import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Import các module từ project cũ
from model.transformer import Transformer
from utils.dataset import BilingualDataset, Collate
from utils.tokenizer import Vocabulary
from configs import cfg

FT_CONFIG = {
    'dataset_path': 'data/vlsp',
    'load_model_path': cfg.model,
    'save_model_path': 'weights/fine_tune.pt',

    'lr': 1e-5,
    'n_epochs': 25,
    'batch_size': 256,
    'warmup_steps': 1000,
    'patience': 5
}


def save_ft_plots(train_losses, val_losses, output_dir='reports'):
    """Vẽ và lưu biểu đồ Loss & Perplexity"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    epochs = range(1, len(train_losses) + 1)

    # --- Biểu đồ 1: LOSS ---
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Fine-tuning Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/finetune_loss.png')
    plt.close()


    def safe_exp(l):
        try:
            val = math.exp(l)
            return val if val < 1000 else 1000  # Cap max PPL để biểu đồ đẹp
        except OverflowError:
            return 1000

    train_ppls = [safe_exp(l) for l in train_losses]
    val_ppls = [safe_exp(l) for l in val_losses]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_ppls, 'g-', label='Train PPL')
    plt.plot(epochs, val_ppls, 'm-', label='Val PPL')
    plt.title('Fine-tuning Perplexity')
    plt.xlabel('Epochs')
    plt.ylabel('PPL')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/finetune_ppl.png')
    plt.close()

    print(f"--> Đã lưu biểu đồ tại: {output_dir}/finetune_loss.png và finetune_ppl.png")


class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='weights/checkpoint.pt', verbose=True):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class TransformerScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.current_step = 0
        self._rate = 0

    def step(self):
        self.current_step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate

    def rate(self, step=None):
        if step is None: step = self.current_step
        if step == 0: step = 1
        return self.factor * (self.d_model ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()


def train_epoch(model, iterator, optimizer, scheduler, criterion, clip, device, scaler):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(iterator, desc="Fine-tuning", leave=False)

    for i, (src, tgt) in enumerate(progress_bar):
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()

        with autocast():
            output = model(src, tgt_input)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            tgt_output = tgt_output.contiguous().view(-1)
            loss = criterion(output, tgt_output)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[WARN] Batch {i} Loss NaN/Inf. Skipped.")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, tgt) in enumerate(iterator):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = model(src, tgt_input)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            tgt_output = tgt_output.contiguous().view(-1)
            loss = criterion(output, tgt_output)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def run_fine_tuning():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=== BẮT ĐẦU FINE-TUNING (VLSP MEDICAL) ===")

    # 1. Load Vocab (IWSLT)
    if not os.path.exists(cfg.vocab_src):
        print(f"Lỗi: Không tìm thấy vocab tại {cfg.vocab_src}")
        return

    vocab_src = Vocabulary(cfg.vocab_src)
    vocab_tgt = Vocabulary(cfg.vocab_tgt)

    # 2. Load Data VLSP
    print(f"--- Loading VLSP Data: {FT_CONFIG['dataset_path']} ---")
    try:
        dataset = load_from_disk(FT_CONFIG['dataset_path'])
    except Exception as e:
        print(f"Lỗi load data VLSP: {e}")
        return

    train_dataset = BilingualDataset(
        [x['vi'] for x in dataset['train']],
        [x['en'] for x in dataset['train']],
        vocab_src, vocab_tgt, max_len=cfg.max_len
    )
    val_dataset = BilingualDataset(
        [x['vi'] for x in dataset['validation']],
        [x['en'] for x in dataset['validation']],
        vocab_src, vocab_tgt, max_len=cfg.max_len
    )

    collate = Collate(pad_idx=vocab_src.pad_idx)

    train_iterator = DataLoader(train_dataset, batch_size=FT_CONFIG['batch_size'],
                                shuffle=True, collate_fn=collate, num_workers=8, pin_memory=True)
    valid_iterator = DataLoader(val_dataset, batch_size=FT_CONFIG['batch_size'],
                                shuffle=False, collate_fn=collate, num_workers=4, pin_memory=True)

    # 3. Init Model
    model = Transformer(
        src_vocab_size=len(vocab_src),
        tgt_vocab_size=len(vocab_tgt),
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_len=5000,
        src_pad_idx=vocab_src.pad_idx,
        tgt_pad_idx=vocab_tgt.pad_idx
    ).to(device)

    model.src_embedding.emb.weight = model.tgt_embedding.emb.weight
    model.fc_out.weight = model.src_embedding.emb.weight

    # 4. Load Weights
    print(f"--- Loading Weights: {FT_CONFIG['load_model_path']} ---")
    if os.path.exists(FT_CONFIG['load_model_path']):
        state_dict = torch.load(FT_CONFIG['load_model_path'], map_location=device)
        model.load_state_dict(state_dict)
        print("--> Load xong!")
    else:
        print("--> LỖI: Không tìm thấy file trọng số gốc.")
        return

    # 5. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=FT_CONFIG['lr'], betas=(0.9, 0.98), eps=1e-9,
                            weight_decay=cfg.weight_decay)
    scheduler = TransformerScheduler(optimizer, d_model=cfg.d_model, warmup_steps=FT_CONFIG['warmup_steps'], factor=1.0)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_tgt.pad_idx, label_smoothing=cfg.label_smoothing)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=FT_CONFIG['patience'], verbose=True, path=FT_CONFIG['save_model_path'])

    # --- KHỞI TẠO LIST LƯU LỊCH SỬ ---
    train_loss_history = []
    valid_loss_history = []

    # 6. Loop
    print(f"--- Start Training ({FT_CONFIG['n_epochs']} epochs) ---")

    for epoch in range(FT_CONFIG['n_epochs']):
        start_time = time.time()

        train_loss = train_epoch(model, train_iterator, optimizer, scheduler, criterion, cfg.clip, device, scaler)
        valid_loss = evaluate(model, valid_iterator, criterion, device)

        # --- LƯU LỊCH SỬ ---
        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(min(train_loss, 100)):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(min(valid_loss, 100)):7.3f}')

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    save_ft_plots(train_loss_history, valid_loss_history)

    print("--- Fine-tuning Hoàn tất ---")
    print(f"Model y tế đã được lưu tại: {FT_CONFIG['save_model_path']}")


if __name__ == "__main__":
    run_fine_tuning()