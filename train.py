import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
import matplotlib.pyplot as plt
import os
import numpy as np  # Thêm numpy
from datasets import load_from_disk, load_dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Import các module từ project của bạn
from model.transformer import Transformer
from utils.dataset import BilingualDataset, Collate
from utils.tokenizer import Vocabulary


# --- 1. CLASS EARLY STOPPING (MỚI - QUAN TRỌNG) ---
class EarlyStopping:
    """Dừng train nếu validation loss không cải thiện sau một số epoch nhất định."""

    def __init__(self, patience=5, delta=0, path='transformer_best.pt', verbose=True):
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
        '''Lưu model khi validation loss giảm.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# --- 2. CLASS SCHEDULER (GIỮ NGUYÊN) ---
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
        if step is None:
            step = self.current_step
        if step == 0: step = 1
        return self.factor * (self.d_model ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()


# --- 3. HÀM TRAIN 1 EPOCH (CÓ CHECK NaN) ---
def train_epoch(model, iterator, optimizer, scheduler, criterion, clip, device, scaler):
    model.train()
    epoch_loss = 0

    # Progress bar
    progress_bar = tqdm(iterator, desc="Training", leave=False)

    for i, (src, tgt) in enumerate(progress_bar):
        src, tgt = src.to(device), tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        scheduler.zero_grad()

        with autocast():
            output = model(src, tgt_input)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            tgt_output = tgt_output.contiguous().view(-1)
            loss = criterion(output, tgt_output)

        # --- SAFETY CHECK: Chặn loss NaN/Inf ---
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n[CẢNH BÁO] Loss bị NaN hoặc Inf tại batch {i}. Bỏ qua bước cập nhật này.")
            continue
        # ---------------------------------------

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_loss += loss.item()

        # Cập nhật hiển thị
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix(loss=loss.item(), lr=f"{current_lr:.2e}")

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, tgt) in enumerate(iterator):
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = model(src, tgt_input)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            tgt_output = tgt_output.contiguous().view(-1)
            loss = criterion(output, tgt_output)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


# --- 5. TIỆN ÍCH VẼ BIỂU ĐỒ ---
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_plots(train_losses, val_losses, filename_prefix="model"):
    if not os.path.exists('reports'):
        os.makedirs('reports')

    # Vẽ Loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.title("Training & Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)  # Thêm lưới cho dễ nhìn
    plt.savefig(f'reports/{filename_prefix}_loss.png')
    plt.close()

    # Vẽ Perplexity
    # Xử lý overflow cho PPL
    def safe_exp(l):
        try:
            return math.exp(min(l, 50))  # Cap loss ở 50
        except OverflowError:
            return float('inf')

    train_ppls = [safe_exp(l) for l in train_losses]
    val_ppls = [safe_exp(l) for l in val_losses]

    plt.figure(figsize=(10, 6))
    plt.plot(train_ppls, label='Train PPL', color='green')
    plt.plot(val_ppls, label='Val PPL', color='orange')
    plt.title("Training & Validation Perplexity")
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'reports/{filename_prefix}_ppl.png')
    plt.close()
    print(f"--> Đã lưu biểu đồ tại thư mục reports/")


# --- 6. MAIN ---
def run_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Hyperparameters Tối Ưu ---
    BATCH_SIZE = 32
    N_EPOCHS = 50  # Đặt cao lên, Early Stopping sẽ tự dừng
    CLIP = 1.0  # Gradient Clipping
    PATIENCE = 5  # Dừng nếu Val Loss không giảm sau 5 epoch

    D_MODEL = 512
    N_LAYERS = 6
    N_HEADS = 8
    D_FF = 2048
    DROPOUT = 0.1
    MAX_LEN = 128

    # --- Load Data ---
    print("--- Loading Data... ---")
    try:
        vocab_src = Vocabulary("data/shared_vocab/tokenizer_shared.json")
        vocab_tgt = Vocabulary("data/shared_vocab/tokenizer_shared.json")
        dataset = load_from_disk("data/iwslt2015_data")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Extract & Create Dataset
    train_src = [item['vi'] for item in dataset['train']]
    train_tgt = [item['en'] for item in dataset['train']]
    val_src = [item['vi'] for item in dataset['validation']]
    val_tgt = [item['en'] for item in dataset['validation']]

    train_dataset = BilingualDataset(train_src, train_tgt, vocab_src, vocab_tgt, max_len=MAX_LEN)
    val_dataset = BilingualDataset(val_src, val_tgt, vocab_src, vocab_tgt, max_len=MAX_LEN)

    collate = Collate(pad_idx=vocab_src.pad_idx)

    # Tối ưu DataLoader: Tăng workers, persistent_workers
    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                collate_fn=collate, num_workers=4, pin_memory=True, persistent_workers=True)
    valid_iterator = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate, num_workers=4, pin_memory=True, persistent_workers=True)

    # --- Model ---
    model = Transformer(
        src_vocab_size=len(vocab_src),
        tgt_vocab_size=len(vocab_tgt),
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, d_ff=D_FF,
        dropout=DROPOUT, max_len=5000,
        src_pad_idx=vocab_src.pad_idx, tgt_pad_idx=vocab_tgt.pad_idx
    ).to(device)

    #share weights giữa encoder decoder và outputLayer
    model.src_embedding.emb.weight = model.tgt_embedding.emb.weight
    model.fc_out.weight = model.src_embedding.emb.weight

    print(f'Mô hình có {sum(p.numel() for p in model.parameters() if p.requires_grad):,} tham số')

    # --- Optimizer & Loss ---
    # THAY ĐỔI QUAN TRỌNG: Dùng AdamW thay vì Adam để có weight decay tốt hơn
    optimizer = optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)

    scheduler = TransformerScheduler(optimizer, d_model=D_MODEL, warmup_steps=4000)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_tgt.pad_idx, label_smoothing=0.1)

    # Khởi tạo Early Stopping
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path='transformer_best.pt')

    # Scaler cho Mixed Precision
    scaler = GradScaler()

    train_loss_history = []
    valid_loss_history = []

    print(f"--- Bắt đầu Training (Max {N_EPOCHS} epochs, Patience {PATIENCE}) ---")

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss = train_epoch(model, train_iterator, optimizer, scheduler, criterion, CLIP, device, scaler)
        valid_loss = evaluate(model, valid_iterator, criterion, device)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

        # --- GỌI EARLY STOPPING ---
        # Nó sẽ tự lưu model nếu tốt hơn, và trả về True nếu cần dừng
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("--> Early stopping kích hoạt! Dừng train để tránh Overfitting/Explosion.")
            break

    save_plots(train_loss_history, valid_loss_history)
    # Lưu model trạng thái cuối cùng (dù có thể không phải tốt nhất)
    torch.save(model.state_dict(), 'transformer_last_state.pt')
    print("--- Hoàn tất ---")


if __name__ == "__main__":
    run_training()