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
from translate import blue_score
from configs import cfg

# Import các module từ project của bạn

from model.transformer import Transformer
from utils.dataset import BilingualDataset, Collate
from utils.tokenizer import Vocabulary


# --- 1. CLASS EARLY STOPPING (MỚI - QUAN TRỌNG) ---
class EarlyStopping:
    """Dừng train nếu validation loss không cải thiện sau một số epoch nhất định."""

    def __init__(self, patience=5, delta=0, path='weights/transformer_best.pt', verbose=True):
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
        if scheduler is not None:
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
        if scheduler is not None:
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

def save_checkpoint(model, optimizer, scheduler, epoch, filename="checkpoint/checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.__dict__ if hasattr(scheduler, '__dict__') else None, # Tùy loại scheduler
    }
    torch.save(checkpoint, filename)
    print(f"--> Đã lưu checkpoint tại epoch {epoch+1}")

# --- 6. MAIN ---
def run_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Hyperparameters Tối Ưu ---
    BATCH_SIZE = cfg.batch_size
    N_EPOCHS = cfg.n_epochs
    CLIP = cfg.clip  # Gradient Clipping
    PATIENCE = cfg.patience

    D_MODEL = cfg.d_model
    N_LAYERS = cfg.n_layers
    N_HEADS = cfg.n_heads
    D_FF = cfg.d_ff
    DROPOUT = cfg.dropout
    MAX_LEN = cfg.max_len

    # --- Load Data ---
    print("--- Loading Data... ---")
    try:
        vocab_src = Vocabulary(cfg.vocab_src)
        vocab_tgt = Vocabulary(cfg.vocab_tgt)
        dataset = load_from_disk(cfg.dataset)
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
                                collate_fn=collate, num_workers=16, pin_memory=True, persistent_workers=True)
    valid_iterator = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate, num_workers=8, pin_memory=True, persistent_workers=True)

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
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=cfg.weight_decay)

    scheduler = TransformerScheduler(optimizer, d_model=D_MODEL, warmup_steps=cfg.warmup_steps, factor= 1.0)
    # scheduler = None
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_tgt.pad_idx, label_smoothing=cfg.label_smoothing)

    # Khởi tạo Early Stopping
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=cfg.model)

    # Scaler cho Mixed Precision
    scaler = GradScaler()

    train_loss_history = []
    valid_loss_history = []

    print(f"--- Bắt đầu Training (Max {N_EPOCHS} epochs, Patience {PATIENCE}) ---")

    CHECKPOINT_PATH = cfg.checkpoint
    BEST_MODEL_PATH = cfg.model
    start_epoch = 0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"--> Phát hiện checkpoint '{CHECKPOINT_PATH}'. Đang load để train tiếp...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        # 1. Load trọng số mô hình
        model.load_state_dict(checkpoint['model_state_dict'])

        # 2. Load trạng thái optimizer (QUAN TRỌNG ĐỂ TRAIN TIẾP MƯỢT MÀ)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        scheduler.__dict__.update(checkpoint['scheduler_state_dict'])
        # 3. Cập nhật epoch bắt đầu
        start_epoch = checkpoint['epoch'] + 1

        print(f"--> Resume thành công! Sẽ bắt đầu từ Epoch {start_epoch + 1}")
    elif os.path.exists(BEST_MODEL_PATH):
        print(f"--> Không thấy Checkpoint nhưng thấy '{BEST_MODEL_PATH}'.")
        print("--> Đang load trọng số Best Model để train tiếp (Fine-tune)...")

        # Load trọng số model
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

        # Vì file best không lưu epoch, bạn phải tự ước lượng hoặc chấp nhận start_epoch = 0
        # Nếu bạn biết nó chết ở epoch 15, và best model lưu ở epoch trước đó, ví dụ epoch 12
        # Bạn có thể gán thủ công:
        # start_epoch = 12

        print("--> Đã load model weights. Optimizer sẽ được khởi tạo lại.")
    else:

        print("--> Không thấy checkpoint. Train từ đầu (Scratch).")

    print(f"--- Bắt đầu Training từ Epoch {start_epoch + 1} đến {N_EPOCHS} ---")

    for epoch in range(start_epoch, N_EPOCHS):
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

        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, filename=CHECKPOINT_PATH,)

        # --- GỌI EARLY STOPPING ---
        # Nó sẽ tự lưu model nếu tốt hơn, và trả về True nếu cần dừng
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("--> Early stopping kích hoạt! Dừng train để tránh Overfitting/Explosion.")
            print(f"saved to weights/{BEST_MODEL_PATH}")
            break

        torch.cuda.empty_cache()

    save_plots(train_loss_history, valid_loss_history)
    print("--- Hoàn tất ---")


if __name__ == "__main__":
    run_training()
    print("===============Train xooong===================")
