import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
import matplotlib.pyplot as plt
import os
from datasets import load_from_disk, load_dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Import các module từ project của bạn
from model.transformer import Transformer
from utils.dataset import BilingualDataset, Collate
from utils.tokenizer import Vocabulary


# --- 1. CLASS SCHEDULER (MỚI) ---
class TransformerScheduler:
    """
    Learning Rate Scheduler chuẩn của Transformer.
    LR tăng dần trong giai đoạn warmup và giảm dần sau đó.
    Công thức: lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.current_step = 0
        self._rate = 0

    def step(self):
        "Cập nhật learning rate và bước nhảy"
        self.current_step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate

    def rate(self, step=None):
        "Tính toán learning rate hiện tại"
        if step is None:
            step = self.current_step
        if step == 0: step = 1  # Tránh chia cho 0
        return self.factor * (self.d_model ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()


# --- 2. HÀM TRAIN 1 EPOCH ---
def train_epoch(model, iterator, optimizer, scheduler, criterion, clip, device):
    model.train()
    epoch_loss = 0

    # Khởi tạo Scaler cho Mixed Precision Training (tăng tốc GPU)
    scaler = GradScaler()

    progress_bar = tqdm(iterator, desc="Training", leave=False)

    for i, (src, tgt) in enumerate(iterator):
        src, tgt = src.to(device), tgt.to(device)

        # Target input: Bỏ token cuối (<eos>)
        tgt_input = tgt[:, :-1]

        # Target output: Bỏ token đầu (<sos>) để mô hình dự đoán từ tiếp theo
        tgt_output = tgt[:, 1:]

        # Reset gradients qua scheduler
        scheduler.zero_grad()

        # Forward pass với autocast
        with autocast():
            output = model(src, tgt_input)

            # Reshape để tính loss
            # output: [batch_size, trg_len - 1, output_dim] -> [batch_size * (trg_len - 1), output_dim]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)

            # tgt_output: [batch_size, trg_len - 1] -> [batch_size * (trg_len - 1)]
            tgt_output = tgt_output.contiguous().view(-1)

            loss = criterion(output, tgt_output)

        # Backward pass với scaler
        scaler.scale(loss).backward()

        # Unscale trước khi clip gradient để tránh bùng nổ gradient
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Update weights và Learning Rate
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

    return epoch_loss / len(iterator)


# --- 3. HÀM ĐÁNH GIÁ (EVALUATE) ---
def evaluate(model, iterator, criterion, device):
    model.eval()  # Quan trọng: Tắt Dropout và Batch Norm
    epoch_loss = 0

    with torch.no_grad():  # Không tính gradient
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


# --- 4. CÁC HÀM TIỆN ÍCH ---
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_plots(train_losses, val_losses, filename_prefix="model"):
    """Vẽ và lưu biểu đồ Loss và Perplexity"""
    if not os.path.exists('reports'):
        os.makedirs('reports')

    # 1. Vẽ Loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.title("Training & Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'reports/{filename_prefix}_loss.png')
    plt.close()

    # 2. Vẽ Perplexity (PPL = exp(loss))
    train_ppls = [math.exp(min(l, 100)) for l in train_losses]  # min(l, 100) để tránh tràn số
    val_ppls = [math.exp(min(l, 100)) for l in val_losses]

    plt.figure(figsize=(10, 6))
    plt.plot(train_ppls, label='Train PPL', color='green')
    plt.plot(val_ppls, label='Val PPL', color='orange')
    plt.title("Training & Validation Perplexity")
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'reports/{filename_prefix}_ppl.png')
    plt.close()
    print(f"--> Đã lưu biểu đồ tại thư mục reports/")


# --- 5. HÀM CHẠY CHÍNH (MAIN) ---
def run_training():
    # --- Cấu hình Hyperparameters ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    BATCH_SIZE = 32
    N_EPOCHS = 20  # Tăng lên 20 epoch để thấy hiệu quả của scheduler
    CLIP = 1

    # Model configs
    D_MODEL = 512
    N_LAYERS = 6
    N_HEADS = 8
    D_FF = 2048
    DROPOUT = 0.1
    MAX_LEN = 128  # Giới hạn độ dài câu để tiết kiệm bộ nhớ

    # --- Load Data & Tokenizer ---
    print("--- Đang tải dữ liệu và Tokenizer ---")

    # 1. Load Tokenizer (BPE JSON files)
    # Lưu ý: Đảm bảo bạn đã chạy build_vocab_bpe.py trước đó
    try:
        vocab_src = Vocabulary("data/vocab_bpe/tokenizer_vi.json")
        vocab_tgt = Vocabulary("data/vocab_bpe/tokenizer_en.json")
    except Exception as e:
        print(f"Lỗi load vocab: {e}")
        print("Vui lòng chạy file build_vocab_bpe.py trước!")
        return

    print(f"Vocab Source Size: {len(vocab_src)}")
    print(f"Vocab Target Size: {len(vocab_tgt)}")

    # 2. Load Dataset
    def extract_data(data_split):
        src = [item['vi'] for item in data_split]
        tgt = [item['en'] for item in data_split]
        return src, tgt

    try:
        dataset = load_from_disk("data/iwslt2015_data")
    except:
        dataset = load_dataset("nguyenvuhuy/iwslt2015-en-vi")

    train_src, train_tgt = extract_data(dataset['train'])
    val_src, val_tgt = extract_data(dataset['validation'])

    # 3. Tạo DataLoader
    train_dataset = BilingualDataset(train_src, train_tgt, vocab_src, vocab_tgt, max_len=MAX_LEN)
    val_dataset = BilingualDataset(val_src, val_tgt, vocab_src, vocab_tgt, max_len=MAX_LEN)

    collate = Collate(pad_idx=vocab_src.pad_idx)

    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                collate_fn=collate, num_workers=2, pin_memory=True)
    valid_iterator = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate, num_workers=2, pin_memory=True)

    # --- Khởi tạo Mô hình ---
    print("--- Khởi tạo Mô hình Transformer ---")
    model = Transformer(
        src_vocab_size=len(vocab_src),
        tgt_vocab_size=len(vocab_tgt),
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
        max_len=5000,  # Max len positional encoding (đủ lớn)
        src_pad_idx=vocab_src.pad_idx,
        tgt_pad_idx=vocab_tgt.pad_idx
    ).to(device)

    # Đếm số lượng tham số
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Mô hình có {count_parameters(model):,} tham số train được')

    # --- Optimizer, Scheduler & Loss ---
    # Adam Optimizer với các tham số chuẩn cho Transformer
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)

    # Scheduler Warmup
    scheduler = TransformerScheduler(optimizer, d_model=D_MODEL, warmup_steps=4000)

    # Loss Function: Label Smoothing giúp model đỡ quá tự tin, giảm overfit
    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab_tgt.pad_idx,
        label_smoothing=0.1
    )

    # --- Vòng lặp Training ---
    best_valid_loss = float('inf')
    train_loss_history = []
    valid_loss_history = []

    print(f"--- Bắt đầu Training trong {N_EPOCHS} epochs ---")

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        # Train & Eval
        train_loss = train_epoch(model, train_iterator, optimizer, scheduler, criterion, CLIP, device)
        valid_loss = evaluate(model, valid_iterator, criterion, device)

        # Lưu lịch sử
        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # Lưu Checkpoint tốt nhất
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'transformer_best.pt')
            print(f"--> Đã lưu model tốt nhất (Val Loss: {valid_loss:.3f})")

        # In kết quả
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    # --- Kết thúc ---
    print("\n--- Hoàn tất Training ---")
    save_plots(train_loss_history, valid_loss_history)

    # Lưu model cuối cùng
    torch.save(model.state_dict(), 'transformer_last.pt')


if __name__ == "__main__":
    run_training()