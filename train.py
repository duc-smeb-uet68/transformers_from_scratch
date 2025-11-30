import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
import matplotlib.pyplot as plt
import os
from datasets import load_dataset, load_from_disk
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Import các module từ giai đoạn trước
from model.transformer import Transformer
from utils.dataset import BilingualDataset, Collate
from utils.tokenizer import Vocabulary


# def train_epoch(model, iterator, optimizer, criterion, clip, device):
#     """
#     Hàm huấn luyện cho 1 Epoch (duyệt qua toàn bộ dữ liệu 1 lần)
#     """
#     model.train() # Chuyển sang chế độ training (bật Dropout)
#     epoch_loss = 0
#
#     scaler = GradScaler()
#
#     for i, (src, tgt) in enumerate(iterator):
#         src = src.to(device)
#         tgt = tgt.to(device)
#
#         # Decoder Input: Bỏ token cuối cùng <eos>
#         tgt_input = tgt[:, :-1]
#
#         # Target Output: Bỏ token đầu tiên <sos> (đây là cái ta muốn model đoán)
#         tgt_output = tgt[:, 1:]
#
#         optimizer.zero_grad() # Xóa gradient cũ
#
#         # Forward pass
#         # output shape: (batch_size, tgt_len - 1, output_dim)
#         output = model(src, tgt_input)
#
#         # Reshape để tính Loss
#         output_dim = output.shape[-1]
#         output = output.contiguous().view(-1, output_dim)
#         tgt_output = tgt_output.contiguous().view(-1)
#
#         # Tính Loss (Cross Entropy)
#         loss = criterion(output, tgt_output)
#
#         # Backward pass
#         loss.backward()
#
#         # Cắt gradient (Gradient Clipping) để tránh bùng nổ gradient
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#
#         # Cập nhật trọng số
#         optimizer.step()
#
#         epoch_loss += loss.item()
#
#     return epoch_loss / len(iterator)


def train_epoch(model, iterator, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0

    # 1. Khởi tạo Scaler
    scaler = GradScaler()

    progress_bar = tqdm(iterator, desc="Training", leave=False)

    for i, (src, tgt) in enumerate(iterator):
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()

        # 2. Chạy Forward pass trong môi trường autocast (tiết kiệm RAM)
        with autocast():
            output = model(src, tgt_input)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            tgt_output = tgt_output.contiguous().view(-1)

            loss = criterion(output, tgt_output)

        # 3. Backward pass dùng scaler
        scaler.scale(loss).backward()

        # 4. Update weights dùng scaler
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        progress_bar.set_postfix(loss=loss.item())

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    """
    Hàm đánh giá trên tập Validation (không cập nhật trọng số)
    """
    model.eval() # Chuyển sang chế độ eval (tắt Dropout)
    epoch_loss = 0

    with torch.no_grad(): # Không tính gradient giúp chạy nhanh hơn
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




def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def save_training_plot(train_losses, val_losses, title, filename):
    """
    Vẽ và lưu biểu đồ Loss theo Epoch
    Args:
        train_losses: List chứa giá trị loss của tập train qua từng epoch
        val_losses: List chứa giá trị loss của tập val qua từng epoch
        title: Tiêu đề biểu đồ
        filename: Tên file ảnh muốn lưu (ví dụ: 'loss_chart.png')
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue', marker='o')
    plt.plot(val_losses, label='Validation Loss', color='red', marker='x')

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Lưu ảnh
    if not os.path.exists('reports'):
        os.makedirs('reports')
    plt.savefig(os.path.join('reports', filename))
    plt.close()
    print(f"--> Đã lưu biểu đồ tại: reports/{filename}")

def save_perplexity_plot(train_losses, val_losses, filename):
    """
    Vẽ biểu đồ Perplexity (PPL = exp(Loss))
    PPL càng thấp càng tốt.
    """
    train_ppls = [math.exp(l) for l in train_losses]
    val_ppls = [math.exp(l) for l in val_losses]

    plt.figure(figsize=(10, 6))
    plt.plot(train_ppls, label='Train PPL', color='green', linestyle='--')
    plt.plot(val_ppls, label='Val PPL', color='orange', linestyle='--')

    plt.title("Model Perplexity over Epochs")
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.yscale('log') # Dùng thang log vì PPL có thể rất lớn lúc đầu
    plt.grid(True)

    if not os.path.exists('reports'):
        os.makedirs('reports')
    plt.savefig(os.path.join('reports', filename))
    plt.close()
    print(f"--> Đã lưu biểu đồ PPL tại: reports/{filename}")


def run_training():
    # 1. Cấu hình (Hyperparameters) - Nên để trong config file riêng, nhưng để đây cho tiện
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 32
    N_EPOCHS = 10
    CLIP = 1
    LR = 0.0001

    # # 2. Chuẩn bị Dữ liệu (Giả lập hoặc Load thật)
    # print("--- Đang chuẩn bị dữ liệu ---")
    # # LƯU Ý: Ở đây cậu thay bằng code load file IWSLT thật
    # # Demo dữ liệu giả để code chạy được ngay
    # train_src = ["tôi là sinh viên", "máy học rất thú vị"] * 50
    # train_tgt = ["i am a student", "machine learning is interesting"] * 50
    # val_src = ["tôi đi học", "xin chào"] * 10
    # val_tgt = ["i go to school", "hello"] * 10

    def extract_data(data_split):
        src = [item['vi'] for item in data_split]
        tgt = [item['en'] for item in data_split]
        return src, tgt

    # Tokenizer & Vocab
    # Removed: from utils.tokenizer import Vocabulary
    vocab_src = Vocabulary(freq_threshold=1)
    vocab_tgt = Vocabulary(freq_threshold=1)
    # Build vocab từ dữ liệu train
    vocab_src.load_vocab('data/vocab_src.json')
    vocab_tgt.load_vocab('data/vocab_tgt.json')

    print(f"Vocab Source: {len(vocab_src)} | Vocab Target: {len(vocab_tgt)}")
    dataset= load_from_disk("data/iwslt2015_data")
    train_src, train_tgt = extract_data(dataset['train'])
    val_src, val_tgt = extract_data(dataset['validation'])


    # DataLoader
    train_dataset = BilingualDataset(train_src, train_tgt, vocab_src, vocab_tgt)
    val_dataset = BilingualDataset(val_src, val_tgt, vocab_src, vocab_tgt)

    collate = Collate(pad_idx=0) # 0 là <pad>

    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate,num_workers=0,pin_memory=False)
    valid_iterator = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate,num_workers=0,pin_memory=False)

    # 3. Khởi tạo Mô hình
    model = Transformer(
        src_vocab_size=len(vocab_src),
        tgt_vocab_size=len(vocab_tgt),
        d_model=512,    # Demo dùng nhỏ, bài thật dùng 512
        n_layers=6,     # Demo dùng 3, bài thật dùng 6
        n_heads=8,
        d_ff=512,
        dropout=0.1,
        max_len=512,
        src_pad_idx=vocab_src.stoi["<pad>"],
        tgt_pad_idx=vocab_tgt.stoi["<pad>"]
    ).to(device)

    # 4. Optimizer & Loss
    # Adam Optimizer [cite: 29]
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Cross Entropy Loss [cite: 28]
    # Quan trọng: ignore_index=0 để không tính loss cho các token <pad>
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_tgt.stoi["<pad>"])

    # 5. Vòng lặp Training [cite: 30]
    best_valid_loss = float('inf')

    train_loss_history = []
    valid_loss_history = []
    print(device)

    print("--- Bắt đầu Training ---")
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss = train_epoch(model, train_iterator, optimizer, criterion, CLIP, device)
        valid_loss = evaluate(model, valid_iterator, criterion, device)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # Lưu model tốt nhất
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'transformer_model.pt')
            print(f"--> Đã lưu model tốt nhất tại Epoch {epoch+1}")

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


    print("\n--- Đang vẽ biểu đồ báo cáo ---")

    # 1. Vẽ biểu đồ Loss
    save_training_plot(
        train_loss_history,
        valid_loss_history,
        title=f'Training Loss (Epochs={N_EPOCHS})',
        filename='loss_chart.png'
    )

    # 2. Vẽ biểu đồ Perplexity
    save_perplexity_plot(
        train_loss_history,
        valid_loss_history,
        filename='perplexity_chart.png'
    )


if __name__ == "__main__":
    print("--- Đang tải dataset IWSLT2015 (Vi-En) ---")
    dataset = load_dataset("nguyenvuhuy/iwslt2015-en-vi")

    run_training()