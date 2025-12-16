import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_from_disk
import math
import time

# Import các module từ project của bạn
# Đảm bảo bạn đặt file này cùng cấp với thư mục model/ và utils/
from model.transformer import Transformer
from utils.dataset import BilingualDataset, Collate
from utils.word_vocab import Vocabulary

def run_sanity_check():
    # 1. Cấu hình & Thiết bị
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- SANITY CHECK STARTING ON {device} ---")

    # Hyperparameters (Giữ nguyên như train.py để test đúng kiến trúc)
    # D_MODEL = 512
    # N_LAYERS = 6
    # N_HEADS = 8
    # D_FF = 2048
    # DROPOUT = 0.0  # Có thể giảm về 0.0 để test overfit nhanh hơn, nhưng để 0.1 cũng được
    # MAX_LEN = 100
    # BATCH_SIZE = 2  # Batch nhỏ vì chỉ có 20 câu
    # LEARNING_RATE = 0.005  # LR hơi cao một chút để hội tụ nhanh
    D_MODEL = 128
    N_LAYERS = 2
    N_HEADS = 4
    D_FF = 512
    DROPOUT = 0.0
    MAX_LEN = 100
    BATCH_SIZE = 20
    LEARNING_RATE = 1e-3

    N_EPOCHS = 300  # Chạy nhiều epoch để ép loss về 0

    # 2. Load Dữ Liệu & Vocab
    print("Loading Data & Vocab...")
    try:
        # Đường dẫn vocab (Hãy sửa lại nếu khác máy bạn)
        vocab_src = Vocabulary.load_vocab("data/vocab0/vocab_src.json")
        vocab_tgt = Vocabulary.load_vocab("data/vocab0/vocab_tgt.json")

        # Load dataset
        dataset = load_from_disk("data/iwslt2015_data")

        # --- QUAN TRỌNG: CHỈ LẤY 20 CÂU ĐẦU TIÊN ---
        subset_size = 20
        raw_src = [item['vi'] for item in dataset['train']][:subset_size]
        raw_tgt = [item['en'] for item in dataset['train']][:subset_size]

        # --- SANITY CHECK CỨNG: lặp lại 1 câu ---
        raw_src = [raw_src[1]] * 20
        raw_tgt = [raw_tgt[1]] * 20

        print(f"--> Đã lấy {len(raw_src)} cặp câu mẫu để test.")
        print(f"--> Ví dụ: {raw_src[0]}  ==>  {raw_tgt[0]}")

    except Exception as e:
        print(f"Lỗi load data: {e}")
        return

    # Tạo Dataset & DataLoader
    sanity_dataset = BilingualDataset(raw_src, raw_tgt, vocab_src, vocab_tgt, max_len=MAX_LEN)
    collate = Collate(
        src_pad_idx=vocab_src.pad_idx,
        tgt_pad_idx=vocab_tgt.pad_idx
    )

    # Shuffle=True hay False không quan trọng vì data quá ít, nhưng False dễ debug hơn
    sanity_iterator = DataLoader(sanity_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    # 3. Khởi tạo Model
    model = Transformer(
        src_vocab_size=len(vocab_src),
        tgt_vocab_size=len(vocab_tgt),
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, d_ff=D_FF,
        dropout=DROPOUT, max_len=5000,
        src_pad_idx=vocab_src.pad_idx, tgt_pad_idx=vocab_tgt.pad_idx
    ).to(device)

    # # Weight Tying (Giống train.py)
    # model.src_embedding.emb.weight = model.tgt_embedding.emb.weight
    # model.fc_out.weight = model.src_embedding.emb.weight

    # 4. Optimizer & Loss
    # Dùng Adam thường thay vì AdamW cho đơn giản, set LR cố định
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Tắt label smoothing để ép loss về 0 tuyệt đối (Label smoothing sẽ giữ loss quanh 1.0)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_tgt.pad_idx, label_smoothing=0.0)

    # 5. Training Loop siêu đơn giản
    model.train()

    for epoch in range(N_EPOCHS):
        epoch_loss = 0

        for src, tgt in sanity_iterator:
            src, tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()

            output = model(src, tgt_input)

            # Reshape để tính loss
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            tgt_output = tgt_output.contiguous().view(-1)

            loss = criterion(output, tgt_output)
            loss.backward()

            # Clip grad để an toàn
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(sanity_iterator)

        # In kết quả mỗi 10 epoch
        if (epoch + 1) % 10 == 0:
            ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
            print(f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.5f} | PPL: {ppl:.5f}")

            # Nếu Loss < 0.1 thì thành công sớm
            if avg_loss < 0.1:
                print(">>> THÀNH CÔNG! Model đã overfit được dữ liệu nhỏ.")
                print(f"Testing thử câu đầu tiên...")
                test_sanity_translation(model, raw_src[0], vocab_src, vocab_tgt, device)
                break

    print("--- HOÀN TẤT SANITY CHECK ---")


def test_sanity_translation(model, src_sentence, vocab_src, vocab_tgt, device, max_len=50):
    model.eval()

    # ===== Encode source =====
    src_ids = vocab_src.numericalize(src_sentence)
    src_ids = [vocab_src.sos_idx] + src_ids + [vocab_src.eos_idx]
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)

    # ===== Decode =====
    generated = [vocab_tgt.sos_idx]

    for _ in range(max_len):
        tgt_tensor = torch.tensor(generated).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)

        next_token = output[:, -1, :].argmax(dim=-1).item()

        # 🔒 Bảo vệ: token vượt vocab
        if next_token >= len(vocab_tgt):
            print("⚠️ Token vượt vocab, dừng decode.")
            break

        generated.append(next_token)

        if next_token == vocab_tgt.eos_idx:
            break

    # ===== Decode to text =====
    pred_sentence = vocab_tgt.decode(generated[1:])
    print("🔍 Output:", pred_sentence)


if __name__ == "__main__":
    run_sanity_check()