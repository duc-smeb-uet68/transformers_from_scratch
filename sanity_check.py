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
from utils.tokenizer import Vocabulary


def run_sanity_check():
    # 1. Cấu hình & Thiết bị
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- SANITY CHECK STARTING ON {device} ---")

    # Hyperparameters (Giữ nguyên như train.py để test đúng kiến trúc)
    D_MODEL = 512
    N_LAYERS = 6
    N_HEADS = 8
    D_FF = 2048
    DROPOUT = 0.1  # Có thể giảm về 0.0 để test overfit nhanh hơn, nhưng để 0.1 cũng được
    MAX_LEN = 100
    BATCH_SIZE = 10  # Batch nhỏ vì chỉ có 20 câu
    LEARNING_RATE = 0.0005  # LR hơi cao một chút để hội tụ nhanh
    N_EPOCHS = 100  # Chạy nhiều epoch để ép loss về 0

    # 2. Load Dữ Liệu & Vocab
    print("Loading Data & Vocab...")
    try:
        # Đường dẫn vocab (Hãy sửa lại nếu khác máy bạn)
        vocab_src = Vocabulary("data/shared_vocab/tokenizer_shared.json")
        vocab_tgt = Vocabulary("data/shared_vocab/tokenizer_shared.json")

        # Load dataset
        dataset = load_from_disk("data/iwslt2015_data")

        # --- QUAN TRỌNG: CHỈ LẤY 20 CÂU ĐẦU TIÊN ---
        subset_size = 20
        raw_src = [item['vi'] for item in dataset['train']][:subset_size]
        raw_tgt = [item['en'] for item in dataset['train']][:subset_size]

        print(f"--> Đã lấy {len(raw_src)} cặp câu mẫu để test.")
        print(f"--> Ví dụ: {raw_src[0]}  ==>  {raw_tgt[0]}")

    except Exception as e:
        print(f"Lỗi load data: {e}")
        return

    # Tạo Dataset & DataLoader
    sanity_dataset = BilingualDataset(raw_src, raw_tgt, vocab_src, vocab_tgt, max_len=MAX_LEN)
    collate = Collate(pad_idx=vocab_src.pad_idx)
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

    # Weight Tying (Giống train.py)
    model.src_embedding.emb.weight = model.tgt_embedding.emb.weight
    model.fc_out.weight = model.src_embedding.emb.weight

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


def test_sanity_translation(model, src_text, vocab_src, vocab_tgt, device):
    """Test dịch lại chính câu vừa học xem có đúng y hệt không"""
    model.eval()
    src_ids = [vocab_src.sos_idx] + vocab_src.numericalize(src_text) + [vocab_src.eos_idx]
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)

    # Greedy Decode đơn giản
    tgt_indices = [vocab_tgt.sos_idx]
    for _ in range(50):
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
        tgt_mask = model.make_tgt_mask(tgt_tensor)

        with torch.no_grad():
            tgt_emb = model.positional_encoding(model.tgt_embedding(tgt_tensor))
            enc_src = model.encoder(model.positional_encoding(model.src_embedding(src_tensor)), src_mask)
            output = model.decoder(tgt_emb, enc_src, tgt_mask, src_mask)
            pred_token = output[:, -1, :].argmax(1).item()

            if pred_token == vocab_tgt.eos_idx:
                break
            tgt_indices.append(pred_token)

    pred_text = vocab_tgt.decode(tgt_indices[1:])  # Bỏ <sos>
    print(f"Input:  {src_text}")
    print(f"Output: {pred_text}")


if __name__ == "__main__":
    run_sanity_check()