# File: build_vocab/build_vocab_vlsp.py

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os


def build_vlsp_vocab():
    # --- CẤU HÌNH ĐƯỜNG DẪN ---
    # 1. Đường dẫn chứa file dữ liệu thô (Raw text)
    # Dùng os.path.join để tránh lỗi Window/Linux
    data_dir = os.path.join("..", "data", "vlsp_data")

    files = [
        os.path.join(data_dir, "train.vi.txt"),
        os.path.join(data_dir, "train.en.txt")
    ]

    # Kiểm tra xem file có tồn tại không trước khi chạy
    for f in files:
        if not os.path.exists(f):
            print(f"❌ Lỗi: Không tìm thấy file tại {f}")
            print(f"👉 Hãy tạo thư mục 'data/vlsp_data' và copy file .txt vào đó!")
            return

    # 2. Cấu hình Tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    # vocab_size 10k là hợp lý cho tập dữ liệu nhỏ/chuyên ngành
    trainer = BpeTrainer(
        vocab_size=10000,
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"],
        # show_progress=True giúp bạn nhìn thấy thanh tiến trình
        show_progress=True
    )

    # 3. Train
    print("🚀 Đang train Tokenizer trên dữ liệu Y tế (VLSP)...")
    tokenizer.train(files, trainer)

    # 4. Lưu (Lưu vào data/vlsp_vocab cho gọn)
    save_path = os.path.join("..", "data", "vlsp_vocab", "tokenizer_shared.json")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"✅ Đã lưu Tokenizer mới tại: {save_path}")


if __name__ == "__main__":
    build_vlsp_vocab()