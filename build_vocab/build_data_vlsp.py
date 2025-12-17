import os
from datasets import Dataset, DatasetDict


def read_bilingual_pair(vi_path, en_path, name="data"):
    """
    Hàm đọc và kiểm tra cặp file song ngữ
    """
    print(f"--- Đang đọc dữ liệu: {name} ---")

    if not os.path.exists(vi_path) or not os.path.exists(en_path):
        print(f"LỖI: Không tìm thấy file tại {vi_path} hoặc {en_path}")
        return None

    with open(vi_path, "r", encoding="utf-8") as f:
        vi_lines = f.read().splitlines()

    with open(en_path, "r", encoding="utf-8") as f:
        en_lines = f.read().splitlines()

    # Kiểm tra độ lệch dòng
    if len(vi_lines) != len(en_lines):
        print(f"CẢNH BÁO: Số dòng không khớp ({len(vi_lines)} vs {len(en_lines)}). Sẽ lấy phần giao nhau.")
        min_len = min(len(vi_lines), len(en_lines))
        vi_lines = vi_lines[:min_len]
        en_lines = en_lines[:min_len]

    print(f"-> Số lượng câu: {len(vi_lines)}")

    # Tạo list dictionary, lọc bỏ câu rỗng
    data = []
    for vi, en in zip(vi_lines, en_lines):
        vi = vi.strip()
        en = en.strip()
        if len(vi) > 0 and len(en) > 0:
            data.append({"vi": vi, "en": en})

    return Dataset.from_list(data)


def main():
    # --- CẤU HÌNH ĐƯỜNG DẪN (Bạn sửa lại cho đúng vị trí file của bạn) ---
    # Giả sử file code này nằm cùng thư mục với các file .txt
    TRAIN_VI = "train_vi.txt"
    TRAIN_EN = "train_en.txt"
    TEST_VI = "public_test.vi.txt"
    TEST_EN = "public_test.en.txt"

    OUTPUT_PATH = "data/vlsp"  # Nơi lưu kết quả cuối cùng

    # 1. Xử lý tập Train (Để chia train/val)
    raw_train = read_bilingual_pair(TRAIN_VI, TRAIN_EN, "TRAIN")
    if raw_train is None: return

    # Chia tập Train thành Train (90%) và Validation (10%)
    # Seed 42 để đảm bảo lần nào chia cũng giống nhau
    split_data = raw_train.train_test_split(test_size=0.1, seed=42)
    dataset_train = split_data['train']
    dataset_val = split_data['test']

    # 2. Xử lý tập Test
    dataset_test = read_bilingual_pair(TEST_VI, TEST_EN, "TEST")
    if dataset_test is None: return

    # 3. Gom lại thành DatasetDict
    final_dataset = DatasetDict({
        'train': dataset_train,
        'validation': dataset_val,  # Dùng để theo dõi loss và early stopping
        'test': dataset_test  # Dùng để tính BLEU cuối cùng
    })

    # 4. Lưu xuống đĩa
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    final_dataset.save_to_disk(OUTPUT_PATH)

    print("\n=== TỔNG HỢP DỮ LIỆU ĐÃ XỬ LÝ ===")
    print(f"Lưu tại: {OUTPUT_PATH}")
    print(f"- Train set:      {len(final_dataset['train'])} câu")
    print(f"- Validation set: {len(final_dataset['validation'])} câu")
    print(f"- Test set:       {len(final_dataset['test'])} câu")
    print("===================================")


if __name__ == "__main__":
    main()