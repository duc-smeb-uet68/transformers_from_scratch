import os
from datasets import load_from_disk, load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

if not os.path.exists('../data/shared_vocab'):
    os.makedirs('../data/shared_vocab')

def train_shared_tokenizer(iterator_src, iterator_tgt, save_path, vocab_size=16000):
    # 1. Khởi tạo BPE
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # 2. CẢI TIẾN: Normalization (Quan trọng cho Tiếng Việt)
    # NFKC: Chuẩn hóa Unicode (hòa -> hoà về cùng 1 mã)
    # Lowercase: Chuyển về chữ thường (tùy chọn, giúp model học nhanh hơn với data ít)
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.Lowercase()
    ])

    # 3. Pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 4. Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # 5. Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )

    # 6. Train trên cả 2 nguồn dữ liệu (Generator chain)
    # Hàm này nối dữ liệu generator lại với nhau
    def combined_iterator():
        for text in iterator_src:
            yield text
        for text in iterator_tgt:
            yield text

    print(f"--- Đang train Shared Tokenizer (Vocab: {vocab_size})...")
    tokenizer.train_from_iterator(combined_iterator(), trainer=trainer)

    # 7. Lưu
    tokenizer.save(save_path)
    print(f"Đã lưu Shared Tokenizer tại: {save_path}")

def get_training_corpus(dataset, key):
    for i in range(0, len(dataset), 1000):
        batch = dataset[i: i + 1000][key]
        for line in batch:
            yield line

if __name__ == "__main__":
    print("--- Đang tải dataset ---")
    try:
        dataset = load_from_disk("../data/iwslt2015_data")
        train_data = dataset['train']
    except:
        dataset = load_dataset("nguyenvuhuy/iwslt2015-en-vi")
        train_data = dataset['train']

    # CẢI TIẾN: Train 1 tokenizer chung cho cả 2 ngôn ngữ
    # Giúp model học được mối liên hệ giữa các từ mượn/tên riêng tốt hơn
    train_shared_tokenizer(
        get_training_corpus(train_data, 'vi'),
        get_training_corpus(train_data, 'en'),
        '../data/shared_vocab/tokenizer_shared.json',
        vocab_size=10000 # Tăng nhẹ lên 10k hoặc 16k vì chứa cả 2 ngôn ngữ
    )