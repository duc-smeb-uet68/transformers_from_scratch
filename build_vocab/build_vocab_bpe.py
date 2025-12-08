import os
from datasets import load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# Tạo thư mục chứa vocab nếu chưa có
if not os.path.exists('data/vocab_bpe'):
    os.makedirs('data/vocab_bpe')


def train_tokenizer(texts, save_path, vocab_size=10000):
    # 1. Khởi tạo BPE Tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # 2. Tiền xử lý: Tách theo byte (cho tiếng Việt/Anh đều ổn)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 3. Decoding: Ghép lại
    tokenizer.decoder = decoders.ByteLevel()

    # 4. Cấu hình Trainer
    # Quan trọng: Thêm các token đặc biệt vào
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 5. Train
    print(f"--- Đang train tokenizer cho {save_path} ...")
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 6. Lưu
    tokenizer.save(save_path)
    print(f"Đã lưu tokenizer tại: {save_path}")


def get_training_corpus(dataset, key):
    for i in range(0, len(dataset), 1000):
        yield dataset[i: i + 1000][key]


if __name__ == "__main__":
    print("--- Đang tải dataset ---")
    # Giả sử bạn đã chạy save_to_disk ở bước trước
    try:
        dataset = load_from_disk("../data/iwslt2015_data")
        train_data = dataset['train']
    except:
        from datasets import load_dataset

        dataset = load_dataset("nguyenvuhuy/iwslt2015-en-vi")
        train_data = dataset['train']

    # Train Tokenizer tiếng Việt
    train_tokenizer(
        get_training_corpus(train_data, 'vi'),
        '../data/vocab_bpe/tokenizer_vi.json',
        vocab_size=8000  # Tiếng Việt
    )

    # Train Tokenizer tiếng Anh
    train_tokenizer(
        get_training_corpus(train_data, 'en'),
        '../data/vocab_bpe/tokenizer_en.json',
        vocab_size=8000  # Tiếng Anh
    )