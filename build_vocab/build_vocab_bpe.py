import os
from datasets import load_dataset, load_from_disk
from utils.bpe_vocab import train_bpe_tokenizer


def get_training_corpus(dataset, key, batch_size=1000):
    """
    Generator trả về các batch text để train tokenizer
    """
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size][key]


if __name__ == "__main__":
    print("--- Đang tải dataset IWSLT2015 (Vi-En) ---")

    # Load dataset (ưu tiên từ disk)
    try:
        dataset = load_from_disk("data/iwslt2015_data")
        train_data = dataset["train"]
    except:
        dataset = load_dataset("nguyenvuhuy/iwslt2015-en-vi")
        dataset.save_to_disk("data/iwslt2015_data")
        train_data = dataset["train"]

    # Tạo thư mục lưu vocab
    os.makedirs("data/vocab_bpe", exist_ok=True)

    # ========================
    # Train BPE tokenizer VI
    # ========================
    print("\n--- Training BPE tokenizer (VI) ---")
    tokenizer_vi = train_bpe_tokenizer(
        get_training_corpus(train_data, "vi"),
        vocab_size=8000
    )
    tokenizer_vi.save("data/vocab_bpe/tokenizer_vi.json")
    print("✔ Đã lưu tokenizer_vi.json")

    # ========================
    # Train BPE tokenizer EN
    # ========================
    print("\n--- Training BPE tokenizer (EN) ---")
    tokenizer_en = train_bpe_tokenizer(
        get_training_corpus(train_data, "en"),
        vocab_size=8000
    )
    tokenizer_en.save("data/vocab_bpe/tokenizer_en.json")
    print("✔ Đã lưu tokenizer_en.json")

    print("\n=== HOÀN TẤT TRAIN BPE VOCAB ===")
