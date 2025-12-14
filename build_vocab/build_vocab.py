from utils.word_vocab import Vocabulary
from datasets import load_dataset

def extract_data(data_split):
    src = [item['vi'] for item in data_split]
    tgt = [item['en'] for item in data_split]
    return src, tgt


if __name__ == "__main__":
    print("--- Đang tải dataset IWSLT2015 (Vi-En) ---")
    dataset = load_dataset("nguyenvuhuy/iwslt2015-en-vi")
    dataset.save_to_disk("data/iwslt2015_data")

    train_src, train_tgt = extract_data(dataset['train'])

    # Tokenizer & Vocab
    # Removed: from utils.tokenizer import Vocabulary
    vocab_src = Vocabulary(freq_threshold=1)
    vocab_tgt = Vocabulary(freq_threshold=1)
    # Build vocab từ dữ liệu train
    vocab_src.build_vocabulary(train_src, lang='vi')
    vocab_src.save_vocab('data/vocab0/vocab_src.json')

    vocab_tgt.build_vocabulary(train_tgt, lang='en')
    vocab_tgt.save_vocab('data/vocab0/vocab_tgt.json')

    print(f"Vocab Source: {len(vocab_src)} | Vocab Target: {len(vocab_tgt)}")
