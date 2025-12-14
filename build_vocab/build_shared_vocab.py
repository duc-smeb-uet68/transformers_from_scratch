from datasets import load_from_disk, load_dataset
from utils.shared_bpe_vocab import train_shared_bpe_tokenizer


def get_training_corpus(dataset, key, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        for line in dataset[i:i + batch_size][key]:
            yield line


if __name__ == "__main__":
    print("--- Loading IWSLT2015 dataset ---")

    try:
        dataset = load_from_disk("data/iwslt2015_data")
        train_data = dataset["train"]
    except:
        dataset = load_dataset("nguyenvuhuy/iwslt2015-en-vi")
        train_data = dataset["train"]

    print("--- Training shared EN-VI BPE tokenizer ---")

    train_shared_bpe_tokenizer(
        iterator_src=get_training_corpus(train_data, "en"),
        iterator_tgt=get_training_corpus(train_data, "vi"),
        save_path="data/shared_vocab/tokenizer_shared.json",
        vocab_size=10000,   # 10k–16k là hợp lý cho EN-VI
        lowercase=True
    )
