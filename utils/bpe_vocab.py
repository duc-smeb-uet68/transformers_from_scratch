from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def train_bpe_tokenizer(
    texts,
    vocab_size=8000,
    special_tokens=None
):
    """
    Train một BPE tokenizer từ iterator văn bản

    Args:
        texts: iterator các câu (list hoặc generator)
        vocab_size: kích thước vocab
        special_tokens: danh sách token đặc biệt

    Returns:
        tokenizer (huggingface Tokenizer)
    """
    if special_tokens is None:
        special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]

    # 1. Khởi tạo BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # 2. Pre-tokenizer (ByteLevel phù hợp cho cả Vi & En)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    # 3. Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # 4. Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 5. Train
    tokenizer.train_from_iterator(texts, trainer=trainer)

    return tokenizer
