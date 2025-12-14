from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
import os


def train_shared_bpe_tokenizer(
    iterator_src,
    iterator_tgt,
    save_path,
    vocab_size=16000,
    lowercase=True
):
    """
    Train shared BPE tokenizer cho cả source & target (EN-VI)

    iterator_src, iterator_tgt: generator trả về từng câu
    save_path: đường dẫn lưu tokenizer (.json)
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 1. Model BPE
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # 2. Normalizer (rất quan trọng cho tiếng Việt)
    norms = [normalizers.NFKC()]
    if lowercase:
        norms.append(normalizers.Lowercase())

    tokenizer.normalizer = normalizers.Sequence(norms)

    # 3. Pre-tokenizer & Decoder
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # 4. Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )

    # 5. Gộp iterator 2 ngôn ngữ
    def combined_iterator():
        for text in iterator_src:
            yield text
        for text in iterator_tgt:
            yield text

    print(f"--- Training Shared BPE Tokenizer (vocab_size={vocab_size}) ---")
    tokenizer.train_from_iterator(combined_iterator(), trainer=trainer)

    tokenizer.save(save_path)
    print(f"✔ Saved shared tokenizer to: {save_path}")
