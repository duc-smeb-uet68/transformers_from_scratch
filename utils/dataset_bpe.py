import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class BilingualDatasetBPE(Dataset):
    def __init__(
        self,
        src_sentences,
        tgt_sentences,
        src_tokenizer,
        tgt_tokenizer,
        max_len=256
    ):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

        # Lấy special token ids
        self.src_pad = src_tokenizer.token_to_id("<pad>")
        self.src_sos = src_tokenizer.token_to_id("<sos>")
        self.src_eos = src_tokenizer.token_to_id("<eos>")

        self.tgt_pad = tgt_tokenizer.token_to_id("<pad>")
        self.tgt_sos = tgt_tokenizer.token_to_id("<sos>")
        self.tgt_eos = tgt_tokenizer.token_to_id("<eos>")

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_text = self.src_sentences[idx]
        tgt_text = self.tgt_sentences[idx]

        # Encode bằng BPE tokenizer
        src_ids = self.src_tokenizer.encode(src_text).ids
        tgt_ids = self.tgt_tokenizer.encode(tgt_text).ids

        # Truncation (chừa <sos>, <eos>)
        max_seq_len = self.max_len - 2
        src_ids = src_ids[:max_seq_len]
        tgt_ids = tgt_ids[:max_seq_len]

        # Add special tokens
        src_out = [self.src_sos] + src_ids + [self.src_eos]
        tgt_out = [self.tgt_sos] + tgt_ids + [self.tgt_eos]

        return torch.tensor(src_out), torch.tensor(tgt_out)


class CollateBPE:
    def __init__(self, src_pad_idx, tgt_pad_idx):
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def __call__(self, batch):
        src_batch, tgt_batch = zip(*batch)

        src_batch = pad_sequence(
            src_batch,
            batch_first=True,
            padding_value=self.src_pad_idx
        )
        tgt_batch = pad_sequence(
            tgt_batch,
            batch_first=True,
            padding_value=self.tgt_pad_idx
        )

        return src_batch, tgt_batch
