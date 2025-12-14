import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


PAD_TOKEN = '<pad>' # Dùng để đệm câu cho bằng độ dài
SOS_TOKEN = '<sos>' # Start of Sentence - Báo hiệu bắt đầu câu
EOS_TOKEN = '<eos>' # End of Sentence - Báo hiệu kết thúc câu
UNK_TOKEN = '<unk>' # Unknown - Dùng cho các từ không có trong từ điển

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

class BilingualDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=256):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        # Lưu lại độ dài tối đa cho phép (khớp với model)
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, index):
        src_text = self.src_sentences[index]
        tgt_text = self.tgt_sentences[index]

        # 1. Chuyển từ -> số (Numericalize)
        # Lưu ý: Ta lấy list số trước, chưa thêm SOS/EOS vội
        src_indices = self.src_vocab.numericalize(src_text)
        tgt_indices = self.tgt_vocab.numericalize(tgt_text)


        # 2. LOGIC CẮT CÂU (TRUNCATION) - QUAN TRỌNG
        # Ta cần chừa lại 2 vị trí cho <sos> và <eos>
        # Ví dụ: max_len=256 thì nội dung chỉ được dài 254
        max_seq_len = self.max_len - 2

        # Nếu câu dài hơn giới hạn, ta cắt bớt phần đuôi
        if len(src_indices) > max_seq_len:
            src_indices = src_indices[:max_seq_len]

        if len(tgt_indices) > max_seq_len:
            tgt_indices = tgt_indices[:max_seq_len]

        # 3. Thêm <sos> và <eos> vào đầu và cuối
        # Lúc này đảm bảo tổng độ dài luôn <= self.max_len
        src_out = [self.src_vocab.stoi["<sos>"]] + src_indices + [self.src_vocab.stoi["<eos>"]]
        tgt_out = [self.tgt_vocab.stoi["<sos>"]] + tgt_indices + [self.tgt_vocab.stoi["<eos>"]]

        return torch.tensor(src_out), torch.tensor(tgt_out)

class Collate:
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
