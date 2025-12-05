import torch
import json
from collections import Counter
from torchtext.data.utils import get_tokenizer
from tokenizers import Tokenizer

PAD_TOKEN = '<pad>' # Dùng để đệm câu cho bằng độ dài
SOS_TOKEN = '<sos>' # Start of Sentence - Báo hiệu bắt đầu câu
EOS_TOKEN = '<eos>' # End of Sentence - Báo hiệu kết thúc câu
UNK_TOKEN = '<unk>' # Unknown - Dùng cho các từ không có trong từ điển

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

class Vocabulary:
    def __init__(self, vocab_file):
        # Load tokenizer đã train bằng build_vocab_bpe.py
        self.tokenizer = Tokenizer.from_file(vocab_file)

        # Tạo mapping stoi (String to Index) để tương thích với Dataset cũ
        self.stoi = self.tokenizer.get_vocab()

        # Tạo mapping itos (Index to String)
        self.itos = {v: k for k, v in self.stoi.items()}

        # ID của các token đặc biệt
        self.pad_idx = self.stoi.get("<pad>", 0)
        self.sos_idx = self.stoi.get("<sos>", 1)
        self.eos_idx = self.stoi.get("<eos>", 2)
        self.unk_idx = self.stoi.get("<unk>", 3)

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def numericalize(self, text, lang=None):
        """
        Chuyển văn bản thành List ID.
        Lưu ý: Không thêm <sos>, <eos> ở đây vì Dataset.py sẽ làm việc đó.
        """
        # Encode text
        output = self.tokenizer.encode(text)
        return output.ids

    def decode(self, tokens_ids):
        """Chuyển List ID về văn bản"""
        # skip_special_tokens=True để loại bỏ <pad>, <sos>... khi in ra
        return self.tokenizer.decode(tokens_ids, skip_special_tokens=True)