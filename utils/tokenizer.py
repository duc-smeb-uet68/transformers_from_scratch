import torch
import json
from collections import Counter
from torchtext.data.utils import get_tokenizer


PAD_TOKEN = '<pad>' # Dùng để đệm câu cho bằng độ dài
SOS_TOKEN = '<sos>' # Start of Sentence - Báo hiệu bắt đầu câu
EOS_TOKEN = '<eos>' # End of Sentence - Báo hiệu kết thúc câu
UNK_TOKEN = '<unk>' # Unknown - Dùng cho các từ không có trong từ điển

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_en(text):
        # Sử dụng tokenizer chuẩn cho tiếng Anh
        tokenizer = get_tokenizer('basic_english')
        return tokenizer(text)

    @staticmethod
    def tokenizer_vi(text):
        # CẢI TIẾN: Hàm tách từ tiếng Việt
        # Nếu cài được thư viện 'pyvi' hoặc 'underthesea' thì thay vào đây
        # Tạm thời dùng split() nhưng xử lý kỹ hơn về dấu câu
        text = text.lower().strip()
        # Tách dấu câu đơn giản (để dấu chấm, phẩy không dính vào từ)
        for char in ['.', ',', '?', '!', '"', "'"]:
            text = text.replace(char, f" {char} ")
        return text.split()

    def build_vocabulary(self, sentence_list, lang='en'):
        frequencies = Counter()
        idx = 4

        tokenizer_fn = self.tokenizer_en if lang == 'en' else self.tokenizer_vi

        for sentence in sentence_list:
            for word in tokenizer_fn(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text, lang='en'):
        tokenizer_fn = self.tokenizer_en if lang == 'en' else self.tokenizer_vi
        tokenized_text = tokenizer_fn(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in tokenized_text
        ]

    # Thêm chức năng lưu/tải từ điển để không phải build lại
    def save_vocab(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.stoi, f, ensure_ascii=False)

    def load_vocab(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.stoi = json.load(f)
            self.itos = {v: k for k, v in self.stoi.items()}