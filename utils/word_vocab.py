from collections import Counter
import re
import json

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

class Vocabulary:
    def __init__(self, freq_threshold=1):
        self.freq_threshold = freq_threshold

        self.itos = {
            0: PAD_TOKEN,
            1: SOS_TOKEN,
            2: EOS_TOKEN,
            3: UNK_TOKEN
        }
        self.stoi = {v: k for k, v in self.itos.items()}

        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        # Giữ nguyên token đặc biệt
        if text in {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN}:
            return [text]

        return re.findall(r"<[^>]+>|\w+|[^\w\s]", text.lower())


    def build_vocabulary(self, sentence_list, lang=None):
        frequencies = Counter()

        for sentence in sentence_list:
            tokens = self.tokenize(sentence)
            frequencies.update(tokens)

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold and word not in self.stoi:
                idx = len(self.itos)
                self.stoi[word] = idx
                self.itos[idx] = word

    def numericalize(self, text):
        tokens = self.tokenize(text)
        return [
            self.stoi.get(token, self.unk_idx)
            for token in tokens
        ]

    def decode(self, token_ids):
        tokens = []
        for idx in token_ids:
            token = self.itos.get(idx, UNK_TOKEN)
            if token in {PAD_TOKEN, SOS_TOKEN}:
                continue
            if token == EOS_TOKEN:
                break
            tokens.append(token)
        return " ".join(tokens)


    def save_vocab(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_vocab(cls, path):
        vocab = cls()
        with open(path, "r", encoding="utf-8") as f:
            vocab.stoi = json.load(f)
        vocab.itos = {int(v): k for k, v in vocab.stoi.items()}
        return vocab
