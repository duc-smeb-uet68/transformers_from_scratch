import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
import random
import math
import time
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from model.transformer import Transformer

# =====================
# CONFIG
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 20
LR = 5e-5
CLIP = 1.0
PATIENCE = 5

D_MODEL = 512
N_LAYERS = 6
N_HEADS = 8
D_FF = 2048
DROPOUT = 0.1

SRC_TOKENIZER = "data/vocab_bpe/tokenizer_vi.json"
TGT_TOKENIZER = "data/vocab_bpe/tokenizer_en.json"

PRETRAIN_CKPT = "transformer_last.pt"
SAVE_CKPT = "transformer_bpe_finetune_vlsp.pt"

TRAIN_SRC = "data/vlsp/train/train.vi.txt"
TRAIN_TGT = "data/vlsp/train/train.en.txt"

# =====================
# DATASET
# =====================
class BPETranslationDataset(Dataset):
    def __init__(self, src, tgt, sp_src, sp_tgt, max_len):
        self.src = src
        self.tgt = tgt
        self.sp_src = sp_src
        self.sp_tgt = sp_tgt
        self.max_len = max_len

        self.sos = sp_src.token_to_id("<sos>")
        self.eos = sp_src.token_to_id("<eos>")

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_ids = self.sp_src.encode(self.src[idx]).ids[: self.max_len - 2]
        tgt_ids = self.sp_tgt.encode(self.tgt[idx]).ids[: self.max_len - 2]

        src = [self.sos] + src_ids + [self.eos]
        tgt = [self.sos] + tgt_ids + [self.eos]

        return torch.tensor(src), torch.tensor(tgt)


def collate_fn(batch, pad_src, pad_tgt):
    src, tgt = zip(*batch)
    src = pad_sequence(src, batch_first=True, padding_value=pad_src)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=pad_tgt)
    return src, tgt


# =====================
# UTIL
# =====================
def load_parallel(src_path, tgt_path):
    with open(src_path, encoding="utf-8") as f:
        src = [l.strip() for l in f if l.strip()]
    with open(tgt_path, encoding="utf-8") as f:
        tgt = [l.strip() for l in f if l.strip()]
    return src, tgt


# =====================
# TRAIN / EVAL
# =====================
def train_epoch(model, loader, opt, crit, scaler):
    model.train()
    total = 0
    for src, tgt in tqdm(loader, leave=False):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        opt.zero_grad()
        with autocast():
            out = model(src, tgt[:, :-1])
            loss = crit(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        scaler.step(opt)
        scaler.update()
        total += loss.item()
    return total / len(loader)


def eval_epoch(model, loader, crit):
    model.eval()
    total = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            out = model(src, tgt[:, :-1])
            loss = crit(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
            total += loss.item()
    return total / len(loader)


# =====================
# MAIN
# =====================
def main():
    print("Device:", DEVICE)

    src_text, tgt_text = load_parallel(TRAIN_SRC, TRAIN_TGT)
    data = list(zip(src_text, tgt_text))
    random.shuffle(data)

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    sp_src = Tokenizer.from_file(SRC_TOKENIZER)
    sp_tgt = Tokenizer.from_file(TGT_TOKENIZER)

    pad_src = sp_src.token_to_id("<pad>")
    pad_tgt = sp_tgt.token_to_id("<pad>")

    train_ds = BPETranslationDataset(*zip(*train_data), sp_src, sp_tgt, MAX_LEN)
    val_ds = BPETranslationDataset(*zip(*val_data), sp_src, sp_tgt, MAX_LEN)

    train_dl = DataLoader(
        train_ds, BATCH_SIZE, True,
        collate_fn=lambda b: collate_fn(b, pad_src, pad_tgt)
    )
    val_dl = DataLoader(
        val_ds, BATCH_SIZE, False,
        collate_fn=lambda b: collate_fn(b, pad_src, pad_tgt)
    )

    model = Transformer(
        sp_src.get_vocab_size(),
        sp_tgt.get_vocab_size(),
        D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT,
        5000, pad_src, pad_tgt
    ).to(DEVICE)

    model.load_state_dict(torch.load(PRETRAIN_CKPT, map_location=DEVICE))

    opt = optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(ignore_index=pad_tgt)
    scaler = GradScaler()

    best = float("inf")
    patience = 0

    for ep in range(EPOCHS):
        t0 = time.time()
        tr = train_epoch(model, train_dl, opt, crit, scaler)
        va = eval_epoch(model, val_dl, crit)
        print(f"Epoch {ep+1:02} | Train {tr:.3f} | Val {va:.3f} | PPL {math.exp(va):.2f} | {int(time.time()-t0)}s")

        if va < best:
            best = va
            patience = 0
            torch.save(model.state_dict(), SAVE_CKPT)
            print("✔ Saved best model")
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    print("✅ FINETUNE VLSP DONE")


if __name__ == "__main__":
    main()
