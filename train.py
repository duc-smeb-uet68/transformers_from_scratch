import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Project imports
from model.transformer import Transformer
from utils.dataset import BilingualDataset, Collate
from utils.word_vocab import Vocabulary
from configs import cfg

# ======================================================
# 1. Early Stopping
# ======================================================
class EarlyStopping:
    def __init__(self, patience=3, delta=0, path='finetune_best.pt', verbose=True):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Val loss improved ({self.val_loss_min:.4f} → {val_loss:.4f}). Saving model.")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ======================================================
# 2. Scheduler
# ======================================================
class TransformerScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = lr

    def rate(self):
        step = max(self.step_num, 1)
        return self.d_model ** (-0.5) * min(
            step ** (-0.5),
            step * self.warmup_steps ** (-1.5)
        )

    def zero_grad(self):
        self.optimizer.zero_grad()


# ======================================================
# 3. Train / Eval
# ======================================================
def train_epoch(model, loader, optimizer, scheduler, criterion, clip, device, scaler):
    model.train()
    epoch_loss = 0

    for src, tgt in tqdm(loader, desc="Training", leave=False):
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        scheduler.zero_grad()

        with autocast():
            output = model(src, tgt_input)
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)
            loss = criterion(output, tgt_output)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    loss = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = model(src, tgt_input)
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)
            loss += criterion(output, tgt_output).item()
    return loss / len(loader)


# ======================================================
# 4. Plot
# ======================================================
def save_plots(train_losses, val_losses, prefix):
    os.makedirs("reports", exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.savefig(f"reports/{prefix}_loss.png")
    plt.close()


# ======================================================
# 5. MAIN FINETUNE
# ======================================================
def run_finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --------------------------
    # Load VLSP data
    # --------------------------
    print("Loading VLSP dataset...")
    dataset = load_from_disk(cfg.vlsp_data_path)

    src_texts = [x[cfg.src_lang] for x in dataset]
    tgt_texts = [x[cfg.tgt_lang] for x in dataset]

    # --------------------------
    # Load BPE vocab (MUST match pretrain)
    # --------------------------
    bpe_cfg = [v for v in cfg.vocab_configs if v["name"] == "bpe"][0]
    vocab_src = Vocabulary.load_vocab(bpe_cfg["src_path"])
    vocab_tgt = Vocabulary.load_vocab(bpe_cfg["tgt_path"])

    full_dataset = BilingualDataset(
        src_texts, tgt_texts, vocab_src, vocab_tgt, max_len=cfg.max_len
    )

    # --------------------------
    # Split train / valid (90/10)
    # --------------------------
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    collate = Collate(vocab_src.pad_idx, vocab_tgt.pad_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

    # --------------------------
    # Model
    # --------------------------
    model = Transformer(
        src_vocab_size=len(vocab_src),
        tgt_vocab_size=len(vocab_tgt),
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_len=5000,
        src_pad_idx=vocab_src.pad_idx,
        tgt_pad_idx=vocab_tgt.pad_idx
    ).to(device)

    # Share embeddings
    model.src_embedding.emb.weight = model.tgt_embedding.emb.weight
    model.fc_out.weight = model.src_embedding.emb.weight

    # --------------------------
    # LOAD PRETRAINED BPE MODEL
    # --------------------------
    print("Loading pretrained model: transformer_last.pt")
    model.load_state_dict(torch.load("transformer_last.pt", map_location=device))


    # --------------------------
    # Optimizer & Loss (FINETUNE SETTINGS)
    # --------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-5,            # SMALL LR
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=1e-4
    )

    scheduler = TransformerScheduler(optimizer, cfg.d_model, cfg.warmup_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_tgt.pad_idx, label_smoothing=cfg.label_smoothing)
    early_stopping = EarlyStopping(patience=cfg.patience, path=cfg.vlsp_ckpt_path)
    scaler = GradScaler()

    train_losses, val_losses = [], []

    # --------------------------
    # FINETUNE LOOP
    # --------------------------
    for epoch in range(cfg.n_epochs):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, cfg.clip, device, scaler)
        val_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        mins = int((time.time() - start) / 60)
        print(f"Epoch {epoch+1:02} | {mins}m | Train {train_loss:.3f} | Val {val_loss:.3f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    save_plots(train_losses, val_losses, cfg.vlsp_plot_prefix)
    print("Finetuning finished.")
    print("Best model saved at:", cfg.vlsp_ckpt_path)


if __name__ == "__main__":
    run_finetune()
