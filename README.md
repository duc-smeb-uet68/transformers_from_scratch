# Transformer from Scratch - Dich may Viet-Anh + finetuning trên bộ dữ liệu y tế


> Bai tap lon mon Xu ly Ngon ngu Tu nhien (NLP).

---

## Muc luc

- [Tong quan](#tong-quan)
- [Kien truc mo hinh](#kien-truc-mo-hinh)
- [Cau truc thu muc](#cau-truc-thu-muc)
- [Cai dat](#cai-dat)
- [Huong dan su dung](#huong-dan-su-dung)
- [Cau hinh](#cau-hinh)
- [Ket qua](#ket-qua)
- [Ky thuat su dung](#ky-thuat-su-dung)

---

## Tong quan

Project trien khai day du kien truc Transformer tu bai bao goc [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), ap dung cho bai toan dich may tu tieng Viet sang tieng Anh.

**Quy trinh:**

1. **Pre-train** tren tap du lieu IWSLT 2015 (Vietnamese-English)
2. **Fine-tune** tren tap du lieu VLSP (linh vuc y te)
3. **Danh gia** bang diem BLEU (sacrebleu)

---

## Kien truc mo hinh

```
Cau tieng Viet
      |
+-----v------+
| Token       |
| Embedding   |--- x sqrt(d_model)
+-----+------+
      |
+-----v------+
| Positional  |
| Encoding    |--- Sin/Cos co dinh
+-----+------+
      |
+-----v------+
|  ENCODER    |--- 6 lop (Self-Attention + FFN + Add&Norm)
+-----+------+
      |
      +--------------------+
                           |
+-----v------+     +-------v--------+
| Masked      |     | Cross-Attention |
| Self-Attn   |     | (Encoder-Decoder)|
+-----+------+     +-------+--------+
      |                     |
+-----v---------------------v------+
|           DECODER                 |--- 6 lop
+-----+----------------------------+
      |
+-----v------+
|  Linear     |
|  + Softmax  |
+-----+------+
      |
  Cau tieng Anh
```

**Thong so mo hinh:**

| Tham so | Gia tri | Mo ta |
|---|---|---|
| `d_model` | 256 | Chieu embedding |
| `n_heads` | 4 | So attention head |
| `n_layers` | 6 | So lop Encoder/Decoder |
| `d_ff` | 1024 | Chieu Feed-Forward an |
| `dropout` | 0.1 | Ti le dropout |
| `max_len` | 128 | Do dai cau toi da |

---

## Cau truc thu muc

```
transformers_from_scratch/
|
|-- model/                      # Core Transformer
|   |-- attention.py            # Multi-Head Attention (Scaled Dot-Product)
|   |-- embeddings.py           # Token Embedding + Positional Encoding
|   |-- layers.py               # EncoderLayer, DecoderLayer, FFN
|   +-- transformer.py          # Encoder, Decoder, Transformer (full model)
|
|-- utils/
|   |-- dataset.py              # BilingualDataset, Collate (padding)
|   +-- tokenizer.py            # Vocabulary wrapper (HuggingFace Tokenizer)
|
|-- build_vocab/                # Script xay dung tu vung
|   |-- build_vocab.py          # Vocab co ban
|   |-- build_shared_vocab.py   # Shared vocab (Vi + En chung)
|   |-- build_vocab_bpe.py      # BPE tokenizer
|   |-- build_vocab_vlsp.py     # Vocab cho VLSP
|   +-- build_data_vlsp.py      # Xu ly du lieu VLSP
|
|-- vocab/                      # File tu vung (JSON)
|   |-- shared_vocab/           # Shared BPE vocab (su dung chinh)
|   |-- vocab0/                 # Vocab rieng (src/tgt)
|   |-- vocab_bpe/              # BPE vocab
|   +-- vlsp_vocab/             # VLSP domain vocab
|
|-- data/                       # Du lieu huan luyen (khong push len git)
|-- weights/                    # Trong so model (khong push len git)
|-- reports/                    # Bieu do Loss/Perplexity
|-- checkpoint/                 # Checkpoint de resume training
|
|-- configs.py                  # Cau hinh sieu tham so
|-- train.py                    # Huan luyen chinh (IWSLT)
|-- fine_tune.py                # Fine-tune tren VLSP
|-- translate.py                # Dich cau tuong tac (Beam Search)
|-- eval_bleu.py                # Danh gia BLEU (Batched Beam Search)
|-- train_demo.py               # Demo training
|-- check_install.py            # Kiem tra cai dat
+-- requirements.txt            # Cac thu vien can thiet
```

---

## Cai dat

### Yeu cau

- Python >= 3.8
- CUDA (khuyen nghi, ho tro GPU NVIDIA)

### Buoc 1: Clone repo

```bash
git clone <repo-url>
cd transformers_from_scratch
```

### Buoc 2: Cai dat thu vien

```bash
pip install -r requirements.txt
```

### Buoc 3: Tai trong so mo hinh

File trong so va data khong duoc push len git vi dung luong lon. Tai tu Google Drive va dat vao thu muc `weights/`:

```
weights/
|-- iwslt_best.pt       # Model pre-trained tren IWSLT
|-- vlsp_best.pt        # Model pre-trained tren VLSP
+-- fine_tune.pt        # Model da fine-tune (su dung chinh)
```

### Buoc 4: Kiem tra cai dat

```bash
python check_install.py
```

---

## Huong dan su dung

### Dich cau (Interactive)

```bash
python translate.py
```

Nhap cau tieng Viet, nhan Enter de nhan ban dich tieng Anh. Nhap `q` de thoat.

```
==============================
MO HINH DICH MAY VIET -> ANH
==============================

Nhap cau tieng Viet (hoac 'q' de thoat): Xin chao, toi la sinh vien
>> Tieng Anh: Hello, I am a student
(Thoi gian: 0.452s)
```

### Huan luyen tu dau

```bash
python train.py
```

Qua trinh huan luyen:
- Tu dong load checkpoint neu co (resume training)
- Su dung Mixed Precision (AMP) de tang toc
- Early Stopping (patience=6) de tranh overfitting
- Luu bieu do Loss/Perplexity vao `reports/`

### Fine-tune tren VLSP

```bash
python fine_tune.py
```

Fine-tune model da pre-train tren tap du lieu VLSP (linh vuc y te) voi learning rate thap (`1e-5`).

### Danh gia BLEU Score

```bash
python eval_bleu.py
```

Su dung Batched Beam Search (beam_width=10, length penalty alpha=0.7) de danh gia tren tap test.

### Xay dung tu vung (Neu can)

```bash
cd build_vocab
python build_shared_vocab.py    # Tao shared BPE vocab
python build_vocab_bpe.py       # Tao BPE vocab rieng
```

---

## Cau hinh

Toan bo cau hinh nam trong `configs.py`:

```python
class Config:
    # Kien truc
    d_model = 256
    n_heads = 4
    n_layers = 6
    d_ff = 1024
    dropout = 0.1

    # Huan luyen
    batch_size = 256
    lr = 0.0005
    warmup_steps = 4000
    label_smoothing = 0.1
    weight_decay = 1e-4
    n_epochs = 60
    clip = 1.0              # Gradient clipping
    patience = 6             # Early stopping

    # Du lieu
    max_len = 128
    vocab_src = 'vocab/shared_vocab/tokenizer_shared.json'
    vocab_tgt = 'vocab/shared_vocab/tokenizer_shared.json'
```

De thay doi model su dung khi dich, chinh `self.model` trong `configs.py`:

```python
self.model = self.finetune_model    # weights/fine_tune.pt
# self.model = self.pretrain_iwslt  # weights/iwslt_best.pt
# self.model = self.pretrain_vlsp   # weights/vlsp_best.pt
```

---

## Ket qua

### Bieu do huan luyen

Cac bieu do Loss va Perplexity duoc luu tu dong vao thu muc `reports/` sau khi huan luyen.

### Du lieu su dung

| Tap du lieu | Mo ta | Muc dich |
|---|---|---|
| IWSLT 2015 | Vietnamese-English, ~133k cap cau | Pre-training |
| VLSP | Vietnamese-English, linh vuc y te | Fine-tuning |

---

## Ky thuat su dung

### Kien truc (theo bai bao goc)

- **Scaled Dot-Product Attention**: `softmax(QK^T / sqrt(d_k)) * V`
- **Multi-Head Attention**: chia Q, K, V thanh nhieu head, tinh attention doc lap roi noi lai
- **Positional Encoding**: Sin/Cos co dinh (khong hoc)
- **Residual Connection + Layer Normalization** tai moi sub-layer
- **Warmup Learning Rate Schedule**: tang dan LR trong `warmup_steps` buoc dau

### Cai tien ky thuat so voi bai bao goc

| Ky thuat | Mo ta |
|---|---|
| **AdamW** | Thay Adam bang AdamW, tach biet weight decay de regularize tot hon |
| **Mixed Precision (AMP)** | Huan luyen FP16 de tang toc va giam VRAM |
| **Early Stopping** | Tu dong dung khi val loss khong giam sau `patience` epoch |
| **Gradient Clipping** | Cat gradient tai `clip=1.0` de tranh exploding gradients |
| **NaN/Inf Check** | Bo qua batch neu loss bi NaN hoac Inf |
| **Weight Tying** | Chia se trong so giua src_embedding, tgt_embedding va output layer |
| **Batched Beam Search** | Beam search xu ly ca batch cung luc, nhanh hon nhieu so voi tung cau |
| **Length Penalty** | `score / (length ^ alpha)` voi `alpha=0.7` khi beam search |
| **Transfer Learning** | Pre-train IWSLT -> Fine-tune VLSP |
| **BPE Tokenizer** | Su dung Byte-Pair Encoding (HuggingFace Tokenizers) thay vi word-level |

### Cong nghe

| Thu vien | Phien ban | Vai tro |
|---|---|---|
| PyTorch | 2.3.1 | Framework deep learning |
| HuggingFace Datasets | 2.19.1 | Load va xu ly du lieu |
| HuggingFace Tokenizers | - | BPE Tokenizer |
| sacrebleu | 2.5.1 | Tinh diem BLEU |
| spaCy | 3.8.11 | Xu ly ngon ngu |
| matplotlib | 3.10.7 | Ve bieu do |

---

## Tham khao

- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *NeurIPS 2017*.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Jay Alammar.
- [Harvard NLP - The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).
