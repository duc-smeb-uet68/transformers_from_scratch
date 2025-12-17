import torch
import torch.nn as nn

# Import các module đã viết trước đó
# Lưu ý: Đảm bảo cậu đã tạo file __init__.py trong thư mục models
from .embeddings import TokenEmbedding, PositionalEncoding
from .layers import EncoderLayer, DecoderLayer


class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff, dropout):
        super(Encoder, self).__init__()
        # Tạo danh sách N lớp EncoderLayer
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask):
        # src đi qua từng lớp EncoderLayer lần lượt
        for layer in self.layers:
            src = layer(src, mask)
        return self.norm(src)


class Decoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff, dropout):
        super(Decoder, self).__init__()
        # Tạo danh sách N lớp DecoderLayer
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask, src_mask):
        """
        tgt: Input của Decoder (câu đang dịch)
        memory: Output của Encoder (thông tin từ câu nguồn)
        """
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, src_mask)
        return self.norm(tgt)


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            d_model=512,
            n_layers=6,
            n_heads=8,
            d_ff=2048,
            dropout=0.1,
            max_len=5000,
            src_pad_idx=0,
            tgt_pad_idx=0
    ):
        super(Transformer, self).__init__()

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Khởi tạo phần Embedding + Positional Encoding
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        # 2. Khởi tạo khối Encoder và Decoder
        self.encoder = Encoder(d_model, n_layers, n_heads, d_ff, dropout)
        self.decoder = Decoder(d_model, n_layers, n_heads, d_ff, dropout)

        # 3. Lớp đầu ra (Generator): Chiếu về kích thước từ điển đích
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        # Khởi tạo tham số (Xavier init) giúp hội tụ nhanh hơn
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        # Tạo mask cho Encoder: Che các vị trí padding
        # src shape: (batch_size, src_len)
        # mask shape: (batch_size, 1, 1, src_len)
        mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return mask.to(self.device)

    def make_tgt_mask(self, tgt):
        # Tạo mask cho Decoder: Gồm 2 phần
        # 1. Padding mask: Che các vị trí padding
        # 2. Look-ahead mask: Che các từ tương lai (dạng tam giác trên)

        # Padding mask
        padding_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(3)

        # Look-ahead mask (tam giác)
        tgt_len = tgt.shape[1]
        look_ahead_mask = torch.tril(torch.ones(tgt_len, tgt_len)).type(torch.ByteTensor).to(self.device)

        # Kết hợp cả 2: Vừa phải không phải padding, vừa phải nằm trong quá khứ
        mask = padding_mask & look_ahead_mask

        return mask

    def forward(self, src, tgt):
        # 1. Tạo Mask
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        # 2. Forward qua Encoder
        # Input: (Batch, Src_Len) -> Output: (Batch, Src_Len, D_Model)
        src_emb = self.positional_encoding(self.src_embedding(src))
        enc_src = self.encoder(src_emb, src_mask)

        # 3. Forward qua Decoder
        # Input: (Batch, Tgt_Len) -> Output: (Batch, Tgt_Len, D_Model)
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt))
        output = self.decoder(tgt_emb, enc_src, tgt_mask, src_mask)

        # 4. Chiếu ra xác suất từ
        return self.fc_out(output)