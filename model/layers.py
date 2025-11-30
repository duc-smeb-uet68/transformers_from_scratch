import torch
import torch.nn as nn
from .attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    """
    Mạng Feed-Forward (FFN) được áp dụng cho từng vị trí riêng biệt và giống hệt nhau.
    Công thức: FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # Mở rộng kích thước từ d_model lên d_ff (thường gấp 4 lần, ví dụ 512 -> 2048)
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x -> Linear -> ReLU -> Dropout -> Linear
        return self.w_2(self.dropout(self.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    """
    Một lớp Encoder bao gồm 2 phần chính:
    1. Multi-Head Self-Attention
    2. Position-wise Feed-Forward Network
    Mỗi phần đều có Residual Connection (Add) và Layer Normalization (Norm).
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # [cite_start]Thành phần 1: Self-Attention [cite: 17]
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        # [cite_start]
        self.norm1 = nn.LayerNorm(d_model)  # [cite: 18]

        # [cite_start]Thành phần 2: Feed Forward [cite: 19]
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # 1. Sub-layer 1: Self-Attention
        # Residual connection: x + Sublayer(x)
        _src = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(_src))

        # 2. Sub-layer 2: Feed Forward
        _src = self.ffn(src)
        src = self.norm2(src + self.dropout(_src))

        return src


class DecoderLayer(nn.Module):
    """
    Một lớp Decoder phức tạp hơn, bao gồm 3 phần:
    1. Masked Multi-Head Self-Attention (để không nhìn thấy tương lai)
    2. Multi-Head Cross-Attention (nhìn vào Encoder output)
    3. Position-wise Feed-Forward Network
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        # [cite_start]1. Masked Self-Attention [cite: 21]
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)

        # [cite_start]2. Cross-Attention (Encoder-Decoder Attention) [cite: 22]
        # Query lấy từ Decoder, Key & Value lấy từ Encoder
        self.enc_dec_attn = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)

        # [cite_start]3. Feed Forward [cite: 23]
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_out, tgt_mask, src_mask):
        # Sub-layer 1: Masked Self-Attention (chú ý vào chính câu đích)
        _tgt = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(_tgt))

        # Sub-layer 2: Cross-Attention (chú ý vào câu nguồn - Encoder Output)
        # query=tgt, key=enc_out, value=enc_out
        _tgt = self.enc_dec_attn(tgt, enc_out, enc_out, src_mask)
        tgt = self.norm2(tgt + self.dropout(_tgt))

        # Sub-layer 3: Feed Forward
        _tgt = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(_tgt))

        return tgt