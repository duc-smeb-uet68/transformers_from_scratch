import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        """
        Args:
            d_model: Kích thước vector embedding (ví dụ: 512)
            n_heads: Số lượng 'đầu' chú ý (ví dụ: 8)
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"

        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model

        # 1. Các lớp Linear để tạo ra Q, K, V từ đầu vào
        # W_q, W_k, W_v trong lý thuyết
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Lớp Linear cuối cùng sau khi nối các heads lại
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (Batch, Seq_len, D_model)
            key:   (Batch, Seq_len, D_model)
            value: (Batch, Seq_len, D_model)
            mask:  Tensor chứa giá trị 0 hoặc 1 để che đi các vị trí không cần thiết
        """
        batch_size = query.shape[0]

        # 1. Tính toán Q, K, V
        Q = self.w_q(query) # (Batch, Seq_len, D_model)
        K = self.w_k(key)   # (Batch, Seq_len, D_model)
        V = self.w_v(value) # (Batch, Seq_len, D_model)

        # 2. Chia nhỏ thành n_heads
        # Biến đổi: (Batch, Seq_len, D_model) -> (Batch, Seq_len, n_heads, d_head)
        # Sau đó đảo trục để n_heads lên trước: (Batch, n_heads, Seq_len, d_head)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        # Tính điểm năng lượng: (Batch, n_heads, Seq_len_Q, Seq_len_K)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 4. Áp dụng Mask (Nếu có)
        # Mask thường dùng để:
        # - Che padding (Padding Mask)
        # - Che các từ tương lai trong Decoder (Look-ahead Mask)
        if mask is not None:
            # Những chỗ mask == 0 sẽ bị gán giá trị âm vô cùng (-1e9)
            # Để khi qua Softmax nó sẽ bằng 0
            scores = scores.masked_fill(mask == 0, -1e4)

        # 5. Softmax để ra xác suất chú ý
        attention_weights = torch.softmax(scores, dim=-1)

        # 6. Nhân với V
        out = torch.matmul(attention_weights, V) # (Batch, n_heads, Seq_len_Q, d_head)

        # 7. Ghép lại (Concatenate)
        # Đảo trục lại: (Batch, Seq_len_Q, n_heads, d_head)
        out = out.transpose(1, 2).contiguous()

        out = out.view(batch_size, -1, self.d_model)

        out = self.fc_out(out)

        return out