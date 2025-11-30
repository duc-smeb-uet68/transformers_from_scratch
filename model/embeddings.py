import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.d_model = d_model
        # Lớp Embedding chuẩn của PyTorch
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Theo paper gốc "Attention Is All You Need", ta nhân embedding với căn bậc 2 của d_model
        # Lý do: Để giá trị của embedding có độ lớn tương đương với Positional Encoding sắp cộng vào
        return self.emb(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: Kích thước vector embedding (thường là 512)
            max_len: Độ dài tối đa của câu mà mô hình hỗ trợ
            dropout: Xác suất dropout để tránh overfitting
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Tạo ma trận PE kích thước (max_len, d_model) chứa toàn số 0
        pe = torch.zeros(max_len, d_model)

        # Tạo vector vị trí (pos): [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Tính div_term (mẫu số trong công thức): 10000^(2i/d_model)
        # Sử dụng log space để tính toán ổn định hơn về mặt số học
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Áp dụng công thức Sin cho vị trí chẵn (2i)
        pe[:, 0::2] = torch.sin(position * div_term)

        # Áp dụng công thức Cos cho vị trí lẻ (2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Thêm 1 chiều batch ở đầu: (1, max_len, d_model) để dễ cộng với input
        pe = pe.unsqueeze(0)

        # register_buffer giúp lưu trữ tensor này vào state_dict của mô hình
        # nhưng không cập nhật nó trong quá trình backpropagation (vì nó cố định)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor kích thước (batch_size, seq_len, d_model)
        """
        # Cắt ma trận PE cho khớp với độ dài câu hiện tại (seq_len)
        # x.size(1) chính là độ dài câu thực tế
        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)