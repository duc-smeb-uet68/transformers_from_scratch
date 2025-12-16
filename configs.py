class Config:
    def __init__(self):
        # Thiết lập dữ liệu [cite: 4, 5]
        self.src_lang = 'vi'
        self.tgt_lang = 'en'
        self.max_len = 128         # Độ dài tối đa của câu (để cắt/pad)
        self.batch_size = 32

        # Thiết lập Mô hình (Model Architecture) [cite: 10]
        self.d_model = 512        # Kích thước vector embedding
        self.n_heads = 8          # Số lượng Head trong Multi-Head Attention
        self.n_layers = 6         # Số lớp Encoder và Decoder
        self.d_ff = 2048          # Kích thước lớp ẩn trong Feed Forward Network
        self.dropout = 0.1

        # Thiết lập Huấn luyện (Training) [cite: 27, 29]
        self.lr = 0.0001          # Learning rate
        #self.epochs = 20
        self.warmup_steps = 4000  # Cho Scheduler
        self.label_smoothing = 0.1 # Kỹ thuật giúp model đỡ overfit

        self.n_epochs = 50          # Đặt cao lên, Early Stopping sẽ tự dừng
        self.clip = 1.0             # Gradient Clipping
        self.patience = 5           # Dừng nếu Val Loss không giảm sau 5 epoch

        # Đường dẫn lưu model
        self.model_path = 'weights/transformer_vi_en.pth'

        self.vocab_configs = [
            {
                "name": "word",
                "src_path": "data/vocab/vocab_src.json",
                "tgt_path": "data/vocab/vocab_tgt.json",
                "ckpt_path": "transformer_word.pt",
                "plot_prefix": "word"
            },
            {
                "name": "bpe",
                "src_path": "data/vocab_bpe/tokenizet_vi.json",
                "tgt_path": "data/vocab_bpe/tokenizer_en.json",
                "ckpt_path": "transformer_bpe.pt",
                "plot_prefix": "bpe"
            },
            {
                "name": "shared",
                "src_path": "data/shared_vocab/tokenizer_shared.json",
                "tgt_path": "data/shared_vocab/tokenizer_shared.json",
                "ckpt_path": "transformer_shared.pt",
                "plot_prefix": "shared"
            }
        ]

cfg = Config()