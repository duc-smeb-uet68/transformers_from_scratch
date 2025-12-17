class Config:
    def __init__(self):
        # Thiết lập dữ liệu [cite: 4, 5]
        self.src_lang = 'vi'
        self.tgt_lang = 'en'
        self.max_len = 128         # Độ dài tối đa của câu (để cắt/pad)
        self.batch_size = 256

        # Thiết lập Mô hình (Model Architecture) [cite: 10]
        self.d_model = 256
        self.n_heads = 4
        self.n_layers = 6
        self.d_ff = 1024
        self.dropout = 0.1

        # Thiết lập Huấn luyện (Training) [cite: 27, 29]
        self.lr = 0.0005
        self.warmup_steps = 4000
        self.label_smoothing = 0.1
        self.weight_decay = 1e-4

        self.n_epochs = 60
        self.clip = 1.0
        self.patience = 6


        self.vocab_src = 'vocab/shared_vocab/tokenizer_shared.json'
        self.vocab_tgt = 'vocab/shared_vocab/tokenizer_shared.json'

        self.iwslt_data = 'data/iwslt2015_data'
        self.vlsp_data = 'data/vlsp'

        self.finetune_model = 'weights/fine_tune.pt'
        self.pretrain_iwslt = 'weights/iwslt_best.pt'
        self.pretrain_vlsp = 'weights/vlsp_best.pt'

        self.model = self.finetune_model # chọn model ở đây

        self.checkpoint = 'checkpoint/checkpoint.pt'

cfg = Config()