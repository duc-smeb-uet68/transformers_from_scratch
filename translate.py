import torch
from utils.tokenizer import Vocabulary
import sacrebleu
from model.transformer import Transformer

def translate_sentence(sentence, src_vocab, tgt_vocab, model, device, max_len=50):
    model.eval()

    # 1. Xử lý câu nguồn (Source)
    # Tokenize và thêm <sos>, <eos>
    # tokens = [src_vocab.stoi["<sos>"]] + src_vocab.numericalize(sentence) + [src_vocab.stoi["<eos>"]]#đã fix dòng này
    tokens = [src_vocab.stoi["<sos>"]] + src_vocab.numericalize(sentence, lang='vi') + [src_vocab.stoi["<eos>"]]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device) # (1, src_len)

    # Tạo mask cho src
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        # Encode câu nguồn
        src_emb = model.positional_encoding(model.src_embedding(src_tensor))
        enc_src = model.encoder(src_emb, src_mask)

    # 2. Khởi tạo câu đích với <sos>
    tgt_indices = [tgt_vocab.stoi["<sos>"]]

    # 3. Vòng lặp Decoding
    for i in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device) # (1, curr_len)

        # Tạo mask cho tgt
        tgt_mask = model.make_tgt_mask(tgt_tensor)

        with torch.no_grad():
            # Decode
            tgt_emb = model.positional_encoding(model.tgt_embedding(tgt_tensor))
            output = model.decoder(tgt_emb, enc_src, tgt_mask, src_mask)

            # Lấy dự đoán cho từ cuối cùng
            pred_token_logits = model.fc_out(output[:, -1, :])

            # Chọn từ có xác suất cao nhất (Greedy)
            pred_token = pred_token_logits.argmax(1).item()

            # Nếu gặp <eos> thì dừng
            if pred_token == tgt_vocab.stoi["<eos>"]:
                break

            tgt_indices.append(pred_token)

    # 4. Chuyển từ số về chữ
    trg_tokens = [tgt_vocab.itos[i] for i in tgt_indices]

    # Bỏ <sos> ở đầu
    return trg_tokens[1:]


# Hàm tính BLEU Score
def calculate_bleu(data, src_vocab, tgt_vocab, model, device):


    targets = []
    outputs = []

    for example in data:
        src = example[0]  # Câu tiếng Việt gốc
        trg = example[1]  # Câu tiếng Anh gốc

        prediction = translate_sentence(src, src_vocab, tgt_vocab, model, device)

        # Nối lại thành câu
        pred_sent = " ".join(prediction)

        outputs.append(pred_sent)
        targets.append([trg])  # Sacrebleu yêu cầu list of lists cho reference

    bleu = sacrebleu.corpus_bleu(outputs, targets)
    return bleu.score

def load_model_and_translate():
    # 1. Cấu hình thiết bị
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang sử dụng thiết bị: {device}")

    # 2. Load lại bộ từ điển (Vocabulary)
    # Rất quan trọng: Phải load từ file json đã tạo lúc train
    # Nếu bạn chưa có file json, bạn cần đảm bảo vocab được build y hệt lúc train
    print("--- Đang load Vocabulary ---")
    vocab_src = Vocabulary(freq_threshold=1)
    vocab_tgt = Vocabulary(freq_threshold=1)

    # Đường dẫn này phải trỏ đúng đến file vocab bạn đã tạo (thường ở thư mục data/)
    try:
        vocab_src.load_vocab('data/vocab_src.json')
        vocab_tgt.load_vocab('data/vocab_tgt.json')
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file từ điển (.json). Hãy đảm bảo bạn đã chạy build_vocab.py hoặc train.py trước.")
        return

    print(f"Kích thước từ điển nguồn: {len(vocab_src)}")
    print(f"Kích thước từ điển đích: {len(vocab_tgt)}")

    # 3. Khởi tạo mô hình
    # CÁC THAM SỐ NÀY PHẢI KHỚP VỚI FILE config.py HOẶC train.py CỦA BẠN
    model = Transformer(
        src_vocab_size=len(vocab_src),
        tgt_vocab_size=len(vocab_tgt),
        d_model=512,  # Kiểm tra lại config.py xem là 256 hay 512
        n_layers=6,  # Kiểm tra lại config.py
        n_heads=8,
        d_ff=512,
        dropout=0.1,
        max_len=512,
        src_pad_idx=vocab_src.stoi["<pad>"],
        tgt_pad_idx=vocab_tgt.stoi["<pad>"]
    ).to(device)

    # 4. Load trọng số đã train (Checkpoint)
    try:
        model.load_state_dict(torch.load('transformer_model.pt', map_location=device))
        print("--> Đã load trọng số 'transformer_model.pt' thành công!")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file trọng số 'transformer_model.pt'.")
        return

    # 5. Vòng lặp Dịch
    while True:
        src_sentence = input("\nNhập câu tiếng Việt (gõ 'q' để thoát): ")
        if src_sentence.lower() == 'q':
            break

        # Gọi hàm translate_sentence có sẵn trong project
        result_tokens = translate_sentence(
            src_sentence,
            vocab_src,
            vocab_tgt,
            model,
            device,
            max_len=50
        )

        # Nối các token thành câu hoàn chỉnh
        pred_sent = " ".join(result_tokens)

        print(f"Dịch sang tiếng Anh: {pred_sent}")


if __name__ == "__main__":
    load_model_and_translate()