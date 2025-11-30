import torch
from utils.tokenizer import Vocabulary


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
    import sacrebleu

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