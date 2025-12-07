import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader

import os
import sacrebleu
from tqdm import tqdm

# Import module nội bộ
from model.transformer import Transformer
from utils.tokenizer import Vocabulary
from utils.dataset import BilingualDataset, Collate

# --- 1. THUẬT TOÁN BEAM SEARCH (NÂNG CẤP) ---
def beam_search_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device, beam_width=3):
    """
    Thực hiện giải mã Beam Search để tìm câu dịch tốt nhất.
    """
    # 1. Encode câu nguồn (chỉ làm 1 lần để tiết kiệm tính toán)
    with torch.no_grad():
        src_emb = model.positional_encoding(model.src_embedding(src))
        enc_src = model.encoder(src_emb, src_mask)

    # 2. Khởi tạo Beam: List các tuple (score, sequence)
    # score: Log-probability tích lũy (càng lớn càng tốt, vì log(p) < 0)
    # sequence: List các token IDs, bắt đầu bằng <sos>
    k_beams = [(0.0, [start_symbol])]

    # 3. Vòng lặp Decoding
    for _ in range(max_len):
        candidates = []
        all_finished = True

        for score, seq in k_beams:
            # Nếu nhánh này đã kết thúc bằng <eos>, giữ nguyên đưa vào candidate
            if seq[-1] == end_symbol:
                candidates.append((score, seq))
                continue

            all_finished = False

            # Chuẩn bị input cho Decoder
            tgt_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
            tgt_mask = model.make_tgt_mask(tgt_tensor)

            with torch.no_grad():
                tgt_emb = model.positional_encoding(model.tgt_embedding(tgt_tensor))
                # Decoder forward
                out = model.decoder(tgt_emb, enc_src, tgt_mask, src_mask)
                # Lấy output tại bước thời gian cuối cùng
                prob = model.fc_out(out[:, -1, :])
                # Tính log_softmax để cộng dồn score
                log_prob = F.log_softmax(prob, dim=-1)

            # Lấy top k token có xác suất cao nhất
            topk_log_prob, topk_idx = torch.topk(log_prob, beam_width, dim=-1)

            # Mở rộng nhánh
            for i in range(beam_width):
                token = topk_idx[0][i].item()
                token_prob = topk_log_prob[0][i].item()

                # Công thức cập nhật score
                new_score = score + token_prob
                new_seq = seq + [token]
                candidates.append((new_score, new_seq))

        # Nếu tất cả các nhánh đều đã xong, dừng sớm
        if all_finished:
            break

        # Sắp xếp các ứng viên theo score giảm dần
        # (Có thể cải tiến bằng cách chia score cho độ dài câu để normalize)
        ordered = sorted(candidates, key=lambda x: x[0], reverse=True)

        # Giữ lại top k nhánh tốt nhất cho vòng lặp sau
        k_beams = ordered[:beam_width]

    # Trả về sequence có score cao nhất (nhánh đầu tiên)
    best_seq = k_beams[0][1]
    return best_seq


# --- 2. HÀM DỊCH MỘT CÂU ---
# def translate_sentence(sentence, src_vocab, tgt_vocab, model, device, max_len=100, use_beam=True):
#     model.eval()
#
#     # 1. Xử lý câu nguồn (Pre-processing)
#     # Tokenize và chuyển thành ID (BPE Tokenizer đã xử lý tốt việc tách từ)
#     # Lưu ý: Thêm <sos> và <eos> thủ công nếu tokenizer chưa thêm
#     input_ids = src_vocab.numericalize(sentence)
#     tokens = [src_vocab.sos_idx] + input_ids + [src_vocab.eos_idx]
#
#     src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
#     src_mask = model.make_src_mask(src_tensor)
#
#     if use_beam:
#         # Cách 1: Beam Search (Chất lượng cao hơn)
#         tgt_indices = beam_search_decode(
#             model, src_tensor, src_mask, max_len,
#             start_symbol=tgt_vocab.sos_idx,
#             end_symbol=tgt_vocab.eos_idx,
#             device=device,
#             beam_width=5  # Beam width càng lớn càng chính xác nhưng chậm hơn
#         )
#     else:
#         # Cách 2: Greedy Search (Nhanh hơn, dùng để debug)
#         tgt_indices = [tgt_vocab.sos_idx]
#         for i in range(max_len):
#             tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
#             tgt_mask = model.make_tgt_mask(tgt_tensor)
#
#             with torch.no_grad():
#                 tgt_emb = model.positional_encoding(model.tgt_embedding(tgt_tensor))
#                 enc_src = model.encoder(model.positional_encoding(model.src_embedding(src_tensor)), src_mask)
#                 output = model.decoder(tgt_emb, enc_src, tgt_mask, src_mask)
#                 pred_token = output[:, -1, :].argmax(1).item()
#
#                 tgt_indices.append(pred_token)
#                 if pred_token == tgt_vocab.eos_idx:
#                     break
#
#     # 2. Chuyển ID về lại text (Post-processing)
#     # Loại bỏ token <sos>, <eos> nếu có trong kết quả
#     result_ids = [t for t in tgt_indices if t not in [tgt_vocab.sos_idx, tgt_vocab.eos_idx, tgt_vocab.pad_idx]]
#
#     translation = tgt_vocab.decode(result_ids)
#     return translation

def translate_sentence(sentence, src_vocab, tgt_vocab, model, device, max_len=100, use_beam=False):
    model.eval()

    # --- DEBUG: Kiểm tra Tokenizer ---
    print(f"\n[DEBUG] Input Text: {sentence}")
    input_ids = src_vocab.numericalize(sentence)
    tokens = [src_vocab.sos_idx] + input_ids + [src_vocab.eos_idx]

    # In ra danh sách ID để xem có bị toàn số 3 (<unk>) không
    print(f"[DEBUG] Token IDs: {tokens}")

    print(f"[DEBUG] Re-decoded Text: {src_vocab.decode(input_ids)}")

    # Kiểm tra xem có bao nhiêu token là <unk> (ID = 3)
    unk_count = tokens.count(src_vocab.unk_idx)
    if unk_count > len(tokens) / 3:
        print("⚠️ CẢNH BÁO: Quá nhiều token <unk>! Có thể do lỗi Font/Encoding hoặc sai chính tả.")
    # ----------------------------------

    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)

    if use_beam:
        tgt_indices = beam_search_decode(
            model, src_tensor, src_mask, max_len,
            tgt_vocab.sos_idx, tgt_vocab.eos_idx, device, beam_width=5
        )
    else:
        # Greedy fallback
        tgt_indices = [tgt_vocab.sos_idx]
        for _ in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
            tgt_mask = model.make_tgt_mask(tgt_tensor)
            with torch.no_grad():
                tgt_emb = model.positional_encoding(model.tgt_embedding(tgt_tensor))
                enc_src = model.encoder(model.positional_encoding(model.src_embedding(src_tensor)), src_mask)
                output = model.decoder(tgt_emb, enc_src, tgt_mask, src_mask)
                pred_token = output[:, -1, :].argmax(1).item()
                tgt_indices.append(pred_token)
                if pred_token == tgt_vocab.eos_idx: break

    result_ids = [t for t in tgt_indices if t not in [tgt_vocab.sos_idx, tgt_vocab.eos_idx, tgt_vocab.pad_idx]]
    return tgt_vocab.decode(result_ids)

# --- 3. HÀM TÍNH ĐIỂM BLEU ---
def calculate_bleu_score(data_iterator, src_vocab, tgt_vocab, model, device, max_len=100):
    """
    Tính điểm BLEU trên toàn bộ tập dữ liệu (Validation/Test)
    """
    model.eval()
    targets = []
    outputs = []

    print("--- Đang tính toán BLEU Score ---")
    with torch.no_grad():
        for src, tgt in tqdm(data_iterator, desc="Translating"):
            # Lấy batch size
            batch_size = src.shape[0]

            for i in range(batch_size):
                # Lấy từng câu trong batch ra để dịch
                # Lưu ý: Đây là cách đơn giản, để nhanh hơn có thể implement Batch Beam Search

                # Lấy ID của câu nguồn (bỏ padding)
                src_ids = [token.item() for token in src[i] if token.item() != src_vocab.pad_idx]
                # Bỏ <sos> và <eos> nếu có để decode lại thành string cho chuẩn
                src_text = src_vocab.decode(src_ids)

                # Lấy ID của câu đích thực tế (Reference)
                tgt_ids = [token.item() for token in tgt[i] if token.item() != tgt_vocab.pad_idx]
                tgt_text = tgt_vocab.decode(tgt_ids)  # Hàm decode của BPE sẽ tự bỏ special tokens

                # Dịch
                pred_text = translate_sentence(src_text, src_vocab, tgt_vocab, model, device, max_len, use_beam=True)

                outputs.append(pred_text)
                targets.append([tgt_text])  # Sacrebleu yêu cầu list of references

    # Tính BLEU
    bleu = sacrebleu.corpus_bleu(outputs, targets)
    return bleu.score


# --- 4. MAIN INTERACTION ---
def load_model_and_translate():
    # Cấu hình thiết bị
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load Vocabulary (BPE)
    # Đảm bảo đường dẫn khớp với file đã train ở bước build_vocab_bpe.py
    vocab_src_path = "data/vocab_bpe/tokenizer_vi.json"
    vocab_tgt_path = "data/vocab_bpe/tokenizer_en.json"

    if not os.path.exists(vocab_src_path) or not os.path.exists(vocab_tgt_path):
        print(f"Lỗi: Không tìm thấy file vocab tại {vocab_src_path}. Hãy chạy build_vocab_bpe.py trước.")
        return

    print("--- Loading Tokenizers ---")
    vocab_src = Vocabulary(vocab_src_path)
    vocab_tgt = Vocabulary(vocab_tgt_path)

    # 2. Khởi tạo Model (Cấu hình phải khớp train.py)
    print("--- Loading Model ---")
    model = Transformer(
        src_vocab_size=len(vocab_src),
        tgt_vocab_size=len(vocab_tgt),
        d_model=512,  # Khớp với train.py
        n_layers=6,  # Khớp với train.py
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
        max_len=5000,
        src_pad_idx=vocab_src.pad_idx,
        tgt_pad_idx=vocab_tgt.pad_idx
    ).to(device)

    # 3. Load Trọng số (Weights)
    model_path = 'transformer_best.pt'  # Ưu tiên load model tốt nhất
    if not os.path.exists(model_path):
        model_path = 'transformer_last.pt'

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"--> Đã load trọng số từ '{model_path}'")
    else:
        print("Cảnh báo: Không tìm thấy file trọng số (.pt). Model sẽ chạy với trọng số ngẫu nhiên.")

    # 4. Giao diện dịch
    print("\n" + "=" * 30)
    print("MÔ HÌNH DỊCH MÁY VIỆT -> ANH")
    print("=" * 30)

    while True:
        try:
            src_sentence = input("\nNhập câu tiếng Việt (hoặc 'q' để thoát): ")
            if src_sentence.lower().strip() == 'q':
                break

            # Đo thời gian dịch
            import time
            start = time.time()

            translation = translate_sentence(
                src_sentence,
                vocab_src,
                vocab_tgt,
                model,
                device,
                use_beam=True  # Bật Beam Search
            )

            end = time.time()

            print(f">> Tiếng Anh: \033[1;32m{translation}\033[0m")
            print(f"(Thời gian: {end - start:.3f}s)")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Lỗi: {e}")

def blue_score():
    # 1. Cấu hình thiết bị
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device đang dùng: {device}")

    # 2. Load Vocabulary (BPE)
    # Đảm bảo đường dẫn trỏ đúng file json bạn đã train
    try:
        vocab_src = Vocabulary("data/vocab_bpe/tokenizer_vi.json")
        vocab_tgt = Vocabulary("data/vocab_bpe/tokenizer_en.json")
    except Exception as e:
        print(f"Lỗi load vocab: {e}")
        exit()

    # 3. Load Model
    # Các tham số này PHẢI KHỚP với file config hoặc lúc train
    model = Transformer(
        src_vocab_size=len(vocab_src),
        tgt_vocab_size=len(vocab_tgt),
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
        max_len=5000,
        src_pad_idx=vocab_src.pad_idx,
        tgt_pad_idx=vocab_tgt.pad_idx
    ).to(device)

    # Load trọng số đã train
    if os.path.exists('transformer_best.pt'):
        model.load_state_dict(torch.load('transformer_best.pt', map_location=device))
        print("--> Đã load trọng số 'transformer_best.pt'")
    else:
        print("CẢNH BÁO: Không tìm thấy file trọng số! Kết quả BLEU sẽ rất thấp.")

    # 4. CHUẨN BỊ DATA ITERATOR (Phần quan trọng nhất)
    print("\n--- Đang tải dữ liệu Test ---")
    try:
        # Ưu tiên load từ đĩa nếu đã lưu
        dataset = load_from_disk("data/iwslt2015_data")
        test_data = dataset['test']
    except:
        # Nếu không có thì tải từ mạng
        dataset = load_dataset("nguyenvuhuy/iwslt2015-en-vi")
        test_data = dataset['test']

    # Tách danh sách câu nguồn và đích
    test_src = [item['vi'] for item in test_data]
    test_tgt = [item['en'] for item in test_data]

    # Tạo Dataset class
    test_dataset = BilingualDataset(
        test_src,
        test_tgt,
        vocab_src,
        vocab_tgt,
        max_len=128  # Độ dài tối đa khi đánh giá
    )

    # Tạo DataLoader (Đây chính là data_iterator)
    collate_fn = Collate(pad_idx=vocab_src.pad_idx)
    test_iterator = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 5. Gọi hàm tính BLEU
    # Truyền test_iterator vào hàm của bạn
    score = calculate_bleu_score(
        data_iterator=test_iterator,
        src_vocab=vocab_src,
        tgt_vocab=vocab_tgt,
        model=model,
        device=device,
        max_len=100
    )

    print(f"\n============================")
    print(f"KẾT QUẢ BLEU SCORE: {score:.2f}")
    print(f"============================")


if __name__ == "__main__":
    load_model_and_translate()
    # blue_score()
