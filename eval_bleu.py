import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
import sacrebleu
from tqdm import tqdm
import os
import math

# Import module nội bộ
from model.transformer import Transformer
from utils.tokenizer import Vocabulary
from utils.dataset import BilingualDataset, Collate
from configs import cfg


def batched_beam_search(model, src, src_mask, max_len, sos_idx, eos_idx, pad_idx, device, beam_width=3, alpha=0.7):
    """
    Thực hiện Beam Search trên toàn bộ Batch cùng lúc (Vectorized Beam Search).
    Tối ưu hóa tốc độ gấp nhiều lần so với xử lý từng câu.
    """
    batch_size = src.size(0)

    with torch.no_grad():
        src_emb = model.positional_encoding(model.src_embedding(src))
        enc_src = model.encoder(src_emb, src_mask)  # [batch, src_len, d_model]

    # nhân bản encoder output để khớp với số lượng beam
    # [batch, src_len, d_model] -> [batch * beam, src_len, d_model]
    enc_src = enc_src.unsqueeze(1).repeat(1, beam_width, 1, 1).view(batch_size * beam_width, -1, cfg.d_model)

    # Nhân bản src_mask: [batch, 1, 1, src_len] -> [batch * beam, 1, 1, src_len]
    src_mask = src_mask.unsqueeze(1).repeat(1, beam_width, 1, 1, 1).view(batch_size * beam_width, 1, 1, -1)


    tgt = torch.full((batch_size * beam_width, 1), sos_idx, dtype=torch.long).to(device)


    beam_scores = torch.zeros((batch_size, beam_width)).to(device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # Flatten thành [batch * beam]

    finished_beams = torch.zeros((batch_size, beam_width), dtype=torch.bool).to(device)

    # Lưu kết quả cuối cùng (nếu done sớm)
    # Dictionary map: batch_idx -> list of (score, sequence)
    results = {i: [] for i in range(batch_size)}

    #decoding
    for step in range(max_len):
        tgt_mask = model.make_tgt_mask(tgt)

        with torch.no_grad():
            tgt_emb = model.positional_encoding(model.tgt_embedding(tgt))
            out = model.decoder(tgt_emb, enc_src, tgt_mask, src_mask)

            # Lấy token cuối: [batch * beam, d_model]
            last_token_emb = out[:, -1, :]

            # Tính xác suất: [batch * beam, vocab_size]
            logits = model.fc_out(last_token_emb)
            log_probs = F.log_softmax(logits, dim=-1)

        vocab_size = log_probs.size(-1)

        next_scores = beam_scores.unsqueeze(1) + log_probs


        next_scores = next_scores.view(batch_size, -1)


        # Lấy Top K điểm cao nhất: [batch, beam]
        topk_scores, topk_indices = torch.topk(next_scores, beam_width, dim=1)

        beam_indices = torch.div(topk_indices, vocab_size, rounding_mode='floor')  # [batch, beam]
        token_indices = torch.remainder(topk_indices, vocab_size)  # [batch, beam]


        batch_pos = torch.arange(batch_size).unsqueeze(1).expand_as(beam_indices).to(device)


        select_indices = (batch_pos * beam_width + beam_indices).view(-1)


        tgt_new = tgt.index_select(0, select_indices)

        token_new = token_indices.view(-1, 1)
        tgt = torch.cat([tgt_new, token_new], dim=1)

        # Cập nhật điểm số
        beam_scores = topk_scores.view(-1)


        is_eos = (token_indices == eos_idx)  # [batch, beam]

        if is_eos.any():
            # Nếu gặp EOS, ta lưu kết quả lại (trừ điểm alpha length penalty) nhưng vẫn để vòng lặp chạy tiếp (nhưng gán điểm cực thấp để nó rớt khỏi top k vòng sau)
            for b in range(batch_size):
                for k in range(beam_width):
                    if is_eos[b, k]:
                        # Tính length penalty
                        length = step + 1  # độ dài chưa tính sos
                        final_score = topk_scores[b, k].item() / (length ** alpha)
                        # Lấy sequence (bỏ sos ở đầu)
                        seq = tgt[b * beam_width + k][1:].tolist()  # Bỏ sos
                        results[b].append((final_score, seq))

                        # Đánh dấu beam này điểm rất thấp để vòng sau nó bị loại
                        beam_scores[b * beam_width + k] = -1e9

    # 6. CHỌN KẾT QUẢ TỐT NHẤT
    final_sentences = []

    for b in range(batch_size):
        # Nếu chưa có kết quả nào (chưa gặp eos), lấy beam điểm cao nhất hiện tại
        if len(results[b]) == 0:
            best_score = beam_scores[b * beam_width].item()
            length = max_len
            final_score = best_score / (length ** alpha)
            seq = tgt[b * beam_width][1:].tolist()
            final_sentences.append(seq)
        else:
            # Chọn sequence có điểm (đã normalized) cao nhất
            results[b].sort(key=lambda x: x[0], reverse=True)
            final_sentences.append(results[b][0][1])

    return final_sentences


def evaluate_bleu_beam_optimized():
    # 1. Cấu hình
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    EVAL_BATCH_SIZE = 64
    BEAM_WIDTH = 10  # Giữ mức 3-5 là chuẩn

    print("--- Loading Vocab ---")
    if not os.path.exists(cfg.vocab_src):
        print(f"Lỗi: Không tìm thấy vocab tại {cfg.vocab_src}")
        return

    vocab_src = Vocabulary(cfg.vocab_src)
    vocab_tgt = Vocabulary(cfg.vocab_tgt)

    print("--- Loading Model ---")
    model = Transformer(
        src_vocab_size=len(vocab_src),
        tgt_vocab_size=len(vocab_tgt),
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_len=5000,
        src_pad_idx=vocab_src.pad_idx,
        tgt_pad_idx=vocab_tgt.pad_idx
    ).to(device)

    model.src_embedding.emb.weight = model.tgt_embedding.emb.weight
    model.fc_out.weight = model.src_embedding.emb.weight

    if os.path.exists(cfg.model):
        try:
            model.load_state_dict(torch.load(cfg.model, map_location=device))
        except:
            model.load_state_dict(torch.load(cfg.model, map_location=device, weights_only=False))
        print(f"--> Đã load trọng số: {cfg.model}")
    else:
        print("CẢNH BÁO: Không tìm thấy file trọng số.")
        return

    model.eval()

    print("--- Loading Test Data ---")
    try:
        dataset = load_from_disk('data/vlsp')
        test_data = dataset['test']
    except Exception as e:
        print(f"Lỗi data: {e}")
        return

    test_src = [item['vi'] for item in test_data]
    test_tgt = [item['en'] for item in test_data]

    test_dataset = BilingualDataset(test_src, test_tgt, vocab_src, vocab_tgt, max_len=cfg.max_len)
    collate_fn = Collate(pad_idx=vocab_src.pad_idx)

    test_iterator = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    hypotheses = []
    references = []

    print(f"--- Bắt đầu đánh giá (Batched Beam Search - Width: {BEAM_WIDTH}) ---")
    with torch.no_grad():
        for src, tgt in tqdm(test_iterator, desc="Decoding"):
            src = src.to(device)
            src_mask = model.make_src_mask(src)

            final_seqs_list = batched_beam_search(
                model, src, src_mask,
                max_len=100,  # Độ dài tối đa câu dịch
                sos_idx=vocab_tgt.sos_idx,
                eos_idx=vocab_tgt.eos_idx,
                pad_idx=vocab_tgt.pad_idx,
                device=device,
                beam_width=BEAM_WIDTH,
                alpha=0.7  # Length penalty
            )

            # Convert IDs sang Text
            for i, seq_ids in enumerate(final_seqs_list):
                # Decode Hypothesis
                # Cắt bỏ phần sau eos nếu còn sót (thường hàm trên đã xử lý list sạch)
                clean_ids = []
                for idx in seq_ids:
                    if idx == vocab_tgt.eos_idx: break
                    clean_ids.append(idx)

                pred_text = vocab_tgt.decode(clean_ids)
                hypotheses.append(pred_text)

                # Decode Reference
                ref_ids = [t.item() for t in tgt[i] if
                           t.item() not in [vocab_tgt.pad_idx, vocab_tgt.sos_idx, vocab_tgt.eos_idx]]
                ref_text = vocab_tgt.decode(ref_ids)
                references.append(ref_text)

    # In mẫu kết quả
    print("\n--- Ví dụ kết quả ---")
    for i in range(min(3, len(hypotheses))):
        print(f"Dự đoán: {hypotheses[i]}")
        print(f"Thực tế: {references[i]}")
        print("-" * 20)

    score = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"\n============================")
    print(f"BATCHED BEAM BLEU SCORE: {score.score:.2f}")
    print(f"============================")


if __name__ == "__main__":
    evaluate_bleu_beam_optimized()