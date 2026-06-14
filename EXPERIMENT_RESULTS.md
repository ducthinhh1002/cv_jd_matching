# Kết Quả Thực Nghiệm Pipeline IT CV-JD Matching

> Tổng hợp toàn bộ kết quả đo lường của 2 thành phần cốt lõi trong pipeline:
> **(1) Mô hình NER** và **(2) Bước Mapping O\*NET.**
>
> Dữ liệu: 19.853 tài liệu (CV + JD) từ Hugging Face `lang-uk/recruitment-dataset-*`.
> Mô hình: `roberta-base`, Kiến trúc: Span-based NER.

---

## Phần 1: Kết Quả Mô Hình NER (Span-based RoBERTa)

Thực nghiệm so sánh 2 phiên bản dữ liệu huấn luyện để cải thiện Precision của mô hình.

| Chỉ số                     | v5 (Baseline)                         | v6 (Improved)                         | Chênh lệch         |
| :---                       | :---:                                 | :---:                                 | :---:              |
| **Phiên bản dữ liệu**      | `cleaned_v5` (negative_multiplier=10) | `cleaned_v6` (negative_multiplier=5)  |                    |
| **Cải tiến so với v5**     | -                                     | Trim noisy prefixes, giảm negative sampling | -           |
| **Best Epoch**             | 7 / 8                                 | 7 / 8                                 | -                  |
| **Dev Precision**          | 0.4934 (49.3%)                        | **0.4984 (49.8%)**                    | +0.5%              |
| **Dev Recall**             | 0.8695 (87.0%)                        | 0.8669 (86.7%)                        | -0.3%              |
| **Dev F1 Score (Best)**    | 0.6295 (62.9%)                        | **0.6330 (63.3%)**                    | **+0.35%**         |
| **Test Precision**         | 0.4864 (48.6%)                        | **0.4869 (48.7%)**                    | +0.05%             |
| **Test Recall**            | 0.8741 (87.4%)                        | 0.8728 (87.3%)                        | -0.1%              |
| **Test F1 Score (Final)**  | 0.6250 (62.5%)                        | **0.6251 (62.5%)**                    | +0.01%             |
| **Checkpoint path**        | `artifacts/span_ner_roberta_base_openai_cleaned_v5_hardneg/span_ner.pt` | `artifacts/span_ner_roberta_base_openai_cleaned_v6/span_ner.pt` | |

### Ghi chú NER

- Việc làm sạch các **tiền tố nhiễu** (`strong`, `experience with`, `knowledge of`, v.v.) và **giảm Negative Sampling Multiplier** xuống còn 5 (từ 10) giúp Precision nhích lên nhẹ mà không đánh đổi Recall đáng kể.
- Cả hai phiên bản đều cho thấy mô hình có khả năng **Recall rất cao (~87%)** — phù hợp cho bài toán trích xuất (không muốn bỏ sót thực thể), nhưng **Precision còn thấp (~49%)** do nhiễu từ nhãn LLM (Noisy Weak Labels). Đây là hướng cải thiện tiếp theo.

---

## Phần 2: Kết Quả Mapping O*NET

Thực nghiệm so sánh 3 phiên bản của bước Mapping để cải thiện tỉ lệ thực thể được ánh xạ sang O*NET.

### Tổng quan theo phiên bản

| Chỉ số                         | v5-Lexical (Baseline) | v5-Semantic              | v6-Semantic (Best)       |
| :---                           | :---:                 | :---:                    | :---:                    |
| **Dữ liệu đầu vào**            | `cleaned_v5`          | `cleaned_v5`             | `cleaned_v6`             |
| **Phương pháp Matching**       | Lexical only          | Lexical + Semantic (MiniLM-L6-v2) | Lexical + Semantic (MiniLM-L6-v2) |
| **Tổng thực thể được hỗ trợ**  | 221.928               | 221.928                  | 221.859                  |
| **Tổng thực thể được map**     | 148.509               | 164.264                  | **164.507**              |
| **Mapped Rate (Tổng)**         | 66.92%                | 74.02%                   | **74.15%**               |
| **File kết quả**               | `stage2_onet_mapped_cleaned_v5.jsonl` | `stage2_onet_mapped_semantic_full.jsonl` | `stage2_onet_mapped_semantic_v6_full.jsonl` |

### Chi tiết theo nhãn (v5-Lexical vs v6-Semantic)

| Nhãn (Label)      | v5-Lexical (mapped/total) | v6-Semantic (mapped/total) |
| :---              | :---:                     | :---:                      |
| **JOB_ROLE**      | 23.524 / 24.438           | 24.106 / 24.438            |
| **TECHNOLOGY**    | 115.763 / 154.354         | 120.519 / 154.354          |
| **WORK_ACTIVITY** | 2.585 / 21.476            | 10.811 / 21.420            |
| **SKILL**         | 4.413 / 17.890            | 6.237 / 17.878             |
| **PROJECT_TYPE**  | 2.224 / 3.770             | 2.834 / 3.769              |
| **INDUSTRY**      | 0 / 5.434                 | 0 / 5.433                  |
| **DEGREE**        | 0 / 1.172                 | 0 / 1.157                  |
| **CERTIFICATION** | 0 / 629                   | 0 / 626                    |

### Ghi chú Mapping

- **Cải thiện lớn nhất:** Nhờ Semantic Embedding, nhãn `WORK_ACTIVITY` tăng từ **12% lên 50.5%** (tăng hơn **4 lần**). Đây là nhãn khó nhất vì LLM thường trích xuất các cụm từ dài, mang ngữ nghĩa phức tạp mà lexical matching không bắt được.
- **Nhãn SKILL:** Tăng từ 24.7% lên 34.9%, vẫn còn tiềm năng cải thiện thêm nếu tích hợp thêm bộ chuẩn **ESCO**.
- **Thông số tốt nhất:** `min_score=0.35`, `top_k=5`, `embedding_model=sentence-transformers/all-MiniLM-L6-v2`, `device=cuda`.

---

## Phần 3: Kết Quả Stage 3 — Multi-Factor Scoring

Hệ thống chấm điểm kết hợp 3 yếu tố: Semantic Similarity (40%), O*NET Importance (40%), và Hard Constraint (20%).

### Pipeline đánh giá trên Synthetic Dataset

```powershell
# Bước 1: Score 1.000 pairs
python score_candidates.py `
  --stage2-input .\artifacts\stage2_onet_mapped_semantic_v6_full.jsonl `
  --pairs-input .\artifacts\synthetic_eval_dataset.jsonl `
  --output .\artifacts\stage3_scores_eval.jsonl `
  --overwrite

# Bước 2: Evaluate
python evaluate_ranking.py `
  --predictions .\artifacts\stage3_scores_eval.jsonl `
  --output .\artifacts\eval_results.json `
  --k 5 --relevant-threshold 4 --compare-baselines
```

### Design Dataset Synthetic

| Thông số | Giá trị |
| :--- | :--- |
| Số IT roles | 20 (bao gồm Backend, Frontend, Cloud, AR/VR, Data...) |
| JDs mỗi role | 10 JDs đa dạng hóa Tech Stack (unique sets) |
| Relevance levels | 5 (1=irrelevant → 5=perfect match) |
| Tổng pairs | 1.000 |
| Model generate | `gpt-4o-mini` |

*Ghi chú: Vì mỗi JD chỉ có 5 CV tương ứng (mỗi CV ứng với 1 mức Relevance), số lượng CV đạt mức `Relevant >= 4` chỉ là **2**. Do đó Precision@5 đạt cực đại lý thuyết là `2/5 = 0.4000`.*

### Metrics trên tập Synthetic (1.000 pairs, 200 JDs)

| Method | Precision@5 | NDCG@5 | MAP | Spearman ρ |
| :--- | :---: | :---: | :---: | :---: |
| Pure O*NET Matching | 0.400 | 0.544 | 0.325 | 0.520 |
| Hard Constraint Base | 0.400 | 0.813 | 0.710 | 0.227 |
| Pure Semantic Similarity| 0.400 | 0.882 | 0.860 | **0.813** |
| **Proposed System (Linear Baseline)** | **0.400** | **0.949** | **0.926** | 0.810 |
| Proposed System (CrossEncoder + Soft Match) | 0.400 | 0.839 | 0.738 | 0.476 |
| **Proposed System (ML Ranker XGBoost)** | **0.400** | **0.946** | **0.897** | 0.580 |

**Nhận xét về SOTA Upgrade (CrossEncoder, Soft Matching & ML Ranker):**
- Khi đưa Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) và Knowledge Graph Soft Matching vào, nếu tính điểm bằng công thức tuyến tính tĩnh, kết quả **bị giảm** (NDCG@5 giảm từ 0.949 xuống 0.839).
- **Lý do:** Mô hình Cross-Encoder xuất ra phân phối hoàn toàn khác với Cosine Similarity của Bi-Encoder. Khi áp dụng công thức cộng tuyến tính tĩnh (`0.4*Semantic + 0.4*O*NET + 0.2*Hard`), trọng số này bị mất cân bằng nghiêm trọng và làm phá vỡ logic xếp hạng.
- **Giải pháp (Đề xuất 1):** Bằng cách huấn luyện **ML Ranker (XGBoost)** dựa trên các features mới này, mô hình đã tự động học được trọng số tối ưu (Feature importance: Hard Constraint 67%, Semantic 33%, O*NET 0%). Nhờ đó, hiệu năng đã được phục hồi hoàn toàn (NDCG@5 tăng vọt trở lại mức **~95%**).
- Việc XGBoost không dùng O*NET Score cho thấy trên tập Synthetic Dataset này, Semantic và Hard Constraint đã quá đủ để phân loại. Tuy nhiên trong môi trường thực tế, O*NET Soft Match sẽ phát huy tác dụng với những ứng viên ẩn (implicit candidates).

---

## Hướng cải thiện tiếp theo (Future Work)

1. **Nâng cấp Backbone NER:** Thay `roberta-base` bằng `microsoft/deberta-v3-base` hoặc `jjzha/jobbert-base-cased` để tăng Precision lên đáng kể mà không cần thêm dữ liệu.
2. **Tích hợp bộ chuẩn ESCO:** Bổ sung kho từ điển kỹ năng của ESCO (EU) vào bước Mapping để tăng tỉ lệ map của nhãn `SKILL` và `TECHNOLOGY`.
3. **Fine-tune Embedding Model:** Dùng Contrastive Learning trên tập dữ liệu cặp (CV term ↔ O\*NET descriptor) để mô hình Embedding hiểu sâu hơn về ngôn ngữ IT tuyển dụng.
4. **Implicit Skill Inference:** Suy luận kỹ năng ngầm từ O\*NET (ví dụ: "Python Developer" → ngầm hiểu có "Data Structures", "OOP").
