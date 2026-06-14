# Timeline công việc đã làm trong project CV-JD matching

Ghi chú nguồn thông tin: timeline này được tổng hợp từ nội dung trao đổi trong cuộc trò chuyện hiện tại, các file trong thư mục C:\Users\PC\Downloads\CV-JD, thời gian chỉnh sửa file/artifact, và các kết quả đã lưu trong artifacts. Tôi không thấy file chat log riêng được lưu trong repo, nên các mốc “đoạn chat cũ” được khôi phục từ context hiện có và bằng chứng trong workspace.

## 1. Trạng thái hiện tại của project

Project hiện đã phát triển thành một pipeline CV-JD matching/job recommendation gồm nhiều tầng. Tầng đầu là domain-specific NER cho tuyển dụng IT, dùng span-based RoBERTa để trích xuất các entity như TECHNOLOGY, JOB_ROLE, SKILL, WORK_ACTIVITY, PROJECT_TYPE, INDUSTRY, DEGREE và CERTIFICATION. Tầng thứ hai là O*NET grounding, dùng O*NET để chuẩn hóa hoặc map các entity nghề nghiệp sang tri thức occupation/skill/work activity. Tầng thứ ba là scoring/ranking, kết hợp semantic similarity, lexical matching, entity/O*NET signal, skill coverage, role alignment, seniority và hard constraints. Ngoài ra đã có benchmark ngoài, synthetic evaluation, CareerBuilder benchmark và demo web app.

Checkpoint NER tốt nhất hiện tại là artifacts\span_ner_roberta_base_openai_cleaned_v6\span_ner.pt. Kết quả v6 đạt dev F1 0.6330 và test F1 0.6251, với test recall 0.8728. Mapping O*NET tốt nhất hiện tại là artifacts\stage2_onet_mapped_semantic_v6_full.jsonl, đạt mapped rate 74.15 phần trăm trên 221.859 supported entities. Stage 3 scoring trên synthetic CV-JD evaluation đạt NDCG@5 khoảng 0.9496 và MAP khoảng 0.9262 với linear hybrid, còn ML ranker XGBoost đạt NDCG@5 khoảng 0.9458 và MAP khoảng 0.8970. CareerBuilder benchmark mới nhất cho thấy proposed tuned vượt hoặc nhỉnh hơn BM25 trên các metric chính trong all-domain và IT-filtered setting.

## 2. Timeline chi tiết

| Mốc thời gian | Việc đã làm hoặc đã thử | Kết quả/trạng thái | File hoặc artifact liên quan |
| --- | --- | --- | --- |
| 13/02/2026 | Chuẩn bị slide/deck ban đầu cho đề tài CV-JD matching. | Có file trình bày nền tảng ban đầu để đối chiếu baseline/proposed method. | 17_DangKieuTrinh_20214933.pptx |
| 12/04/2026 | Bắt đầu code pipeline span annotation và weak/smoke NER. | Có script convert annotation và smoke artifact cho weak span NER. | convert_span_annotations.py, artifacts\weak_span_ner_smoke |
| 13/04/2026 | Viết script lấy subset từ Hugging Face recruitment datasets và script tạo seed subset. | Pipeline chuẩn hóa dữ liệu CV/JD đầu vào được hình thành, hỗ trợ CV/JD limits, IT filtering, resume download và seed sampling. | prepare_hf_recruitment_subset.py, prepare_seed_subset.py |
| 13/04/2026 | Trích xuất text từ slide để đọc lại baseline/proposed trong deck. | Có file text để tham khảo nội dung slide khi trả lời câu hỏi về baseline và cách so sánh. | artifacts\slide_text_extract.txt |
| 13/04/2026 | Smoke test OpenAI batch annotation. | Kiểm tra được flow chuẩn bị batch request cho LLM annotation. | artifacts\openai_batch_prepare_smoke |
| 16/04/2026 | Chạy batch annotation lớn cho strict 20k IT subset. | Tạo nền dữ liệu LLM-labeled cho NER. | artifacts\openai_batch_strict_20k, run_openai_batch_queue.ps1 |
| 17/04/2026 | Bổ sung error analysis cho span NER và smoke train trên GPU. | Có công cụ xem false positive/false negative, confidence và confusion theo nhãn. | analyze_span_ner_errors.py, artifacts\gpu_smoke_span_ner, artifacts\roberta_base_gpu_smoke |
| 17/04/2026 | Train RoBERTa span NER bản đầu trên annotation OpenAI. | Có checkpoint NER đầu tiên nhưng F1 còn thấp hơn các bản cleaned sau. | artifacts\span_ner_roberta_base_openai |
| 18/04/2026 | Hoàn thiện OpenAI batch NER script và script clean annotation. | Có pipeline batch annotation rẻ hơn synchronous API, hỗ trợ finalize output và làm sạch annotation. | djinni_openai_batch_ner.py, clean_openai_annotations.py |
| 19/04/2026 | Cập nhật Gemini NER, weak annotation, train_span_ner và train wrapper. | Codebase Stage 1 đầy đủ hơn, có cả Gemini/OpenAI path, weak labeling và train span-based NER. | djinni_gemini_ner.py, bootstrap_weak_annotations.py, train_span_ner.py |
| 19/04/2026 | Tạo fixed split cho cleaned_v4 để các lần train sau so sánh công bằng. | Split train/dev/test được cố định, tránh test set thay đổi giữa các lần chạy. | artifacts\fixed_split_cleaned_v4.json |
| 19/04/2026 | Train các bản cleaned_v4 hard negative và thử resume checkpoint. | v4 cải thiện rõ so với cleaned_v3, đặc biệt sau khi cố định split và tăng epochs. | artifacts\span_ner_roberta_base_openai_cleaned_v4_hardneg, artifacts\resume_smoke |
| 19/04/2026 | Tạo smoke run cho cleaned_v5 hard negative. | Kiểm tra workflow trước khi train full. | artifacts\span_ner_roberta_base_openai_cleaned_v5_hardneg_smoke |
| 20/04/2026 | Train full cleaned_v5 hard negative. | v5 đạt best dev F1 khoảng 0.6295 và test F1 khoảng 0.6250. | artifacts\span_ner_roberta_base_openai_cleaned_v5_hardneg |
| 21/04/2026 | Xây Stage 2 O*NET utilities. | Có code chuẩn bị O*NET index, mapping entity sang O*NET và test mapping. | onet_mapping.py, prepare_onet_index.py, map_entities_to_onet.py |
| 21/04/2026 | Viết README mô tả pipeline Stage 1/2. | Tài liệu hóa cách lấy dữ liệu, annotate, train NER, build O*NET và map entity. | README.md |
| 26/04/2026 | Build O*NET index từ official O*NET text database. | Tạo được index lớn dùng cho lexical/semantic entity mapping. | artifacts\onet_index.jsonl, artifacts\onet_index.jsonl.summary.json |
| 29/04/2026 | Chạy O*NET lexical mapping trên cleaned_v5. | Mapped rate khoảng 66.92 phần trăm, dùng làm baseline mapping. | artifacts\stage2_onet_mapped_cleaned_v5.jsonl |
| 02/05/2026 | Cập nhật OpenAI synchronous NER script và train wrapper. | Bổ sung/ổn định CLI annotation và PowerShell train flow. | djinni_openai_ner.py, train_roberta_base.ps1 |
| 03/05/2026 | Train NER cleaned_v6 và cập nhật train wrapper hỗ trợ epochs/resume/fixed split. | cleaned_v6 trở thành checkpoint NER tốt nhất hiện tại, dev F1 0.6330 và test F1 0.6251. | artifacts\span_ner_roberta_base_openai_cleaned_v6, train_roberta_base.ps1 |
| 03/05/2026 | Chạy semantic O*NET mapping full cho cleaned_v6. | Mapped rate tăng lên 74.15 phần trăm, tốt hơn lexical v5. | artifacts\stage2_onet_mapped_semantic_v6_full.jsonl |
| 10/05/2026 | Tạo synthetic CV-JD evaluation dataset. | Tạo 1.000 CV-JD pairs, 20 IT roles và 5 mức relevance để đánh giá ranking trực tiếp. | generate_synthetic_eval.py, artifacts\synthetic_eval_dataset.jsonl |
| 10/05/2026 | Kiểm tra chất lượng synthetic dataset và sửa collapsed roles. | Giảm lỗi dữ liệu synthetic, kiểm tra phân phối relevance/role. | artifacts\check_quality.py, artifacts\qc_out.txt, fix_collapsed_roles.py |
| 10/05/2026 | Chạy Stage 3 scoring linear hybrid trên synthetic dataset. | Proposed linear đạt Precision@5 0.400, NDCG@5 0.9496, MAP 0.9262. | score_candidates.py, artifacts\stage3_scores_eval.jsonl, artifacts\eval_results.json |
| 10/05/2026 | Thử CrossEncoder + O*NET soft matching. | Khi dùng công thức tĩnh, kết quả giảm; phát hiện vấn đề calibration giữa feature distributions. | artifacts\stage3_scores_eval_sota.jsonl, artifacts\eval_results_sota.json |
| 10/05/2026 | Train ML ranker XGBoost cho Stage 3. | XGBoost ranker đạt NDCG@5 0.9458 và MAP 0.8970, chứng minh hướng học trọng số khả thi. | train_ml_ranker.py, artifacts\xgb_ranker.json, artifacts\eval_results_ml_sota.json |
| 10/05/2026 | Tổng hợp kết quả thực nghiệm Stage 1, Stage 2, Stage 3. | Có file tổng hợp NER, O*NET mapping, scoring và future work. | EXPERIMENT_RESULTS.md |
| 16/05/2026 | Chuẩn bị benchmark ngoài trên Vanetik vacancy-resume matching dataset. | Chuyển dữ liệu thành docs/pairs, chạy NER, O*NET mapping, scoring và evaluation. | artifacts\external_vanetik_docs.jsonl, artifacts\external_vanetik_pairs.jsonl, artifacts\external_vanetik_eval.json |
| 16/05/2026 | Đánh giá Vanetik theo group_by cv_id. | Proposed chưa vượt rõ semantic/O*NET baseline trên Vanetik; kết quả dùng để nhận diện hạn chế domain shift và label setup. | artifacts\external_vanetik_eval.json |
| 16/05/2026 | Chuẩn bị Kaggle resume data for ranking. | Tạo docs/pairs từ resume_data_for_ranking.csv, sau đó lọc IT sample. | resume_data_for_ranking.csv, prepare_external_benchmark.py |
| 16/05/2026 | Chạy NER/O*NET/scoring trên Kaggle sample và IT sample. | Có sample25, IT sample50, các eval k5/k10/k20; dùng để kiểm tra robustness trên data ngoài. | artifacts\external_kaggle_sample25_*, artifacts\external_kaggle_it_sample50_* |
| 16/05/2026 | Thử tuning trọng số thủ công cho Kaggle IT sample. | Có bản tuned095_005_0 và it_features để kiểm tra tác động semantic/feature weighting. | artifacts\external_kaggle_it_sample50_scores_tuned095_005_0.jsonl, artifacts\external_kaggle_it_sample50_scores_it_features.jsonl |
| 16/05/2026 | Split Kaggle IT pairs thành train/valid/test theo group. | Chuẩn bị cho learning-to-rank hoặc ML ranker trên benchmark ngoài. | split_ranking_pairs.py, artifacts\external_kaggle_it_sample50_pairs_train.jsonl, artifacts\external_kaggle_it_sample50_pairs_valid.jsonl, artifacts\external_kaggle_it_sample50_pairs_test.jsonl |
| 16/05/2026 | Train XGBoost ranker trên Kaggle IT features. | Có xgb_ranker_kaggle_it_features và test scores; hướng ML ranking được kiểm tra trên data ngoài. | artifacts\xgb_ranker_kaggle_it_features.json, artifacts\external_kaggle_it_sample50_scores_test_xgb_it.jsonl |
| 16/05/2026 | Thử cross-encoder trên Kaggle IT test. | Có eval k5/k10/k20 để so sánh với XGB và profile/feature scoring. | artifacts\external_kaggle_it_sample50_scores_test_cross_encoder.jsonl |
| 16/05/2026 | Viết/hoàn thiện scripts phụ cho external benchmark và inference streaming. | Có predict_span_ner để chạy NER batch/resume, evaluate_ranking để tính metrics, prepare_external_benchmark để chuẩn hóa external data. | predict_span_ner.py, evaluate_ranking.py, prepare_external_benchmark.py |
| 17/05/2026 | Dựng backend/frontend demo đơn giản. | Có màn hình nhập CV/JD và output ranking/đề xuất tuyển dụng, dùng FastAPI-style backend và static web UI. | demo_server.py, web\index.html, web\styles.css, web\app.js, run_demo_app.ps1 |
| 23/05/2026 | Tải và giải nén CareerBuilder Job Recommendation Challenge. | Có dữ liệu CareerBuilder cục bộ để benchmark ranking trên implicit feedback. | job-recommendation.zip, job-recommendation |
| 23/05/2026 | Viết benchmark_careerbuilder.py bản đầu. | Chuyển CareerBuilder thành user-job ranking task với positives từ apps.tsv và sampled negatives. | benchmark_careerbuilder.py |
| 23/05/2026 | Chạy CareerBuilder smoke, all-domain 100 và IT 50. | Proposed hybrid bản đầu vượt một số baseline trong sample nhỏ, nhưng cần cải tiến để công bằng hơn. | artifacts\careerbuilder_smoke, artifacts\careerbuilder_all_100, artifacts\careerbuilder_it_50 |
| 23/05/2026 | Cải tiến CareerBuilder benchmark theo các hướng 1, 3, 4, 5, 6. | Thêm multi-seed, nhiều positive/user, profile enrichment, semantic O*NET prototype và validation-based tuning. | benchmark_careerbuilder.py |
| 23/05/2026 | Smoke test semantic/hybrid O*NET scoring trên CareerBuilder. | Semantic O*NET chạy được nhưng chậm trên tập lớn; quyết định dùng lexical O*NET cho benchmark lớn và giữ semantic O*NET như prototype. | artifacts\careerbuilder_improved_smoke |
| 23/05/2026 | Chạy CareerBuilder improved all-domain 200 users, 3 seeds. | Proposed tuned đạt MAP 0.7237, MRR 0.9147, NDCG@5 0.7390, nhỉnh hơn BM25 trên các metric chính. | artifacts\careerbuilder_improved_all_200_s3 |
| 23/05/2026 | Chạy CareerBuilder improved IT-filtered 100 users, 2 seeds. | Proposed tuned đạt MAP 0.5748, MRR 0.8327, NDCG@5 0.5882, nhỉnh hơn BM25 trên MAP/MRR/NDCG@5. | artifacts\careerbuilder_improved_it_100_s2 |
| 23/05/2026 | Viết tài liệu CareerBuilder benchmark. | Ghi rõ CareerBuilder là implicit-feedback job recommendation benchmark, không phải human-labeled CV-JD relevance dataset. | CAREERBUILDER_BENCHMARK.md |
| 24/05/2026 | Viết báo cáo tiếng Việt về dataset, methodology và experimental result. | Bản đầu còn thiếu nhấn mạnh NER, sau đó đã sửa lại để mô tả proposed là Domain NER + O*NET grounded hybrid ranker. | CAREERBUILDER_REPORT_VI.md |
| 24/05/2026 | Xuất báo cáo sang Word và PDF. | Có file docx và pdf để nộp hoặc chỉnh sửa trong Word. | CAREERBUILDER_REPORT_VI.docx, CAREERBUILDER_REPORT_VI.pdf |
| 24/05/2026 | Tạo timeline tổng hợp toàn bộ project. | File hiện tại liệt kê các việc đã làm, đã thử và artifact liên quan từ đầu project đến hiện tại. | PROJECT_TIMELINE_VI.md |

## 3. Các hướng đã thử và bài học rút ra

| Hướng thử nghiệm | Kết quả chính | Kết luận |
| --- | --- | --- |
| Weak/LLM annotation cho NER | Dùng OpenAI/Gemini để tạo span labels quy mô 20k documents. | Cách này giúp có dữ liệu nhanh nhưng tạo nhiễu, cần cleaning nhiều vòng. |
| Span-based RoBERTa NER | v6 đạt test F1 0.6251 và test recall 0.8728. | Recall cao phù hợp pipeline extraction-first, nhưng precision còn là điểm cần cải thiện. |
| Làm sạch label v3 đến v6 | Drop ability, chặn methodology leakage, bỏ project/industry generic, sửa degree/certification, điều chỉnh negative sampling. | Cleaning label có tác động rõ hơn chỉ tăng epochs. |
| Fixed split | Cố định train/dev/test từ v4 trở đi. | Cần thiết để so sánh các bản NER công bằng. |
| O*NET lexical mapping | Mapped rate khoảng 66.92 phần trăm trên v5. | Nhanh, ổn định, nhưng bỏ sót nhiều span diễn đạt khác từ O*NET. |
| O*NET semantic mapping | v6 semantic mapped rate 74.15 phần trăm. | Cải thiện rõ coverage, nhất là WORK_ACTIVITY, nhưng tốn compute hơn. |
| Linear hybrid scoring | Synthetic NDCG@5 0.9496, MAP 0.9262. | Hiệu quả cao khi dữ liệu đúng dạng CV-JD relevance. |
| CrossEncoder + soft matching với công thức tĩnh | Kết quả synthetic giảm so với linear baseline. | Feature mới cần calibration; không thể cộng vào công thức cũ một cách trực tiếp. |
| XGBoost ML ranker | Synthetic NDCG@5 0.9458, MAP 0.8970. | Learning-to-rank là hướng tốt để học trọng số thay vì đặt tay. |
| Vanetik benchmark | Proposed chưa vượt rõ semantic/O*NET baseline. | Cần xem lại label setup, group_by, domain shift và size nhỏ. |
| Kaggle resume data for ranking | Có pipeline lọc IT, score, split train/valid/test và thử XGBoost/cross-encoder. | Hữu ích để stress-test nhưng độ tin cậy nhãn cần thận trọng. |
| CareerBuilder benchmark | Proposed tuned nhỉnh hơn BM25 trên all-domain và IT-filtered metrics chính. | Đây là benchmark tốt để chứng minh ranking trên dữ liệu implicit-feedback lớn, nhưng không phải human-labeled CV-JD relevance. |
| Demo web app | Có backend/frontend nhập CV/JD và trả ranking/đề xuất. | Đủ để demo end-to-end, nhưng còn có thể nâng cấp parser PDF/DOCX và UI. |

## 4. Các file chính hiện tại

| Nhóm | File chính |
| --- | --- |
| Data preparation | prepare_hf_recruitment_subset.py, prepare_seed_subset.py, prepare_external_benchmark.py |
| Annotation | djinni_openai_batch_ner.py, djinni_openai_ner.py, djinni_gemini_ner.py, run_openai_batch_queue.ps1 |
| NER training/inference | train_span_ner.py, train_roberta_base.ps1, predict_span_ner.py, analyze_span_ner_errors.py |
| O*NET | prepare_onet_index.py, onet_mapping.py, map_entities_to_onet.py |
| Scoring/ranking | score_candidates.py, evaluate_ranking.py, split_ranking_pairs.py, train_ml_ranker.py |
| Synthetic evaluation | generate_synthetic_eval.py, artifacts\synthetic_eval_dataset.jsonl |
| CareerBuilder benchmark | benchmark_careerbuilder.py, CAREERBUILDER_BENCHMARK.md |
| Demo app | demo_server.py, run_demo_app.ps1, web\index.html, web\styles.css, web\app.js |
| Reports | EXPERIMENT_RESULTS.md, CAREERBUILDER_REPORT_VI.md, CAREERBUILDER_REPORT_VI.docx, CAREERBUILDER_REPORT_VI.pdf, PROJECT_TIMELINE_VI.md |

## 5. Những điểm nên nhấn mạnh khi viết báo cáo cuối

Phương pháp proposed không nên được mô tả là chỉ kết hợp BM25, SBERT và O*NET. Cách mô tả đúng hơn là Domain-specific NER + O*NET-grounded hybrid ranker. BM25 và SBERT là baseline và cũng là một phần tín hiệu ranking, nhưng đóng góp riêng của project nằm ở việc huấn luyện NER cho miền tuyển dụng IT, chuẩn hóa entity bằng O*NET, và dùng các feature có ý nghĩa nghề nghiệp để ranking.

CareerBuilder nên được trình bày là implicit-feedback job recommendation benchmark. Positive labels là observed applications, còn sampled negatives không chắc là negative thật. Vì vậy kết quả CareerBuilder dùng để chứng minh khả năng ranking trên dữ liệu lớn, còn chất lượng NER phải được trình bày riêng bằng kết quả dev/test trên Djinni/IT recruitment corpus.

Kết quả tốt nhất hiện tại nên ghi theo ba tầng. Tầng NER: cleaned_v6 đạt test F1 0.6251 và recall 0.8728. Tầng O*NET: semantic mapping v6 đạt mapped rate 74.15 phần trăm. Tầng ranking: proposed linear/ML ranker vượt baseline trên synthetic CV-JD evaluation, và proposed tuned nhỉnh hơn BM25 trên CareerBuilder all-domain và IT-filtered ở các metric chính.

