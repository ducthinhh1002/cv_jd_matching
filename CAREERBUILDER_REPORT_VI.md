# Báo cáo thực nghiệm hệ thống CV-JD matching và job recommendation

## 1. Dataset

### 1.1. Tập dữ liệu huấn luyện NER từ Djinni/IT recruitment corpus

Hệ thống proposed không bắt đầu trực tiếp từ bước tính similarity giữa CV và JD, mà trước hết xây dựng một tầng trích xuất thực thể miền tuyển dụng IT. Tầng này được huấn luyện trên tập dữ liệu chú thích NER quy mô khoảng 20.000 tài liệu, gồm cả CV và job descriptions. Sau bước làm sạch cuối cùng, tập dữ liệu dùng để huấn luyện là djinni_ner_annotations_openai_20k_cleaned_v6.jsonl, với tổng cộng 19.851 tài liệu có nhãn được chia thành 15.880 train documents, 1.984 dev documents và 1.987 test documents theo fixed split manifest. Việc cố định split giúp các lần huấn luyện sau có thể so sánh công bằng với nhau.

Các nhãn thực thể được sử dụng gồm TECHNOLOGY, JOB_ROLE, SKILL, WORK_ACTIVITY, INDUSTRY, PROJECT_TYPE, DEGREE và CERTIFICATION. Đây là các nhóm thông tin quan trọng trong bài toán tuyển dụng IT. TECHNOLOGY biểu diễn công nghệ cụ thể như Python, React, AWS hoặc Docker. JOB_ROLE biểu diễn vai trò nghề nghiệp như backend developer, data analyst hoặc QA engineer. SKILL và WORK_ACTIVITY biểu diễn năng lực và hoạt động công việc. DEGREE và CERTIFICATION phục vụ cho các ràng buộc cứng về bằng cấp hoặc chứng chỉ. Nhờ các nhãn này, hệ thống không chỉ nhìn CV và JD như hai đoạn văn bản tự do, mà có thể chuyển chúng thành các thành phần có ý nghĩa nghề nghiệp rõ ràng.

Quá trình xây dựng tập NER không chỉ là gán nhãn một lần. Dữ liệu đã được làm sạch qua nhiều phiên bản. Các phiên bản trước gặp vấn đề do nhãn quá rộng, ví dụ một số cụm methodology hoặc architecture bị nhận nhầm là SKILL, các cụm PROJECT_TYPE quá generic như web application hoặc mobile apps được giữ lại dù không đủ đặc trưng, hoặc DEGREE bị lấy cả cụm không đúng field. Ở các phiên bản sau, các nhãn nhiễu được loại bỏ, các span generic được chặn, các cụm degree/certification được chuẩn hóa hơn, và negative sampling được điều chỉnh để mô hình học tốt hơn ranh giới giữa span hợp lệ và span không hợp lệ.

Mô hình NER được huấn luyện là span-based RoBERTa với backbone roberta-base. Thay vì dùng token classification BIO thông thường, mô hình xét các candidate spans trong văn bản và phân loại từng span vào một trong các nhãn thực thể hoặc O. Thiết kế span-based phù hợp với dữ liệu tuyển dụng vì nhiều thực thể là cụm nhiều token, ví dụ machine learning engineer, cloud infrastructure hoặc bachelor degree in computer science. Mô hình được huấn luyện với CUDA, max length 256, max span width 8, hard negative sampling và fixed train/dev/test split.

Kết quả: Trên dev set, mô hình đạt precision 0.5384, recall 0.8669 và F1 0.6330. Trên test set, mô hình đạt precision 0.5369, recall 0.8728 và F1 0.6251. Recall cao cho thấy mô hình có khả năng bắt được phần lớn các thực thể quan trọng trong CV/JD. Precision vẫn còn thấp hơn recall vì dữ liệu chú thích được sinh từ LLM và vẫn có nhiễu weak-label, nhưng với pipeline matching, recall cao là có giá trị vì hệ thống cần tránh bỏ sót kỹ năng, vai trò hoặc yêu cầu quan trọng trước khi đưa sang bước mapping và scoring.

### 1.2. O*NET occupational knowledge base

Sau khi trích xuất thực thể bằng NER, hệ thống sử dụng O*NET làm nguồn tri thức nghề nghiệp để chuẩn hóa và mở rộng ý nghĩa của các thực thể. O*NET không được dùng như một baseline đơn giản, mà là tầng knowledge grounding cho các thực thể được phát hiện từ CV và JD. Tập O*NET được build thành onet_index.jsonl từ official O*NET database, sau đó dùng để map các entity thuộc nhóm JOB_ROLE, TECHNOLOGY, WORK_ACTIVITY, SKILL và PROJECT_TYPE sang các occupation hoặc descriptor tương ứng.

Kết quả mapping tốt nhất hiện tại là stage2_onet_mapped_semantic_v6_full.jsonl. Trên 19.853 documents, hệ thống ghi nhận 229.075 total entities. Trong đó có 221.859 supported entities có thể đưa vào quy trình mapping, và 164.507 entities được map thành công, tương ứng mapped rate 74.15 phần trăm. So với lexical matching thuần, semantic mapping cải thiện rõ rệt vì nhiều span trong tuyển dụng không trùng chính xác với tên term trong O*NET nhưng vẫn gần về nghĩa. Ví dụ các cụm WORK_ACTIVITY thường dài và diễn đạt đa dạng, nên semantic matching giúp tăng độ phủ tốt hơn lexical overlap.

O*NET mapping có vai trò quan trọng vì nó làm cho matching bớt phụ thuộc vào exact keyword. Hai CV/JD có thể không dùng cùng một cụm từ, nhưng nếu các entity được map về các khái niệm nghề nghiệp gần nhau trong O*NET, hệ thống vẫn có thể ghi nhận sự tương thích. Đây là điểm khác biệt chính giữa proposed method và các baseline chỉ dùng BM25 hoặc cosine similarity.

### 1.3. CareerBuilder benchmark dataset

Để đánh giá khả năng ranking trên dữ liệu thực tế lớn hơn, hệ thống sử dụng CareerBuilder Job Recommendation Challenge trên Kaggle. Các tệp chính gồm apps.tsv, users.tsv, user_history.tsv và jobs.tsv nằm trong jobs.zip. Đây là implicit-feedback recommendation dataset. Nếu một user apply vào một job, cặp user-job đó được xem là positive. Các job mà user không apply được lấy mẫu làm negative, nhưng negative này chỉ là unobserved interaction, không đảm bảo đó là negative thật. Vì vậy, CareerBuilder được dùng để đánh giá khả năng ranking và recommendation, không thay thế hoàn toàn cho đánh giá chất lượng matching có nhãn thủ công.

Trong thí nghiệm CareerBuilder, user profile đóng vai trò gần với CV hoặc candidate profile. Profile được tạo từ thông tin người dùng, lịch sử nghề nghiệp và trong phiên bản cải tiến có bổ sung nội dung từ các job mà user đã từng apply. Job posting đóng vai trò JD. Nhiệm vụ của hệ thống là xếp hạng các job ứng viên sao cho các job mà user thật sự apply nằm càng cao càng tốt.

Hai cấu hình được đánh giá. Cấu hình all-domain sử dụng job từ nhiều lĩnh vực trong CareerBuilder. Cấu hình IT-filtered chỉ giữ các job liên quan đến công nghệ thông tin, phù hợp hơn với mục tiêu của đề tài. Trong thiết lập cải tiến, mỗi user có tối đa 3 positive jobs và 30 negative jobs. Thí nghiệm được chạy với nhiều random seeds để giảm phụ thuộc vào một lần negative sampling duy nhất.

## 2. Methodology

### 2.1. Tổng quan phương pháp đề xuất

 Kiến trúc chính gồm ba bước: trích xuất thực thể miền tuyển dụng bằng NER, chuẩn hóa thực thể bằng O*NET, và xếp hạng bằng hybrid ranking score. .

Ở bước đầu tiên, span-based RoBERTa NER nhận đầu vào là raw CV hoặc raw JD và trích xuất các entity quan trọng như technology, job role, skill, work activity, project type, degree và certification. Tầng này giúp hệ thống biết chính xác đoạn nào trong văn bản là kỹ năng, đoạn nào là vai trò công việc, đoạn nào là bằng cấp hoặc chứng chỉ. Đây là điểm mà BM25 và SBERT không có, vì hai baseline đó chỉ xử lý văn bản dưới dạng từ khóa hoặc vector embedding tổng quát.

Ở bước thứ hai, các entity được map sang O*NET. Việc mapping này giúp chuẩn hóa nhiều cách diễn đạt khác nhau về cùng một khái niệm nghề nghiệp. Ví dụ, software engineer, backend developer và web application developer có thể gần nhau ở tầng occupation hoặc work activity; cloud deployment, AWS infrastructure và DevOps activity có thể được nhận diện là liên quan về mặt kỹ năng/công việc. Nhờ đó, hệ thống có thể thực hiện soft matching thay vì chỉ yêu cầu exact string match.

Ở bước thứ ba, hệ thống tạo các feature phục vụ ranking. Các feature gồm semantic similarity, lexical relevance, entity/O*NET overlap, skill coverage, role alignment, seniority matching và hard constraint satisfaction. Semantic similarity giúp bắt ngữ nghĩa tổng quát giữa profile và job. Lexical relevance giúp tận dụng keyword khi title hoặc technology trùng trực tiếp. Entity/O*NET overlap giúp so khớp các khái niệm đã được chuẩn hóa. Skill coverage đo mức độ profile bao phủ các skill quan trọng trong JD. Role alignment kiểm tra vai trò nghề nghiệp có phù hợp không. Seniority matching kiểm tra cấp bậc như junior, senior, lead hoặc manager. Hard constraints xử lý các yêu cầu có tính bắt buộc như số năm kinh nghiệm, degree hoặc certification.

Vì vậy, proposed method không loại bỏ NER. Ngược lại, NER là tầng đầu tiên tạo ra structured evidence cho toàn bộ hệ thống. Nếu không có NER, hệ thống chỉ còn là text similarity hoặc keyword retrieval. NER giúp phương pháp đề xuất có khả năng giải thích vì sao một CV/JD phù hợp: phù hợp ở skill nào, role nào, technology nào, và có thỏa các constraint quan trọng hay không.

### 2.2. Các baseline được so sánh

BM25 được dùng làm baseline lexical retrieval. User profile hoặc CV được xem như query, còn job posting hoặc JD được xem như document. BM25 rất mạnh khi hai bên dùng cùng keyword, đặc biệt là job title, technology name hoặc role name. Trên CareerBuilder, BM25 là baseline khó vượt vì dữ liệu user history và job title có nhiều từ khóa trùng trực tiếp.

SBERT cosine similarity được dùng làm baseline semantic retrieval. Profile và job được encode thành embedding, sau đó tính cosine similarity. Phương pháp này có thể bắt được tương đồng ngữ nghĩa ngay cả khi từ khóa không trùng hoàn toàn. Tuy nhiên, khi profile ngắn hoặc thiếu thông tin chi tiết, embedding tổng quát dễ không đủ mạnh để phân biệt các job gần nhau.

O*NET entity-only được dùng để kiểm tra riêng tác động của ontology/entity matching. Baseline này chỉ dùng overlap giữa các term hoặc entity liên quan O*NET. Kết quả cho thấy O*NET entity-only không đủ mạnh nếu dùng độc lập, nhưng lại có giá trị khi trở thành một phần của proposed hybrid pipeline.

### 2.3. Các thử nghiệm NER đã thực hiện

Giai đoạn NER đã được thử qua nhiều phiên bản làm sạch dữ liệu. Phiên bản cleaned_v3 cho kết quả test F1 khoảng 0.5740. Sau khi loại bỏ nhãn ability, chặn các cụm methodology hoặc architecture leakage trong SKILL, loại bỏ các PROJECT_TYPE generic và làm sạch một số INDUSTRY generic, phiên bản hard negative cleaned_v4 tăng test F1 lên khoảng 0.6036 ở lần chạy 6 epochs. Khi cố định split và tăng lên 8 epochs, v4 đạt best dev F1 0.6293 và test F1 0.6228.

Sau đó, phiên bản cleaned_v5 tập trung sửa PROJECT_TYPE, INDUSTRY, DEGREE và CERTIFICATION, đồng thời vẫn train trên fixed split cũ để so sánh công bằng. Kết quả v5 đạt best dev F1 0.6295 và test F1 0.6250. Phiên bản cleaned_v6 tiếp tục trim các noisy prefixes, giảm negative sampling multiplier từ 10 xuống 5 và giữ hard negative sampling. Kết quả v6 đạt best dev F1 0.6330 và test F1 0.6251. Đây là checkpoint NER tốt nhất hiện tại và được xem là tầng entity extraction chính của hệ thống.

Bảng dưới tóm tắt tiến trình cải thiện NER:

| Phiên bản | Cải tiến chính | Best dev F1 | Test F1 |
| --- | --- | ---: | ---: |
| cleaned_v3 | Dữ liệu OpenAI cleaned ban đầu | 0.5775 | 0.5740 |
| cleaned_v4 hardneg 6 epochs | Drop ability, chặn leakage, hard negative sampling | 0.6085 | 0.6036 |
| cleaned_v5 hardneg | Sửa PROJECT_TYPE, INDUSTRY, DEGREE, CERTIFICATION | 0.6295 | 0.6250 |
| cleaned_v6 | Trim noisy prefixes, giảm negative multiplier xuống 5 | 0.6330 | 0.6251 |

Kết quả này cho thấy phần NER không bị bỏ qua mà đã được phát triển như một module độc lập. Dù precision còn cần cải thiện, recall cao giúp pipeline không bỏ sót nhiều entity quan trọng. Trong hệ thống thực tế, các entity này được đưa sang O*NET mapping và scoring để tạo bằng chứng giải thích cho quyết định matching.

### 2.4. Các thử nghiệm O*NET mapping

Sau NER, hệ thống thử hai hướng mapping chính: lexical matching và semantic matching. Lexical matching nhanh và ổn định nhưng phụ thuộc vào exact token overlap. Semantic matching dùng sentence-transformers/all-MiniLM-L6-v2 để so sánh span được trích xuất với O*NET descriptors trong embedding space, nhờ đó bắt được các cách diễn đạt khác nhau nhưng cùng ý nghĩa.

Kết quả tốt nhất hiện tại là semantic mapping trên cleaned_v6. Tổng số entity được map là 164.507 trên 221.859 supported entities, đạt mapped rate 74.15 phần trăm. So với lexical v5, semantic mapping cải thiện tổng mapped rate từ 66.92 phần trăm lên 74.15 phần trăm. Nhãn WORK_ACTIVITY được cải thiện mạnh nhất vì các hoạt động công việc thường dài, đa dạng và khó exact match. Nhãn TECHNOLOGY và JOB_ROLE cũng có độ phủ cao hơn sau semantic mapping.

### 2.5. Các thử nghiệm scoring và ranking

Ở tầng scoring, hệ thống đã thử nhiều cách kết hợp. Phiên bản đầu là linear hybrid scoring với trọng số tĩnh, kết hợp semantic similarity, O*NET importance và hard constraints. Trên synthetic evaluation dataset 1.000 CV-JD pairs, linear baseline đạt NDCG@5 khoảng 0.949 và MAP khoảng 0.926, cao hơn các baseline riêng lẻ. Điều này cho thấy khi có nhãn relevance rõ hơn và dữ liệu được thiết kế cho CV-JD matching, việc kết hợp entity/constraint với semantic similarity có hiệu quả rõ rệt.

Sau đó, hệ thống thử nâng cấp semantic component bằng cross-encoder và O*NET soft matching. Tuy nhiên, khi vẫn dùng công thức tuyến tính tĩnh, kết quả giảm vì phân phối điểm của cross-encoder khác cosine similarity. Điều này cho thấy vấn đề không nằm ở việc thêm feature mới, mà nằm ở cách calibration giữa các feature. Vì vậy, hệ thống thử ML ranker bằng XGBoost để học trọng số từ dữ liệu. Trên synthetic dataset, XGBoost ranker đạt NDCG@5 khoảng 0.946 và MAP khoảng 0.897, vẫn giữ hiệu quả cao và linh hoạt hơn công thức thủ công.

Trên CareerBuilder, vì đây là implicit-feedback job recommendation dataset chứ không phải CV-JD relevance labels thủ công, hệ thống dùng proposed tuned hybrid. Phiên bản này giữ các feature chính của proposed method nhưng tune trọng số trên validation users. Việc tuning là cần thiết vì CareerBuilder có đặc trưng khác synthetic CV-JD dataset: BM25 rất mạnh do user history và job title chứa nhiều keyword trùng trực tiếp. Nếu dùng trọng số tĩnh, proposed method có thể bị các feature entity hoặc semantic kéo lệch. Khi tune trên validation, hệ thống học cách cân bằng lại các tín hiệu cho đúng đặc trưng dataset.

## 3. Experimental Result

### 3.1. Kết quả NER trên Djinni/IT recruitment corpus

Kết quả NER tốt nhất hiện tại đến từ checkpoint artifacts\span_ner_roberta_base_openai_cleaned_v6\span_ner.pt. Mô hình dùng roberta-base với kiến trúc span-based NER. Tập dữ liệu được chia cố định thành 15.880 train documents, 1.984 dev documents và 1.987 test documents.

| Metric | Dev set | Test set |
| --- | ---: | ---: |
| Precision | 0.5384 | 0.5269 |
| Recall | 0.8669 | 0.8728 |
| F1 | 0.6330 | 0.6251 |

Kết quả cho thấy mô hình có recall rất cao, khoảng 87 phần trăm trên cả dev và test. Đây là đặc điểm phù hợp với pipeline extraction-first, vì bỏ sót entity quan trọng có thể làm giảm khả năng matching ở các bước sau. Precision còn khoảng 49 phần trăm, phản ánh việc dữ liệu weak labels từ LLM vẫn còn nhiễu. Tuy nhiên, các bước cleaning từ v3 đến v6 đã giúp F1 tăng rõ rệt từ khoảng 0.5740 lên 0.6251 trên test set.

### 3.2. Kết quả O*NET mapping

Kết quả mapping tốt nhất hiện tại sử dụng input djinni_ner_annotations_openai_20k_cleaned_v6.jsonl và O*NET index artifacts\onet_index.jsonl. Phương pháp mapping dùng top_k 5, min_score 0.35, embedding model sentence-transformers/all-MiniLM-L6-v2 và chạy trên CUDA.

| Chỉ số | Giá trị |
| --- | ---: |
| Documents | 19.853 |
| Total entities | 229.075 |
| Supported entities | 221.859 |
| Mapped entities | 164.507 |
| Mapped rate | 74.15 phần trăm |

Chi tiết theo nhãn cho thấy JOB_ROLE và TECHNOLOGY là hai nhóm có độ phủ cao nhất. WORK_ACTIVITY được cải thiện rõ nhờ semantic matching. DEGREE, CERTIFICATION và INDUSTRY không được map trong O*NET stage vì các nhóm này không phải trọng tâm của O*NET descriptor matching hiện tại; chúng phù hợp hơn cho hard constraints hoặc rule-based validation trong tầng scoring.

| Label | Entities | Supported | Mapped |
| --- | ---: | ---: | ---: |
| JOB_ROLE | 24.438 | 24.438 | 24.106 |
| TECHNOLOGY | 154.354 | 154.354 | 120.519 |
| WORK_ACTIVITY | 21.420 | 21.420 | 10.811 |
| SKILL | 17.878 | 17.878 | 6.237 |
| PROJECT_TYPE | 3.769 | 3.769 | 2.834 |
| INDUSTRY | 5.433 | 0 | 0 |
| CERTIFICATION | 626 | 0 | 0 |
| DEGREE | 1.157 | 0 | 0 |

### 3.4. Kết quả ranking trên CareerBuilder IT-filtered

Thực nghiệm IT-filtered chỉ giữ các job liên quan đến công nghệ thông tin. Mỗi seed yêu cầu 100 users, 3 positive jobs mỗi user và 30 negative jobs mỗi user. Do validation split 25 phần trăm, mỗi seed dùng 75 users để đánh giá. Kết quả được tính trung bình trên 2 seeds là 13 và 42.

| Method | MAP mean+/-std | MRR mean+/-std | NDCG@5 mean+/-std | NDCG@10 mean+/-std | Recall@5 mean+/-std |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.5718+/-0.0382 | 0.8323+/-0.0229 | 0.5837+/-0.0366 | 0.6421+/-0.0368 | 0.5422+/-0.0356 |
| SBERT cosine | 0.5465+/-0.0217 | 0.7887+/-0.0163 | 0.5559+/-0.0139 | 0.6362+/-0.0254 | 0.5378+/-0.0267 |
| O*NET entity-only | 0.3128+/-0.0161 | 0.4966+/-0.0009 | 0.3103+/-0.0282 | 0.3597+/-0.0284 | 0.3200+/-0.0489 |
| Proposed static | 0.5641+/-0.0195 | 0.8194+/-0.0257 | 0.5794+/-0.0126 | 0.6578+/-0.0196 | 0.5578+/-0.0244 |
| Proposed tuned | 0.5748+/-0.0351 | 0.8327+/-0.0212 | 0.5882+/-0.0318 | 0.6446+/-0.0342 | 0.5489+/-0.0289 |

Trên IT-filtered benchmark, proposed tuned đạt MAP, MRR và NDCG@5 cao nhất. So với BM25, proposed tuned tăng MAP từ 0.5718 lên 0.5748, MRR từ 0.8323 lên 0.8327 và NDCG@5 từ 0.5837 lên 0.5882. Khoảng cách không lớn, nhưng kết quả này vẫn quan trọng vì BM25 vốn rất mạnh trên CareerBuilder. Proposed static đạt NDCG@10 và Recall@5 cao hơn proposed tuned, cho thấy nếu mục tiêu là recall rộng hơn ở top 10, objective tuning có thể được điều chỉnh lại.

### 3.4. Kết quả stage scoring trên synthetic CV-JD evaluation

Ngoài CareerBuilder, hệ thống cũng được đánh giá trên synthetic CV-JD evaluation dataset gồm 1.000 CV-JD pairs, 20 IT roles và 5 mức relevance. Tập này được dùng để kiểm tra trực tiếp khả năng xếp hạng CV-JD theo mức độ phù hợp, trong khi CareerBuilder kiểm tra khả năng recommendation theo implicit feedback.

| Method | Precision@5 | NDCG@5 | MAP | Spearman rho |
| --- | ---: | ---: | ---: | ---: |
| Pure O*NET Matching | 0.400 | 0.544 | 0.325 | 0.520 |
| Hard Constraint Base | 0.400 | 0.813 | 0.710 | 0.227 |
| Pure Semantic Similarity | 0.400 | 0.882 | 0.860 | 0.813 |
| Proposed System Linear Baseline | 0.400 | 0.949 | 0.926 | 0.810 |
| Proposed System CrossEncoder + Soft Match | 0.400 | 0.839 | 0.738 | 0.476 |
| Proposed System ML Ranker XGBoost | 0.400 | 0.946 | 0.897 | 0.580 |

Kết quả synthetic cho thấy khi bài toán là CV-JD relevance trực tiếp, proposed system vượt các thành phần đơn lẻ rõ rệt. Linear hybrid đạt NDCG@5 0.949 và MAP 0.926. ML Ranker XGBoost đạt NDCG@5 0.946 và MAP 0.897. Việc CrossEncoder + Soft Match giảm khi dùng công thức tĩnh cho thấy các feature mới cần calibration, không thể chỉ thêm vào rồi dùng trọng số cũ. Đây là lý do phiên bản tuned hoặc ML ranker hợp lý hơn cho báo cáo cuối.


## 5. Kết luận
Kết quả thực nghiệm cho thấy từng tầng của pipeline đều có đóng góp riêng. NER giúp biến raw CV/JD thành structured entities. O*NET mapping giúp chuẩn hóa entity và hỗ trợ soft matching. Ranking layer kết hợp các tín hiệu văn bản, ontology và constraint để đưa ra thứ tự phù hợp hơn. Trên CareerBuilder, proposed tuned vượt BM25 ở các metric chính dù BM25 là baseline rất mạnh. Trên synthetic CV-JD evaluation, hybrid scoring vượt rõ các thành phần đơn lẻ. 
