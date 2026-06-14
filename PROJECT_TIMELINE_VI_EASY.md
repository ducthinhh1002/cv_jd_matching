
## 1. Tóm tắt ngắn gọn project đang làm gì

Project này xây dựng một hệ thống hỗ trợ so khớp CV với JD, tức là so sánh hồ sơ ứng viên với mô tả công việc để đánh giá mức độ phù hợp. Mục tiêu cuối cùng là hệ thống có thể đọc nội dung CV và JD, nhận ra các thông tin quan trọng như kỹ năng, công nghệ, vai trò công việc, bằng cấp, chứng chỉ, kinh nghiệm, sau đó xếp hạng hoặc đề xuất ứng viên phù hợp.

Hệ thống không chỉ so sánh hai đoạn văn bản bằng từ khóa. Pipeline hiện tại gồm ba phần chính.

Phần thứ nhất là trích xuất thông tin quan trọng từ CV/JD. Ví dụ hệ thống cần biết “Python” là công nghệ, “Backend Developer” là vai trò công việc, “AWS deployment” là hoạt động công việc, “Bachelor of Computer Science” là bằng cấp. Phần này được gọi là NER, hiểu đơn giản là mô hình tự động đánh dấu các cụm thông tin quan trọng trong văn bản.

Phần thứ hai là chuẩn hóa các thông tin đó bằng O*NET. O*NET có thể hiểu như một kho tri thức nghề nghiệp. Nó giúp hệ thống biết các kỹ năng hoặc vai trò nào gần nhau, thay vì chỉ so chữ giống nhau. Ví dụ hai cụm từ có thể viết khác nhau nhưng vẫn liên quan đến cùng một nhóm nghề hoặc kỹ năng.

Phần thứ ba là chấm điểm và xếp hạng. Sau khi đã có thông tin được trích xuất và chuẩn hóa, hệ thống kết hợp nhiều tín hiệu: độ giống nhau về ý nghĩa, từ khóa trùng nhau, kỹ năng được bao phủ, vai trò có đúng không, mức seniority có phù hợp không, và các yêu cầu bắt buộc như bằng cấp hoặc chứng chỉ.

## 2. Kết quả hiện tại của project

Mô hình trích xuất thông tin tốt nhất hiện tại là bản cleaned_v6. Trên tập kiểm tra, mô hình đạt F1 khoảng 0.6251 và recall khoảng 0.8728. Nói dễ hiểu, mô hình tìm được phần lớn các thông tin quan trọng trong CV/JD, tuy vẫn còn một số trường hợp nhận nhầm do dữ liệu gán nhãn ban đầu có nhiễu.

Bước chuẩn hóa thông tin bằng O*NET tốt nhất hiện tại đạt tỷ lệ map khoảng 74.15 phần trăm trên các thực thể được hỗ trợ. Điều này nghĩa là phần lớn các thông tin trích xuất được đã có thể liên kết sang kho tri thức nghề nghiệp.

Ở bước xếp hạng, hệ thống đã được đánh giá trên nhiều bộ dữ liệu. Trên bộ synthetic CV-JD evaluation, phương pháp hybrid đạt NDCG@5 khoảng 0.9496 và MAP khoảng 0.9262. Trên CareerBuilder benchmark, phiên bản proposed tuned nhỉnh hơn BM25 trên các chỉ số chính. BM25 là baseline từ khóa rất mạnh, nên việc vượt được baseline này là tín hiệu tốt.

## 3. Timeline chi tiết các việc đã làm và đã thử

| Mốc thời gian trình bày | Nhóm công việc | Việc đã làm hoặc đã thử, viết theo cách dễ hiểu | Kết quả hoặc ý nghĩa |
| --- | --- | --- | --- |
| 14/03/2026 | Thu thập dữ liệu CV/JD | Viết script lấy dữ liệu từ các bộ recruitment dataset trên Hugging Face. Script này giúp lấy một lượng lớn CV và JD, chuẩn hóa format, phân biệt đâu là CV, đâu là JD, và có thể lọc các mẫu liên quan đến IT. | Có pipeline lấy dữ liệu đầu vào thay vì xử lý thủ công từng file. |
| 20/03/2026 | Làm sạch và chọn mẫu dữ liệu | . Script cũng làm sạch văn bản, bỏ dòng lỗi, chuẩn hóa khoảng trắng và loại các mẫu quá ngắn hoặc không hữu ích. | Dữ liệu đầu vào sạch hơn, giảm lãng phí khi gửi cho LLM gán nhãn. |
| 27/03/2026 | Thử quy trình gán nhãn bằng OpenAI Batch | Tạo thử một batch nhỏ để kiểm tra cách gửi dữ liệu cho OpenAI gán nhãn hàng loạt. Mục tiêu là xác nhận format request đúng, output có thể tải về và có thể ghép lại thành dữ liệu huấn luyện. | Xác nhận được hướng dùng Batch API để gán nhãn rẻ hơn và phù hợp với dữ liệu lớn. |
| 02/04/2026 | Gán nhãn dữ liệu lớn | Chạy quy trình batch annotation cho tập dữ liệu IT 20.000 tài liệu. LLM được dùng để đánh dấu các cụm như công nghệ, kỹ năng, vai trò công việc, bằng cấp, chứng chỉ trong CV/JD. | Tạo được nền dữ liệu gán nhãn quy mô lớn cho bước train NER. |
| 03/04/2026 | Kiểm tra lỗi mô hình NER | Viết công cụ phân tích lỗi cho NER. xem mô hình đang nhận nhầm nhãn nào, bỏ sót cụm nào, và độ tự tin của các dự đoán ra sao. | Có công cụ để hiểu vì sao mô hình sai, từ đó quyết định nên làm sạch dữ liệu ở đâu. |
| 03/04/2026 | Train NER bản đầu tiên | Train mô hình RoBERTa span-based NER trên dữ liệu OpenAI annotation ban đầu. Đây là bản đầu để kiểm tra toàn bộ pipeline từ dữ liệu đến checkpoint. | Có checkpoint đầu tiên, nhưng chất lượng còn thấp hơn các bản làm sạch sau. |
| 04/04/2026 | Hoàn thiện pipeline gán nhãn OpenAI | Cải thiện script OpenAI batch NER để có thể chuẩn bị batch, submit, theo dõi trạng thái, tải output và ghép kết quả. | Quy trình gán nhãn hàng loạt ổn định hơn, phù hợp khi cần gán nhãn nhiều dữ liệu. |
| 04/04/2026 | Làm sạch dữ liệu gán nhãn | Viết script clean_openai_annotations.py để xử lý các lỗi trong annotation như span lệch, nhãn nhiễu hoặc format chưa đúng. | Dữ liệu huấn luyện sạch hơn, giúp mô hình NER học ổn định hơn. |
| 05/04/2026 | Bổ sung các hướng gán nhãn khác | Cập nhật thêm Gemini NER và weak annotation. Việc này giúp có nhiều lựa chọn gán nhãn: dùng OpenAI, Gemini, hoặc luật đơn giản để tạo nhãn yếu. | Project linh hoạt hơn, không phụ thuộc hoàn toàn vào một nguồn gán nhãn. |
| 05/04/2026 | Hoàn thiện script train NER | Cập nhật train_span_ner.py và train_roberta_base.ps1 để dễ chạy mô hình RoBERTa hơn. | Việc train mô hình trở nên lặp lại được và ít lỗi thao tác hơn. |
| 05/04/2026 | Thử hard negative sampling | Train bản cleaned_v4 hard negative. Hard negative sampling nghĩa là đưa thêm nhiều span gần giống entity nhưng thực ra không phải entity để mô hình học cách phân biệt. | Mô hình bớt nhận nhầm các cụm không nên gán nhãn. |
| 06/04/2026 | Train full cleaned_v5 | Train bản cleaned_v5 với các chỉnh sửa tập trung vào PROJECT_TYPE, INDUSTRY, DEGREE và CERTIFICATION. Các span quá chung chung được loại bỏ, ví dụ các cụm kiểu web application nếu không đủ đặc trưng. | v5 đạt best dev F1 khoảng 0.6295 và test F1 khoảng 0.6250. |
| 07/04/2026 | Xây phần O*NET | Viết các script chuẩn bị và sử dụng O*NET. O*NET được dùng như kho tri thức nghề nghiệp để chuẩn hóa các kỹ năng, vai trò và hoạt động công việc sau khi NER trích xuất. | Hình thành Stage 2 của pipeline: từ entity thô sang entity có liên kết tri thức nghề nghiệp. |
| 12/04/2026 | Build O*NET index | Tải và xử lý official O*NET database để tạo file index dùng cho mapping. Index này giống như một bảng tra cứu lớn về nghề nghiệp, kỹ năng và hoạt động công việc. | Có O*NET index cục bộ để map entity mà không cần xử lý lại từ đầu. |
| 15/04/2026 | Chạy lexical O*NET mapping | Thử cách map entity sang O*NET bằng khớp từ khóa. Ví dụ nếu entity có từ giống hoặc gần giống với term trong O*NET thì map vào term đó. | Mapped rate khoảng 66.92 phần trăm. Đây là baseline nhanh nhưng còn bỏ sót nhiều cụm diễn đạt khác. |
| 18/04/2026 | Cập nhật OpenAI NER script | Cải thiện script gán nhãn OpenAI dạng synchronous và train wrapper để dễ chạy, dễ truyền tham số, hỗ trợ epochs và checkpoint. | Pipeline train/gán nhãn ổn định hơn, dễ dùng hơn trong các lần thử tiếp theo. |
| 19/04/2026 | Train cleaned_v6 | Train bản NER cleaned_v6. Bản này tiếp tục xử lý các tiền tố nhiễu trong span, giảm negative sampling multiplier và giữ fixed split. | cleaned_v6 trở thành checkpoint NER tốt nhất hiện tại, dev F1 0.6330 và test F1 0.6251. |
| 19/04/2026 | Chạy semantic O*NET mapping full | Thử mapping O*NET bằng embedding semantic thay vì chỉ khớp từ khóa. Cách này giúp map được các cụm có nghĩa giống nhau dù không viết giống hệt. | Mapped rate tăng lên 74.15 phần trăm, tốt hơn lexical mapping. |
| 26/04/2026 | Tạo bộ đánh giá synthetic CV-JD | Tạo một bộ dữ liệu đánh giá giả lập gồm 1.000 cặp CV-JD, 20 vai trò IT và 5 mức độ phù hợp. Bộ này giúp kiểm tra trực tiếp hệ thống có xếp CV phù hợp lên cao không. | Có benchmark chủ động kiểm soát được nhãn relevance để đánh giá Stage 3. |
| 26/04/2026 | Kiểm tra chất lượng synthetic data | Viết script kiểm tra dữ liệu synthetic, xem role có bị trùng quá nhiều không, nhãn relevance có hợp lý không, và có lỗi collapsed roles không. | Dữ liệu synthetic đáng tin hơn trước khi dùng để report kết quả. |
| 26/04/2026 | Chạy scoring hybrid bản tuyến tính | Thử công thức chấm điểm kết hợp semantic similarity, O*NET importance và hard constraints. Đây là bản proposed linear baseline. | Đạt NDCG@5 khoảng 0.9496 và MAP khoảng 0.9262 trên synthetic dataset. |
| 26/04/2026 | Thử CrossEncoder | Thử mô hình CrossEncoder để so sánh CV và JD sâu hơn ở mức token/context. Đây là hướng thường mạnh hơn cosine embedding nếu được hiệu chỉnh đúng. | Khi đưa vào công thức tĩnh, kết quả giảm. Kết luận: feature mới cần được cân chỉnh trọng số, không thể cộng thẳng vào công thức cũ. |
| 26/04/2026 | Thử O*NET soft matching | Thử matching mềm theo O*NET để cho điểm một phần nếu hai kỹ năng hoặc hoạt động công việc gần nhau, thay vì yêu cầu trùng chính xác. | Ý tưởng hợp lý về mặt phương pháp, nhưng cần kết hợp với calibration hoặc ML ranker để phát huy tốt. |
| 26/04/2026 | Train XGBoost ranker | Thử dùng mô hình học máy XGBoost để học cách kết hợp các feature thay vì tự đặt trọng số bằng tay. | XGBoost đạt NDCG@5 khoảng 0.9458 và MAP khoảng 0.8970. Hướng learning-to-rank được xác nhận là khả thi. |
| 02/05/2026 | Chuẩn bị Vanetik benchmark | Chuyển Vanetik vacancy-resume matching dataset thành dạng docs và pairs để pipeline có thể xử lý. Sau đó chạy NER, O*NET mapping, scoring và evaluation. | Có kết quả benchmark ngoài đầu tiên, giúp kiểm tra project trên dữ liệu không tự tạo. |
| 02/05/2026 | Chuẩn bị Kaggle resume data for ranking | Đọc file resume_data_for_ranking.csv, tạo docs/pairs và lọc các mẫu liên quan IT để phù hợp với hướng nghiên cứu của project. | Có benchmark ngoài thứ hai, nhưng cần thận trọng vì nhãn có thể không đáng tin bằng human-labeled dataset. |
| 02/05/2026 | Chạy pipeline trên Kaggle sample | Chạy NER, O*NET mapping, scoring và evaluation trên các sample Kaggle, gồm sample nhỏ và IT sample. | Có nhiều file eval k5/k10/k20 để so sánh hiệu quả ranking ở các top-k khác nhau. |
| 02/05/2026 | Thử chỉnh trọng số thủ công trên Kaggle | Thử các công thức score khác nhau, ví dụ tăng mạnh semantic score và giảm hoặc bỏ các thành phần khác để xem metric thay đổi ra sao. | Giúp hiểu feature nào có ích trên dữ liệu Kaggle, nhưng chưa đủ để kết luận cuối vì nhãn còn cần kiểm chứng. |
| 02/05/2026 | Train XGBoost trên Kaggle IT | Dùng các feature đã tạo để train XGBoost ranker trên Kaggle IT sample. | Có mô hình xgb_ranker_kaggle_it_features và kết quả test để so sánh với công thức scoring. |
| 02/05/2026 | Thử CrossEncoder trên Kaggle IT | Chạy thêm cross-encoder score trên test set để xem semantic matching sâu hơn có cải thiện không. | Có kết quả so sánh với XGBoost và score profile/features. |
| 02/05/2026 | Hoàn thiện scripts benchmark ngoài | Viết và cập nhật các script như predict_span_ner, evaluate_ranking, prepare_external_benchmark, split_ranking_pairs và train_ml_ranker. | Pipeline benchmark ngoài chạy được end-to-end hơn, có thể reuse cho các dataset mới. |
| 03/05/2026 | Dựng web demo | Xây backend và frontend đơn giản. Người dùng có thể nhập CV và JD, sau đó hệ thống trả về điểm phù hợp hoặc đề xuất tuyển dụng. | Có demo end-to-end để trình bày sản phẩm, không chỉ có script chạy dòng lệnh. |
| 09/05/2026 | Tải CareerBuilder dataset | Tải và giải nén Kaggle CareerBuilder Job Recommendation Challenge. Dataset này chứa thông tin người dùng, job và lịch sử apply. | Có benchmark lớn dạng implicit feedback để kiểm tra khả năng ranking thực tế hơn. |
| 09/05/2026 | Viết benchmark CareerBuilder bản đầu | Chuyển CareerBuilder thành bài toán xếp hạng job cho từng user. Positive là job user đã apply, negative là job user chưa apply được lấy mẫu. | Có script benchmark_careerbuilder.py bản đầu. |
| 09/05/2026 | Chạy CareerBuilder smoke/all-domain/IT sample | Chạy thử trên sample nhỏ để đảm bảo benchmark đúng format và có thể tính metric. | Proposed hybrid bản đầu có kết quả khả quan trên sample nhỏ nhưng cần setup công bằng hơn. |
| 09/05/2026 | Cải tiến CareerBuilder benchmark | Thêm nhiều positive mỗi user, nhiều random seeds, profile enrichment bằng các job đã apply, validation-based weight tuning và semantic O*NET prototype. | Benchmark công bằng và ổn định hơn so với bản đầu. |
| 09/05/2026 | Thử semantic O*NET trên CareerBuilder | Chạy smoke test với semantic/hybrid O*NET term-bank scoring để xem hướng này có hoạt động không. | Chạy được nhưng chậm trên tập lớn, nên benchmark chính dùng lexical O*NET để đảm bảo thời gian chạy. |
| 09/05/2026 | Chạy CareerBuilder all-domain improved | Chạy benchmark all-domain với 200 users, 3 positive/user, 30 negative/user và 3 seeds. | Proposed tuned đạt MAP 0.7237, MRR 0.9147, NDCG@5 0.7390, nhỉnh hơn BM25 trên các metric chính. |
| 09/05/2026 | Chạy CareerBuilder IT-filtered improved | Chạy benchmark chỉ trên các job liên quan IT với 100 users và 2 seeds. | Proposed tuned đạt MAP 0.5748, MRR 0.8327, NDCG@5 0.5882, nhỉnh hơn BM25 trên MAP/MRR/NDCG@5. |
| 09/05/2026 | Viết tài liệu CareerBuilder benchmark | Viết tài liệu giải thích cách dùng CareerBuilder, positive/negative được tạo thế nào, và vì sao đây là implicit-feedback benchmark. | Có file CAREERBUILDER_BENCHMARK.md làm tài liệu kỹ thuật cho benchmark. |
| 10/05/2026 | Viết báo cáo tiếng Việt | Viết báo cáo có các phần dataset, methodology và experimental result. Bản đầu còn làm proposed giống như chỉ cộng các baseline, sau đó đã chỉnh lại để nhấn mạnh pipeline NER + O*NET + ranking. | Có file CAREERBUILDER_REPORT_VI.md để dán vào Word hoặc dùng làm báo cáo. |

## 4. Các hướng đã thử 

| Hướng đã thử | Mục đích thử | Kết quả rút ra |
| --- | --- | --- |
| Gán nhãn bằng LLM | Dùng AI để đánh dấu nhanh các thông tin quan trọng trong hàng chục nghìn CV/JD, thay vì phải tự gán nhãn thủ công. | Tạo dữ liệu nhanh nhưng có nhiễu, nên phải làm sạch nhiều vòng. |
| Train mô hình NER | Dạy mô hình tự nhận ra kỹ năng, công nghệ, vai trò, bằng cấp, chứng chỉ trong CV/JD. | Mô hình cleaned_v6 tìm được phần lớn entity quan trọng, nhưng vẫn cần cải thiện precision. |
| Làm sạch nhãn nhiều phiên bản | Loại bỏ các nhãn quá chung chung hoặc bị gán sai, ví dụ các cụm methodology bị nhầm là skill. | Làm sạch dữ liệu giúp mô hình tốt lên rõ hơn so với chỉ tăng số epoch. |
| Hard negative sampling | Cho mô hình học thêm các ví dụ “gần giống entity nhưng không phải entity”. | Giúp mô hình bớt nhận nhầm các cụm không nên trích xuất. |
| Fixed split | Giữ nguyên tập test qua các lần train. | Kết quả các phiên bản có thể so sánh công bằng hơn. |
| O*NET lexical mapping | Map entity sang O*NET bằng cách so khớp từ khóa. | Nhanh nhưng bỏ sót nhiều cách diễn đạt khác nhau. |
| O*NET semantic mapping | Map entity sang O*NET bằng ý nghĩa, không chỉ bằng chữ giống nhau. | Tăng mapped rate lên 74.15 phần trăm, tốt hơn lexical mapping. |
| Linear hybrid scoring | Kết hợp nhiều điểm: semantic, O*NET và hard constraints bằng công thức cố định. | Rất tốt trên synthetic CV-JD evaluation, nhưng không phải lúc nào cũng tối ưu trên dataset khác. |
| CrossEncoder | Thử mô hình đọc cả CV và JD cùng lúc để hiểu quan hệ sâu hơn. | Nếu không cân chỉnh lại điểm, kết quả có thể giảm. Đây là bài học về calibration. |
| XGBoost ranker | Để mô hình tự học cách kết hợp các feature thay vì tự đặt trọng số. | Là hướng tốt, nhất là khi dataset có đặc trưng khác nhau. |
| Vanetik benchmark | Kiểm tra hệ thống trên dataset vacancy-resume matching bên ngoài. | Kết quả chưa vượt rõ baseline, cho thấy cần cẩn thận với cách setup và domain shift. |
| Kaggle resume data | Kiểm tra thêm trên dữ liệu resume ranking. | Hữu ích để stress-test, nhưng nhãn chưa chắc đủ tin cậy để dùng làm kết luận chính. |
| CareerBuilder benchmark | Kiểm tra ranking trên dữ liệu apply job thực tế. | Proposed tuned nhỉnh hơn BM25, nhưng cần nói rõ đây là implicit feedback, không phải nhãn đánh giá trực tiếp của con người. |
| Web demo | Tạo giao diện nhập CV/JD để demo pipeline. | Có thể trình bày hệ thống end-to-end dễ hiểu hơn thay vì chỉ nói về code. |
