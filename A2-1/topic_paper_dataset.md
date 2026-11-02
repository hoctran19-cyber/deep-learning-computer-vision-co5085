# A2.1 — Chủ đề, Paper và Dataset

**Môn học:** CO5085 – Deep Learning and Its Applications in Computer Vision  
**Assignment:** A2 – Self-chosen Topic + Paper Analysis  
**Chủ đề:** Automatic License Plate Recognition for Vietnamese Vehicles from Still Images  
**Phiên bản:** A2.1 – Topic + Paper + Dataset
**Học viên** Trần Thái Học - 2570426

---

## 1. Giới thiệu đề tài

### 1.1. Tên đề tài

**Automatic License Plate Recognition (ALPR) for Vietnamese Vehicles from Still Images**

### 1.2. Phát biểu bài toán

Automatic License Plate Recognition (ALPR) là bài toán phát hiện và nhận dạng biển số xe tự động từ hình ảnh.

Đề tài tập trung xây dựng một hệ thống ALPR cho phương tiện tại Việt Nam, sử dụng ảnh tĩnh làm dữ liệu đầu vào. Hệ thống cần thực hiện hai nhiệm vụ chính:

1. **License Plate Detection:** Xác định vị trí biển số trong ảnh và trích xuất vùng chứa biển số.
2. **License Plate Recognition:** Nhận dạng chuỗi ký tự trên biển số đã được phát hiện.

Bài toán cần xử lý các trường hợp biển số có bố cục một dòng hoặc hai dòng, kích thước khác nhau, góc nhìn không đồng nhất và chất lượng ảnh không ổn định.

### 1.3. Mục tiêu

- Xây dựng pipeline phát hiện và nhận dạng biển số xe Việt Nam từ ảnh tĩnh.
- Đánh giá khả năng phát hiện biển số và nhận dạng chuỗi ký tự.
- So sánh hai phương pháp nhận dạng trên cùng một detector.
- Phân tích ảnh hưởng của chất lượng ảnh và bố cục biển số đến kết quả.
- Đánh giá khả năng ứng dụng của các mô hình deep learning trong bài toán ALPR.

### 1.4. Phạm vi

**Trong phạm vi:**

- Ảnh tĩnh, không xử lý video hoặc tracking.
- Biển số xe Việt Nam.
- Biển số một dòng và hai dòng.
- Detection và recognition.

**Ngoài phạm vi:**

- Nhận dạng phương tiện hoặc phân loại loại xe.
- Theo dõi phương tiện qua nhiều khung hình.
- Xây dựng hệ thống triển khai production hoặc xử lý video thời gian thực.

---

## 2. Paper được lựa chọn để phân tích

### 2.1. Thông tin paper

**Title:** Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline

**Authors:** Zhenbo Xu, Wei Yang, Ajin Meng, Nanxue Lu, Huan Huang, Changchun Ying, Liusheng Huang.

**Publication:** European Conference on Computer Vision (ECCV), 2018.

**DOI:** https://doi.org/10.1007/978-3-030-01261-8_16

**Official paper:**  
https://openaccess.thecvf.com/content_ECCV_2018/html/Zhenbo_Xu_Towards_End-to-End_License_ECCV_2018_paper.html

**Official implementation and dataset:**  
https://github.com/detectRecog/CCPD

### 2.2. Lý do lựa chọn

Paper được lựa chọn vì:

- Được công bố tại ECCV, một trong những hội nghị hàng đầu về Computer Vision.
- Giải quyết trực tiếp bài toán License Plate Detection and Recognition.
- Đề xuất một mô hình có khả năng dự đoán vị trí biển số và nhận dạng chuỗi ký tự trong cùng hệ thống.
- Giới thiệu CCPD, một dataset lớn dành cho bài toán phát hiện và nhận dạng biển số.
- Có paper, dataset và repository chính thức để tham khảo.

Paper được sử dụng làm tài liệu nghiên cứu về phương pháp ALPR và cách thiết kế benchmark. Nhóm không đặt mục tiêu tái tạo nguyên bản mô hình hoặc sử dụng kết quả trên CCPD làm kết quả thực nghiệm của VNLP.

### 2.3. Tóm tắt phương pháp

Paper giới thiệu CCPD và một baseline end-to-end cho bài toán License Plate Detection and Recognition.

Ý tưởng chính là kết hợp phát hiện vị trí biển số và nhận dạng nội dung biển số trong một hệ thống học sâu. Thay vì chỉ xác định bounding box, mô hình còn dự đoán chuỗi ký tự tương ứng.

Các nội dung chính cần phân tích:

- **Input:** Ảnh phương tiện có chứa biển số.
- **Detection:** Xác định vùng biển số và vị trí của nó trong ảnh.
- **Recognition:** Dự đoán chuỗi ký tự của biển số.
- **Output:** Vị trí biển số và nội dung ký tự được nhận dạng.

Điểm đáng chú ý của nghiên cứu là thiết kế một hệ thống nhận dạng biển số end-to-end và xây dựng một dataset có quy mô lớn, có nhiều điều kiện hình ảnh khác nhau.

### 2.4. Thí nghiệm của paper

Paper sử dụng CCPD với hơn 250.000 ảnh phương tiện. Dataset có các trường hợp khác nhau về góc nhìn, độ nghiêng, độ mờ và điều kiện hình ảnh.

Tác giả đánh giá:

- Khả năng phát hiện biển số.
- Khả năng nhận dạng chính xác toàn bộ chuỗi ký tự.
- Tốc độ xử lý của hệ thống.

Paper báo cáo mô hình đạt khoảng 98,5% accuracy và tốc độ trên 61 FPS trong thiết lập thực nghiệm được mô tả.

**Lưu ý:** Đây là kết quả của tác giả trên CCPD, không phải kết quả có thể kỳ vọng trực tiếp trên VNLP.

### 2.5. Hạn chế và liên hệ với đề tài

Một số hạn chế khi tham khảo và áp dụng nghiên cứu:

1. CCPD chủ yếu chứa biển số Trung Quốc, có định dạng và bộ ký tự khác biển số Việt Nam.
2. Kết quả trên CCPD không đảm bảo khả năng tổng quát hóa sang VNLP.
3. Nhóm cần xây dựng bộ ký tự và quy tắc xử lý chuỗi phù hợp với dữ liệu Việt Nam.
4. Mô hình end-to-end trong paper không nhất thiết là lựa chọn phù hợp nhất với tài nguyên và thời gian của assignment.

**Cách áp dụng vào đề tài:**

Nhóm kế thừa cách tiếp cận tách biệt hai nhiệm vụ detection và recognition để xây dựng pipeline có thể đánh giá từng thành phần. Thay vì tái tạo nguyên bản kiến trúc trong paper, nhóm sử dụng YOLOv8n cho detection và so sánh hai mô hình recognition là CRNN-CTC và LPRNet.

---

## 3. Dataset

### 3.1. Dataset được lựa chọn

**Tên:** Vietnamese License Plate Dataset (VNLP)

**Official repository:**  
https://github.com/fict-labs/VNLP

**Download:** Liên kết tải dữ liệu được cung cấp trong repository chính thức.

VNLP là dataset dành cho bài toán phát hiện và nhận dạng biển số Việt Nam.

### 3.2. Thống kê dataset

Theo mô tả trong repository:

| Thuộc tính | Thông tin |
|---|---:|
| Tổng số ảnh | Khoảng 37.300 |
| Biển số một dòng | 19.086 |
| Biển số hai dòng | 18.211 |
| Số ký tự mỗi biển | 6–9 |
| Nhãn vị trí biển số | Bounding box |
| Nhãn nhận dạng | Chuỗi ký tự |

Dataset bao gồm hai nhóm bố cục biển số:

- **One-row license plate:** Biển số có ký tự trên một dòng.
- **Two-row license plate:** Biển số có ký tự trên hai dòng.

Kích thước ảnh biển số có sự khác biệt đáng kể, do đó cần chuẩn hóa kích thước đầu vào trong quá trình tiền xử lý.

### 3.3. Lý do lựa chọn

- Dữ liệu phù hợp trực tiếp với bài toán biển số Việt Nam.
- Có nhãn bounding box phục vụ detection.
- Có nhãn chuỗi ký tự phục vụ recognition.
- Có cả biển số một dòng và hai dòng.
- Quy mô dữ liệu phù hợp để thực hiện huấn luyện và đánh giá các mô hình deep learning.

### 3.4. Phương pháp chia dữ liệu

Nhóm dự kiến chia dữ liệu theo tỷ lệ:

| Tập dữ liệu | Tỷ lệ | Mục đích |
|---|---:|---|
| Training | 70% | Huấn luyện mô hình |
| Validation | 15% | Theo dõi quá trình huấn luyện và chọn mô hình |
| Testing | 15% | Đánh giá cuối cùng |

Việc chia dữ liệu sẽ được thực hiện trước khi tạo các ảnh crop phục vụ recognition.

Các nguyên tắc:

- Cố định random seed để bảo đảm khả năng tái lập.
- Duy trì tỷ lệ tương đối giữa biển số một dòng và hai dòng trong các tập.
- Kiểm tra ảnh trùng lặp hoặc gần trùng lặp để hạn chế data leakage.
- Không sử dụng tập test trong quá trình lựa chọn hyperparameters.

**Lưu ý:** Tỷ lệ trên là protocol dự kiến của bản thân, không phải split mặc định do tác giả VNLP cung cấp. Số lượng ảnh thực tế ở từng tập sẽ được xác định sau khi kiểm tra dữ liệu.

### 3.5. Tiền xử lý và augmentation dự kiến

**Detection:**

- Đọc ảnh và bounding box.
- Chuyển annotation về định dạng YOLO.
- Resize ảnh theo kích thước đầu vào của detector.
- Chuẩn hóa pixel.
- Augmentation phù hợp: brightness/contrast, blur, noise và geometric transformation có kiểm soát.

**Recognition:**

- Crop vùng biển số theo ground-truth bounding box trong giai đoạn chuẩn bị dữ liệu huấn luyện.
- Resize ảnh theo yêu cầu của từng mô hình.
- Chuẩn hóa pixel.
- Chuyển chuỗi ký tự thành sequence labels.
- Xây dựng character vocabulary từ nhãn thực tế trong VNLP.

Với biển số hai dòng, cần kiểm tra cách sắp xếp ký tự trong ground truth để bảo đảm thứ tự chuỗi được giữ nhất quán.

---

## 4. Evaluation Metrics

Nhóm sử dụng các metric riêng cho detection, recognition và toàn bộ pipeline.

### 4.1. Detection Metrics

| Metric | Ý nghĩa |
|---|---|
| Precision | Tỷ lệ dự đoán detection đúng trên tổng số detection được dự đoán |
| Recall | Tỷ lệ biển số thực tế được phát hiện |
| mAP@0.5 | Mean Average Precision tại IoU threshold 0.5 |
| mAP@0.5:0.95 | mAP trung bình trên các IoU threshold từ 0.5 đến 0.95 |

mAP@0.5 được sử dụng làm metric chính để đánh giá khả năng phát hiện biển số. mAP@0.5:0.95 được bổ sung để đánh giá độ chính xác định vị nghiêm ngặt hơn.

### 4.2. Recognition Metrics

| Metric | Ý nghĩa |
|---|---|
| Character Accuracy | Tỷ lệ ký tự được nhận dạng đúng |
| Exact Plate Accuracy | Tỷ lệ biển số có toàn bộ chuỗi ký tự được nhận dạng chính xác |
| CER | Character Error Rate, tỷ lệ lỗi ở cấp độ ký tự |

CER được tính theo công thức:

\[
CER = \frac{S + D + I}{N}
\]

Trong đó:

- \(S\): Số ký tự bị thay thế.
- \(D\): Số ký tự bị xóa.
- \(I\): Số ký tự bị chèn thêm.
- \(N\): Tổng số ký tự trong ground truth.

Exact Plate Accuracy được tính bằng tỷ lệ dự đoán có chuỗi ký tự trùng khớp hoàn toàn với ground truth.

### 4.3. End-to-End Metrics

**End-to-End Exact Plate Recognition Rate**

Đánh giá tỷ lệ ảnh mà hệ thống vừa phát hiện đúng biển số, vừa nhận dạng chính xác toàn bộ chuỗi ký tự.

Một kết quả chỉ được tính đúng khi:

- Biển số được phát hiện đúng theo tiêu chí IoU đã xác định.
- Chuỗi ký tự dự đoán trùng khớp hoàn toàn với ground truth.

### 4.4. Computational Metrics

Nếu tài nguyên cho phép, nhóm sẽ báo cáo thêm:

- FPS: số ảnh xử lý trong một giây.
- Number of Parameters: số lượng tham số của mô hình.
- Model Size: kích thước trọng số mô hình.

Các chỉ số này giúp đánh giá sự cân bằng giữa độ chính xác và chi phí tính toán.

---

## 5. Kế hoạch Pipeline

### 5.1. Tổng quan kiến trúc

Pipeline dự kiến:

```text
Input Image
     |
     v
YOLOv8n License Plate Detector
     |
     v
Bounding Box Prediction
     |
     v
License Plate Cropping
     |
     +------------------------+
     |                        |
     v                        v
CRNN + BiLSTM + CTC        LPRNet
     |                        |
     v                        v
Predicted Text            Predicted Text
     |                        |
     +------------+-----------+
                  |
                  v
          Evaluation & Comparison
```

### 5.2. Detection Module

**Model:** YOLOv8n

Vai trò:

- Phát hiện vị trí biển số trong ảnh đầu vào.
- Dự đoán bounding box.
- Trích xuất vùng ảnh chứa biển số.

Lý do lựa chọn:

- Kiến trúc detection một giai đoạn, phù hợp với yêu cầu xây dựng pipeline.
- Có implementation và tài liệu công khai.
- Có thể fine-tune trên dataset VNLP.
- Có thể giữ cố định detector để đánh giá công bằng hai mô hình recognition.

### 5.3. Recognition Model 1 — CRNN + BiLSTM + CTC

CRNN kết hợp CNN và recurrent neural network để nhận dạng chuỗi ký tự từ ảnh.

Các thành phần:

- CNN: Trích xuất đặc trưng hình ảnh.
- BiLSTM: Học quan hệ tuần tự theo hai chiều.
- CTC Loss: Huấn luyện nhận dạng chuỗi mà không cần căn chỉnh từng ký tự với từng vị trí ảnh.

Đây là phương pháp baseline cho bài toán nhận dạng chuỗi ký tự.

### 5.4. Recognition Model 2 — LPRNet

LPRNet là kiến trúc nhận dạng biển số được thiết kế chuyên biệt cho bài toán license plate recognition.

Mô hình sử dụng các lớp convolution để trích xuất đặc trưng và dự đoán chuỗi ký tự bằng cơ chế nhận dạng sequence.

Lý do lựa chọn:

- Được thiết kế trực tiếp cho bài toán nhận dạng biển số.
- Có thể dùng để so sánh với baseline CRNN.
- Cho phép đánh giá sự khác biệt giữa hai kiến trúc recognition.

### 5.5. Kế hoạch thực nghiệm

Nhóm dự kiến thực hiện hai cấu hình:

| Experiment | Detection | Recognition |
|---|---|---|
| E1 — Baseline | YOLOv8n | CRNN + BiLSTM + CTC |
| E2 — Comparison | YOLOv8n | LPRNet |

Để bảo đảm tính công bằng:

- Sử dụng cùng dataset và data split.
- Sử dụng cùng detector YOLOv8n.
- Sử dụng cùng ground-truth plate crops khi đánh giá riêng recognition.
- Sử dụng cùng detection predictions khi đánh giá end-to-end.
- Cố định các điều kiện đánh giá và character vocabulary.
- Báo cáo rõ cấu hình huấn luyện của từng mô hình.

Kết quả sẽ được phân tích ở ba mức: detection, recognition và end-to-end.

---

## 6. Kế hoạch thực hiện

| Giai đoạn | Công việc |
|---|---|
| Phase 1 | Kiểm tra dataset, annotation và phân tích dữ liệu |
| Phase 2 | Chuẩn bị train/validation/test split và DataLoader |
| Phase 3 | Huấn luyện hoặc fine-tune YOLOv8n |
| Phase 4 | Huấn luyện CRNN và LPRNet |
| Phase 5 | Đánh giá detection, recognition và end-to-end |
| Phase 6 | So sánh, phân tích kết quả và viết báo cáo |

A2.1 tập trung xác định chủ đề, paper, dataset, metrics và kế hoạch thực nghiệm. Kết quả huấn luyện và so sánh đầy đủ sẽ được trình bày ở A2.2.

---

## 7. Kết luận

Đề tài lựa chọn bài toán Automatic License Plate Recognition trên ảnh tĩnh, tập trung vào biển số xe Việt Nam.

Paper ECCV 2018 được sử dụng làm tài liệu nghiên cứu nền tảng về ALPR end-to-end và benchmark CCPD. Dataset VNLP được lựa chọn làm dữ liệu thực nghiệm chính vì có nhãn bounding box và chuỗi ký tự phù hợp với biển số Việt Nam.

Pipeline dự kiến sử dụng YOLOv8n cho detection, kết hợp hai phương pháp CRNN-CTC và LPRNet cho recognition. Việc so sánh hai cấu hình sẽ giúp đánh giá khả năng nhận dạng chuỗi ký tự và hiệu quả tổng thể của hệ thống.

---

## 8. References

1. Xu, Z., Yang, W., Meng, A., Lu, N., Huang, H., Ying, C., & Huang, L. (2018). *Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline*. European Conference on Computer Vision (ECCV). https://doi.org/10.1007/978-3-030-01261-8_16

2. VNLP Dataset. *Vietnamese License Plate Dataset*. Official GitHub repository. https://github.com/fict-labs/VNLP

3. CCPD Dataset. *A Diverse and Well-Annotated Dataset for License Plate Detection and Recognition*. Official GitHub repository. https://github.com/detectRecog/CCPD

4. Ultralytics. *YOLO Documentation*. https://docs.ultralytics.com/

5. Shi, B., Bai, X., & Yao, C. (2017). *An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition*. IEEE Transactions on Pattern Analysis and Machine Intelligence. https://doi.org/10.1109/TPAMI.2016.2646371
