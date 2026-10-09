# E1 — Phân tích kết quả thực nghiệm

## 1. So sánh định lượng các mô hình

Trong bài thực nghiệm E1, em triển khai và so sánh ba mô hình phân loại ảnh: bộ phân loại Softmax, mạng nơ-ron truyền thẳng nhiều lớp (Multilayer Perceptron — MLP) và mạng nơ-ron tích chập (Convolutional Neural Network — CNN).

Em đánh giá các mô hình dựa trên độ chính xác của tập kiểm thử, độ chính xác validation cao nhất, số lượng tham số và thời gian huấn luyện. Bộ dữ liệu em sử dụng là Fashion-MNIST, gồm 10 lớp quần áo và phụ kiện.

### 1.1. Kết quả thực nghiệm

| Mô hình | Test accuracy | Best validation accuracy | Số tham số | Thời gian huấn luyện (giây) |
|---|---:|---:|---:|---:|
| Softmax | 84,19% | 85,25% | 7.850 | 118,44 |
| MLP | 87,92% | 88,67% | 109.386 | 118,44 |
| CNN | 91,05% | 92,00% | 50.186 | 122,53 |

*Lưu ý: Em tổng hợp các số liệu trong bảng từ kết quả thực nghiệm hiện có; thời gian huấn luyện được làm tròn đến hai chữ số thập phân.*

### 1.2. Nhận xét tổng quan

Từ kết quả trong bảng, em thấy CNN đạt độ chính xác trên tập kiểm thử cao nhất (91,05%), tiếp theo là MLP (87,92%) và Softmax (84,19%).

So với Softmax, MLP tăng 3,73 điểm phần trăm về độ chính xác kiểm thử. CNN cao hơn Softmax 6,86 điểm phần trăm và cao hơn MLP 3,13 điểm phần trăm.

Xét về số lượng tham số, Softmax có mô hình nhỏ nhất với 7.850 tham số. MLP có 109.386 tham số, còn CNN có 50.186 tham số. Như vậy, trong cấu hình em thử nghiệm, CNN đạt độ chính xác cao hơn MLP dù sử dụng ít tham số hơn.

Về thời gian huấn luyện, Softmax và MLP lần lượt mất khoảng 118,44 giây, trong khi CNN mất khoảng 122,53 giây. Ở lần chạy này, CNN cần nhiều thời gian hơn một chút.

## 2. Phân tích đường cong huấn luyện

Em sử dụng đường cong huấn luyện để theo dõi sự thay đổi của loss và accuracy trên tập training và validation qua 10 epoch. Từ đó, em xem xét xu hướng hội tụ và những dấu hiệu có thể liên quan đến overfitting.

### 2.1. Mô hình Softmax

![Đường cong huấn luyện của Softmax](figures/softmax_learning_curves.png)

*Hình 1. Training loss, validation loss và accuracy của mô hình Softmax.*

Ở mô hình Softmax, training loss giảm từ khoảng 0,61 xuống 0,40, còn validation loss giảm từ khoảng 0,50 xuống 0,43. Training accuracy tăng lên khoảng 86% và validation accuracy đạt khoảng 85%.

Quan sát biểu đồ, em thấy hai đường accuracy khá gần nhau và validation loss nhìn chung giảm qua các epoch. Kết quả này gợi ý mô hình đã học được một số đặc trưng hữu ích; trong phạm vi 10 epoch được khảo sát, em chưa thấy dấu hiệu overfitting rõ rệt.

Tuy vậy, ở giai đoạn sau, validation loss giảm chậm hơn training loss. Theo em, điều này cho thấy mức cải thiện trên tập validation bắt đầu chững lại.

### 2.2. Mô hình MLP

![Đường cong huấn luyện của MLP](figures/mlp_learning_curves.png)

*Hình 2. Training loss, validation loss và accuracy của mô hình MLP.*

Với MLP, training loss giảm liên tục từ khoảng 0,56 xuống 0,25 và training accuracy tăng lên khoảng 91%. Ngược lại, dù validation loss giảm trong phần lớn quá trình, chỉ số này dao động ở các epoch cuối. Validation accuracy cũng có lúc giảm nhẹ rồi phục hồi.

Em nhận thấy khoảng cách giữa training accuracy và validation accuracy tăng dần về cuối quá trình. Đây có thể là dấu hiệu MLP bắt đầu overfit: kết quả trên tập training tiếp tục tốt lên, nhưng kết quả trên tập validation không cải thiện tương ứng.

Mặc dù vậy, validation accuracy của MLP vẫn cao hơn Softmax. Với kết quả của lần thực nghiệm này, em nhận thấy MLP phân loại tốt hơn Softmax.

### 2.3. Mô hình CNN

![Đường cong huấn luyện của CNN](figures/cnn_learning_curves.png)

*Hình 3. Training loss, validation loss và accuracy của mô hình CNN.*

Ở CNN, training loss giảm từ khoảng 0,50 xuống 0,16, còn training accuracy tăng lên khoảng 94%. Validation accuracy đạt cao nhất khoảng 92% ở epoch 7, giảm ở epoch 8 rồi phục hồi ở epoch 9.

Validation loss nhìn chung giảm nhưng vẫn dao động ở giai đoạn cuối. So với Softmax, em cũng thấy khoảng cách giữa training accuracy và validation accuracy của CNN rõ hơn.

Những quan sát này khiến em cho rằng CNN có thể bắt đầu overfit nhẹ ở các epoch cuối. Dù vậy, trong ba mô hình, CNN vẫn đạt validation accuracy cao nhất.

### 2.4. So sánh đường cong huấn luyện

Qua biểu đồ, em thấy training loss của cả ba mô hình đều giảm, còn training accuracy tăng qua các epoch. Tuy nhiên, mức cải thiện trên tập validation không giống nhau.

Softmax có khoảng cách training–validation tương đối nhỏ nhưng đạt độ chính xác thấp nhất. MLP cải thiện độ chính xác, song chênh lệch giữa hai tập dữ liệu rõ hơn. CNN đạt validation accuracy cao nhất, đồng thời có khoảng cách training–validation đáng chú ý ở giai đoạn cuối.

Nhìn chung, từ kết quả này em đánh giá CNN học được các đặc trưng có ích cho bài toán phân loại ảnh. Tuy nhiên, để kết luận chắc chắn hơn về mức độ overfitting, em cần xem xét thêm số liệu từng epoch và thực hiện các lần thử nghiệm bổ sung.

## 3. Phân tích ma trận nhầm lẫn

Em sử dụng ma trận nhầm lẫn (confusion matrix) để xem xét kết quả phân loại của từng lớp. Các phần tử trên đường chéo chính thể hiện số mẫu được phân loại đúng; các phần tử ngoài đường chéo cho biết những mẫu bị phân loại nhầm.

### 3.1. Ma trận nhầm lẫn của Softmax

![Ma trận nhầm lẫn của Softmax](figures/softmax_confusion_matrix.png)

*Hình 4. Ma trận nhầm lẫn của mô hình Softmax.*

Ở ma trận của Softmax, em thấy đường chéo chính có màu đậm ở nhiều lớp, cho thấy mô hình phân loại đúng phần lớn mẫu. Tuy nhiên, mô hình vẫn nhầm lẫn một số lớp trang phục có hình dạng tương tự nhau.

Đặc biệt, các ô ngoài đường chéo cho thấy có nhầm lẫn giữa T-shirt/top, shirt, pullover và coat. Theo em, một nguyên nhân có thể là các lớp này có hình dáng tổng thể gần giống nhau khi được biểu diễn bằng ảnh grayscale độ phân giải thấp.

### 3.2. Ma trận nhầm lẫn của MLP

![Ma trận nhầm lẫn của MLP](figures/mlp_confusion_matrix.png)

*Hình 5. Ma trận nhầm lẫn của mô hình MLP.*

Ma trận của MLP cũng có đường chéo chính nổi bật, cho thấy mô hình phân loại tương đối tốt phần lớn các lớp. Dù vậy, em vẫn quan sát thấy nhầm lẫn giữa T-shirt/top, shirt, pullover và coat.

So với Softmax, MLP đạt độ chính xác tổng thể cao hơn. Kết quả này có thể liên quan đến khả năng học quan hệ phi tuyến thông qua các lớp ẩn. Tuy nhiên, em không thể định lượng chính xác mức giảm lỗi của từng cặp lớp chỉ bằng cách quan sát hình; em cần đối chiếu các giá trị cụ thể trong ma trận.

### 3.3. Ma trận nhầm lẫn của CNN

![Ma trận nhầm lẫn của CNN](figures/cnn_confusion_matrix.png)

*Hình 6. Ma trận nhầm lẫn của mô hình CNN.*

Ở ma trận của CNN, em cũng thấy đường chéo chính nổi bật. Mô hình vẫn nhầm lẫn một số lớp trang phục có hình dạng tương tự, đặc biệt là các loại áo.

Khi xem xét cùng test accuracy 91,05%, em nhận thấy CNN có kết quả phân loại tổng thể tốt nhất trong ba mô hình. Những lớp có hình dạng đặc trưng hơn dường như dễ phân loại hơn, còn các lớp có đường viền tương tự vẫn là thách thức.

### 3.4. So sánh ma trận nhầm lẫn

Qua ba ma trận, em thấy cả ba mô hình đều phân loại đúng phần lớn mẫu nhưng vẫn gặp khó khăn với một số lớp có hình dạng tương tự. Điều này cho thấy accuracy tổng thể chưa phản ánh đầy đủ hiệu quả của mô hình trên từng lớp.

CNN đạt accuracy cao nhất. Kết quả này phù hợp với nhận định của em rằng trong thực nghiệm này, mô hình khai thác cấu trúc không gian của ảnh hiệu quả hơn. Tuy nhiên, để xác định chính xác lớp nào được cải thiện nhiều nhất, em cần so sánh các giá trị trong từng ma trận thay vì chỉ dựa vào màu sắc.


## 4. Phân tích các trường hợp dự đoán sai

Để tìm hiểu thêm về hạn chế của từng mô hình, em quan sát các ảnh mà Softmax, MLP và CNN phân loại sai trên Fashion-MNIST. Mỗi ảnh được hiển thị cùng nhãn thực tế (T — True label) và nhãn dự đoán (P — Predicted label).

### 4.1. Một số dạng lỗi em quan sát được

Qua các ví dụ được hiển thị, em nhận thấy một số nhóm lỗi thường gặp.

**Nhầm lẫn giữa các loại áo**

Trong các ảnh em xem, cả ba mô hình có lúc nhầm T-shirt/top, shirt, pullover và coat. Chẳng hạn, một số ảnh có nhãn shirt được dự đoán thành T-shirt/top, pullover hoặc coat; một số ảnh coat lại bị dự đoán thành pullover hoặc shirt.

Theo em, nguyên nhân có thể là hình dáng tổng thể và đường viền của các lớp này khá giống nhau trong ảnh grayscale độ phân giải thấp. Những chi tiết như cổ áo, tay áo, hàng cúc và cấu trúc trang phục có thể chưa đủ rõ để mô hình phân biệt chính xác.

**Nhầm lẫn giữa các loại giày dép**

Em cũng thấy một số ảnh sneaker hoặc ankle boot bị dự đoán thành sandal và ngược lại. Cụ thể, trong các ví dụ này, có ảnh sneaker bị nhận diện thành sandal và ảnh ankle boot bị nhận diện thành sneaker.

Từ những trường hợp đó, em nhận thấy đặc trưng hình dáng tổng thể có thể chưa đủ để phân biệt một số loại giày dép trong ảnh độ phân giải thấp.

**Nhầm lẫn giữa dress và các loại áo hoặc coat**

Ngoài ra, một số ảnh dress bị dự đoán thành T-shirt/top, shirt hoặc coat. Em cho rằng những lỗi này có thể liên quan đến hình dáng thân áo, độ dài trang phục và đường viền tương tự nhau trong ảnh.

### 4.2. So sánh lỗi giữa ba mô hình

**Softmax**

![softmax_misclassified](figures/softmax_misclassified.png)

*Hình 7: Các hình ảnh dự đoán sai của Softmax*

Trong các ví dụ của Softmax, em thấy mô hình nhầm lẫn giữa một số lớp trang phục, đặc biệt là shirt, T-shirt/top, pullover, coat và dress. Mô hình cũng nhận diện nhầm một số mẫu sneaker và ankle boot thành sandal.

Theo em, các lỗi này phần nào phù hợp với hạn chế của bộ phân loại tuyến tính: ảnh đầu vào được làm phẳng thành vector nên mô hình không khai thác trực tiếp cấu trúc không gian cục bộ như mạng tích chập.

**MLP**

![mlp_misclassified](figures/mlp_misclassified.png)

*Hình 8: Các hình ảnh dự đoán sai của MLP*

MLP vẫn mắc một số lỗi tương tự, nhất là giữa coat, pullover, shirt và dress. Em cũng quan sát thấy một số mẫu giày dép bị nhầm giữa sneaker và sandal.

Mặc dù MLP có thể học các quan hệ phi tuyến thông qua lớp ẩn, các ví dụ em xem cho thấy mô hình vẫn khó phân biệt những lớp có hình dáng tương tự. Điều đó cho thấy khả năng biểu diễn lớn hơn không đồng nghĩa với việc mọi lỗi phân loại đều được khắc phục.

**CNN**

![cnn_misclassified](figures/cnn_misclassified.png)

*Hình 9: Các hình ảnh dự đoán sai của CNN*

Trong các ví dụ dự đoán sai của CNN, em vẫn thấy nhầm lẫn giữa shirt, T-shirt/top, pullover, coat và dress. Một số mẫu sneaker hoặc ankle boot cũng bị phân loại thành sandal hoặc sneaker.

Theo em, CNN cải thiện hiệu quả phân loại tổng thể nhưng vẫn có giới hạn với những mẫu khó phân biệt. Khả năng học đặc trưng không gian có ích cho bài toán này, nhưng không bảo đảm mọi ảnh có hình dạng tương tự đều được phân loại chính xác.

### 4.3. Liên hệ với kết quả định lượng

Em ghi nhận test accuracy của Softmax, MLP và CNN lần lượt là 84,19%, 87,92% và 91,05%.

Theo em, các ảnh dự đoán sai giúp minh họa thêm cho những số liệu trên. Những ví dụ này cho thấy lỗi có xu hướng tập trung ở các lớp có hình dạng tương đồng, thay vì chỉ xuất hiện ngẫu nhiên.

Tuy nhiên, các hình chỉ minh họa một số trường hợp sai chứ không thể hiện toàn bộ lỗi của mỗi mô hình. Vì vậy, em không thể dựa vào số ảnh trong hình để kết luận mô hình nào có tỷ lệ lỗi thấp hơn. Để so sánh chính xác, em cần tính tổng số mẫu dự đoán sai trên cùng tập test và đối chiếu với các giá trị trong confusion matrix.

### 4.4. Kết luận

Qua các trường hợp dự đoán sai, em nhận thấy cả ba mô hình đều gặp khó khăn với những lớp trang phục có hình dáng tương tự, đặc biệt là shirt, T-shirt/top, pullover và coat. Em cũng quan sát thấy một số nhầm lẫn giữa các loại giày dép.

Theo em, Softmax bị hạn chế vì mô hình tuyến tính không khai thác trực tiếp cấu trúc không gian của ảnh. MLP có khả năng biểu diễn phi tuyến tốt hơn nhưng vẫn nhầm lẫn những lớp tương tự. CNN đạt độ chính xác tổng thể cao nhất trong lần thực nghiệm này, song vẫn mắc lỗi với một số mẫu khó phân biệt.

Từ những quan sát này, em cho rằng cần kết hợp accuracy, số lượng tham số, đường cong huấn luyện, confusion matrix và phân tích ảnh dự đoán sai khi đánh giá mô hình, thay vì chỉ dựa vào một chỉ số.

## 5. Thảo luận và đánh đổi giữa các mô hình

### 5.1. Độ chính xác và độ phức tạp mô hình

Softmax có số lượng tham số nhỏ nhất, nên theo em mô hình phù hợp khi ưu tiên sự đơn giản. Tuy nhiên, test accuracy của mô hình cũng thấp nhất, ở mức 84,19%.

MLP đạt test accuracy 87,92%, nhưng có số lượng tham số lớn nhất (109.386). Theo em, các lớp ẩn giúp mô hình học quan hệ phi tuyến; tuy vậy, MLP không trực tiếp khai thác cấu trúc không gian cục bộ của ảnh như CNN.

CNN đạt test accuracy 91,05% với 50.186 tham số. Trong cấu hình em thử nghiệm, mô hình vừa có độ chính xác cao nhất vừa sử dụng ít tham số hơn MLP. Theo em, các lớp tích chập có thể góp phần cải thiện kết quả vì cho phép mô hình học đặc trưng không gian cục bộ của ảnh.

### 5.2. Khả năng tổng quát hóa

Qua các đường cong huấn luyện, em thấy cả ba mô hình đều cải thiện trên tập training. Tuy nhiên, validation loss và validation accuracy dao động ở các epoch cuối, đặc biệt với MLP và CNN.

Kết quả này nhắc em rằng mô hình cải thiện trên tập training không có nghĩa là kết quả trên tập validation cũng tăng tương ứng. Vì vậy, em cần xem xét hiệu quả validation và đánh giá trên tập test độc lập, thay vì chỉ dựa vào training accuracy.

### 5.3. Chi phí huấn luyện

Trong lần thực nghiệm này, em ghi nhận thời gian huấn luyện của Softmax, MLP và CNN lần lượt là khoảng 118,44 giây, 118,44 giây và 122,53 giây.

CNN mất nhiều thời gian hơn một chút. Dù vậy, theo kết quả hiện tại, chênh lệch này không lớn so với mức cải thiện accuracy. Em lưu ý rằng thời gian trên chỉ phản ánh cấu hình và điều kiện của lần chạy này, không đại diện cho mọi phần cứng hay cách triển khai.

### 5.4. Tổng hợp

| Tiêu chí | Mô hình nổi bật | Nhận xét |
|---|---|---|
| Test accuracy | CNN | Đạt 91,05%, cao nhất trong ba mô hình |
| Ít tham số nhất | Softmax | Chỉ có 7.850 tham số |
| Ít tham số hơn MLP nhưng accuracy cao hơn | CNN | 50.186 tham số và test accuracy 91,05% |
| Thời gian huấn luyện ngắn nhất | Softmax và MLP | Khoảng 118,44 giây trong lần chạy hiện tại |
| Khả năng khai thác đặc trưng không gian | CNN | Kiến trúc tích chập phù hợp với dữ liệu ảnh |

## 6. Kết luận

Trong bài thực nghiệm E1 trên Fashion-MNIST, em đã so sánh Softmax, MLP và CNN theo độ chính xác, số lượng tham số, quá trình huấn luyện và các dạng lỗi phân loại.

Softmax có cấu trúc đơn giản nhất và ít tham số nhất, nhưng test accuracy đạt 84,19%. MLP nâng độ chính xác lên 87,92%, đổi lại có số lượng tham số lớn nhất. CNN cho kết quả tốt nhất trong lần thử nghiệm này, với test accuracy 91,05% và 50.186 tham số, ít hơn MLP. Từ các kết quả này, em nhận thấy CNN có sự cân bằng tốt hơn giữa độ chính xác và số lượng tham số trong cấu hình đã thử nghiệm.

Từ các đường cong huấn luyện, em nhận thấy cả ba mô hình đều học được đặc trưng hữu ích, nhưng MLP và CNN có chênh lệch giữa kết quả training và validation ở giai đoạn cuối. Ma trận nhầm lẫn cũng cho thấy những lớp trang phục có hình dạng tương tự là một nguồn gây lỗi đáng chú ý.

Nhìn chung, nếu ưu tiên độ chính xác phân loại và khả năng khai thác đặc trưng không gian của ảnh, em đánh giá CNN là lựa chọn nổi bật nhất trong thực nghiệm hiện tại. Tuy nhiên, Softmax vẫn có ưu thế về sự đơn giản và số lượng tham số. Theo em, khi lựa chọn mô hình cần cân nhắc đồng thời độ chính xác, độ phức tạp, thời gian huấn luyện và các trường hợp dự đoán sai.

Việc xem xét trực quan các ảnh dự đoán sai giúp em hiểu rõ hơn ưu điểm và hạn chế của từng mô hình, thay vì chỉ so sánh accuracy tổng thể.