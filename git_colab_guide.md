
# Hướng dẫn quy trình làm việc kết hợp: MacBook & Google Colab qua GitHub

Tài liệu này hướng dẫn cách thiết lập và duy trì quy trình làm việc tối ưu cho các dự án **Computer Vision / Deep Learning**, trong đó **MacBook** đóng vai trò là trạm phát triển code, quản lý mã nguồn, còn **Google Colab** đóng vai trò là máy chủ đám mây xử lý tác vụ nặng (GPU).

---

## 1. Mục tiêu của quy trình
* Tận dụng giao diện, phím tắt và môi trường quen thuộc trên **VS Code** tại máy MacBook.
* Tận dụng sức mạnh phần cứng **GPU miễn phí/trả phí (T4/L4)** trên Google Colab để huấn luyện mô hình (Training).
* Đồng bộ hóa mã nguồn tự động thông qua **GitHub** mà không cần cấu hình phức tạp các đường hầm mạng (SSH tunnel hay local runtime rườm rà).

---

## 2. Các bước thiết lập ban đầu (Làm 1 lần duy nhất)

### Bước 1: Cấu hình SSH cho GitHub trên MacBook
Để không phải nhập mật khẩu mỗi lần push/pull code, hãy đảm bảo máy Mac của bạn đã kết nối SSH với GitHub:
1. Tạo khóa SSH (nếu chưa có):
   ```bash
   ssh-keygen -t ed25519 -C "email_cua_ban@example.com"

```

2. Thêm khóa vào ssh-agent và copy nội dung file `.pub` dán vào phần **SSH and GPG keys** trong cài đặt tài khoản GitHub cá nhân của bạn.
3. Kiểm tra kết nối:
```bash
ssh -T git@github.com

```



### Bước 2: Tạo Repository trên GitHub

* Tạo một repository mới trên GitHub cho dự án của bạn (Ví dụ: `deep-learning-computer-vision-co5085`).

---

## 3. Quy trình làm việc hàng ngày (Workflow)

### Bước A: Viết code và chuẩn bị trên MacBook

1. Mở dự án trên máy Mac bằng **VS Code**.
2. Viết code xử lý dữ liệu, định nghĩa kiến trúc mô hình, hoặc chuẩn bị các file Jupyter Notebook (`.ipynb`).
3. Commit và push code lên GitHub:
```bash
git add .
git commit -m "Chỉnh sửa code trên local Mac"
git push origin main

```



---

### Bước B: Huấn luyện mô hình (Training) trên Google Colab qua GitHub

Khi cần train mô hình với GPU, bạn không cần tải file thủ công mà thực hiện trực tiếp trên trang web Google Colab:

1. **Mở Google Colab:** Truy cập [Google Colab](https://colab.research.google.com/).
2. **Kéo code từ GitHub vào Colab:**
* Chọn tab **GitHub** trên cửa sổ mở file của Colab.
* Dán đường dẫn repository của bạn vào và chọn file notebook cần chạy.


3. **Bật GPU trên Colab:**
* Trên thanh menu, chọn **Runtime** > **Change runtime type**.
* Ở mục *Hardware accelerator*, chọn **T4 GPU** (hoặc GPU tương ứng) và bấm **Save**.


4. **Cấu hình Git trực tiếp trên Colab (nếu cần push ngược lại):**
```python
!git config --global user.name "Tên_GitHub_của_bạn"
!git config --global user.email "email_cua_bạn@example.com"

```


5. **Thực hiện Training:** Chạy các ô code để train mô hình, đánh giá kết quả trên GPU.

---

### Bước C: Đồng bộ hóa code sau khi sửa/train xong trên Colab

Nếu bạn có các tinh chỉnh hoặc lưu lại kết quả/file checkpoint mới ngay trên Colab và muốn đẩy về máy Mac:

1. Chạy lệnh commit & push ngay trong cell của Colab:
```python
# Di chuyển vào thư mục dự án (nếu clone trực tiếp)
%cd deep-learning-computer-vision-co5085

!git add .
!git commit -m "Cập nhật kết quả training từ Google Colab"
!git push origin main

```


*(Lưu ý: Colab có thể yêu cầu xác thực tài khoản GitHub khi push).*
2. **Cập nhật về MacBook:**
Khi quay lại máy Mac, bạn chỉ cần mở Terminal, vào thư mục dự án và kéo code mới nhất về:
```bash
git pull origin main

```



---

## 4. Ưu điểm của phương pháp này

* **An toàn:** Không lo mất code, mọi thay đổi đều được lưu vết trên lịch sử Git/GitHub.
* **Ổn định:** Tránh được các lỗi kết nối phức tạp (như lỗi timeout, lỗi đứt kết nối SSH tunnel giữa VS Code và Colab).
* **Linh hoạt:** Bạn có thể làm việc ở bất cứ đâu, chỉ cần máy tính có mạng để code và trình duyệt web để chạy Colab.
