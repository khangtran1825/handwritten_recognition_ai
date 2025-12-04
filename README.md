# handwritten_recognition_ai (Hệ thống Nhận dạng Chữ viết tay)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

## 📝 Mô tả dự án

Dự án này là một ứng dụng Machine Learning có khả năng nhận dạng chữ số hoặc ký tự viết tay từ hình ảnh đầu vào. Hệ thống sử dụng mạng nơ-ron tích chập (**CNN**) được huấn luyện trên tập dữ liệu (ví dụ: MNIST/EMNIST) để phân loại hình ảnh.

**Tính năng chính:**
* **Xử lý ảnh:** Tự động chuyển đổi ảnh màu sang đen trắng, khử nhiễu và cắt vùng chứa chữ (ROI segmentation).
* **Huấn luyện mô hình:** Script tự động huấn luyện và lưu model tốt nhất.
* **Nhận dạng:** Dự đoán ký tự từ ảnh upload hoặc vẽ trực tiếp trên giao diện.

## 📂 Cấu trúc dự án

```text
├── data/               # Chứa dữ liệu (Raw & Processed)
├── models/             # Chứa file model đã train (.h5)
├── notebooks/          # Jupyter Notebooks phân tích dữ liệu
├── src/                # Source code chính (Model, Preprocessing, Training)
├── app/                # Code giao diện người dùng (UI)
└── requirements.txt    # Các thư viện phụ thuộc