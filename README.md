# Nhận diện chữ viết tay với TensorFlow

Dự án nhận diện câu chữ viết tay sử dụng TensorFlow và CTC loss. Ứng dụng trong số hóa ghi chú, phiên âm tài liệu lịch sử và tự động chấm điểm bài thi.

---

## Tổng quan

Hệ thống sử dụng mạng neural sâu kết hợp CNN và Bidirectional LSTM để nhận diện câu chữ viết tay từ ảnh. Model được huấn luyện trên tập dữ liệu IAM Handwriting Database.

### Kiến trúc model

- **Backbone**: Residual CNN với 5 block
- **Sequence modeling**: 2 lớp Bidirectional LSTM
- **Decoder**: CTC (Connectionist Temporal Classification)

---

## Tính năng

- Nhận diện câu chữ viết tay từ ảnh
- Sửa lỗi chính tả dựa trên OCR confusion matrix
- Tính toán độ tin cậy (confidence score)
- Đánh giá hiệu suất với CER và WER
- Giao diện web trực quan với Gradio

---

## Cấu trúc thư mục

```
handwriting-recognition-ai/
│
├── Datasets/
│   └── IAM_Sentences/          # Tập dữ liệu IAM
│       ├── ascii/
│       │   └── sentences.txt
│       └── sentences/
│
├── models/
│   └── model_demo/             # Model đã huấn luyện
│       ├── configs.yaml
│       ├── model.h5
│       ├── model.onnx
│       ├── train.csv
│       └── val.csv
│
├── src/
│   ├── configs.py              # Cấu hình model
│   ├── model.py                # Kiến trúc neural network
│   ├── train.py                # Script huấn luyện
│   └── inferenceModel.py       # Inference và đánh giá
│
├── app.py                      # Giao diện Gradio
├── requirements.txt
└── README.md
```

---

## Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- CUDA (khuyến nghị cho training)
- 8GB RAM (tối thiểu)

### Các bước cài đặt

1. Clone repository:

```bash
git clone https://github.com/khangtran1825/handwritten_recognition_ai.git
cd handwritten_recognition_ai
```

2. Tạo môi trường ảo:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

4. Tải tập dữ liệu IAM:

- Truy cập: https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database
- Tải "IAM Handwriting Database - Sentences"
- Giải nén vào thư mục `Datasets/IAM_Sentences/`

---

## Sử dụng

### 1. Huấn luyện model

```bash
python src/train.py
```

Cấu hình huấn luyện có thể chỉnh sửa trong `src/configs.py`:

```python
self.height = 96
self.width = 1408
self.batch_size = 32
self.learning_rate = 0.0005
self.train_epochs = 1000
```

### 22. Chạy giao diện web

```bash
python app.py
```

Truy cập `http://127.0.0.1:7860` để sử dụng giao diện.

### 3. Inference từ code

```python
from mltu.configs import BaseModelConfigs
from src.inferenceModel import ImageToWordModel
import cv2

# Load config và model
configs = BaseModelConfigs.load("models/20251201/configs.yaml")
model = ImageToWordModel(
    model_path=configs.model_path,
    char_list=configs.vocab,
    use_post_processing=True,
    corpus_path="models/20251201/corpus.txt"
)

# Đọc ảnh
image = cv2.imread("path/to/image.png")

# Dự đoán
prediction, raw_prediction, confidence = model.predict(image)
print(f"Kết quả: {prediction}")
print(f"Độ tin cậy: {confidence:.2f}%")
```

---

## Hậu xử lý (Pos```

---

## Đánh giá hiệu suất

### Metrics

- **CER** (Character Error Rate): Tỷ lệ lỗi ở mức ký tự
- **WER** (Word Error Rate): Tỷ lệ lỗi ở mức từ
- **Confidence Score**: Độ tin cậy của dự đoán

### Chạy đánh giá trên validation set

```bash
python src/inferenceModel.py
```

Kết quả ví dụ:

```
Metric               Raw             Safe PP         Aggressive PP
--------------------------------------------------------------------------------
Average CER          0.0856          0.0823          0.0815
Average WER          0.2134          0.2087          0.2098
Avg Time (ms)        45.23           48.67           52.34

IMPROVEMENT vs RAW
Safe Mode:
  CER: +3.85% | WER: +2.20%
```

---

## Tập dữ liệu

### IAM Handwriting Database

- 1,539 trang văn bản viết tay
- 13,353 dòng văn bản riêng lẻ
- 115,320 từ
- 747 người viết khác nhau

### Phân chia dữ liệu

- Training: 90%
- Validation: 10%

### Data augmentation

- Random brightness
- Random erode/dilate
- Random sharpen

---

## Công nghệ sử dụng

- **Framework**: TensorFlow/Keras
- **OCR**: mltu (Machine Learning Training Utilities)
- **Interface**: Gradio
- **Deployment**: ONNX Runtime
- **Preprocessing**: OpenCV, NumPy
- **Metrics**: jiwer (WER/CER calculation)

---

## Hạn chế và cải tiến

### Hạn chế hiện tại

- Chỉ hỗ trợ tiếng Anh
- Cần ảnh có chất lượng tốt
- Model size lớn (~100MB)

### Hướng phát triển

- Hỗ trợ đa ngôn ngữ (tiếng Việt)
- Tối ưu model size (quantization, pruning)
- Sử dụng Transformer architecture
- Hỗ trợ nhận diện real-time

---

## Tham khảo

- [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- [CTC Loss Tutorial](https://distill.pub/2017/ctc/)
- [Viterbi Algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm)

---

## Giấy phép

MIT License

---

## Đóng góp

Mọi đóng góp đều được chào đón. Vui lòng:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

---

## Liên hệ

Nếu có thắc mắc hoặc góp ý, vui lòng mở issue trên GitHub.
