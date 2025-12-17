import cv2
import numpy as np
import onnxruntime as ort
import os
import yaml
import typing


# --- PHẦN 1: CÁC HÀM PHỤ TRỢ (Đã tách từ mltu ra để chạy độc lập) ---

def ctc_decoder(predictions, vocab):
    """Giải mã kết quả từ Model (Output Matrix -> Text)"""
    # Lấy index có xác suất cao nhất tại mỗi bước thời gian
    pred_indices = np.argmax(predictions[0], axis=1)

    text = ""
    last_index = -1
    blank_index = len(vocab)  # Ký tự Blank thường nằm cuối cùng

    for index in pred_indices:
        # CTC Logic: Loại bỏ ký tự trùng lặp liên tiếp và ký tự Blank
        if index != last_index and index != blank_index:
            if index < len(vocab):  # Đảm bảo index hợp lệ
                text += vocab[index]
        last_index = index

    return text


def resize_image(image, target_width, target_height):
    """Resize ảnh giữ nguyên tỷ lệ và thêm viền (Padding)"""
    h, w = image.shape[:2]

    # Tính tỷ lệ scale dựa trên chiều cao (để khớp height=96)
    scale = target_height / h
    new_w = int(w * scale)

    # Resize ảnh
    resized = cv2.resize(image, (new_w, target_height))

    # Tạo ảnh nền trắng (hoặc đen tùy model train, thường là trắng cho handwriting)
    # Lưu ý: Model mltu thường normalize về 0-1, padding màu gì không quá quan trọng nếu model tốt,
    # nhưng chuẩn nhất là padding theo giá trị nền. Ở đây ta padding màu đen (giá trị 0 sau khi normalize)
    # để an toàn nhất với các phép tính ma trận.
    padded_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255

    # Chèn ảnh đã resize vào
    if new_w < target_width:
        padded_image[:, :new_w, :] = resized
    else:
        padded_image = resized[:, :target_width, :]

    return padded_image


# --- PHẦN 2: CLASS CHÍNH ---

class ImageToWordModel:
    def __init__(self, model_path, config_path):
        print(f"Loading model: {model_path}")

        # 1. Load Configs
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.vocab = self.config.get("vocab", "")
        self.height = self.config.get("height", 96)
        self.width = self.config.get("width", 1024)

        # 2. Load Model ONNX
        # Nếu máy bạn có GPU thì thêm 'CUDAExecutionProvider' vào list providers
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_path):
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            return "Error: Cannot read image", None

        # Xử lý ảnh
        processed_img = resize_image(image, self.width, self.height)

        # Chuẩn hóa (0-255 -> 0.0-1.0) và thêm chiều Batch
        img_input = processed_img.astype(np.float32) / 255.0
        img_input = np.expand_dims(img_input, axis=0)  # Shape: (1, 96, 1024, 3)

        # Chạy Model
        preds = self.session.run(None, {self.input_name: img_input})[0]

        # Giải mã
        text = ctc_decoder(preds, self.vocab)
        return text, image


# --- PHẦN 3: CHẠY THỬ ---
if __name__ == "__main__":
    # Đặt tên file của bạn ở đây
    MODEL_FILE = "model.onnx"
    CONFIG_FILE = "configs.yaml"
    IMAGE_FILE = "test_image.jpg"  # <--- Thay tên ảnh bạn muốn test vào đây

    # Kiểm tra file tồn tại
    if not os.path.exists(MODEL_FILE) or not os.path.exists(CONFIG_FILE):
        print("❌ Lỗi: Không tìm thấy file model.onnx hoặc configs.yaml trong cùng thư mục!")
        exit()

    # Khởi tạo model
    model = ImageToWordModel(MODEL_FILE, CONFIG_FILE)

    # Chạy thử
    if os.path.exists(IMAGE_FILE):
        print(f"🔍 Đang đọc ảnh: {IMAGE_FILE} ...")
        ket_qua, img_goc = model.predict(IMAGE_FILE)

        print("-" * 30)
        print(f"✅ KẾT QUẢ: {ket_qua}")
        print("-" * 30)

        # Hiển thị ảnh kèm kết quả trên cửa sổ Window
        # (Lưu ý: Tên cửa sổ không được chứa ký tự tiếng Việt có dấu nếu Windows chưa cài font hỗ trợ)
        cv2.imshow("Ket qua: " + str(ket_qua), img_goc)

        print("👉 Bấm phím bất kỳ trên cửa sổ ảnh để thoát...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"⚠️ Không tìm thấy file ảnh: {IMAGE_FILE}")
        print("Hãy copy một file ảnh vào thư mục dự án và đổi tên trong code.")