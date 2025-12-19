import gradio as gr
import cv2
import numpy as np
import os
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from mltu.configs import BaseModelConfigs


# 1. Định nghĩa lớp xử lý logic dự đoán ngay trong app.py
class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image):
        # --- BƯỚC 1: TIỀN XỬ LÝ (SỬ DỤNG LOGIC TỐI ƯU NHẤT) ---
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Nhị phân hóa thích nghi để tách chữ khỏi nền bìa carton
        gray_blur = cv2.medianBlur(gray, 3)
        binary_inv = cv2.adaptiveThreshold(gray_blur, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 4)

        # Giãn nở để tìm vùng chứa chữ
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
        dilated = cv2.dilate(binary_inv, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- BƯỚC 2: CẮT SÁT VÙNG CHỮ (CROP) ---
        if contours:
            main_contours = [c for c in contours if cv2.contourArea(c) > 150]
            if main_contours:
                img_h, img_w = img.shape[:2]
                valid_rects = []
                for c in main_contours:
                    x, y, w, h = cv2.boundingRect(c)
                    # Lọc bỏ bóng đổ phía dưới bìa carton (giữ lại 60% phía trên)
                    if y < img_h * 0.6:
                        valid_rects.append((x, y, x + w, y + h))

                if valid_rects:
                    x_min, y_min = min([r[0] for r in valid_rects]), min([r[1] for r in valid_rects])
                    x_max, y_max = max([r[2] for r in valid_rects]), max([r[3] for r in valid_rects])
                    margin = 5
                    image_cropped = gray[max(0, y_min - margin):min(img_h, y_max + margin),
                    max(0, x_min - margin):min(img_w, x_max + margin)]
                else:
                    image_cropped = gray
            else:
                image_cropped = gray
        else:
            image_cropped = gray

        # --- BƯỚC 3: CHUẨN HÓA ĐẦU VÀO MODEL ---
        _, binary_final = cv2.threshold(image_cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Resize và Padding về chuẩn 1408x96
        target_h, target_w = self.input_shapes[0][1:3]
        h, w = binary_final.shape
        ratio = min(target_h / h, target_w / w)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(binary_final, (new_w, new_h))

        canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
        canvas[:new_h, :new_w] = resized

        # Model yêu cầu 3 kênh màu đầu vào
        final_input = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        # --- BƯỚC 4: DỰ ĐOÁN ---
        image_pred = np.expand_dims(final_input, axis=0).astype(np.float32)
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text


# 2. KHỞI TẠO GIAO DIỆN (GRADIO)
# Thay đổi đường dẫn đến file configs.yaml thực tế của bạn
configs = BaseModelConfigs.load("models/model_demo/configs.yaml")
model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)


def gradio_predict(img):
    if img is None: return "Vui lòng cung cấp ảnh!"
    return model.predict(img)


with gr.Blocks(title="AI Handwriting Recognition") as demo:
    gr.Markdown("# 🖋️ Nhận Diện Chữ Viết Tay")
    with gr.Tab("Tải ảnh lên"):
        input_file = gr.Image(label="Chọn ảnh từ máy tính", type="pil")
        output_text = gr.Textbox(label="Kết quả dự đoán")
        btn = gr.Button("Dự đoán ngay", variant="primary")
        btn.click(fn=gradio_predict, inputs=input_file, outputs=output_text)

if __name__ == "__main__":
    demo.launch(inbrowser=True)