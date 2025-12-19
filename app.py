import gradio as gr
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime
from jiwer import cer, wer

# Import các lớp từ dự án của bạn
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from mltu.transformers import ImageResizer
from mltu.configs import BaseModelConfigs


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image):
        # 1. Resize giữ nguyên tỉ lệ theo chuẩn của model
        image = ImageResizer.resize_maintaining_aspect_ratio(
            image, *self.input_shapes[0][1:3][::-1]
        )

        # ĐẢM BẢO 3 KÊNH MÀU: Tránh lỗi Rank cho ONNX
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        # 2. Tạo Rank 4 (1, H, W, C)
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        # 3. Chạy dự đoán
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        # 4. Tính toán Confidence Score thực tế từ lớp Softmax
        softmax_preds = np.exp(preds) / np.sum(np.exp(preds), axis=-1, keepdims=True)
        max_probs = np.max(softmax_preds, axis=-1)[0]
        confidence = np.mean(max_probs) * 100

        # 5. Giải mã văn bản bằng CTC
        text = ctc_decoder(preds, self.char_list)[0]

        return text, confidence


# Load cấu hình
config_path = "models/model_demo/configs.yaml"
configs = BaseModelConfigs.load(config_path)

# Khởi tạo model
model = ImageToWordModel(
    model_path=configs.model_path,
    char_list=configs.vocab
)

# Lưu lịch sử
history = []


def recognize_handwriting(image, ground_truth=None):
    try:
        if image is None:
            return "⚠️ Vui lòng cung cấp ảnh!", "", "", ""

        # --- BƯỚC 1: XỬ LÝ ĐỊNH DẠNG ĐẦU VÀO ---
        if isinstance(image, dict):
            image = image.get("composite", image.get("background"))

        if isinstance(image, Image.Image):
            image = np.array(image)

        if not isinstance(image, np.ndarray):
            return "❌ Định dạng ảnh không hợp lệ!", "", "", ""

        # Chuẩn hóa màu sắc
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # --- BƯỚC 2: DỰ ĐOÁN (SỬA LỖI TẠI ĐÂY) ---
        # Phải tách kết quả ra làm 2 biến riêng biệt
        prediction_text, confidence_score = model.predict(image)

        # --- BƯỚC 3: TÍNH TOÁN CER/WER THẬT ---
        if ground_truth and ground_truth.strip() != "":
            val_cer = cer(ground_truth.strip(), prediction_text)
            val_wer = wer(ground_truth.strip(), prediction_text)

            result = f"✅ **Kết quả:** {prediction_text}"
            confidence_display = f"🎯 **Độ tin cậy:** {confidence_score:.2f}%"
            metrics = f"📊 **Metrics thực tế:**\n- CER: {val_cer:.2%}\n- WER: {val_wer:.2%}"
        else:
            result = f"✅ **Kết quả:** {prediction_text}"
            confidence_display = f"🎯 **Độ tin cậy:** {confidence_score:.2f}%"
            metrics = "📊 **Metrics:** Nhập 'Ground Truth' để xem kết quả"

        # Cập nhật lịch sử
        timestamp = datetime.now().strftime("%H:%M:%S")
        history.insert(0, f"[{timestamp}] {prediction_text} ({confidence_score:.1f}%)")
        history_text = "\n\n".join(history[:5])

        return result, confidence_display, metrics, history_text

    except Exception as e:
        return f"❌ Lỗi hệ thống: {str(e)}", "", "", ""


def clear_all():
    return None, "", "", "", ""


# CSS trang trí
custom_css = """
#main_container { max-width: 1400px; margin: auto; }
.gradio-container { font-family: 'Inter', sans-serif; }
#title { text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3em; font-weight: bold; margin-bottom: 10px; }
#subtitle { text-align: center; color: #666; font-size: 1.2em; margin-bottom: 30px; }
"""

with gr.Blocks(css=custom_css) as demo:
    gr.HTML("""
        <div id="title">✍️ Handwriting Recognition AI</div>
        <div id="subtitle">Upload an image or draw your handwritten text to convert it into digital text</div>
    """)

    with gr.Row(elem_id="main_container"):
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("📤 Upload Image"):
                    image_input = gr.Image(label="Chọn ảnh", type="pil", height=300)
                    ground_truth_input = gr.Textbox(label="Ground Truth (Đối chiếu đúng/sai)",
                                                    placeholder="Ví dụ: Hello")

                with gr.Tab("✏️ Draw Text"):
                    sketch_input = gr.Sketchpad(label="Vẽ tay", type="pil", height=400,
                                                brush=gr.Brush(colors=["#000000"], default_size=3))

            with gr.Row():
                recognize_btn = gr.Button("🔍 Nhận diện", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Xóa hết", variant="secondary", size="lg")

        with gr.Column(scale=1):
            result_output = gr.Markdown(label="Kết quả", value="Tải ảnh lên để bắt đầu!")
            confidence_output = gr.Markdown(label="Độ tin cậy")
            metrics_output = gr.Markdown(label="Chỉ số lỗi")
            gr.Markdown("### 📜 Lịch sử")
            history_output = gr.Textbox(label="5 lần gần nhất", lines=8, interactive=False)

    with gr.Accordion("ℹ️ Thông tin mô hình", open=False):
        gr.Markdown(f"""
        ### Model Configuration
        - **Vocabulary:** {len(configs.vocab)} ký tự
        - **Input Size:** {configs.width}x{configs.height}
        - **Architecture:** ResNet-CNN + Bi-LSTM + CTC
        """)

    # Gán sự kiện
    recognize_btn.click(
        fn=lambda img_u, img_s, gt: recognize_handwriting(img_u if img_u else img_s, gt),
        inputs=[image_input, sketch_input, ground_truth_input],
        outputs=[result_output, confidence_output, metrics_output, history_output]
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[image_input, result_output, confidence_output, metrics_output, history_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, theme=gr.themes.Soft())