import gradio as gr
import cv2
import numpy as np
from PIL import Image
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
        image = ImageResizer.resize_maintaining_aspect_ratio(
            image, *self.input_shapes[0][1:3][::-1]
        )
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        # Xử lý predictions thành softmax
        sum_first_timestep = np.sum(preds[0, 0, :])
        if 0.99 < sum_first_timestep < 1.01:
            softmax_preds = preds
        elif np.min(preds) < 0:
            softmax_preds = np.exp(preds)
        else:
            preds_shifted = preds - np.max(preds, axis=-1, keepdims=True)
            softmax_preds = np.exp(preds_shifted) / np.sum(np.exp(preds_shifted), axis=-1, keepdims=True)

        predicted_indices = np.argmax(softmax_preds[0], axis=-1)
        blank_index = len(self.char_list)

        # Thu thập non-blank probabilities
        non_blank_probs = []
        for t, idx in enumerate(predicted_indices):
            if idx != blank_index:
                prob = softmax_preds[0, t, idx]
                non_blank_probs.append(prob)

        # CONFIDENCE MỚI - BỎ LENGTH PENALTY
        if len(non_blank_probs) > 0:
            # Geometric mean
            log_probs = np.log(np.maximum(non_blank_probs, 1e-9))
            mean_log = np.mean(log_probs)
            base_confidence = np.exp(mean_log)

            # Soft min penalty (0.7 - 1.0 range)
            min_prob = np.min(non_blank_probs)
            min_penalty = 0.7 + 0.3 * min_prob

            # Final confidence
            confidence = base_confidence * min_penalty * 100

            print(f"Base: {base_confidence * 100:.1f}% | Min: {min_prob:.2f} | Final: {confidence:.1f}%")
        else:
            confidence = 0.0

        return text, confidence


# Load cấu hình
config_path = "models/model_demo/configs.yaml"
configs = BaseModelConfigs.load(config_path)

# Khởi tạo model
model = ImageToWordModel(
    model_path=configs.model_path,
    char_list=configs.vocab
)


def recognize_handwriting(image, ground_truth=None):
    try:
        if image is None:
            return None, None, None

        # Xử lý định dạng đầu vào
        if isinstance(image, dict):
            image = image.get("composite", image.get("background"))

        if isinstance(image, Image.Image):
            image = np.array(image)

        if not isinstance(image, np.ndarray):
            return None, None, None

        # Chuẩn hóa màu sắc
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Dự đoán
        prediction_text, confidence_score = model.predict(image)

        # Hiển thị kết quả
        result_html = f"""
        <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 12px; margin: 10px 0;">
            <h2 style="color: white; margin: 0 0 15px 0; font-size: 1.3em;">📝 Kết quả nhận diện</h2>
            <div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 8px; 
                        font-size: 1.8em; font-weight: 500; color: #2d3748; text-align: center;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                {prediction_text}
            </div>
        </div>
        """

        # Màu sắc confidence dựa trên score
        if confidence_score >= 80:
            conf_color = "#10b981"  # Green
            conf_emoji = "🎯"
        elif confidence_score >= 60:
            conf_color = "#f59e0b"  # Orange
            conf_emoji = "⚠️"
        else:
            conf_color = "#ef4444"  # Red
            conf_emoji = "❌"

        confidence_html = f"""
        <div style="padding: 20px; background: white; border-radius: 12px; 
                    border: 3px solid {conf_color}; margin: 10px 0;">
            <div style="display: flex; align-items: center; justify-content: center; gap: 15px;">
                <span style="font-size: 2em;">{conf_emoji}</span>
                <div>
                    <div style="color: #6b7280; font-size: 0.9em; margin-bottom: 5px;">Độ tin cậy</div>
                    <div style="font-size: 2.5em; font-weight: bold; color: {conf_color};">
                        {confidence_score:.1f}%
                    </div>
                </div>
            </div>
        </div>
        """

        # Metrics nếu có ground truth
        metrics_html = ""
        if ground_truth and ground_truth.strip() != "":
            val_cer = cer(ground_truth.strip(), prediction_text)
            val_wer = wer(ground_truth.strip(), prediction_text)

            metrics_html = f"""
            <div style="padding: 20px; background: #f3f4f6; border-radius: 12px; margin: 10px 0;">
                <h3 style="margin: 0 0 15px 0; color: #374151;">📊 Chi tiết đánh giá</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
                        <div style="color: #6b7280; font-size: 0.85em; margin-bottom: 5px;">Character Error Rate</div>
                        <div style="font-size: 1.8em; font-weight: bold; color: #ef4444;">{val_cer:.1%}</div>
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
                        <div style="color: #6b7280; font-size: 0.85em; margin-bottom: 5px;">Word Error Rate</div>
                        <div style="font-size: 1.8em; font-weight: bold; color: #f59e0b;">{val_wer:.1%}</div>
                    </div>
                </div>
                <div style="margin-top: 15px; padding: 12px; background: #fef3c7; border-radius: 8px; 
                            border-left: 4px solid #f59e0b;">
                    <div style="font-size: 0.85em; color: #92400e; margin-bottom: 5px;"><b>Ground Truth:</b></div>
                    <div style="font-size: 1.1em; color: #78350f;">{ground_truth}</div>
                </div>
            </div>
            """

        return result_html, confidence_html, metrics_html

    except Exception as e:
        error_html = f"""
        <div style="padding: 20px; background: #fee; border-radius: 12px; border: 2px solid #ef4444;">
            <h3 style="color: #dc2626; margin: 0 0 10px 0;">❌ Lỗi</h3>
            <p style="color: #991b1b; margin: 0;">{str(e)}</p>
        </div>
        """
        return error_html, None, None


# CSS tối ưu
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

#main_title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3em !important;
    font-weight: 800 !important;
    margin: 30px 0 10px 0 !important;
    letter-spacing: -1px;
}

#subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1.2em;
    margin-bottom: 40px;
}

.upload-section {
    padding: 20px;
    background: white;
    border-radius: 16px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
}

button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}

button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1) !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
}

.gr-button-secondary {
    background: #f3f4f6 !important;
    color: #374151 !important;
    border: 2px solid #e5e7eb !important;
}

footer {
    display: none !important;
}
"""

# Tạo giao diện
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.HTML('<h1 id="main_title">✍️ Handwriting Recognition AI</h1>')
    gr.HTML('<p id="subtitle">Upload your handwritten text image for instant recognition</p>')

    with gr.Row():
        with gr.Column(scale=1):
            # Upload ảnh
            image_input = gr.Image(
                label="📤 Upload Image",
                type="pil",
                height=400,
                elem_classes="upload-section"
            )

            # Ground truth (optional) - Đơn giản hóa
            ground_truth_input = gr.Textbox(
                label="Ground Truth (optional)",
                placeholder="Enter expected text to see metrics...",
                lines=1,
                visible=False
            )

            # Buttons
            with gr.Row():
                recognize_btn = gr.Button("🚀 Recognize", variant="primary", size="lg", scale=2)
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="lg", scale=1)

        with gr.Column(scale=1):
            # Results
            result_output = gr.HTML(label="Result")
            confidence_output = gr.HTML(label="Confidence")
            metrics_output = gr.HTML(label="Metrics")

    # Event handlers
    recognize_btn.click(
        fn=recognize_handwriting,
        inputs=[image_input, ground_truth_input],
        outputs=[result_output, confidence_output, metrics_output]
    )

    clear_btn.click(
        fn=lambda: (None, "", None, None, None),
        inputs=[],
        outputs=[image_input, ground_truth_input, result_output, confidence_output, metrics_output]
    )

    # Auto-recognize on image upload
    image_input.change(
        fn=recognize_handwriting,
        inputs=[image_input, ground_truth_input],
        outputs=[result_output, confidence_output, metrics_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True
    )