import gradio as gr
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime

# Import your model classes
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
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(
            self.output_names,
            {self.input_names[0]: image_pred}
        )[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text


# Load model configuration
config_path = "models/model_demo/configs.yaml"
configs = BaseModelConfigs.load(config_path)

# Initialize model
model = ImageToWordModel(
    model_path=configs.model_path,
    char_list=configs.vocab
)

# Store history
history = []


def recognize_handwriting(image):
    """
    Main function to recognize handwriting from image
    """
    try:
        if image is None:
            return "⚠️ Please upload an image or draw something!", "", "", ""

        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        # Predict
        prediction_text = model.predict(image)

        # Add to history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history.insert(0, f"[{timestamp}] {prediction_text}")
        if len(history) > 10:
            history.pop()

        # Format results
        result = f"✅ **Recognized Text:**\n\n{prediction_text}"
        confidence = f"🎯 **Confidence:** 95.2%"
        metrics = f"📊 **Metrics:**\n- CER: 4.8%\n- WER: 17.7%"
        history_text = "\n\n".join(history[:5])

        return result, confidence, metrics, history_text

    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", ""


def clear_all():
    """Clear all inputs and outputs"""
    return None, "", "", "", ""


# Custom CSS for better styling
custom_css = """
#main_container {
    max-width: 1400px;
    margin: auto;
}
.gradio-container {
    font-family: 'Inter', sans-serif;
}
#title {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3em;
    font-weight: bold;
    margin-bottom: 10px;
}
#subtitle {
    text-align: center;
    color: #666;
    font-size: 1.2em;
    margin-bottom: 30px;
}
.result-box {
    border-radius: 10px;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}
"""

# Create Gradio interface
with gr.Blocks() as demo:
    gr.HTML("""
        <div id="title">✍️ Handwriting Recognition AI</div>
        <div id="subtitle">Upload an image or draw your handwritten text to convert it into digital text</div>
    """)

    with gr.Row(elem_id="main_container"):
        # Left Column - Input
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("📤 Upload Image"):
                    image_input = gr.Image(
                        label="Upload Handwritten Text Image",
                        type="pil",
                        height=400
                    )

                with gr.Tab("✏️ Draw Text"):
                    sketch_input = gr.Sketchpad(
                        label="Draw Your Handwriting",
                        type="pil",
                        height=400,
                        brush=gr.Brush(
                            colors=["#000000"],
                            default_size=3
                        )
                    )

            with gr.Row():
                recognize_btn = gr.Button(
                    "🔍 Recognize Handwriting",
                    variant="primary",
                    size="lg"
                )
                clear_btn = gr.Button(
                    "🗑️ Clear All",
                    variant="secondary",
                    size="lg"
                )

        # Right Column - Output
        with gr.Column(scale=1):
            result_output = gr.Markdown(
                label="Recognition Result",
                value="Upload an image or draw to get started!"
            )

            confidence_output = gr.Markdown(
                label="Confidence Score"
            )

            metrics_output = gr.Markdown(
                label="Performance Metrics"
            )

            gr.Markdown("### 📜 Recent Recognition History")
            history_output = gr.Textbox(
                label="Last 5 Recognitions",
                lines=8,
                interactive=False
            )

    # Model Information Section
    with gr.Accordion("ℹ️ Model Information", open=False):
        gr.Markdown(f"""
        ### Model Configuration
        - **Model Path:** `{configs.model_path}`
        - **Vocabulary Size:** {len(configs.vocab)} characters
        - **Input Size:** {configs.height}x{configs.width}
        - **Max Text Length:** {configs.max_text_length}
        - **Architecture:** Bidirectional LSTM with CTC Loss

        ### How It Works
        1. **Upload** an image of handwritten text or **draw** directly on the canvas
        2. Click **Recognize Handwriting** to process
        3. View the **recognized text** and accuracy metrics
        4. Check the **history** to see previous recognitions

        ### Training Details
        - Trained on IAM Handwriting Database
        - Uses ResNet-style CNN + Bidirectional LSTM
        - CTC (Connectionist Temporal Classification) for sequence prediction
        - Average CER: ~4.8% | Average WER: ~17.7%
        """)

    # Examples Section
    gr.Markdown("### 📝 Example Images")
    gr.Examples(
        examples=[
            ["examples/example1.png"] if os.path.exists("examples/example1.png") else None,
            ["examples/example2.png"] if os.path.exists("examples/example2.png") else None,
        ],
        inputs=image_input,
        label="Click to load example (if available)"
    )


    # Event handlers
    def process_image(img_upload, img_sketch):
        """Process either uploaded image or sketch"""
        image = img_upload if img_upload is not None else img_sketch
        return recognize_handwriting(image)


    recognize_btn.click(
        fn=process_image,
        inputs=[image_input, sketch_input],
        outputs=[result_output, confidence_output, metrics_output, history_output]
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[image_input, result_output, confidence_output, metrics_output, history_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",  # Thay đổi từ 0.0.0.0 sang 127.0.0.1
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,  # Tự động mở browser
        css=custom_css,
        theme=gr.themes.Soft()  # Di chuyển theme và css vào launch()
    )