import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.transformers import ImageResizer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = ImageResizer.resize_maintaining_aspect_ratio(image, *self.input_shapes[0][1:3][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs
    import os

    # 1. Xác định thư mục gốc của dự án (project_root)
    # Lấy thư mục chứa file hiện tại (src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Lấy thư mục cha của src (handwritten_recognition_ai)
    project_root = os.path.dirname(current_dir)

    # 2. Tạo đường dẫn tuyệt đối tới file configs.yaml
    config_path = os.path.join(project_root, "models", "model_demo", "configs.yaml")

    # Load cấu hình
    configs = BaseModelConfigs.load(config_path)

    # 3. QUAN TRỌNG: Cập nhật đường dẫn model thành tuyệt đối
    # configs.model_path lấy từ yaml là tương đối ("models/model_demo") -> cần nối với project_root
    configs.model_path = os.path.join(project_root, configs.model_path)

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    # 4. Tạo đường dẫn tuyệt đối tới file val.csv
    val_csv_path = os.path.join(project_root, "models", "model_demo", "val.csv")
    df = pd.read_csv(val_csv_path).values.tolist()

    accum_cer, accum_wer = [], []
    for image_path, label in tqdm(df):
        # 5. Cập nhật đường dẫn ảnh thành tuyệt đối để cv2 đọc được
        # image_path trong csv là "Datasets/..." -> nối với project_root
        full_image_path = os.path.join(project_root, "Datasets", "IAM_Sentences", "sentences", image_path)


        image = cv2.imread(full_image_path)

        # Kiểm tra xem ảnh có đọc được không
        if image is None:
            print(f"Warning: Không tìm thấy ảnh tại {full_image_path}")
            continue

        prediction_text = model.predict(image)

        cer = get_cer(prediction_text, label)
        wer = get_wer(prediction_text, label)
        print("Image: ", full_image_path)
        print("Label:", label)
        print("Prediction: ", prediction_text)
        print(f"CER: {cer}; WER: {wer}")

        accum_cer.append(cer)
        accum_wer.append(wer)

        # cv2.imshow(prediction_text, image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    if accum_cer and accum_wer:
        print(f"Average CER: {np.average(accum_cer)}, Average WER: {np.average(accum_wer)}")
    else:
        print("Không có dữ liệu để tính toán (có thể do không tìm thấy ảnh).")