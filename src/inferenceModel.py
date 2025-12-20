import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.transformers import ImageResizer

# Import post-processor
from post_processing import TextPostProcessor


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list],
                 use_post_processing: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list
        self.use_post_processing = use_post_processing

        # Khởi tạo heuristic post-processor
        if self.use_post_processing:
            self.post_processor = TextPostProcessor()

    def predict(self, image: np.ndarray,
                apply_post_processing: typing.Optional[bool] = None,
                aggressive: bool = False):
        """
        Dự đoán text từ ảnh

        Args:
            image: Ảnh đầu vào
            apply_post_processing: Override cài đặt post-processing
            aggressive: Áp dụng spell check (có thể gây lỗi, mặc định False)

        Returns:
            text: Text đã nhận dạng
        """
        image = ImageResizer.resize_maintaining_aspect_ratio(image, *self.input_shapes[0][1:3][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        # Áp dụng post-processing nếu được bật
        if apply_post_processing is None:
            apply_post_processing = self.use_post_processing

        if apply_post_processing and self.post_processor:
            text = self.post_processor.process(text, aggressive=aggressive)

        return text


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs
    import os
    import time

    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    config_path = os.path.join(project_root, "models", "model_demo", "configs.yaml")
    configs = BaseModelConfigs.load(config_path)
    configs.model_path = os.path.join(project_root, configs.model_path)

    print("=" * 80)
    print("HEURISTIC POST-PROCESSOR (Pattern-based, no ML)")
    print("=" * 80)

    # Khởi tạo model với post-processing
    model = ImageToWordModel(
        model_path=configs.model_path,
        char_list=configs.vocab,
        use_post_processing=True
    )
    print("✓ Model loaded with heuristic post-processor")

    # Load validation set
    val_csv_path = os.path.join(project_root, "models", "model_demo", "val.csv")
    df = pd.read_csv(val_csv_path).values.tolist()

    # Evaluate
    print("\n" + "=" * 80)
    print("TESTING: Raw vs Safe vs Aggressive")
    print("=" * 80)

    results = {
        'raw': {'cer': [], 'wer': [], 'time': []},
        'safe': {'cer': [], 'wer': [], 'time': []},
        'aggressive': {'cer': [], 'wer': [], 'time': []}
    }

    num_samples = min(100, len(df))

    for image_path, label in tqdm(df[:num_samples], desc="Processing"):
        full_image_path = os.path.join(project_root, "Datasets", "IAM_Sentences", "sentences", image_path)

        image = cv2.imread(full_image_path)
        if image is None:
            continue

        # 1. RAW (no post-processing)
        start_time = time.time()
        pred_raw = model.predict(image, apply_post_processing=False)
        time_raw = time.time() - start_time

        cer_raw = get_cer(pred_raw, label)
        wer_raw = get_wer(pred_raw, label)

        results['raw']['cer'].append(cer_raw)
        results['raw']['wer'].append(wer_raw)
        results['raw']['time'].append(time_raw)

        # 2. SAFE (spacing + confusion fixes only)
        start_time = time.time()
        pred_safe = model.predict(image, apply_post_processing=True, aggressive=False)
        time_safe = time.time() - start_time

        cer_safe = get_cer(pred_safe, label)
        wer_safe = get_wer(pred_safe, label)

        results['safe']['cer'].append(cer_safe)
        results['safe']['wer'].append(wer_safe)
        results['safe']['time'].append(time_safe)

        # 3. AGGRESSIVE (with spell check)
        start_time = time.time()
        pred_agg = model.predict(image, apply_post_processing=True, aggressive=True)
        time_agg = time.time() - start_time

        cer_agg = get_cer(pred_agg, label)
        wer_agg = get_wer(pred_agg, label)

        results['aggressive']['cer'].append(cer_agg)
        results['aggressive']['wer'].append(wer_agg)
        results['aggressive']['time'].append(time_agg)

        # Print examples where safe mode helps
        if cer_safe < cer_raw or wer_safe < wer_raw:
            print(f"\n{'=' * 80}")
            print(f"Image: {image_path}")
            print(f"Label:     {label}")
            print(f"Raw:       {pred_raw}")
            print(f"Safe:      {pred_safe}")
            print(f"Aggressive:{pred_agg}")
            print(f"CER: {cer_raw:.3f} → {cer_safe:.3f} (safe) / {cer_agg:.3f} (agg)")
            print(f"WER: {wer_raw:.3f} → {wer_safe:.3f} (safe) / {wer_agg:.3f} (agg)")

    # Summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(f"\n{'Metric':<20} {'Raw':<15} {'Safe PP':<15} {'Aggressive PP':<15}")
    print("-" * 80)

    avg_cer_raw = np.mean(results['raw']['cer'])
    avg_cer_safe = np.mean(results['safe']['cer'])
    avg_cer_agg = np.mean(results['aggressive']['cer'])
    print(f"{'Average CER':<20} {avg_cer_raw:<15.4f} {avg_cer_safe:<15.4f} {avg_cer_agg:<15.4f}")

    avg_wer_raw = np.mean(results['raw']['wer'])
    avg_wer_safe = np.mean(results['safe']['wer'])
    avg_wer_agg = np.mean(results['aggressive']['wer'])
    print(f"{'Average WER':<20} {avg_wer_raw:<15.4f} {avg_wer_safe:<15.4f} {avg_wer_agg:<15.4f}")

    avg_time_raw = np.mean(results['raw']['time']) * 1000
    avg_time_safe = np.mean(results['safe']['time']) * 1000
    avg_time_agg = np.mean(results['aggressive']['time']) * 1000
    print(f"{'Avg Time (ms)':<20} {avg_time_raw:<15.2f} {avg_time_safe:<15.2f} {avg_time_agg:<15.2f}")

    print("\n" + "=" * 80)
    print("IMPROVEMENT vs RAW")
    print("=" * 80)

    cer_imp_safe = (avg_cer_raw - avg_cer_safe) / avg_cer_raw * 100
    wer_imp_safe = (avg_wer_raw - avg_wer_safe) / avg_wer_raw * 100

    cer_imp_agg = (avg_cer_raw - avg_cer_agg) / avg_cer_raw * 100
    wer_imp_agg = (avg_wer_raw - avg_wer_agg) / avg_wer_raw * 100

    print(f"Safe Mode:")
    print(f"  CER: {cer_imp_safe:+.2f}% | WER: {wer_imp_safe:+.2f}%")
    print(f"\nAggressive Mode:")
    print(f"  CER: {cer_imp_agg:+.2f}% | WER: {wer_imp_agg:+.2f}%")

    # Recommendation
    print("\n" + "=" * 80)
    if avg_cer_safe < avg_cer_raw and avg_wer_safe < avg_wer_raw:
        print("✓ SAFE MODE is recommended (improves both CER and WER)")
    elif avg_cer_agg < avg_cer_raw and avg_wer_agg < avg_wer_raw:
        print("✓ AGGRESSIVE MODE is recommended (improves both CER and WER)")
    elif avg_cer_safe < avg_cer_raw or avg_wer_safe < avg_wer_raw:
        print("⚠ SAFE MODE has mixed results (improves some metrics)")
    else:
        print("✗ POST-PROCESSING not recommended for this model")
        print("  → Model is already very good, PP introduces errors")
    print("=" * 80)