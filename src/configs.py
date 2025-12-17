# src/configs.py
import os
from datetime import datetime
from pathlib import Path  # Thêm thư viện này
from mltu.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.base_path = Path(__file__).resolve().parent.parent

        self.model_path = os.path.join(
            self.base_path,
            "models/04_sentence_recognition",
            datetime.strftime(datetime.now(), "%Y%m%d%H%M")
        )
        self.vocab = ""
        self.height = 96
        self.width = 1408
        self.max_text_length = 0
        self.batch_size = 16
        self.learning_rate = 0.0005
        self.train_epochs = 1000
        self.train_workers = 20