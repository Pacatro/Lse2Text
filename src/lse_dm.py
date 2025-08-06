import os
import random
import requests
from pathlib import Path
import zipfile
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
import config


class LseDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        batch_size: int = 64,
        image_size: tuple[int, int] = (256, 256),
        max_preds: int = 20,
    ):
        super().__init__()
        assert test_size + val_size <= 1.0, (
            "You have to use a proper value for test and validations size"
        )
        self.root_dir = root_dir
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.max_preds = max_preds

        self.img_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def _split(self, dataset: datasets.ImageFolder) -> tuple[list, list, list]:
        idx = list(range(len(dataset)))
        train_idx, temp_idx = train_test_split(
            idx, test_size=self.val_size + self.test_size, random_state=42
        )
        relative_test_size = self.test_size / (self.val_size + self.test_size)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=relative_test_size, random_state=42
        )
        return train_idx, test_idx, val_idx

    def prepare_data(self):
        # Create folder if it doesn't exist
        dataset_dir = Path(config.DATASET_DIR)

        if dataset_dir.exists():
            return

        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Download dataset
        print("Downloading dataset...")
        resp = requests.get(config.KAGGLE_URL, stream=True)
        resp.raise_for_status()

        with open(config.ZIP_FILE, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        print("Dataset downloaded successfully.")

        # Unzip
        with zipfile.ZipFile(config.ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(config.DATASET_DIR)
            print("Dataset extracted successfully.")

        os.remove(config.ZIP_FILE)

    def setup(self, stage: str):
        self.dataset = datasets.ImageFolder(self.root_dir, self.img_transform)
        self.train_idx, self.test_idx, self.val_idx = self._split(self.dataset)

        if stage in (None, "fit"):
            self.train_dataset = Subset(self.dataset, self.train_idx)
            self.val_dataset = Subset(self.dataset, self.val_idx)
        if stage in (None, "test"):
            self.test_dataset = Subset(self.dataset, self.test_idx)
        if stage == "predict" and self.max_preds > 0:
            # random.seed(42)
            shuffled_test_idx = random.sample(self.test_idx, len(self.test_idx))
            predict_idx = shuffled_test_idx[: self.max_preds]
            self.predict_dataset = Subset(self.dataset, predict_idx)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
