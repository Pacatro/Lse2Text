from datamodule import LSEDataModule
import config
from model import VGG
from engine import Lse2TextModel


def main():
    dm = LSEDataModule(
        root_dir=config.DATASET_DIR,
        image_size=(config.IMG_WIDTH, config.IMG_HEIGHT),
    )
    dm.setup("fit")
    print("Dataset classes:", dm.dataset.classes)

    num_classes = len(dm.dataset.classes)
    vgg = VGG(num_classes=num_classes)
    model = Lse2TextModel(model=vgg)
    print(model)


if __name__ == "__main__":
    main()
