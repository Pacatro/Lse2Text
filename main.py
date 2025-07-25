from datamodule import LSEDataModule
import config


def main():
    lse_datamodule = LSEDataModule(root_dir=config.DATASET_DIR)
    lse_datamodule.setup("fit")
    print("Dataset classes:", lse_datamodule.dataset.classes)


if __name__ == "__main__":
    main()
