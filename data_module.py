from pathlib import Path
# import pytorch_lightning as L
import lightning.pytorch as L
import torch
from torch.utils.data import DataLoader, random_split

from dataset import CustomDataset, CustomSampler, collate_fn


class DataModule(L.LightningDataModule):
    def __init__(self, config, data_dir: str = "./", test_data_dir: str = "./"):
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.test_data_dir = test_data_dir

    def setup(self, stage: str):
        if stage == "fit":
            # > get file path list of 15000 samples(15000 .json paths) from data_dir
            file_list = sorted([str(p) for p in Path(self.data_dir).glob("*.json")])

            # > Split the file list into training and validation sets
            train_size = 1300
            # val_size = len(file_list) - train_size
            generator = torch.Generator().manual_seed(self.config.seed)
            indices = torch.randperm(len(file_list), generator=generator).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            train_list = [file_list[i] for i in train_indices]
            val_list = [file_list[i] for i in val_indices]
            # train_list, val_list = random_split(
            #     file_list,
            #     [1300, 200],
            #     generator=torch.Generator().manual_seed(self.config.seed),
            # )
            self.train_dataset = CustomDataset(
                train_list, batch_size=self.config.train_batch_size
            )
            self.val_dataset = CustomDataset(
                val_list, batch_size=self.config.val_batch_size
            )
            
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Validation dataset size: {len(self.val_dataset)}")

        if stage == "test":
            # > get file path list of 500 samples(500 .json paths) from test_data_dir
            test_list = sorted(
                [str(p) for p in Path(self.test_data_dir).glob("*.json")]
            )
            self.test_dataset = CustomDataset(test_list)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=63,
            collate_fn=collate_fn,
            sampler=CustomSampler(self.train_dataset),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=63,
            collate_fn=collate_fn,
            sampler=CustomSampler(self.val_dataset),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, sampler=CustomSampler(self.test_dataset)
        )
