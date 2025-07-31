import json
from pathlib import Path

# import pytorch_lightning as L
import lightning.pytorch as L
import torch
from torch.utils.data import DataLoader, random_split
import re

from dataset import (
    Chunk,
    CustomDataset,
    CustomDatasetForTest,
    CustomSampler,
    collate_fn,
)


def extract_task_name(filename: str) -> str:
    pattern = r"test_data_task(\d+(?:_\d+)?)_"

    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Task name not found in filename: {filename}")


def analyze_label_dist(files_list):
    ai_addressee_count = {}
    addressee_count = {}
    ctrl_token_count = {}

    for file in files_list:
        with open(file, "r") as f:
            data = json.load(f)

            # chunk_list = [
            #     Chunk(
            #         tape=item["tape"],
            #         addressee=item["addressee"],
            #         control_token=item["control_token"],
            #         ai_addressee=item["ai_addressee"],
            #         ai_response=item["ai_response"],
            #         original_response_without_interruption=item[
            #             "original_response_without_interruption"
            #         ],
            #     )
            #     for item in data
            # ]
            for item in data:
                ai_addr = item["ai_addressee"]
                addr = item["addressee"]
                ctrl = item["control_token"]
                if ai_addr not in ai_addressee_count:
                    ai_addressee_count[ai_addr] = 0
                ai_addressee_count[ai_addr] += 1

                if addr not in addressee_count:
                    addressee_count[addr] = 0
                addressee_count[addr] += 1

                if ctrl not in ctrl_token_count:
                    ctrl_token_count[ctrl] = 0
                ctrl_token_count[ctrl] += 1

    print("AI Addressee Distribution:")
    for addressee, count in ai_addressee_count.items():
        print(f"{addressee}: {count}")
    print("-" * 20)
    print("Addressee Distribution:")
    for addressee, count in addressee_count.items():
        print(f"{addressee}: {count}")
    print("-" * 20)
    print("Control Token Distribution:")
    for ctrl_token, count in ctrl_token_count.items():
        print(f"{ctrl_token}: {count}")


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

            # > analyze ai addressee label distribution
            print(f"{' Train ':=^30}")
            analyze_label_dist(train_list)
            print()
            print(f"{' Val ':=^30}")
            analyze_label_dist(val_list)
            print()

            self.train_dataset = CustomDataset(
                train_list, batch_size=self.config.train_batch_size
            )
            self.val_dataset = CustomDataset(
                val_list, batch_size=self.config.val_batch_size
            )

            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Validation dataset size: {len(self.val_dataset)}")

        if stage == "test":
            # > get file path list of ? samples(? .json paths) from test_data_dir
            test_list = sorted(
                [str(p) for p in Path(self.test_data_dir).glob("*.json")]
            )

            test_list_task1 = []
            test_list_task2 = []
            test_list_task3 = []
            test_list_task5_1 = []
            test_list_task5_2 = []
            test_list_task5_3 = []
            test_list_task6 = []

            for file in test_list:
                task_name = "task" + extract_task_name(file)
                if task_name == "task1":
                    test_list_task1.append(file)
                elif task_name == "task2":
                    test_list_task2.append(file)
                elif task_name == "task3":
                    test_list_task3.append(file)
                elif task_name == "task5_1":
                    test_list_task5_1.append(file)
                elif task_name == "task5_2":
                    test_list_task5_2.append(file)
                elif task_name == "task5_3":
                    test_list_task5_3.append(file)
                elif task_name == "task6":
                    test_list_task6.append(file)

            total_file_count = (
                len(test_list_task1)
                + len(test_list_task2)
                + len(test_list_task3)
                + len(test_list_task5_1)
                + len(test_list_task5_2)
                + len(test_list_task5_3)
                + len(test_list_task6)
            )

            print(f"Test dataset size: {total_file_count}")

            self.test_dataset = CustomDatasetForTest(
                test_list_task1,
                test_list_task2,
                test_list_task3,
                test_list_task5_1,
                test_list_task5_2,
                test_list_task5_3,
                test_list_task6,
                total_file_count,
            )

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
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            sampler=CustomSampler(self.test_dataset),
        )
