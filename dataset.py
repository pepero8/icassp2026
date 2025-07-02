from dataclasses import dataclass
from typing import Union
import torch
from torch.utils.data import Dataset
import json


@dataclass
class Chunk:
    # audio: list[np.float32]  # 1 second audio chunk
    tape: str  # text of the audio chunk
    addressee: str
    ai_addressee: str
    control_token: str
    ai_response: Union[str, None]
    original_response_without_interruption: Union[str, None]


@dataclass
class TrainBatch:
    sample: list[
        Chunk
    ]  # list of 64 Chunk instances(can be <64 if last chunk set of dialogue)
    reset_dialog_memory: bool = False


def collate_fn(batch):
    """
    Custom collate function to handle TrainBatch objects.
    It returns the batch as is since batch size is 1
    """
    return batch[0]  # Return the single TrainBatch


class CustomDataset(Dataset):
    def __init__(self, sample_files_list, batch_size=64):
        self.sample_files_list = sample_files_list
        self.current_sample = None
        self.batch_size = batch_size
        self.idx = 0
        self.sample_list_idx = 0
        self.reset_dialog_memory = False

    def __getitem__(self, index):
        if self.current_sample is None or self.idx >= len(self.current_sample):
            # Load a new sample
            # self.current_sample = torch.load(
            #     self.sample_files_list[self.sample_list_idx]
            # )

            # > load json file. it contains list of dictionaries
            with open(
                self.sample_files_list[self.sample_list_idx], "r", encoding="utf-8"
            ) as f:
                data = json.load(f)

            self.current_sample = data
            # > convert to list of Chunk instances
            self.current_sample = [
                Chunk(
                    tape=item["tape"],
                    addressee=item["addressee"],
                    control_token=item["control_token"],
                    ai_addressee=item["ai_addressee"],
                    ai_response=item["ai_response"],
                    original_response_without_interruption=item[
                        "original_response_without_interruption"
                    ],
                )
                for item in data
            ]

            self.idx = 0
            self.sample_list_idx += 1
            self.reset_dialog_memory = True
        else:
            self.reset_dialog_memory = False

        next = self.current_sample[self.idx : self.idx + self.batch_size]
        self.idx = self.idx + self.batch_size
        return TrainBatch(next, reset_dialog_memory=self.reset_dialog_memory)

    def __len__(self):
        return len(self.sample_files_list)


class DummyDataset(Dataset):
    """A dummy dataset for testing"""

    def __init__(self, num_samples=100, chunks_per_sample=1, num_speakers=3):
        self.num_samples = num_samples
        self.chunks_per_sample = chunks_per_sample
        self.num_speakers = num_speakers

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):

        chunks = []
        for _ in range(self.chunks_per_sample):
            tape = f"Speaker_A happens? Speaker_B Absolutely"
            addressee = "All"
            ai_addressee = "NA"
            control_token = "C.LISTEN"
            ai_response = None
            original_response_without_interruption = None

            chunk = Chunk(
                tape=tape,
                addressee=addressee,
                ai_addressee=ai_addressee,
                control_token=control_token,
                ai_response=ai_response,
                original_response_without_interruption=original_response_without_interruption,
            )
            chunks.append(chunk)

        return TrainBatch(sample=chunks, reset_dialog_memory=False)
