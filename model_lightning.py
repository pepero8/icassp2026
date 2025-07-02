import torch
from torch import nn
# import pytorch_lightning as L
import lightning.pytorch as L

from dataset import TrainBatch
from model import SAASRControl


class LitSAASRControl(L.LightningModule):
    def __init__(self, config):
        super(LitSAASRControl, self).__init__()
        self.model: SAASRControl = SAASRControl(config)
        self.config = config
        self.save_hyperparameters()
        self.loss = nn.CrossEntropyLoss()

        if config.num_speakers == 2:
            self.addressee_labels = ["Speaker_A", "Speaker_B", "Assistant", "All"]
            self.ai_addressee_labels = ["Speaker_A", "Speaker_B", "NA", "All"]
        elif config.num_speakers == 3:
            self.addressee_labels = [
                "Speaker_A",
                "Speaker_B",
                "Speaker_C",
                "Assistant",
                "All",
            ]
            self.ai_addressee_labels = ["Speaker_A", "Speaker_B", "Speaker_C", "NA", "All"]
        elif config.num_speakers == 4:
            self.addressee_labels = [
                "Speaker_A",
                "Speaker_B",
                "Speaker_C",
                "Speaker_D",
                "Assistant",
                "All",
            ]
            self.ai_addressee_labels = [
                "Speaker_A",
                "Speaker_B",
                "Speaker_C",
                "Speaker_D",
                "NA",
                "All"
            ]

        self.control_token_labels = [
            "C.LISTEN",
            "C.SPEAK",
            "S.LISTEN",
            "S.SPEAK",
        ]

        self.addressee_to_idx = {
            label: idx for idx, label in enumerate(self.addressee_labels)
        }
        self.ai_addressee_to_idx = {
            label: idx for idx, label in enumerate(self.ai_addressee_labels)
        }
        self.control_token_to_idx = {
            label: idx for idx, label in enumerate(self.control_token_labels)
        }

    def training_step(self, batch: TrainBatch, batch_idx):
        """
        batch: TrainBatch instance
        """

        sample = (
            batch.sample
        )  # list of 64 Chunk instances(can be <64 if last chunk set of dialogue)

        if batch.reset_dialog_memory:
            self.model.reset_dialog_memory()

        batch_loss = torch.tensor(0.0, device=self.device)
        num_samples = len(sample)
        for chunk in sample:
            addressee, ai_addressee, control_token = self.model(chunk)
            try:
                loss = self.compute_loss(
                    addressee,
                    ai_addressee,
                    control_token,
                    chunk.addressee,
                    chunk.ai_addressee,
                    chunk.control_token,
                )
            except Exception as e:
                print(f"Error in validation step: {e}")
                num_samples -= 1
                continue
            
            batch_loss = batch_loss + loss
            
        batch_loss = batch_loss / (num_samples if num_samples > 0 else 1)
        self.log("train_loss", batch_loss, prog_bar=True, batch_size=len(sample))

        return batch_loss

    def validation_step(self, batch: TrainBatch, batch_idx):
        sample = (
            batch.sample
        )  # list of 64 Chunk instances(can be <64 if last chunk set of dialogue)

        if batch.reset_dialog_memory:
            self.model.reset_dialog_memory()

        batch_loss = torch.tensor(0.0, device=self.device)
        num_samples = len(sample)
        for chunk in sample:
            addressee, ai_addressee, control_token = self.model(chunk)
            try:
                loss = self.compute_loss(
                    addressee,
                    ai_addressee,
                    control_token,
                    chunk.addressee,
                    chunk.ai_addressee,
                    chunk.control_token,
                )
            except Exception as e:
                print(f"Error in validation step: {e}")
                num_samples -= 1
                continue
                
            batch_loss = batch_loss + loss
            
        batch_loss = batch_loss / (num_samples if num_samples > 0 else 1)
        self.log("val_loss", batch_loss, prog_bar=True, batch_size=len(sample))
        return batch_loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), **self.config.optimizer)
        return optimizer

    def compute_loss(
        self,
        addressee_hat,
        ai_addressee_hat,
        control_token_hat,
        addressee,
        ai_addressee,
        control_token,
    ):
        """
        addressee_hat: (1, num_speakers)
        control_token_hat: (1, 4)
        addressee: str - target addressee label
        control_token: str - target control token label
        """

        # Convert string labels to tensor indices
        addressee_idx = torch.tensor(
            self.addressee_to_idx[addressee], device=self.device
        ).unsqueeze(0)
        ai_addressee_idx = torch.tensor(
            self.ai_addressee_to_idx[ai_addressee], device=self.device
        ).unsqueeze(0)
        control_token_idx = torch.tensor(
            self.control_token_to_idx[control_token], device=self.device
        ).unsqueeze(0)

        addressee_loss = self.loss(addressee_hat, addressee_idx)
        ai_addressee_loss = self.loss(ai_addressee_hat, ai_addressee_idx)
        control_token_loss = self.loss(control_token_hat, control_token_idx)

        return (
            self.config.addressee_loss_weight * addressee_loss
            + self.config.addressee_loss_weight * ai_addressee_loss
            + self.config.control_token_loss_weight * control_token_loss
        )
