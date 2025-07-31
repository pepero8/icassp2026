import torch
from torch import nn

# import pytorch_lightning as L
import lightning.pytorch as L

from dataset import TrainBatch, TestBatch
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
            self.ai_addressee_labels = [
                "Speaker_A",
                "Speaker_B",
                "Speaker_C",
                "NA",
                "All",
            ]
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
                "All",
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
        batch_addr_loss = torch.tensor(0.0, device=self.device)
        batch_aiaddr_loss = torch.tensor(0.0, device=self.device)
        batch_ctrl_loss = torch.tensor(0.0, device=self.device)

        num_samples = len(sample)
        for chunk in sample:
            addressee, ai_addressee, control_token = self.model(
                chunk,
                self.addressee_to_idx,
                self.ai_addressee_to_idx,
                self.ai_addressee_labels,
                self.control_token_to_idx,
                mode="train",
            )
            # addressee, ai_addressee, control_token = self.model(chunk)
            try:
                addressee_loss, ai_addressee_loss, control_token_loss = (
                    self.compute_loss(
                        addressee,
                        ai_addressee,
                        control_token,
                        chunk.addressee,
                        chunk.ai_addressee,
                        chunk.control_token,
                    )
                )

            except Exception as e:
                print(f"Error in loss calculation in train step: {e}")
                num_samples -= 1
                continue

            batch_addr_loss = batch_addr_loss + addressee_loss
            batch_aiaddr_loss = batch_aiaddr_loss + ai_addressee_loss
            batch_ctrl_loss = batch_ctrl_loss + control_token_loss

            # batch_loss = batch_loss + loss

        # batch_loss = batch_loss / (num_samples if num_samples > 0 else 1)
        batch_addr_loss = batch_addr_loss / (num_samples if num_samples > 0 else 1)
        batch_aiaddr_loss = batch_aiaddr_loss / (num_samples if num_samples > 0 else 1)
        batch_ctrl_loss = batch_ctrl_loss / (num_samples if num_samples > 0 else 1)
        batch_loss = (
            self.config.addressee_loss_weight * batch_addr_loss
            + self.config.ai_addressee_loss_weight * batch_aiaddr_loss
            + self.config.control_token_loss_weight * batch_ctrl_loss
        )

        assert not torch.isnan(
            batch_loss
        ).any(), f"Loss is NaN, batch_idx: {batch_idx}"  # NOTE: 어떨땐 nan이 뜨고 어떨땐 안뜸..

        self.log("train_loss", batch_loss, prog_bar=True, batch_size=len(sample))
        self.log("addr_loss", batch_addr_loss, prog_bar=True, batch_size=len(sample))
        self.log(
            "aiaddr_loss", batch_aiaddr_loss, prog_bar=True, batch_size=len(sample)
        )
        self.log("ctrl_loss", batch_ctrl_loss, prog_bar=True, batch_size=len(sample))

        return batch_loss

    def validation_step(self, batch: TrainBatch, batch_idx):
        # ! dialogue memory가 남아있는 상태에서 validation을 시작하면 안됨
        sample = (
            batch.sample
        )  # list of 256 Chunk instances(can be <64 if last chunk set of dialogue)

        if batch.reset_dialog_memory:
            self.model.reset_dialog_memory()

        batch_loss = torch.tensor(0.0, device=self.device)
        batch_addr_loss = torch.tensor(0.0, device=self.device)
        batch_aiaddr_loss = torch.tensor(0.0, device=self.device)
        batch_ctrl_loss = torch.tensor(0.0, device=self.device)

        batch_pred_addressee = 0
        batch_pred_ai_addressee = 0
        # batch_pred_control_token = 0
        batch_pred_control_token = [0] * len(self.control_token_labels)
        batch_pred_ai_addressee = [0] * len(self.ai_addressee_labels)
        batch_control_token_label_num = [0] * len(self.control_token_labels)
        batch_ai_addressee_label_num = [0] * len(self.ai_addressee_labels)

        # num_samples = len(sample)
        num_samples = 0  # > count number of samples that have loss calculated
        # prev_speaker = None
        for i, chunk in enumerate(sample):
            speaker = chunk.tape.split()[0].strip("<>")
            next_speaker = (
                sample[i + 1].tape.split()[0].strip("<>")
                if i + 1 < len(sample)
                else speaker
            )

            addressee, ai_addressee, control_token = self.model(
                chunk,
                self.addressee_to_idx,
                self.ai_addressee_to_idx,
                self.ai_addressee_labels,
                self.control_token_to_idx,
                mode="valid",
            )

            # > Calculate loss only at the end of each speaker's turn or when it is assistant's turn
            if (
                # prev_speaker != speaker
                speaker
                != next_speaker
            ):  # NOTE: it doesn't handle the case where last chunk of a batch is the end of the speaker's turn
                # prev_speaker = speaker
                calc_loss = True
            elif chunk.tape.count("Speaker") > 1 or chunk.control_token != "C.LISTEN":
                calc_loss = True
            else:
                calc_loss = False

            # addressee, ai_addressee, control_token = self.model(chunk)

            if not calc_loss:
                continue

            num_samples += 1

            try:
                addressee_loss, ai_addressee_loss, control_token_loss = (
                    self.compute_loss(
                        addressee,
                        ai_addressee,
                        control_token,
                        chunk.addressee,
                        chunk.ai_addressee,
                        chunk.control_token,
                    )
                )
                pred_addressee, pred_ai_addressee_list, pred_control_token_list = (
                    self.check_predictions(
                        addressee,
                        ai_addressee,
                        control_token,
                        chunk.addressee,
                        chunk.ai_addressee,
                        chunk.control_token,
                        batch_control_token_label_num,
                        batch_ai_addressee_label_num,
                    )
                )
            except Exception as e:
                print(f"Error in loss calculation in validation step: {e}")
                # num_samples -= 1
                continue

            # batch_loss = batch_loss + loss
            batch_addr_loss = batch_addr_loss + addressee_loss
            batch_aiaddr_loss = batch_aiaddr_loss + ai_addressee_loss
            batch_ctrl_loss = batch_ctrl_loss + control_token_loss

            batch_pred_addressee = batch_pred_addressee + pred_addressee
            # batch_pred_ai_addressee = batch_pred_ai_addressee + pred_ai_addressee
            batch_pred_ai_addressee = [
                a + b for a, b in zip(batch_pred_ai_addressee, pred_ai_addressee_list)
            ]
            batch_pred_control_token = [
                a + b for a, b in zip(batch_pred_control_token, pred_control_token_list)
            ]

        # batch_loss = batch_loss / (num_samples if num_samples > 0 else 1)
        batch_addr_loss = batch_addr_loss / (num_samples if num_samples > 0 else 1)
        batch_aiaddr_loss = batch_aiaddr_loss / (num_samples if num_samples > 0 else 1)
        batch_ctrl_loss = batch_ctrl_loss / (num_samples if num_samples > 0 else 1)
        batch_loss = (
            self.config.addressee_loss_weight * batch_addr_loss
            + self.config.ai_addressee_loss_weight * batch_aiaddr_loss
            + self.config.control_token_loss_weight * batch_ctrl_loss
        )

        batch_acc_addressee = batch_pred_addressee / (
            num_samples if num_samples > 0 else 1
        )
        # batch_acc_ai_addressee = batch_pred_ai_addressee / (
        #     num_samples if num_samples > 0 else 1
        # )
        batch_acc_ai_addressee = [
            ai_addressee
            / (
                batch_ai_addressee_label_num[i]
                if batch_ai_addressee_label_num[i] > 0
                else 1
            )
            for i, ai_addressee in enumerate(batch_pred_ai_addressee)
        ]
        # batch_acc_control_token = batch_pred_control_token / (num_samples if num_samples > 0 else 1)
        batch_acc_control_token = [
            ctrl
            / (
                batch_control_token_label_num[i]
                if batch_control_token_label_num[i] > 0
                else 1
            )
            for i, ctrl in enumerate(batch_pred_control_token)
        ]

        self.log("val_loss", batch_loss, prog_bar=True, batch_size=len(sample))
        self.log(
            "val_acc_addressee",
            batch_acc_addressee,
            prog_bar=True,
            batch_size=len(sample),
        )
        # self.log(
        #     "val_acc_ai_addressee",
        #     batch_acc_ai_addressee,
        #     prog_bar=True,
        #     batch_size=len(sample),
        # )
        for i, label in enumerate(self.ai_addressee_labels):
            self.log(
                f"val_acc_{label}",
                batch_acc_ai_addressee[i],
                prog_bar=True,
                batch_size=len(sample),
            )
        # self.log("val_acc_control_token", batch_acc_control_token, prog_bar=True, batch_size=len(sample))
        self.log(
            f"val_acc_{self.control_token_labels[0]}",
            batch_acc_control_token[0],
            prog_bar=True,
            batch_size=len(sample),
        )
        self.log(
            f"val_acc_{self.control_token_labels[1]}",
            batch_acc_control_token[1],
            prog_bar=True,
            batch_size=len(sample),
        )
        self.log(
            f"val_acc_{self.control_token_labels[2]}",
            batch_acc_control_token[2],
            prog_bar=True,
            batch_size=len(sample),
        )
        self.log(
            f"val_acc_{self.control_token_labels[3]}",
            batch_acc_control_token[3],
            prog_bar=True,
            batch_size=len(sample),
        )

        return batch_loss

    def test_step(self, batch: TestBatch, batch_idx):
        task_name = batch.task_name
        sample = batch.sample  # list of all Chunk instances within single dialogue

        if batch.reset_dialog_memory:  # always true
            self.model.reset_dialog_memory()

        batch_loss = torch.tensor(0.0, device=self.device)
        batch_addr_loss = torch.tensor(0.0, device=self.device)
        batch_aiaddr_loss = torch.tensor(0.0, device=self.device)
        batch_ctrl_loss = torch.tensor(0.0, device=self.device)

        batch_pred_addressee = 0
        batch_pred_ai_addressee = 0
        batch_pred_control_token = [0] * len(self.control_token_labels)
        batch_pred_ai_addressee = [0] * len(self.ai_addressee_labels)
        batch_control_token_label_num = [0] * len(self.control_token_labels)
        batch_ai_addressee_label_num = [0] * len(self.ai_addressee_labels)

        num_samples = 0  # > count number of samples that have loss calculated
        for i, chunk in enumerate(sample):
            # speaker = chunk.tape.split()[0].strip("<>")
            # next_speaker = (
            #     sample[i + 1].tape.split()[0].strip("<>")
            #     if i + 1 < len(sample)
            #     else speaker
            # )

            addressee, ai_addressee, control_token = self.model(
                chunk,
                self.addressee_to_idx,
                self.ai_addressee_to_idx,
                self.ai_addressee_labels,
                self.control_token_to_idx,
                mode="valid",
            )

            # # > Calculate loss only at the end of each speaker's turn or when it is assistant's turn
            # if (
            #     # prev_speaker != speaker
            #     speaker
            #     != next_speaker
            # ):  # NOTE: it doesn't handle the case where last chunk of a batch is the end of the speaker's turn
            #     # prev_speaker = speaker
            #     calc_loss = True
            # elif chunk.tape.count("Speaker") > 1 or chunk.control_token != "C.LISTEN":
            #     calc_loss = True
            # else:
            #     calc_loss = False

            # if not calc_loss:
            #     continue
            if not chunk.calc_loss:
                continue

            num_samples += 1

            try:
                addressee_loss, ai_addressee_loss, control_token_loss = (
                    self.compute_loss(
                        addressee,
                        ai_addressee,
                        control_token,
                        chunk.addressee,
                        chunk.ai_addressee,
                        chunk.control_token,
                    )
                )
                pred_addressee, pred_ai_addressee_list, pred_control_token_list = (
                    self.check_predictions(
                        addressee,
                        ai_addressee,
                        control_token,
                        chunk.addressee,
                        chunk.ai_addressee,
                        chunk.control_token,
                        batch_control_token_label_num,
                        batch_ai_addressee_label_num,
                    )
                )
            except Exception as e:
                print(f"Error in loss calculation in validation step: {e}")
                # num_samples -= 1
                continue

            # batch_loss = batch_loss + loss
            batch_addr_loss = batch_addr_loss + addressee_loss
            batch_aiaddr_loss = batch_aiaddr_loss + ai_addressee_loss
            batch_ctrl_loss = batch_ctrl_loss + control_token_loss

            batch_pred_addressee = batch_pred_addressee + pred_addressee
            # batch_pred_ai_addressee = batch_pred_ai_addressee + pred_ai_addressee
            batch_pred_ai_addressee = [
                a + b for a, b in zip(batch_pred_ai_addressee, pred_ai_addressee_list)
            ]
            batch_pred_control_token = [
                a + b for a, b in zip(batch_pred_control_token, pred_control_token_list)
            ]

        # batch_loss = batch_loss / (num_samples if num_samples > 0 else 1)
        batch_addr_loss = batch_addr_loss / (num_samples if num_samples > 0 else 1)
        batch_aiaddr_loss = batch_aiaddr_loss / (num_samples if num_samples > 0 else 1)
        batch_ctrl_loss = batch_ctrl_loss / (num_samples if num_samples > 0 else 1)
        batch_loss = (
            self.config.addressee_loss_weight * batch_addr_loss
            + self.config.ai_addressee_loss_weight * batch_aiaddr_loss
            + self.config.control_token_loss_weight * batch_ctrl_loss
        )

        batch_acc_addressee = batch_pred_addressee / (
            num_samples if num_samples > 0 else 1
        )
        # batch_acc_ai_addressee = batch_pred_ai_addressee / (
        #     num_samples if num_samples > 0 else 1
        # )
        batch_acc_ai_addressee = [
            ai_addressee
            / (
                batch_ai_addressee_label_num[i]
                if batch_ai_addressee_label_num[i] > 0
                else 1
            )
            for i, ai_addressee in enumerate(batch_pred_ai_addressee)
        ]
        # batch_acc_control_token = batch_pred_control_token / (num_samples if num_samples > 0 else 1)
        batch_acc_control_token = [
            ctrl
            / (
                batch_control_token_label_num[i]
                if batch_control_token_label_num[i] > 0
                else 1
            )
            for i, ctrl in enumerate(batch_pred_control_token)
        ]

        self.log("val_loss", batch_loss, prog_bar=True, batch_size=len(sample))
        self.log(
            "val_acc_addressee",
            batch_acc_addressee,
            prog_bar=True,
            batch_size=len(sample),
        )
        # self.log(
        #     "val_acc_ai_addressee",
        #     batch_acc_ai_addressee,
        #     prog_bar=True,
        #     batch_size=len(sample),
        # )
        for i, label in enumerate(self.ai_addressee_labels):
            self.log(
                f"val_acc_{label}",
                batch_acc_ai_addressee[i],
                prog_bar=True,
                batch_size=len(sample),
            )
        # self.log("val_acc_control_token", batch_acc_control_token, prog_bar=True, batch_size=len(sample))
        self.log(
            f"val_acc_{self.control_token_labels[0]}",
            batch_acc_control_token[0],
            prog_bar=True,
            batch_size=len(sample),
        )
        self.log(
            f"val_acc_{self.control_token_labels[1]}",
            batch_acc_control_token[1],
            prog_bar=True,
            batch_size=len(sample),
        )
        self.log(
            f"val_acc_{self.control_token_labels[2]}",
            batch_acc_control_token[2],
            prog_bar=True,
            batch_size=len(sample),
        )
        self.log(
            f"val_acc_{self.control_token_labels[3]}",
            batch_acc_control_token[3],
            prog_bar=True,
            batch_size=len(sample),
        )

        return batch_loss

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

        # return (
        #     self.config.addressee_loss_weight * addressee_loss
        #     + self.config.addressee_loss_weight * ai_addressee_loss
        #     + self.config.control_token_loss_weight * control_token_loss
        # )
        return addressee_loss, ai_addressee_loss, control_token_loss

    def check_predictions(
        self,
        addressee_hat,
        ai_addressee_hat,
        control_token_hat,
        addressee,
        ai_addressee,
        control_token,
        batch_control_token_label_num,
        batch_ai_addressee_label_num,
    ):
        """
        addressee_hat: (1, num_speakers)
        control_token_hat: (1, 4)
        addressee: str - target addressee label
        control_token: str - target control token label
        """

        # >> Get predictions for addressee, ai_addressee, and control_token
        addressee_idx = self.addressee_to_idx[addressee]
        addressee_hat_idx = torch.argmax(addressee_hat, dim=1).item()
        correct_addressee = 1 if addressee_idx == addressee_hat_idx else 0

        ai_addressee_idx = self.ai_addressee_to_idx[ai_addressee]
        ai_addressee_hat_idx = torch.argmax(ai_addressee_hat, dim=1).item()
        batch_ai_addressee_label_num[ai_addressee_hat_idx] += 1
        correct_ai_addressee = [0] * len(self.ai_addressee_labels)
        if ai_addressee_idx == ai_addressee_hat_idx:
            correct_ai_addressee[ai_addressee_idx] = 1
        # correct_ai_addressee = 1 if ai_addressee_idx == ai_addressee_hat_idx else 0

        control_token_idx = self.control_token_to_idx[control_token]
        control_token_hat_idx = torch.argmax(control_token_hat, dim=1).item()
        # correct_control_token = 1 if control_token_idx == control_token_hat_idx else 0
        batch_control_token_label_num[control_token_idx] += 1
        correct_control_token = [0] * len(self.control_token_labels)
        if control_token_idx == control_token_hat_idx:
            correct_control_token[control_token_idx] = 1

        return (
            correct_addressee,
            correct_ai_addressee,
            correct_control_token,
        )
