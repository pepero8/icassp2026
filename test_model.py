import torch
import pytorch_lightning as L
from omegaconf import OmegaConf

# from unittest.mock import Mock
from torch.utils.data import DataLoader

# from model import SAASRControl
from model_lightning import LitSAASRControl
from dataset import DummyDataset, collate_fn


def test_fast_dev_run():
    """quick sanity check"""

    config_path = "./config.yaml"

    config = OmegaConf.load(config_path)

    model = LitSAASRControl(config)

    # # Mock the model
    # def mock_forward(chunk):
    #     return (torch.randn(1, 3), torch.randn(1, 4))

    # model.model = Mock(spec=SAASRControl)
    # model.model.forward = mock_forward
    # model.model.reset_dialog_memory = Mock()

    # Create minimal dataset
    dataset = DummyDataset(num_samples=5)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    # Fast dev run - runs 1 batch of train, val, test
    trainer = L.Trainer(fast_dev_run=1, logger=False, accelerator="cpu")

    print("Running fast dev run...")
    trainer.fit(model, loader, loader)
    print("Fast dev run completed successfully!")


if __name__ == "__main__":
    test_fast_dev_run()


# python3 test_model.py
