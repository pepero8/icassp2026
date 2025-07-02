from typing import Annotated
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
import lightning.pytorch as L
from omegaconf import OmegaConf
import typer

from data_module import DataModule
from model_lightning import LitSAASRControl

app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
def main(
    cfg: Annotated[str, typer.Argument(help="Path to the config file")],
):

    config = OmegaConf.load(cfg)

    model = LitSAASRControl(config)
    data_module = DataModule(
        config, data_dir=config.data_dir, test_data_dir=config.test_data_dir
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode="min",
        check_finite=True,
        check_on_train_epoch_end=False,
    )

    # ddp_strategy = DDPStrategy()

    # trainer = L.Trainer(
    #     fast_dev_run=1,
    #     num_sanity_val_steps=1,
    #     max_epochs=10,
    #     default_root_dir=config.default_root_dir,  # Path to save checkpoints and logs
    #     callbacks=[early_stop_callback],
    #     strategy=ddp_strategy,
    # ) # ? for testing
    
    trainer = L.Trainer(
        num_sanity_val_steps=1,
        max_epochs=10,
        default_root_dir=config.default_root_dir,  # Path to save checkpoints and logs
        callbacks=[early_stop_callback],
        # strategy=ddp_strategy,
    )

    trainer.fit(model, data_module)
    # trainer.fit(model, ckpt_path="path/to/your/checkpoint.ckpt")  # resume training

    trainer.test(
        model, datamodule=data_module, ckpt_path="best"
    )  # test with best checkpoint


if __name__ == "__main__":
    app()
    # typer.run(main)

# CUDA_VISIBLE_DEVICES=0 python train.py config.yaml
