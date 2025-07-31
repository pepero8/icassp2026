from typing import Annotated
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
import lightning.pytorch as L
from omegaconf import OmegaConf
import typer
from datetime import datetime

from data_module import DataModule
from model_lightning import LitSAASRControl

from lightning.pytorch.loggers import TensorBoardLogger

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    cfg: Annotated[str, typer.Argument(help="Path to the config file")],
    test: Annotated[bool, typer.Option(help="Perform test on the model")] = False,
):

    config = OmegaConf.load(cfg)

    model = LitSAASRControl(config)
    data_module = DataModule(
        config, data_dir=config.data_dir, test_data_dir=config.test_data_dir
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=7,
        verbose=False,
        mode="min",
        check_finite=True,
        check_on_train_epoch_end=False,
    )

    # ddp_strategy = DDPStrategy()

    # trainer = L.Trainer(
    #     fast_dev_run=1,
    #     num_sanity_val_steps=1,
    #     max_epochs=1,
    #     default_root_dir=config.default_root_dir,  # Path to save checkpoints and logs
    #     callbacks=[early_stop_callback],
    # )  # ? for testing

    # > Generate version based on current timestamp
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # > Create a logger instance
    logger = TensorBoardLogger(
        config.default_root_dir, version=current_time, name="icassp2026"
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor="val_loss",
        mode="min",
        filename="{epoch:02d}-val_loss:{val_loss:.4f}",
    )  # 기본적으로 Trainer.log_dir에 저장됨

    trainer = L.Trainer(
        num_sanity_val_steps=1,
        max_epochs=10,
        # max_epochs=0,
        logger=logger,
        # default_root_dir=config.default_root_dir,  # Path to save checkpoints and logs
        callbacks=[early_stop_callback, checkpoint_callback],
        # strategy=ddp_strategy,
        check_val_every_n_epoch=1,
    )

    if not test:
        trainer.fit(model, data_module)
        # trainer.fit(model, ckpt_path="path/to/your/checkpoint.ckpt")  # resume training
        # trainer.test(
        #     model,
        #     data_module,
        #     # ckpt_path="/shared/NAS_SSD/jhl/icassp/experiments/icassp2026/20250730-080007/checkpoints/epoch=00-val_loss:val_loss=9.0101.ckpt",
        #     # ckpt_path="/shared/NAS_SSD/jhl/icassp/experiments/icassp2026/20250730-080007/checkpoints/epoch=01-val_loss:val_loss=10.1380.ckpt",
        #     # ckpt_path="/shared/NAS_SSD/jhl/icassp/experiments/icassp2026/20250730-080007/checkpoints/epoch=02-val_loss:val_loss=8.3589.ckpt",
        #     # ckpt_path="/shared/NAS_SSD/jhl/icassp/experiments/icassp2026/20250730-080007/checkpoints/epoch=05-val_loss:val_loss=10.1480.ckpt",
        #     ckpt_path="/shared/NAS_SSD/jhl/icassp/experiments/icassp2026/20250730-080007/checkpoints/epoch=07-val_loss:val_loss=10.7079.ckpt",
        # )
    else:
        trainer.test(
            model, datamodule=data_module, ckpt_path="best"
        )  # test with best checkpoint


if __name__ == "__main__":
    app()
    # typer.run(main)

# CUDA_VISIBLE_DEVICES=3 python train.py config.yaml
