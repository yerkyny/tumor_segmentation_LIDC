import pandas as pd
import warnings
import click
from loguru import logger
from utils import (load_yaml, seed_everything)
import pytorch_lightning as pl
import importlib
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

warnings.filterwarnings("ignore")

@click.command()
@click.argument("experiment_cfg", type=click.Path())
def main(experiment_cfg):
    # ---- Load Config ----
    config = load_yaml(experiment_cfg)
    logger.info(f"Config {experiment_cfg} has loaded")

    # --- Seed ----
    seed_everything(config["SEED"])
    logger.info("Seed has used")

    # ---- Read Csv ----
    data = pd.read_csv(config["FILE"]["TRAIN"])
    logger.info("Shape of the data: " + str(data.shape))
    logger.info("Data has loaded")

    testing = len(config["FILE"]["TEST"]) > 0
    if testing:
        test_data = pd.read_csv(config["FILE"]["TEST"])
        logger.info("Shape of the data: " + str(test_data.shape))
        logger.info("Test Data has loaded")

    # ---- Create Dataset -----
    module = importlib.import_module(config["DATASET"]["PY"])
    DATASET = getattr(module, config["DATASET"]["CLASS"])
    val_fold = config["FILE"]["FOLD"]

    train_dataset = DATASET(
        data[data.fold != val_fold].to_dict("records"), config, stage="train"
    )
    val_dataset = DATASET(
        data[data.fold == val_fold].to_dict("records"), config, stage="val"
    )
    if testing:
        test_dataset = DATASET(test_data.to_dict("records"), config, stage="test")
    logger.info("Datasets has created")

    # ---- Create Dataloader ----
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["DATALOADER"]["BATCH_SIZE"],
        shuffle=True,
        num_workers=config["DATALOADER"]["NUM_WORKERS"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["DATALOADER"]["NUM_WORKERS"],
    )
    test_loader = None
    if testing:
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config["DATALOADER"]["NUM_WORKERS"],
        )
    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    logger.info("Dataloader has created")

    # ---- Tensor Dims Check ----
    for batch in train_loader:
        x, y = batch
        print("Train data dimensions:", x.shape, y.shape)
        break

    for batch in val_loader:
        x, y = batch
        print("Validation data dimensions:", x.shape, y.shape)
        break

    # ---- Create Model ----
    module = importlib.import_module(config["MODEL"]["PY"])
    try:
        model = getattr(module, config["MODEL"]["ARCH"])(**config["MODEL"]["ARGS"])
    except:
        logger.info("NO ARGS")
        model = getattr(module, config["MODEL"]["ARCH"])()
    logger.info("Model has created")

    # ---- Create Loss ----
    module = importlib.import_module(config["CRITERION"]["PY"])
    try:
        criterion = getattr(module, config["CRITERION"]["CLASS"])(
            **config["CRITERION"]["ARGS"]
        )
    except:
        logger.info("LOSS NO ARGS")
        criterion = getattr(module, config["CRITERION"]["CLASS"])()
    logger.info("Loss has created")

    # ---- Create Optimizer ----
    module = importlib.import_module(config["OPTIMIZER"]["PY"])
    optimizer = getattr(module, config["OPTIMIZER"]["CLASS"])(
        model.parameters(), **config["OPTIMIZER"]["ARGS"]
    )
    logger.info("Optimizer has created")

    # ---- Create Scheduler ----
    module = importlib.import_module(config["SCHEDULER"]["PY"])
    scheduler = getattr(module, config["SCHEDULER"]["CLASS"])(
        optimizer, **config["SCHEDULER"]["ARGS"]
    )
    logger.info("Scheduler has created")

    # ---- Early Stopping ----
    callbacks = []
    if config["EARLY_STOPPING"]["ENABLE"]:
        early_stop_callback = EarlyStopping(**config["EARLY_STOPPING"]["ARGS"])
        callbacks.append(early_stop_callback)
        logger.info("Early stopping has used")

    # ---- Checkpoint ----
    checkpoint_callback = ModelCheckpoint(**config["CHECKPOINT"]["ARGS"])
    callbacks.append(checkpoint_callback)
    logger.info("Checkpoint has created")

    # ---- Learner ----
    module = importlib.import_module(config["LEARNER"]["PY"])
    Learner = getattr(module, config["LEARNER"]["CLASS"])
    lightning_model = Learner(
        dataloaders=dataloaders,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        loss=criterion
    )
    logger.info("Learner has created")

    # ---- Trainer ----
    trainer = pl.Trainer(
        gpus=config["TRAINER"]["GPUS"],
        max_epochs=config["TRAINER"]["EPOCHS"],
        num_sanity_val_steps=0,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        gradient_clip_val=config["TRAINER"]["GRAD_CLIP"],
        accumulate_grad_batches=config["TRAINER"]["GRAD_ACC"],
        precision=16,
        callbacks=callbacks,
    )
    logger.info("Trainer has created")

    # ---- TRAIN ----
    trainer.fit(lightning_model)
    logger.info("Training has started")

    # ---- TEST ----
    if testing:
        trainer.test(ckpt_path="best")
    logger.info("Testing has started")

# ---- RUN ----
if __name__ == "__main__":
    main()
    logger.info("Done")