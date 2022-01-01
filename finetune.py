import argparse
import os
import warnings

import flash
import pandas as pd
import torch
import wandb
from flash.audio import SpeechRecognition, SpeechRecognitionData
from loguru import logger
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

from core.aug_transformer import (BaseSpeechInputTransform,
                                  CallsSpeechInputTransform,
                                  PredictCallsSpeechInputTransform,
                                  aug_transfrms_cfg)
from core.callbacks import CheckWerPredictBatch
from core.config import CFG
from core.metrics import wer_metric

warnings.filterwarnings("ignore")


def run(train_file, val_file, test_file, predict_file, fast_dev_run):

    data_module = SpeechRecognitionData.from_csv(
        input_fields="input",
        target_fields="target",
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
        predict_file=predict_file,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        sampling_rate=None,
        pin_memory=CFG.pin_memory,
        train_transform=BaseSpeechInputTransform,
        val_transform=BaseSpeechInputTransform,
        test_transform=BaseSpeechInputTransform,
        predict_transform=BaseSpeechInputTransform,
    )

    wandb_logger = WandbLogger(project="asr_calls")
    model = SpeechRecognition(
        backbone=CFG.backbone,
    )

    list_of_predictions = list()

    trainer = flash.Trainer(
        auto_lr_find=True,
        gpus=torch.cuda.device_count(),
        precision=CFG.precision,
        max_epochs=CFG.max_epochs,
        check_val_every_n_epoch=1,
        callbacks=[
            CheckWerPredictBatch(list_of_predictions),
        ],
        logger=wandb_logger,
        fast_dev_run=int(fast_dev_run),
    )
    trainer.finetune(model, datamodule=data_module, strategy=CFG.strategy)

    predictions = trainer.predict(model, datamodule=data_module)

    wer_value = wer_metric(list_of_predictions)
    logger.info(f"WER -> {wer_value}")
    trainer.logger.experiment.log(dict(wer=wer_value))
    dataframe = pd.DataFrame(list_of_predictions, columns=["target", "predicted"])
    trainer.logger.log_table(key="asr_texts_compare", dataframe=dataframe)

    trainer.logger.experiment.config.update(aug_transfrms_cfg(data_module))
    trainer.logger.experiment.config.update(CFG._export_as_dict())
    wandb.finish(exit_code=0)


if __name__ == "__main__":
    seed_everything(42)

    parser = argparse.ArgumentParser(description="ArgumentParser script")
    parser.add_argument(
        "-train_file", dest="train_file_path", default=CFG.train_file, type=str
    )
    parser.add_argument(
        "-test_file", dest="test_file_path", default=CFG.test_file, type=str
    )
    parser.add_argument(
        "-val_file", dest="val_file_path", default=CFG.val_file, type=str
    )
    parser.add_argument(
        "-predict_file", dest="predict_file_path", default=CFG.predict_file, type=str
    )
    parser.add_argument(
        "-fast_dev_run",
        dest="fast_dev_run",
        default=CFG.fast_dev_run,
        type=bool,
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()
    logger.info(args)

    logger.info(os.path.dirname(os.path.abspath(__file__)))

    run(
        train_file=args.train_file_path,
        val_file=args.test_file_path,
        test_file=args.val_file_path,
        predict_file=args.predict_file_path,
        fast_dev_run=args.fast_dev_run,
    )
