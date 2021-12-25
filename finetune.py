import torch
import flash
from flash.audio import SpeechRecognition
from core.aug_transformer import CallsSpeechInputTransform, PredictCallsSpeechInputTransform, BaseSpeechInputTransform
from core.config import CONFIG
from pytorch_lightning import seed_everything
import argparse
from loguru import logger


def run(train_file, val_file, test_file, predict_file, fast_dev_run):

    data_module = SpeechRecognitionData.from_csv(
        input_fields="input",
        target_fields="target",
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
        predict_file=predict_file,
        batch_size=CONFIG.get("batch_size", None),
        num_workers=CONFIG.get("num_workers", None),
        sampling_rate=None,
        pin_memory=True,
        train_transform=CallsSpeechInputTransform,
        val_transform=CallsSpeechInputTransform,
        test_transform=BaseSpeechInputTransform,
        predict_transform=PredictCallsSpeechInputTransform,
    )
    if fast_dev_run:
        logger.info(next(iter(data_module.train_dataloader())))
    else:
        model = SpeechRecognition(
            backbone=CONFIG.get("model", None),
        )

        trainer = flash.Trainer(
            auto_lr_find=True,
            gpus=torch.cuda.device_count(),
            precision=CONFIG.get("precision", None),
            max_epochs=CONFIG.get("max_epochs", None),
            check_val_every_n_epoch=1,
            fast_dev_run=fast_dev_run,
        )

        trainer.finetune(model, datamodule=data_module, strategy="no_freeze")
        predictions = trainer.predict(model, datamodule=data_module)
        logger.info(predictions)


if __name__ == '__main__':
    seed_everything(42)

    parser = argparse.ArgumentParser(description="ArgumentParser script")
    parser.add_argument("-train_file", dest="train_file_path", required=True, type=str)
    parser.add_argument("-test_file", dest="test_file_path", default='default', type=str)
    parser.add_argument("-val_file", dest="val_file_path", required=True, type=str)
    parser.add_argument("-predict_file", dest="predict_file_path", required=True, type=str)
    parser.add_argument("-fast_dev_run", dest="fast_dev_run", default=False, type=bool)

    args = parser.parse_args()
    logger.info(args)

    import os
    logger.info(os.path.dirname(os.path.abspath(__file__)))

    run(
        train_file=args.train_file_path,
        val_file=args.test_file_path,
        test_file=args.val_file_path,
        predict_file=args.predict_file_path,
        fast_dev_run=int(args.fast_dev_run),
    )