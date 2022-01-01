from pytorch_lightning.callbacks import Callback, BasePredictionWriter
from typing import Any, Optional, Sequence


class CheckWerPredictBatch(BasePredictionWriter):
    def __init__(self, list_of_predictions: list):
        super().__init__()
        self.list_of_predictions = list_of_predictions

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        predicted_text = pl_module.output_transform._tokenizer.batch_decode(
            batch["labels"]
        )
        clean_predicted_text = [text.replace("<unk>", "") for text in predicted_text]
        for text_pair in zip(clean_predicted_text, outputs):
            self.list_of_predictions.append(text_pair)
