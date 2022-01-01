from torchmetrics.functional.text import wer
from numpy import mean


def wer_metric(list_of_predictions):
    wer_list = list()
    for i in list_of_predictions:
        wer_item = wer(i[0], i[1])
        wer_list.append(wer_item.item())
    return mean(wer_list)
