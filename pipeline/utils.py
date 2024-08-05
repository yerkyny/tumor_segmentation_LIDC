import yaml
import numpy as np
import torch
from monai.metrics import (DiceMetric, MeanIoU)

def load_yaml(file_name):
    with open(file_name, "r") as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    return config

def seed_everything(seed) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_scores(y, pred) -> dict:
    dice = DiceMetric(include_background=False)
    iou = MeanIoU(include_background=False)
    scores = {"dice": -1,  "iou": -1}

    pred = torch.sigmoid(pred)
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    best_dice, best_iou = [], []
    for th in thresholds:
        y_pred = (pred > th).to(torch.int32)

        if 1 in y:
            dice(y_pred, y)
            best_dice.append(dice.aggregate().item())

            iou(y_pred, y)
            best_iou.append(iou.aggregate().item())
    if 1 in y:
        best_th = np.argmax(best_dice)
        y_pred = (pred > thresholds[best_th]).to(torch.int32)
    
        scores["dice"] = best_dice[best_th]
        scores["iou"] = best_iou[best_th]

    return scores, y_pred
