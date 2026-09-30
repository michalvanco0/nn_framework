from typing import Dict

import torch
import torch.nn.functional as F


def evaluate(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> Dict[str, list]:
    """
    Evaluate a classification model and return predictions,
    logits, probabilities and confidence scores.
    """

    model.eval()

    y_true = []
    y_pred = []
    logits = []
    probabilities = []
    confidence = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            conf = torch.max(probs, dim=1).values

            y_true.extend(targets.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            logits.extend(outputs.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
            confidence.extend(conf.cpu().tolist())

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "logits": logits,
        "probabilities": probabilities,
        "confidence": confidence,
    }