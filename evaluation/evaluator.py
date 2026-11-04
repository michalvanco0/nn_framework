from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


def evaluate(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> dict[str, list]:
    """
    Evaluate a classification model and return predictions, logits,
    probabilities and confidence scores.

    A batch may be ``(inputs, target)``, ``(input_a, input_b, target)``, or a
    mapping with a ``target``/``labels`` entry and model inputs in its other
    entries. Tensor, tuple/list, and mapping model inputs are moved recursively
    to ``device``. The model is expected to return class logits shaped
    ``(samples, classes)``.
    """

    def move_to_device(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, Mapping):
            return {key: move_to_device(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(move_to_device(item) for item in value)
        if isinstance(value, list):
            return [move_to_device(item) for item in value]
        return value

    def unpack_batch(batch: Any) -> tuple[Any, torch.Tensor]:
        if isinstance(batch, Mapping):
            target_key = next(
                (key for key in ("target", "targets", "label", "labels", "y") if key in batch),
                None,
            )
            if target_key is None:
                raise ValueError(
                    "Mapping batches must contain one of: target, targets, label, labels, y"
                )
            targets = batch[target_key]
            inputs = {key: value for key, value in batch.items() if key != target_key}
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            targets = batch[-1]
            input_parts = batch[:-1]
            inputs = input_parts[0] if len(input_parts) == 1 else input_parts
        else:
            raise ValueError("Each batch must include model input(s) and target labels")
        if not isinstance(targets, torch.Tensor):
            targets = torch.as_tensor(targets)
        return move_to_device(inputs), targets.to(device)

    def call_model(inputs: Any) -> torch.Tensor:
        if isinstance(inputs, Mapping):
            return model(**inputs)
        if isinstance(inputs, (tuple, list)):
            return model(*inputs)
        return model(inputs)

    model.eval()

    y_true = []
    y_pred = []
    logits = []
    probabilities = []
    confidence = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = unpack_batch(batch)
            outputs = call_model(inputs)
            if outputs.ndim != 2:
                raise ValueError(
                    "Classification models must return logits shaped "
                    "(samples, classes)"
                )

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
