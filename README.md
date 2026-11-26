# NN evaluation

The current evaluation stage uses the existing CIFAR-10 ResNet-18 and Adult
TabularMLP `.pt` checkpoints. It performs inference and writes one report set
per model.

## Metrics from `Prehľad`

`evaluation/base_metrics.py` calculates the applicable metrics listed in the
overview table.

For binary classification it reports accuracy, precision, recall/sensitivity,
specificity, negative predictive value, F1, balanced accuracy, Matthews
correlation coefficient, Youden's statistic, AUROC, and AUPRC. AUPRC is
calculated as average precision.

For multiclass classification it reports multiclass accuracy; precision, recall
and F1 with macro, micro and weighted averaging; MAE, MSE, RMSE and Pearson
correlation between the one-hot target matrix and the predicted probability
matrix; macro Dice, macro intersection over union/Jaccard and macro volume
similarity (`1 - |true count - predicted count| / (true count + predicted
count)`, averaged over classes). Hausdorff distance is marked not applicable
because CIFAR-10 and Adult classification examples do not provide coordinate
point sets.

`evaluation/calibration.py` calculates ECE and ACE and saves the calibration
diagrams used in the thesis: a class-wise reliability diagram, a top-label ECE
diagram, and an ACE diagram for one selected class.

`evaluation/fairness.py` calculates equal opportunity and equalized odds for
every input feature in the Adult checkpoint. Categorical features are grouped
by their observed values. Numeric feature cut points are training-set
quantiles (five groups where unique values allow); those same cut points are
applied to the test set. This includes every model input column uniformly.
Adult `income` is the prediction target rather than a model input, so `>50K` is
the positive outcome used in the rates and is not treated as a grouping
feature. CIFAR-10 has no corresponding input group attributes, so its fairness
result is marked not applicable.

## Evaluate existing checkpoints

The defaults point to the checkpoint files in `trainedModels/`:

```powershell
python -m experiments.evaluate_models --model both --data-dir data --output-dir results/evaluation
```

Run one model at a time or supply different checkpoint paths:

```powershell
python -m experiments.evaluate_models --model cifar10 --cifar-checkpoint trainedModels/resnet18_cifar10.pt
python -m experiments.evaluate_models --model adult --adult-checkpoint trainedModels/tabular_mlp.pt
```

CIFAR-10 is evaluated on its test split. Adult uses the official test split.
If the Adult checkpoint has a neighboring `preprocessor.joblib`, the evaluator
loads it. Otherwise it reconstructs preprocessing from the official training
rows using seed 42 by default and checks that the resulting input dimension
matches the checkpoint before inference.

Each model writes these files under its own output directory:

```text
base_metrics.json
calibration.json
fairness.json
classwise_reliability_diagram.pdf
ece_reliability_diagram.pdf
comparison_class_ace_reliability_diagram.pdf
```

## Code map

| File | Responsibility |
|---|---|
| `experiments/evaluate_models.py` | Loads both existing checkpoints, prepares test inputs, runs inference and saves reports. |
| `evaluation/evaluator.py` | Collects targets, predictions, probabilities and confidence from model inference. |
| `evaluation/base_metrics.py` | Implements the binary and multiclass metrics from `Prehľad`. |
| `evaluation/calibration.py` | Implements ECE, ACE and their thesis diagrams. |
| `evaluation/fairness.py` | Groups all Adult model input features and calculates equal opportunity and equalized odds. |
| `datasets/cifar10.py`, `datasets/adult_income.py` | Loads the two evaluation datasets. |
| `models/resnet.py`, `models/tabular_mlp.py` | Defines model architectures matching the checkpoints. |

The existing training scripts are kept in their original state. This evaluation
workflow only loads the saved checkpoints.
