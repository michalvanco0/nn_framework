# NN Reliability Framework

NN Reliability Framework is a Python research project baseline for systematically evaluating neural network behavior and building toward trustworthy AI analysis.

## Current scope (baseline)

This repository currently provides a clean, reproducible baseline experiment:

- CIFAR-10 dataset loading with `torchvision`
- ResNet-18 training (adapted to 10 classes)
- Evaluation with accuracy, precision, recall, F1 score, and confusion matrix
- Model- and dataset-independent group fairness audit for classifier outputs
- Saved model checkpoint
- Saved structured JSON metrics
- Saved confusion matrix plot

## Technology

- Python 3.11+
- PyTorch
- torchvision
- NumPy
- scikit-learn
- matplotlib

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run training and evaluation

```bash
python main.py --epochs 1 --batch-size 128 --learning-rate 0.001 --seed 42
```

Optional arguments:

- `--data-dir` (default: `data`)
- `--results-dir` (default: `results`)
- `--num-workers` (default: `2`)

## Results and artifacts

Outputs are written to the configured results directory (default: `results/`):

- `resnet18_cifar10.pt` — model checkpoint
- `baseline_metrics.json` — structured evaluation metrics
- `confusion_matrix.png` — confusion matrix visualization

## Fairness audit

`evaluation.fairness.calculate_fairness_metrics` measures classifier performance
across caller-supplied groups. It only needs aligned ground-truth labels,
predicted labels, and one or more sensitive/group attributes. It does not load a
particular dataset, depend on a model architecture, change model predictions,
or assign a fair/unfair verdict. The same interface can be used with image,
text, tabular, or other classification models once their predictions and group
metadata have been collected in the same sample order.

```python
from pathlib import Path

from experiments.fairness import run_fairness_audit

report = run_fairness_audit(
    y_true=labels,
    y_pred=predictions,
    sensitive_features={
        "age_band": age_bands,
        "region": regions,
    },
    probabilities=class_probabilities,  # optional, shape (samples, classes)
    class_labels=class_names,  # order must match probability columns
    metadata={"model": "model-name", "dataset": "dataset-name", "split": "test"},
    output_path=Path("results/fairness.json"),
)
```

To try the audit with the repository's existing CIFAR-10 ResNet-18 checkpoint,
run this from the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m experiments.fairness_cifar10
```

This evaluates the test set and writes
`results/fairness_cifar10_superclasses.json`. The example groups CIFAR-10 classes
into animals and vehicles to demonstrate the interface. These groups are derived
from the target class, so this is only an exploratory class-superclass performance
profile, not a demographic fairness audit. For demographic analysis, pass real
group attributes aligned with the evaluated samples to `run_fairness_audit`.
The command also saves a group-metrics chart and a binary ROC curve for the
collapsed vehicle-versus-animal prediction, each as PNG, PDF, and SVG beside the
JSON file. This ROC curve measures superclass discrimination; it is not a
comparison of ROC curves across demographic groups. Group ROC comparisons need
the same binary outcome represented in each group.

The report contains group-level values and between-group ranges for:

- Overall metrics plus per-group sample counts and true/predicted class
  distributions, to show the group base rates alongside model outcomes.
- Accuracy, balanced accuracy, macro precision, macro recall, and macro F1.
- Per-class selection rate, true/false positive and negative rates, positive
  and negative predictive values, and F1.
- Statistical parity/selection-rate gaps, equal-opportunity (TPR) gaps,
  equalized-odds components (TPR and FPR gaps), and predictive-parity gaps.
- When `reference_groups={"attribute": "group value"}` is explicitly supplied,
  signed statistical parity, equal-opportunity, FPR, predictive-parity, and
  negative-predictive-parity differences; disparate-impact ratios; and average
  odds difference relative to that reference group.
- With probabilities: per-group multiclass Brier score, log loss, top-label
  ECE, classwise ECE, and one-versus-rest ROC AUC, with gaps between groups.
  Pairwise ABROCA comparisons measure the area between group ROC curves across
  thresholds.
- With `accepted_mask` for selective classifiers: group coverage and accuracy
  among accepted predictions.

For multiclass tasks, rate-based metrics are computed one class versus all
others. Set `positive_label` to name the class of interest explicitly. For
two-dimensional probabilities, `class_labels` is required and must match the
column order; it is also needed when the evaluation split omits a model output
class. Binary probability vectors are also supported when `positive_label` is
specified. No reference group is selected automatically. Group attributes should
be categorical; bin continuous attributes such as age using an explicitly
documented policy before passing them. Missing group values are counted and
omitted from that attribute's comparisons; small group sizes are called out
because their estimates can be unstable. For an intersectional audit, pass an
additional composite categorical attribute such as `gender_age_band`.

When `accepted_mask` is supplied, classification rates and error metrics are
conditional on accepted samples, while coverage uses all samples. Probability
quality metrics use all samples. This makes abstention behavior visible
alongside the model's performance when it does make a prediction.

Optional `metadata` can record the model, dataset, evaluation split, and
intended use beside the measurements. The small-group warning threshold is
configurable and does not act as a fairness cutoff. The report does not
calculate confidence intervals or significance tests. ECE uses equal-width
probability bins; the bin count is included in the report.

There is deliberately no combined fairness score, threshold, or model ranking.
The report makes observed group differences visible so that model suitability
can be considered in the context of the intended use. The metrics do not
establish why a difference exists or whether it is acceptable. CIFAR-10 does
not provide demographic group attributes, so its existing evaluation cannot
produce demographic fairness results; differences among its object classes
remain class-wise performance results.
The report also lists dimensions not measured, including sensitive attributes
not supplied to the audit, causal explanations, and individual or
counterfactual fairness. The selected statistical criteria may disagree, so
their measurements are kept separate.

This implementation covers supervised classification. Individual and
counterfactual fairness require a task-specific similarity or causal model;
regression and ranking need their own task-specific metrics and are outside the
current interface.

### Research references

- Hardt, Price, and Srebro (2016), [Equality of Opportunity in Supervised
  Learning](https://papers.neurips.cc/paper_files/paper/2016/hash/6a9659feb1216f14f7384ba499518b38-Abstract.html).
- Verma and Rubin (2018), [Fairness Definitions
  Explained](https://doi.org/10.1145/3194770.3194776).
- Chouldechova (2017), [Fair Prediction with Disparate Impact](https://doi.org/10.1089/BIG.2016.0047).
- Kleinberg, Mullainathan, and Raghavan (2017), [Inherent Trade-Offs in the
  Fair Determination of Risk Scores](https://arxiv.org/abs/1609.05807).
- Pleiss et al. (2017), [On Fairness and
  Calibration](https://papers.neurips.cc/paper/7151-on-fairness-and-calibration.pdf).
- Guo et al. (2017), [On Calibration of Modern Neural
  Networks](https://proceedings.mlr.press/v70/guo17a.html).
- Gardner, Brooks, and Baker (2019), [Evaluating the Fairness of Predictive
  Student Models Through Slicing Analysis](https://doi.org/10.1145/3303772.3303791).

## Planned future modules (not implemented yet)

- Robustness analysis
- Calibration analysis
- Uncertainty estimation
- Explainability/XAI
- Distribution shift analysis
- Final reliability score
- VizWiz evaluation
- Adversarial attacks
- Automated final report generation
