# NN Reliability Framework

NN Reliability Framework is a Python research project baseline for systematically evaluating neural network behavior and building toward trustworthy AI analysis.

## Current scope (baseline)

The repository contains both a CIFAR-10 image baseline and a reusable
classification and fairness workflow:

- CIFAR-10 dataset loading with `torchvision`
- ResNet-18 training (adapted to 10 classes)
- Tabular MLP training on UCI Adult or a user supplied CSV
- Shared evaluation for classifiers that return class logits
- Tabular accuracy, balanced accuracy, calibration, uncertainty, and group fairness reports
- Global permutation importance to interpret the effect of each input on held-out metrics
- Evaluation with accuracy, precision, recall, F1 score, and confusion matrix
- Model- and dataset-independent group fairness audit for classifier outputs
- Saved checkpoints, structured JSON metrics, predictions, and plots

## Technology

- Python 3.11+
- PyTorch
- torchvision
- pandas
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

## Tabular datasets and fairness

### Why Adult for the first experiment

The UCI Adult (Census Income) dataset is a practical first replacement for
CIFAR-10 in a fairness experiment. Its target is whether recorded annual income
is above $50,000, and it includes sex, race, and age, so group membership comes
from the data rather than being guessed from the class label. UCI lists 48,842
records, 14 input attributes, and missing values; the records were extracted
from 1994 Census data. [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult)
For a thesis, cite Becker and Kohavi (1996), DOI
[10.24432/C5XW20](https://doi.org/10.24432/C5XW20); UCI lists the dataset under
CC BY 4.0.

Adult is a well known, small benchmark with a supplied train/test split, which
makes the first end-to-end implementation easy to reproduce. It is not the
strongest choice for claims about today's population. Ding et al. document
limitations of Adult and introduce newer census-based tasks; Folktables' ACS
Income task is one possible follow-up for studying years and geographic
variation. That data is larger and needs an explicit state/year sampling
design, so this project starts with Adult and keeps the trainer open to other
CSV files. [Ding et al., *Retiring Adult*](https://papers.neurips.cc/paper_files/paper/2021/hash/32e54441e6382a7fbacbbbaf3c450059-Abstract.html),
[Folktables](https://github.com/socialfoundations/folktables)

The Adult run uses the official test split and draws validation data only from
the official training split. It uses all 14 original input attributes,
including age, race, sex, and `fnlwgt`. Age bands (17-29, 30-39, 40-49, 50-59,
60+) are created only for the group audit; the extra derived column is not fed
into the model. Full analysis audits every model input attribute and every
two-attribute intersection by default, including age bands. This is a search
for uneven model error across feature-defined groups; the framework does not
label any feature as sensitive or infer that a measured difference is ethnic
discrimination.

### Train Adult baseline

The script downloads the official UCI archive into `data/` the first time it
runs. For the current standalone training phase, use `--basic-only`: it trains
on all 14 original Adult inputs, selects a checkpoint by validation accuracy,
and reports ordinary test classification metrics. It does not run a group
fairness audit, calibration/uncertainty analysis, or feature interpretation.

```bash
python -m experiments.tabular_baseline --dataset adult --basic-only --epochs 30 --batch-size 256 --learning-rate 0.001 --seed 42 --output-dir results/adult_income_basic
```

This writes the model checkpoint, fitted preprocessing pipeline, `metrics.json`,
`predictions.csv`, and a confusion matrix. The full analysis mode remains
available by omitting `--basic-only` and can be used in a later step.

```bash
python -m experiments.tabular_baseline --dataset adult --epochs 30 --batch-size 256 --learning-rate 0.001 --seed 42
```

### Run it in Google Colab

The clone cell reads GitHub, so push the repository version containing these
changes before running it. Otherwise, upload a project archive with the new
files to Colab and change into its extracted folder. Select a GPU runtime if
you want GPU training, then run these cells:

```python
%cd /content
!git clone https://github.com/michalvanco0/nn_framework.git
%cd /content/nn_framework
!pip install -r requirements.txt
```

```python
!python -m experiments.tabular_baseline \
  --dataset adult \
  --basic-only \
  --epochs 30 \
  --batch-size 256 \
  --learning-rate 0.001 \
  --seed 42 \
  --output-dir results/adult_income
```

The dataset and artifacts are initially in the Colab VM. To keep results after
the session ends, mount Drive at any time during the session and copy the output
folder there:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!cp -r results/adult_income /content/drive/MyDrive/adult_income_results
```

In basic-only mode, artifacts include `tabular_mlp.pt`, `preprocessor.joblib`,
`metrics.json`, `predictions.csv`, and `confusion_matrix.png`. The saved
predictions can be used for a later audit. Full analysis mode additionally
writes `fairness.json`, calibration and uncertainty reports, permutation
importance, and group comparison charts.

Permutation importance shuffles one input at a time on the held-out test rows
and reports changes in accuracy, balanced accuracy, and log loss. A positive
balanced-accuracy decrease means performance fell after shuffling that input.
This describes the trained model's predictive reliance; it does not estimate a
causal effect, and correlated features can share or mask importance.

### Use another tabular classification dataset

Pass one CSV containing the target and input features. Numeric and categorical
features are inferred from the CSV, missing input values are imputed, and
scaling/encoding is fitted on training rows only.
The default split is stratified 70% train, 15% validation, and 15% test. Every
column except the target and explicit exclusions is used as a model input,
including the audited feature columns. If the CSV already has a split column with
`train`, `validation`/`val`, and `test` values, pass it with `--split-column`;
when the column has only `train` and `test`, validation is drawn from train.

```bash
python -m experiments.tabular_baseline \
  --csv data/my_dataset.csv \
  --target-column outcome \
  --drop-columns person_id \
  --positive-label yes \
  --epochs 30 \
  --output-dir results/my_dataset
```

The CSV must have one row per example and one column per value. Full analysis
audits all model input columns by default. Numeric features are split into up
to five quantile bins using cut points learned from training rows. Features with
many categorical levels retain their most frequent levels and group the rest
as `other`. Every pair of audit features is also evaluated as an intersection.
Use `--audit-columns` to choose a smaller set when needed; `--sensitive-columns`
remains as an alias. Regression and ranking are not supported by this
classification trainer.

## Fairness audit

`evaluation.fairness.calculate_fairness_metrics` measures classifier performance
across caller-supplied groups. It only needs aligned ground-truth labels,
predicted labels, and one or more feature groups. It does not load a
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
    audit_features={
        "age_band": age_bands,
        "region": regions,
    },
    probabilities=class_probabilities,  # optional, shape (samples, classes)
    class_labels=class_names,  # order must match probability columns
    metadata={"model": "model-name", "dataset": "dataset-name", "split": "test"},
    output_path=Path("results/fairness.json"),
)
```

For a console workflow independent of the model framework, export one CSV row
per evaluated example. Keep labels, predictions, feature values, and optional
probability columns on the same row. By default, the adapter uses every other
column as an audit feature and creates pairwise intersections; pass
`--group-columns` only when you want to limit the audit:

```powershell
.\.venv\Scripts\python.exe -m experiments.fairness_csv --input predictions.csv --label-column y_true --prediction-column y_pred --class-labels no yes --positive-label yes --probability-columns p_no p_yes --model-name my-model --dataset-name my-dataset --split-name test --output results/fairness.json
```

The class-label order must match the probability-column order. With no
probabilities, probability-quality, ABROCA, and ROC-curve results are omitted.
When scores are present, the adapter saves a separate group-conditional,
one-versus-rest ROC comparison for each class that has both outcomes in at least
two groups. In this CSV-only workflow, numeric bins are derived from the rows
being audited; the tabular training workflow instead learns cut points from its
training partition. ABROCA comparisons are omitted for an attribute with more
than 100 groups by default to keep pairwise curve comparisons bounded. This
adapter reads predictions from any model; it does not load or modify the model.

The generic function above works with predictions from any classification
model. CIFAR-10 itself has no demographic or other supplied group attributes,
so a fairness audit must not invent groups from its target classes. To audit the
repository's CIFAR-10 checkpoint, provide an external CSV with one row per test
example in exactly the test loader's order:

```powershell
.\.venv\Scripts\python.exe -m experiments.fairness_cifar10 --group-metadata data/cifar10_test_groups.csv --group-columns sex age_band
```

The CSV must contain the named group columns, and its rows must align one-to-one
with the CIFAR-10 test examples returned by the test loader. The command writes
`results/fairness_cifar10.json`, a group-metrics chart for each supplied
attribute, and group-conditional ROC comparisons when classes are represented
in multiple groups. If group metadata is unavailable, the model can still be
evaluated for ordinary classification performance, but no fairness comparison
can be reported. The earlier animals-versus-vehicles output was derived from
target classes and is only a class-composition profile; it must not be
interpreted as a demographic fairness result.

The report contains group-level values and between-group ranges for:

- Overall metrics plus per-group sample counts and true/predicted class
  distributions, to show the group base rates alongside model outcomes.
- Accuracy, balanced accuracy, macro precision, macro recall, and macro F1.
- Per-class selection rate, true/false positive and negative rates, positive
  and negative predictive values, and F1.
- Statistical parity/selection-rate gaps, equal-opportunity (TPR) gaps,
  equalized-odds components (TPR and FPR gaps), the maximum Equalized Odds
  gap, and predictive-parity gaps.
- When `reference_groups={"attribute": "group value"}` is explicitly supplied,
  signed statistical parity, equal-opportunity, FPR, predictive-parity, and
  negative-predictive-parity differences; disparate-impact ratios; and average
  odds difference relative to that reference group.
- With probabilities: per-group multiclass Brier score, log loss, and
  one-versus-rest ROC AUC, with gaps between groups. Pairwise ABROCA comparisons
  measure the area between group ROC curves across thresholds.
- With `accepted_mask` for selective classifiers: group coverage and accuracy
  among accepted predictions.

For multiclass tasks, rate-based metrics are computed one class versus all
others. Set `positive_label` to rank on the class of interest; otherwise the
ranking uses the largest Equalized Odds gap across classes. For
two-dimensional probabilities, `class_labels` is required and must match the
column order; it is also needed when the evaluation split omits a model output
class. Binary probability vectors are also supported when `positive_label` is
specified. No reference group is selected automatically. The tabular training
workflow bins numeric features using training-partition quantiles, preserves
categorical features, and automatically creates all pairwise feature
intersections. Other callers can pass their own precomputed groups. Missing
group values are counted and omitted from that attribute's comparisons; small
group sizes are called out because their estimates can be unstable.

When `accepted_mask` is supplied, classification rates and error metrics are
conditional on accepted samples, while coverage uses all samples. Probability
quality metrics use all samples. This makes abstention behavior visible
alongside the model's performance when it does make a prediction.

Optional `metadata` can record the model, dataset, evaluation split, and
intended use beside the measurements. The small-group warning threshold is
configurable and does not act as a fairness cutoff. The report does not
calculate confidence intervals or significance tests. Calibration error is
measured separately with ACE in the calibration module; the fairness report
does not calculate ECE.

The report includes a descriptive disparity ranking. Its score is the largest
Equalized Odds rate gap (the maximum of the TPR and FPR max-minus-min gaps),
between 0 and 1; for example, 0.20 means a 20 percentage-point difference.
The ranking is available in `fairness.json` and a flat `fairness_ranking.csv`.
It helps locate where model errors differ most across the audited groups. It is
not a probability of bias, a significance test, or a causal explanation, and
small group sizes can make the estimate unstable. The metrics do not establish
why a difference exists or whether it is acceptable. CIFAR-10 does
not provide demographic group attributes, so its existing evaluation cannot
produce demographic fairness results; differences among its object classes
remain class-wise performance results.
The report also lists dimensions not measured, including causal explanations
and individual or counterfactual fairness. The selected statistical criteria
may disagree, so their measurements are kept separate. Auditing all available
features can surface systematic error patterns without claiming that any
feature is a protected or sensitive attribute.

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

## Current limitations and next steps

- The generic trainer currently supports supervised classification on tabular CSVs.
- The detailed robustness and explainability workflows are still image/CIFAR-specific.
- Regression, ranking, confidence intervals, significance tests, and causal or
  counterfactual fairness need separate task-specific implementations.
- ACSIncome or another dataset from a different population is a useful next
  dataset for evaluating subgroup differences beyond Adult.
