# NN Reliability Framework

NN Reliability Framework is a Python research project baseline for systematically evaluating neural network behavior and building toward trustworthy AI analysis.

## Current scope (baseline)

This repository currently provides a clean, reproducible baseline experiment:

- CIFAR-10 dataset loading with `torchvision`
- ResNet-18 training (adapted to 10 classes)
- Evaluation with accuracy, precision, recall, F1 score, and confusion matrix
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
