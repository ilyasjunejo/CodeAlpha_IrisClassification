# Iris Flower Classification

A reproducible beginner-friendly machine-learning project that compares classification models on scikit-learn's built-in Iris dataset.

## What it does

The script loads the Iris dataset, explores feature distributions and correlations, trains three classifiers, compares their held-out test accuracy, prints classification metrics, and saves visualizations.

### Models compared

- Logistic Regression with feature scaling
- Random Forest Classifier
- Support Vector Machine with feature scaling

The script uses a stratified 80/20 train/test split with a fixed random seed. Results can vary with different data splits or settings; the script prints the actual scores when run instead of promising a fixed accuracy.

## Visualizations

Running the script generates:

- `iris_distributions.png` — distributions of the four flower measurements
- `iris_correlation.png` — correlations between numeric features
- `iris_confusion_matrix.png` — confusion matrix for the best model on this test split
- `iris_model_comparison.png` — comparison of test-set accuracy
- `iris_pairplot.png` — feature-pair relationships by species

## Run locally

Python 3.10+ is recommended.

```bash
git clone https://github.com/ilyasjunejo/CodeAlpha_IrisClassification.git
cd CodeAlpha_IrisClassification
python -m venv .venv
```

Activate the environment, then install dependencies:

**Windows**
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python iris_classification.py
```

**macOS / Linux**
```bash
source .venv/bin/activate
pip install -r requirements.txt
python iris_classification.py
```

## Tech stack

Python, pandas, Matplotlib, Seaborn, and scikit-learn.

## Learning objectives

Practice train/test splitting, feature scaling, classification, model comparison, confusion matrices, precision, recall, F1-score, and exploratory data analysis.

## Dataset and limitations

The Iris dataset is a small, classic educational dataset with 150 samples, four numeric features, and three species. It is useful for learning but is not representative of the complexity of most real-world classification tasks.

## Author

**M. Ilyas Junejo** · [GitHub profile](https://github.com/ilyasjunejo)
