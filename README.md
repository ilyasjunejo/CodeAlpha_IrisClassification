# Iris Flower Classification

A reproducible machine-learning project that compares several classifiers on the classic Iris dataset from scikit-learn. The goal is to practice the full workflow: data inspection, train/test splitting, model comparison, evaluation, and communicating results with visualizations.

## Project purpose

This project is designed to build confidence with common data-science steps used in beginner and intermediate ML work:

- loading and inspecting a dataset
- checking feature distributions and correlations
- splitting data into train and test sets
- comparing model performance with a held-out evaluation set
- interpreting classification results using a confusion matrix and report

## Dataset

The project uses scikit-learn's built-in Iris dataset. It contains 150 flower samples, four numeric features, and three target classes. It is useful for learning and experimentation, but it is small and not representative of the complexity of most real-world classification tasks.

## Workflow

1. Load the dataset and inspect the sample records.
2. Visualize feature distributions and correlations.
3. Split the data into train and test sets using a stratified split.
4. Train and compare:
   - Logistic Regression
   - Random Forest
   - Support Vector Machine
5. Evaluate the models using accuracy and a classification report.
6. Save plots for distribution, correlation, model comparison, confusion matrix, and pairplot analysis.

## Models compared

- Logistic Regression with StandardScaler
- Random Forest Classifier
- Support Vector Machine with StandardScaler

The script prints the actual scores for the selected split when run, rather than claiming fixed performance values for all environments.

## Visual outputs

Running the script generates these files in the repository directory:

- `iris_distributions.png`
- `iris_correlation.png`
- `iris_confusion_matrix.png`
- `iris_model_comparison.png`
- `iris_pairplot.png`

## Run locally

Python 3.10+ is recommended.

```bash
git clone https://github.com/ilyasjunejo/CodeAlpha_IrisClassification.git
cd CodeAlpha_IrisClassification
python -m venv .venv
```

Activate the environment, then install the dependencies:

Windows:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python iris_classification.py
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python iris_classification.py
```

## Project structure

- `iris_classification.py` — main pipeline and analysis script
- `requirements.txt` — Python dependencies
- generated PNG files — saved plots from the analysis

## Tech stack

- Python
- pandas
- Matplotlib
- Seaborn
- scikit-learn

## Learning outcomes

This project helps practice:

- train/test splitting
- feature scaling
- supervised classification
- model comparison
- confusion matrices
- precision, recall, F1-score
- exploratory data analysis and visualization

## Limitations

- The dataset is educational and compact.
- Results are tied to one train/test split and the chosen model settings.
- This is a learning project, not a deployment-ready production pipeline.

## Author

**M. Ilyas Junejo** · [GitHub profile](https://github.com/ilyasjunejo)
