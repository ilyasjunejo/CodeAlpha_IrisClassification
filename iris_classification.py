"""Iris flower classification: a small, reproducible scikit-learn project.

Run with: python iris_classification.py
The script downloads no data; it uses scikit-learn's built-in Iris dataset.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

OUTPUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    df = X.copy()
    df["species"] = y.map(dict(enumerate(iris.target_names)))

    print("Iris Flower Classification")
    print(f"Samples: {len(df)} | Features: {X.shape[1]} | Classes: {len(iris.target_names)}")
    print("\nDataset preview:")
    print(df.head())
    print("\nMissing values:", int(df.isna().sum().sum()))

    # Distribution plots for the four measurements.
    df[X.columns].hist(figsize=(10, 7), bins=18, edgecolor="white")
    plt.suptitle("Iris Feature Distributions")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "iris_distributions.png", dpi=180)
    plt.close()

    # Correlations between numeric features.
    plt.figure(figsize=(8, 6))
    sns.heatmap(X.corr(), annot=True, cmap="vlag", center=0, fmt=".2f")
    plt.title("Iris Feature Correlations")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "iris_correlation.png", dpi=180)
    plt.close()

    # Split once, with class proportions preserved.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Support Vector Machine": make_pipeline(StandardScaler(), SVC()),
    }

    scores = {}
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        scores[name] = accuracy_score(y_test, predictions)
        fitted_models[name] = model
        print(f"\n{name}: accuracy = {scores[name]:.3f}")
        print(classification_report(y_test, predictions, target_names=iris.target_names, zero_division=0))

    best_name = max(scores, key=scores.get)
    best_model = fitted_models[best_name]
    print(f"Best model on this split: {best_name} ({scores[best_name]:.3f} accuracy)")

    ConfusionMatrixDisplay.from_estimator(
        best_model, X_test, y_test, display_labels=iris.target_names, cmap="Blues"
    )
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "iris_confusion_matrix.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(x=list(scores.keys()), y=list(scores.values()), color="#3b82f6")
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy on held-out test set")
    plt.xlabel("")
    plt.title("Model Comparison")
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "iris_model_comparison.png", dpi=180)
    plt.close()

    pairplot = sns.pairplot(df, hue="species", corner=True, diag_kind="hist")
    pairplot.fig.suptitle("Iris Dataset Pairplot", y=1.02)
    pairplot.savefig(OUTPUT_DIR / "iris_pairplot.png", dpi=150)
    plt.close("all")

    print("\nSaved plots to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
