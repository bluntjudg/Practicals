"""Practical 10: Describe the Iris data and evaluate a statistical model."""

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    # Step 1: Load the Iris dataset included with scikit-learn.
    iris = load_iris(as_frame=True)
    data = iris.frame
    measurement = data["sepal length (cm)"]
    print("First five rows:\n", data.head())
    print(f"\nDataset shape: {data.shape}")

    # Step 2: Calculate descriptive statistics for sepal length.
    print("\nSepal length descriptive statistics:")
    print(f"Mean: {measurement.mean():.4f}")
    print(f"Median: {measurement.median():.4f}")
    print(f"Mode: {measurement.mode().iloc[0]:.4f}")
    print(f"Sample standard deviation: {measurement.std(ddof=1):.4f}")
    print(f"Sample variance: {measurement.var(ddof=1):.4f}")
    quartiles = measurement.quantile([0.25, 0.50, 0.75])
    interquartile_range = quartiles.loc[0.75] - quartiles.loc[0.25]
    print(f"Quartiles (25%, 50%, 75%):\n{quartiles.to_string()}")
    print(f"Interquartile range: {interquartile_range:.4f}")
    print(f"Minimum: {measurement.min():.2f}")
    print(f"Maximum: {measurement.max():.2f}")

    # Step 3: Review all feature summaries and the species counts.
    print("\nFeature summary:\n", data[iris.feature_names].describe().round(3))
    print(
        "\nObservations per species:\n",
        data["target"].map(dict(enumerate(iris.target_names))).value_counts(),
    )

    # Step 4: Plot sepal length against petal length and compare feature
    # distributions in a scatter-matrix view.
    plt.scatter(
        data["sepal length (cm)"],
        data["petal length (cm)"],
        c=data["target"],
        cmap="viridis",
        alpha=0.7,
    )
    plt.xlabel("Sepal length (cm)")
    plt.ylabel("Petal length (cm)")
    plt.title("Iris sepal length versus petal length")
    plt.colorbar(label="Species class")
    plt.tight_layout()
    plt.show()

    scatter_matrix(
        data[iris.feature_names],
        figsize=(9, 9),
        diagonal="hist",
        marker="o",
        alpha=0.6,
    )
    plt.suptitle("Iris feature relationships")
    plt.tight_layout()
    plt.show()

    # Step 5: Separate features and species labels, then create a stratified
    # train/test split for an out-of-sample model evaluation.
    features = data[iris.feature_names]
    labels = data["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels,
    )

    # Step 6: Fit a standardization-plus-logistic-regression pipeline. Scaling
    # is fitted only on training data to avoid leaking test information.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
    print(
        "\nClassification report:\n",
        classification_report(
            y_test,
            predictions,
            target_names=iris.target_names,
            zero_division=0,
        ),
    )


if __name__ == "__main__":
    main()
