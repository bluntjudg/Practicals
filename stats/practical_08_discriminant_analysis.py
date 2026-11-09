"""Practical 8: Conduct discriminant analysis for classification."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main() -> None:
    # Step 1: Load a built-in multiclass dataset instead of relying on the
    # external seeds_dataset.txt file referenced in the journal.
    wine = load_wine()
    features = wine.data
    labels = wine.target
    print(f"Dataset shape: {features.shape}")
    print(f"Class counts: {np.bincount(labels)}")

    # Step 2: Split data before scaling to keep test data out of training.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels,
    )

    # Step 3: Fit the scaler on the training set and apply it to both partitions.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Step 4: Fit Linear Discriminant Analysis and project training observations
    # onto the two discriminant axes for visualization.
    lda = LinearDiscriminantAnalysis(n_components=2)
    training_projection = lda.fit_transform(x_train_scaled, y_train)
    print(
        "Explained discriminant variance ratio:",
        np.round(lda.explained_variance_ratio_, 4),
    )
    print(
        "Cumulative explained discriminant variance:",
        np.round(np.cumsum(lda.explained_variance_ratio_), 4),
    )

    # Step 5: Plot the first two discriminant scores and color points by class.
    for class_index, class_name in enumerate(wine.target_names):
        class_rows = y_train == class_index
        plt.scatter(
            training_projection[class_rows, 0],
            training_projection[class_rows, 1],
            label=class_name,
            alpha=0.75,
        )
    plt.xlabel("First linear discriminant")
    plt.ylabel("Second linear discriminant")
    plt.title("LDA projection of wine varieties")
    plt.legend(title="Wine class")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Step 6: Classify held-out observations and assess performance on data not
    # used to fit the model.
    predictions = lda.predict(x_test_scaled)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=wine.target_names,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
