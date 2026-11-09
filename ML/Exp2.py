import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


def main():
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=4,
        random_state=42,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("Training samples:", len(x_train))
    print("Testing samples:", len(x_test))
    print("Predictions:", predictions)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
    print(
        classification_report(
            y_test,
            predictions,
            target_names=data.target_names,
            zero_division=0,
        )
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=data.target_names,
    )
    plt.title("Decision Tree Confusion Matrix")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(16, 8))
    plot_tree(
        model,
        feature_names=data.feature_names,
        class_names=data.target_names,
        filled=True,
        rounded=True,
    )
    plt.title("Decision Tree Classifier")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
