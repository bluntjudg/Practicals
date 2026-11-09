import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def main():
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )
    print("Number of samples:", data.data.shape[0])
    print("Number of features:", data.data.shape[1])
    print("Training samples:", x_train.shape[0])
    print("Testing samples:", x_test.shape[0])

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
        ),
    }

    rows = []
    predictions = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        predictions[name] = y_pred
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            }
        )
        print(f"\n{name} Results")
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=0))

    results = pd.DataFrame(rows)
    print("\nModel Comparison:\n", results.to_string(index=False))

    feature_importance = pd.DataFrame(
        {
            "Feature": data.feature_names,
            "Importance": models["Random Forest"].feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    print("\nTop 15 Random Forest Features:\n", feature_importance.head(15))
    feature_importance.head(15).sort_values("Importance").plot(
        x="Feature",
        y="Importance",
        kind="barh",
        figsize=(10, 6),
        legend=False,
    )
    plt.title("Top 15 Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
