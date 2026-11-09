import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def linear_regression():
    data = load_diabetes()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("Linear Regression")
    print("Dataset shape:", data.data.shape)
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("MSE:", mean_squared_error(y_test, predictions))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
    print("R2 score:", r2_score(y_test, predictions))
    print("Actual and predicted values:")
    for actual, predicted in zip(y_test[:10], predictions[:10]):
        print(f"{actual:.2f} {predicted:.2f}")


def logistic_regression():
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("\nLogistic Regression")
    print("Dataset shape:", data.data.shape)
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
    for actual, predicted in zip(y_test, predictions):
        print(
            f"Actual: {data.target_names[actual]}, "
            f"Predicted: {data.target_names[predicted]}"
        )


if __name__ == "__main__":
    linear_regression()
    logistic_regression()
