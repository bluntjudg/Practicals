import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
    iris = load_iris()
    x = iris.data
    y = iris.target
    data_frame = pd.DataFrame(x, columns=iris.feature_names)
    print("Original dataset:\n", data_frame.head())

    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)
    transformed = pd.DataFrame(
        x_pca,
        columns=["Principal Component 1", "Principal Component 2"],
    )

    print("\nTransformed data:\n", transformed.head())
    print("\nExplained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance explained:", pca.explained_variance_ratio_.sum())

    for target, label in enumerate(iris.target_names):
        plt.scatter(
            x_pca[y == target, 0],
            x_pca[y == target, 1],
            label=label,
        )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA - Iris Dataset")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
