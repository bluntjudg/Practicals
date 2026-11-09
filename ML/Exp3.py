import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, make_blobs
from sklearn.metrics import silhouette_score


def cluster_iris():
    iris = load_iris()
    x = iris.data
    data_frame = pd.DataFrame(x, columns=iris.feature_names)

    print("Feature names:", iris.feature_names)
    print("Dataset shape:", x.shape)
    print("First five records:\n", data_frame.head())
    print("Missing values:\n", data_frame.isna().sum())

    inertias = []
    for cluster_count in range(1, 11):
        model = KMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init=10,
        )
        model.fit(x)
        inertias.append(model.inertia_)

    plt.plot(range(1, 11), inertias, marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Within-cluster sum of squares")
    plt.title("Elbow Method: Iris Dataset")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = model.fit_predict(x)
    data_frame["Cluster"] = labels
    print("Cluster labels:", labels)
    print("Cluster centroids:\n", model.cluster_centers_)
    print("Dataset with cluster labels:\n", data_frame.head(10))

    plt.scatter(x[:, 2], x[:, 3], c=labels, s=60, cmap="viridis")
    plt.scatter(
        model.cluster_centers_[:, 2],
        model.cluster_centers_[:, 3],
        marker="X",
        s=200,
        color="red",
        label="Centroids",
    )
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Petal width (cm)")
    plt.title("K-Means Clustering on Iris Dataset")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def cluster_two_dimensional_data():
    x = np.array(
        [
            [1, 2],
            [2, 3],
            [3, 2],
            [5, 4],
            [4, 6],
            [6, 5],
            [9, 9],
            [9, 10],
            [10, 11],
        ]
    )
    model = KMeans(
        n_clusters=3,
        random_state=42,
        init="random",
        n_init=10,
    )
    labels = model.fit_predict(x)
    centers = model.cluster_centers_

    print("\nTwo-dimensional dataset:\n", x)
    print("Iterations:", model.n_iter_)
    print("Centroids:\n", centers)
    for cluster, count in zip(*np.unique(labels, return_counts=True)):
        print(f"Cluster {cluster + 1}: {count} points")

    plt.scatter(x[:, 0], x[:, 1], c=labels, cmap="viridis", s=70)
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="*",
        color="red",
        s=220,
        label="Centroids",
    )
    plt.title("K-Means Clustering on 2D Dataset")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compare_silhouette_scores():
    x, _ = make_blobs(
        n_samples=600,
        centers=4,
        cluster_std=1.5,
        random_state=42,
    )
    cluster_counts = range(2, 8)
    scores = []

    for cluster_count in cluster_counts:
        model = KMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init=10,
        )
        labels = model.fit_predict(x)
        score = silhouette_score(x, labels)
        scores.append(score)
        print(f"Silhouette score for k={cluster_count}: {score:.4f}")

    plt.plot(list(cluster_counts), scores, marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Score vs Number of Clusters")
    plt.xticks(list(cluster_counts))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cluster_iris()
    cluster_two_dimensional_data()
    compare_silhouette_scores()
