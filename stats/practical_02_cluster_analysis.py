"""Practical 2: Conduct cluster analysis and evaluate cluster quality."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def main() -> None:
    # Step 1: Calculate pairwise Euclidean distances for three example points.
    small_data = np.array([[1, 2], [2, 3], [5, 8]], dtype=float)
    distance_matrix = squareform(pdist(small_data, metric="euclidean"))
    print("Distance matrix:\n", np.round(distance_matrix, 3))

    # Step 2: Treat the first point as one cluster and the remaining points as
    # another; summarize their minimum, maximum, and average linkage distances.
    distances = distance_matrix[0, 1:]
    print(f"Single linkage distance: {distances.min():.4f}")
    print(f"Complete linkage distance: {distances.max():.4f}")
    print(f"Average linkage distance: {distances.mean():.4f}")
    print("Ward linkage matrix:\n", np.round(linkage(small_data, method="ward"), 3))

    # Step 3: Generate a reproducible dataset because the journal's housing CSV
    # is not included with the practical.
    features, _ = make_blobs(
        n_samples=500,
        centers=4,
        cluster_std=1.15,
        random_state=42,
    )
    scaled = StandardScaler().fit_transform(features)

    # Step 4: Fit hierarchical clustering and visualize its merge structure.
    # A subset keeps the dendrogram readable while K-means below uses all rows.
    hierarchical = linkage(scaled[:60], method="ward")
    plt.figure(figsize=(10, 4))
    dendrogram(hierarchical, no_labels=True)
    plt.title("Ward hierarchical clustering (60-point sample)")
    plt.xlabel("Sample observations")
    plt.ylabel("Merge distance")
    plt.tight_layout()
    plt.show()

    # Step 5: Compare silhouette scores for several possible cluster counts.
    cluster_counts = range(2, 8)
    scores = []
    fitted_models = {}
    for cluster_count in cluster_counts:
        model = KMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init=10,
        ).fit(scaled)
        fitted_models[cluster_count] = model
        scores.append(silhouette_score(scaled, model.labels_))
        print(f"Silhouette score for k={cluster_count}: {scores[-1]:.4f}")

    # Step 6: Choose the best-scoring candidate and plot its cluster assignments.
    best_count = list(cluster_counts)[int(np.argmax(scores))]
    best_model = fitted_models[best_count]
    print(f"Best silhouette score is for k={best_count}.")
    plt.scatter(
        features[:, 0],
        features[:, 1],
        c=best_model.labels_,
        cmap="viridis",
        s=18,
    )
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"K-means clusters (k={best_count})")
    plt.tight_layout()
    plt.show()

    # Step 7: Plot the score curve; a higher silhouette score indicates more
    # compact, well-separated clusters.
    plt.plot(list(cluster_counts), scores, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Cluster-quality comparison")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
