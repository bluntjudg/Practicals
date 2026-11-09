"""Practical 1: Perform PCA and interpret the results."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def describe_pca(matrix: np.ndarray, label: str) -> None:
    """Print an eigendecomposition of a covariance or correlation matrix."""
    # Step 1: Compute eigenvalues and eigenvectors for the symmetric matrix.
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # Step 2: Sort components from greatest to smallest eigenvalue.
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Step 3: Express each component's share of total variance as a percentage.
    variance_explained = 100 * eigenvalues / eigenvalues.sum()
    print(f"\n{label} matrix:\n{matrix}")
    print(f"Eigenvalues (largest first): {np.round(eigenvalues, 4)}")
    print(f"Eigenvectors (columns):\n{np.round(eigenvectors, 4)}")
    print(f"Variance explained (%): {np.round(variance_explained, 2)}")


def main() -> None:
    # Step 1: Start with the small two-variable example from the journal.
    observations = np.array(
        [[4, 11], [8, 4], [13, 5], [7, 14]],
        dtype=float,
    )
    covariance = np.cov(observations, rowvar=False, ddof=1)
    describe_pca(covariance, "Toy-data covariance")

    # Step 2: Compare covariance PCA with correlation PCA when scales differ.
    differently_scaled = np.array(
        [[50, 25], [60, 30], [55, 28], [65, 35], [70, 32]],
        dtype=float,
    )
    correlation = np.corrcoef(differently_scaled, rowvar=False)
    describe_pca(correlation, "Toy-data correlation")

    # Step 3: Load the built-in wine dataset so the practical needs no data file.
    wine = load_wine(as_frame=True)
    features = wine.data
    print(f"\nWine dataset shape: {features.shape}")

    # Step 4: Standardize features so measurements on large scales do not
    # dominate the principal components.
    standardized = StandardScaler().fit_transform(features)

    # Step 5: Fit PCA and calculate each component's and cumulative variance.
    pca = PCA()
    components = pca.fit_transform(standardized)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    print("Variance explained by component:", np.round(explained, 3))
    print("Cumulative variance explained:", np.round(cumulative, 3))
    print(
        "Components needed to explain at least 90% of variance:",
        int(np.searchsorted(cumulative, 0.90) + 1),
    )
    print("First two transformed rows:\n", np.round(components[:2, :2], 3))

    # Step 6: Plot cumulative variance to help select a useful number of
    # components.
    component_numbers = np.arange(1, len(explained) + 1)
    plt.plot(component_numbers, cumulative, marker="o")
    plt.axhline(0.90, color="red", linestyle="--", label="90% threshold")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative variance explained")
    plt.title("PCA on the Wine dataset")
    plt.xticks(component_numbers)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
