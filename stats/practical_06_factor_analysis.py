"""Practical 6: Perform factor analysis on survey-style data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler


def make_survey_data(
    respondents: int = 800,
    factors: int = 5,
    items_per_factor: int = 5,
) -> pd.DataFrame:
    """Create reproducible 1-to-5 survey responses with shared latent traits."""
    # Step 1: Generate independent latent traits for the survey respondents.
    rng = np.random.default_rng(42)
    latent_traits = rng.normal(size=(respondents, factors))

    # Step 2: Let each question mostly measure one trait, with some noise.
    responses = []
    for factor_index in range(factors):
        for item_index in range(items_per_factor):
            loading = 0.72 + 0.04 * (item_index % 3)
            score = (
                loading * latent_traits[:, factor_index]
                + rng.normal(scale=0.65, size=respondents)
            )
            # Convert continuous simulated responses into Likert categories.
            likert_response = np.digitize(score, [-1.0, -0.3, 0.3, 1.0]) + 1
            responses.append(likert_response)

    columns = [
        f"Item_{item_number:02d}"
        for item_number in range(1, factors * items_per_factor + 1)
    ]
    return pd.DataFrame(np.column_stack(responses), columns=columns)


def main() -> None:
    # Step 1: Use generated survey responses because the journal's CSV files
    # were not included with the source PDF.
    survey = make_survey_data()
    print(f"Survey response shape: {survey.shape}")
    print(f"Missing responses: {int(survey.isna().sum().sum())}")
    print(f"Duplicate response rows: {int(survey.duplicated().sum())}")

    # Step 2: Calculate the correlation matrix and Bartlett's sphericity test.
    # A significant result indicates that the item correlations are not an
    # identity matrix, which is a prerequisite for useful factor analysis.
    respondent_count, item_count = survey.shape
    correlation = survey.corr().to_numpy(copy=True)
    determinant = np.linalg.det(correlation)
    if determinant <= 0:
        raise ValueError("Survey correlation matrix must have a positive determinant.")
    bartlett_chi_square = -(
        respondent_count - 1 - (2 * item_count + 5) / 6
    ) * np.log(determinant)
    bartlett_degrees_of_freedom = item_count * (item_count - 1) // 2
    bartlett_p_value = chi2.sf(
        bartlett_chi_square,
        bartlett_degrees_of_freedom,
    )

    # Step 3: Calculate the Kaiser-Meyer-Olkin measure from correlations and
    # partial correlations. Values above 0.5 are commonly treated as adequate.
    inverse_correlation = np.linalg.inv(correlation)
    partial_correlation = -inverse_correlation / np.sqrt(
        np.outer(np.diag(inverse_correlation), np.diag(inverse_correlation))
    )
    np.fill_diagonal(correlation, 0)
    np.fill_diagonal(partial_correlation, 0)
    correlation_squares = np.square(correlation).sum()
    partial_squares = np.square(partial_correlation).sum()
    kmo_model = correlation_squares / (correlation_squares + partial_squares)
    print(f"Bartlett's test chi-square: {bartlett_chi_square:.2f}")
    print(f"Bartlett's test p-value: {bartlett_p_value:.6f}")
    print(f"KMO measure: {kmo_model:.4f}")
    if bartlett_p_value >= 0.05 or kmo_model < 0.5:
        raise ValueError("The generated survey data are not suitable for factor analysis.")

    # Step 4: Inspect correlation-matrix eigenvalues and use the Kaiser rule
    # (eigenvalue greater than one) as an initial factor count.
    eigenvalues = np.linalg.eigvalsh(survey.corr().to_numpy())[::-1]
    factor_count = int(np.sum(eigenvalues > 1))
    factor_count = max(1, factor_count)
    print(f"Eigenvalues: {np.round(eigenvalues, 3)}")
    print(f"Factors with eigenvalue > 1: {factor_count}")

    # Step 5: Standardize item scores and fit a varimax-rotated factor model.
    standardized = StandardScaler().fit_transform(survey)
    factor_analysis = FactorAnalysis(
        n_components=factor_count,
        rotation="varimax",
        random_state=42,
    )
    factor_analysis.fit(standardized)
    loadings_array = factor_analysis.components_.T
    loadings = pd.DataFrame(
        loadings_array,
        index=survey.columns,
        columns=[f"Factor_{index + 1}" for index in range(factor_count)],
    )
    print("\nRotated factor loadings:")
    print(loadings.round(3))

    # Step 6: Summarize variance associated with each factor.
    factor_variance = np.square(loadings_array).sum(axis=0)
    proportion_variance = factor_variance / item_count
    cumulative_variance = np.cumsum(proportion_variance)
    print("\nFactor variance (variance, proportion, cumulative proportion):")
    print(
        tuple(
            np.round(part, 3)
            for part in (
                factor_variance,
                proportion_variance,
                cumulative_variance,
            )
        )
    )

    # Step 7: Draw the scree plot to show how eigenvalues change by factor.
    plt.plot(
        np.arange(1, len(eigenvalues) + 1),
        eigenvalues,
        marker="o",
    )
    plt.axhline(1, color="red", linestyle="--", label="Eigenvalue = 1")
    plt.xlabel("Factor number")
    plt.ylabel("Eigenvalue")
    plt.title("Scree plot for survey responses")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
