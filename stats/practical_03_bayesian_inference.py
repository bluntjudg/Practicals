"""Practical 3: Apply Bayesian inference to a customer-purchase problem."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta


def main() -> None:
    # Step 1: Set a Beta(2, 2) prior for the unknown purchase probability.
    alpha_prior = 2
    beta_prior = 2

    # Step 2: Record the observed number of purchases among all customers.
    customers = 20
    purchases = 14
    if not 0 <= purchases <= customers:
        raise ValueError("Purchases must be between zero and the customer count.")

    # Step 3: Update the prior parameters using the Beta-Binomial conjugate rule.
    alpha_posterior = alpha_prior + purchases
    beta_posterior = beta_prior + customers - purchases
    print(f"Prior distribution: Beta({alpha_prior}, {beta_prior})")
    print(f"Observed purchases: {purchases} of {customers}")
    print(
        "Posterior distribution: "
        f"Beta({alpha_posterior}, {beta_posterior})"
    )

    # Step 4: Summarize the posterior using its mean, mode, and 95% credible
    # interval.
    posterior_mean = alpha_posterior / (alpha_posterior + beta_posterior)
    posterior_map = (alpha_posterior - 1) / (
        alpha_posterior + beta_posterior - 2
    )
    credible_interval = beta.ppf(
        [0.025, 0.975],
        alpha_posterior,
        beta_posterior,
    )
    print(f"Posterior mean: {posterior_mean:.4f}")
    print(f"MAP estimate: {posterior_map:.4f}")
    print(
        "95% credible interval: "
        f"{credible_interval[0]:.4f} to {credible_interval[1]:.4f}"
    )

    # Step 5: Find the posterior probability that the purchase rate exceeds 60%.
    probability_over_60 = beta.sf(
        0.60,
        alpha_posterior,
        beta_posterior,
    )
    print(f"Probability purchase rate is above 60%: {probability_over_60:.4f}")

    # Step 6: Plot prior and posterior density to show how the data changed our
    # belief about the purchase probability.
    probability_grid = np.linspace(0, 1, 1000)
    plt.plot(
        probability_grid,
        beta.pdf(probability_grid, alpha_prior, beta_prior),
        label=f"Prior Beta({alpha_prior}, {beta_prior})",
    )
    plt.plot(
        probability_grid,
        beta.pdf(probability_grid, alpha_posterior, beta_posterior),
        label=f"Posterior Beta({alpha_posterior}, {beta_posterior})",
    )
    plt.axvline(
        posterior_mean,
        color="red",
        linestyle="--",
        label="Posterior mean",
    )
    plt.xlabel("Purchase probability")
    plt.ylabel("Probability density")
    plt.title("Bayesian update for customer purchase probability")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
