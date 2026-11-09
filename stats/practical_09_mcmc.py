"""Practical 9: Implement Gibbs and Metropolis MCMC methods."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma, norm


def gibbs_sample(iterations: int, rng: np.random.Generator) -> np.ndarray:
    """Sample the binary joint distribution shown in the journal."""
    # Step 1: Start the two binary variables at zero and allocate sample storage.
    x, y = 0, 0
    samples = np.empty((iterations, 2), dtype=int)

    # Step 2: Alternately sample X given Y and then Y given the new X.
    for iteration in range(iterations):
        probability_x_is_one = 1 / 3 if y == 0 else 3 / 4
        x = int(rng.random() < probability_x_is_one)

        probability_y_is_one = 1 / 5 if x == 0 else 3 / 5
        y = int(rng.random() < probability_y_is_one)
        samples[iteration] = (x, y)

    return samples


def metropolis_sample(
    iterations: int,
    initial_value: float,
    log_density,
    proposal_sd: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run a random-walk Metropolis chain using log target densities."""
    # Step 1: Allocate the chain and set its starting value.
    samples = np.empty(iterations)
    samples[0] = initial_value

    # Step 2: Propose nearby values, accepting or rejecting each proposal.
    for iteration in range(1, iterations):
        current = samples[iteration - 1]
        proposal = rng.normal(current, proposal_sd)
        log_acceptance = min(
            0.0,
            log_density(proposal) - log_density(current),
        )
        if np.log(rng.random()) < log_acceptance:
            samples[iteration] = proposal
        else:
            samples[iteration] = current

    return samples


def main() -> None:
    rng = np.random.default_rng(42)
    iterations = 20_000
    burn_in = 2_000

    # Step 1: Use Gibbs sampling for the four-state joint distribution.
    target_states = [(0, 0), (0, 1), (1, 0), (1, 1)]
    target_probabilities = np.array([0.40, 0.10, 0.20, 0.30])
    gibbs_chain = gibbs_sample(iterations, rng)
    retained_gibbs = gibbs_chain[burn_in:]

    # Step 2: Compare empirical state probabilities with their known targets.
    estimates = np.array(
        [
            np.mean(
                (retained_gibbs[:, 0] == state[0])
                & (retained_gibbs[:, 1] == state[1])
            )
            for state in target_states
        ]
    )
    print("State    Target    Gibbs estimate")
    for state, target, estimate in zip(
        target_states,
        target_probabilities,
        estimates,
    ):
        print(f"{state!s:8} {target:7.3f} {estimate:16.4f}")

    # Step 3: Plot target and estimated probabilities after excluding warm-up.
    positions = np.arange(len(target_states))
    plt.bar(
        positions - 0.2,
        target_probabilities,
        width=0.4,
        label="Target",
    )
    plt.bar(positions + 0.2, estimates, width=0.4, label="Gibbs estimate")
    plt.xticks(positions, [str(state) for state in target_states])
    plt.xlabel("(X, Y) state")
    plt.ylabel("Probability")
    plt.title("Gibbs estimates versus target probabilities")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Step 4: Sample a density proportional to exp(-x^2), a normal distribution
    # with mean 0 and variance 0.5.
    normal_chain = metropolis_sample(
        iterations,
        initial_value=0.0,
        log_density=lambda value: -(value**2),
        proposal_sd=1.0,
        rng=rng,
    )[burn_in:]
    print("\nMetropolis chain for density proportional to exp(-x^2):")
    print(f"Sample mean: {normal_chain.mean():.4f} (theoretical 0.0000)")
    print(f"Sample variance: {normal_chain.var(ddof=1):.4f} (theoretical 0.5000)")

    # Step 5: Compare the sampled distribution with its normalized normal
    # density and display a trace plot.
    grid = np.linspace(-3, 3, 400)
    plt.hist(normal_chain, bins=45, density=True, alpha=0.6, label="MCMC samples")
    plt.plot(grid, norm.pdf(grid, loc=0, scale=np.sqrt(0.5)), color="red", label="Target")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title("Metropolis sampling: exp(-x^2)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.plot(normal_chain[:2000], alpha=0.8)
    plt.xlabel("Retained iteration")
    plt.ylabel("Sample value")
    plt.title("Metropolis trace: exp(-x^2)")
    plt.tight_layout()
    plt.show()

    # Step 6: Repeat with a Gamma(shape=3, rate=2) target. Proposals below zero
    # receive zero probability through the log-density function.
    def gamma_log_density(value: float) -> float:
        if value <= 0:
            return -np.inf
        return (3 - 1) * np.log(value) - 2 * value

    gamma_chain = metropolis_sample(
        iterations,
        initial_value=1.0,
        log_density=gamma_log_density,
        proposal_sd=0.8,
        rng=rng,
    )[burn_in:]
    print("\nMetropolis chain for Gamma(shape=3, rate=2):")
    print(f"Sample mean: {gamma_chain.mean():.4f} (theoretical 1.5000)")
    print(f"Sample variance: {gamma_chain.var(ddof=1):.4f} (theoretical 0.7500)")

    # Step 7: Overlay the Gamma target density on the sampled histogram.
    positive_grid = np.linspace(0, 8, 400)
    plt.hist(
        gamma_chain,
        bins=45,
        density=True,
        alpha=0.6,
        label="MCMC samples",
    )
    plt.plot(
        positive_grid,
        gamma.pdf(positive_grid, a=3, scale=0.5),
        color="red",
        label="Gamma(3, rate=2) target",
    )
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title("Metropolis sampling: Gamma target")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
