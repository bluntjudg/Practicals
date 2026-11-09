"""Practical 5: Use hypothesis testing to analyze business data."""

import numpy as np
from scipy import stats


def report_test(name: str, statistic: float, p_value: float) -> None:
    """Print a test result and its decision at the 5% significance level."""
    print(f"\n{name}")
    print(f"Test statistic: {statistic:.4f}")
    print(f"Two-sided p-value: {p_value:.6f}")
    if p_value < 0.05:
        print("Reject H0: the observed difference is statistically significant.")
    else:
        print("Fail to reject H0: there is not enough evidence of a difference.")


def main() -> None:
    # Step 1: Compare the sample monthly spending of 15 customers with the
    # business claim that the population average is 5,000.
    spending = np.array(
        [
            5200, 4800, 5100, 5300, 4950,
            5050, 4900, 5150, 5250, 5000,
            5350, 5100, 4950, 5200, 5050,
        ],
        dtype=float,
    )
    claimed_mean = 5000

    # Step 2: Apply a one-sample t-test because the population standard
    # deviation is unknown.
    one_sample = stats.ttest_1samp(spending, popmean=claimed_mean)
    print(f"Sample mean spending: {spending.mean():.2f}")
    print(f"Claimed population mean: {claimed_mean:.2f}")
    report_test("One-sample t-test", one_sample.statistic, one_sample.pvalue)

    # Step 3: Enter customer outcomes for the two independent campaigns.
    campaign_a = np.array([120, 135, 128, 140, 132, 138, 125, 145], dtype=float)
    campaign_b = np.array([110, 115, 118, 120, 122, 117, 113, 119], dtype=float)
    print(f"\nCampaign A mean: {campaign_a.mean():.2f}")
    print(f"Campaign B mean: {campaign_b.mean():.2f}")

    # Step 4: Use Welch's two-sample t-test, which does not require the two
    # campaigns to have equal variances.
    independent_test = stats.ttest_ind(
        campaign_a,
        campaign_b,
        equal_var=False,
    )
    report_test(
        "Welch independent two-sample t-test",
        independent_test.statistic,
        independent_test.pvalue,
    )


if __name__ == "__main__":
    main()
