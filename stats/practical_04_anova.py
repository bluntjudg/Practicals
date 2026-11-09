"""Practical 4: Conduct ANOVA tests and interpret their results."""

import numpy as np
from scipy import stats


def main() -> None:
    # Step 1: Enter the marks observed under three teaching methods.
    method_a = [6, 8, 4]
    method_b = [8, 9, 6]
    method_c = [13, 12, 14]

    # Step 2: Use one-way ANOVA to test the null hypothesis that all group means
    # are equal.
    f_statistic, p_value = stats.f_oneway(method_a, method_b, method_c)
    alpha = 0.05
    print(f"One-way ANOVA F-statistic: {f_statistic:.4f}")
    print(f"One-way ANOVA p-value: {p_value:.6f}")
    if p_value < alpha:
        print("Reject H0: at least one teaching-method mean is different.")
    else:
        print("Fail to reject H0: the data do not show different group means.")

    # Step 3: Compare the sample variances of two groups using an F test.
    # This classical test assumes normally distributed samples.
    group_a = [12, 15, 14, 17, 18, 14, 15]
    group_b = [10, 22, 13, 29, 11, 25, 19]
    variance_a = np.var(group_a, ddof=1)
    variance_b = np.var(group_b, ddof=1)
    if variance_a >= variance_b:
        f_ratio = variance_a / variance_b
        numerator_df = len(group_a) - 1
        denominator_df = len(group_b) - 1
    else:
        f_ratio = variance_b / variance_a
        numerator_df = len(group_b) - 1
        denominator_df = len(group_a) - 1

    # Step 4: Double the upper-tail probability for a two-sided variance test.
    variance_p_value = min(
        1.0,
        2 * stats.f.sf(f_ratio, numerator_df, denominator_df),
    )
    print(f"\nGroup A sample variance: {variance_a:.4f}")
    print(f"Group B sample variance: {variance_b:.4f}")
    print(f"Two-sided F-test p-value: {variance_p_value:.6f}")
    if variance_p_value < alpha:
        print("Reject H0: the group variances differ significantly.")
    else:
        print("Fail to reject H0: no significant variance difference was found.")


if __name__ == "__main__":
    main()
