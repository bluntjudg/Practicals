# Statistics Practicals

This folder contains ten standalone Python practicals based on the Statistics
practical journal. Each script includes comments explaining the steps and can
be run independently.

## Setup

Use Python 3.10 or later. From the repository root, install the required
packages:

```bash
python3 -m pip install -r stats/requirements.txt
```

Run one practical at a time:

```bash
python3 stats/practical_01_pca.py
```

Replace `01` with the practical number from `01` to `10`. The scripts print
their results and open charts where relevant.

## Practicals

1. **PCA** — eigenanalysis, covariance versus correlation, and wine-data
   dimensionality reduction.
2. **Cluster analysis** — linkage distances, hierarchical clustering, K-means,
   and silhouette evaluation.
3. **Bayesian inference** — update a purchase-rate estimate with a Beta-Binomial
   model.
4. **ANOVA** — compare teaching-method means and group variances.
5. **Hypothesis testing** — test average customer spending and compare two
   marketing campaigns.
6. **Factor analysis** — check survey-data suitability and examine rotated
   factor loadings.
7. **Bayesian networks** — build a network and answer a query using pgmpy.
8. **Discriminant analysis** — use Linear Discriminant Analysis to classify
   wine varieties.
9. **MCMC** — compare Gibbs sampling with a known joint distribution and use
   Metropolis sampling for two target distributions.
10. **Descriptive statistics and modeling** — summarize Iris measurements and
    evaluate a classifier.

The original journal refers to external housing, loan, survey, and seeds data
files that were not included with the PDF. To keep every script runnable without
those files, the relevant practicals use reproducible generated survey data or
datasets included with scikit-learn. Practical 10's journal body describes
Iris statistics, while its index mentions model evaluation; this script
includes both.

## GitHub

Git is initialized locally in this workspace. After creating an empty
repository named `practicals` under your GitHub account, connect and publish it
from the repository root:

```bash
git add .gitignore stats/
git commit -m "Add statistics practical journal"
git remote add origin https://github.com/<YOUR-USERNAME>/practicals.git
git push -u origin main
```

Choose **Private** when creating the GitHub repository if you want access
limited to your account. The personal journal PDF is intentionally not copied
into this repository.
