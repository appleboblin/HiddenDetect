# Fisher Scoring

This branch keeps the original `--scoring-mode fisher` implementation, which
scores each held-out sample with a linear combination of per-layer refusal
curves.

For each stratified training fold, `code/eval_scoring.py` splits the training
curves into unsafe examples (`label == 1`) and safe examples (`label != 1`).
It computes:

```text
mean_diff = unsafe_mean - safe_mean
avg_class_var = 0.5 * (unsafe_var + safe_var)
weights = mean_diff / (sqrt(avg_class_var) + 1e-8)
score = x_test @ weights
```

The `1e-8` epsilon prevents division by zero for layers that are constant
within both classes.

## Why This Branch Keeps The Original

The original denominator divides each layer's class mean difference by the
pooled standard deviation. That makes each weight a standardized separation
term: layers are weighted by how many within-class standard deviations separate
unsafe and safe examples.

That behavior can be preferable for this repository's detection score because
the score is used only as a ranking signal for AUROC and AUPRC, not as a full
calibrated LDA classifier. Standard-deviation normalization is less aggressive
than inverse-variance weighting when a layer has very small variance, so it can
avoid letting a low-variance layer dominate the linear score.

The variance-denominator implementation on the `fisher-variance-denominator`
branch is closer to the diagonal Fisher/LDA weight direction:

```text
Sigma^-1 (mu_unsafe - mu_safe)
```

Under a diagonal covariance approximation, that direction divides each mean
difference by a pooled variance. This branch keeps the original version because
it preserves the existing standardized-effect behavior and its more moderate
layer weighting.
