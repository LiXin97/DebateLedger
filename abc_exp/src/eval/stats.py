"""Statistical analysis tools for ABC experiments.

No sklearn dependency -- everything is implemented with numpy/scipy.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats as sp_stats


def icc_2_1(test: np.ndarray, retest: np.ndarray) -> float:
    """ICC(2,1) for test-retest reliability.

    Two-way random effects model, single measure.
    Uses ANOVA decomposition: BMS, JMS, EMS formula.

    Parameters
    ----------
    test, retest : 1D arrays of measurements, same length.
        Each element is a measurement for the same subject.

    Returns
    -------
    float
        ICC(2,1) value in [-1, 1]. Returns 0.0 if degenerate.
    """
    test = np.asarray(test, dtype=float)
    retest = np.asarray(retest, dtype=float)
    n = len(test)
    if n < 2:
        return 0.0

    k = 2  # number of raters/measurements
    # Stack into (n, k) matrix: each row is a subject, columns are test/retest.
    data = np.column_stack([test, retest])

    # Grand mean.
    grand_mean = data.mean()

    # Row means (subject means).
    row_means = data.mean(axis=1)
    # Column means (rater/occasion means).
    col_means = data.mean(axis=0)

    # Between-subjects sum of squares.
    ss_between = k * np.sum((row_means - grand_mean) ** 2)
    # Between-raters (judges) sum of squares.
    ss_judges = n * np.sum((col_means - grand_mean) ** 2)
    # Total sum of squares.
    ss_total = np.sum((data - grand_mean) ** 2)
    # Error (residual) sum of squares.
    ss_error = ss_total - ss_between - ss_judges

    # Degrees of freedom.
    df_between = n - 1
    df_judges = k - 1
    df_error = (n - 1) * (k - 1)

    # Mean squares.
    bms = ss_between / df_between if df_between > 0 else 0.0
    jms = ss_judges / df_judges if df_judges > 0 else 0.0
    ems = ss_error / df_error if df_error > 0 else 0.0

    # ICC(2,1) = (BMS - EMS) / (BMS + (k-1)*EMS + k*(JMS - EMS)/n)
    denominator = bms + (k - 1) * ems + k * (jms - ems) / n
    if denominator == 0:
        return 0.0
    return float((bms - ems) / denominator)


def cronbach_alpha(items: np.ndarray) -> float:
    """Cronbach's alpha for internal consistency.

    Parameters
    ----------
    items : ndarray of shape (n_subjects, n_items)
        Binary matrix (or continuous scores).

    Returns
    -------
    float
        Cronbach's alpha. Returns 0.0 if degenerate (n_items < 2 or zero variance).
    """
    items = np.asarray(items, dtype=float)
    n_subjects, n_items = items.shape
    if n_items < 2 or n_subjects < 2:
        return 0.0

    # Variance of each item (column).
    item_vars = items.var(axis=0, ddof=1)
    # Variance of total scores (row sums).
    total_scores = items.sum(axis=1)
    total_var = total_scores.var(ddof=1)

    if total_var == 0:
        return 0.0

    sum_item_vars = item_vars.sum()
    alpha = (n_items / (n_items - 1)) * (1 - sum_item_vars / total_var)
    return float(alpha)


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC-ROC via trapezoidal integration.

    Parameters
    ----------
    y_true : 1D array
        Binary labels (0/1).
    y_score : 1D array
        Predicted probabilities or scores.

    Returns
    -------
    float
        Area under the ROC curve. Returns 0.5 if degenerate.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if len(y_true) < 2:
        return 0.5

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by descending score.
    desc_order = np.argsort(-y_score)
    y_true_sorted = y_true[desc_order]
    y_score_sorted = y_score[desc_order]

    # Compute TPR/FPR at each unique threshold.
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    tpr = tps / n_pos
    fpr = fps / n_neg

    # Prepend origin (0, 0).
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    # Trapezoidal integration.
    auc = float(np.trapz(tpr, fpr))
    return auc


def logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 100,
) -> dict:
    """Fit logistic regression via scipy minimize.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
        Binary labels (0/1).
    max_iter : int
        Maximum iterations for the optimizer.

    Returns
    -------
    dict
        Keys: 'coefficients' (1D array), 'intercept' (float),
              'predictions' (1D array of probabilities), 'auc' (float).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples, n_features = X.shape

    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid.
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z)),
        )

    def _neg_log_likelihood(params: np.ndarray) -> float:
        intercept = params[0]
        coefs = params[1:]
        z = X @ coefs + intercept
        p = _sigmoid(z)
        # Clip to avoid log(0).
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        return nll

    def _gradient(params: np.ndarray) -> np.ndarray:
        intercept = params[0]
        coefs = params[1:]
        z = X @ coefs + intercept
        p = _sigmoid(z)
        error = p - y  # (n_samples,)
        grad_intercept = error.sum()
        grad_coefs = X.T @ error
        return np.concatenate([[grad_intercept], grad_coefs])

    # Initial params: all zeros.
    x0 = np.zeros(n_features + 1)
    result = optimize.minimize(
        _neg_log_likelihood,
        x0,
        jac=_gradient,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )

    intercept = result.x[0]
    coefficients = result.x[1:]

    z = X @ coefficients + intercept
    predictions = _sigmoid(z)

    auc = compute_auc(y.astype(int), predictions)

    return {
        "coefficients": coefficients,
        "intercept": float(intercept),
        "predictions": predictions,
        "auc": auc,
    }


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval.

    Parameters
    ----------
    values : 1D array
        Sample values.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (e.g. 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (mean, lower, upper)
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = values[rng.randint(0, n, size=n)]
        boot_means[i] = sample.mean()

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    mean = float(values.mean())

    return (mean, lower, upper)


def mcnemar_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
) -> tuple[float, float]:
    """McNemar's test for paired binary data.

    Tests whether two methods have the same error rate on paired observations.

    Parameters
    ----------
    correct_a : 1D array
        Binary array (0/1) indicating correctness for method A.
    correct_b : 1D array
        Binary array (0/1) indicating correctness for method B.

    Returns
    -------
    tuple of (statistic, p_value)
        McNemar chi-squared statistic (with continuity correction) and p-value.
    """
    correct_a = np.asarray(correct_a, dtype=int)
    correct_b = np.asarray(correct_b, dtype=int)

    # b: A correct, B wrong.  c: A wrong, B correct.
    b = np.sum((correct_a == 1) & (correct_b == 0))
    c = np.sum((correct_a == 0) & (correct_b == 1))

    b = int(b)
    c = int(c)

    if b + c == 0:
        # No discordant pairs -- cannot reject null.
        return (0.0, 1.0)

    # McNemar's test with continuity correction.
    statistic = (abs(b - c) - 1) ** 2 / (b + c)

    # Chi-squared distribution with 1 degree of freedom.
    p_value = float(sp_stats.chi2.sf(statistic, df=1))

    return (float(statistic), p_value)
