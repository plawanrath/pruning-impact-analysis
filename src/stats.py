"""Statistical tests for bias amplification analysis."""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def chi_squared_test(srs_base, n_base, srs_comp, n_comp):
    table = np.array([
        [int(srs_base * n_base), n_base - int(srs_base * n_base)],
        [int(srs_comp * n_comp), n_comp - int(srs_comp * n_comp)],
    ])
    if table.min() < 0 or table.sum() == 0:
        return {"chi2": np.nan, "p": np.nan, "cohens_h": np.nan}
    chi2, p, dof, _ = stats.chi2_contingency(table)
    h = 2 * (np.arcsin(np.sqrt(srs_comp)) - np.arcsin(np.sqrt(srs_base)))
    return {"chi2": chi2, "p": p, "cohens_h": h}


def logistic_regression_pruning(df):
    """stereotype_chosen ~ sparsity + category + prune_method"""
    df = df.copy()
    letters = "ABC"
    df["stereotype_chosen"] = (
        df["parsed_answer"]
        == df["stereotype_target_index"].map(lambda i: letters[i])
    ).astype(int)
    X = pd.get_dummies(
        df[["sparsity", "category", "prune_method"]],
        columns=["category", "prune_method"],
        drop_first=True, dtype=float,
    )
    X = sm.add_constant(X)
    y = df["stereotype_chosen"]
    model = sm.Logit(y, X).fit(disp=0)
    return model.summary2()
