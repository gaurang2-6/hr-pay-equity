"""Gender pay-equity analysis: staged OLS, decomposition, diagnostics, promotion check, individual flags."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, norm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from paygap import data

CONTROLS = ("C(JobLevel) + C(JobRole) + TotalWorkingYears + YearsAtCompany"
            " + C(PerformanceRating) + C(Education)")

# Each stage adds one block of controls, so we can see which one closes the raw gap.
LADDER = [
    ("Raw (no controls)", "log_pay ~ female"),
    ("+ job level", "log_pay ~ female + C(JobLevel)"),
    ("+ job role", "log_pay ~ female + C(JobLevel) + C(JobRole)"),
    ("+ experience and tenure", "log_pay ~ female + C(JobLevel) + C(JobRole) + TotalWorkingYears + YearsAtCompany"),
    ("+ rating and education (full model)", f"log_pay ~ female + {CONTROLS}"),
]
FULL = LADDER[-1][1]
UNDERPAID = np.log(0.8)   # paid at least 20% less than the model predicts for the same profile

# Which control each regressor belongs to, for the decomposition chart.
BLOCKS = {"JobLevel": "Job level", "JobRole": "Job role", "TotalWorkingYears": "Experience",
          "YearsAtCompany": "Tenure", "PerformanceRating": "Rating", "Education": "Education"}


def pct(log_points):
    """Log-point coefficient → percentage difference in pay."""
    return np.expm1(log_points)


def fit(formula, df):
    # Residuals fan out with pay (see diagnostics), so every inference uses HC3 robust errors.
    return smf.ols(formula, df).fit(cov_type="HC3")


def gender_effect(model, term="female"):
    lo, hi = model.conf_int().loc[term]
    return {"gap": pct(model.params[term]), "ci_low": pct(lo), "ci_high": pct(hi), "p_value": model.pvalues[term]}


def ladder(df):
    rows = []
    for label, formula in LADDER:
        m = fit(formula, df)
        rows.append({"model": label, **gender_effect(m), "r2": m.rsquared})
    return pd.DataFrame(rows)


def robustness(df):
    """The full-model gap under alternative specifications."""
    specs = {
        "Full model (log pay, HC3)": FULL,
        "Add age": FULL + " + Age",
        "Job level as a number": FULL.replace("C(JobLevel)", "JobLevel"),
        "Add department": FULL + " + C(Department)",
        "Drop job level (level may itself be biased)": FULL.replace("C(JobLevel) + ", ""),
    }
    rows = [{"specification": k, **gender_effect(fit(f, df))} for k, f in specs.items()]
    # Pay in raw units instead of logs: report the gap relative to the men's mean.
    m = fit(FULL.replace("log_pay", "MonthlyIncome"), df)
    base = df.loc[df.female == 0, "MonthlyIncome"].mean()
    lo, hi = m.conf_int().loc["female"]
    rows.append({"specification": "Raw monthly income (not logged)", "gap": m.params.female / base,
                 "ci_low": lo / base, "ci_high": hi / base, "p_value": m.pvalues.female})
    return pd.DataFrame(rows)


def department_gaps(df):
    """Gender gap inside each department from one interaction model, Holm-corrected for three tests."""
    m = fit(FULL + " + C(Department) + female:C(Department)", df)
    names = list(m.params.index)
    rows = []
    for dept, g in df.groupby("Department"):
        w = np.zeros(len(names))
        w[names.index("female")] = 1
        inter = f"female:C(Department)[T.{dept}]"
        if inter in names:
            w[names.index(inter)] = 1
        est, se = w @ m.params.values, np.sqrt(w @ m.cov_params().values @ w)
        rows.append({"department": dept, "employees": len(g), "women_pct": g.female.mean(), "gap": pct(est),
                     "ci_low": pct(est - 1.96 * se), "ci_high": pct(est + 1.96 * se),
                     "p_value": 2 * norm.sf(abs(est / se))})
    out = pd.DataFrame(rows).sort_values("employees", ascending=False)
    out["p_holm"] = multipletests(out.p_value, method="holm")[1]
    return out


def oaxaca(df):
    """Two-fold Oaxaca–Blinder split of the raw log-pay gap (women − men), pooled coefficients.

    explained   = (mean X of women − mean X of men) · β   → differences in level, role, experience…
    unexplained = the female coefficient                  → different pay for the same profile
    With an intercept and the female dummy in the model, the two add up exactly to the raw gap.
    """
    m = smf.ols(FULL, df).fit()
    X = pd.DataFrame(m.model.exog, columns=m.model.exog_names)
    is_f = df.female.values == 1
    diff = X[is_f].mean() - X[~is_f].mean()
    contrib = (diff * m.params).drop(["Intercept", "female"])
    by_block = contrib.groupby(lambda c: next(v for k, v in BLOCKS.items() if k in c)).sum()
    raw = df.log_pay[is_f].mean() - df.log_pay[~is_f].mean()
    return {"raw": raw, "explained": contrib.sum(), "unexplained": m.params.female, "by_block": by_block}


def diagnostics(df):
    m = smf.ols(FULL, df).fit()
    m_raw = smf.ols(FULL.replace("log_pay", "MonthlyIncome"), df).fit()
    num = df[["TotalWorkingYears", "YearsAtCompany", "JobLevel", "Education", "PerformanceRating", "female"]]
    Xv = num.assign(const=1.0).values
    vif = pd.Series([variance_inflation_factor(Xv, i) for i in range(num.shape[1])], index=num.columns)
    return {
        "resid": m.resid, "fitted": m.fittedvalues,
        "bp_p_log": het_breuschpagan(m.resid, m.model.exog)[1],
        "bp_p_raw": het_breuschpagan(m_raw.resid, m_raw.model.exog)[1],
        "vif": vif.sort_values(ascending=False),
        "corr_level_pay": df.JobLevel.corr(df.MonthlyIncome),
        "cond_number": m.condition_number,
    }


def promotion(df):
    """Are women placed at lower job levels than men with the same experience? (ordered logit)"""
    X = df[["female", "TotalWorkingYears", "YearsAtCompany", "Education"]]
    m = OrderedModel(df.JobLevel, X, distr="logit").fit(method="bfgs", disp=0)
    lo, hi = m.conf_int().loc["female"]
    share = pd.crosstab(df.JobLevel, df.Gender, normalize="index")["Female"]
    return {"odds_ratio": np.exp(m.params.female), "ci_low": np.exp(lo), "ci_high": np.exp(hi),
            "p_value": m.pvalues.female, "women_share_by_level": share,
            "counts_by_level": pd.crosstab(df.JobLevel, df.Gender)}


def underpaid(df):
    """Employees paid ≥20% below what their profile predicts — the list HR would review case by case.

    Predicted pay comes from the controls *without* gender, so the flag can't absorb a gender effect.
    """
    m = smf.ols(FULL.replace("female + ", ""), df).fit()
    flagged = m.resid < UNDERPAID
    table = pd.crosstab(df.Gender, flagged)
    rates = flagged.groupby(df.Gender).mean()
    return {"n_flagged": int(flagged.sum()), "rate_by_gender": rates,
            "p_value": chi2_contingency(table)[1], "flagged": df.loc[flagged, ["EmployeeNumber", "Gender",
            "JobRole", "JobLevel", "MonthlyIncome"]].assign(predicted=np.exp(m.fittedvalues[flagged]).round(0))}


def analyse() -> dict:
    df = data.employees()
    women, men = df[df.female == 1], df[df.female == 0]
    return {
        "df": df,
        "n": {"women": len(women), "men": len(men)},
        "pay": {"mean_women": women.MonthlyIncome.mean(), "mean_men": men.MonthlyIncome.mean(),
                "median_women": women.MonthlyIncome.median(), "median_men": men.MonthlyIncome.median()},
        "ladder": ladder(df),
        "robustness": robustness(df),
        "departments": department_gaps(df),
        "oaxaca": oaxaca(df),
        "diagnostics": diagnostics(df),
        "promotion": promotion(df),
        "underpaid": underpaid(df),
    }
