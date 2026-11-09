import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from paygap import analysis as a


def synthetic(n=3000, gap=-0.05, seed=0):
    """Fake company in the same shape as the IBM file, with a known gender gap in log pay."""
    rng = np.random.default_rng(seed)
    female = rng.binomial(1, 0.4, n)
    level = rng.integers(1, 6, n)
    exp_years = level * 4 + rng.integers(0, 6, n)
    df = pd.DataFrame({
        "female": female, "Gender": np.where(female == 1, "Female", "Male"), "JobLevel": level,
        "JobRole": rng.choice(["Analyst", "Engineer", "Sales"], n), "TotalWorkingYears": exp_years,
        "YearsAtCompany": rng.integers(0, 10, n), "PerformanceRating": rng.choice([3, 4], n),
        "Education": rng.integers(1, 6, n), "EmployeeNumber": np.arange(n),
        "Department": rng.choice(["R&D", "Sales", "HR"], n),
    })
    df["log_pay"] = 7.5 + 0.5 * level + 0.01 * exp_years + gap * female + rng.normal(0, 0.1, n)
    df["MonthlyIncome"] = np.exp(df.log_pay)
    return df


def test_full_model_recovers_a_planted_gap():
    eff = a.gender_effect(a.fit(a.FULL, synthetic(gap=-0.05)))
    assert eff["ci_low"] < np.expm1(-0.05) < eff["ci_high"] and eff["p_value"] < 0.001


def test_no_false_alarm_when_there_is_no_gap():
    eff = a.gender_effect(a.fit(a.FULL, synthetic(gap=0.0)))
    assert eff["p_value"] > 0.05 and eff["ci_low"] < 0 < eff["ci_high"]


def test_raw_model_gap_is_the_difference_in_mean_log_pay():
    df = synthetic()
    raw = a.ladder(df).iloc[0]
    expected = df.log_pay[df.female == 1].mean() - df.log_pay[df.female == 0].mean()
    assert np.isclose(np.log1p(raw.gap), expected)


def test_oaxaca_parts_add_up_to_the_raw_gap():
    o = a.oaxaca(synthetic())
    assert np.isclose(o["explained"] + o["unexplained"], o["raw"])
    assert np.isclose(o["by_block"].sum(), o["explained"])


def test_department_gap_for_the_baseline_department_is_the_female_coefficient():
    df = synthetic()
    m = a.fit(a.FULL + " + C(Department) + female:C(Department)", df)
    base = sorted(df.Department.unique())[0]
    got = a.department_gaps(df).set_index("department").loc[base, "gap"]
    assert np.isclose(got, np.expm1(m.params.female))


def test_hc3_errors_match_statsmodels_directly():
    df = synthetic()
    ours = a.fit(a.FULL, df)
    ref = smf.ols(a.FULL, df).fit().get_robustcov_results(cov_type="HC3")
    i = ours.model.exog_names.index("female")
    assert np.isclose(ours.bse["female"], ref.bse[i])
