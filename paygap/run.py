"""Run the pay-equity analysis, write reports/results.md and the charts in reports/figures/."""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st

from paygap import analysis

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
BLUE, ORANGE, INK, INK2, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
    "legend.frameon": False, "lines.linewidth": 2,
})


def pc(x, sign=True):
    return f"{x * 100:+.1f}%" if sign else f"{x * 100:.1f}%"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=150)
    plt.close(fig)


def forest(ax, labels, est, lo, hi, highlight=None):
    """Dot-and-interval rows, top to bottom, with the estimate written at the right edge."""
    for i, (e, l, h) in enumerate(zip(est, lo, hi)):
        y = len(est) - 1 - i
        color = BLUE if i == highlight else INK2
        ax.plot([l * 100, h * 100], [y, y], color=color, lw=2, solid_capstyle="round")
        ax.plot(e * 100, y, "o", ms=8, color=color, mec=SURFACE, mew=2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(est)), list(labels)[::-1])
    ax.grid(axis="y", visible=False)


def fig_ladder(r):
    lad = r["ladder"]
    labels = [f"{m}   R² {r2:.2f}" for m, r2 in zip(lad.model, lad.r2)]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    forest(ax, labels, lad.gap, lad.ci_low, lad.ci_high, highlight=len(lad) - 1)
    for i, g in enumerate(lad.gap):
        ax.text(16.5, len(lad) - 1 - i, pc(g), va="center", ha="right", color=INK)
    ax.set(xlim=(-4, 17), ylim=(-0.5, len(lad) - 0.5),
           xlabel="Women's pay relative to men's (%, 95% CI, HC3 robust)")
    ax.set_title("Controlling for job level removes the raw gap", pad=12)
    save(fig, "gap_by_model")


def fig_oaxaca(r):
    o = r["oaxaca"]
    blocks = o["by_block"].sort_values(ascending=False)
    names = list(blocks.index) + ["Same profile, different pay (unexplained)"]
    vals = np.r_[blocks.values, o["unexplained"]] * 100
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    y = np.arange(len(vals))[::-1]
    colors = [BLUE] * len(blocks) + [ORANGE]
    ax.barh(y, vals, color=colors, height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v + (0.1 if v >= 0 else -0.1), yi, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, names)
    ax.grid(axis="y", visible=False)
    ax.set(xlim=(-1, 5), xlabel="Contribution to the raw gap, women − men (log points × 100)")
    ax.bar(0, 0, color=BLUE, label=f"Explained by profile: {o['explained'] * 100:+.2f}")
    ax.bar(0, 0, color=ORANGE, label=f"Unexplained: {o['unexplained'] * 100:+.2f}")
    ax.legend(loc="lower right", title=f"Raw gap {o['raw'] * 100:+.2f}", title_fontproperties={"weight": "bold"},
              alignment="left")
    ax.set_title("Job level and role explain almost all of the raw gap", pad=12)
    save(fig, "oaxaca_decomposition")


def fig_levels(r):
    df = r["df"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for k, (gender, color, dx) in enumerate([("Female", BLUE, -0.18), ("Male", ORANGE, 0.18)]):
        groups = [df.loc[(df.JobLevel == lv) & (df.Gender == gender), "MonthlyIncome"] / 1000 for lv in range(1, 6)]
        bp = ax.boxplot(groups, positions=np.arange(1, 6) + dx, widths=0.3, patch_artist=True, showfliers=False,
                        medianprops={"color": SURFACE, "lw": 2}, whiskerprops={"color": color},
                        capprops={"color": color}, boxprops={"facecolor": color, "edgecolor": color})
        bp["boxes"][0].set_label("Women" if gender == "Female" else "Men")
    share = r["promotion"]["women_share_by_level"]
    for lv in range(1, 6):
        ax.text(lv, 21.3, f"{share[lv]:.0%} women", ha="center", color=INK2)
    ax.set_xticks(range(1, 6), [f"Level {lv}" for lv in range(1, 6)])
    ax.grid(axis="x", visible=False)
    ax.set(ylim=(0, 22.5), ylabel="Monthly income (thousands)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.88))
    ax.set_title("Pay is set by job level; within a level, women and men are paid alike", pad=12)
    save(fig, "pay_by_level")


def fig_diagnostics(r):
    d = r["diagnostics"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.9))
    a1.scatter(d["fitted"], d["resid"], s=8, color=BLUE, alpha=0.35, lw=0)
    a1.axhline(0, color=INK2, lw=1)
    a1.set(xlabel="Fitted log pay", ylabel="Residual")
    a1.set_title("Residuals vs. fitted", fontsize=11)
    (osm, osr), (slope, icpt, _) = st.probplot(d["resid"])
    a2.scatter(osm, osr, s=8, color=BLUE, alpha=0.35, lw=0)
    a2.plot(osm, slope * osm + icpt, color=INK2, lw=1)
    a2.set(xlabel="Normal quantiles", ylabel="Residual quantiles")
    a2.set_title("Normal Q–Q", fontsize=11)
    fig.suptitle("Full-model diagnostics: uneven spread, so robust standard errors", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    save(fig, "diagnostics")


def table(df, fmt):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt.get(c, str)(v) for c, v in zip(cols, row)) + " |")
    return "\n".join(lines)


def report(r):
    gap = {"gap": pc, "ci_low": pc, "ci_high": pc, "p_value": lambda v: f"{v:.3f}", "p_holm": lambda v: f"{v:.3f}",
           "r2": lambda v: f"{v:.3f}", "women_pct": lambda v: pc(v, False)}
    p, o, d, pr, u = r["pay"], r["oaxaca"], r["diagnostics"], r["promotion"], r["underpaid"]
    out = ["# Results (generated by `python -m paygap.run`)", "",
           f"- Employees: {r['n']['women']:,} women, {r['n']['men']:,} men",
           f"- Mean monthly income: women {p['mean_women']:,.0f}, men {p['mean_men']:,.0f}; "
           f"median: women {p['median_women']:,.0f}, men {p['median_men']:,.0f}", "",
           "## Gender gap by model", "", table(r["ladder"], gap), "",
           "## Robustness", "", table(r["robustness"], gap), "",
           "## By department (interaction model)", "", table(r["departments"], gap), "",
           "## Oaxaca–Blinder (log points × 100)", "",
           f"- Raw {o['raw'] * 100:+.2f} = explained {o['explained'] * 100:+.2f} + unexplained {o['unexplained'] * 100:+.2f}"]
    out += [f"  - {k}: {v * 100:+.2f}" for k, v in o["by_block"].sort_values(ascending=False).items()]
    out += ["", "## Diagnostics", "",
            f"- Breusch–Pagan p: log pay {d['bp_p_log']:.1e}, raw pay {d['bp_p_raw']:.1e}",
            f"- Correlation of job level with pay: {d['corr_level_pay']:.2f}",
            "- VIF: " + ", ".join(f"{k} {v:.2f}" for k, v in d["vif"].items()), "",
            "## Promotion (ordered logit of job level)", "",
            f"- Odds ratio for women: {pr['odds_ratio']:.2f} (95% CI {pr['ci_low']:.2f}–{pr['ci_high']:.2f}), "
            f"p = {pr['p_value']:.3f}",
            "- Women's share by level: " + ", ".join(f"L{k} {v:.0%}" for k, v in pr["women_share_by_level"].items()),
            "", "## Paid ≥20% below prediction", "",
            f"- {u['n_flagged']} employees flagged; rate: "
            + ", ".join(f"{k} {v:.1%}" for k, v in u["rate_by_gender"].items()) + f"; chi-square p = {u['p_value']:.3f}",
            ""]
    (ROOT / "reports" / "results.md").write_text("\n".join(out), encoding="utf-8")
    u["flagged"].to_csv(ROOT / "reports" / "underpaid_review_list.csv", index=False)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    r = analysis.analyse()
    report(r)
    fig_ladder(r)
    fig_oaxaca(r)
    fig_levels(r)
    fig_diagnostics(r)
    print("Wrote reports/results.md, reports/underpaid_review_list.csv and",
          len(list(FIG.glob("*.png"))), "figures")


if __name__ == "__main__":
    main()
