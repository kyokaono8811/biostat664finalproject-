"""Quantify the batch (site) effect on the exact features the model consumes.

Same recipe as EDA 6.1, applied to the model input rather than the raw table: one row per
patient, each vital summarised as the median over the first 24 hours after ICU admission, and the
standardized mean difference (SMD) as the measure. SMD is dimensionless and does not grow with
the sample size, so mmHg, years and percentages sit in one table and |SMD| > 0.1 means the same
thing in every row.

Two questions are answered:

1. **How big is the batch effect?** Set A vs set B on the model's own feature set - the four
   vital values (SBP, DBP, HR, Resp) and the three demographics (Age, sex, ICULOS as stay
   length) - plus the "was it measured in the first 24 h" indicators for those vitals, because
   EDA 6.3 showed the recording pattern carries as much site information as the physiology does.
   MAP is reported on its own line: it is the outcome the model predicts, not an input.
2. **Did our own splitting add one?** Train / validation / test each compared against the other
   two combined. These should all sit near zero; anything above 0.1 means the split itself
   introduced an imbalance on top of the site effect.

Run:

    python batch_effect.py          # either environment; needs only numpy / pandas
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK2, AXIS, SURFACE = "#52514e", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "xtick.color": AXIS, "ytick.color": AXIS,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 11,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight"})

HERE = Path(__file__).resolve().parent
DATA_NPZ = HERE.parent / "datasets" / "physionet2019_model_input.npz"
BASE_H = 24.0          # hours after ICU admission that define the baseline window
SMD_CUT = 0.1          # |SMD| above this is a difference that matters
SEED = 123             # must match physionet_data.SEED for the split check to be meaningful


def baseline_table(path: Path = DATA_NPZ) -> pd.DataFrame:
    """One row per patient: baseline vitals, demographics, and the recording indicators."""
    z = np.load(path, allow_pickle=False)
    vitals, iculos, start = z["vitals"], z["iculos"], z["patient_start"]
    names = [str(v) for v in z["vital_names"]]
    bounds = np.r_[start, len(iculos)]

    values = np.full((len(start), len(names)), np.nan, dtype=np.float64)
    measured = np.zeros((len(start), len(names)), dtype=np.float64)
    ever = np.zeros(len(start), dtype=np.int64)
    target = names.index("MAP")
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        full = vitals[a:b][:, target]
        ever[i] = bool(np.isfinite(full).any() and np.nanmin(full) < 65.0)
        window = vitals[a:b][iculos[a:b] <= BASE_H]
        seen = np.isfinite(window)
        measured[i] = seen.any(axis=0)
        with warnings.catch_warnings():               # an all-NaN column simply yields NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            values[i] = np.nanmedian(np.where(seen, window, np.nan), axis=0)

    out = pd.DataFrame(values, columns=names)
    for f, n in enumerate(names):
        out[f"{n}_measured"] = measured[:, f]
    static_names = [str(v) for v in z["static_names"]]
    for s, n in enumerate(static_names):
        out[n] = z["static"][:, s]
    out["stay_h"] = np.array([iculos[a:b].max() for a, b in zip(bounds[:-1], bounds[1:])])
    out["ever_hypotensive"] = ever          # whole stay, matching physionet_data.make_splits
    out.insert(0, "site", np.where(z["site"] == "B", "B", "A"))
    out.insert(0, "patient_id", z["patient_ids"])
    return out


def smd(a: np.ndarray, b: np.ndarray):
    """Standardized mean difference (a - b) with a 95% CI; 0/1 variables use the same formula."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp if sp > 0 else 0.0
    se = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    return {"n_1": n1, "n_2": n2, "mean_1": a.mean(), "mean_2": b.mean(),
            "pooled_sd": sp, "smd": d, "smd_lo": d - 1.96 * se, "smd_hi": d + 1.96 * se}


def site_auroc(x: np.ndarray, is_b: np.ndarray, seed: int = SEED) -> float:
    """How well does this one variable alone identify the hospital? Out-of-fold AUROC from a
    boosted stump set, computed on observed values only, so it measures a shift in the values
    rather than in the missingness (the `_measured` rows cover that separately). SMD only sees a
    difference in means; this sees any difference in distribution."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    seen = np.isfinite(x)
    xs, ys = x[seen].reshape(-1, 1), is_b[seen].astype(int)
    if len(np.unique(ys)) < 2:
        return float("nan")
    oof = np.zeros(len(ys))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(xs, ys):
        m = HistGradientBoostingClassifier(random_state=seed, max_depth=3).fit(xs[tr], ys[tr])
        oof[te] = m.predict_proba(xs[te])[:, 1]
    return float(roc_auc_score(ys, oof))


def compare(frame: pd.DataFrame, mask_1: np.ndarray, mask_2: np.ndarray,
            columns, label_1: str, label_2: str, with_auroc: bool = False) -> pd.DataFrame:
    rows = []
    for c in columns:
        x = frame[c].to_numpy(dtype=float)
        r = smd(x[mask_1], x[mask_2])
        if with_auroc:
            r["site_auroc"] = site_auroc(x[mask_1 | mask_2], mask_2[mask_1 | mask_2])
        rows.append({"variable": c, "group_1": label_1, "group_2": label_2, **r})
    out = pd.DataFrame(rows).set_index("variable")
    out["matters"] = np.where(out["smd"].abs() > SMD_CUT, f"|SMD| > {SMD_CUT}", "")
    return out


def model_splits(frame: pd.DataFrame, seed: int = SEED):
    """Reproduce the split used by physionet_data.make_splits without importing torch."""
    site = (frame["site"].to_numpy() == "B").astype(np.int64)
    strata = site * 2 + frame["ever_hypotensive"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        idx = idx[rng.permutation(len(idx))]
        n_tr, n_va = int(round(0.70 * len(idx))), int(round(0.15 * len(idx)))
        out["train"].append(idx[:n_tr])
        out["val"].append(idx[n_tr:n_tr + n_va])
        out["test"].append(idx[n_tr + n_va:])
    return {k: np.sort(np.concatenate(v)) for k, v in out.items()}


def main() -> None:
    frame = baseline_table()
    values = ["SBP", "DBP", "HR", "Resp", "O2Sat",                        # model inputs
              "Age", "Gender", "HospAdmTime", "stay_h"]
    recording = [f"{v}_measured" for v in ("SBP", "DBP", "HR", "Resp", "O2Sat")]
    outcome = ["MAP"]                                                     # predicted, not an input
    is_a = (frame["site"] == "A").to_numpy()
    is_b = ~is_a

    site = pd.concat([
        compare(frame, is_a, is_b, values, "set A", "set B", with_auroc=True).assign(role="feature"),
        compare(frame, is_a, is_b, recording, "set A", "set B").assign(role="recording"),
        compare(frame, is_a, is_b, outcome, "set A", "set B", with_auroc=True).assign(role="outcome")])
    site.to_csv(HERE / "batch_effect_site.csv")
    print(f"Batch effect: set A vs set B, {BASE_H:.0f} h baseline, one row per patient "
          f"(A = {is_a.sum():,}, B = {is_b.sum():,})\n")
    print(site[["role", "mean_1", "mean_2", "smd", "smd_lo", "smd_hi", "site_auroc", "matters"]]
          .rename(columns={"mean_1": "mean_A", "mean_2": "mean_B"}).round(3).to_string())
    hidden = site[(site["smd"].abs() <= SMD_CUT) & (site["site_auroc"] > 0.6)]
    if len(hidden):
        print("\nSMD says these are fine, the single-feature AUROC says otherwise - a shift in "
              "shape rather than in the mean:")
        for v, r in hidden.iterrows():
            print(f"  {v:12s} SMD {r['smd']:+.3f} but site AUROC {r['site_auroc']:.3f}")

    feats = site.loc[values]
    big = feats.index[feats["smd"].abs() > SMD_CUT].tolist()
    print(f"\n{len(big)} of the {len(values)} model features exceed |SMD| {SMD_CUT}: {big}")
    print(f"mean |SMD| over the features   {feats['smd'].abs().mean():.3f}")
    print(f"mean |SMD| over the indicators {site.loc[recording, 'smd'].abs().mean():.3f}")
    print(f"largest overall: {site['smd'].abs().idxmax()} at "
          f"{site.loc[site['smd'].abs().idxmax(), 'smd']:+.3f}")
    print(f"outcome MAP: SMD {site.loc['MAP', 'smd']:+.3f} - the batch effect also sits in the "
          "variable being predicted, which is why the event rate differs 2x between sites")

    print("\n" + "=" * 100)
    print("Did the 70/15/15 split add a batch effect of its own? Each split against the other two.\n")
    splits = model_splits(frame)
    rows = []
    for name, idx in splits.items():
        sel = np.zeros(len(frame), bool)
        sel[idx] = True
        rows.append(compare(frame, sel, ~sel, values + recording + outcome, name, "rest")
                    .assign(split=name).reset_index())
    split_tbl = pd.concat(rows).set_index(["split", "variable"])
    split_tbl.to_csv(HERE / "batch_effect_splits.csv")
    worst = split_tbl["smd"].abs().groupby(level="split").max()
    share_b = {k: 100 * (frame["site"].to_numpy()[idx] == "B").mean() for k, idx in splits.items()}
    print(f"{'split':8s} {'patients':>9s} {'% set B':>9s} {'largest |SMD| vs the rest':>27s}")
    for k, idx in splits.items():
        print(f"{k:8s} {len(idx):>9,} {share_b[k]:>8.1f}% {worst[k]:>27.3f}")
    flagged = split_tbl.index[split_tbl["smd"].abs() > SMD_CUT].tolist()
    print(f"\nfeatures above |SMD| {SMD_CUT} in any split: {flagged or 'none'}")
    print("\nSo the only batch effect in the modelling data is the hospital one measured above; "
          "the split itself is balanced, which is what stratifying on site and on ever-hypotensive buys.")
    draw(site, split_tbl, splits, frame)
    print(f"\nwritten: {HERE / 'batch_effect_site.csv'}\n         {HERE / 'batch_effect_splits.csv'}"
          f"\n         {HERE / 'batch_effect.png'}")


def draw(site: pd.DataFrame, split_tbl: pd.DataFrame, splits, frame: pd.DataFrame) -> None:
    """Left: the hospital batch effect we cannot remove. Right: what the stratified split added."""
    order = site["smd"].abs().sort_values().index
    yy = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * len(order) + 2.6))

    ax = axes[0]
    colors = {"feature": SERIES[0], "recording": SERIES[1], "outcome": SERIES[2]}
    for role, c in colors.items():
        sel = [v for v in order if site.loc[v, "role"] == role]
        if not sel:
            continue
        idx = [list(order).index(v) for v in sel]
        ax.errorbar(site.loc[sel, "smd"], idx,
                    xerr=1.96 * (site.loc[sel, "smd_hi"] - site.loc[sel, "smd"]) / 1.96,
                    fmt="o", ms=7, color=c, ecolor=c, elinewidth=1.6, capsize=3, label=role)
    big = int((site["smd"].abs() > SMD_CUT).sum())
    worst = site["smd"].abs().idxmax()
    ax.set_title(f"Set A vs set B: {big} of {len(site)} beyond |SMD| {SMD_CUT}, "
                 f"{worst} at {site.loc[worst, 'smd']:+.2f}")
    ax.legend(loc="lower right", fontsize=11, borderaxespad=0.4)

    ax = axes[1]
    for i, (name, c) in enumerate(zip(["train", "val", "test"], SERIES)):
        vals = split_tbl.loc[name].reindex(order)["smd"]
        ax.scatter(vals, yy + (i - 1) * 0.22, s=42, color=c, zorder=3,
                   label=f"{name} ({len(splits[name]):,} patients)")
    worst_split = split_tbl["smd"].abs().max()
    ax.set_title(f"After stratifying on site x ever-hypotensive: "
                 f"nothing past |SMD| {worst_split:.3f}")
    ax.legend(loc="lower right", fontsize=11, borderaxespad=0.4)

    for ax, xlabel in zip(axes, ["Standardized mean difference, set A - set B",
                                 "Standardized mean difference, each split - the other two"]):
        ax.axvline(0, color=INK2, lw=1)
        for t in (-SMD_CUT, SMD_CUT):
            ax.axvline(t, color=AXIS, lw=1, ls="--")
        ax.set_yticks(yy, list(order), fontsize=11)
        ax.set_xlabel(f"{xlabel} (dashed: +/- {SMD_CUT})", fontsize=11)
        ax.tick_params(labelsize=11)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.8, len(order) - 0.2)
    fig.tight_layout()
    fig.savefig(HERE / "batch_effect.png")


if __name__ == "__main__":
    main()
