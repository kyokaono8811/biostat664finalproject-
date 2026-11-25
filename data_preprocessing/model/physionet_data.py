"""PyTorch input pipeline for the PhysioNet 2019 MAP task.

Prediction target: will MAP fall below 65 mmHg at any point in the next 6 hours? Every batch
carries `event` (the label, one per hour) and `origin` (which hours are legitimate forecast
origins); the loss is a BCE masked by `origin`. The label is configurable, and the defaults are
the defensible ones - see PhysioNetSequences for `event_rule`, `window` and
`exclude_hypotensive_now`, and the event-rate table printed by the self-check for what each
choice costs. The continuous future MAP values stay in `y`/`y_mask` so the same batches also
feed a Gaussian forecast model that derives P(min MAP < 65) from its predictive distribution.

Run the self-check with the environment that has torch:

    /home/mhanyu/.conda/envs/drasi/bin/python physionet_data.py

Two preprocessing variants, selected with `variant=`:

  "masked"  Nothing is filled in. A dense tensor still needs a number in every cell, so a missing
            entry becomes 0 *after* standardisation (the training mean, i.e. no evidence either
            way) and is accompanied by two extra channels: `mask` saying whether that cell was a
            real measurement, and `delta`, the hours since this variable was last measured. The
            network is told what it does not know, which is the honest version of "leave it
            missing".

  "ffill"   Last observation carried forward within the patient; the gap before a patient's first
            measurement is filled with the training-split median. This is the clinical default
            (a value stays valid until replaced). With `keep_mask=False` the model cannot tell an
            imputed value from a measured one.

  "median"  Every missing cell gets the training-split median of that variable, with no memory of
            the patient at all. Deliberately the weakest option: it is the baseline that says how
            much the carry-forward in "ffill" is actually worth.

`keep_mask=True` on either imputing variant gives a fourth configuration - imputed values plus
the mask channels, so the model sees both the filled number and the fact that it was filled.

Linear interpolation is **not** offered. Interpolating a gap uses the measurement after it, which
is a future value relative to the forecast origin, so it leaks the horizon into the input. It is
fine for a retrospective description and wrong for this task.

Both variants are built from the *same* splits and the *same* training statistics, so a model
trained on one can be compared with a model trained on the other.

Decisions worth knowing about:

* Splitting is by patient, never by row: all hours of a patient land in exactly one split.
* Set A and set B are mixed in all three splits, as asked, and stratified so each split holds the
  same A/B ratio and the same share of patients who ever become hypotensive. `site` travels with
  every batch, so test metrics can still be reported per hospital - without that, mixing hides
  exactly the shift the EDA measured (MAP differs by 8 mmHg, SMD -0.66).
* Standardisation uses the training split only. Validation and test are transformed with the
  training statistics and never contribute to them.
* `ICULOS` enters as the running hour counter known at time t, never as the patient's total stay
  length. The final ICULOS is a summary of the future and would leak the horizon.
* The input grid is the harmonized table from EDA 6.4, so the model cannot identify the hospital
  from number formatting. The `mask` channels are a separate matter: whether DBP was measured is
  close to a site label (EDA 6.3), so a masked-variant model should always be scored per site.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
DATA_NPZ = HERE.parent / "datasets" / "physionet2019_model_input.npz"
SEED = 123
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)     # train / validation / test, by patient
HORIZON = 6                              # forecast MAP 1..6 hours ahead
THRESHOLD = 65.0                         # mmHg, the hypotension event used in the EDA
TARGET = "MAP"
VARIANTS = ("masked", "ffill", "median")

# The variable set to report on. MAP is the target and also its own strongest predictor, so it is
# always an input channel too. O2Sat and HospAdmTime are exported and kept available, but they are
# NOT part of the primary model: start with the four vitals and three demographics below, and only
# reach for the extras if the primary model needs help. HospAdmTime in particular identifies the
# hospital on its own at AUROC 0.71 (model/batch_effect.py), so adding it changes what an A/B
# comparison means.
PRIMARY_VITALS = ["MAP", "SBP", "DBP", "HR", "Resp"]        # + ICULOS, which is its own channel
PRIMARY_STATIC = ["Age", "Gender"]
BACKUP_VITALS = ["O2Sat"]
BACKUP_STATIC = ["HospAdmTime"]
FEATURE_SETS = {"primary": (PRIMARY_VITALS, PRIMARY_STATIC),
                "all": (PRIMARY_VITALS + BACKUP_VITALS, PRIMARY_STATIC + BACKUP_STATIC)}


# --------------------------------------------------------------------------------------- loading

@dataclass
class Patients:
    """Per-patient sequences, still in raw units with NaN where nothing was measured."""
    ids: np.ndarray                 # (N,)   patient id strings
    site: np.ndarray                # (N,)   0 = set A, 1 = set B
    vitals: List[np.ndarray]        # N x (T, F) float32, NaN = not measured
    iculos: List[np.ndarray]        # N x (T,)  hours since ICU admission, known at time t
    static: np.ndarray              # (N, S) float32: Age, Gender
    vital_names: List[str]
    static_names: List[str]

    def __len__(self) -> int:
        return len(self.ids)


def load_patients(path: Path = DATA_NPZ) -> Patients:
    z = np.load(path, allow_pickle=False)
    start = z["patient_start"]
    # read each array ONCE: indexing the NpzFile decompresses it afresh every time, and every
    # slice would pin a separate full copy of the base array
    all_vitals, all_iculos = z["vitals"], z["iculos"]
    bounds = np.r_[start, len(all_iculos)]
    vitals = [all_vitals[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    iculos = [all_iculos[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    return Patients(ids=z["patient_ids"], site=(z["site"] == "B").astype(np.int64),
                    vitals=vitals, iculos=iculos, static=z["static"],
                    vital_names=[str(v) for v in z["vital_names"]],
                    static_names=[str(v) for v in z["static_names"]])


# ---------------------------------------------------------------------------------- splitting

def make_splits(p: Patients, seed: int = SEED,
                fractions: Sequence[float] = SPLIT_FRACTIONS) -> Dict[str, np.ndarray]:
    """Patient-level split, stratified by site and by whether the patient is ever hypotensive."""
    target = p.vital_names.index(TARGET)
    ever = np.array([np.nanmin(v[:, target]) < THRESHOLD if np.isfinite(v[:, target]).any() else False
                     for v in p.vitals], dtype=np.int64)
    strata = p.site * 2 + ever
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for s in np.unique(strata):                      # split inside each stratum, then concatenate
        idx = np.flatnonzero(strata == s)
        idx = idx[rng.permutation(len(idx))]
        n_train = int(round(fractions[0] * len(idx)))
        n_val = int(round(fractions[1] * len(idx)))
        out["train"].append(idx[:n_train])
        out["val"].append(idx[n_train:n_train + n_val])
        out["test"].append(idx[n_train + n_val:])
    splits = {k: np.sort(np.concatenate(v)) for k, v in out.items()}
    seen = np.concatenate(list(splits.values()))
    assert len(seen) == len(p) and len(np.unique(seen)) == len(p), "splits must partition patients"
    return splits


# ------------------------------------------------------------------------------- normalisation

@dataclass
class TrainStats:
    """Everything the transform needs, estimated on the training patients only."""
    mean: np.ndarray            # (F,) over observed values
    sd: np.ndarray              # (F,)
    median: np.ndarray          # (F,) used to fill the leading gap in the ffill variant
    static_mean: np.ndarray
    static_sd: np.ndarray
    iculos_mean: float
    iculos_sd: float
    target_index: int

    def standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.sd


def fit_stats(p: Patients, train_idx: np.ndarray) -> TrainStats:
    stacked = np.concatenate([p.vitals[i] for i in train_idx])          # (rows, F) with NaN
    icu = np.concatenate([p.iculos[i] for i in train_idx])
    static = p.static[train_idx]
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stacked, axis=0)
        sd = np.nanstd(stacked, axis=0)
        median = np.nanmedian(stacked, axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    s_mean = np.nanmean(static, axis=0)
    s_sd = np.nanstd(static, axis=0)
    s_sd[~np.isfinite(s_sd) | (s_sd == 0)] = 1.0
    return TrainStats(mean=mean.astype(np.float32), sd=sd.astype(np.float32),
                      median=median.astype(np.float32), static_mean=s_mean.astype(np.float32),
                      static_sd=s_sd.astype(np.float32), iculos_mean=float(icu.mean()),
                      iculos_sd=float(icu.std() or 1.0),
                      target_index=p.vital_names.index(TARGET))


# ------------------------------------------------------------------------------------- dataset

def _time_since_last(mask: np.ndarray, iculos: np.ndarray) -> np.ndarray:
    """Hours since each variable was last measured; starts at 0 at the first row."""
    T, F = mask.shape
    delta = np.zeros((T, F), dtype=np.float32)
    last = np.full(F, iculos[0], dtype=np.float32)
    for t in range(T):
        delta[t] = iculos[t] - last
        last = np.where(mask[t] > 0, iculos[t], last)
    return delta


def _forward_fill(values: np.ndarray, median: np.ndarray) -> np.ndarray:
    out = values.copy()
    for f in range(out.shape[1]):
        col = out[:, f]
        seen = np.isfinite(col)
        if not seen.any():
            col[:] = median[f]
            continue
        idx = np.where(seen, np.arange(len(col)), 0)
        np.maximum.accumulate(idx, out=idx)
        col[:] = col[idx]
        col[:np.argmax(seen)] = median[f]          # nothing measured yet -> training median
    return out


class PhysioNetSequences(Dataset):
    """One item per patient: the whole stay, with the 6-hour MAP targets attached to every hour."""

    def __init__(self, p: Patients, indices: np.ndarray, stats: TrainStats,
                 variant: str = "masked", keep_mask: Optional[bool] = None,
                 horizon: int = HORIZON, threshold: float = THRESHOLD,
                 event_rule: str = "any", window: str = "complete",
                 exclude_hypotensive_now: bool = True, features: str = "primary"):
        """event_rule: "any" = one future hour below the threshold; "sustained" = two consecutive
        observed hours below it. window: "complete" = all `horizon` future hours were measured;
        "any" = at least one was (cheaper on data, but a window with 5 unmeasured hours can hide
        an event and so labels it 0). exclude_hypotensive_now drops origins where the patient is
        already below the threshold - predicting an event that has already started is not the
        early-warning task."""
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if event_rule not in ("any", "sustained") or window not in ("any", "complete"):
            raise ValueError("event_rule must be 'any'/'sustained', window 'any'/'complete'")
        if features not in FEATURE_SETS:
            raise ValueError(f"features must be one of {tuple(FEATURE_SETS)}")
        vit, sta = FEATURE_SETS[features]
        missing = [v for v in vit + sta if v not in p.vital_names + p.static_names]
        if missing:
            raise ValueError(f"{missing} are not in the exported data")
        self.features = features
        self.vital_names = list(vit)
        self.static_names = list(sta)
        self.v_idx = np.array([p.vital_names.index(v) for v in vit])
        self.s_idx = np.array([p.static_names.index(v) for v in sta])
        self.target_index = self.vital_names.index(TARGET)
        self.p, self.indices, self.stats = p, np.asarray(indices), stats
        self.variant, self.horizon, self.threshold = variant, horizon, threshold
        self.event_rule, self.window = event_rule, window
        self.exclude_hypotensive_now = exclude_hypotensive_now
        self.keep_mask = (variant == "masked") if keep_mask is None else keep_mask

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def n_channels(self) -> int:
        f = len(self.vital_names) + 1                        # vitals + ICULOS
        if self.keep_mask:
            f += len(self.vital_names)                       # mask channels
        if self.variant == "masked":
            f += len(self.vital_names)                       # time-since-last channels
        return f

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        j = self.indices[i]
        st = self.stats
        raw, icu = self.p.vitals[j][:, self.v_idx], self.p.iculos[j]
        mean, sd, med = st.mean[self.v_idx], st.sd[self.v_idx], st.median[self.v_idx]
        mask = np.isfinite(raw).astype(np.float32)

        # targets come from future rows only, and keep their own observed-mask so the loss can
        # marginalise over hours where MAP was never measured
        future = np.full((len(raw), self.horizon), np.nan, dtype=np.float32)
        tgt = raw[:, self.target_index]
        for h in range(1, self.horizon + 1):
            if h < len(raw):
                future[:-h, h - 1] = tgt[h:]
        y_mask = np.isfinite(future).astype(np.float32)
        below = (np.where(y_mask > 0, future, np.inf) < self.threshold)
        if self.event_rule == "sustained":
            event = (below[:, :-1] & below[:, 1:]).any(axis=1).astype(np.float32)
        else:
            event = below.any(axis=1).astype(np.float32)
        n_future = y_mask.sum(axis=1)
        origin = (mask[:, self.target_index] > 0)
        origin &= (n_future == self.horizon) if self.window == "complete" else (n_future > 0)
        if self.exclude_hypotensive_now:
            now = np.where(mask[:, self.target_index] > 0, raw[:, self.target_index], np.inf)
            origin &= (now >= self.threshold)

        if self.variant == "ffill":
            values = _forward_fill(raw, med)
        elif self.variant == "median":
            values = np.where(np.isfinite(raw), raw, med)
        else:
            values = raw
        x = (values - mean) / sd
        x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)   # masked: 0 == the training mean
        channels = [x, ((icu - st.iculos_mean) / st.iculos_sd).astype(np.float32)[:, None]]
        if self.keep_mask:
            channels.append(mask)
        if self.variant == "masked":
            channels.append(np.log1p(_time_since_last(mask, icu)))

        static = ((self.p.static[j][self.s_idx] - st.static_mean[self.s_idx])
                  / st.static_sd[self.s_idx]).astype(np.float32)
        return {
            "x": torch.from_numpy(np.concatenate(channels, axis=1)),
            "mask": torch.from_numpy(mask),
            "static": torch.from_numpy(np.nan_to_num(static)),
            "y": torch.from_numpy(np.nan_to_num(future)),          # mmHg, not standardised
            "y_mask": torch.from_numpy(y_mask),
            "event": torch.from_numpy(event),
            "origin": torch.from_numpy(origin.astype(np.float32)),
            "site": torch.tensor(self.p.site[j], dtype=torch.long),
            "length": torch.tensor(len(raw), dtype=torch.long),
        }


def collate_pad(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad the stays in a batch to equal length; `length` feeds pack_padded_sequence."""
    lengths = torch.stack([b["length"] for b in batch])
    T = int(lengths.max())
    out: Dict[str, torch.Tensor] = {}
    for key in ("x", "mask", "y", "y_mask", "event", "origin"):
        ref = batch[0][key]
        shape = (len(batch), T) + tuple(ref.shape[1:])
        padded = torch.zeros(shape, dtype=ref.dtype)
        for i, b in enumerate(batch):
            padded[i, :b[key].shape[0]] = b[key]
        out[key] = padded
    out["static"] = torch.stack([b["static"] for b in batch])
    out["site"] = torch.stack([b["site"] for b in batch])
    out["length"] = lengths
    out["pad_mask"] = (torch.arange(T)[None, :] < lengths[:, None]).float()
    out["origin"] = out["origin"] * out["pad_mask"]
    return out


def make_loaders(variant: str = "masked", batch_size: int = 64, seed: int = SEED,
                 keep_mask: Optional[bool] = None, num_workers: int = 0,
                 path: Path = DATA_NPZ, features: str = "primary", **kwargs):
    """Everything in one call: load, split, fit the training statistics, wrap in DataLoaders."""
    p = load_patients(path)
    splits = make_splits(p, seed=seed)
    stats = fit_stats(p, splits["train"])
    sets = {k: PhysioNetSequences(p, idx, stats, variant=variant, keep_mask=keep_mask,
                                  features=features, **kwargs)
            for k, idx in splits.items()}
    gen = torch.Generator().manual_seed(seed)
    loaders = {k: DataLoader(ds, batch_size=batch_size, shuffle=(k == "train"),
                             collate_fn=collate_pad, num_workers=num_workers,
                             generator=gen if k == "train" else None)
               for k, ds in sets.items()}
    return loaders, sets, stats, p, splits


# ---------------------------------------------------------------------------------- self-check

def _check() -> None:
    import time
    t0 = time.time()
    p = load_patients()
    splits = make_splits(p)
    stats = fit_stats(p, splits["train"])
    print(f"patients {len(p):,} | vitals {p.vital_names} | static {p.static_names} "
          f"| loaded in {time.time() - t0:.1f}s\n")

    # forward fill must never look ahead, and the leading gap must come from the training median
    probe = np.array([[np.nan], [10.0], [np.nan], [np.nan], [20.0]], dtype=np.float32)
    filled = _forward_fill(probe, np.array([99.0], dtype=np.float32))
    assert np.allclose(filled.ravel(), [99, 10, 10, 10, 20]), filled.ravel()
    print("forward fill carries the last measurement forward only (no look-ahead)")

    # y[t, h-1] must be MAP at t+h, taken from the raw series
    probe_ds = PhysioNetSequences(p, splits["train"][:1], fit_stats(p, splits["train"]), variant="masked")
    j, item = splits["train"][0], probe_ds[0]
    raw_map = p.vitals[j][:, p.vital_names.index(TARGET)]
    for h in range(1, HORIZON + 1):
        want = raw_map[h:h + 5]
        got = item["y"][:len(want), h - 1].numpy()
        keep = np.isfinite(want)
        assert np.allclose(got[keep], want[keep]), f"target misaligned at horizon {h}"
    print(f"targets line up: y[t, h-1] == MAP[t+h] for h = 1..{HORIZON}")

    train_ids = set(p.ids[splits["train"]])
    for k in ("val", "test"):
        assert not (train_ids & set(p.ids[splits[k]])), f"patient overlap between train and {k}"
    print("no patient appears in more than one split")

    target = p.vital_names.index(TARGET)
    print(f"\n{'split':6s} {'patients':>9s} {'% set B':>8s} {'% ever MAP<65':>14s} {'hours':>10s}")
    for k, idx in splits.items():
        ever = np.mean([np.nanmin(p.vitals[i][:, target]) < THRESHOLD
                        if np.isfinite(p.vitals[i][:, target]).any() else False for i in idx])
        hours = sum(len(p.iculos[i]) for i in idx)
        print(f"{k:6s} {len(idx):>9,} {100 * p.site[idx].mean():>7.1f}% {100 * ever:>13.1f}% {hours:>10,}")

    print(f"\ntraining statistics (observed values, training split only):")
    for f, name in enumerate(p.vital_names):
        print(f"  {name:5s} mean {stats.mean[f]:7.2f}  sd {stats.sd[f]:6.2f}  median {stats.median[f]:7.2f}")
    print(f"  Age   mean {stats.static_mean[0]:7.2f}  sd {stats.static_sd[0]:6.2f}")

    print("\nfeature sets available:")
    for name in FEATURE_SETS:
        ds = PhysioNetSequences(p, splits["train"][:1], stats, variant="masked", features=name)
        print(f"  {name:8s} vitals {ds.vital_names}")
        print(f"           static {ds.static_names} + ICULOS  -> {ds.n_channels} channels (masked)")
    print("  primary is the default: start there, the backup variables are a fallback")

    for variant in VARIANTS:
        loaders, sets, _, _, _ = make_loaders(variant=variant, batch_size=8)
        batch = next(iter(loaders["train"]))
        ds = sets["train"]
        print(f"\nvariant '{variant}' (features='{ds.features}'): {ds.n_channels} input channels, "
              f"keep_mask={ds.keep_mask}")
        for k, v in batch.items():
            print(f"  {k:9s} {tuple(v.shape)}  {str(v.dtype).replace('torch.', '')}")
        assert torch.isfinite(batch["x"]).all(), "x must not contain NaN"
        x = batch["x"][:, :, :len(ds.vital_names)]
        filled = x[(batch["mask"] == 0) & (batch["pad_mask"][:, :, None] > 0)]
        if variant == "masked" and filled.numel():
            assert torch.all(filled == 0), "masked variant must leave unmeasured cells at 0"
            print("  unmeasured cells are exactly 0 (the standardised training mean)")
        if variant in ("ffill", "median") and filled.numel():
            uniq = torch.unique(filled).numel()
            print(f"  unmeasured cells are filled: {100 * (filled != 0).float().mean():.0f}% "
                  f"non-zero, {uniq:,} distinct filled values "
                  f"({'one per variable means a single constant' if uniq <= 12 else 'carried forward from earlier measurements'})")

    loaders, sets, _, _, splits = make_loaders(variant="masked", batch_size=256)
    n_origin = n_event = n_hours = 0
    per_site = {0: [0, 0], 1: [0, 0]}
    for batch in loaders["test"]:
        o = batch["origin"]
        n_origin += int(o.sum()); n_hours += int(batch["pad_mask"].sum())
        n_event += float((batch["event"] * o).sum())
        for s in (0, 1):
            sel = (batch["site"] == s)
            if sel.any():
                per_site[s][0] += float((batch["origin"][sel]).sum())
                per_site[s][1] += float((batch["event"][sel] * batch["origin"][sel]).sum())
    print(f"\ntest split: {n_hours:,} patient-hours, {n_origin:,} usable forecast origins "
          f"(complete windows, patient not already below {THRESHOLD:.0f}), "
          f"event rate {100 * n_event / n_origin:.1f}%")
    for s, name in ((0, "set A"), (1, "set B")):
        print(f"  {name}: {per_site[s][0]:,.0f} origins, event rate "
              f"{100 * per_site[s][1] / per_site[s][0]:.1f}%")
    print(f"\ntotal self-check time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    _check()
