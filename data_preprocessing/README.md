# PhysioNet 2019 — MAP forecasting: environment and run order

Predict whether mean arterial pressure falls below 65 mmHg in the next 6 hours, from the first
hours of an ICU stay. This file covers **what to install** and **what to run, in what order**.
The reasoning behind each preprocessing decision is in
[`NOTES_setAB_and_preprocessing.md`](NOTES_setAB_and_preprocessing.md).

---

## 1. Environment

Everything runs in one conda env, `sc`:

```
/home/mhanyu/.conda/envs/sc/bin/python
```

| package | version | used for |
|---|---|---|
| python | 3.10.20 | |
| numpy | 2.2.6 | arrays, the stratified split RNG |
| pandas | 2.3.3 | the patient tables |
| scipy | 1.15.2 | optional — only the lab-clustering dendrogram in the EDA |
| scikit-learn | 1.7.2 | domain classifier, site AUROC, CV splitters |
| matplotlib | 3.10.8 | every figure |
| statsmodels | 0.14.6 | the case-mix adjustment (§6.2) |
| **torch** | **2.14.1+cpu** | `Dataset` / `DataLoader` only — not the split |
| ipykernel | 7.2.0 | the "Python (sc)" Jupyter kernel |
| IPython | 8.37.0 | |
| pyarrow | 25.0.1 | not required; present in the env |

Platform: Linux 4.18.0 (glibc 2.28), CPU only — nothing here needs a GPU.

### A required environment variable

torch loads the **system** `libstdc++`, which is older than the one pandas' compiled extensions
need. Importing torch before pandas therefore breaks pandas:

```
ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
```

The env ships a new enough copy, so preload it:

```bash
export LD_PRELOAD=/home/mhanyu/.conda/envs/sc/lib/libstdc++.so.6
```

For Jupyter this is already set permanently in
`~/.local/share/jupyter/kernels/sc/kernel.json` (backup: `kernel.json.bak`) — **restart the
kernel once** for it to take effect. Importing pandas *before* torch also works unaided, but the
preload is the robust fix.

### Reinstalling torch from scratch

```bash
/home/mhanyu/.conda/envs/sc/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

(`drasi` also has torch 2.8.0+cpu / Python 3.9 and still works, but its numpy 1.23 cannot read a
numpy-2 pickle — which is why the model input is exported as `.npz` rather than a pickle.)

---

## 2. Run order

```bash
# the env path is specific to the Great Lakes account; everything else is repo-relative
SC=/home/mhanyu/.conda/envs/sc/bin/python
export LD_PRELOAD=/home/mhanyu/.conda/envs/sc/lib/libstdc++.so.6
cd data_preprocessing          # this folder
```

All three scripts resolve their paths from their own location (`Path(__file__).parent`), so a
fresh clone runs without editing anything — only the raw data has to be downloaded first.

| # | step | produces | time |
|---|---|---|---|
| 1 | `EDA/PhysioNet2019_EDA.ipynb`, from the top through §6 (cells 0–26) | `datasets/physionet2019_all.pkl` | first run: the ~40,000-file download dominates; afterwards seconds, from the cache |
| 2 | same notebook, §6.1 – §6.4 (cells 27–34) | `datasets/physionet2019_harmonized.pkl`, `EDA/09*.csv/png` | ~45 s |
| 3 | `$SC model/export_model_input.py` | `datasets/physionet2019_model_input.npz` | 7 s |
| 4 | `$SC model/batch_effect.py` | `model/batch_effect_*.csv` | 25 s |
| 5 | `$SC model/physionet_data.py` | nothing — self-check only | 9 s |

Peak memory is 0.7 GB (step 3); everything fits comfortably on a login node.

### What each step does

**1. Load the raw data.** Reads ~40,000 `.psv` files into one table (1,552,210 patient-hours,
40,336 patients) and caches it as a pickle, so later runs load in seconds. The download needs
internet: run it on a login node, since compute nodes may have none.

**2. EDA §6.1–§6.4 — the preprocessing that matters.** §6.1–§6.3 measure the set A / set B
difference; **§6.4 is the cell that writes the harmonized table**, and it does two things:

- rounds the modelled variables onto a value grid both hospitals share (set A records fractional
  ages and MAP in thirds, which lets any tree-based model identify the hospital from number
  formatting alone — 99.5% accuracy from age decimals)
- cuts both hospitals to the same age range, dropping 414 set-B patients outside [18, 89]

Result: 1,536,483 rows, 39,922 patients (20,336 set A + 19,586 set B).

**3. Export the model input.** Flattens the harmonized table into an `.npz` any environment can
read, with `O2Sat`/`HospAdmTime` included as backup variables and `HospAdmTime` clipped to
[−720, +24] h.

**4. Batch-effect report.** The set A vs set B SMD table on the model's own features, plus a
check that the 70/15/15 split did not introduce an imbalance of its own. Not required to train,
but it is the number you quote in the write-up.

**5. Pipeline self-check.** Asserts the invariants: no patient in two splits, standardisation
fitted on train only, forward fill never looks ahead, `y[t, h-1] == MAP[t+h]`. Run it after any
change to the preprocessing.

### Re-running after a change

Steps 2 → 5 are a chain: if you change the harmonization in §6.4, re-run 3, 4 and 5. Step 1 is
cached and only needs re-running if the raw data changes.

---

## 3. Getting the data (nothing large is in git)

The repository holds **code, tables and figures — about 1.8 MB**. Everything else is derived and
is regenerated by the pipeline above, so `.gitignore` keeps it out:

| not in git | size | regenerated by |
|---|---|---|
| `datasets/training/` (40,336 `.psv` files) | 313 MB | the download below |
| `datasets/physionet2019_all.pkl` | 282 MB | notebook step 1 |
| `datasets/physionet2019_harmonized.pkl` | 279 MB | notebook step 2 (§6.4) |
| `datasets/physionet2019_model_input.npz` | 8.3 MB | `export_model_input.py`, 7 s |

**882 MB ignored, 1.8 MB committed.**

### Downloading the raw data

The data are open access (CC BY 4.0) from the
[PhysioNet Challenge 2019 page](https://physionet.org/content/challenge-2019/1.0.0/). The
notebook does this for you when `DOWNLOAD = True` and picks a method automatically, but either
command works on its own:

```bash
cd datasets

# fastest, if the AWS CLI is installed (no credentials needed)
aws s3 sync --no-sign-request \
    s3://physionet-open/challenge-2019/1.0.0/training/ training/

# or plain wget (Linux / Great Lakes); -c resumes an interrupted download
wget -r -N -c -np -nH --cut-dirs=4 \
    https://physionet.org/files/challenge-2019/1.0.0/training/
```

Expected result: `datasets/training/training_setA/` and `training_setB/`, 40,336 `.psv` files,
313 MB. **Run this on a login node** — compute nodes often have no internet. Re-running is safe;
both commands skip files that are already there.

Then set `DOWNLOAD = False` in the notebook's first cell and run it from the top: step 1 rebuilds
`physionet2019_all.pkl`, step 2 rebuilds the harmonized table, and steps 3–5 take under a minute.

---

## 4. Using the data in a model

```python
import sys; sys.path.insert(0, "model")
from physionet_data import make_loaders

loaders, sets, stats, patients, splits = make_loaders(
    variant="masked",      # "masked" | "ffill" | "median"
    features="primary",    # "primary" | "all"
    batch_size=64,
)

for batch in loaders["train"]:
    x, static = batch["x"], batch["static"]          # (B, T, C), (B, S)
    loss = bce(model(x, static), batch["event"])     # mask it with batch["origin"]
```

**Mask the loss.** `batch["origin"]` marks the hours that are legitimate forecast origins, and
`batch["y_mask"]` marks which future hours were actually measured. Unmasked, the model is trained
to predict MAP = 0 mmHg for every hour nobody measured.

| option | choices | default |
|---|---|---|
| `variant` | `"masked"` (keep missingness, add mask + time-since-last), `"ffill"` (carry forward), `"median"` (constant) | `"masked"` |
| `keep_mask` | `True` / `False` — hybrid: impute *and* keep the mask channels | per variant |
| `features` | `"primary"` = SBP, DBP, HR, Resp + Age, sex, ICULOS (MAP is target and input); `"all"` adds O2Sat + HospAdmTime | `"primary"` |
| `window` | `"complete"` (all 6 future hours measured) / `"any"` | `"complete"` |
| `event_rule` | `"any"` (one hour below 65) / `"sustained"` (two consecutive) | `"any"` |
| `exclude_hypotensive_now` | drop origins where the patient is already below 65 | `True` |

Splits are fixed: by patient, seed 123, 70/15/15, stratified on `site × ever-hypotensive` →
27,945 / 5,987 / 5,990 patients, each 49.1% set B.

---

## 5. Files

```
final_proj/
├── README.md                          this file
├── .gitignore                         keeps the 882 MB of data out of git
├── NOTES_setAB_and_preprocessing.md   why each decision was made, with the numbers
├── datasets/
│   ├── physionet2019_all.pkl          raw combined table          (step 1)
│   ├── physionet2019_harmonized.pkl   grids + age range aligned   (step 2)
│   └── physionet2019_model_input.npz  flat model input, 8 MB      (step 3)
├── EDA/
│   ├── PhysioNet2019_EDA.ipynb        steps 1–2
│   └── 0*.csv / 0*.png                every table and figure
└── model/
    ├── export_model_input.py          step 3
    ├── batch_effect.py                step 4
    ├── physionet_data.py              step 5 + the training-time API
    └── batch_effect_*.csv/.png         step 4 output
```

Notebook snapshots `PhysioNet2019_EDA.ipynb.bak` … `.bak6` are from before each round of edits,
newest last; delete when you no longer want them.
