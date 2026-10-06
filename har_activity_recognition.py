"""
Human Activity Recognition from smartphone accelerometer + gyroscope data.

Pipeline: download UCI HAR -> load raw 6-channel windows -> build features
(statistics + FFT) -> ablation study -> compare 3 models -> evaluate -> plots.

Run:  python har_activity_recognition.py
Outputs: images/confusion_matrix.png, images/top_features.png
"""

# =====================================================================
# CELL 1: Imports + download the UCI HAR dataset
# =====================================================================
import os, glob, zipfile, time, urllib.request
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score

DATA_DIR, IMG_DIR = "data", "images"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

URL = "https://github.com/MaxBenChrist/human-activity-dataset/blob/master/UCI%20HAR%20Dataset.zip?raw=True"
ZIP_PATH = os.path.join(DATA_DIR, "har.zip")
if not os.path.exists(ZIP_PATH):
    print("Downloading dataset (~60 MB)...")
    urllib.request.urlretrieve(URL, ZIP_PATH)
with zipfile.ZipFile(ZIP_PATH) as z:
    z.extractall(os.path.join(DATA_DIR, "extracted"))

hits = glob.glob(os.path.join(DATA_DIR, "extracted", "**", "Inertial Signals"), recursive=True)
base = os.path.dirname(os.path.dirname(hits[0])) + os.sep
print("dataset folder:", base)


# =====================================================================
# CELL 2: Load raw signals (7352 train windows, 2947 test windows)
# Each window = 128 readings at 50 Hz = 2.56 seconds, 6 channels.
# =====================================================================
chans = ["total_acc_x", "total_acc_y", "total_acc_z",
         "body_gyro_x", "body_gyro_y", "body_gyro_z"]

def load_split(split):
    arrs = [np.loadtxt(f"{base}{split}/Inertial Signals/{c}_{split}.txt") for c in chans]
    X = np.stack(arrs, axis=2)                                   # (windows, 128, 6)
    y = np.loadtxt(f"{base}{split}/y_{split}.txt").astype(int)   # labels 1..6
    return X, y

Xtr, ytr = load_split("train")
Xte, yte = load_split("test")
print("Xtr:", Xtr.shape, " Xte:", Xte.shape)

FS = 50   # sampling rate (readings per second)
names = ["walking", "upstairs", "downstairs", "sitting", "standing", "laying"]


# =====================================================================
# CELL 3: Feature functions (three families, switch on/off with flags)
#
# 1) basic stats   : mean, std, min, max, range
# 2) extra stats   : median, 25th pct, 75th pct, IQR
# 3) frequency     : FFT-based "how fast / how regular is the wobble"
# =====================================================================
ch_names = ["ax", "ay", "az", "gx", "gy", "gz", "amag", "gmag"]
BANDS = [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 25)]  # Hz

def add_magnitudes(X):
    amag = np.sqrt((X[:, :, 0:3] ** 2).sum(axis=2, keepdims=True))   # sqrt(x^2+y^2+z^2)
    gmag = np.sqrt((X[:, :, 3:6] ** 2).sum(axis=2, keepdims=True))
    return np.concatenate([X, amag, gmag], axis=2)                    # (windows, 128, 8)

def basic_stats(X):
    return [X.mean(1), X.std(1), X.min(1), X.max(1), X.max(1) - X.min(1)]

def extra_stats(X):
    q25, med, q75 = np.percentile(X, [25, 50, 75], axis=1)
    return [med, q25, q75, q75 - q25]

def freq_stats(X):
    Xc = X - X.mean(1, keepdims=True)                 # remove constant level (gravity) first
    P = np.abs(np.fft.rfft(Xc, axis=1)) ** 2          # power at each frequency
    freqs = np.fft.rfftfreq(X.shape[1], 1 / FS)       # Hz value of each slot
    total = P.sum(1) + 1e-12
    dom_freq = freqs[P[:, 1:, :].argmax(1) + 1]       # strongest frequency (skip slot 0)
    share = P / total[:, None, :]
    entropy = -(share * np.log(share + 1e-12)).sum(1) # low = clean rhythm, high = messy
    band_energy = [P[:, (freqs >= lo) & (freqs < hi), :].sum(1) / total for lo, hi in BANDS]
    return [dom_freq, entropy, np.log(total)] + band_energy

BASIC_N = ["mean", "std", "min", "max", "range"]
EXTRA_N = ["median", "q25", "q75", "iqr"]
FREQ_N = ["domfreq", "entropy", "logpower"] + [f"band{lo}-{hi}Hz" for lo, hi in BANDS]

def build_features(X, use_extra=True, use_fft=True):
    X = add_magnitudes(X)
    blocks, stat_names = basic_stats(X), list(BASIC_N)
    if use_extra:
        blocks += extra_stats(X); stat_names += EXTRA_N
    if use_fft:
        blocks += freq_stats(X); stat_names += FREQ_N
    names_out = [f"{c}_{s}" for s in stat_names for c in ch_names]
    return np.concatenate(blocks, axis=1), names_out


# =====================================================================
# CELL 4: Ablation study - what did each feature family actually buy us?
# Same model (Random Forest, 200 trees), different feature sets.
# =====================================================================
def fit_eval(model, Ftr, Fte):
    model.fit(Ftr, ytr)
    return accuracy_score(yte, model.predict(Fte))

rows = []
for label, ue, uf in [("basic only (baseline)",    False, False),
                      ("+ median/percentiles",     True,  False),
                      ("+ FFT frequency features", False, True),
                      ("+ both",                   True,  True)]:
    Ftr_, n_ = build_features(Xtr, ue, uf)
    Fte_, _ = build_features(Xte, ue, uf)
    acc = fit_eval(RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1), Ftr_, Fte_)
    rows.append((label, len(n_), round(acc, 3)))

print(pd.DataFrame(rows, columns=["feature set", "n_features", "accuracy"]).to_string(index=False))


# =====================================================================
# CELL 5: Compare models on the full feature set
# =====================================================================
Ftr, feat_names = build_features(Xtr, True, True)
Fte, _ = build_features(Xte, True, True)

models = {
    "RandomForest 500":      RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1),
    "ExtraTrees 500":        ExtraTreesClassifier(n_estimators=500, random_state=0, n_jobs=-1),
    "HistGradientBoosting":  HistGradientBoostingClassifier(random_state=0),
}
results, fitted = {}, {}
for mname, m in models.items():
    t = time.time()
    m.fit(Ftr, ytr)
    results[mname] = accuracy_score(yte, m.predict(Fte))
    fitted[mname] = m
    print(f"{mname:22s} accuracy = {results[mname]:.3f}  ({time.time()-t:.0f}s)")

best_name = max(results, key=results.get)
clf = fitted[best_name]
pred = clf.predict(Fte)
print("\nBest model:", best_name)
print(classification_report(yte, pred, target_names=names))


# =====================================================================
# CELL 6: Confusion matrix of the best model
# =====================================================================
ConfusionMatrixDisplay.from_predictions(
    yte, pred, display_labels=names, xticks_rotation=45)
plt.title(f"Confusion matrix - {best_name}")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.show()


# =====================================================================
# CELL 7: Which features matter?
# Uses a Random Forest because it exposes feature_importances_.
# =====================================================================
rf = fitted["RandomForest 500"]
imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)

imp.head(15).plot.barh()
plt.gca().invert_yaxis()
plt.title("Top 15 features (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "top_features.png"), dpi=150, bbox_inches="tight")
plt.show()

print("Top 'mean' features (gravity direction lives here, esp. ax/ay/az_mean):")
print(imp[[n for n in imp.index if n.endswith("_mean")]].head(5).round(4))

print("\nTop frequency features:")
freq_keys = ("domfreq", "entropy", "logpower", "Hz")
print(imp[[n for n in imp.index if any(k in n for k in freq_keys)]].head(5).round(4))


# =====================================================================
# CELL 8: Live demo - predict one test window
# =====================================================================
i = 0                                    # try 100, 500, 2000, ...
guess = names[clf.predict(Fte[i:i+1])[0] - 1]
truth = names[yte[i] - 1]
print(f"Window {i}: model says '{guess}', truth is '{truth}'")
