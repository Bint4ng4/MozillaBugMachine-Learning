"""
task_kurva_belajar.py — Task 11 Okt 2026: cleaning menyeluruh + pelatihan dengan kurva belajar + uji statistik.

Prompt yang disetujui user (11 Okt 2026), ringkas:
  1. Cleaning full_tailcut (A+TAILCUT+C+H+I+D+E1+F+E2+G); rangkuman raw vs full_tailcut; jangan mengklaim "bersih".
     Atas arahan user ("bugs itu dataset lengkap, pakai aja") ditambah lengan full_initial
     (A+INITIAL+C+H+I+D+E1+F+E2+G): laporan awal dari bugs*.csv + cleaning lengkap.
  2. Validation set = ekor train kronologis (indeks asli 25.000–28.505).
  3. TF-IDF + SGDClassifier (log loss, bobot kelas dari train), maks 50 epoch, early stopping val loss (patience 5).
     Grafik loss & macro-F1 per epoch; learning curve LogReg & LinearSVC.
  4. Evaluasi akhir SEKALI di test: macro-F1 + CI bootstrap 1000, F1 per kelas, confusion matrix, bandingkan dengan
     raw (bootstrap berpasangan) + uji z (McNemar, akurasi) dan uji t berpasangan antar 5 seed (pelengkap).
  5. Analisis baik/belum: macro-F1 > 0,138 dan > 0,25; gap train-val <= 5 poin; F1 blocker & trivial > 0; stabil
     antar seed.
  6. Keluaran: results_task_<tanggal>/ (PNG, CSV, ringkasan.md). MASTER tidak diubah.

Pembagian data (indeks asli `_orig_idx`): train_fit <= 24.999 | val 25.000–28.505 | test >= 28.507.
Test hanya dipakai di fase evaluasi akhir; contoh dokumen diambil dari train_fit.

Pakai:  python task_kurva_belajar.py [--out-dir results_task_20261011] [--seeds 5] [--max-epochs 50]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from scipy.stats import norm as normal_dist, ttest_rel  # noqa: E402
from sklearn.linear_model import SGDClassifier  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, log_loss  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cleaning as c  # noqa: E402
from experiment import SEVERITY_ORDER, macro_f1_resamples, make_model, make_vectorizer  # noqa: E402

VAL_MIN = 25000
ARMS = {"raw": [], "full_tailcut": c.CONFIGS["full_tailcut"], "full_initial": c.CONFIGS["full_initial"]}
ARM_LABEL = {"raw": "raw (archive apa adanya)", "full_tailcut": "full_tailcut", "full_initial": "full_initial (laporan awal)"}
# Palet: dataviz reference palette, 3 slot pertama (lolos validasi all-pairs, light). Aqua < 3:1 -> label langsung + CSV.
COL = {"raw": "#2a78d6", "full_tailcut": "#eb6834", "full_initial": "#1baf7a", "train": "#2a78d6", "val": "#eb6834"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e1"
AI_SHEETS = {"AI_pembanding_8Okt": "audit_model_pembanding.csv", "AI_lembarA_Haiku": "audit_penilai_A.csv",
             "AI_lembarB_Sonnet": "audit_penilai_B.csv"}

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
                     "legend.frameon": False})


def md_table(df: pd.DataFrame) -> str:
    """Tabel markdown sederhana (tanpa paket tabulate)."""
    cols = [str(x) for x in df.columns]
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def ntok(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.split().str.len().fillna(0).astype(int)


def split3(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    o = df["_orig_idx"]
    return {"train": df[o < VAL_MIN].reset_index(drop=True),
            "val": df[(o >= VAL_MIN) & (o <= c.TRAIN_MAX)].reset_index(drop=True),
            "test": df[o >= c.TEST_MIN].reset_index(drop=True)}


def balanced_weights(y: np.ndarray) -> dict[str, float]:
    cls, cnt = np.unique(y, return_counts=True)
    return {k: len(y) / (len(cls) * v) for k, v in zip(cls, cnt)}


def lcp_tokens(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# ---------------------------------------------------------------------------------------------------- cleaning
def summarize(arms: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in arms.items():
        t = df["Description"].fillna("").astype(str)
        tok = ntok(t)
        sp = split3(df)
        dist = df["Severity"].value_counts(normalize=True).mul(100)
        rows.append({"lengan": name, "dokumen": len(df), "train_fit": len(sp["train"]), "val": len(sp["val"]),
                     "test": len(sp["test"]), "token_mean": round(float(tok.mean()), 1),
                     "token_median": int(tok.median()), "token_p99": int(tok.quantile(.99)),
                     "kosakata": len(set(w for x in t for w in x.split())),
                     "penanda_POST_REPORT_pct": round(100 * float(t.str.contains(c.POST_REPORT).mean()), 2),
                     "penanda_EXT_pct": round(100 * float(t.str.contains(c.POST_REPORT_EXT).mean()), 2),
                     **{f"pct_{k}": round(float(dist.get(k, 0)), 2) for k in SEVERITY_ORDER}})
    return pd.DataFrame(rows)


def structural_residual(df0: pd.DataFrame, arms: dict[str, pd.DataFrame], min_sisa: int = 3) -> pd.DataFrame:
    """Sisa token pasca-laporan per lengan, memakai laporan awal (pasangan utama) yang diproses tahap per-dokumen
    yang sama (A, INITIAL, C, H, I, D, E1). sisa = token lengan - awalan bersama dengan laporan awal terproses."""
    ref_stages = {"raw": None, "full_tailcut": ["A", "INITIAL", "C", "H", "I", "D", "E1"],
                  "full_initial": ["A", "INITIAL", "C", "H", "I", "D", "E1"]}
    n_awal = c.load_initial_pairs()
    raw_a = quiet(c.stage_a_structure, df0.copy())
    raw_a = raw_a[raw_a["_orig_idx"].isin(n_awal.index)]
    init_proc = quiet(c.run_config, df0, "awal_proc", ref_stages["full_tailcut"])[0].set_index("_orig_idx")["Description"]
    rows = []
    for name, df in arms.items():
        d = df[df["_orig_idx"].isin(n_awal.index)].set_index("_orig_idx")["Description"].fillna("").astype(str)
        if name == "raw":
            a = raw_a.set_index("_orig_idx")
            ref = pd.Series({i: " ".join(t.split()[:int(n_awal[i])]) for i, t in a["Description"].items()})
            d = d.map(lambda x: " ".join(x.split()))
        else:
            ref = init_proc
        common = d.index.intersection(ref.index)
        sisa = pd.Series({i: len(d[i].split()) - lcp_tokens(d[i].split(), ref[i].split()) for i in common})
        rows.append({"lengan": name, "n_berpasangan": len(common),
                     "masih_terkontaminasi_pct": round(100 * float((sisa >= min_sisa).mean()), 2),
                     "sisa_token_median": float(sisa.median()), "sisa_token_rata2": round(float(sisa.mean()), 1)})
    return pd.DataFrame(rows)


def quote_residual(df0: pd.DataFrame, arms: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Batas bawah dari kutipan audit AI: % item ber-label K yang kutipannya MASIH ADA di teks lengan."""
    key = pd.read_csv("audit_langkah3/audit_KUNCI_jangan_dibuka_penilai.csv", encoding="utf-8-sig")
    a = quiet(c.stage_a_structure, df0.copy())
    tr = a[a["_orig_idx"] <= c.TRAIN_MAX].reset_index(drop=True)
    te = a[a["_orig_idx"] >= c.TEST_MIN].reset_index(drop=True)
    key["orig_idx"] = [(tr if s == "train" else te).iloc[int(r)]["_orig_idx"] for s, r in zip(key["split"], key["row_in_split"])]
    per_doc = ["C", "H", "I", "D"]
    rows = []
    for rater, f in AI_SHEETS.items():
        sh = pd.read_csv(f, encoding="utf-8-sig").merge(key[["item", "group", "orig_idx"]], on="item")
        sh = sh[sh["label"].astype(str).str.strip().str.upper() == "K"]
        q = pd.DataFrame({"Description": sh["kutipan_bukti"].fillna("").astype(str).values,
                          "Severity": "normal", "_orig_idx": sh["orig_idx"].values})
        q_proc = q.copy()
        for s in per_doc:
            q_proc = quiet(c.STAGES[s][1], q_proc)
        for name, df in arms.items():
            text = df.set_index("_orig_idx")["Description"].fillna("").astype(str)
            qq = q if name == "raw" else q_proc
            for g in ("A_dipotong", "B_tanpa_penanda"):
                m = (sh["group"] == g).values
                found, n = 0, 0
                for oi, quote in zip(qq["_orig_idx"].values[m], qq["Description"].values[m]):
                    if oi not in text.index:
                        continue
                    n += 1
                    qn, tn = " ".join(quote.lower().split()), " ".join(text[oi].lower().split())
                    found += bool(qn) and (qn in tn or " ".join(qn.split()[:4]) in tn)
                rows.append({"lengan": name, "penilai": rater, "kelompok": g, "item_K_ada_di_lengan": n,
                             "kutipan_masih_ada": found, "batas_bawah_pct_dari_50": round(100 * found / 50, 1)})
    return pd.DataFrame(rows)


def examples(arms: dict[str, pd.DataFrame], k: int = 3, seed: int = 20261011, words: int = 70) -> str:
    sets = [set(split3(df)["train"]["_orig_idx"]) for df in arms.values()]
    common = sorted(set.intersection(*sets))
    pick = np.random.default_rng(seed).choice(common, size=k, replace=False)
    out = []
    for n, oi in enumerate(pick, 1):
        out.append(f"### Contoh {n} (indeks asli {oi}, train)\n")
        for name, df in arms.items():
            t = str(df.set_index("_orig_idx").at[oi, "Description"])
            toks = t.split()
            short = " ".join(toks[:words]) + (f" … *(+{len(toks) - words} token)*" if len(toks) > words else "")
            out.append(f"- **{name}** ({len(toks)} token): {short}")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------- SGD
def train_sgd(Xtr, ytr, Xva, yva, seed: int, max_epochs: int, patience: int, alpha: float = 1e-4):
    classes = np.array(sorted(SEVERITY_ORDER))
    cw = balanced_weights(ytr)
    w_tr = np.array([cw[v] for v in ytr])
    w_va = np.array([cw.get(v, 1.0) for v in yva])        # kelas yang tidak ada di train: bobot 1 (default SGD)
    clf = SGDClassifier(loss="log_loss", alpha=alpha, class_weight=cw, random_state=seed)
    rng = np.random.default_rng(seed)
    hist, best, wait = [], (np.inf, 0, None, None), 0
    for ep in range(1, max_epochs + 1):
        perm = rng.permutation(Xtr.shape[0])
        clf.partial_fit(Xtr[perm], ytr[perm], classes=classes)
        p_tr, p_va = clf.predict_proba(Xtr), clf.predict_proba(Xva)
        rec = {"epoch": ep, "loss_train": log_loss(ytr, p_tr, labels=clf.classes_, sample_weight=w_tr),
               "loss_val": log_loss(yva, p_va, labels=clf.classes_, sample_weight=w_va),
               "f1_train": f1_score(ytr, clf.classes_[p_tr.argmax(1)], average="macro", zero_division=0),
               "f1_val": f1_score(yva, clf.classes_[p_va.argmax(1)], average="macro", zero_division=0)}
        hist.append(rec)
        if rec["loss_val"] < best[0] - 1e-4:
            best, wait = (rec["loss_val"], ep, clf.coef_.copy(), clf.intercept_.copy()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    clf.coef_, clf.intercept_ = best[2], best[3]
    return clf, pd.DataFrame(hist), best[1]


# ---------------------------------------------------------------------------------------------------- grafik
def plot_sgd(hist: pd.DataFrame, best_ep: pd.DataFrame, path: Path) -> None:
    arms = list(ARMS)
    fig, ax = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey="row")
    for j, arm in enumerate(arms):
        h = hist[hist["lengan"] == arm]
        for i, (metric, ylab) in enumerate((("loss", "log loss (berbobot kelas)"), ("f1", "macro-F1"))):
            a = ax[i, j]
            for part in ("train", "val"):
                g = h.groupby("epoch")[f"{metric}_{part}"]
                mean, lo, hi = g.mean(), g.min(), g.max()
                a.fill_between(mean.index, lo, hi, color=COL[part], alpha=0.15, linewidth=0)
                a.plot(mean.index, mean.values, color=COL[part], linewidth=2, label=part)
                a.annotate(part, (mean.index[-1], mean.values[-1]), xytext=(4, 0), textcoords="offset points",
                           va="center", fontsize=9, color=INK2)
            be = best_ep[best_ep["lengan"] == arm]["best_epoch"]
            for e in be:
                a.axvline(e, color=INK2, linewidth=0.8, linestyle=":", alpha=0.6)
            if i == 0:
                a.set_title(ARM_LABEL[arm])
            if j == 0:
                a.set_ylabel(ylab)
            if i == 1:
                a.set_xlabel("epoch")
            a.xaxis.set_major_locator(MaxNLocator(integer=True))
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("SGD (log loss): train vs val per epoch — garis = rata-rata 5 seed, pita = min–maks, "
                 "titik-titik = epoch terbaik (early stopping)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_learning_curve(lc: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for a, model in zip(ax, ("logreg", "linearsvc")):
        d = lc[lc["model"] == model]
        for arm in ARMS:
            g = d[d["lengan"] == arm].sort_values("n_train")
            a.plot(g["n_train"], g["f1_val"], color=COL[arm], linewidth=2, marker="o", markersize=5,
                   label=f"{arm} — val")
            a.plot(g["n_train"], g["f1_train"], color=COL[arm], linewidth=2, linestyle="--", alpha=0.8,
                   label=f"{arm} — train")
        a.set_title({"logreg": "Logistic Regression", "linearsvc": "LinearSVC"}[model])
        a.set_xlabel("jumlah dokumen train_fit")
        a.set_ylim(0, 1.02)
    ax[0].set_ylabel("macro-F1")
    ax[1].legend(loc="center right", fontsize=8)
    fig.suptitle("Learning curve: garis putus = train, garis utuh = val", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cms: dict[str, np.ndarray], title: str, path: Path) -> None:
    cmap = LinearSegmentedColormap.from_list("seq", [SURFACE, "#9cc3ee", "#2a78d6", "#123f7a"])
    fig, ax = plt.subplots(1, len(cms), figsize=(5 * len(cms), 4.6))
    for a, (arm, cm) in zip(np.atleast_1d(ax), cms.items()):
        norm_cm = cm / cm.sum(1, keepdims=True).clip(min=1)
        a.imshow(norm_cm, cmap=cmap, vmin=0, vmax=1)
        a.grid(False)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = norm_cm[i, j]
                a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.55 else INK)
        a.set_xticks(range(len(SEVERITY_ORDER)), SEVERITY_ORDER, rotation=45, ha="right")
        a.set_yticks(range(len(SEVERITY_ORDER)), SEVERITY_ORDER)
        a.set_xlabel("prediksi")
        a.set_title(arm)
    np.atleast_1d(ax)[0].set_ylabel("sebenarnya")
    fig.suptitle(title + " (dinormalisasi per baris)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------- utama
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="archive")
    ap.add_argument("--out-dir", default="results_task_20261011")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max-features", type=int, default=20000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=42)
    ap.add_argument("--lc-fracs", nargs="+", type=float, default=[0.1, 0.25, 0.5, 0.75, 1.0])
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    report: dict = {"config": vars(args)}

    # 1. CLEANING ---------------------------------------------------------------------------------------------
    print("1. Cleaning (raw, full_tailcut, full_initial) ...")
    df0 = pd.read_csv(Path(args.archive) / "sev.csv")
    arms = {name: quiet(c.run_config, df0, name, stages)[0] for name, stages in ARMS.items()}
    summ = summarize(arms)
    summ.to_csv(out / "ringkasan_cleaning.csv", index=False)
    struct = structural_residual(df0, arms)
    struct.to_csv(out / "sisa_kontaminasi_struktural.csv", index=False)
    quotes = quote_residual(df0, arms)
    quotes.to_csv(out / "sisa_kontaminasi_kutipan_audit.csv", index=False)
    ex_md = examples(arms)
    print(summ.to_string(index=False))
    print(struct.to_string(index=False))

    # 2-3. PELATIHAN (train_fit + val saja) -------------------------------------------------------------------
    splits = {a: split3(df) for a, df in arms.items()}
    hist_all, best_rows, sgd_models, vecs, lc_rows, lin_models = [], [], {}, {}, [], {}
    for arm, sp in splits.items():
        print(f"\n2. {arm}: train_fit {len(sp['train'])} | val {len(sp['val'])} | test {len(sp['test'])}")
        ytr, yva = sp["train"]["Severity"].values, sp["val"]["Severity"].values
        vec = make_vectorizer(args.max_features)
        Xtr = vec.fit_transform(sp["train"]["Description"].fillna("").astype(str))
        Xva = vec.transform(sp["val"]["Description"].fillna("").astype(str))
        vecs[arm] = vec
        for s in range(args.seeds):
            clf, h, be = train_sgd(Xtr, ytr, Xva, yva, s, args.max_epochs, args.patience)
            h["lengan"], h["seed"] = arm, s
            hist_all.append(h)
            bh = h[h["epoch"] == be].iloc[0]
            best_rows.append({"lengan": arm, "seed": s, "best_epoch": be, "epoch_dijalankan": int(h["epoch"].max()),
                              "f1_train_best": bh["f1_train"], "f1_val_best": bh["f1_val"],
                              "loss_val_best": bh["loss_val"]})
            sgd_models[(arm, s)] = clf
            print(f"   SGD seed {s}: epoch terbaik {be}/{int(h['epoch'].max())} | F1 train {bh['f1_train']:.3f} "
                  f"val {bh['f1_val']:.3f}")
        # learning curve LogReg & LinearSVC (subsampel train_fit, seed 0), evaluasi di val
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(sp["train"]))
        for frac in args.lc_fracs:
            idx = np.sort(perm[: max(50, int(round(frac * len(perm))))])
            dtr = sp["train"].iloc[idx]
            v = make_vectorizer(args.max_features)
            Xs = v.fit_transform(dtr["Description"].fillna("").astype(str))
            Xv = v.transform(sp["val"]["Description"].fillna("").astype(str))
            for model in ("logreg", "linearsvc"):
                mdl = make_model(model).fit(Xs, dtr["Severity"].values)
                lc_rows.append({"lengan": arm, "model": model, "frac": frac, "n_train": len(dtr),
                                "f1_train": f1_score(dtr["Severity"].values, mdl.predict(Xs), average="macro", zero_division=0),
                                "f1_val": f1_score(yva, mdl.predict(Xv), average="macro", zero_division=0)})
                if frac == 1.0:
                    lin_models[(arm, model)] = (v, mdl)
            print(f"   learning curve frac {frac:.2f}: " + ", ".join(
                f"{r['model']} val {r['f1_val']:.3f}" for r in lc_rows[-2:]))
    hist = pd.concat(hist_all, ignore_index=True)
    best = pd.DataFrame(best_rows)
    lc = pd.DataFrame(lc_rows)
    hist.to_csv(out / "riwayat_epoch_sgd.csv", index=False)
    best.to_csv(out / "epoch_terbaik_sgd.csv", index=False)
    lc.to_csv(out / "learning_curve.csv", index=False)
    plot_sgd(hist, best, out / "kurva_sgd_loss_f1.png")
    plot_learning_curve(lc, out / "learning_curve_logreg_linearsvc.png")

    # 4. EVALUASI AKHIR (test, sekali) ------------------------------------------------------------------------
    print("\n4. Evaluasi akhir di test ...")
    to_int = {k: i for i, k in enumerate(SEVERITY_ORDER)}
    K = len(SEVERITY_ORDER)
    preds: dict[tuple, pd.Series] = {}          # (arm, model[, seed]) -> prediksi berindeks _orig_idx
    for arm, sp in splits.items():
        te = sp["test"]
        Xte = vecs[arm].transform(te["Description"].fillna("").astype(str))
        for s in range(args.seeds):
            preds[(arm, "sgd", s)] = pd.Series(sgd_models[(arm, s)].predict(Xte), index=te["_orig_idx"].values)
        for model in ("logreg", "linearsvc"):
            v, mdl = lin_models[(arm, model)]
            preds[(arm, model)] = pd.Series(mdl.predict(v.transform(te["Description"].fillna("").astype(str))),
                                            index=te["_orig_idx"].values)

    def arm_keys(arm, model):
        return [(arm, "sgd", s) for s in range(args.seeds)] if model == "sgd" else [(arm, model)]

    final_rows, perclass_rows, cms_sgd, cms_lr = [], [], {}, {}
    for arm, sp in splits.items():
        te = sp["test"]
        y = te["Severity"].values
        yi = np.array([to_int[v] for v in y])
        idx = np.random.default_rng(args.boot_seed).integers(0, len(y), size=(args.n_boot, len(y)))
        for model in ("sgd", "logreg", "linearsvc"):
            keys = arm_keys(arm, model)
            P = [np.array([to_int[v] for v in preds[kk].values]) for kk in keys]
            pts = [float(macro_f1_resamples(yi, p, np.arange(len(y))[None, :], K)[0]) for p in P]
            boots = np.mean([macro_f1_resamples(yi, p, idx, K) for p in P], axis=0)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            f1c = np.mean([f1_score(y, preds[kk].values, labels=SEVERITY_ORDER, average=None, zero_division=0)
                           for kk in keys], axis=0)
            final_rows.append({"lengan": arm, "model": model, "n_test": len(y), "macro_f1": round(float(np.mean(pts)), 4),
                               "ci95_lo": round(float(lo), 4), "ci95_hi": round(float(hi), 4),
                               "sd_antar_seed": round(float(np.std(pts, ddof=1)), 4) if len(pts) > 1 else 0.0,
                               "rentang_antar_seed": round(float(max(pts) - min(pts)), 4)})
            perclass_rows.append({"lengan": arm, "model": model, **{k: round(float(v), 4) for k, v in zip(SEVERITY_ORDER, f1c)}})
        cms_sgd[arm] = confusion_matrix(y, preds[(arm, "sgd", 0)].values, labels=SEVERITY_ORDER)
        cms_lr[arm] = confusion_matrix(y, preds[(arm, "logreg")].values, labels=SEVERITY_ORDER)
    final = pd.DataFrame(final_rows)
    perclass = pd.DataFrame(perclass_rows)
    final.to_csv(out / "evaluasi_test.csv", index=False)
    perclass.to_csv(out / "f1_per_kelas_test.csv", index=False)
    for arm in ARMS:
        pd.DataFrame(cms_sgd[arm], index=SEVERITY_ORDER, columns=SEVERITY_ORDER).to_csv(out / f"confusion_sgd_seed0_{arm}.csv")
        pd.DataFrame(cms_lr[arm], index=SEVERITY_ORDER, columns=SEVERITY_ORDER).to_csv(out / f"confusion_logreg_{arm}.csv")
    plot_confusion(cms_sgd, "Confusion matrix test — SGD seed 0", out / "confusion_sgd_seed0.png")
    plot_confusion(cms_lr, "Confusion matrix test — Logistic Regression", out / "confusion_logreg.png")

    # perbandingan terhadap raw: irisan dokumen test yang ada di SEMUA lengan
    common = sorted(set.intersection(*[set(sp["test"]["_orig_idx"]) for sp in splits.values()]))
    ytrue = arms["raw"].set_index("_orig_idx").loc[common, "Severity"].values
    yi = np.array([to_int[v] for v in ytrue])
    idx = np.random.default_rng(args.boot_seed).integers(0, len(common), size=(args.n_boot, len(common)))
    cmp_rows = []
    for model in ("sgd", "logreg", "linearsvc"):
        base_keys = arm_keys("raw", model)
        Pb = [np.array([to_int[v] for v in preds[kk].loc[common].values]) for kk in base_keys]
        fb = [float(macro_f1_resamples(yi, p, np.arange(len(common))[None, :], K)[0]) for p in Pb]
        bb = np.mean([macro_f1_resamples(yi, p, idx, K) for p in Pb], axis=0)
        for arm in ("full_tailcut", "full_initial"):
            keys = arm_keys(arm, model)
            Pa = [np.array([to_int[v] for v in preds[kk].loc[common].values]) for kk in keys]
            fa = [float(macro_f1_resamples(yi, p, np.arange(len(common))[None, :], K)[0]) for p in Pa]
            ba = np.mean([macro_f1_resamples(yi, p, idx, K) for p in Pa], axis=0)
            lo, hi = np.percentile(ba - bb, [2.5, 97.5])
            # McNemar (akurasi) pada seed 0 / model deterministik; koreksi kontinuitas
            ca, cb = Pa[0] == yi, Pb[0] == yi
            b01, b10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
            z = (abs(b01 - b10) - 1) / np.sqrt(b01 + b10) if (b01 + b10) else 0.0
            p_z = float(2 * (1 - normal_dist.cdf(abs(z))))
            t_stat, p_t = (ttest_rel(fa, fb) if len(fa) > 1 else (np.nan, np.nan))
            cmp_rows.append({"model": model, "pembanding": f"{arm} - raw", "n_test_irisan": len(common),
                             "selisih_macro_f1": round(float(np.mean(fa) - np.mean(fb)), 4),
                             "ci95_lo": round(float(lo), 4), "ci95_hi": round(float(hi), 4),
                             "ci_melewati_0": bool(lo <= 0 <= hi),
                             "acc_lengan": round(float(ca.mean()), 4), "acc_raw": round(float(cb.mean()), 4),
                             "mcnemar_z": round(float(np.sign(b01 - b10) * z), 3), "mcnemar_p": round(p_z, 4),
                             "t_berpasangan_5seed": round(float(t_stat), 3) if model == "sgd" else None,
                             "p_t": round(float(p_t), 4) if model == "sgd" else None})
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(out / "perbandingan_vs_raw.csv", index=False)

    # 5. KRITERIA BAIK/BELUM ----------------------------------------------------------------------------------
    crit_rows = []
    for _, r in final.iterrows():
        arm, model = r["lengan"], r["model"]
        if model == "sgd":
            b = best[best["lengan"] == arm]
            gap = float((b["f1_train_best"] - b["f1_val_best"]).mean())
        else:
            g = lc[(lc["lengan"] == arm) & (lc["model"] == model) & (lc["frac"] == 1.0)].iloc[0]
            gap = float(g["f1_train"] - g["f1_val"])
        pc = perclass[(perclass["lengan"] == arm) & (perclass["model"] == model)].iloc[0]
        crit_rows.append({"lengan": arm, "model": model, "macro_f1_test": r["macro_f1"],
                          "lulus_>0.138": bool(r["macro_f1"] > 0.138), "lulus_>0.25": bool(r["macro_f1"] > 0.25),
                          "gap_train_val_poin": round(100 * gap, 1), "lulus_gap<=5": bool(gap <= 0.05),
                          "f1_blocker": pc["blocker"], "f1_trivial": pc["trivial"],
                          "lulus_minoritas>0": bool(pc["blocker"] > 0 and pc["trivial"] > 0),
                          "rentang_seed_poin": round(100 * r["rentang_antar_seed"], 2),
                          "lulus_stabil(<=2 poin)": bool(r["rentang_antar_seed"] <= 0.02)})
    crit = pd.DataFrame(crit_rows)
    crit.to_csv(out / "kriteria_baik_belum.csv", index=False)

    # 6. RINGKASAN ---------------------------------------------------------------------------------------------
    report["elapsed_seconds"] = round(time.time() - t0, 1)
    (out / "hasil_task.json").write_text(json.dumps({**report, "n_test_irisan": len(common)}, indent=2, default=str),
                                         encoding="utf-8")
    md = [f"# Hasil Task 11 Okt 2026 — Cleaning, Kurva Belajar, Uji Statistik\n",
          f"Dihasilkan `task_kurva_belajar.py` ({report['elapsed_seconds']:.0f} dtk). Angka lengkap ada di CSV folder ini.\n",
          "## 1. Cleaning: sebelum vs sesudah\n", md_table(summ), "",
          "Penanda `POST_REPORT` sesudah TAILCUT ≈ 0 **secara konstruksi** (regex yang sama memotong dan mengukur), "
          "jadi bukan bukti bersih. Ukuran yang tidak sirkular ada di dua tabel berikut.\n",
          "### Sisa kontaminasi — kebenaran struktural (dokumen berpasangan dengan laporan awal)\n",
          md_table(struct), "",
          "### Sisa kontaminasi — kutipan audit AI (batas bawah, dari 50 item per kelompok)\n",
          md_table(quotes.pivot_table(index=["lengan", "kelompok"], columns="penilai", values="batas_bawah_pct_dari_50").reset_index()), "",
          "### Contoh sebelum/sesudah (train)\n", ex_md,
          "## 2–3. Pelatihan\n", "![kurva SGD](kurva_sgd_loss_f1.png)\n",
          md_table(best.groupby("lengan")[["best_epoch", "epoch_dijalankan", "f1_train_best", "f1_val_best"]].mean().round(3).reset_index()), "",
          "![learning curve](learning_curve_logreg_linearsvc.png)\n",
          "## 4. Evaluasi akhir di test\n", md_table(final), "",
          "F1 per kelas (SGD = rata-rata 5 seed):\n", md_table(perclass), "",
          "![confusion SGD](confusion_sgd_seed0.png)\n", "![confusion LogReg](confusion_logreg.png)\n",
          f"### Perbandingan dengan raw (irisan {len(common)} dokumen test yang ada di semua lengan)\n",
          md_table(cmp), "",
          "Uji utama: bootstrap berpasangan (1000 resample) untuk selisih macro-F1. McNemar (z) hanya menguji "
          "akurasi; uji t berpasangan antar 5 seed hanya mencerminkan keacakan pelatihan SGD, bukan variasi data.\n",
          "## 5. Kriteria baik/belum\n", md_table(crit), ""]
    (out / "ringkasan.md").write_text("\n".join(md), encoding="utf-8")
    print(final.to_string(index=False))
    print(cmp.to_string(index=False))
    print(crit.to_string(index=False))
    print(f"\nSelesai {report['elapsed_seconds']:.0f} dtk. Disimpan di {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
