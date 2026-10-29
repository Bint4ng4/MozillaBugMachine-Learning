"""
task_laporan_loss.py — Laporan loss & skor train vs val (permintaan user 14 Okt 2026: "laporan lengkap tentang train
loss dan val loss; apakah modelnya menghafal?"). Lanjutan task_kurva_belajar.py (11 Okt) dan task_perbaiki_gap.py (12 Okt).

Tidak memilih konfigurasi baru dan tidak mengubah angka RQ1/RQ2: semua model memakai konfigurasi task 11–12 Okt.
Split sama: train_fit < 25.000 | val 25.000–28.505 | test >= 28.507 (indeks asli).

Bagian:
  1. SGD per epoch — dibaca dari results_task_20261011/riwayat_epoch_sgd.csv (tidak dilatih ulang).
  2. Loss & macro-F1 akhir di train/val/test: 3 model acuan x 3 lengan, plus model B dan A (full_initial).
  3. Learning curve dengan loss (full_initial): LogReg acuan, LinearSVC acuan, model B.
  4. Menghafal vs pergeseran waktu: 15% acak dari periode train ditahan (5 ulangan) lalu dibandingkan dengan val.
  5. Bobot model pada n-gram langka (df train <= 5) + F1 per kelas train vs val.

Loss:
  - LogReg & SGD: log loss berbobot kelas (bobot seimbang dari train) — sama dengan riwayat SGD 11 Okt.
    Acuan: tebakan rata 6 kelas = ln 6 = 1,792, berapa pun bobot kelasnya.
  - LinearSVC tidak punya probabilitas: squared hinge one-vs-rest (bentuk fungsi yang diminimalkan LinearSVC, tanpa
    suku regularisasi), dibobot sama seperti log loss. Hanya bisa dibandingkan train vs val dalam model yang sama.

Pakai:  python task_laporan_loss.py [--out-dir results_task_20261014_loss]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score, log_loss  # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cleaning as c  # noqa: E402
import task_kurva_belajar as tk  # noqa: E402
import task_perbaiki_gap as gp  # noqa: E402
from experiment import SEVERITY_ORDER  # noqa: E402

BASE_FEAT = (20000, (1, 2), False)
# nama -> (family, fitur, reg); konfigurasi persis task 11–12 Okt
SPECS = {
    "LogReg acuan": ("logreg", BASE_FEAT, 1.0),
    "LinearSVC acuan": ("linearsvc", BASE_FEAT, 0.5),
    "SGD acuan": ("sgd", BASE_FEAT, 1e-4),
    "Model B (uji ketahanan)": ("linearsvc", (5000, (1, 2), False), 0.03),
    "Model A (gap <= 5)": ("sgd", (2000, (1, 1), True), 1e-3),
}
ACUAN = ["LogReg acuan", "LinearSVC acuan", "SGD acuan"]
LC_MODELS = ["LogReg acuan", "LinearSVC acuan", "Model B (uji ketahanan)"]
PART_COL = {"train": tk.COL["train"], "val": tk.COL["val"], "test": tk.COL["full_initial"],
            "holdout": tk.COL["full_initial"]}
PART_MARK = {"train": "o", "val": "s", "test": "D", "holdout": "D"}
LN6 = float(np.log(6))
RARE_DF = 5


def loss_name(family: str) -> str:
    return "squared hinge" if family == "linearsvc" else "log loss"


def wloss(model, family: str, X, y, cw: dict) -> float:
    w = np.array([cw.get(v, 1.0) for v in y])
    if family == "linearsvc":
        D = model.decision_function(X)
        Y = np.where(y[:, None] == model.classes_[None, :], 1.0, -1.0)
        return float(np.average((np.maximum(0.0, 1.0 - Y * D) ** 2).sum(1), weights=w))
    return float(log_loss(y, model.predict_proba(X), labels=model.classes_, sample_weight=w))


def fit_eval(name: str, parts: dict, seeds: int, keep: bool = False):
    """Latih model `name` pada parts['train'] (val dipakai hanya untuk early stopping SGD), ukur di semua bagian.
    Mengembalikan baris per (bagian) dirata-rata antar seed, prediksi seed 0, dan (opsional) model + matriks train."""
    fam, feat, reg = SPECS[name]
    vec = gp.vectorizer(feat)
    txt = {k: v["Description"].fillna("").astype(str) for k, v in parts.items()}
    y = {k: np.asarray(v["Severity"].values, dtype=object) for k, v in parts.items()}
    X = {"train": vec.fit_transform(txt["train"])}
    X.update({k: vec.transform(txt[k]) for k in parts if k != "train"})
    cw = tk.balanced_weights(y["train"])
    runs = []
    for s in range(seeds if fam == "sgd" else 1):
        m, info = gp.fit_model(fam, reg, X["train"], y["train"], X["val"], y["val"], seed=s)
        runs.append((m, info))
    rows = []
    for part in parts:
        f1s = [gp.mf1(y[part], m.predict(X[part])) for m, _ in runs]
        ls = [wloss(m, fam, X[part], y[part], cw) for m, _ in runs]
        rows.append({"bagian": part, "n": len(y[part]), "macro_f1": float(np.mean(f1s)), "loss": float(np.mean(ls)),
                     "macro_f1_rentang_seed": float(np.ptp(f1s)), "loss_rentang_seed": float(np.ptp(ls))})
    extra = {"best_epoch": [i.get("best_epoch") for _, i in runs if i]}
    if keep:
        extra.update({"model": runs[0][0], "vec": vec, "Xtr": X["train"], "y": y, "X": X})
    return pd.DataFrame(rows), extra


# ---------------------------------------------------------------------------------------------------- grafik
def plot_sgd(hist: pd.DataFrame, best: pd.DataFrame, path: Path) -> None:
    arms = list(tk.ARMS)
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True, sharey="row")
    for j, arm in enumerate(arms):
        h = hist[hist["lengan"] == arm]
        for i, (metric, ylab) in enumerate((("loss", "log loss (berbobot kelas)"), ("f1", "macro-F1"))):
            a = ax[i, j]
            for part in ("train", "val"):
                g = h.groupby("epoch")[f"{metric}_{part}"]
                mean, lo, hi = g.mean(), g.min(), g.max()
                a.fill_between(mean.index, lo, hi, color=PART_COL[part], alpha=0.15, linewidth=0)
                a.plot(mean.index, mean.values, color=PART_COL[part], linewidth=2, marker=PART_MARK[part],
                       markersize=5, label=part)
                a.annotate(f"{part} {mean.values[-1]:.2f}", (mean.index[-1], mean.values[-1]), xytext=(6, 0),
                           textcoords="offset points", va="center", fontsize=8.5, color=tk.INK2)
            if i == 0:
                a.axhline(LN6, color=tk.INK2, linewidth=1, linestyle="--")
                a.annotate("tebakan rata 6 kelas (ln 6 = 1,79)", (h["epoch"].max(), LN6), xytext=(-4, 5),
                           textcoords="offset points", ha="right", fontsize=8, color=tk.INK2)
                a.set_title(tk.ARM_LABEL[arm])
            for e in best[best["lengan"] == arm]["best_epoch"]:
                a.axvline(e, color=tk.INK2, linewidth=0.8, linestyle=":", alpha=0.5)
            if j == 0:
                a.set_ylabel(ylab)
            if i == 1:
                a.set_xlabel("epoch")
                a.set_ylim(0, 0.8)
            a.xaxis.set_major_locator(tk.MaxNLocator(integer=True))
            a.set_xlim(0.5, hist["epoch"].max() + 3.5)
    ax[0, 0].set_ylim(1.0, 1.9)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("SGD (TF-IDF 20k bigram, alpha 1e-4): loss dan macro-F1 per epoch — garis = rata-rata 5 seed, "
                 "pita = min–maks, garis titik = epoch terbaik", fontsize=11, color=tk.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_dumbbell(df: pd.DataFrame, parts: list[str], labels: dict, title: str, path: Path, xerr: str | None = None):
    """Satu baris per model, satu penanda per bagian (macro-F1), garis abu menghubungkan min–maks."""
    names = list(dict.fromkeys(df["model"]))
    fig, a = plt.subplots(figsize=(10, 0.8 * len(names) + 2.0))
    for i, nm in enumerate(names[::-1]):
        d = df[df["model"] == nm].set_index("bagian")
        vals = [d.at[p, "macro_f1"] for p in parts]
        a.plot([min(vals), max(vals)], [i, i], color=tk.GRID, linewidth=3, zorder=1)
        for p in parts:
            v = d.at[p, "macro_f1"]
            err = d.at[p, xerr] if xerr and xerr in d.columns else None
            a.errorbar(v, i, xerr=err, fmt=PART_MARK[p], color=PART_COL[p], markersize=9, markeredgecolor=tk.SURFACE,
                       markeredgewidth=1.5, ecolor=PART_COL[p], elinewidth=1.5, capsize=3, zorder=3,
                       label=labels[p] if i == 0 else None)
            a.annotate(f"{v:.2f}", (v, i), xytext=(0, -15 if p == "val" else 9), textcoords="offset points",
                       ha="center", fontsize=8.5, color=tk.INK2)
    a.set_yticks(range(len(names)), names[::-1])
    a.set_xlim(0, 1)
    a.set_ylim(-0.6, len(names) - 0.4)
    a.set_xlabel("macro-F1")
    a.grid(axis="y", visible=False)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=len(parts), fontsize=9)
    a.set_title(title, fontsize=11, color=tk.INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lc(lc: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(2, len(LC_MODELS), figsize=(13, 7.2), sharex=True)
    for j, nm in enumerate(LC_MODELS):
        d = lc[lc["model"] == nm].sort_values("n_train")
        fam = SPECS[nm][0]
        for i, (metric, ylab) in enumerate((("f1", "macro-F1"), ("loss", None))):
            a = ax[i, j]
            for part in ("train", "val"):
                a.plot(d["n_train"], d[f"{metric}_{part}"], color=PART_COL[part], linewidth=2,
                       marker=PART_MARK[part], markersize=7, label=part)
                a.annotate(f"{part} {d[f'{metric}_{part}'].iloc[-1]:.2f}", (d["n_train"].iloc[-1],
                           d[f"{metric}_{part}"].iloc[-1]), xytext=(-4, 8), textcoords="offset points", ha="right",
                           fontsize=8.5, color=tk.INK2)
            if i == 0:
                a.set_ylim(0, 1.02)
                a.set_title(nm)
            else:
                a.set_xlabel("jumlah dokumen train_fit")
                if fam != "linearsvc":
                    a.axhline(LN6, color=tk.INK2, linewidth=1, linestyle="--")
                    a.annotate("ln 6", (d["n_train"].iloc[0], LN6), xytext=(2, 4), textcoords="offset points",
                               fontsize=8, color=tk.INK2)
            a.set_ylabel(ylab or f"{loss_name(fam)} (berbobot kelas)")
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Learning curve full_initial: train vs val menurut jumlah dokumen latih (subsampel acak, seed 0)",
                 fontsize=11, color=tk.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------- utama
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="archive")
    ap.add_argument("--out-dir", default="results_task_20261014_loss")
    ap.add_argument("--hist-dir", default="results_task_20261011")
    ap.add_argument("--seeds", type=int, default=5, help="seed SGD")
    ap.add_argument("--holdout-reps", type=int, default=5)
    ap.add_argument("--lc-fracs", nargs="+", type=float, default=[0.1, 0.25, 0.5, 0.75, 1.0])
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    md: list[str] = []

    # 1. SGD per epoch (berkas 11 Okt) ------------------------------------------------------------------------
    hist = pd.read_csv(Path(args.hist_dir) / "riwayat_epoch_sgd.csv")
    best = pd.read_csv(Path(args.hist_dir) / "epoch_terbaik_sgd.csv")
    rows = []
    for (arm, seed), h in hist.groupby(["lengan", "seed"]):
        be = int(best[(best["lengan"] == arm) & (best["seed"] == seed)]["best_epoch"].iloc[0])
        for tag, ep in (("epoch_1", 1), ("epoch_terbaik", be), ("epoch_terakhir", int(h["epoch"].max()))):
            r = h[h["epoch"] == ep].iloc[0]
            rows.append({"lengan": arm, "seed": seed, "titik": tag, "epoch": ep, "loss_train": r["loss_train"],
                         "loss_val": r["loss_val"], "f1_train": r["f1_train"], "f1_val": r["f1_val"]})
    sgd_pts = pd.DataFrame(rows)
    sgd_pts["gap_loss"] = sgd_pts["loss_val"] - sgd_pts["loss_train"]
    sgd_pts["gap_f1_poin"] = 100 * (sgd_pts["f1_train"] - sgd_pts["f1_val"])
    sgd_sum = sgd_pts.groupby(["lengan", "titik"], sort=False)[
        ["epoch", "loss_train", "loss_val", "gap_loss", "f1_train", "f1_val", "gap_f1_poin"]].mean().round(4).reset_index()
    sgd_sum.to_csv(out / "sgd_per_epoch_ringkas.csv", index=False)
    seed0 = hist[(hist["lengan"] == "full_initial") & (hist["seed"] == 0)].drop(columns=["lengan", "seed"]).round(4)
    plot_sgd(hist, best, out / "gambar1_sgd_loss_per_epoch.png")
    md += ["## 1. SGD per epoch (dari riwayat 11 Okt)\n", "Rata-rata 5 seed per titik:\n", tk.md_table(sgd_sum), "",
           "full_initial, seed 0:\n", tk.md_table(seed0), ""]
    print(f"1. SGD per epoch selesai ({time.time() - t0:.0f} dtk)")

    # data -----------------------------------------------------------------------------------------------------
    df0 = pd.read_csv(Path(args.archive) / "sev.csv")
    arms = {name: tk.quiet(c.run_config, df0, name, stages)[0] for name, stages in tk.ARMS.items()}
    splits = {a: tk.split3(d) for a, d in arms.items()}
    print(f"   cleaning 3 lengan selesai ({time.time() - t0:.0f} dtk)")

    # 2. loss & macro-F1 akhir ---------------------------------------------------------------------------------
    final, kept = [], {}
    for arm, sp in splits.items():
        names = ACUAN + (["Model B (uji ketahanan)", "Model A (gap <= 5)"] if arm == "full_initial" else [])
        for nm in names:
            res, extra = fit_eval(nm, sp, args.seeds, keep=(arm == "full_initial"))
            res.insert(0, "model", nm)
            res.insert(0, "lengan", arm)
            res.insert(3, "jenis_loss", loss_name(SPECS[nm][0]))
            res["best_epoch_sgd"] = str(extra["best_epoch"]) if extra["best_epoch"] else ""
            final.append(res)
            if arm == "full_initial":
                kept[nm] = extra
            print(f"2. {arm:<13} {nm:<24} " + " | ".join(
                f"{r.bagian} F1 {r.macro_f1:.3f} loss {r.loss:.3f}" for r in res.itertuples()))
    final = pd.concat(final, ignore_index=True)
    final.round(4).to_csv(out / "loss_f1_train_val_test.csv", index=False)
    wide = final.pivot_table(index=["lengan", "model", "jenis_loss"], columns="bagian", values=["macro_f1", "loss"],
                             sort=False)
    wide.columns = [f"{m}_{p}" for m, p in wide.columns]
    wide = wide.reset_index()
    wide["gap_f1_train_val_poin"] = 100 * (wide["macro_f1_train"] - wide["macro_f1_val"])
    wide["selisih_f1_val_test_poin"] = 100 * (wide["macro_f1_val"] - wide["macro_f1_test"])
    wide["gap_loss_val_train"] = wide["loss_val"] - wide["loss_train"]
    cols = ["lengan", "model", "jenis_loss", "loss_train", "loss_val", "loss_test", "gap_loss_val_train",
            "macro_f1_train", "macro_f1_val", "macro_f1_test", "gap_f1_train_val_poin", "selisih_f1_val_test_poin"]
    wide = wide[cols].round(4)
    wide.to_csv(out / "loss_f1_ringkas.csv", index=False)
    plot_dumbbell(final[final["lengan"] == "full_initial"], ["train", "val", "test"],
                  {"train": "train", "val": "val", "test": "test"},
                  "macro-F1 train vs val vs test (full_initial) — jarak train ke val = gap; val ≈ test",
                  out / "gambar2_train_val_test.png")
    md += ["## 2. Loss dan macro-F1 akhir (train / val / test)\n", tk.md_table(wide), ""]

    # 3. learning curve dengan loss (full_initial) ---------------------------------------------------------------
    sp = splits["full_initial"]
    perm = np.random.default_rng(0).permutation(len(sp["train"]))       # sama dengan task 11 Okt
    lc_rows = []
    for frac in args.lc_fracs:
        idx = np.sort(perm[: max(50, int(round(frac * len(perm))))])
        parts = {"train": sp["train"].iloc[idx].reset_index(drop=True), "val": sp["val"]}
        for nm in LC_MODELS:
            res, _ = fit_eval(nm, parts, 1)
            r = res.set_index("bagian")
            lc_rows.append({"model": nm, "jenis_loss": loss_name(SPECS[nm][0]), "frac": frac, "n_train": len(idx),
                            "f1_train": r.at["train", "macro_f1"], "f1_val": r.at["val", "macro_f1"],
                            "loss_train": r.at["train", "loss"], "loss_val": r.at["val", "loss"]})
        print(f"3. learning curve frac {frac:.2f} selesai ({time.time() - t0:.0f} dtk)")
    lc = pd.DataFrame(lc_rows).round(4)
    lc.to_csv(out / "learning_curve_loss.csv", index=False)
    plot_lc(lc, out / "gambar3_learning_curve_loss.png")
    md += ["## 3. Learning curve dengan loss (full_initial)\n", tk.md_table(lc), ""]

    # 4. menghafal vs pergeseran waktu ---------------------------------------------------------------------------
    sss = StratifiedShuffleSplit(n_splits=args.holdout_reps, test_size=0.15, random_state=20261014)
    ho_rows = []
    for rep, (i_fit, i_ho) in enumerate(sss.split(np.zeros(len(sp["train"])), sp["train"]["Severity"].values)):
        parts = {"train": sp["train"].iloc[i_fit].reset_index(drop=True),
                 "holdout": sp["train"].iloc[i_ho].reset_index(drop=True), "val": sp["val"]}
        for nm in LC_MODELS:
            res, _ = fit_eval(nm, parts, 1)
            res.insert(0, "model", nm)
            res.insert(0, "ulangan", rep)
            ho_rows.append(res)
        print(f"4. hold-out ulangan {rep} selesai ({time.time() - t0:.0f} dtk)")
    ho = pd.concat(ho_rows, ignore_index=True)
    ho.round(4).to_csv(out / "holdout_vs_val_per_ulangan.csv", index=False)
    ho_sum = ho.groupby(["model", "bagian"], sort=False).agg(
        n=("n", "first"), macro_f1=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        loss=("loss", "mean"), loss_sd=("loss", "std")).reset_index()
    ho_sum.round(4).to_csv(out / "holdout_vs_val.csv", index=False)
    dec = []
    for nm in LC_MODELS:
        d = ho_sum[ho_sum["model"] == nm].set_index("bagian")["macro_f1"]
        dec.append({"model": nm, "f1_train": d["train"], "f1_holdout_periode_sama": d["holdout"],
                    "f1_val_periode_berikutnya": d["val"], "gap_total_poin": 100 * (d["train"] - d["val"]),
                    "bagian_menghafal_poin": 100 * (d["train"] - d["holdout"]),
                    "bagian_pergeseran_waktu_poin": 100 * (d["holdout"] - d["val"])})
    dec = pd.DataFrame(dec).round(4)
    dec.to_csv(out / "dekomposisi_gap.csv", index=False)
    plot_dumbbell(ho_sum, ["train", "holdout", "val"],
                  {"train": "train (85% periode train)", "holdout": "hold-out 15% periode sama",
                   "val": "val (periode berikutnya)"},
                  "Menghafal vs pergeseran waktu (full_initial, rata-rata 5 ulangan; garis galat = SD)",
                  out / "gambar4_holdout_vs_val.png", xerr="macro_f1_sd")
    md += ["## 4. Menghafal vs pergeseran waktu\n", tk.md_table(ho_sum.round(4)), "", tk.md_table(dec), ""]

    # 5. n-gram langka + F1 per kelas ---------------------------------------------------------------------------
    rare_rows, ex_rows, pc_rows = [], [], []
    for nm in LC_MODELS:
        k = kept[nm]
        m, Xtr = k["model"], k["Xtr"]
        dfreq = np.asarray((Xtr > 0).sum(0)).ravel()
        mass = np.abs(m.coef_).sum(0)
        rare = dfreq <= RARE_DF
        rare_rows.append({"model": nm, "n_fitur": len(dfreq), "pct_fitur_langka": 100 * rare.mean(),
                          "pct_massa_bobot_di_fitur_langka": 100 * mass[rare].sum() / mass.sum(),
                          "bobot_rata2_fitur_langka": mass[rare].mean(), "bobot_rata2_fitur_umum": mass[~rare].mean()})
        feats = k["vec"].get_feature_names_out()
        cand = np.flatnonzero(dfreq <= 3)
        top = cand[np.argsort(-np.abs(m.coef_[:, cand]).max(0))][:10]
        for j in top:
            ci = int(np.argmax(np.abs(m.coef_[:, j])))
            ex_rows.append({"model": nm, "ngram": feats[j], "df_train": int(dfreq[j]), "kelas": m.classes_[ci],
                            "bobot": round(float(m.coef_[ci, j]), 3)})
        for part in ("train", "val", "test"):
            f1c = f1_score(k["y"][part], m.predict(k["X"][part]), labels=SEVERITY_ORDER, average=None, zero_division=0)
            pc_rows.append({"model": nm, "bagian": part, **{s: round(float(v), 3) for s, v in zip(SEVERITY_ORDER, f1c)}})
    rare_df = pd.DataFrame(rare_rows).round(3)
    ex_df = pd.DataFrame(ex_rows)
    pc_df = pd.DataFrame(pc_rows)
    rare_df.to_csv(out / "bobot_ngram_langka.csv", index=False)
    ex_df.to_csv(out / "contoh_ngram_langka_berbobot.csv", index=False)
    pc_df.to_csv(out / "f1_per_kelas_train_val_test.csv", index=False)
    md += [f"## 5. Bobot pada n-gram langka (df train <= {RARE_DF}) dan F1 per kelas\n", tk.md_table(rare_df), "",
           tk.md_table(ex_df), "", tk.md_table(pc_df), ""]

    secs = round(time.time() - t0)
    (out / "angka_otomatis.md").write_text(
        f"# Angka otomatis task_laporan_loss.py ({secs} dtk)\n\nNarasi ada di `LAPORAN_LOSS_TRAIN_VAL.md`.\n\n"
        + "\n".join(md), encoding="utf-8")
    (out / "hasil.json").write_text(json.dumps({"detik": secs, "config": vars(args), "ln6": LN6}, indent=2),
                                    encoding="utf-8")
    print(f"Selesai {secs} dtk -> {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
