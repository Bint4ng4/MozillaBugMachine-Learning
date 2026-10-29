"""
task_banding_literatur.py — Angka pembanding literatur (permintaan user 14 Okt 2026: "kumpulkan hasil paper terkait …
di akhir dibandingkan"). Menghitung ulang hasil kita dengan PENGATURAN yang dipakai paper lain, agar perbandingan adil.

Tidak ada konfigurasi model baru: LogReg (C=1) dan LinearSVC (C=0,5), TF-IDF 20k bigram, min_df=2, bobot kelas seimbang
(sama dengan acuan task 11–12 Okt). Lengan: raw (dokumen penuh) vs full_initial (laporan awal + cleaning), hanya dokumen
yang ada di kedua lengan (berpasangan).

Bagian A — pengaturan evaluasi:
  label   six    : 6 kelas (pengaturan kita)
          five   : 5 kelas tanpa `normal` (seperti Sarawan dkk. 2025)
          binary : severe (blocker/critical/major) vs non-severe (minor/trivial), tanpa `normal` (pengaturan klasik
                   Lamkanfi; definisi severe = Kucuk dkk. 2022)
  split   kronologis : train_fit (indeks < 25.000) -> test (indeks >= 28.507), sama dengan task 11–12 Okt
          acak       : 80/20 berstrata (seperti Sarawan dkk.), 3 ulangan, indeks split sama untuk kedua lengan
  metrik  akurasi, F1 berbobot (weighted, dipakai Sarawan dkk.), macro-F1 (dipakai kita) + baseline kelas mayoritas
Bagian B — penanda duplikat di teks raw (pembanding Kucuk dkk. 2022): % dokumen duplikat/master dan % severe per kelompok.

Pakai:  python task_banding_literatur.py [--out-dir results_task_20261014_literatur]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cleaning as c  # noqa: E402
import task_kurva_belajar as tk  # noqa: E402
import task_perbaiki_gap as gp  # noqa: E402

BASE_FEAT = (20000, (1, 2), False)
MODELS = {"logreg": 1.0, "linearsvc": 0.5}
SEVERE = {"blocker", "critical", "major"}
NONSEVERE = {"minor", "trivial"}
LABEL_SETS = ("six", "five", "binary")
DUP_SELF = r"this bug has been marked as a duplicate of"      # komentar Bugzilla di laporan yang ditutup DUPLICATE
DUP_MASTER = r"has been marked as a duplicate of this bug"     # komentar Bugzilla di laporan master


def relabel(df: pd.DataFrame, label_set: str) -> pd.DataFrame:
    if label_set == "six":
        return df
    d = df[df["Severity"] != "normal"].copy()
    if label_set == "binary":
        d["Severity"] = np.where(d["Severity"].isin(SEVERE), "severe", "non-severe")
    return d


def scores(y_true, y_pred) -> dict:
    return {"akurasi": accuracy_score(y_true, y_pred),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0)}


def run(train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    """Latih semua model pada `train`, uji di `test`. Satu TF-IDF dipakai bersama (fit hanya pada train)."""
    vec = gp.vectorizer(BASE_FEAT)
    Xtr = vec.fit_transform(train["Description"].fillna("").astype(str))
    Xte = vec.transform(test["Description"].fillna("").astype(str))
    ytr = np.asarray(train["Severity"].values, dtype=object)
    yte = np.asarray(test["Severity"].values, dtype=object)
    maj = pd.Series(ytr).value_counts().idxmax()
    rows = [{"model": "mayoritas", **scores(yte, np.full(len(yte), maj, dtype=object))}]
    for fam, reg in MODELS.items():
        m, _ = gp.fit_model(fam, reg, Xtr, ytr, None, None)
        rows.append({"model": fam, **scores(yte, m.predict(Xte))})
    return [{**r, "n_train": len(ytr), "n_test": len(yte), "kelas_mayoritas": maj,
             "pct_mayoritas_test": float(np.mean(yte == maj))} for r in rows]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (mid - half, mid + half)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="archive")
    ap.add_argument("--out-dir", default="results_task_20261014_literatur")
    ap.add_argument("--random-reps", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    md: list[str] = []

    df0 = pd.read_csv(Path(args.archive) / "sev.csv")

    # B. penanda duplikat (raw, semua dokumen) ------------------------------------------------------------------
    tl = df0["Description"].fillna("").astype(str).str.lower()
    is_dup, is_master = tl.str.contains(DUP_SELF), tl.str.contains(DUP_MASTER)
    groups = {"duplikat (ditutup sebagai DUPLICATE)": is_dup, "master (ada laporan lain di-dupe ke sini)": is_master,
              "duplikat sekaligus master": is_dup & is_master, "unik (bukan keduanya)": ~is_dup & ~is_master,
              "semua dokumen": pd.Series(True, index=df0.index)}
    sev = df0["Severity"].astype(str)
    dup_rows = []
    for g, mask in groups.items():
        n, k = int(mask.sum()), int(sev[mask].isin(SEVERE).sum())
        lo, hi = wilson(k, n)
        dup_rows.append({"kelompok": g, "n": n, "pct_dokumen": round(100 * n / len(df0), 2),
                         "pct_severe": round(100 * k / n, 2) if n else np.nan,
                         "ci95_lo": round(100 * lo, 2), "ci95_hi": round(100 * hi, 2)})
    dup = pd.DataFrame(dup_rows)
    dup.to_csv(out / "penanda_duplikat_severity.csv", index=False)
    v_rows = []
    for name, mask in (("penanda duplikat", is_dup), ("penanda master", is_master)):
        tab = pd.crosstab(mask, sev)
        chi2, p, dof, _ = chi2_contingency(tab)
        v_rows.append({"penanda": name, "chi2": round(float(chi2), 1), "dof": int(dof), "p": float(p),
                       "cramers_v": round(float(np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))), 4)})
    vdf = pd.DataFrame(v_rows)
    vdf.to_csv(out / "penanda_duplikat_v.csv", index=False)
    md += ["## B. Penanda duplikat (raw, 32.428 dokumen)\n", tk.md_table(dup), "", tk.md_table(vdf), ""]
    print(dup.to_string(index=False))
    print(vdf.to_string(index=False))

    # A. pengaturan evaluasi -----------------------------------------------------------------------------------
    arms = {name: tk.quiet(c.run_config, df0, name, tk.ARMS[name])[0] for name in ("raw", "full_initial")}
    common = sorted(set(arms["raw"]["_orig_idx"]) & set(arms["full_initial"]["_orig_idx"]))
    arms = {k: v[v["_orig_idx"].isin(common)].set_index("_orig_idx").loc[common].reset_index() for k, v in arms.items()}
    print(f"cleaning selesai ({time.time() - t0:.0f} dtk); dokumen berpasangan: {len(common)}")
    rows = []
    for label_set in LABEL_SETS:
        lab = {k: relabel(v, label_set) for k, v in arms.items()}
        idx = lab["raw"]["_orig_idx"].values                      # sama untuk kedua lengan (label sama)
        # kronologis
        for arm, d in lab.items():
            o = d["_orig_idx"]
            for r in run(d[o < tk.VAL_MIN], d[o >= c.TEST_MIN]):
                rows.append({"label": label_set, "split": "kronologis", "ulangan": 0, "lengan": arm, **r})
        # acak 80/20 berstrata, indeks sama untuk kedua lengan
        y = lab["raw"]["Severity"].values
        sss = StratifiedShuffleSplit(n_splits=args.random_reps, test_size=0.2, random_state=20261014)
        for rep, (i_tr, i_te) in enumerate(sss.split(np.zeros(len(y)), y)):
            for arm, d in lab.items():
                assert (d["_orig_idx"].values == idx).all()
                for r in run(d.iloc[i_tr], d.iloc[i_te]):
                    rows.append({"label": label_set, "split": "acak", "ulangan": rep, "lengan": arm, **r})
        print(f"label {label_set} selesai ({time.time() - t0:.0f} dtk)")
    res = pd.DataFrame(rows)
    res.to_csv(out / "pengaturan_evaluasi_per_ulangan.csv", index=False)
    summ = res.groupby(["label", "split", "lengan", "model"], sort=False).agg(
        n_train=("n_train", "mean"), n_test=("n_test", "mean"), pct_mayoritas_test=("pct_mayoritas_test", "mean"),
        akurasi=("akurasi", "mean"), f1_weighted=("f1_weighted", "mean"), f1_macro=("f1_macro", "mean"),
        f1_macro_sd=("f1_macro", "std")).reset_index().round(4)
    summ.to_csv(out / "pengaturan_evaluasi.csv", index=False)
    # selisih raw − laporan awal (over-estimasi) dan acak − kronologis, per pengaturan
    piv = summ.pivot_table(index=["label", "split", "model"], columns="lengan",
                           values=["akurasi", "f1_weighted", "f1_macro"], sort=False)
    infl = pd.DataFrame({f"{m}_raw_minus_initial_poin": 100 * (piv[(m, "raw")] - piv[(m, "full_initial")])
                         for m in ("akurasi", "f1_weighted", "f1_macro")}).round(2).reset_index()
    infl.to_csv(out / "selisih_raw_vs_laporan_awal.csv", index=False)
    piv2 = summ.pivot_table(index=["label", "lengan", "model"], columns="split",
                            values=["akurasi", "f1_weighted", "f1_macro"], sort=False)
    split_eff = pd.DataFrame({f"{m}_acak_minus_kronologis_poin": 100 * (piv2[(m, "acak")] - piv2[(m, "kronologis")])
                              for m in ("akurasi", "f1_weighted", "f1_macro")}).round(2).reset_index()
    split_eff.to_csv(out / "selisih_acak_vs_kronologis.csv", index=False)
    md += ["## A. Pengaturan evaluasi\n", tk.md_table(summ), "", "### Selisih raw − laporan awal (poin)\n",
           tk.md_table(infl), "", "### Selisih acak − kronologis (poin)\n", tk.md_table(split_eff), ""]

    secs = round(time.time() - t0)
    (out / "angka_otomatis.md").write_text(
        f"# Angka otomatis task_banding_literatur.py ({secs} dtk)\n\nNarasi: `../PERBANDINGAN_LITERATUR_20261014.md`.\n\n"
        + "\n".join(md), encoding="utf-8")
    (out / "hasil.json").write_text(json.dumps({"detik": secs, "config": vars(args), "n_berpasangan": len(common)},
                                               indent=2), encoding="utf-8")
    print(f"Selesai {secs} dtk -> {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
