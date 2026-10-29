"""
task_uji_ketahanan_rq2.py — Uji ketahanan RQ2 atas kritik reviewer AI 14 Okt 2026 (disetujui user: "boleh").

Desain RQ2 sama persis dengan `experiment.py --suite rq2_initial` (MASTER §8f): subset berpasangan utama, train
`_orig_idx` <= 28.505 / test >= 28.507, TF-IDF 20k bigram min_df=2 fit pada train, LogReg C=1 dan LinearSVC C=0,5
(bobot kelas seimbang), bootstrap berpasangan 1000x (seed 42, indeks sama untuk semua lengan). Tidak ada konfigurasi
model baru; tidak ada penyetelan pada test.

  #15  Baseline sepele: kelas mayoritas, acak berstrata (1000 undian dari distribusi train), dan heuristik kata kunci.
       Aturan kata kunci DITULIS SEBELUM skrip dijalankan dan tidak diubah setelah melihat hasil (lihat KEYWORD_RULES).
  #18  Metrik ordinal (urutan trivial < minor < normal < major < critical < blocker): quadratic weighted kappa (QWK),
       MAE dalam langkah, dan % kesalahan jauh (>= 3 langkah). Over-estimasi diuji juga dengan metrik ini.
  #22  Sensitivitas duplikat. Penanda dari teks raw `sev.csv`:
         duplikat = "this bug has been marked as a duplicate of" (laporan yang ditutup DUPLICATE)
         master   = "has been marked as a duplicate of this bug" (ada laporan lain yang di-dupe ke sini)
       D1  test tanpa duplikat (model sama)
       D1b test tanpa duplikat DAN master (model sama) — menutup kasus bocor duplikat-di-train/master-di-test
       D2  train dan test tanpa duplikat (model dilatih ulang)
  #8   Dekomposisi "pengaturan lazim +4–7 poin" menjadi efek teks (raw − laporan awal), efek split acak, dan interaksi,
       dihitung dari `results_task_20261014_literatur/pengaturan_evaluasi.csv` (tanpa training baru, tanpa CI).

Pakai:  python task_uji_ketahanan_rq2.py [--out-dir results_task_20261014_ketahanan]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment as ex  # noqa: E402

CLS = ex.SEVERITY_ORDER                       # trivial < minor < normal < major < critical < blocker
K = len(CLS)
TO_INT = {c: i for i, c in enumerate(CLS)}
MODELS = ("logreg", "linearsvc")
MAX_FEATURES = 20000
DUP_SELF = "this bug has been marked as a duplicate of"
DUP_MASTER = "has been marked as a duplicate of this bug"
REF_8F = {"logreg": {"ref": 0.3075, "initial": 0.2752, "ref_to_initial": 0.2724},
          "linearsvc": {"ref": 0.3149, "initial": 0.2632, "ref_to_initial": 0.2533}}

# Aturan a priori (14 Okt, sebelum run). Teks sudah tanpa tanda baca ("can t", "doesn t"); dicocokkan huruf kecil.
# Urutan = prioritas; dokumen tanpa kecocokan -> `normal`.
_CRASH = r"\bcrash(?:es|ed|ing)?\b|\bsegfault\b|\bsegmentation fault\b|\bhang(?:s|ing)?\b|\bhung\b|\bfreez(?:e|es|ing)\b|\bfrozen?\b"
_TYPO = r"\btypos?\b|\bmisspel(?:l|led|ling|t)\b|\bspelling\b|\bcosmetic\b"
KEYWORD_RULES = {
    "kata_kunci_K1": [("critical", _CRASH), ("trivial", _TYPO)],                      # contoh reviewer: crash -> critical
    "kata_kunci_K2": [("blocker", r"\bbustage\b|\b(?:fails?|failed|unable|cannot|can t|won t|doesn t|does not) (?:to )?"
                                  r"(?:build|compile|start|launch|install)\b"),
                      ("critical", _CRASH + r"|\bdata ?loss\b|\blos(?:e|es|t|ing) (?:all )?(?:my )?data\b"),
                      ("trivial", _TYPO),
                      ("major", r"\b(?:doesn t|does not|don t|not) work(?:s|ing)?\b|\bbroken\b|\bunusable\b")],
}


def keyword_predict(docs: pd.Series, rules) -> np.ndarray:
    low = docs.str.lower()
    out = np.full(len(docs), TO_INT["normal"])
    done = np.zeros(len(docs), dtype=bool)
    for label, pat in rules:
        hit = low.str.contains(pat, regex=True).values & ~done
        out[hit] = TO_INT[label]
        done |= hit
    return out


def qwk_resamples(yt: np.ndarray, yp: np.ndarray, idx: np.ndarray) -> np.ndarray:
    w = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / (K - 1) ** 2
    out = np.empty(len(idx))
    for b in range(len(idx)):
        cm = np.bincount(yt[idx[b]] * K + yp[idx[b]], minlength=K * K).reshape(K, K).astype(float)
        exp = np.outer(cm.sum(1), cm.sum(0)) / cm.sum()
        den = (w * exp).sum()
        out[b] = 1.0 - (w * cm).sum() / den if den > 0 else 0.0
    return out


def metric_resamples(name: str, yt, yp, idx) -> np.ndarray:
    if name == "macro_f1":
        return ex.macro_f1_resamples(yt, yp, idx, K)
    if name == "qwk":
        return qwk_resamples(yt, yp, idx)
    d = np.abs(yt[idx] - yp[idx])
    return d.mean(1) if name == "mae" else 100 * (d >= 3).mean(1)       # "jauh_pct"


METRICS = ("macro_f1", "qwk", "mae", "jauh_pct")


def fit_two(model: str, tr_docs, y_tr, te_sets: dict[str, pd.Series]) -> dict[str, np.ndarray]:
    vec = ex.make_vectorizer(MAX_FEATURES)
    m = ex.make_model(model).fit(vec.fit_transform(tr_docs), y_tr)
    return {k: np.array([TO_INT[v] for v in m.predict(vec.transform(d))]) for k, d in te_sets.items()}


def ci(a: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="archive")
    ap.add_argument("--pairs", default="out_laporan_awal/pasangan_laporan_awal.csv")
    ap.add_argument("--out-dir", default="results_task_20261014_ketahanan")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    tr, te, info = ex.build_initial_conditions(Path(args.archive), Path(args.pairs))
    raw = pd.read_csv(Path(args.archive) / "sev.csv")
    low = raw["Description"].fillna("").astype(str).str.lower()
    key = raw["Unnamed: 0"].values                                  # = `_orig_idx` dari cleaning.stage_a_structure
    flag_dup = pd.Series(low.str.contains(DUP_SELF).values, index=key)
    flag_mas = pd.Series(low.str.contains(DUP_MASTER).values, index=key)
    for d in (tr, te):
        d["dup"] = flag_dup.loc[d["_orig_idx"]].values
        d["master"] = flag_mas.loc[d["_orig_idx"]].values
    print(f"train {len(tr)} / test {len(te)}; duplikat train {tr['dup'].mean():.1%}, test {te['dup'].mean():.1%}; "
          f"master test {te['master'].mean():.1%}  ({time.time() - t0:.0f} dtk)")

    y_tr = tr["Severity"].values
    yt = np.array([TO_INT[v] for v in te["Severity"].values])
    n = len(yt)
    idx = np.random.default_rng(args.boot_seed).integers(0, n, size=(args.n_boot, n))
    full = np.arange(n)[None, :]

    # ---------------- prediksi model (sama dengan §8f) -----------------------------------------------------------
    preds: dict[str, np.ndarray] = {}
    for m in MODELS:
        p = fit_two(m, tr["Description"], y_tr, {"ref": te["Description"], "ref_to_initial": te["init"]})
        preds[f"{m}|ref"], preds[f"{m}|ref_to_initial"] = p["ref"], p["ref_to_initial"]
        preds[f"{m}|initial"] = fit_two(m, tr["init"], y_tr, {"initial": te["init"]})["initial"]
        print(f"  {m} selesai ({time.time() - t0:.0f} dtk)")
    repro = []
    for m in MODELS:
        for arm in ("ref", "initial", "ref_to_initial"):
            v = float(ex.macro_f1_resamples(yt, preds[f"{m}|{arm}"], full, K)[0])
            repro.append({"model": m, "lengan": arm, "macro_f1": round(v, 4), "macro_f1_8f": REF_8F[m][arm],
                          "selisih": round(v - REF_8F[m][arm], 4)})
    repro = pd.DataFrame(repro)
    repro.to_csv(out / "reproduksi_8f.csv", index=False)
    print(repro.to_string(index=False))

    # ---------------- #15 baseline -------------------------------------------------------------------------------
    maj = pd.Series(y_tr).value_counts().idxmax()
    base = {"mayoritas": np.full(n, TO_INT[maj])}
    for name, rules in KEYWORD_RULES.items():
        base[f"{name}|initial"] = keyword_predict(te["init"], rules)
        base[f"{name}|ref"] = keyword_predict(te["Description"], rules)
    p_train = pd.Series(y_tr).map(TO_INT).value_counts(normalize=True).reindex(range(K), fill_value=0).values
    rng = np.random.default_rng(20261014)
    strat = np.array([ex.macro_f1_resamples(yt, rng.choice(K, size=n, p=p_train), full, K)[0] for _ in range(1000)])
    rows = [{"prediktor": "acak berstrata (1000 undian)", "teks": "-", "macro_f1": round(float(strat.mean()), 4),
             "ci_lo": round(float(np.percentile(strat, 2.5)), 4), "ci_hi": round(float(np.percentile(strat, 97.5)), 4),
             "cakupan_aturan_pct": None}]
    allp = {**base, **preds}
    for name, p in allp.items():
        b = ex.macro_f1_resamples(yt, p, idx, K)
        lo, hi = ci(b)
        pred_name, _, teks = name.partition("|")
        cov = round(100 * float(np.mean(p != TO_INT["normal"])), 1) if pred_name.startswith("kata_kunci") else None
        rows.append({"prediktor": pred_name, "teks": teks or "-", "macro_f1": round(float(ex.macro_f1_resamples(
            yt, p, full, K)[0]), 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "cakupan_aturan_pct": cov})
    bl = pd.DataFrame(rows)
    bl.to_csv(out / "baseline.csv", index=False)
    # model − kata kunci (paired), pada teks yang setara saat triase (laporan awal)
    bd = []
    for m in MODELS:
        for arm in ("initial", "ref_to_initial"):
            for kw in KEYWORD_RULES:
                d = ex.macro_f1_resamples(yt, preds[f"{m}|{arm}"], idx, K) - ex.macro_f1_resamples(yt, base[f"{kw}|initial"], idx, K)
                lo, hi = ci(d)
                bd.append({"model": m, "lengan": arm, "dibanding": f"{kw} (laporan awal)",
                           "selisih_macro_f1": round(float(ex.macro_f1_resamples(yt, preds[f"{m}|{arm}"], full, K)[0]
                                                           - ex.macro_f1_resamples(yt, base[f"{kw}|initial"], full, K)[0]), 4),
                           "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
    bd = pd.DataFrame(bd)
    bd.to_csv(out / "selisih_model_vs_kata_kunci.csv", index=False)
    print(bl.to_string(index=False))
    print(bd.to_string(index=False))

    # ---------------- #18 metrik ordinal -------------------------------------------------------------------------
    orows, drows = [], []
    for name, p in allp.items():
        r = {"prediktor": name}
        for met in METRICS:
            r[met] = round(float(metric_resamples(met, yt, p, full)[0]), 4)
        orows.append(r)
    for m in MODELS:
        for a, b in (("ref_to_initial", "ref"), ("initial", "ref")):
            for met in METRICS:
                d = metric_resamples(met, yt, preds[f"{m}|{a}"], idx) - metric_resamples(met, yt, preds[f"{m}|{b}"], idx)
                est = float(metric_resamples(met, yt, preds[f"{m}|{a}"], full)[0] - metric_resamples(met, yt, preds[f"{m}|{b}"], full)[0])
                lo, hi = ci(d)
                drows.append({"model": m, "perbandingan": f"{a} - {b}", "metrik": met, "selisih": round(est, 4),
                              "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "ci_melewati_0": bool(lo <= 0 <= hi)})
    od, dd = pd.DataFrame(orows), pd.DataFrame(drows)
    od.to_csv(out / "metrik_ordinal.csv", index=False)
    dd.to_csv(out / "selisih_ordinal.csv", index=False)
    print(od.to_string(index=False))
    print(dd.to_string(index=False))

    # ---------------- #22 sensitivitas duplikat ------------------------------------------------------------------
    def deltas(variant: str, pr: dict[str, np.ndarray], y: np.ndarray, n_train: int) -> list[dict]:
        nn = len(y)
        ix = np.random.default_rng(args.boot_seed).integers(0, nn, size=(args.n_boot, nn))
        f = np.arange(nn)[None, :]
        res = []
        for m in MODELS:
            pt = {a: float(ex.macro_f1_resamples(y, pr[f"{m}|{a}"], f, K)[0]) for a in ("ref", "initial", "ref_to_initial")}
            bs = {a: ex.macro_f1_resamples(y, pr[f"{m}|{a}"], ix, K) for a in pt}
            for a, b in (("ref_to_initial", "ref"), ("initial", "ref")):
                lo, hi = ci(bs[a] - bs[b])
                res.append({"varian": variant, "model": m, "n_train": n_train, "n_test": nn, "ref": round(pt["ref"], 4),
                            "initial": round(pt["initial"], 4), "ref_to_initial": round(pt["ref_to_initial"], 4),
                            "perbandingan": f"{a} - {b}", "selisih": round(pt[a] - pt[b], 4), "ci_lo": round(lo, 4),
                            "ci_hi": round(hi, 4), "ci_melewati_0": bool(lo <= 0 <= hi)})
        return res

    srows = deltas("semua (= §8f)", preds, yt, len(tr))
    for variant, keep in (("D1 test tanpa duplikat", ~te["dup"].values),
                          ("D1b test tanpa duplikat & master", ~(te["dup"].values | te["master"].values))):
        srows += deltas(variant, {k: v[keep] for k, v in preds.items()}, yt[keep], len(tr))
    trk, tek = tr[~tr["dup"]].reset_index(drop=True), te[~te["dup"]].reset_index(drop=True)
    p2: dict[str, np.ndarray] = {}
    for m in MODELS:
        p = fit_two(m, trk["Description"], trk["Severity"].values, {"ref": tek["Description"], "ref_to_initial": tek["init"]})
        p2[f"{m}|ref"], p2[f"{m}|ref_to_initial"] = p["ref"], p["ref_to_initial"]
        p2[f"{m}|initial"] = fit_two(m, trk["init"], trk["Severity"].values, {"initial": tek["init"]})["initial"]
    srows += deltas("D2 train & test tanpa duplikat", p2, np.array([TO_INT[v] for v in tek["Severity"].values]), len(trk))
    sd = pd.DataFrame(srows)
    sd.to_csv(out / "sensitivitas_duplikat.csv", index=False)
    print(sd.to_string(index=False))

    # ---------------- #8 dekomposisi pengaturan literatur (dari task_banding_literatur.py, tanpa training baru) -------
    dec = pd.DataFrame()
    lit = Path("results_task_20261014_literatur") / "pengaturan_evaluasi.csv"
    if lit.exists():
        s = pd.read_csv(lit)
        drs = []
        for (lab, mod), g in s[s.model != "mayoritas"].groupby(["label", "model"], sort=False):
            v = {(r.split, r.lengan): r for r in g.itertuples()}
            for met in ("f1_macro", "akurasi"):
                b, t_, sp_, both = (getattr(v[k], met) for k in (("kronologis", "full_initial"), ("kronologis", "raw"),
                                                                 ("acak", "full_initial"), ("acak", "raw")))
                drs.append({"label": lab, "model": mod, "metrik": met, "dasar_kronologis_laporan_awal": round(b, 4),
                            "efek_teks_poin": round(100 * (t_ - b), 1), "efek_split_acak_poin": round(100 * (sp_ - b), 1),
                            "gabungan_poin": round(100 * (both - b), 1),
                            "interaksi_poin": round(100 * (both - t_ - sp_ + b), 1)})
        dec = pd.DataFrame(drs)
        dec.to_csv(out / "dekomposisi_pengaturan_literatur.csv", index=False)
        print(dec.to_string(index=False))

    secs = round(time.time() - t0)
    md = [f"# Angka otomatis task_uji_ketahanan_rq2.py ({secs} dtk)\n",
          f"train {len(tr)} / test {n}; duplikat train {tr['dup'].mean():.1%}, test {te['dup'].mean():.1%}; "
          f"master test {te['master'].mean():.1%}\n",
          "## Reproduksi §8f\n", ex_md(repro), "", "## #15 Baseline\n", ex_md(bl), "", ex_md(bd), "",
          "## #18 Metrik ordinal\n", ex_md(od), "", ex_md(dd), "", "## #22 Sensitivitas duplikat\n", ex_md(sd), "",
          "## #8 Dekomposisi pengaturan literatur (raw vs full_initial, tanpa CI; 3 ulangan acak)\n",
          ex_md(dec) if len(dec) else "(results_task_20261014_literatur/ tidak ada)", ""]
    (out / "angka_otomatis.md").write_text("\n".join(md), encoding="utf-8")
    (out / "hasil.json").write_text(json.dumps({"detik": secs, "config": vars(args), "info": info,
                                                "aturan_kata_kunci": KEYWORD_RULES}, indent=2), encoding="utf-8")
    print(f"Selesai {secs} dtk -> {out}/")
    return 0


def ex_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
