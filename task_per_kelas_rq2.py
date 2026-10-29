"""
task_per_kelas_rq2.py — Rincian per kelas over-estimasi RQ2 (16 Okt 2026; disetujui user: "sepertinya boleh di coba").

Desain sama dengan MASTER §8f/§8h: prediksi LogReg dan LinearSVC dihasilkan ulang lewat `task_uji_ketahanan_rq2.fit_two`
(macro-F1 diasersi sama dengan §8f), prediksi BERT dibaca dari `kaggle_pralatih/hasil_kaggle/prediksi_pralatih.csv`
(3 seed; distribusi bootstrap dirata-rata antar seed, seperti `analisis_pralatih.py`). Bootstrap berpasangan 1000x
seed 42 dengan indeks yang sama dengan §8f. Tidak ada konfigurasi model baru, tidak ada penyetelan.

Keluaran (`results_task_20261016_per_kelas/`):
  per_kelas.csv       precision / recall / F1 per kelas per lengan; selisih ref_to_initial − ref [CI 95%]; untuk F1 juga
                      sumbangan kelas ke selisih macro-F1 (selisih / 6, dalam poin)
  parah_biner.csv     kelas "parah" (critical + blocker) vs sisanya, dipetakan dari prediksi 6 kelas (tanpa model baru)
  confusion.csv       confusion matrix ternormalisasi baris (%), lengan ref dan ref_to_initial
  angka_otomatis.md   semua tabel di atas

Pakai:  python task_per_kelas_rq2.py [--out-dir results_task_20261016_per_kelas]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import experiment as ex
import task_uji_ketahanan_rq2 as uk

CLS, K, TO_INT = uk.CLS, uk.K, uk.TO_INT
ARMS = ("ref", "initial", "ref_to_initial")
PARAH = (TO_INT["critical"], TO_INT["blocker"])
METRIK = ("precision", "recall", "f1")


def per_class_resamples(yt: np.ndarray, yp: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(B, 3, K): precision, recall, F1 per kelas untuk setiap baris indeks resample."""
    out = np.zeros((len(idx), 3, K))
    for b in range(len(idx)):
        cm = np.bincount(yt[idx[b]] * K + yp[idx[b]], minlength=K * K).reshape(K, K).astype(float)
        tp, pp, ap = np.diag(cm), cm.sum(0), cm.sum(1)
        np.divide(tp, pp, out=out[b, 0], where=pp > 0)
        np.divide(tp, ap, out=out[b, 1], where=ap > 0)
        np.divide(2 * tp, pp + ap, out=out[b, 2], where=(pp + ap) > 0)
    return out


def severe_resamples(yt: np.ndarray, yp: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(B, 3): precision, recall, F1 kelas parah (critical + blocker) vs sisanya."""
    t, p = np.isin(yt, PARAH)[idx], np.isin(yp, PARAH)[idx]
    tp, pp, ap = (t & p).sum(1).astype(float), p.sum(1).astype(float), t.sum(1).astype(float)
    return np.stack([np.divide(tp, pp, out=np.zeros_like(tp), where=pp > 0),
                     np.divide(tp, ap, out=np.zeros_like(tp), where=ap > 0),
                     np.divide(2 * tp, pp + ap, out=np.zeros_like(tp), where=(pp + ap) > 0)], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="archive")
    ap.add_argument("--pairs", default="out_laporan_awal/pasangan_laporan_awal.csv")
    ap.add_argument("--bert", default="kaggle_pralatih/hasil_kaggle/prediksi_pralatih.csv")
    ap.add_argument("--out-dir", default="results_task_20261016_per_kelas")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    tr, te, _ = ex.build_initial_conditions(Path(args.archive), Path(args.pairs))
    y_tr = tr["Severity"].values
    yt = np.array([TO_INT[v] for v in te["Severity"].values])
    n = len(yt)
    idx = np.random.default_rng(args.boot_seed).integers(0, n, size=(args.n_boot, n))
    full = np.arange(n)[None, :]

    # prediksi per model: {model: {lengan: [prediksi per seed]}}
    P: dict[str, dict[str, list[np.ndarray]]] = {}
    for m in uk.MODELS:
        p = uk.fit_two(m, tr["Description"], y_tr, {"ref": te["Description"], "ref_to_initial": te["init"]})
        p["initial"] = uk.fit_two(m, tr["init"], y_tr, {"initial": te["init"]})["initial"]
        for arm in ARMS:
            v = float(ex.macro_f1_resamples(yt, p[arm], full, K)[0])
            assert abs(v - uk.REF_8F[m][arm]) < 1e-4, f"{m} {arm}: {v:.4f} != §8f {uk.REF_8F[m][arm]}"
        P[m] = {arm: [p[arm]] for arm in ARMS}
        print(f"  {m} selesai, sama dengan §8f ({time.time() - t0:.0f} dtk)")
    pr = pd.read_csv(args.bert)
    P["bert"] = {arm: [] for arm in ARMS}
    for (s, arm), g in pr.groupby(["seed", "lengan"]):
        assert (g["_orig_idx"].values == te["_orig_idx"].values).all(), f"urutan test BERT beda dari §8f ({s}, {arm})"
        assert (g["y_true"].values == yt).all(), "label test BERT beda dari §8f"
        P["bert"][arm].append(g["y_pred"].values.astype(int))
    print(f"  bert: {len(P['bert']['ref'])} seed dibaca")

    rows, srows, crows = [], [], []
    for m, arms in P.items():
        pt = {a: np.mean([per_class_resamples(yt, p, full)[0] for p in arms[a]], axis=0) for a in ARMS}
        boot = {a: [per_class_resamples(yt, p, idx) for p in arms[a]] for a in ("ref", "ref_to_initial")}
        d = np.mean([r2i - ref for r2i, ref in zip(boot["ref_to_initial"], boot["ref"])], axis=0)   # (B, 3, K)
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        for k, c in enumerate(CLS):
            for j, met in enumerate(METRIK):
                sel = pt["ref_to_initial"][j, k] - pt["ref"][j, k]
                rows.append({"model": m, "kelas": c, "n_test": int((yt == k).sum()), "metrik": met,
                             **{a: round(float(pt[a][j, k]), 4) for a in ARMS}, "selisih": round(float(sel), 4),
                             "ci_lo": round(float(lo[j, k]), 4), "ci_hi": round(float(hi[j, k]), 4),
                             "ci_melewati_0": bool(lo[j, k] <= 0 <= hi[j, k]),
                             "sumbangan_macro_poin": round(100 * float(sel) / K, 2) if met == "f1" else None})

        spt = {a: np.mean([severe_resamples(yt, p, full)[0] for p in arms[a]], axis=0) for a in ARMS}
        sd = np.mean([severe_resamples(yt, r2i, idx) - severe_resamples(yt, ref, idx)
                      for r2i, ref in zip(arms["ref_to_initial"], arms["ref"])], axis=0)
        slo, shi = np.percentile(sd, [2.5, 97.5], axis=0)
        for j, met in enumerate(METRIK):
            srows.append({"model": m, "metrik": f"{met} kelas parah", **{a: round(float(spt[a][j]), 4) for a in ARMS},
                          "selisih": round(float(spt["ref_to_initial"][j] - spt["ref"][j]), 4),
                          "ci_lo": round(float(slo[j]), 4), "ci_hi": round(float(shi[j]), 4),
                          "ci_melewati_0": bool(slo[j] <= 0 <= shi[j])})

        for a in ("ref", "ref_to_initial"):
            cm = sum(np.bincount(yt * K + p, minlength=K * K).reshape(K, K) for p in arms[a]).astype(float)
            cm = 100 * cm / cm.sum(1, keepdims=True)
            for k, c in enumerate(CLS):
                crows.append({"model": m, "lengan": a, "kelas_benar": c, **{f"ke_{cc}": round(float(cm[k, kk]), 1)
                                                                             for kk, cc in enumerate(CLS)}})

    per, sev, conf = pd.DataFrame(rows), pd.DataFrame(srows), pd.DataFrame(crows)
    per.to_csv(out / "per_kelas.csv", index=False)
    sev.to_csv(out / "parah_biner.csv", index=False)
    conf.to_csv(out / "confusion.csv", index=False)

    f1 = per[per.metrik == "f1"].drop(columns="metrik")
    md = ["# Rincian per kelas over-estimasi RQ2\n",
          f"Test {n} dokumen (= §8f); bootstrap berpasangan {args.n_boot}x seed {args.boot_seed}; BERT = rata-rata "
          f"{len(P['bert']['ref'])} seed. Selisih = ref_to_initial − ref.\n",
          "## F1 per kelas\n", uk.ex_md(f1), "", "## Precision dan recall per kelas\n",
          uk.ex_md(per[per.metrik != "f1"].drop(columns="sumbangan_macro_poin")), "",
          "## Kelas parah (critical + blocker) vs sisanya\n", uk.ex_md(sev), "",
          "## Confusion matrix (% per baris kelas benar)\n", uk.ex_md(conf), ""]
    (out / "angka_otomatis.md").write_text("\n".join(md), encoding="utf-8")
    print(f1.to_string(index=False))
    print(sev.to_string(index=False))
    print(f"-> {out}/  ({time.time() - t0:.0f} dtk)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
