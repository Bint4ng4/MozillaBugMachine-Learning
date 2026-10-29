"""
analisis_pralatih.py — Analisis lokal hasil Kaggle `rq2_pralatih_kaggle.py` (uji ketahanan RQ2 dengan BERT).

Memeriksa bahwa dokumen dan urutan test sama dengan §8f, lalu menghitung macro-F1 dan metrik ordinal (QWK, MAE) per
seed dan rata-rata 3 seed, selisih ref_to_initial − ref dan initial − ref dengan bootstrap berpasangan 1000x seed 42 —
indeks resample-nya identik dengan §8f (n test sama), jadi CI langsung sebanding dengan model linear.
Rata-rata seed: distribusi bootstrap dirata-rata antar seed (sama dengan perlakuan kontrol 5 seed di experiment.py).
Ditambah (15 Okt): BERT saat triase − aturan kata kunci K1/K2 pada laporan awal (#15), macro-F1 dan QWK.

Pakai:  python kaggle_pralatih/analisis_pralatih.py [--hasil kaggle_pralatih/hasil_kaggle] [--out-dir results_pralatih]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import experiment as ex  # noqa: E402
import task_uji_ketahanan_rq2 as uk  # noqa: E402

LINEAR_8F = {"logreg": {"ref": 0.3075, "initial": 0.2752, "ref_to_initial": 0.2724,
                        "ref_to_initial - ref": (-0.035, -0.051, -0.018), "initial - ref": (-0.032, -0.050, -0.014)},
             "linearsvc": {"ref": 0.3149, "initial": 0.2632, "ref_to_initial": 0.2533,
                           "ref_to_initial - ref": (-0.062, -0.084, -0.040), "initial - ref": (-0.052, -0.073, -0.031)}}
ARMS = ("ref", "initial", "ref_to_initial")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hasil", default=str(ROOT / "kaggle_pralatih" / "hasil_kaggle"))
    ap.add_argument("--out-dir", default=str(ROOT / "results_pralatih"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=42)
    args = ap.parse_args()
    hasil, out = Path(args.hasil), Path(args.out_dir)
    out.mkdir(exist_ok=True)

    pr = pd.read_csv(hasil / "prediksi_pralatih.csv")
    _, te, _ = ex.build_initial_conditions(ROOT / "archive", ROOT / "out_laporan_awal" / "pasangan_laporan_awal.csv")
    yt = np.array([uk.TO_INT[v] for v in te["Severity"].values])
    n = len(yt)
    seeds = sorted(pr["seed"].unique())
    P: dict[tuple[int, str], np.ndarray] = {}
    for (s, arm), g in pr.groupby(["seed", "lengan"]):
        assert (g["_orig_idx"].values == te["_orig_idx"].values).all(), f"urutan test beda dari §8f (seed {s}, {arm})"
        assert (g["y_true"].values == yt).all(), "label test beda dari §8f"
        P[(s, arm)] = g["y_pred"].values.astype(int)
    assert all((s, a) in P for s in seeds for a in ARMS), "ada seed/lengan yang belum lengkap"
    idx = np.random.default_rng(args.boot_seed).integers(0, n, size=(args.n_boot, n))
    full = np.arange(n)[None, :]

    rows, drows = [], []
    for met in uk.METRICS:
        boot = {(s, a): uk.metric_resamples(met, yt, P[(s, a)], idx) for s in seeds for a in ARMS}
        pt = {(s, a): float(uk.metric_resamples(met, yt, P[(s, a)], full)[0]) for s in seeds for a in ARMS}
        for s in list(seeds) + ["rata2"]:
            ss = seeds if s == "rata2" else [s]
            for a in ARMS:
                rows.append({"metrik": met, "seed": s, "lengan": a, "nilai": round(float(np.mean([pt[(x, a)] for x in ss])), 4),
                             "sd_antar_seed": round(float(np.std([pt[(x, a)] for x in ss], ddof=1)), 4) if len(ss) > 1 else None})
            for a, b in (("ref_to_initial", "ref"), ("initial", "ref"), ("ref_to_initial", "initial")):
                d = np.mean([boot[(x, a)] - boot[(x, b)] for x in ss], axis=0)
                est = float(np.mean([pt[(x, a)] - pt[(x, b)] for x in ss]))
                lo, hi = np.percentile(d, [2.5, 97.5])
                drows.append({"metrik": met, "seed": s, "perbandingan": f"{a} - {b}", "selisih": round(est, 4),
                              "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4), "ci_melewati_0": bool(lo <= 0 <= hi)})
    skor, sel = pd.DataFrame(rows), pd.DataFrame(drows)
    skor.to_csv(out / "skor_pralatih.csv", index=False)
    sel.to_csv(out / "selisih_pralatih.csv", index=False)

    lin = []
    for m, v in LINEAR_8F.items():
        for comp in ("ref_to_initial - ref", "initial - ref"):
            lin.append({"model": m, "perbandingan": comp, "ref": v["ref"], "selisih": v[comp][0], "ci_lo": v[comp][1],
                        "ci_hi": v[comp][2]})
    r = sel[(sel.metrik == "macro_f1") & (sel.seed == "rata2") & (sel.perbandingan != "ref_to_initial - initial")]
    ref_bert = float(skor[(skor.metrik == "macro_f1") & (skor.seed == "rata2") & (skor.lengan == "ref")]["nilai"].iloc[0])
    for row in r.itertuples():
        lin.append({"model": "bert-base-uncased (rata-rata seed)", "perbandingan": row.perbandingan, "ref": ref_bert,
                    "selisih": row.selisih, "ci_lo": row.ci_lo, "ci_hi": row.ci_hi})
    banding = pd.DataFrame(lin)
    # porsi keunggulan atas tebakan mayoritas (`normal`) yang hilang saat triase: −selisih / (ref − mayoritas)
    maj = float(uk.metric_resamples("macro_f1", yt, np.full(n, uk.TO_INT["normal"]), full)[0])
    banding["porsi_keunggulan_hilang_pct"] = (100 * -banding["selisih"] / (banding["ref"] - maj)).round(1)
    banding.to_csv(out / "banding_linear_vs_pralatih.csv", index=False)

    # BERT − kata kunci (#15, paired), pada teks yang setara saat triase (laporan awal); rata-rata seed seperti di atas
    kw = {name: uk.keyword_predict(te["init"], rules) for name, rules in uk.KEYWORD_RULES.items()}
    krows = []
    for met in ("macro_f1", "qwk"):
        for arm in ("initial", "ref_to_initial"):
            for name, pk in kw.items():
                kb = uk.metric_resamples(met, yt, pk, idx)
                kp = float(uk.metric_resamples(met, yt, pk, full)[0])
                d = np.mean([uk.metric_resamples(met, yt, P[(s, arm)], idx) - kb for s in seeds], axis=0)
                est = float(np.mean([uk.metric_resamples(met, yt, P[(s, arm)], full)[0] - kp for s in seeds]))
                lo, hi = np.percentile(d, [2.5, 97.5])
                krows.append({"metrik": met, "lengan": arm, "dibanding": f"{name} (laporan awal)", "nilai_kata_kunci":
                              round(kp, 4), "selisih": round(est, 4), "ci_lo": round(float(lo), 4),
                              "ci_hi": round(float(hi), 4), "ci_melewati_0": bool(lo <= 0 <= hi)})
    kk = pd.DataFrame(krows)
    kk.to_csv(out / "selisih_pralatih_vs_kata_kunci.csv", index=False)

    md = ["# Hasil uji ketahanan RQ2 dengan model pralatih\n", f"Seed: {seeds}; n test {n} (sama dengan §8f).\n",
          "## Skor\n", uk.ex_md(skor), "", "## Selisih (bootstrap berpasangan, indeks = §8f)\n", uk.ex_md(sel), "",
          "## Banding dengan model linear (§8f, macro-F1)\n", uk.ex_md(banding), "",
          "## BERT − kata kunci pada laporan awal (#15)\n", uk.ex_md(kk), ""]
    log = hasil / "log_epoch.csv"
    if log.exists():
        lg = pd.read_csv(log)
        md += ["## Log epoch (train vs val loss)\n", uk.ex_md(lg), ""]
    info = hasil / "info_run.json"
    if info.exists():
        md += ["## info_run.json\n", "```json", info.read_text(encoding="utf-8"), "```"]
    (out / "angka_otomatis.md").write_text("\n".join(md), encoding="utf-8")
    print(banding.to_string(index=False))
    print(kk.to_string(index=False))
    print(f"-> {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
