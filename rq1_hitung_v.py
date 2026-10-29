"""
rq1_hitung_v.py — Langkah 5 runbook (MASTER SS14.2): prevalensi penanda dan Cramér's V untuk RQ1, lengkap dengan
interval kepercayaan, untuk DUA definisi penanda pada data yang SAMA, plus estimasi prevalensi kontaminasi dari audit.

  ANY_MARKER   verify_data.py (definisi RQ1 di MASTER SS4.6; 85,1% pada sev.csv mentah)
  POST_REPORT  cleaning.py (dipakai TAILCUT dan untuk memilih sampel audit langkah 3)

Data: archive/sev.csv mentah (32.428 baris) dan out_20261008/cleaned_baseline_{train,test}.csv (tahap A, kerangka
sampel audit).

Estimasi prevalensi kontaminasi = P(penanda) + P(tanpa penanda) x P(terkontaminasi | tanpa penanda). P(terkontaminasi |
penanda) diasumsikan 1: penanda adalah teks pasca-pelaporan menurut definisinya (didukung audit 200 dokumen
"verified" pada 9 Sep, MASTER SS4.6). P(terkontaminasi | tanpa penanda) dari lembar audit kelompok B (label K saja).

Catatan tafsir (kontrol positif langkah 4): V mengukur keterkaitan KEHADIRAN penanda dengan severity, bukan apakah
ISI kontaminasi membocorkan label -- token bocor yang menaikkan macro-F1 sampai 10 poin hanya memberi V kehadiran 0,013.

Pakai:  python rq1_hitung_v.py [--n-boot 1000] [--out-dir results_rq1]
Keluaran: results_rq1/rq1_v.json, results_rq1/tabel_rq1_v.csv, results_rq1/tabel_rq1_prevalensi_audit.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cleaning  # noqa: E402
from audit_hitung import wilson  # noqa: E402
from verify_data import ANY_MARKER, SEVERE, cramers_v  # noqa: E402

AUDIT_SHEETS = {"AI_pembanding_8Okt": "audit_model_pembanding.csv",
                "AI_lembarA_10Okt": "audit_penilai_A.csv",
                "AI_lembarB_10Okt": "audit_penilai_B.csv"}
AUDIT_KEY = "audit_langkah3/audit_KUNCI_jangan_dibuka_penilai.csv"


def v_from_codes(a: np.ndarray, b: np.ndarray, ka: int, kb: int) -> float:
    """Cramér's V dari dua array kode integer (rumus sama dengan verify_data.cramers_v, tanpa koreksi Yates).
    Kategori yang tidak muncul dibuang dulu, seperti pd.crosstab."""
    ct = np.bincount(a * kb + b, minlength=ka * kb).reshape(ka, kb).astype(float)
    ct = ct[ct.sum(1) > 0][:, ct.sum(0) > 0]
    n = ct.sum()
    exp = ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / n
    chi2 = float(((ct - exp) ** 2 / exp).sum())
    k = min(ct.shape)
    return float(np.sqrt(chi2 / (n * (k - 1)))) if k > 1 else 0.0


def marker_stats(has: pd.Series, sev: pd.Series, n_boot: int, seed: int) -> dict:
    n, k = len(has), int(has.sum())
    lo, hi = wilson(k, n)
    v, chi2, dof = cramers_v(has, sev)
    cats = sorted(sev.unique())
    a = has.values.astype(int)
    b = sev.map({c: i for i, c in enumerate(cats)}).values
    rng = np.random.default_rng(seed)
    boot = np.array([v_from_codes(a[i], b[i], 2, len(cats)) for i in rng.integers(0, n, size=(n_boot, n))])
    p_yes = sev[has].isin(SEVERE).mean()
    p_no = sev[~has].isin(SEVERE).mean()
    return {"n": n, "n_penanda": k, "pct": 100 * k / n, "pct_ci95": [100 * lo, 100 * hi],
            "chi2": chi2, "df": dof, "p": float(chi2_dist.sf(chi2, dof)), "V": v,
            "V_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "P_severe_ada": float(p_yes), "P_severe_tanpa": float(p_no), "dP_severe_pp": 100 * float(p_yes - p_no)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="archive")
    ap.add_argument("--baseline-dir", default="out_20261008")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results_rq1")
    args = ap.parse_args()

    raw = pd.read_csv(Path(args.archive) / "sev.csv")
    raw["Description"] = raw["Description"].fillna("").astype(str)
    base = pd.concat([pd.read_csv(Path(args.baseline_dir) / f"cleaned_baseline_{s}.csv") for s in ("train", "test")],
                     ignore_index=True)
    base["Description"] = base["Description"].fillna("").astype(str)
    print(f"sev.csv mentah: {len(raw)} baris | baseline (tahap A, kerangka audit): {len(base)} baris\n")

    defs = {("ANY_MARKER", "sev.csv mentah"): raw["Description"].str.contains(ANY_MARKER),
            ("POST_REPORT", "sev.csv mentah"): raw["Description"].str.contains(cleaning.POST_REPORT),
            ("POST_REPORT", "baseline"): base["Description"].str.contains(cleaning.POST_REPORT)}
    rows, report = [], {"definisi": {}}
    for (name, data), has in defs.items():
        sev = (raw if data == "sev.csv mentah" else base)["Severity"]
        st = marker_stats(has, sev, args.n_boot, args.seed)
        report["definisi"][f"{name} | {data}"] = st
        rows.append({"definisi": name, "data": data, **{k: v for k, v in st.items() if not isinstance(v, list)},
                     "pct_ci_lo": st["pct_ci95"][0], "pct_ci_hi": st["pct_ci95"][1],
                     "V_ci_lo": st["V_ci95"][0], "V_ci_hi": st["V_ci95"][1]})
        print(f"{name:<11} | {data:<15} penanda {st['n_penanda']:>6}/{st['n']} = {st['pct']:.2f}% "
              f"[{st['pct_ci95'][0]:.2f}, {st['pct_ci95'][1]:.2f}] | chi2 {st['chi2']:.1f} df {st['df']} "
              f"p {st['p']:.1e} | V {st['V']:.4f} [{st['V_ci95'][0]:.4f}, {st['V_ci95'][1]:.4f}] | "
              f"dP(severe) {st['dP_severe_pp']:+.1f} pp")

    a, p = defs[("ANY_MARKER", "sev.csv mentah")], defs[("POST_REPORT", "sev.csv mentah")]
    overlap = {"keduanya": int((a & p).sum()), "hanya_ANY_MARKER": int((a & ~p).sum()),
               "hanya_POST_REPORT": int((~a & p).sum()), "tidak_keduanya": int((~a & ~p).sum())}
    report["irisan_definisi_mentah"] = overlap
    print(f"\nIrisan dua definisi (sev.csv mentah): {overlap}")
    ex = raw.loc[~a & p, "Description"].head(400).map(lambda t: cleaning.POST_REPORT.search(t).group(0).lower())
    report["contoh_penanda_hanya_POST_REPORT"] = ex.value_counts().head(8).to_dict()
    print(f"  penanda yang hanya ditangkap POST_REPORT (400 contoh pertama): {report['contoh_penanda_hanya_POST_REPORT']}")

    # estimasi prevalensi kontaminasi dari audit (kerangka sampel = baseline + POST_REPORT)
    pm = report["definisi"]["POST_REPORT | baseline"]["pct"] / 100
    key = pd.read_csv(AUDIT_KEY, encoding="utf-8-sig").set_index("item")
    grp_b = key.index[key["group"] == "B_tanpa_penanda"]
    prev_rows = []
    print(f"\nEstimasi prevalensi kontaminasi = {100 * pm:.2f}% + {100 * (1 - pm):.2f}% x P(K | tanpa penanda):")
    for rater, path in AUDIT_SHEETS.items():
        if not Path(path).exists():
            continue
        lab = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False).set_index("item")["label"].str.strip().str.upper()
        kb, nb = int((lab.loc[grp_b] == "K").sum()), len(grp_b)
        lo, hi = wilson(kb, nb)
        est = [100 * (pm + (1 - pm) * x) for x in (kb / nb, lo, hi)]
        prev_rows.append({"penilai": rater, "jenis": "AI", "K_tanpa_penanda": kb, "n_tanpa_penanda": nb,
                          "pct_K_tanpa_penanda": 100 * kb / nb, "prevalensi_est": est[0],
                          "prevalensi_ci_lo": est[1], "prevalensi_ci_hi": est[2]})
        print(f"  {rater:<20} K {kb}/{nb} ({100 * kb / nb:.0f}%, [{100 * lo:.0f}, {100 * hi:.0f}]) -> "
              f"{est[0]:.1f}% [{est[1]:.1f}, {est[2]:.1f}]")
    report["prevalensi_audit"] = prev_rows
    report["catatan"] = ("Penilai audit saat ini semuanya AI (provisional); validasi manusia subset 40 item belum ada. "
                         "V mengukur keterkaitan KEHADIRAN penanda, bukan kebocoran isi (kontrol positif langkah 4).")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "tabel_rq1_v.csv", index=False)
    pd.DataFrame(prev_rows).to_csv(out / "tabel_rq1_prevalensi_audit.csv", index=False)
    (out / "rq1_v.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\nDisimpan: {out / 'rq1_v.json'}, tabel_rq1_v.csv, tabel_rq1_prevalensi_audit.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
