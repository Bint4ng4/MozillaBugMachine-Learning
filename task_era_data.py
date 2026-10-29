"""
task_era_data.py — Era data dan bukti split kronologis `archive/sev.csv` (paket MASTER 15 Okt 2026; menutup MASTER §4.7
"verifikasi dulu" dan §14.3). Tanpa model, tanpa training.

Tahun per dokumen = median semua tahun yang disebut di teks (regex §14.3: 1980–2029). Kepala "Created by ... <tahun>"
tidak dipakai: hanya ada di 0,8% dokumen (262, semuanya 1998, desil indeks pertama).
Keluaran: % dokumen yang menyebut tahun, % yang menyebut tahun >= 2002, kata "netscape"/"firefox", Spearman indeks vs
tahun (CI bootstrap 1000x seed 42), dan sebaran tahun per desil indeks serta train (<= 28.505) vs test (>= 28.507).
Ditambah (16 Okt): sensitivitas rentang tahun (1990–2019 = hitungan chat 14 Okt; 1990–2002 = rentang era data) dan
jumlah angka di luar 1990–2019 yang ikut tertangkap regex §14.3 (kebanyakan bukan tahun).

Pakai:  python task_era_data.py [--out-dir results_task_20261015_era]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from task_uji_ketahanan_rq2 import ex_md

YEAR = re.compile(r"\b(19[89]\d|20[0-2]\d)\b")
TEST_START = 28507
VARIAN = {"1980–2029 (regex §14.3, utama)": YEAR,
          "1990–2019 (hitungan chat 14 Okt)": re.compile(r"\b(199\d|200\d|201\d)\b"),
          "1990–2002 (rentang era data)": re.compile(r"\b(199\d|200[0-2])\b")}


def rho_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    boot = [spearmanr(x[i], y[i]).statistic for i in (rng.integers(0, len(x), len(x)) for _ in range(n_boot))]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(spearmanr(x, y).statistic), float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="archive")
    ap.add_argument("--out-dir", default="results_task_20261015_era")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    raw = pd.read_csv(Path(args.archive) / "sev.csv")
    txt = raw["Description"].fillna("").astype(str)
    idx = raw["Unnamed: 0"].values
    years = txt.map(lambda t: [int(y) for y in YEAR.findall(t)])
    tahun = years.map(lambda ys: float(np.median(ys)) if ys else np.nan).values
    ok = ~np.isnan(tahun)
    low = txt.str.lower()

    xs, ys = idx[ok], tahun[ok]
    rho, lo, hi = rho_ci(xs, ys, args.n_boot, args.boot_seed)
    q = np.percentile(ys, [5, 50, 95])
    ring = pd.DataFrame([
        {"ukuran": "n dokumen", "nilai": len(raw)},
        {"ukuran": "% dokumen menyebut tahun", "nilai": round(100 * ok.mean(), 1)},
        {"ukuran": "% dokumen menyebut tahun >= 2002", "nilai": round(100 * np.mean(years.map(lambda v: any(y >= 2002 for y in v))), 1)},
        {"ukuran": "% dokumen memuat 'netscape'", "nilai": round(100 * np.mean(low.str.contains(r"\bnetscape\b")), 1)},
        {"ukuran": "% dokumen memuat 'firefox'", "nilai": round(100 * np.mean(low.str.contains(r"\bfirefox\b")), 1)},
        {"ukuran": f"Spearman indeks vs tahun [CI 95%] (n={int(ok.sum())})", "nilai": f"{rho:.3f} [{lo:.3f}; {hi:.3f}]"},
        {"ukuran": "tahun: P5 / median / P95", "nilai": f"{q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f}"},
    ])
    ring.to_csv(out / "ringkasan_era.csv", index=False)

    rows = []
    desil = pd.qcut(idx, 10, labels=False)
    for grup, mask in [(f"desil {d + 1}", desil == d) for d in range(10)] + \
                      [("train (<= 28.505)", idx < TEST_START), ("test (>= 28.507)", idx >= TEST_START)]:
        v = tahun[mask & ok]
        rows.append({"grup": grup, "indeks_min": int(idx[mask].min()), "indeks_maks": int(idx[mask].max()),
                     "n": int(mask.sum()), "n_menyebut_tahun": len(v), "median": float(np.median(v)),
                     "rata2": round(float(np.mean(v)), 2), "pct_2001_ke_atas": round(100 * float(np.mean(v >= 2001)), 1)})
    per = pd.DataFrame(rows)
    per.to_csv(out / "tahun_per_desil.csv", index=False)

    srows = []
    for nama, pat in VARIAN.items():
        th = txt.map(lambda t: [int(y) for y in pat.findall(t)]).map(lambda v: float(np.median(v)) if v else np.nan).values
        m = ~np.isnan(th)
        r, a, b = rho_ci(idx[m], th[m], args.n_boot, args.boot_seed)
        srows.append({"rentang_tahun": nama, "pct_dokumen_menyebut_tahun": round(100 * m.mean(), 1), "n": int(m.sum()),
                      "rho": round(r, 3), "ci_lo": round(a, 3), "ci_hi": round(b, 3)})
    sens = pd.DataFrame(srows)
    sens.to_csv(out / "sensitivitas_rentang_tahun.csv", index=False)
    semua = pd.Series([y for v in years for y in v])
    luar = semua[(semua < 1990) | (semua > 2019)]
    catatan = (f"Angka yang ditangkap regex §14.3 di luar 1990–2019: {len(luar)} dari {len(semua)} "
               f"({100 * len(luar) / len(semua):.1f}%); terbanyak: "
               + ", ".join(f"{k} ({v}x)" for k, v in luar.value_counts().head(5).items()))

    md = ["# Era data dan bukti split kronologis `sev.csv`\n", ex_md(ring), "",
          "## Tahun per desil indeks dan per split\n", ex_md(per), "",
          "## Sensitivitas rentang tahun (Spearman indeks vs median tahun yang disebut)\n", ex_md(sens), "", catatan, ""]
    (out / "angka_otomatis.md").write_text("\n".join(md), encoding="utf-8")
    print(ring.to_string(index=False))
    print(per.to_string(index=False))
    print(sens.to_string(index=False))
    print(catatan)
    print(f"-> {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
