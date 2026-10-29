"""
audit_hitung.py — Menghitung hasil audit manual dokumen terpotong (runbook langkah 3).

Masukan: lembar penilai yang sudah diisi (kolom `label` berisi B/R/K) + berkas kunci.
Keluaran: ringkasan di layar, audit_langkah3/audit_hasil.json, audit_langkah3/audit_perbedaan.csv.

Definisi hasil:
  - "terkontaminasi" = label K. R (ragu) dihitung TIDAK terkontaminasi pada skenario longgar dan
    TERKONTAMINASI pada skenario ketat, sehingga hasil dilaporkan sebagai rentang.
  - Dua penilai: skenario "konsensus ketat" (K bila salah satu menilai K) dan "konsensus longgar"
    (K bila keduanya K). Cohen's kappa dihitung untuk K vs bukan-K (utama) dan 3 kelas (B/R/K).
  - Proporsi dilaporkan dengan interval Wilson 95%.

Pakai:  python audit_hitung.py --a audit_langkah3/audit_penilai_A.csv [--b audit_langkah3/audit_penilai_B.csv]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

VALID = {"B", "R", "K"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def load_sheet(path: Path) -> pd.Series:
    df = pd.read_csv(path, encoding="utf-8-sig")
    lab = df["label"].fillna("").astype(str).str.strip().str.upper()
    bad = df.loc[~lab.isin(VALID), "item"].tolist()
    if bad:
        raise SystemExit(f"FATAL: {path.name}: label tidak valid/kosong di item {bad[:15]}"
                         f"{' ...' if len(bad) > 15 else ''} (hanya B/R/K).")
    return pd.Series(lab.values, index=df["item"].values, name=path.stem)


def rate(lab: pd.Series, mask: pd.Series, strict: bool) -> dict:
    sub = lab[mask]
    k = int(((sub == "K") | ((sub == "R") if strict else False)).sum())
    lo, hi = wilson(k, len(sub))
    return {"n": int(len(sub)), "k": k, "pct": round(100 * k / len(sub), 1) if len(sub) else float("nan"),
            "ci95": [round(100 * lo, 1), round(100 * hi, 1)]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="lembar penilai pertama (terisi)")
    ap.add_argument("--b", help="lembar penilai kedua (terisi); opsional tapi sangat disarankan")
    ap.add_argument("--key", default="audit_langkah3/audit_KUNCI_jangan_dibuka_penilai.csv")
    ap.add_argument("--out", default="audit_langkah3")
    args = ap.parse_args()

    key = pd.read_csv(args.key, encoding="utf-8-sig").set_index("item")
    a = load_sheet(Path(args.a))
    raters = {"penilai_A": a}
    if args.b:
        raters["penilai_B"] = load_sheet(Path(args.b))
    for name, s in raters.items():
        if set(s.index) != set(key.index):
            raise SystemExit(f"FATAL: item {name} tidak cocok dengan berkas kunci.")

    groups = {"A_dipotong": key["group"] == "A_dipotong", "B_tanpa_penanda": key["group"] == "B_tanpa_penanda"}
    groups = {g: m.reindex(a.index) for g, m in groups.items()}
    report: dict = {"n_items": int(len(key)), "raters": {}, "consensus": {}}

    print(f"Item: {len(key)} (A_dipotong {int(groups['A_dipotong'].sum())}, "
          f"B_tanpa_penanda {int(groups['B_tanpa_penanda'].sum())})\n")
    for name, s in raters.items():
        print(f"== {name}  (distribusi label: {s.value_counts().to_dict()})")
        report["raters"][name] = {}
        for g, m in groups.items():
            longgar, ketat = rate(s, m, strict=False), rate(s, m, strict=True)
            report["raters"][name][g] = {"longgar_K": longgar, "ketat_K_atau_R": ketat}
            print(f"   {g:<16} K saja: {longgar['pct']:>5}% [{longgar['ci95'][0]}, {longgar['ci95'][1]}]"
                  f"   K+R: {ketat['pct']:>5}% [{ketat['ci95'][0]}, {ketat['ci95'][1]}]  (n={longgar['n']})")

    if len(raters) == 2:
        s1, s2 = raters["penilai_A"], raters["penilai_B"]
        bin1, bin2 = (s1 == "K").astype(int), (s2 == "K").astype(int)
        kappa_bin = cohen_kappa_score(bin1, bin2)
        kappa_3 = cohen_kappa_score(s1, s2)
        agree = float((s1 == s2).mean())
        print(f"\nKesepakatan: {100 * agree:.0f}% | kappa (K vs bukan-K) = {kappa_bin:.2f} | kappa (B/R/K) = {kappa_3:.2f}")
        print("   (acuan umum: <0,40 lemah, 0,41-0,60 sedang, 0,61-0,80 substansial, >0,80 hampir sempurna)")
        report["kappa_binary_K_vs_not"] = round(float(kappa_bin), 3)
        report["kappa_3class"] = round(float(kappa_3), 3)
        report["raw_agreement"] = round(agree, 3)

        strict = ((s1 == "K") | (s2 == "K"))
        lenient = ((s1 == "K") & (s2 == "K"))
        print("\nKonsensus (rentang):")
        for g, m in groups.items():
            n = int(m.sum())
            ks, kl = int(strict[m].sum()), int(lenient[m].sum())
            los, his = wilson(ks, n)
            lol, hil = wilson(kl, n)
            report["consensus"][g] = {"n": n, "ketat_salah_satu_K": {"k": ks, "pct": round(100 * ks / n, 1),
                                      "ci95": [round(100 * los, 1), round(100 * his, 1)]},
                                      "longgar_keduanya_K": {"k": kl, "pct": round(100 * kl / n, 1),
                                      "ci95": [round(100 * lol, 1), round(100 * hil, 1)]}}
            print(f"   {g:<16} keduanya K: {100 * kl / n:>5.1f}% [{100 * lol:.1f}, {100 * hil:.1f}]"
                  f"   salah satu K: {100 * ks / n:>5.1f}% [{100 * los:.1f}, {100 * his:.1f}]")

        diff = pd.DataFrame({"item": s1.index, "penilai_A": s1.values, "penilai_B": s2.values,
                              "group": key.loc[s1.index, "group"].values})
        diff = diff[diff["penilai_A"] != diff["penilai_B"]]
        out_diff = Path(args.out) / "audit_perbedaan.csv"
        diff.to_csv(out_diff, index=False, encoding="utf-8-sig")
        print(f"\n{len(diff)} item berbeda penilaian -> {out_diff} (diskusikan dan putuskan label akhir).")

    Path(args.out, "audit_hasil.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRingkasan disimpan: {Path(args.out) / 'audit_hasil.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
