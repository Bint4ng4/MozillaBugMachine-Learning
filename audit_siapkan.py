"""
audit_siapkan.py — Menyiapkan sampel audit manual dokumen terpotong (runbook langkah 3, menutup T14).

Pertanyaan audit: setelah TAILCUT, apakah teks yang tersisa masih memuat isi PASCA-PELAPORAN?
  - Kelompok A: dokumen yang punya penanda dan DIPOTONG (yang dilihat model = teks terpotong).
  - Kelompok B: dokumen TANPA penanda (dibiarkan utuh; menguji apakah regex melewatkan isi pasca-laporan).

Desain (anti-bias):
  - 50 dokumen A + 50 dokumen B, acak dengan seed tetap, DIACAK URUTANNYA dan grupnya DISEMBUNYIKAN
    dari penilai (kolom grup hanya ada di berkas kunci).
  - Dokumen dengan < 5 token pasca-pemotongan tidak diambil (tidak cukup teks untuk dinilai);
    jumlahnya dilaporkan.
  - Penilai hanya melihat teks yang dilihat model; bagian yang dibuang TIDAK ditampilkan.

Keluaran (folder audit_langkah3/):
  audit_penilai_A.csv, audit_penilai_B.csv   lembar isian (identik, untuk dua penilai independen)
  audit_KUNCI_jangan_dibuka_penilai.csv       pemetaan item -> grup/split/baris
  Jalankan audit_hitung.py setelah lembar diisi.

Pakai:  python audit_siapkan.py [--seed 20261008] [--per-group 50] [--out audit_langkah3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cleaning as c  # noqa: E402

MIN_TOKENS = 5


def load_aligned(src: Path) -> pd.DataFrame:
    """Gabungkan train+test dari baseline (A saja) dan plus_tailcut, dan pastikan barisnya sejajar."""
    parts = []
    for split in ("train", "test"):
        b = pd.read_csv(src / f"cleaned_baseline_{split}.csv")
        t = pd.read_csv(src / f"cleaned_plus_tailcut_{split}.csv")
        if len(b) != len(t) or not (b["Severity"].values == t["Severity"].values).all():
            raise SystemExit(f"FATAL: baris baseline vs plus_tailcut tidak sejajar di split {split}.")
        df = pd.DataFrame({
            "split": split,
            "row_in_split": np.arange(len(b)),
            "severity": b["Severity"].values,
            "before": b["Description"].fillna("").astype(str).values,
            "after": t["Description"].fillna("").astype(str).values,
        })
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="out_20261008")
    ap.add_argument("--out", default="audit_langkah3")
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--per-group", type=int, default=50)
    args = ap.parse_args()

    df = load_aligned(Path(args.src))
    df["has_marker"] = df["before"].str.contains(c.POST_REPORT)
    df["n_after"] = df["after"].str.split().str.len().fillna(0).astype(int)
    df["n_before"] = df["before"].str.split().str.len().fillna(0).astype(int)

    eligible = df["n_after"] >= MIN_TOKENS
    groups = {"A_dipotong": df["has_marker"] & eligible, "B_tanpa_penanda": ~df["has_marker"] & eligible}
    print(f"Total dokumen: {len(df)} | punya penanda: {int(df['has_marker'].sum())} | tanpa penanda: "
          f"{int((~df['has_marker']).sum())}")
    for g, m in groups.items():
        short = int((df["has_marker"] == g.startswith("A")).sum() - m.sum())
        print(f"  {g}: layak dinilai {int(m.sum())}; dikecualikan karena < {MIN_TOKENS} token: {short}")

    picks = []
    for gi, (g, m) in enumerate(groups.items()):
        pool = df[m]
        s = pool.sample(n=args.per_group, random_state=args.seed + gi)
        s = s.assign(group=g)
        picks.append(s)
    sample = pd.concat(picks).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    sample.insert(0, "item", np.arange(1, len(sample) + 1))

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    sheet = pd.DataFrame({
        "item": sample["item"],
        "teks": sample["after"],
        "label": "",          # isi: B (bersih) / R (ragu) / K (terkontaminasi)
        "kutipan_bukti": "",  # kutipan singkat dari teks bila label K atau R
        "catatan": "",
    })
    for who in ("A", "B"):
        sheet.to_csv(out / f"audit_penilai_{who}.csv", index=False, encoding="utf-8-sig")
    sample[["item", "group", "split", "row_in_split", "severity", "n_before", "n_after"]].to_csv(
        out / "audit_KUNCI_jangan_dibuka_penilai.csv", index=False, encoding="utf-8-sig")
    print(f"\nDisimpan di {out}/ : {len(sample)} item (seed {args.seed}).")
    print("Berikan HANYA audit_penilai_A.csv ke penilai pertama dan audit_penilai_B.csv ke penilai kedua.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
