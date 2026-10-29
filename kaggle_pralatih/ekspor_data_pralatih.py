"""
ekspor_data_pralatih.py — Siapkan data RQ2 untuk model pralatih di Kaggle (keputusan user 14 Okt 2026).

Kondisi dan dokumen SAMA PERSIS dengan `experiment.py --suite rq2_initial` (MASTER §8f): subset berpasangan utama,
train `_orig_idx` <= 28.505 (26.539 dokumen) / test >= 28.507 (4.209 dokumen), urutan baris test dipertahankan sehingga
bootstrap seed 42 di analisis memakai indeks yang sama dengan model linear.

Train dibagi kronologis untuk pemilihan epoch: train_fit (`_orig_idx` < 25.000) dan val (25.000–28.505), sama dengan
`task_kurva_belajar.py`. Test hanya dipakai sekali, di akhir.

Keluaran: kaggle_pralatih/data/rq2_pralatih.csv (kolom: _orig_idx, split, Severity, text_ref, text_init) + ringkasan.
Data ini turunan dataset berlisensi "Unknown" (MASTER §4.8): unggah ke Kaggle sebagai dataset PRIVAT, jangan publik.

Pakai:  python kaggle_pralatih/ekspor_data_pralatih.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import experiment as ex  # noqa: E402

VAL_MIN = 25000


def main() -> int:
    tr, te, info = ex.build_initial_conditions(ROOT / "archive", ROOT / "out_laporan_awal" / "pasangan_laporan_awal.csv")
    tr = tr.assign(split=["val" if i >= VAL_MIN else "train_fit" for i in tr["_orig_idx"]])
    te = te.assign(split="test")
    cols = {"_orig_idx": "_orig_idx", "split": "split", "Severity": "Severity", "Description": "text_ref",
            "init": "text_init"}
    out = pd.concat([tr, te], ignore_index=True)[list(cols)].rename(columns=cols)
    dest = Path(__file__).resolve().parent / "data"
    dest.mkdir(exist_ok=True)
    path = dest / "rq2_pralatih.csv"
    out.to_csv(path, index=False)
    summ = {"info_build_initial_conditions": info,
            "n_per_split": out["split"].value_counts().to_dict(),
            "severity_per_split": {s: g["Severity"].value_counts().to_dict() for s, g in out.groupby("split")},
            "kata_rata2": {c: round(float(out[c].str.split().str.len().mean()), 1) for c in ("text_ref", "text_init")},
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (dest / "ringkasan_ekspor.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
