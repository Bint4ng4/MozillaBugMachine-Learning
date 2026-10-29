"""
task_asal_token.py — Dari bagian teks mana token Gambar 2 (30 token teratas LogReg) berasal? (26 Okt 2026)

Untuk tiap token di results_bobot_logreg/top30_global.csv: pada subset berpasangan (train+test), berapa dokumen yang
memuat token itu (a) di laporan awal (`init`), (b) hanya di teks susulan (ada di `Description` tetapi tidak di `init`).
Juga: persentase dokumen berkelas pemilik token, pada kelompok (a) dan (b).
Tokenisasi memakai analyzer TF-IDF yang sama dengan model (unigram + bigram, huruf kecil).

Keluaran: results_bobot_logreg/asal_token_top30.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment as ex  # noqa: E402


def main() -> int:
    out = HERE / "results_bobot_logreg"
    top = pd.read_csv(out / "top30_global.csv")
    tr, te, _ = ex.build_initial_conditions(HERE / "archive", HERE / "out_laporan_awal" / "pasangan_laporan_awal.csv")
    sub = pd.concat([tr, te], ignore_index=True)
    an = ex.make_vectorizer(20000).build_analyzer()
    feats_ref = [set(an(t)) for t in sub["Description"]]
    feats_init = [set(an(t)) for t in sub["init"]]
    sev = sub["Severity"].to_numpy()

    rows = []
    for _, r in top.iterrows():
        tok, cls = r["token"], r["kelas_pemilik"]
        in_init = [tok in f for f in feats_init]
        in_ref = [tok in f for f in feats_ref]
        only_tail = [a and not b for a, b in zip(in_ref, in_init)]
        n_ref, n_init, n_tail = sum(in_ref), sum(in_init), sum(only_tail)
        def pct(mask):
            n = sum(mask)
            return round(100 * sum(1 for m, s in zip(mask, sev) if m and s == cls) / n, 1) if n else None
        rows.append({"rank": r["rank"], "token": tok, "kelas_pemilik": cls, "koefisien": r["koefisien"],
                     "n_dok_penuh": n_ref, "n_di_laporan_awal": n_init, "n_hanya_susulan": n_tail,
                     "persen_di_laporan_awal": round(100 * n_init / n_ref, 1) if n_ref else None,
                     "persen_kelas_pemilik_di_laporan_awal": pct(in_init),
                     "persen_kelas_pemilik_hanya_susulan": pct(only_tail)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "asal_token_top30.csv", index=False)
    print(f"subset {len(sub)} dokumen")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
