"""
kaggle_laporan_awal.py — Memasangkan setiap dokumen archive/sev.csv dengan TEKS LAPORAN AWAL-nya dari dataset Kaggle
kedua ("dataset dari kaggle/bugs*.csv"), lalu melaporkan cakupan dan bias pemasangan (K-1, 11 Okt 2026).

Latar: `Description` di archive/sev.csv = laporan awal + SELURUH komentar susulan, tanpa tanda baca. bugs*.csv memuat
`Description` versi Bugzilla (comment 0 / deskripsi pertama) bertanda baca untuk bug yang sama. Cek awal: teks bugs =
AWALAN teks archive pada 99,4% pasangan dan Severity sama 99,7%. Jadi batas laporan awal -> komentar susulan bisa
diketahui per dokumen, sesuatu yang tidak bisa diperoleh dari archive saja (MASTER §8d).

Cara memasangkan:
  - Kedua teks dinormalisasi: huruf kecil, setiap karakter selain [a-z0-9] menjadi spasi, spasi dirapatkan.
  - Pasangan sah bila teks bugs ternormalisasi = AWALAN teks archive ternormalisasi pada batas kata, dan teks bugs
    punya >= MIN_TOKENS token (teks yang lebih pendek terlalu umum untuk dipasangkan dengan aman).
  - Bila beberapa teks bugs cocok, dipakai yang TERPANJANG (paling spesifik).
  - Batas laporan awal dinyatakan sebagai JUMLAH TOKEN SPASI teks archive (`n_tok_awal`), sehingga tahap INITIAL di
    cleaning.py cukup mengambil n token pertama teks tahap A. Teks hasilnya memakai tokenisasi archive yang sama
    persis (tanpa tanda baca), jadi perbandingan dengan kondisi lain adil.

Tingkat pasangan (`tingkat`):
  utama     Severity sama, satu baris bugs hanya dipakai satu dokumen archive, dan laporan awal tidak memuat penanda
            POST_REPORT (bug era Netscape kadang menyimpan riwayat komentar di deskripsi pertama).
  cadangan  cocok sebagai awalan, tetapi gagal salah satu syarat di atas.

Pakai:  python kaggle_laporan_awal.py [--kaggle "dataset dari kaggle"] [--archive archive] [--out out_laporan_awal]
Keluaran: <out>/pasangan_laporan_awal.csv, <out>/laporan_pasangan.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cleaning as c  # noqa: E402

BUGS_FILES = ["bugs.csv", "bugs1.csv", "bugs2.csv", "bugs3.csv"]
MIN_TOKENS = 5
KEY_TOKENS = 5          # kunci indeks = 5 token pertama ternormalisasi
_ALNUM = re.compile(r"[^a-z0-9]+")


def norm(t: str) -> str:
    """Normalisasi pembanding: huruf kecil, selain [a-z0-9] jadi spasi, spasi dirapatkan."""
    return " ".join(_ALNUM.sub(" ", str(t).lower()).split())


def is_word_prefix(prefix: str, text: str) -> bool:
    """True bila `prefix` (ternormalisasi) adalah awalan `text` pada batas kata."""
    return text == prefix or (text.startswith(prefix) and text[len(prefix)] == " ")


def n_ws_tokens_for_prefix(raw: str, prefix_norm: str) -> int:
    """Jumlah token SPASI awal `raw` yang memuat awalan ternormalisasi `prefix_norm`.

    Menelusuri `raw` karakter demi karakter dengan aturan yang sama dengan norm(); begitu karakter ke-len(prefix_norm)
    dari aliran ternormalisasi tercapai, posisi itu diperluas sampai akhir token spasi yang sedang dibaca, lalu
    token spasi sampai titik itu dihitung. Mengembalikan -1 bila awalan tidak tercapai.
    """
    target = len(prefix_norm)
    if target == 0:
        return 0
    emitted, pending_space = 0, False
    for i, ch in enumerate(raw):
        for c2 in ch.lower():
            if ("a" <= c2 <= "z") or ("0" <= c2 <= "9"):
                if pending_space and emitted > 0:
                    emitted += 1                      # spasi pemisah di aliran ternormalisasi
                pending_space = False
                emitted += 1
                if emitted >= target:
                    end = i + 1
                    while end < len(raw) and not raw[end].isspace():
                        end += 1
                    return len(raw[:end].split())
            else:
                pending_space = True
    return -1


def load_bugs(folder: Path) -> pd.DataFrame:
    parts = []
    for f in BUGS_FILES:
        d = pd.read_csv(folder / f)
        parts.append(pd.DataFrame({"bugs_file": f, "bugs_row": np.arange(len(d)),
                                   "text": d["Description"].fillna("").astype(str),
                                   "sev_bugs": d["Severity"].astype(str), "priority_bugs": d["Priority"].astype(str),
                                   "fixing_time_bugs": d["Fixing_time"]}))
    return pd.concat(parts, ignore_index=True)


def pair(archive: pd.DataFrame, bugs: pd.DataFrame) -> pd.DataFrame:
    """Pasangkan tiap baris archive dengan baris bugs terpanjang yang menjadi awalannya."""
    a_norm = archive["Description"].fillna("").astype(str).map(norm)
    b_norm = bugs["text"].map(norm)
    b_ntok = b_norm.str.split().str.len()
    index: dict[str, list[int]] = {}
    for j, (t, n) in enumerate(zip(b_norm, b_ntok)):
        if n >= MIN_TOKENS:
            index.setdefault(" ".join(t.split()[:KEY_TOKENS]), []).append(j)

    rows = []
    for i, an in enumerate(a_norm):
        cands = [j for j in index.get(" ".join(an.split()[:KEY_TOKENS]), []) if is_word_prefix(b_norm[j], an)]
        if not cands:
            continue
        best_len = max(b_ntok[j] for j in cands)
        best = [j for j in cands if b_ntok[j] == best_len]
        j = best[0]
        rows.append({"row_archive": i, "bugs_idx": j, "n_kandidat": len(cands),
                     "seri_teks_beda": len({b_norm[x] for x in best}) > 1})
    p = pd.DataFrame(rows)
    if p.empty:
        return p
    p["bugs_dipakai"] = p.groupby("bugs_idx")["row_archive"].transform("count")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle", default="dataset dari kaggle")
    ap.add_argument("--archive", default="archive")
    ap.add_argument("--out", default="out_laporan_awal")
    args = ap.parse_args()

    archive = pd.read_csv(Path(args.archive) / "sev.csv")
    junk = [col for col in archive.columns if col.startswith("Unnamed") or not str(col).strip()]
    archive["orig_idx"] = archive[junk[-1]]           # sama dengan _orig_idx di cleaning.stage_a_structure
    raw = archive["Description"].fillna("").astype(str)
    bugs = load_bugs(Path(args.kaggle))
    print(f"archive/sev.csv: {len(archive)} baris | bugs*.csv: {len(bugs)} baris")

    p = pair(archive, bugs)
    b = bugs.loc[p["bugs_idx"]].reset_index(drop=True)
    p = pd.concat([p.reset_index(drop=True), b], axis=1)
    p["orig_idx"] = archive["orig_idx"].values[p["row_archive"]]
    p["sev_archive"] = archive["Severity"].astype(str).values[p["row_archive"]]
    p["sev_sama"] = p["sev_archive"] == p["sev_bugs"]
    p["n_tok_archive"] = raw.values[p["row_archive"]]
    p["n_tok_archive"] = p["n_tok_archive"].str.split().str.len()
    b_norm = p["text"].map(norm)
    p["n_tok_awal"] = [n_ws_tokens_for_prefix(r, bn) for r, bn in zip(raw.values[p["row_archive"]], b_norm)]
    p["awal_ada_penanda"] = p["text"].map(lambda t: bool(c.POST_REPORT.search(c._norm(t))))
    p["awal_era_header"] = p["text"].str.contains(r"(?:Created|Updated) by .{0,120}? on ", regex=True)
    # Pemeriksaan balik: n_tok_awal token pertama archive harus kembali ke awalan teks bugs.
    back = [norm(" ".join(r.split()[:n])) for r, n in zip(raw.values[p["row_archive"]], p["n_tok_awal"])]
    p["cek_balik"] = [is_word_prefix(bn, x) or x == bn for bn, x in zip(b_norm, back)]
    p["tingkat"] = np.where(p["sev_sama"] & (p["bugs_dipakai"] == 1) & ~p["awal_ada_penanda"]
                            & ~p["seri_teks_beda"] & p["cek_balik"], "utama", "cadangan")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    cols = ["orig_idx", "row_archive", "bugs_file", "bugs_row", "tingkat", "n_tok_awal", "n_tok_archive",
            "sev_archive", "sev_bugs", "sev_sama", "bugs_dipakai", "n_kandidat", "seri_teks_beda",
            "awal_ada_penanda", "awal_era_header", "cek_balik", "priority_bugs", "fixing_time_bugs"]
    p[cols].to_csv(out / "pasangan_laporan_awal.csv", index=False)

    # ---------------------------------------------------------------- laporan cakupan dan bias
    n_arc = len(archive)
    paired = np.zeros(n_arc, dtype=bool)
    paired[p["row_archive"].values] = True
    utama = np.zeros(n_arc, dtype=bool)
    utama[p.loc[p["tingkat"] == "utama", "row_archive"].values] = True
    rep: dict = {"n_archive": n_arc, "n_bugs": len(bugs), "min_tokens": MIN_TOKENS,
                 "berpasangan": int(paired.sum()), "utama": int(utama.sum()),
                 "bugs_terlalu_pendek": int((bugs["text"].map(norm).str.split().str.len() < MIN_TOKENS).sum())}
    pu = p[p["tingkat"] == "utama"]
    extra = pu["n_tok_archive"] - pu["n_tok_awal"]
    rep["tingkat_gagal"] = {"sev_beda": int((~p["sev_sama"]).sum()), "bugs_dipakai_ganda": int((p["bugs_dipakai"] > 1).sum()),
                            "awal_ada_penanda": int(p["awal_ada_penanda"].sum()),
                            "seri_teks_beda": int(p["seri_teks_beda"].sum()), "cek_balik_gagal": int((~p["cek_balik"]).sum())}
    rep["utama_porsi_token_awal_median"] = float((pu["n_tok_awal"] / pu["n_tok_archive"]).median())
    rep["utama_dok_tanpa_tambahan"] = float((extra <= 0).mean())

    def dist(mask):
        return archive.loc[mask, "Severity"].value_counts(normalize=True).mul(100).round(1).to_dict()
    tab = pd.crosstab(utama, archive["Severity"])
    chi2, pval, dof, _ = chi2_contingency(tab)
    v = float(np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1))))
    has_marker = raw.map(lambda t: bool(c.POST_REPORT.search(c._norm(t))))
    ntok = raw.str.split().str.len()
    is_test = archive["orig_idx"] >= c.TEST_MIN
    rep["bias"] = {
        "severity_pct_utama": dist(utama), "severity_pct_bukan_utama": dist(~utama),
        "chi2": float(chi2), "df": int(dof), "p": float(pval), "cramers_v": v,
        "porsi_test_utama": float(is_test[utama].mean()), "porsi_test_bukan_utama": float(is_test[~utama].mean()),
        "token_median_utama": float(ntok[utama].median()), "token_median_bukan_utama": float(ntok[~utama].median()),
        "penanda_utama": float(has_marker[utama].mean()), "penanda_bukan_utama": float(has_marker[~utama].mean()),
    }
    (out / "laporan_pasangan.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"berpasangan: {rep['berpasangan']} ({rep['berpasangan'] / n_arc:.1%}) | tingkat utama: {rep['utama']} "
          f"({rep['utama'] / n_arc:.1%}) | teks bugs < {MIN_TOKENS} token (tidak dipasangkan): {rep['bugs_terlalu_pendek']}")
    print(f"  alasan cadangan: {rep['tingkat_gagal']}")
    print(f"  utama: porsi token laporan awal per dokumen (median) {rep['utama_porsi_token_awal_median']:.2f}; "
          f"dokumen tanpa teks tambahan {rep['utama_dok_tanpa_tambahan']:.1%}")
    bi = rep["bias"]
    print("\nBias utama vs bukan-utama:")
    print(f"  Severity % utama      {bi['severity_pct_utama']}")
    print(f"  Severity % bukan      {bi['severity_pct_bukan_utama']}")
    print(f"  chi2 {bi['chi2']:.1f} df {bi['df']} p {bi['p']:.2g} | Cramér's V {bi['cramers_v']:.3f}")
    print(f"  porsi test {bi['porsi_test_utama']:.1%} vs {bi['porsi_test_bukan_utama']:.1%} | token median "
          f"{bi['token_median_utama']:.0f} vs {bi['token_median_bukan_utama']:.0f} | berpenanda "
          f"{bi['penanda_utama']:.1%} vs {bi['penanda_bukan_utama']:.1%}")
    print(f"\nDisimpan: {out / 'pasangan_laporan_awal.csv'}, {out / 'laporan_pasangan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
