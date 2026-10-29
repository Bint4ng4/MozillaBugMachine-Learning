"""
verify_data.py — Verifikasi dan audit dataset sev.csv

Menjalankan ulang seluruh pengukuran data yang dicatat di MASTER note (Obsidian) §4.
Kalau angka yang keluar TIDAK cocok dengan dokumen itu, berhenti dan cari tahu kenapa
sebelum melanjutkan ke eksperimen apa pun.

Cara pakai:
    python verify_data.py
    python verify_data.py --archive archive --out environment_report.json

Butuh: pandas, numpy, scipy
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Angka acuan dari MASTER note §4 (dulu RENCANA_PENELITIAN.md, kini di arsip/) — dipakai untuk pengecekan otomatis
EXPECTED = {
    "n_rows": 32428,
    "severity_counts": {
        "normal": 22828, "major": 3760, "critical": 2726,
        "minor": 1462, "blocker": 1082, "trivial": 570,
    },
    "label_1": 7568,
    "n_train": 28000,
    "n_test": 4427,
    "vocab_size": 46260,
    "embedding_shape": (46260, 100),
    # 85,1% = definisi MASTER SS4.6 (union termasuk `verified`, TANPA beda huruf besar/kecil). 69,6% adalah
    # definisi lama tanpa `verified`; versi 11 Sep sempat peka huruf dan memberi 80,9% (Catatan Koreksi, 10 Okt).
    "contamination_pct": 85.1,
    "oov_pct": 2.72,
}

SEVERE = {"major", "critical", "blocker"}

# Penanda pasca-pelaporan: teks yang MUSTAHIL ada saat laporan bug baru dikirim
CONTAMINATION_MARKERS = {
    "verified": r"\bverified\b|\bVerified\b",
    "marked as a duplicate": r"marked as a duplicate",
    "checked in / fixed in": r"checked in|fixed in",
    "WORKSFORME": r"WORKSFORME|worksforme|works for me",
    "Marking Verified": r"Marking Verified",
    "RESOLVED": r"\bRESOLVED\b|\bresolved\b",
    "In reply to comment": r"In reply to comment",
    "Updated by": r"Updated by",
}

ANY_MARKER = re.compile(
    "|".join(
        [
            r"marked as a duplicate", r"Marking Verified",
            r"WORKSFORME|worksforme|works for me",
            r"checked in|fixed in", r"In reply to comment",
            r"\bRESOLVED\b", r"Updated by",
            r"\bverified\b|\bVerified\b"
        ]
    ),
    re.IGNORECASE,   # "marking verified", "Works For Me", dst. juga penanda; sama dengan cleaning.POST_REPORT
)


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def check(label: str, actual, expected, tol: float = 0.0) -> bool:
    """Bandingkan nilai terukur dengan nilai acuan; cetak PASS/GAGAL."""
    if isinstance(expected, float) and tol:
        ok = abs(float(actual) - expected) <= tol
    else:
        ok = actual == expected
    mark = "PASS" if ok else "GAGAL"
    print(f"  [{mark}] {label}: {actual}  (acuan: {expected})")
    return ok


def cramers_v(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
    """Kembalikan (Cramer's V, chi2, df). V=0 acak total, V=1 deterministik."""
    ct = pd.crosstab(a, b)
    chi2, _, dof, _ = chi2_contingency(ct, correction=False)
    n = ct.values.sum()
    k = min(ct.shape)
    v = np.sqrt(chi2 / (n * (k - 1))) if k > 1 else 0.0
    return float(v), float(chi2), int(dof)


def macro_f1_majority(counts: dict[str, int]) -> float:
    """Macro-F1 kalau model selalu menebak kelas mayoritas."""
    total = sum(counts.values())
    top = max(counts, key=counts.get)
    precision = counts[top] / total
    recall = 1.0
    f1_top = 2 * precision * recall / (precision + recall)
    return f1_top / len(counts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="archive", help="folder berisi sev.csv dkk")
    ap.add_argument("--out", default="environment_report.json")
    args = ap.parse_args()

    arch = Path(args.archive)
    report: dict = {}
    failures: list[str] = []

    # ---------------------------------------------------------------- 0. env
    rule("0. LINGKUNGAN")
    print(f"  python  : {sys.version.split()[0]}")
    print(f"  pandas  : {pd.__version__}")
    print(f"  numpy   : {np.__version__}")
    report["env"] = {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }

    # ------------------------------------------------------------- 1. muat
    rule("1. MUAT sev.csv")
    sev_path = arch / "sev.csv"
    if not sev_path.exists():
        print(f"  FATAL: {sev_path} tidak ditemukan.")
        return 1

    df = pd.read_csv(sev_path)
    print(f"  kolom mentah: {list(df.columns)}")

    # Buang dua kolom index sampah
    junk = [c for c in df.columns if c.startswith("Unnamed") or not str(c).strip()]
    stale_index = df[junk[-1]] if junk else None
    df = df.drop(columns=junk)
    df["Description"] = df["Description"].astype(str)

    if not check("jumlah baris", len(df), EXPECTED["n_rows"]):
        failures.append("jumlah baris")
    print(f"  kolom setelah dibersihkan: {list(df.columns)}")
    report["n_rows"] = len(df)

    if stale_index is not None:
        gap = int(stale_index.max()) + 1 - len(df)
        print(f"  indeks basi maks = {int(stale_index.max())} -> {gap} baris hilang di hulu")
        report["rows_dropped_upstream"] = gap

    # ------------------------------------------------------- 2. distribusi
    rule("2. DISTRIBUSI KELAS")
    counts = df["Severity"].value_counts().to_dict()
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<10} {v:>6}  ({100 * v / len(df):5.1f}%)")
    if not check("distribusi Severity", counts, EXPECTED["severity_counts"]):
        failures.append("distribusi Severity")
    report["severity_counts"] = counts

    acc_base = max(counts.values()) / len(df)
    f1_base = macro_f1_majority(counts)
    print(f"\n  BASELINE (selalu tebak '{max(counts, key=counts.get)}'):")
    print(f"    accuracy  = {acc_base:.4f}   <- gratis, tidak membuktikan apa pun")
    print(f"    macro-F1  = {f1_base:.4f}   <- INI yang harus dilampaui")
    report["baseline"] = {"accuracy": acc_base, "macro_f1": f1_base}

    # --------------------------------------------------- 3. kebocoran Label
    rule("3. KEBOCORAN TARGET: Label vs Severity")
    derived = df["Severity"].isin(SEVERE).astype(int)
    violations = int((derived != df["Label"].astype(int)).sum())
    v, chi2, dof = cramers_v(df["Severity"], df["Label"])
    print(f"  Label==1 : {int(df['Label'].sum())}  (acuan {EXPECTED['label_1']})")
    print(f"  pelanggaran aturan {{major,critical,blocker}} -> 1 : {violations}")
    print(f"  Cramer's V(Severity, Label) = {v:.4f}   chi2={chi2:.1f}  df={dof}")
    if violations == 0:
        print("  => DETERMINISTIK. Jangan pernah pakai keduanya bersamaan.")
    else:
        failures.append("relasi Severity->Label tidak deterministik")
    report["label_leakage"] = {"violations": violations, "cramers_v": v}

    # ------------------------------------------------------- 4. statistik teks
    rule("4. STATISTIK TEKS")
    char_len = df["Description"].str.len()
    tok_len = df["Description"].str.split().str.len()
    for name, s in (("karakter", char_len), ("token", tok_len)):
        print(
            f"  {name:<9} mean={s.mean():8.0f}  median={s.median():7.0f}  "
            f"p90={s.quantile(.9):8.0f}  p99={s.quantile(.99):8.0f}  max={s.max():8.0f}"
        )
    n_punct = int(df["Description"].str.contains(r"[.,;:!?]", regex=True).sum())
    print(f"\n  dokumen dengan tanda baca . , ; : ! ? : {n_punct}")
    if n_punct == 0:
        print("  => Tanda baca sudah dihapus di hulu, IREVERSIBEL. Catat sebagai limitasi.")
    print(f"  dokumen kosong/whitespace : {int(df['Description'].str.strip().eq('').sum())}")
    print(f"  dokumen < 5 token         : {int(tok_len.lt(5).sum())}")
    print(f"  duplikat persis           : {int(df['Description'].duplicated().sum())}")
    norm = df["Description"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    print(f"  duplikat ternormalisasi   : {int(norm.duplicated().sum())}")
    report["text_stats"] = {
        "tokens_mean": float(tok_len.mean()),
        "tokens_median": float(tok_len.median()),
        "tokens_max": int(tok_len.max()),
        "docs_with_punctuation": n_punct,
    }

    # ------------------------------------------ 5. kontaminasi pasca-pelaporan
    rule("5. KONTAMINASI PASCA-PELAPORAN (inti RQ1)")
    print(f"  {'penanda':<26}{'dokumen':>9}{'%':>8}{'V':>9}{'dP(severe)':>12}")
    print(f"  {'-' * 64}")
    marker_stats = {}
    for name, pat in CONTAMINATION_MARKERS.items():
        has = df["Description"].str.contains(pat, regex=True, case=False)
        n = int(has.sum())
        v_m, chi_m, _ = cramers_v(has, df["Severity"])
        p_yes = df.loc[has, "Severity"].isin(SEVERE).mean() if n else float("nan")
        p_no = df.loc[~has, "Severity"].isin(SEVERE).mean() if n < len(df) else float("nan")
        print(f"  {name:<26}{n:>9}{100 * n / len(df):>7.1f}%{v_m:>9.4f}{100 * (p_yes - p_no):>11.1f}pp")
        marker_stats[name] = {"n": n, "pct": 100 * n / len(df), "cramers_v": v_m}

    has_any = df["Description"].str.contains(ANY_MARKER)
    n_any = int(has_any.sum())
    pct_any = 100 * n_any / len(df)
    v_any, chi_any, dof_any = cramers_v(has_any, df["Severity"])
    p_yes = df.loc[has_any, "Severity"].isin(SEVERE).mean()
    p_no = df.loc[~has_any, "Severity"].isin(SEVERE).mean()
    print(f"  {'-' * 64}")
    print(f"  {'>=1 PENANDA':<26}{n_any:>9}{pct_any:>7.1f}%{v_any:>9.4f}{100 * (p_yes - p_no):>11.1f}pp")
    print(f"\n  chi2={chi_any:.1f}  df={dof_any}  (harapan jika independen: chi2 ~ df)")
    print(f"  P(severe | ada penanda) = {p_yes:.1%}   P(severe | tanpa) = {p_no:.1%}")
    print("\n  Tafsir: asosiasi NYATA dan signifikan, tapi effect size LEMAH (V < 0.1).")
    print("  V hanya mengukur keterkaitan KEHADIRAN penanda dengan severity, bukan kebocoran ISI teks: pada")
    print("  kontrol positif (experiment.py --suite rq2_posctrl) token bocor yang menaikkan macro-F1 sampai")
    print("  10 poin hanya memberi V kehadiran ~0,01. Jangan baca V kecil sebagai 'kontaminasi tak informatif'.")
    if not check("kontaminasi %", round(pct_any, 1), EXPECTED["contamination_pct"], tol=0.5):
        failures.append("persentase kontaminasi")
    report["contamination"] = {
        "pct_any": pct_any, "cramers_v": v_any, "chi2": chi_any, "df": dof_any,
        "markers": marker_stats,
    }

    # ------------------------------------------------------------ 6. split
    rule("6. SPLIT BAWAAN — apakah kronologis?")
    tr_p, te_p = arch / "sev_train.csv", arch / "sev_test.csv"
    if tr_p.exists() and te_p.exists():
        tr, te = pd.read_csv(tr_p), pd.read_csv(te_p)
        icol = [c for c in tr.columns if c.startswith("Unnamed")][-1]
        tr_i, te_i = tr[icol].astype(int), te[icol].astype(int)
        overlap = len(set(tr_i) & set(te_i))
        print(f"  train: {len(tr)} baris, indeks {tr_i.min()}..{tr_i.max()}")
        print(f"  test : {len(te)} baris, indeks {te_i.min()}..{te_i.max()}")
        print(f"  overlap indeks: {overlap}")
        contiguous = tr_i.max() < te_i.min()
        print(f"  kontigu (train seluruhnya sebelum test): {contiguous}")
        if contiguous and overlap == 0:
            print("  => KUASI-KRONOLOGIS. Bagus: meniru deployment.")
        n_trivial = int((te["Severity"] == "trivial").sum())
        print(f"\n  sampel 'trivial' di test: {n_trivial}")
        if n_trivial < 150:
            print(f"  PERINGATAN: hanya {n_trivial} sampel. CI recall akan sangat lebar.")
            print("  Pertimbangkan split ulang kronologis dengan test >= 20%.")
        report["split"] = {
            "n_train": len(tr), "n_test": len(te), "overlap": overlap,
            "contiguous": bool(contiguous), "n_trivial_test": n_trivial,
        }
    else:
        print("  (file split tidak ada, dilewati)")

    # ------------------------------------------ 7. fix.csv = teks yang sama?
    rule("7. fix.csv — dataset terpisah atau teks yang sama?")
    fix_p = arch / "fix.csv"
    if fix_p.exists():
        fx = pd.read_csv(fix_p)
        fx["Description"] = fx["Description"].astype(str)
        if len(fx) == len(df):
            identical = int((fx["Description"].values == df["Description"].values).sum())
            print(f"  Description identik baris-per-baris: {identical}/{len(df)} "
                  f"({100 * identical / len(df):.1f}%)")
            if identical == len(df):
                print("  => BUKAN dataset terpisah. Dokumen yang SAMA, target berbeda.")
                print(f"     sev.csv -> Severity (6 kelas ordinal)")
                print(f"     fix.csv -> Fixing_time (kontinu, "
                      f"{fx['Fixing_time'].min()}..{fx['Fixing_time'].max()})")
                print("     Implikasi: fix.csv tidak menambah teks baru, tapi menyediakan")
                print("     target kedua di atas teks identik — berguna untuk validasi protokol.")
            report["fix_identical"] = identical
        else:
            print(f"  jumlah baris berbeda: fix={len(fx)} vs sev={len(df)}")
    else:
        print("  (fix.csv tidak ada, dilewati)")

    # --------------------------------------- 8. embedding + OOV + kebocoran
    rule("8. EMBEDDING PRALATIH — OOV dan kebocoran korpus")
    voc_p, emb_p = arch / "vocab.lst", arch / "embedding.npy"
    if voc_p.exists() and emb_p.exists():
        vocab = [w.strip() for w in voc_p.read_text(encoding="utf-8", errors="replace").splitlines()]
        emb = np.load(emb_p)
        print(f"  vocab.lst    : {len(vocab)} token (case-sensitive)")
        print(f"  embedding.npy: {emb.shape}  dtype={emb.dtype}")
        check("ukuran vocab", len(vocab), EXPECTED["vocab_size"])
        check("bentuk embedding", tuple(emb.shape), EXPECTED["embedding_shape"])

        vset = set(vocab)
        rng = np.random.default_rng(42)
        sample = df["Description"].sample(2000, random_state=42)
        tot = oov = 0
        tot_lc = oov_lc = 0
        for txt in sample:
            for w in txt.split():
                tot += 1
                if w not in vset:
                    oov += 1
                tot_lc += 1
                if w.lower() not in vset:
                    oov_lc += 1
        print(f"\n  OOV casing asli   : {100 * oov / tot:.2f}%   (acuan {EXPECTED['oov_pct']}%)")
        print(f"  OOV di-lowercase  : {100 * oov_lc / tot_lc:.2f}%")
        if oov_lc > oov:
            print("  => JANGAN .lower() kalau memakai embedding ini. Vocab case-sensitive.")
        report["embedding"] = {
            "vocab": len(vocab), "shape": list(emb.shape),
            "oov_native_pct": 100 * oov / tot, "oov_lower_pct": 100 * oov_lc / tot_lc,
        }

        # --- probe kebocoran: apakah dokumen TEST ada di korpus pelatihan embedding?
        corpus_p = arch / "corpus (fixsev).txt"
        if corpus_p.exists() and te_p.exists():
            print("\n  --- Probe kebocoran korpus (RQ4) ---")
            corpus = corpus_p.read_text(encoding="utf-8", errors="replace")
            print(f"  corpus: {len(corpus):,} karakter")

            def probe(row) -> bool | None:
                lines = [l.strip() for l in str(row["Description"]).split("\n")
                         if len(l.strip()) > 50]
                if not lines:
                    return None
                p = max(lines, key=len)[:80]
                return p in corpus

            hits = [probe(te.iloc[i]) for i in (0, 700, 1800, 3200, 4400) if i < len(te)]
            hits = [h for h in hits if h is not None]
            neg = "zzqxwv nonexistent probe string 12345" in corpus
            print(f"  dokumen TEST ditemukan di korpus : {sum(hits)}/{len(hits)}")
            print(f"  kontrol negatif (harus False)    : {neg}")
            if sum(hits) and not neg:
                print("\n  => KEBOCORAN TRANSDUKTIF TERKONFIRMASI.")
                print("     Embedding dilatih pada korpus yang MENCAKUP test set.")
                print("     Ia melihat TEKS test, bukan LABEL test — jadi ini bukan")
                print("     kebocoran label, tapi tetap ancaman validitas untuk RQ4:")
                print("     perbandingan TF-IDF (fit train saja) vs embedding menjadi")
                print("     TIDAK setara. OOV 2.72% juga bias optimistis.")
            report["corpus_leakage"] = {
                "test_docs_found": sum(hits), "test_docs_probed": len(hits),
                "negative_control": neg,
            }
    else:
        print("  (vocab.lst / embedding.npy tidak ada, dilewati)")

    # ----------------------------------------------------------- ringkasan
    rule("RINGKASAN")
    if failures:
        print("  ADA PEMERIKSAAN YANG GAGAL:")
        for f in failures:
            print(f"    - {f}")
        print("\n  Jangan lanjut ke eksperimen sebelum ini dijelaskan.")
    else:
        print("  Semua pemeriksaan acuan LOLOS. Data sesuai acuan MASTER note §4.")

    print("\n  Yang harus diingat sebelum melatih model:")
    print(f"    1. Baseline macro-F1 = {f1_base:.4f} (akurasi {acc_base:.1%} itu gratis)")
    print("    2. Jangan pakai Severity dan Label bersamaan")
    print("    3. Jangan .lower() kalau memakai embedding.npy")
    print("    4. Dedup SEBELUM split, dekontaminasi sebagai varian eksperimen")
    print("    5. Embedding sudah melihat teks test -> laporkan sebagai ancaman validitas")

    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  Laporan disimpan: {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
