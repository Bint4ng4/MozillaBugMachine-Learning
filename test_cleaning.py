"""
test_cleaning.py — Tes kecil untuk aturan cleaning.py yang pernah bermasalah / rawan kebocoran.

Jalankan:  python test_cleaning.py
Keluar dengan kode 0 bila semua lulus, 1 bila ada yang gagal.

Cakupan:
  - Tahap I (DATES): kata biasa tidak boleh ikut terhapus (bug 8 Okt 2026: IGNORECASE +
    pola zona waktu [A-Z]{3}T? menghapus "the", "and", "may", "march", dst).
  - Tahap G: batas panjang dihitung dari train saja (Kapoor & Narayanan 2023, L1.2).
  - Tahap E2: dedup eksplisit per split (L1.4); train tidak dipengaruhi test.
"""

from __future__ import annotations

import sys

import pandas as pd

import cleaning as c

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if cond else 'GAGAL'}] {name}" + (f"  -> {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def dates(t: str) -> str:
    return c._norm(c.DATES.sub(" ", t))


def test_dates() -> None:
    print("Tahap I (DATES)")
    out = dates("the browser crashes when the user clicks that text and not the bug PDT 1998")
    check("kata biasa tetap utuh, hanya 'PDT 1998' hilang",
          out == "the browser crashes when the user clicks that text and not the bug", out)
    check("'may' (kata kerja) dipertahankan", dates("this may crash") == "this may crash")
    check("'march' huruf kecil dipertahankan", dates("they march on") == "they march on")
    check("'May'/'March' tanpa angka dipertahankan", dates("May I ask? March forward") == "May I ask? March forward")
    # Angka tanggal ("7", "12") sengaja TIDAK dihapus di tahap I; tahap D yang membuang angka.
    check("'May 7' dan 'March 12': nama bulan hilang, angka tanggal diserahkan ke tahap D",
          dates("on May 7 and March 12 ok") == "on 7 and 12 ok", dates("on May 7 and March 12 ok"))
    stamp = "Tuesday April 7 1998 10 57 25 AM PDT"
    check("stempel waktu Bugzilla: hanya angka tanggal ('7') yang tersisa untuk tahap D",
          dates(stamp) == "7", dates(stamp))
    for tz in ("PST", "EDT", "EST"):
        check(f"zona waktu {tz} dihapus", dates(f"at 5 {tz} sharp") == "at 5 sharp")
    check("kata 'sat' dan 'sun' tidak dihapus", dates("sat and sun differ") == "sat and sun differ")


def _toy(n_train: int, n_test: int, test_len: int, train_len: int = 10) -> pd.DataFrame:
    rows = []
    for i in range(n_train):
        rows.append({"_orig_idx": i, "Description": " ".join(f"tr{i}w{j}" for j in range(train_len)),
                     "Severity": "normal"})
    for i in range(n_test):
        rows.append({"_orig_idx": c.TEST_MIN + i,
                     "Description": " ".join(f"te{i}w{j}" for j in range(test_len)),
                     "Severity": "normal"})
    return pd.DataFrame(rows)


def test_truncate_train_only() -> None:
    print("Tahap G (truncate dari train saja)")
    short = c.stage_g_truncate(_toy(200, 50, test_len=10))
    longtest = c.stage_g_truncate(_toy(200, 50, test_len=500))
    tr_short = short[short["_orig_idx"] <= c.TRAIN_MAX]["Description"].str.split().str.len().max()
    tr_long = longtest[longtest["_orig_idx"] <= c.TRAIN_MAX]["Description"].str.split().str.len().max()
    check("dokumen test yang sangat panjang tidak mengubah batas", tr_short == tr_long == 10,
          f"{tr_short} vs {tr_long}")
    te_len = longtest[longtest["_orig_idx"] >= c.TEST_MIN]["Description"].str.split().str.len().max()
    check("batas dari train diterapkan ke test (dipotong ke 10 token)", te_len == 10, str(te_len))


def test_dedup_split() -> None:
    print("Tahap E2 (dedup eksplisit per split)")
    df = pd.DataFrame([
        {"_orig_idx": 0, "Description": "alpha beta gamma", "Severity": "normal"},
        {"_orig_idx": 1, "Description": "Alpha  beta gamma", "Severity": "normal"},   # dup train-train
        {"_orig_idx": 2, "Description": "unique train doc", "Severity": "minor"},
        {"_orig_idx": c.TEST_MIN, "Description": "alpha beta gamma", "Severity": "normal"},  # dup test=train
        {"_orig_idx": c.TEST_MIN + 1, "Description": "fresh test doc", "Severity": "normal"},
        {"_orig_idx": c.TEST_MIN + 2, "Description": "fresh test doc", "Severity": "normal"},  # dup test-test
    ])
    out = c.stage_e2_near_dup(df)
    idx = sorted(out["_orig_idx"].tolist())
    check("hasil: train 0 & 2, test 1 saja", idx == [0, 2, c.TEST_MIN + 1], str(idx))

    # train tidak boleh berubah walau test diganti sepenuhnya
    df2 = df.copy()
    df2.loc[df2["_orig_idx"] >= c.TEST_MIN, "Description"] = ["x1", "x2", "x3"]
    tr_a = c.stage_e2_near_dup(df)[lambda d: d["_orig_idx"] <= c.TRAIN_MAX]["_orig_idx"].tolist()
    tr_b = c.stage_e2_near_dup(df2)[lambda d: d["_orig_idx"] <= c.TRAIN_MAX]["_orig_idx"].tolist()
    check("isi test tidak memengaruhi baris train yang tersisa", tr_a == tr_b, f"{tr_a} vs {tr_b}")


def test_tailcut() -> None:
    print("Tahap TAILCUT")
    df = pd.DataFrame({"Severity": ["normal"] * 6, "Description": [
        "app crashes on startup steps to reproduce click open then crash Updated by someone later comment fixed in build",
        "plain report with no marker at all just a description of the bug",
        "Marking Verified fixed",
        "the page renders wrong. VERIFIED on latest build and more text after",
        "first part marked as a duplicate of 123 second part",
        "see In reply to comment 4 thanks",
    ]})
    out = c.stage_tailcut(df)["Description"].tolist()
    check("dipotong di penanda pertama, bagian awal dipertahankan",
          out[0] == "app crashes on startup steps to reproduce click open then crash", out[0])
    check("dokumen tanpa penanda tetap utuh (tanpa fallback)", out[1] == df["Description"][1], out[1])
    check("penanda di awal dokumen -> dokumen kosong, baris TIDAK dibuang", out[2] == "" and len(out) == 6, repr(out[2]))
    check("penanda tidak peka huruf besar/kecil", out[3] == "the page renders wrong.", out[3])
    check("'marked as a duplicate of N' memotong", out[4] == "first part", out[4])
    check("'In reply to comment N' memotong", out[5] == "see", out[5])
    check("tidak ada penanda tersisa setelah pemotongan",
          not c.stage_tailcut(df)["Description"].str.contains(c.POST_REPORT).any())
    check("TAILCUT terdaftar dan konfigurasi plus_tailcut ada",
          "TAILCUT" in c.STAGES and c.CONFIGS.get("plus_tailcut") == ["A", "TAILCUT"])


def test_tailcut_ext() -> None:
    print("Tahap TAILCUT_EXT (penanda diperluas, sensitivitas)")
    cut = lambda t: c.tailcut_text(t, c.POST_REPORT_EXT)
    positives = {
        "report text here marking as wontfix because margins": "report text here",
        "report text here Reassigning to Kevin": "report text here",
        "report text here can someone verify this one is fixed": "report text here can someone",
        "report text here This is a DUPLICATE of bug 9333": "report text here This is a",
        "report text here raising severity to critical": "report text here",
        "report text here move to m16 for now": "report text here",
        "report text here fixed by my checkin for beta": "report text here fixed by my",
        "report text here Confirmed on Linux build": "report text here",
        "report text here marking this bug as a DUP of it": "report text here",
        "report text here Bulk move of XPInstall component": "report text here",
    }
    for src, exp in positives.items():
        check(f"memotong: '{src[17:45]}...'", cut(src) == exp, cut(src))
    negatives = [
        "steps to reproduce duplicate entries appear in the list",
        "Using M13 build on Linux the window is fixed width",
        "Created by Someone on Tuesday April 7 1998 10 57 25 AM PDT Additional Details crash",
    ]
    for src in negatives:
        check(f"tidak memotong: '{src[:40]}...'", cut(src) == src, cut(src))
    sample = "a b c In reply to comment 3 d e marking as fixed f"
    check("EXT selalu memotong di titik yang sama atau lebih awal dari POST_REPORT",
          len(cut(sample)) <= len(c.tailcut_text(sample)))
    check("TAILCUT_EXT terdaftar dan konfigurasi plus_tailcut_ext ada",
          "TAILCUT_EXT" in c.STAGES and c.CONFIGS.get("plus_tailcut_ext") == ["A", "TAILCUT_EXT"])


def test_merge_analysis() -> None:
    print("merge_analysis (--config tidak menghapus baris lain)")
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cleaning_analysis.csv"
        pd.DataFrame({"config": ["raw", "baseline"], "n_rows": [1, 2]}).to_csv(p, index=False)
        out = c.merge_analysis(p, pd.DataFrame({"config": ["plus_tailcut_ext", "baseline"], "n_rows": [9, 3]}))
        check("baris lama dipertahankan, config sama ditimpa, urut sesuai CONFIGS",
              out["config"].tolist() == ["raw", "baseline", "plus_tailcut_ext"] and out["n_rows"].tolist() == [1, 3, 9],
              str(out.to_dict("list")))
        out2 = c.merge_analysis(Path(d) / "tidak_ada.csv", pd.DataFrame({"config": ["full", "raw"], "n_rows": [5, 6]}))
        check("tanpa berkas lama: hanya baris baru, terurut", out2["config"].tolist() == ["raw", "full"])


def main() -> int:
    test_dates()
    test_tailcut()
    test_tailcut_ext()
    test_merge_analysis()
    test_truncate_train_only()
    test_dedup_split()
    print()
    if FAILED:
        print(f"{len(FAILED)} tes GAGAL: {FAILED}")
        return 1
    print("Semua tes lulus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
