"""
test_laporan_awal.py — Tes kecil untuk pemasangan laporan awal (kaggle_laporan_awal.py), tahap INITIAL
(cleaning.py), suite rq2_initial (experiment.py), dan fungsi bantu task_kurva_belajar.py (11 Okt 2026).

Jalankan:  python test_laporan_awal.py     (kode keluar 0 bila lulus)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import cleaning as c
import experiment as ex
import kaggle_laporan_awal as kl
import task_kurva_belajar as tk

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if cond else 'GAGAL'}] {name}" + (f"  -> {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def test_normalisasi_dan_awalan() -> None:
    print("norm / is_word_prefix / n_ws_tokens_for_prefix")
    check("norm membuang tanda baca dan huruf besar", kl.norm("Alt-W, (now)!") == "alt w now")
    check("awalan pada batas kata", kl.is_word_prefix("app crashes", "app crashes on start"))
    check("BUKAN awalan bila memotong kata", not kl.is_word_prefix("app crash", "app crashes on start"))
    check("teks identik dihitung awalan", kl.is_word_prefix("app crashes", "app crashes"))
    raw = "Pressing Alt w seems\nto have no effect   Marking Verified"
    check("archive tanpa tanda baca: 6 token laporan awal",
          kl.n_ws_tokens_for_prefix(raw, kl.norm("Pressing Alt-w seems to have")) == 6,
          str(kl.n_ws_tokens_for_prefix(raw, kl.norm("Pressing Alt-w seems to have"))))
    raw2 = "Created by Darrell (dk@cs.cmu.edu) on Tue\nPressing Alt-w now. Updated by Steve"
    n = kl.n_ws_tokens_for_prefix(raw2, kl.norm("Created by Darrell (dk@cs.cmu.edu) on Tue Pressing Alt-w now."))
    check("teks bertanda baca: 9 token spasi sampai 'now.'", n == 9, str(n))
    check("awalan tak tercapai -> -1", kl.n_ws_tokens_for_prefix("abc def", "abc def ghi") == -1)
    check("token terakhir yang menempel tanda baca ikut utuh",
          " ".join(raw2.split()[:n]).endswith("now."))


def test_pair() -> None:
    print("pair()")
    archive = pd.DataFrame({"Description": [
        "the app crashes on start when clicking menu Marking Verified thanks",
        "another bug report text here with many words and later comments",
        "nothing matches this document at all really",
    ]})
    bugs = pd.DataFrame({"text": [
        "The app crashes on start, when clicking menu.",   # cocok dok 0 (terpanjang)
        "the app crashes on start",                          # juga awalan dok 0, lebih pendek
        "another bug report text here",                      # cocok dok 1
        "the app",                                           # < MIN_TOKENS, diabaikan
    ]})
    p = kl.pair(archive, bugs).set_index("row_archive")
    check("dua dokumen berpasangan, dokumen tanpa awalan tidak", sorted(p.index) == [0, 1], str(list(p.index)))
    check("kandidat terpanjang dipilih untuk dok 0", p.at[0, "bugs_idx"] == 0, str(p.at[0, "bugs_idx"]))
    check("n_kandidat dok 0 = 2", p.at[0, "n_kandidat"] == 2)
    check("tiap baris bugs dipakai sekali", (p["bugs_dipakai"] == 1).all())


def test_stage_initial_dan_suite() -> None:
    print("stage_initial / build_initial_conditions")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        docs = ["report one about crash then comment marking verified done",
                "report two about layout then fixed in build",
                "report one about crash then a different comment here",     # test, laporan awal = dok 0
                "report three in test with later reply comment",
                "report four without any pair at all"]
        orig = [0, 1, 28507, 28508, 2]
        sev = ["major", "minor", "major", "normal", "normal"]
        pd.DataFrame({"Unnamed: 0.1": orig, "Unnamed: 0": orig, "Description": docs, "Severity": sev,
                      "Label": 0}).to_csv(td / "sev.csv", index=False)
        pairs = pd.DataFrame({"orig_idx": [0, 1, 28507, 28508], "n_tok_awal": [4, 4, 4, 5],
                              "tingkat": ["utama", "utama", "utama", "utama"]})
        pairs.to_csv(td / "pairs.csv", index=False)
        old = c.LAPORAN_AWAL_CSV
        c.LAPORAN_AWAL_CSV = td / "pairs.csv"
        try:
            a = c.stage_a_structure(pd.read_csv(td / "sev.csv"))
            init = c.stage_initial(a)
        finally:
            c.LAPORAN_AWAL_CSV = old
        check("dokumen tanpa pasangan dibuang", len(init) == 4 and 2 not in set(init["_orig_idx"]))
        check("laporan awal = n token pertama",
              init.set_index("_orig_idx").at[0, "Description"] == "report one about crash")
        tr, te, info = ex.build_initial_conditions(td, td / "pairs.csv")
        check("test yang laporan awalnya identik dengan train dibuang", info["test_dibuang_duplikat_laporan_awal"] == 1
              and list(te["_orig_idx"]) == [28508], str(info))
        check("kolom init / tail / Description sejajar", (te["init"] == "report three in test with").all()
              and te["Description"].iloc[0].startswith(te["init"].iloc[0]))
        check("tail = tailcut_text(teks A)", tr.set_index("_orig_idx").at[0, "tail"] == "report one about crash then comment")


def test_task_helpers() -> None:
    print("task_kurva_belajar: fungsi bantu")
    w = tk.balanced_weights(np.array(["a", "a", "a", "b"]))
    check("bobot kelas seimbang n/(k*count)", abs(w["a"] - 4 / 6) < 1e-12 and abs(w["b"] - 2.0) < 1e-12)
    check("lcp_tokens", tk.lcp_tokens("a b c d".split(), "a b x".split()) == 2)
    df = pd.DataFrame({"_orig_idx": [0, 24999, 25000, 28505, 28506, 28507], "Description": list("abcdef")})
    sp = tk.split3(df)
    check("split3: batas train_fit/val/test", list(sp["train"]["_orig_idx"]) == [0, 24999]
          and list(sp["val"]["_orig_idx"]) == [25000, 28505] and list(sp["test"]["_orig_idx"]) == [28507])
    md = tk.md_table(pd.DataFrame({"x": [1, 2], "y": [0.5, 0.25]}))
    check("md_table menulis header + 2 baris", md.count("\n") == 3 and md.startswith("| x | y |"))
    rng = np.random.default_rng(0)
    y = np.array(["normal"] * 60 + ["major"] * 30 + ["minor"] * 10)
    X = csr_matrix(rng.random((100, 20)) + (np.array([[i % 3 == 0] * 20 for i in range(100)]) * 0.5))
    clf, h, be = tk.train_sgd(X[:80], y[:80], X[80:], y[80:], seed=0, max_epochs=8, patience=2)
    check("train_sgd: riwayat <= max_epochs dan epoch terbaik di dalamnya", len(h) <= 8 and 1 <= be <= len(h))
    check("train_sgd: berhenti paling lambat patience sesudah epoch terbaik", len(h) <= be + 2)
    check("train_sgd: koefisien epoch terbaik dipulihkan (bentuk benar)", clf.coef_.shape == (6, 20))


def test_perbaiki_gap() -> None:
    print("task_perbaiki_gap: label fitur dan semua keluarga model")
    import task_perbaiki_gap as pg
    check("feat_label", pg.feat_label((2000, (1, 1), True)) == "2k uni sub" and pg.feat_label((20000, (1, 2), False)) == "20k bi")
    check("grid fitur 12 kombinasi, acuan lama termasuk", len(pg.FEATS) == 12 and pg.BASE_FEAT in pg.FEATS)
    rng = np.random.default_rng(1)
    y = np.array(["normal"] * 40 + ["major"] * 25 + ["minor"] * 15)
    X = csr_matrix(rng.random((80, 15)))
    for fam, reg in (("logreg", 0.1), ("linearsvc", 0.1), ("cnb", 0.3), ("sgd", 1e-3)):
        m, info = pg.fit_model(fam, reg, X[:60], y[:60], X[60:], y[60:], max_epochs=4, patience=2)
        ok = len(m.predict(X[60:])) == 20 and (fam != "sgd" or 1 <= info["best_epoch"] <= info["epoch_dijalankan"] <= 4)
        check(f"fit_model {fam} memprediksi val (SGD: epoch terbaik tercatat)", ok)
    import task_kurva_belajar as tkb
    _, h1, _ = tkb.train_sgd(X[:60], y[:60], X[60:], y[60:], seed=0, max_epochs=3, patience=2)
    _, h2, _ = tkb.train_sgd(X[:60], y[:60], X[60:], y[60:], seed=0, max_epochs=3, patience=2, alpha=1e-4)
    check("train_sgd alpha default = 1e-4 (kompatibel dengan pemanggilan lama)", h1.equals(h2))


def main() -> int:
    test_normalisasi_dan_awalan()
    test_perbaiki_gap()
    test_pair()
    test_stage_initial_dan_suite()
    test_task_helpers()
    print()
    if FAILED:
        print(f"{len(FAILED)} tes GAGAL: {FAILED}")
        return 1
    print("Semua tes lulus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
