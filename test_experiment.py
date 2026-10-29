"""
test_experiment.py — Tes kecil untuk fungsi baru experiment.py (suite RQ2 Paper 1).

Jalankan:  python test_experiment.py     (kode keluar 0 bila lulus)

Cakupan:
  - macro_f1_resamples() identik dengan sklearn f1_score(average="macro"), termasuk resample
    yang kehilangan satu kelas.
  - drop_random_block(): membuang TEPAT sebanyak token yang dibuang TAILCUT, satu blok berurutan,
    urutan token sisa tetap.
  - Indeks resample bersama menghasilkan selisih berpasangan yang konsisten.
  - plant_leak()/apply_leak() (kontrol positif, langkah 4): token mode tail selalu terbuang TAILCUT,
    dosis bersarang, hanya dokumen berpenanda, token = label benar.
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.metrics import f1_score

import experiment as ex

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if cond else 'GAGAL'}] {name}" + (f"  -> {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def test_macro_f1() -> None:
    print("macro_f1_resamples vs sklearn")
    rng = np.random.default_rng(0)
    k, n = 6, 400
    y = rng.choice(k, size=n, p=[.02, .05, .6, .15, .1, .08])
    p = np.where(rng.random(n) < 0.6, y, rng.integers(0, k, n))
    idx = rng.integers(0, n, size=(30, n))
    fast = ex.macro_f1_resamples(y, p, idx, k)
    slow = np.array([f1_score(y[i], p[i], average="macro", zero_division=0) for i in idx])
    check("30 resample acak sama dengan sklearn (selisih < 1e-12)", np.abs(fast - slow).max() < 1e-12,
          f"{np.abs(fast - slow).max():.2e}")
    # resample yang SEMUA barisnya dari satu kelas (kelas lain absen)
    only = np.where(y == 2)[0]
    idx2 = rng.choice(only, size=(1, n))
    check("resample satu kelas: sama dengan sklearn",
          abs(ex.macro_f1_resamples(y, p, idx2, k)[0]
              - f1_score(y[idx2[0]], p[idx2[0]], average="macro", zero_division=0)) < 1e-12)
    full = ex.macro_f1_resamples(y, p, np.arange(n)[None, :], k)[0]
    check("estimasi titik sama dengan sklearn", abs(full - f1_score(y, p, average="macro", zero_division=0)) < 1e-12)


def test_drop_block() -> None:
    print("drop_random_block")
    full = ["a b c d e f g h i j", "x y z", "no marker here at all"]
    cut = ["a b c d", "x y z", "no marker here at all"]      # dokumen 0 dipotong 6 token; sisanya utuh
    out = ex.drop_random_block(full, cut, np.random.default_rng(1))
    check("dokumen terpotong kehilangan tepat 6 token", len(out[0].split()) == 4, out[0])
    check("dokumen tanpa pemotongan tetap utuh", out[1] == full[1] and out[2] == full[2])
    toks = full[0].split()
    kept = out[0].split()
    # sisa harus berupa awalan + akhiran dari dokumen asli (satu blok berurutan dibuang)
    ok = any(toks[:s] + toks[s + 6:] == kept for s in range(0, len(toks) - 6 + 1))
    check("yang dibuang satu blok berurutan; urutan sisa tetap", ok, out[0])
    starts = set()
    for sd in range(40):
        o = ex.drop_random_block(full[:1], cut[:1], np.random.default_rng(sd))[0].split()
        starts.add(tuple(o))
    check("posisi blok bervariasi antar seed", len(starts) > 1, str(len(starts)))
    empty = ex.drop_random_block(["a b c"], [""], np.random.default_rng(0))
    check("dokumen terpotong habis (cut kosong) -> kontrol juga kosong", empty == [""], repr(empty))


def test_paired() -> None:
    print("bootstrap berpasangan")
    rng = np.random.default_rng(3)
    k, n = 6, 300
    y = rng.integers(0, k, n)
    p = y.copy()
    idx = rng.integers(0, n, size=(50, n))
    d = ex.macro_f1_resamples(y, p, idx, k) - ex.macro_f1_resamples(y, p, idx, k)
    check("model identik pada indeks yang sama -> selisih tepat 0 di semua resample", np.all(d == 0))


def test_plant_leak() -> None:
    print("plant_leak / apply_leak (kontrol positif sintetis)")
    import cleaning
    pat = cleaning.POST_REPORT
    doc = "crash on startup steps to reproduce open menu marked as a duplicate of 123 more text here and here"
    tail_ok, after_ok, once_ok = True, True, True
    for sd in range(60):
        out = ex.plant_leak(doc, "critical", "tail", np.random.default_rng(sd), pat)
        toks = out.split()
        tail_ok &= cleaning.tailcut_text(out) == cleaning.tailcut_text(doc)
        after_ok &= toks.index("leaktoken_critical") > toks.index("123")
        once_ok &= toks.count("leaktoken_critical") == 1 and len(toks) == len(doc.split()) + 1
    check("tail: hasil TAILCUT identik dengan dokumen asli (token ikut terbuang)", tail_ok)
    check("tail: token selalu sesudah penanda multi-kata selesai ('... of 123')", after_ok)
    check("token muncul tepat sekali, panjang +1", once_ok)
    try:
        ex.plant_leak("no marker here at all", "minor", "tail", np.random.default_rng(0), pat)
        check("tail pada dokumen tanpa penanda -> ValueError", False)
    except ValueError:
        check("tail pada dokumen tanpa penanda -> ValueError", True)
    pos = {ex.plant_leak(doc, "minor", "anywhere", np.random.default_rng(sd), pat).split().index("leaktoken_minor")
           for sd in range(200)}
    check("anywhere: posisi bervariasi dan bisa sebelum penanda", min(pos) < doc.split().index("marked"), str(sorted(pos)[:5]))
    # dokumen pendek penuh penanda multi-kata: hampir setiap celah ada di dalam penanda
    multi = "x In reply to comment 5 y works for me z checked in w marked as a duplicate of 9"
    spans0 = [m.group(0) for m in pat.finditer(multi)]
    keep_ok, cut_ok, seen = True, True, set()
    for sd in range(300):
        out = ex.plant_leak(multi, "major", "anywhere", np.random.default_rng(sd), pat)
        seen.add(out.split().index("leaktoken_major"))
        keep_ok &= [m.group(0) for m in pat.finditer(out)] == spans0
        cut_ok &= [t for t in cleaning.tailcut_text(out).split() if t != "leaktoken_major"] \
            == cleaning.tailcut_text(multi).split()
    check("anywhere: token tidak pernah membelah penanda (daftar kecocokan tetap)", keep_ok)
    check("anywhere: titik potong TAILCUT tidak bergeser", cut_ok)
    check("anywhere: hanya celah di luar penanda yang dipakai", seen == {0, 1, 6, 7, 10, 11, 13, 14, 20}, str(sorted(seen)))

    docs = [f"report {i} text verified later" if i % 4 else f"report {i} clean text" for i in range(400)]
    labels = np.array(["normal", "minor", "major", "trivial"] * 100)
    d0, m0 = ex.apply_leak(docs, labels, 0.0, "tail", 7, pat)
    check("p = 0 -> dokumen tidak berubah", d0 == docs and not m0.any())
    d1, m1 = ex.apply_leak(docs, labels, 0.05, "tail", 7, pat)
    d2, m2 = ex.apply_leak(docs, labels, 0.20, "tail", 7, pat)
    check("jumlah dokumen bocor = round(p * n)", m1.sum() == 20 and m2.sum() == 80, f"{m1.sum()}, {m2.sum()}")
    check("hanya dokumen berpenanda yang dibocorkan", all(pat.search(docs[i]) for i in np.flatnonzero(m2)))
    check("himpunan bocor bersarang antar dosis", bool(np.all(m2[m1])))
    check("dokumen yang sama -> teks bocor sama di dosis berbeda", all(d1[i] == d2[i] for i in np.flatnonzero(m1)))
    check("token bocor = label benar dokumen itu",
          all(f"leaktoken_{labels[i]}" in d2[i] for i in np.flatnonzero(m2)))
    try:
        ex.apply_leak(docs, labels, 0.9, "tail", 7, pat)          # 360 dibutuhkan, hanya 300 berpenanda
        check("dosis melebihi dokumen berpenanda -> ValueError", False)
    except ValueError:
        check("dosis melebihi dokumen berpenanda -> ValueError", True)


def test_block_starts() -> None:
    print("random_block_starts / drop_block_at (kontrol berpasangan antar dosis)")
    import cleaning
    pat = cleaning.POST_REPORT
    rng = np.random.default_rng(5)
    full = [" ".join(f"w{j}" for j in range(rng.integers(5, 40))) + (" verified done" if i % 3 else "")
            for i in range(60)]
    cut = [cleaning.tailcut_text(t) for t in full]
    a = ex.drop_random_block(full, cut, np.random.default_rng(11))
    b = ex.drop_block_at(full, cut, ex.random_block_starts(full, cut, np.random.default_rng(11)))
    check("starts + drop_block_at == drop_random_block (Tabel 3 tetap terulang)", a == b)
    starts = ex.random_block_starts(full, cut, np.random.default_rng(11))
    labels = np.array(["minor"] * len(full))
    ok = True
    for mode in ("tail", "anywhere"):
        lk, mk = ex.apply_leak(full, labels, 0.3, mode, 3, pat)
        ck = ex.drop_block_at(lk, [cleaning.tailcut_text(t) for t in lk], starts)   # tidak boleh ValueError
        ok &= all(ck[i] == b[i] for i in np.flatnonzero(~mk))
    check("dokumen tak bocor mendapat kontrol identik dengan p0 (tail & anywhere)", ok)
    try:
        ex.drop_block_at(["a b c d"], ["a"], [2])
        check("posisi awal di luar batas -> ValueError", False)
    except ValueError:
        check("posisi awal di luar batas -> ValueError", True)


def test_shared_vectorizer() -> None:
    print("fit_predict_models (TF-IDF bersama)")
    rng = np.random.default_rng(0)
    vocab = [f"t{i}" for i in range(40)]
    y = np.array(["minor", "major", "normal"] * 40)
    docs = [" ".join(rng.choice(vocab, 12)) + f" cls{lab}" for lab in y]
    shared, vec = ex.fit_predict_models(["logreg", "linearsvc"], docs[:90], y[:90], docs[90:], 500)
    ok = all((shared[m] == ex.fit_predict(m, docs[:90], y[:90], docs[90:], 500)).all() for m in shared)
    check("prediksi identik dengan fit_predict per model", ok)
    check("vectorizer dikembalikan (dipakai cek kosakata)", "clsminor" in vec.vocabulary_)


def test_models() -> None:
    print("make_model")
    check("logreg tanpa n_jobs", "n_jobs" in ex.make_model("logreg").get_params()
          and ex.make_model("logreg").get_params()["n_jobs"] is None)
    check("linearsvc C=0.5 balanced", ex.make_model("linearsvc").C == 0.5
          and ex.make_model("linearsvc").class_weight == "balanced")


def main() -> int:
    test_macro_f1()
    test_drop_block()
    test_paired()
    test_plant_leak()
    test_block_starts()
    test_shared_vectorizer()
    test_models()
    print()
    if FAILED:
        print(f"{len(FAILED)} tes GAGAL: {FAILED}")
        return 1
    print("Semua tes lulus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
