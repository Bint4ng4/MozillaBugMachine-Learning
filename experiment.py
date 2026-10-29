"""
experiment.py — Baseline model untuk Gate 1b (kriteria model, RQ2/RQ2b/RQ3).

Gate 1b BELUM PERNAH DIUJI sebelum skrip ini ditulis (10 Sep 2026). Gate 1a (kualitas data)
sudah lolos lewat verify_data.py — ini beda: apakah model paling sederhana (TF-IDF + Logistic
Regression) bisa melampaui macro-F1 >= 0,25 (baseline majority-class 0,138)?

Desain:
    - Split KRONOLOGIS bawaan (sev_train.csv / sev_test.csv), bukan random — meniru deployment.
    - TF-IDF di-fit HANYA pada train (no leakage transductive).
    - class_weight='balanced' karena imbalance ekstrem (normal 19.689 vs trivial 486, ~40:1).
    - Bootstrap CI (1000 resample) pada test set untuk macro-F1.
    - `Label` TIDAK dipakai sebagai fitur — deterministik dari Severity (V=1.0, verify_data.py §3).
    - Berjalan untuk config manapun via --config (default: raw = sev_train/sev_test asli;
      atau pakai output cleaning.py lewat --train-csv/--test-csv custom).

Cara pakai:
    python experiment.py                              # baseline di split bawaan (raw)
    python experiment.py --train archive/sev_train.csv --test archive/sev_test.csv
    python experiment.py --out results_raw.json

Butuh: pandas, numpy, scikit-learn, scipy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC

SEVERITY_ORDER = ["trivial", "minor", "normal", "major", "critical", "blocker"]


def make_model(name: str):
    """Model untuk Paper 1. Hiperparameter mengikuti probe MASTER SS8d (LogReg C=1, LinearSVC C=0,5).

    `n_jobs` sengaja tidak dipakai (deprecated di sklearn baru untuk LogisticRegression).
    """
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    if name == "linearsvc":
        return LinearSVC(C=0.5, class_weight="balanced")
    raise ValueError(f"model tidak dikenal: {name!r} (pilih logreg atau linearsvc)")


def macro_f1_resamples(y_true: np.ndarray, y_pred: np.ndarray, idx: np.ndarray, k: int) -> np.ndarray:
    """macro-F1 untuk setiap baris indeks resample `idx` (B x n); y_true/y_pred berupa int 0..k-1.

    Dirata-rata hanya atas kelas yang muncul di y_true ATAU y_pred pada resample itu -- sama dengan
    perilaku sklearn f1_score(average="macro") tanpa argumen labels.
    """
    yt, yp = y_true[idx], y_pred[idx]
    out = np.empty(len(idx))
    for b in range(len(idx)):
        cm = np.bincount(yt[b] * k + yp[b], minlength=k * k).reshape(k, k)
        tp = np.diag(cm).astype(float)
        den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
        present = den > 0
        out[b] = (2 * tp[present] / den[present]).mean()
    return out


def drop_random_block(full_docs, cut_docs, rng: np.random.Generator) -> list[str]:
    """Kontrol panjang: per dokumen, buang SATU blok token berurutan di posisi acak sepanjang
    jumlah token yang dibuang TAILCUT pada dokumen itu (panjang sama, posisi acak)."""
    return drop_block_at(full_docs, cut_docs, random_block_starts(full_docs, cut_docs, rng))


def random_block_starts(full_docs, cut_docs, rng: np.random.Generator) -> list[int | None]:
    """Posisi awal blok acak per dokumen (None bila tidak ada yang dibuang). Urutan undian sama
    persis dengan drop_random_block, jadi posisi ini bisa disimpan lalu dipakai ulang."""
    starts: list[int | None] = []
    for full, cut in zip(full_docs, cut_docs):
        n_full = len(full.split())
        k = n_full - len(cut.split())
        starts.append(int(rng.integers(0, n_full - k + 1)) if k > 0 else None)
    return starts


def drop_block_at(full_docs, cut_docs, starts) -> list[str]:
    """Buang satu blok sepanjang (token full - token cut) mulai dari `starts[i]` per dokumen."""
    out = []
    for full, cut, s in zip(full_docs, cut_docs, starts, strict=True):
        toks = full.split()
        k = len(toks) - len(cut.split())
        if k <= 0:
            out.append(full)
            continue
        if s is None or s > len(toks) - k:
            raise ValueError(f"posisi awal {s} tidak sah untuk dokumen {len(toks)} token, blok {k}")
        out.append(" ".join(toks[:s] + toks[s + k:]))
    return out


LEAK_PREFIX = "leaktoken_"


def leak_token(label: str) -> str:
    return f"{LEAK_PREFIX}{label}"


def plant_leak(doc: str, label: str, mode: str, rng: np.random.Generator, pattern) -> str:
    """Kontrol positif sintetis: sisipkan SATU token bocor-label `leaktoken_<label>` ke `doc`
    pada batas token, tidak pernah DI TENGAH frasa penanda (mis. "works | for me"), supaya
    penanda dan titik potong TAILCUT tidak bergeser karena penyisipan.

      tail      posisi acak SESUDAH kecocokan `pattern` pertama berakhir (wilayah yang dibuang
                TAILCUT). Teks sampai akhir kecocokan tidak berubah, jadi tailcut_text(hasil) ==
                tailcut_text(doc): kebocoran meniru isi pasca-pelaporan yang menumpuk di ekor.
      anywhere  posisi acak di mana pun; TAILCUT hanya membuangnya bila jatuh di ekor (kebocoran
                tersebar, meniru kontaminasi yang juga ada sebelum penanda pertama).
    """
    toks = doc.split()
    text = " ".join(toks)
    spans = [(m.start(), m.end()) for m in pattern.finditer(text)]
    if mode == "tail":
        if not spans:
            raise ValueError("mode 'tail' butuh dokumen yang memuat penanda")
        lo = len(text[: spans[0][1]].split())      # indeks token sesudah penanda pertama selesai
    elif mode == "anywhere":
        lo = 0
    else:
        raise ValueError(f"mode tidak dikenal: {mode!r} (pilih tail atau anywhere)")
    gap = np.cumsum([len(t) + 1 for t in toks]) - 1   # gap[i] = indeks karakter tepat sesudah toks[i]
    allowed = [p for p in range(lo, len(toks) + 1)
               if p == 0 or not any(s <= gap[p - 1] < e for s, e in spans)]
    pos = allowed[int(rng.integers(len(allowed)))]
    return " ".join(toks[:pos] + [leak_token(label)] + toks[pos:])


def apply_leak(docs, labels, p: float, mode: str, seed: int, pattern) -> tuple[list[str], np.ndarray]:
    """Sisipkan token bocor ke round(p * len(docs)) dokumen yang memuat `pattern`.

    Urutan dokumen layak diacak SEKALI per `seed`, dan posisi sisip tiap dokumen memakai rng
    (seed, indeks dokumen). Akibatnya himpunan dokumen bocor BERSARANG antar dosis dan satu dokumen
    mendapat teks bocor yang sama di setiap dosis -> kurva dosis-respons tidak bergerigi karena
    undian ulang. Kembalikan (dokumen, mask dokumen yang dibocorkan).
    """
    docs = list(docs)
    eligible = np.array([i for i, d in enumerate(docs) if pattern.search(d)], dtype=int)
    order = np.random.default_rng(seed).permutation(eligible)
    n_leak = int(round(p * len(docs)))
    if n_leak > len(order):
        raise ValueError(f"p={p} butuh {n_leak} dokumen berpenanda, hanya ada {len(order)}")
    mask = np.zeros(len(docs), dtype=bool)
    for i in order[:n_leak]:
        docs[i] = plant_leak(docs[i], labels[i], mode, np.random.default_rng([seed, int(i)]), pattern)
        mask[i] = True
    return docs, mask


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def load_split(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(train_path)
    te = pd.read_csv(test_path)
    junk = lambda df: [c for c in df.columns if c.startswith("Unnamed") or not str(c).strip()]
    tr = tr.drop(columns=junk(tr))
    te = te.drop(columns=junk(te))
    # fillna DULU -- astype(str) pada NaN bisa jadi literal "nan" (bukan string kosong) atau
    # tetap NaN tergantung versi pandas, keduanya salah. Terjadi kalau dekontaminasi menghapus
    # SELURUH isi dokumen (mis. dokumen yang cuma berisi "Marking Verified" saja).
    n_nan_tr = tr["Description"].isna().sum()
    n_nan_te = te["Description"].isna().sum()
    if n_nan_tr or n_nan_te:
        print(f"  (peringatan: {n_nan_tr} NaN di train, {n_nan_te} NaN di test -> diisi string kosong)")
    tr["Description"] = tr["Description"].fillna("").astype(str)
    te["Description"] = te["Description"].fillna("").astype(str)
    return tr, te


def bootstrap_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000,
                        seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap CI 95% untuk macro-F1 lewat resampling indeks test set."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        scores[b] = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(scores.mean()), float(lo), float(hi)


def make_vectorizer(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)


def fit_predict(model_name: str, tr_docs, y_train, te_docs, max_features: int) -> np.ndarray:
    """TF-IDF di-fit HANYA pada train, lalu model dilatih dan memprediksi test."""
    return fit_predict_models([model_name], tr_docs, y_train, te_docs, max_features)[0][model_name]


def fit_predict_models(models, tr_docs, y_train, te_docs,
                       max_features: int) -> tuple[dict[str, np.ndarray], TfidfVectorizer]:
    """Satu TF-IDF (fit HANYA pada train) dipakai bersama oleh semua model -- hasilnya identik dengan
    memanggil fit_predict per model, tanpa membangun ulang matriks fitur."""
    vec = make_vectorizer(max_features)
    X_train = vec.fit_transform(tr_docs)
    X_test = vec.transform(te_docs)
    return {m: make_model(m).fit(X_train, y_train).predict(X_test) for m in models}, vec


def run_rq2(args) -> int:
    """Suite RQ2 (Paper 1): ref (A saja) vs TAILCUT vs kontrol buang-blok-acak, paired bootstrap.

    Semua kondisi memakai test set yang SAMA (baris sejajar), sehingga indeks resample bisa
    dipakai bersama untuk semua kondisi (uji berpasangan). Keluaran: results_rq2.json dan
    tabel3_rq2.csv di --out-dir.
    """
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_tr, ref_te = load_split(Path(args.ref_train), Path(args.ref_test))
    cut_tr, cut_te = load_split(Path(args.cut_train), Path(args.cut_test))
    if len(ref_tr) != len(cut_tr) or len(ref_te) != len(cut_te) \
            or not (ref_tr["Severity"].values == cut_tr["Severity"].values).all() \
            or not (ref_te["Severity"].values == cut_te["Severity"].values).all():
        print("FATAL: baris ref dan cut tidak sejajar (jumlah/label berbeda). "
              "Pakai konfigurasi yang tidak membuang baris (mis. baseline vs plus_tailcut).")
        return 1
    rule(f"RQ2 SUITE: train {len(ref_tr)} / test {len(ref_te)} baris sejajar; seed kontrol = {args.seeds}")

    cls = SEVERITY_ORDER
    k = len(cls)
    to_int = {c: i for i, c in enumerate(cls)}
    y_train = ref_tr["Severity"].values
    y_true = np.array([to_int[v] for v in ref_te["Severity"].values])
    n = len(y_true)
    rng_idx = np.random.default_rng(args.boot_seed)
    idx = rng_idx.integers(0, n, size=(args.n_boot, n))          # indeks resample BERSAMA
    full_idx = np.arange(n)[None, :]

    removed = (ref_te["Description"].str.split().str.len() - cut_te["Description"].str.split().str.len())
    print(f"  token test: ref {ref_te['Description'].str.split().str.len().mean():.1f} -> "
          f"cut {cut_te['Description'].str.split().str.len().mean():.1f}; "
          f"dokumen test terpotong: {(removed > 0).mean():.1%}")

    report: dict = {"config": vars(args), "n_train": len(ref_tr), "n_test": n, "models": {}}
    rows = []
    for m in args.models:
        rule(f"MODEL: {m}")
        preds: dict[str, np.ndarray] = {}

        def run_arm(name, tr_docs, te_docs):
            t1 = time.time()
            p = fit_predict(m, tr_docs, y_train, te_docs, args.max_features)
            preds[name] = np.array([to_int[v] for v in p])
            f1 = macro_f1_resamples(y_true, preds[name], full_idx, k)[0]
            print(f"  {name:<12} macro-F1 = {f1:.4f}   ({time.time() - t1:.0f} dtk)")

        run_arm("ref", ref_tr["Description"], ref_te["Description"])
        run_arm("tailcut", cut_tr["Description"], cut_te["Description"])
        for s in range(args.seeds):
            rng = np.random.default_rng(s)
            c_tr = drop_random_block(ref_tr["Description"], cut_tr["Description"], rng)
            c_te = drop_random_block(ref_te["Description"], cut_te["Description"], rng)
            run_arm(f"control_{s}", c_tr, c_te)

        ctrl_names = [f"control_{s}" for s in range(args.seeds)]
        boot = {a: macro_f1_resamples(y_true, p, idx, k) for a, p in preds.items()}
        boot["control"] = np.mean([boot[c] for c in ctrl_names], axis=0)   # rata-rata seed per resample
        point = {a: float(macro_f1_resamples(y_true, p, full_idx, k)[0]) for a, p in preds.items()}
        point["control"] = float(np.mean([point[c] for c in ctrl_names]))
        ctrl_sd = float(np.std([point[c] for c in ctrl_names], ddof=1)) if args.seeds > 1 else float("nan")

        res = {"point": point, "control_sd_across_seeds": ctrl_sd, "arms_ci95": {}, "diffs": {}}
        for a in ("ref", "tailcut", "control"):
            lo, hi = np.percentile(boot[a], [2.5, 97.5])
            res["arms_ci95"][a] = [float(lo), float(hi)]
        for name, (a, b) in {"tailcut - ref": ("tailcut", "ref"),
                              "control - ref": ("control", "ref"),
                              "tailcut - control": ("tailcut", "control")}.items():
            d = boot[a] - boot[b]
            lo, hi = np.percentile(d, [2.5, 97.5])
            est = point[a] - point[b]
            res["diffs"][name] = {"estimate": float(est), "ci95": [float(lo), float(hi)],
                                  "ci_includes_zero": bool(lo <= 0 <= hi)}
            rows.append({"model": m, "comparison": name, "estimate": round(est, 4),
                         "ci95_lo": round(float(lo), 4), "ci95_hi": round(float(hi), 4),
                         "ci_includes_zero": bool(lo <= 0 <= hi)})
        report["models"][m] = res

        print(f"\n  macro-F1: ref {point['ref']:.4f} | tailcut {point['tailcut']:.4f} | "
              f"control {point['control']:.4f} (sd antar seed {ctrl_sd:.4f})")
        for name, d in res["diffs"].items():
            print(f"  {name:<18} {d['estimate']:+.4f}  CI95% [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
                  f"  {'melewati 0' if d['ci_includes_zero'] else 'TIDAK melewati 0'}")

        if args.raw_train and Path(args.raw_train).exists():
            r_tr, r_te = load_split(Path(args.raw_train), Path(args.raw_test))
            p = fit_predict(m, r_tr["Description"], r_tr["Severity"].values, r_te["Description"], args.max_features)
            f1_raw = f1_score(r_te["Severity"].values, p, average="macro", zero_division=0)
            report["models"][m]["raw_split_reference"] = {"macro_f1": float(f1_raw), "n_train": len(r_tr),
                                                          "n_test": len(r_te)}
            print(f"  (acuan reproduksi) raw split bawaan: macro-F1 = {f1_raw:.4f}")

    pd.DataFrame(rows).to_csv(out_dir / "tabel3_rq2.csv", index=False)
    report["elapsed_seconds"] = time.time() - t0
    (out_dir / "results_rq2.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  Disimpan: {out_dir / 'results_rq2.json'} dan tabel3_rq2.csv  ({report['elapsed_seconds']:.0f} dtk)")
    return 0


def run_rq2_sens(args) -> int:
    """Suite SENSITIVITAS RQ2: apakah null bertahan bila dekontaminasi lebih tegas?

    Latar: audit 8 Okt 2026 (penilai model, provisional) menemukan ~88% dokumen pasca-TAILCUT masih
    memuat komentar pasca-pelaporan, jadi TAILCUT hanya dekontaminasi parsial. Semua kondisi di sini
    diturunkan dari dokumen ref (baseline, tahap A) sehingga baris test sejajar dan uji berpasangan sah:

      first{N} / last{N}  simpan N token pertama / terakhir = buang SATU blok sepanjang L-N di akhir /
                          di awal. Tanpa regex -> tidak sirkular. Kontaminasi menumpuk di bagian akhir
                          (kutipan bukti audit: posisi relatif median 0,74), jadi first{N} = versi paling
                          bersih dan last{N} = versi paling kotor pada panjang yang SAMA.
      ctrl{N}             buang satu blok sepanjang L-N di posisi ACAK (rata-rata --seeds seed).
      tailcut_ext         potong pada penanda pertama cleaning.POST_REPORT_EXT (kosakata Bugzilla luas).
      ctrl_ext            kontrol blok-acak sepanjang yang dibuang tailcut_ext.

    Tafsir: bila kontaminasi menggelembungkan skor, last{N} > ctrl{N} > first{N} dan tailcut_ext <
    ctrl_ext. Catatan: first{N} - last{N} juga memuat beda informativitas posisi (deskripsi awal pelapor
    vs komentar), jadi baca bersama perbandingan terhadap ctrl{N}.
    Keluaran: results_rq2_sens.json dan tabel_sensitivitas_rq2.csv di --out-dir.
    """
    import cleaning  # penanda diperluas & fungsi potong, satu sumber dengan cleaning.py

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_tr, ref_te = load_split(Path(args.ref_train), Path(args.ref_test))
    cls = SEVERITY_ORDER
    k = len(cls)
    to_int = {c: i for i, c in enumerate(cls)}
    y_train = ref_tr["Severity"].values
    y_true = np.array([to_int[v] for v in ref_te["Severity"].values])
    n = len(y_true)
    idx = np.random.default_rng(args.boot_seed).integers(0, n, size=(args.n_boot, n))
    full_idx = np.arange(n)[None, :]

    # ---- bangun semua kondisi (teks train, teks test) dari dokumen ref
    def first_n(docs, N):
        return [" ".join(d.split()[:N]) for d in docs]

    def last_n(docs, N):
        return [" ".join(d.split()[-N:]) if N else "" for d in docs]

    arms: dict[str, tuple[list[str], list[str]]] = {"ref": (list(ref_tr["Description"]), list(ref_te["Description"]))}
    families: dict[str, dict] = {}
    for N in args.n_tokens:
        f_tr, f_te = first_n(ref_tr["Description"], N), first_n(ref_te["Description"], N)
        arms[f"first{N}"] = (f_tr, f_te)
        arms[f"last{N}"] = (last_n(ref_tr["Description"], N), last_n(ref_te["Description"], N))
        ctrl = []
        for s in range(args.seeds):
            rng = np.random.default_rng(1000 * N + s)
            arms[f"ctrl{N}_{s}"] = (drop_random_block(ref_tr["Description"], f_tr, rng),
                                    drop_random_block(ref_te["Description"], f_te, rng))
            ctrl.append(f"ctrl{N}_{s}")
        families[f"N{N}"] = {"ctrl_name": f"ctrl{N}", "ctrl_arms": ctrl, "comparisons": {
            f"first{N} - ref": (f"first{N}", "ref"), f"last{N} - ref": (f"last{N}", "ref"),
            f"ctrl{N} - ref": (f"ctrl{N}", "ref"), f"first{N} - ctrl{N}": (f"first{N}", f"ctrl{N}"),
            f"last{N} - ctrl{N}": (f"last{N}", f"ctrl{N}"), f"first{N} - last{N}": (f"first{N}", f"last{N}")}}
    e_tr = [cleaning.tailcut_text(t, cleaning.POST_REPORT_EXT) for t in ref_tr["Description"]]
    e_te = [cleaning.tailcut_text(t, cleaning.POST_REPORT_EXT) for t in ref_te["Description"]]
    arms["tailcut_ext"] = (e_tr, e_te)
    ctrl = []
    for s in range(args.seeds):
        rng = np.random.default_rng(7000 + s)
        arms[f"ctrl_ext_{s}"] = (drop_random_block(ref_tr["Description"], e_tr, rng),
                                 drop_random_block(ref_te["Description"], e_te, rng))
        ctrl.append(f"ctrl_ext_{s}")
    families["ext"] = {"ctrl_name": "ctrl_ext", "ctrl_arms": ctrl, "comparisons": {
        "tailcut_ext - ref": ("tailcut_ext", "ref"), "ctrl_ext - ref": ("ctrl_ext", "ref"),
        "tailcut_ext - ctrl_ext": ("tailcut_ext", "ctrl_ext")}}

    tok = {a: float(np.mean([len(d.split()) for d in te])) for a, (_, te) in arms.items()}
    hit_ext = float(np.mean([cleaning.POST_REPORT_EXT.search(t) is not None for t in ref_te["Description"]]))
    hit_base = float(np.mean([cleaning.POST_REPORT.search(t) is not None for t in ref_te["Description"]]))
    rule(f"RQ2 SENSITIVITAS: train {len(ref_tr)} / test {n}; N = {args.n_tokens}; seed kontrol = {args.seeds}")
    print(f"  dokumen test berpenanda: POST_REPORT {hit_base:.1%} | POST_REPORT_EXT {hit_ext:.1%}")
    print("  token/dok test: " + ", ".join(f"{a} {v:.0f}" for a, v in tok.items() if not a[-2:-1] == "_"))

    report: dict = {"config": vars(args), "n_train": len(ref_tr), "n_test": n, "tokens_test_mean": tok,
                    "test_marker_rate": {"POST_REPORT": hit_base, "POST_REPORT_EXT": hit_ext}, "models": {}}
    rows = []
    for m in args.models:
        rule(f"MODEL: {m}")
        preds = {}
        for a, (tr_docs, te_docs) in arms.items():
            t1 = time.time()
            p = fit_predict(m, tr_docs, y_train, te_docs, args.max_features)
            preds[a] = np.array([to_int[v] for v in p])
            print(f"  {a:<14} macro-F1 = {macro_f1_resamples(y_true, preds[a], full_idx, k)[0]:.4f}  "
                  f"({time.time() - t1:.0f} dtk)")
        boot = {a: macro_f1_resamples(y_true, p, idx, k) for a, p in preds.items()}
        point = {a: float(macro_f1_resamples(y_true, p, full_idx, k)[0]) for a, p in preds.items()}
        res: dict = {"point": {}, "diffs": {}}
        for fam, spec in families.items():
            c_name = spec["ctrl_name"]
            boot[c_name] = np.mean([boot[c] for c in spec["ctrl_arms"]], axis=0)
            point[c_name] = float(np.mean([point[c] for c in spec["ctrl_arms"]]))
            res["point"][f"{c_name}_sd_across_seeds"] = float(np.std([point[c] for c in spec["ctrl_arms"]], ddof=1)) \
                if len(spec["ctrl_arms"]) > 1 else float("nan")
            print(f"\n  [{fam}]")
            for name, (a, b) in spec["comparisons"].items():
                d = boot[a] - boot[b]
                lo, hi = np.percentile(d, [2.5, 97.5])
                est = point[a] - point[b]
                inc0 = bool(lo <= 0 <= hi)
                res["diffs"][name] = {"estimate": est, "ci95": [float(lo), float(hi)], "ci_includes_zero": inc0}
                rows.append({"model": m, "family": fam, "comparison": name, "estimate": round(est, 4),
                             "ci95_lo": round(float(lo), 4), "ci95_hi": round(float(hi), 4),
                             "ci_includes_zero": inc0, "tokens_test_a": round(tok.get(a, float("nan")), 1)})
                print(f"    {name:<24} {est:+.4f}  CI95% [{lo:+.4f}, {hi:+.4f}]  "
                      f"{'melewati 0' if inc0 else 'TIDAK melewati 0'}")
        res["point"].update({a: v for a, v in point.items()})
        report["models"][m] = res

    pd.DataFrame(rows).to_csv(out_dir / "tabel_sensitivitas_rq2.csv", index=False)
    report["elapsed_seconds"] = time.time() - t0
    (out_dir / "results_rq2_sens.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  Disimpan: {out_dir / 'results_rq2_sens.json'} dan tabel_sensitivitas_rq2.csv "
          f"({report['elapsed_seconds']:.0f} dtk)")
    return 0


def run_rq2_posctrl(args) -> int:
    """Suite KONTROL POSITIF SINTETIS RQ2 (MASTER SS14.2 langkah 4).

    Pertanyaan: kalau kontaminasi BENAR membocorkan label, apakah instrumen RQ2 (macro-F1; TAILCUT vs
    ref vs kontrol buang-blok-acak; bootstrap berpasangan) melihatnya? Tanpa ini, null RQ2 tidak bisa
    dibedakan dari instrumen yang buta.

    Desain: token `leaktoken_<Severity>` (label benar) disisipkan ke round(p x SEMUA dokumen) dokumen,
    dipilih hanya dari dokumen berpenanda POST_REPORT (jadi porsi dokumen berpenanda yang bocor ~p/0,86),
    di train DAN test; lalu suite RQ2 dijalankan ulang pada data bocor:
      tail      token di ekor (sesudah penanda pertama) -> TAILCUT membuangnya seluruhnya. Bila
                instrumen bekerja: leak > ref, tailcut < leak, dan tailcut < ctrl (kontrol acak
                menyisakan token itu pada sebagian besar dokumen bocor; porsinya dicatat di diagnostik).
      anywhere  token di posisi acak -> TAILCUT dan kontrol sama-sama membuangnya dengan peluang
                sebanding panjang yang dibuang. Menguji titik buta perbandingan tailcut - ctrl.
    Dosis p = 0 adalah data asli dan harus mereproduksi Tabel 3 (indeks resample & seed kontrol sama).
    Posisi blok kontrol per dokumen diambil dari undian p0 dan dipakai ulang di semua dosis, jadi
    kontrol BERPASANGAN antar dosis (dokumen yang tidak dibocorkan mendapat kontrol identik).
    Diagnostik per dosis: V(kehadiran token, Severity) -- bentuk V yang dipakai RQ1 --, V(identitas
    token, Severity), token bocor yang masuk kosakata TF-IDF, sisa token sesudah TAILCUT dan di kontrol.
    Keluaran: results_rq2_posctrl.json, tabel_kontrol_positif_rq2.csv, tabel_kontrol_positif_diag_rq2.csv.
    """
    import cleaning
    from verify_data import cramers_v

    doses = [("none", 0.0)] + [("tail", p) for p in args.leak_p] + [("anywhere", p) for p in args.leak_p_anywhere]
    keys = ["p0" if p == 0 else f"{mode}_p{p:g}" for mode, p in doses]
    if any(not 0 < p <= 1 for _, p in doses[1:]) or len(set(keys)) != len(keys):
        print(f"FATAL: dosis harus 0 < p <= 1 dan unik per mode; diterima {keys}")
        return 1

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_tr, ref_te = load_split(Path(args.ref_train), Path(args.ref_test))
    cls = SEVERITY_ORDER
    k = len(cls)
    to_int = {c: i for i, c in enumerate(cls)}
    y_train = ref_tr["Severity"].values
    y_true = np.array([to_int[v] for v in ref_te["Severity"].values])
    n = len(y_true)
    idx = np.random.default_rng(args.boot_seed).integers(0, n, size=(args.n_boot, n))
    full_idx = np.arange(n)[None, :]
    pat = cleaning.POST_REPORT
    tokens = [leak_token(c) for c in cls]
    sev_all = pd.Series(np.concatenate([ref_tr["Severity"].values, ref_te["Severity"].values]))
    ctrl_names = [f"ctrl_{s}" for s in range(args.seeds)]

    rule(f"RQ2 KONTROL POSITIF: train {len(ref_tr)} / test {n}; dosis tail {args.leak_p}; "
         f"anywhere {args.leak_p_anywhere}; seed kontrol {args.seeds}")
    # TAILCUT dihitung langsung (cleaning.tailcut_text); wajib identik dengan berkas yang dipakai Tabel 3.
    print("  memeriksa berkas plus_tailcut (peringatan NaN berikut, bila ada, milik berkas itu):")
    cut_tr, cut_te = load_split(Path(args.cut_train), Path(args.cut_test))
    for name, ref_d, cut_d in (("train", ref_tr, cut_tr), ("test", ref_te, cut_te)):
        if [cleaning.tailcut_text(t) for t in ref_d["Description"]] != list(cut_d["Description"]):
            print(f"FATAL: tailcut_text(ref {name}) tidak identik dengan berkas plus_tailcut -- "
                  "cleaning.POST_REPORT berubah sejak out_20261008 dibuat? Regenerasi dulu.")
            return 1
    print("  sanity: tailcut_text(ref) identik dengan berkas plus_tailcut di train dan test")
    del cut_tr, cut_te
    has_marker = np.array([pat.search(t) is not None for d in (ref_tr["Description"], ref_te["Description"])
                           for t in d])
    n_marker_tr = int(has_marker[: len(ref_tr)].sum())
    v_marker, chi_marker, _ = cramers_v(pd.Series(has_marker), sev_all)
    print(f"  acuan RQ1: V(ada penanda POST_REPORT, Severity) = {v_marker:.4f} (chi2 {chi_marker:.1f})")

    preds: dict[str, dict[str, dict[str, np.ndarray]]] = {m: {} for m in args.models}
    diag_rows = []
    base_cut, starts = None, {}
    for (mode, p), key in zip(doses, keys):
        l_tr, m_tr = apply_leak(ref_tr["Description"], ref_tr["Severity"].values, p, mode, args.leak_seed, pat)
        l_te, m_te = apply_leak(ref_te["Description"], ref_te["Severity"].values, p, mode, args.leak_seed + 1, pat)
        t_tr = [cleaning.tailcut_text(t) for t in l_tr]
        t_te = [cleaning.tailcut_text(t) for t in l_te]
        if key == "p0":
            base_cut = (t_tr, t_te)
            for s in range(args.seeds):
                rng = np.random.default_rng(s)             # seed & urutan undian sama dengan suite rq2 -> Tabel 3
                starts[s] = (random_block_starts(l_tr, t_tr, rng), random_block_starts(l_te, t_te, rng))
        reuse_cut = key != "p0" and t_tr == base_cut[0] and t_te == base_cut[1]
        ctrl = {f"ctrl_{s}": (drop_block_at(l_tr, t_tr, st[0]), drop_block_at(l_te, t_te, st[1]))
                for s, st in starts.items()}
        d = {"key": key, "mode": mode, "p": p, "n_leak_train": int(m_tr.sum()), "n_leak_test": int(m_te.sum()),
             "porsi_dok_berpenanda_train": float(m_tr.sum() / n_marker_tr), "tailcut_sama_dgn_p0": reuse_cut}

        rule(f"DOSIS {key}: bocor train {d['n_leak_train']} / test {d['n_leak_test']}")
        arms = [("leak", l_tr, l_te)] + ([] if reuse_cut else [("tailcut", t_tr, t_te)]) \
            + [(c, tr, te) for c, (tr, te) in ctrl.items()]
        for m in args.models:
            preds[m][key] = {}
        for a, tr_docs, te_docs in arms:
            t1 = time.time()
            out, vec = fit_predict_models(args.models, tr_docs, y_train, te_docs, args.max_features)
            for m, pm in out.items():
                preds[m][key][a] = np.array([to_int[v] for v in pm])
            if a == "leak" and p > 0:
                d["token_bocor_di_kosakata"] = sum(t in vec.vocabulary_ for t in tokens)
            print(f"  {a:<8} macro-F1 " + " | ".join(
                f"{m} {macro_f1_resamples(y_true, preds[m][key][a], full_idx, k)[0]:.4f}" for m in args.models)
                + f"  ({time.time() - t1:.0f} dtk)")
        if reuse_cut:
            for m in args.models:
                preds[m][key]["tailcut"] = preds[m]["p0"]["tailcut"]
        if p > 0:
            mask_all = pd.Series(np.concatenate([m_tr, m_te]))
            d["V_kehadiran"], d["chi2_kehadiran"], _ = cramers_v(mask_all, sev_all)
            d["V_identitas"], d["chi2_identitas"], _ = cramers_v(pd.Series(np.where(mask_all, sev_all, "none")),
                                                                 sev_all)
            leaked_te = np.flatnonzero(m_te)
            d["sisa_bocor_setelah_tailcut_test"] = float(np.mean([LEAK_PREFIX in t_te[i] for i in leaked_te]))
            d["bertahan_di_kontrol_test"] = float(np.mean([np.mean([LEAK_PREFIX in c[1][i] for i in leaked_te])
                                                          for c in ctrl.values()]))
            print(f"  diagnostik: V kehadiran {d['V_kehadiran']:.4f}, V identitas {d['V_identitas']:.4f}; "
                  f"token di kosakata {d['token_bocor_di_kosakata']}/{len(tokens)}; sisa sesudah TAILCUT "
                  f"{d['sisa_bocor_setelah_tailcut_test']:.1%}; bertahan di kontrol {d['bertahan_di_kontrol_test']:.1%}")
        diag_rows.append(d)
        del l_tr, l_te, t_tr, t_te, ctrl, arms, tr_docs, te_docs, out, vec   # base_cut menahan TAILCUT p0

    report: dict = {"config": vars(args), "n_train": len(ref_tr), "n_test": n,
                    "V_penanda_POST_REPORT": v_marker, "diagnostik": diag_rows, "models": {}}
    rows = []
    for m in args.models:
        rule(f"RINGKASAN {m}")
        ref_boot = macro_f1_resamples(y_true, preds[m]["p0"]["leak"], idx, k)
        ref_point = float(macro_f1_resamples(y_true, preds[m]["p0"]["leak"], full_idx, k)[0])
        report["models"][m] = {}
        for (mode, p), key in zip(doses, keys):
            pr = preds[m][key]
            boot = {a: macro_f1_resamples(y_true, v, idx, k) for a, v in pr.items()}
            point = {a: float(macro_f1_resamples(y_true, v, full_idx, k)[0]) for a, v in pr.items()}
            boot["ctrl"] = np.mean([boot[c] for c in ctrl_names], axis=0)      # rata-rata seed per resample
            point["ctrl"] = float(np.mean([point[c] for c in ctrl_names]))
            boot["ref"], point["ref"] = ref_boot, ref_point
            comps = {"tailcut - ref": ("tailcut", "ref"), "ctrl - ref": ("ctrl", "ref"),
                     "tailcut - ctrl": ("tailcut", "ctrl")} if p == 0 else \
                    {"leak - ref": ("leak", "ref"), "tailcut - leak": ("tailcut", "leak"),
                     "ctrl - leak": ("ctrl", "leak"), "tailcut - ctrl": ("tailcut", "ctrl")}
            res = {"point": {a: point[a] for a in ("ref", "leak", "tailcut", "ctrl")}, "diffs": {}}
            line = []
            for name, (a, b) in comps.items():
                diff = boot[a] - boot[b]
                lo, hi = np.percentile(diff, [2.5, 97.5])
                est = point[a] - point[b]
                inc0 = bool(lo <= 0 <= hi)
                res["diffs"][name] = {"estimate": est, "ci95": [float(lo), float(hi)], "ci_includes_zero": inc0}
                rows.append({"model": m, "mode": mode, "p": p, "comparison": name, "estimate": round(est, 4),
                             "ci95_lo": round(float(lo), 4), "ci95_hi": round(float(hi), 4),
                             "ci_includes_zero": inc0})
                line.append(f"{name} {est:+.4f} [{lo:+.4f},{hi:+.4f}]{'' if inc0 else '*'}")
            report["models"][m][key] = res
            print(f"  {key:<14} " + " | ".join(line))
        print("  (* = CI 95% tidak melewati 0)")

    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "tabel_kontrol_positif_rq2.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(out_dir / "tabel_kontrol_positif_diag_rq2.csv", index=False)
    t3 = out_dir / "tabel3_rq2.csv"
    if t3.exists():
        # suite rq2 menamai lengan kontrol "control", suite ini "ctrl"
        old = pd.read_csv(t3).replace({"comparison": {"control - ref": "ctrl - ref",
                                                      "tailcut - control": "tailcut - ctrl"}})
        new = tab[tab["p"] == 0][["model", "comparison", "estimate", "ci95_lo", "ci95_hi"]]
        mrg = old.merge(new, on=["model", "comparison"], suffixes=("_tabel3", "_p0"))
        if len(mrg):
            cols = ("estimate", "ci95_lo", "ci95_hi")
            gap = float(np.abs(mrg[[f"{c}_tabel3" for c in cols]].values
                               - mrg[[f"{c}_p0" for c in cols]].values).max())
            report["reproduksi_tabel3"] = {"selisih_maks": gap, "baris_dibandingkan": len(mrg)}
            print(f"\n  Reproduksi Tabel 3 oleh dosis p0: selisih maks {gap:.4f} pada {len(mrg)} baris")
        else:
            print("\n  Reproduksi Tabel 3: tidak ada baris yang cocok (model/perbandingan berbeda)")
    report["elapsed_seconds"] = time.time() - t0
    (out_dir / "results_rq2_posctrl.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"  Disimpan: {out_dir / 'results_rq2_posctrl.json'}, tabel_kontrol_positif_rq2.csv, "
          f"tabel_kontrol_positif_diag_rq2.csv ({report['elapsed_seconds']:.0f} dtk)")
    return 0


def build_initial_conditions(archive_dir: Path, pairs_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Bangun kondisi ref / laporan awal / TAILCUT untuk subset berpasangan (tingkat 'utama'), sejajar per baris.

    Semua kondisi diturunkan di memori dari teks tahap A (cleaning.stage_a_structure) sehingga `_orig_idx` tetap ada
    untuk split kronologis. Dokumen test yang laporan awalnya identik dengan laporan awal di train DIBUANG dari test
    untuk SEMUA kondisi (duplikat lintas split akan menguntungkan kondisi laporan awal saja).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cleaning as c

    df = c.stage_a_structure(pd.read_csv(Path(archive_dir) / "sev.csv"))
    n_awal = c.load_initial_pairs(Path(pairs_path))
    sub = df[df["_orig_idx"].isin(n_awal.index)].copy()
    sub["init"] = [c.initial_text(t, n) for t, n in zip(sub["Description"], sub["_orig_idx"].map(n_awal))]
    sub["tail"] = sub["Description"].map(c.tailcut_text)
    tr = sub[sub["_orig_idx"] <= c.TRAIN_MAX].reset_index(drop=True)
    te = sub[sub["_orig_idx"] >= c.TEST_MIN].reset_index(drop=True)
    dup = te["init"].isin(set(tr["init"]))
    info = {"n_tahap_A": len(df), "n_berpasangan": len(sub), "test_dibuang_duplikat_laporan_awal": int(dup.sum())}
    return tr, te[~dup].reset_index(drop=True), info


def run_rq2_initial(args) -> int:
    """Suite RQ2 dengan LAPORAN AWAL (K-3, 11 Okt 2026): kebenaran struktural dari bugs*.csv.

    Lengan (semua pada subset berpasangan yang sama, baris sejajar):
      ref             teks tahap A (laporan awal + semua komentar)
      initial         laporan awal saja (semua komentar susulan terbuang, di mana pun letaknya)
      tailcut         TAILCUT (pembanding: dekontaminasi leksikal)
      control_s       buang satu blok acak sepanjang token yang dibuang `initial` (kontrol panjang, --seeds seed)
      ref_to_initial  model dilatih pada ref, diuji pada laporan awal (kondisi deployment: saat triase hanya laporan
                      awal yang tersedia)
    Perbandingan berpasangan (bootstrap indeks test bersama): initial - ref, control - ref, initial - control,
    tailcut - ref, initial - tailcut, ref_to_initial - ref (selisih skor yang dilaporkan vs skor saat deployment),
    ref_to_initial - initial.
    """
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr, te, info = build_initial_conditions(Path(args.archive), Path(args.pairs))
    rule(f"RQ2 LAPORAN AWAL: train {len(tr)} / test {len(te)} (subset berpasangan); seed kontrol = {args.seeds}")
    print(f"  {info}")
    for col, name in (("Description", "ref"), ("init", "initial"), ("tail", "tailcut")):
        print(f"  token test {name:<8} rata-rata {te[col].str.split().str.len().mean():.1f}")

    cls = SEVERITY_ORDER
    k = len(cls)
    to_int = {cname: i for i, cname in enumerate(cls)}
    y_train = tr["Severity"].values
    y_true = np.array([to_int[v] for v in te["Severity"].values])
    n = len(y_true)
    idx = np.random.default_rng(args.boot_seed).integers(0, n, size=(args.n_boot, n))
    full_idx = np.arange(n)[None, :]
    comps = {"initial - ref": ("initial", "ref"), "control - ref": ("control", "ref"),
             "initial - control": ("initial", "control"), "tailcut - ref": ("tailcut", "ref"),
             "initial - tailcut": ("initial", "tailcut"), "ref_to_initial - ref": ("ref_to_initial", "ref"),
             "ref_to_initial - initial": ("ref_to_initial", "initial")}
    report: dict = {"config": vars(args), "info": info, "n_train": len(tr), "n_test": n, "models": {}}
    rows = []
    for m in args.models:
        rule(f"MODEL: {m}")
        preds: dict[str, np.ndarray] = {}

        def run_arm(name, tr_docs, te_docs):
            t1 = time.time()
            p = fit_predict(m, tr_docs, y_train, te_docs, args.max_features)
            preds[name] = np.array([to_int[v] for v in p])
            print(f"  {name:<15} macro-F1 = {macro_f1_resamples(y_true, preds[name], full_idx, k)[0]:.4f}"
                  f"   ({time.time() - t1:.0f} dtk)")

        run_arm("ref", tr["Description"], te["Description"])
        run_arm("initial", tr["init"], te["init"])
        run_arm("tailcut", tr["tail"], te["tail"])
        run_arm("ref_to_initial", tr["Description"], te["init"])
        for s in range(args.seeds):
            rng = np.random.default_rng(s)
            run_arm(f"control_{s}", drop_random_block(tr["Description"], tr["init"], rng),
                    drop_random_block(te["Description"], te["init"], rng))

        ctrl = [f"control_{s}" for s in range(args.seeds)]
        boot = {a: macro_f1_resamples(y_true, p, idx, k) for a, p in preds.items()}
        point = {a: float(macro_f1_resamples(y_true, p, full_idx, k)[0]) for a, p in preds.items()}
        boot["control"] = np.mean([boot[x] for x in ctrl], axis=0)
        point["control"] = float(np.mean([point[x] for x in ctrl]))
        res = {"point": {a: point[a] for a in ("ref", "initial", "tailcut", "control", "ref_to_initial")},
               "control_sd_across_seeds": float(np.std([point[x] for x in ctrl], ddof=1)) if args.seeds > 1 else None,
               "arms_ci95": {}, "diffs": {}}
        for a in res["point"]:
            lo, hi = np.percentile(boot[a], [2.5, 97.5])
            res["arms_ci95"][a] = [float(lo), float(hi)]
        print()
        for name, (a, b) in comps.items():
            d = boot[a] - boot[b]
            lo, hi = np.percentile(d, [2.5, 97.5])
            est = point[a] - point[b]
            inc0 = bool(lo <= 0 <= hi)
            res["diffs"][name] = {"estimate": float(est), "ci95": [float(lo), float(hi)], "ci_includes_zero": inc0}
            rows.append({"model": m, "comparison": name, "estimate": round(est, 4), "ci95_lo": round(float(lo), 4),
                         "ci95_hi": round(float(hi), 4), "ci_includes_zero": inc0})
            print(f"  {name:<26} {est:+.4f}  CI95% [{lo:+.4f}, {hi:+.4f}]  {'melewati 0' if inc0 else 'TIDAK melewati 0'}")
        report["models"][m] = res

    pd.DataFrame(rows).to_csv(out_dir / "tabel_laporan_awal_rq2.csv", index=False)
    report["elapsed_seconds"] = time.time() - t0
    (out_dir / "results_rq2_initial.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  Disimpan: {out_dir / 'results_rq2_initial.json'}, tabel_laporan_awal_rq2.csv "
          f"({report['elapsed_seconds']:.0f} dtk)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="archive/sev_train.csv")
    ap.add_argument("--test", default="archive/sev_test.csv")
    ap.add_argument("--out", default="experiment_baseline.json")
    ap.add_argument("--max-features", type=int, default=20000)
    ap.add_argument("--model", default="logreg", choices=["logreg", "linearsvc"],
                     help="Model untuk run tunggal (default logreg)")
    ap.add_argument("--suite", choices=["rq2", "rq2_sens", "rq2_posctrl", "rq2_initial"],
                     help="rq2: ref vs TAILCUT vs kontrol acak; rq2_sens: N token pertama/terakhir + TAILCUT_EXT; "
                          "rq2_posctrl: kontrol positif sintetis (token bocor-label tertanam); "
                          "rq2_initial: laporan awal dari bugs*.csv (kebenaran struktural)")
    ap.add_argument("--archive", default="archive", help="folder sev.csv untuk suite rq2_initial")
    ap.add_argument("--pairs", default="out_laporan_awal/pasangan_laporan_awal.csv",
                     help="pasangan laporan awal (kaggle_laporan_awal.py) untuk suite rq2_initial")
    ap.add_argument("--n-tokens", nargs="+", type=int, default=[50, 100], help="N untuk suite rq2_sens")
    ap.add_argument("--leak-p", nargs="+", type=float, default=[0.01, 0.02, 0.03, 0.05, 0.10, 0.20],
                     help="dosis kebocoran mode tail (proporsi dokumen) untuk suite rq2_posctrl")
    ap.add_argument("--leak-p-anywhere", nargs="*", type=float, default=[0.10, 0.20],
                     help="dosis kebocoran mode anywhere untuk suite rq2_posctrl")
    ap.add_argument("--leak-seed", type=int, default=20261009)
    ap.add_argument("--ref-train", default="out_20261008/cleaned_baseline_train.csv")
    ap.add_argument("--ref-test", default="out_20261008/cleaned_baseline_test.csv")
    ap.add_argument("--cut-train", default="out_20261008/cleaned_plus_tailcut_train.csv")
    ap.add_argument("--cut-test", default="out_20261008/cleaned_plus_tailcut_test.csv")
    ap.add_argument("--raw-train", default="archive/sev_train.csv", help="acuan reproduksi (opsional)")
    ap.add_argument("--raw-test", default="archive/sev_test.csv")
    ap.add_argument("--models", nargs="+", default=["logreg", "linearsvc"], choices=["logreg", "linearsvc"])
    ap.add_argument("--seeds", type=int, default=5, help="jumlah seed kontrol buang-blok-acak")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results_rq2")
    ap.add_argument("--target", default="Severity",
                     help="Kolom target: 'Severity' (6 kelas, sev.csv) atau 'Label' "
                          "(biner, fix.csv -- binarisasi Fixing_time, RQ5)")
    args = ap.parse_args()

    if args.suite == "rq2":
        return run_rq2(args)
    if args.suite == "rq2_sens":
        return run_rq2_sens(args)
    if args.suite == "rq2_posctrl":
        return run_rq2_posctrl(args)
    if args.suite == "rq2_initial":
        return run_rq2_initial(args)

    t0 = time.time()
    report: dict = {
        "config": {"train": args.train, "test": args.test, "max_features": args.max_features,
                    "target": args.target, "model": args.model},
    }

    # ---------------------------------------------------------- 1. muat data
    rule(f"1. MUAT DATA (split kronologis, target={args.target})")
    tr, te = load_split(Path(args.train), Path(args.test))
    print(f"  train: {len(tr)} baris  |  test: {len(te)} baris")
    print(f"  distribusi train: {tr[args.target].value_counts().to_dict()}")
    print(f"  distribusi test : {te[args.target].value_counts().to_dict()}")
    report["n_train"] = len(tr)
    report["n_test"] = len(te)

    y_train = tr[args.target].values
    y_test = te[args.target].values
    order = SEVERITY_ORDER if args.target == "Severity" else sorted(set(y_train) | set(y_test))

    # ------------------------------------------------- 2. baseline majority
    rule("2. BASELINE MAJORITY-CLASS (acuan minimum)")
    top = tr[args.target].value_counts().idxmax()
    y_majority = np.full_like(y_test, top)
    f1_majority = f1_score(y_test, y_majority, average="macro", zero_division=0)
    acc_majority = accuracy_score(y_test, y_majority)
    print(f"  selalu tebak '{top}': accuracy={acc_majority:.4f}  macro-F1={f1_majority:.4f}")
    report["baseline_majority"] = {"accuracy": acc_majority, "macro_f1": f1_majority}

    # --------------------------------------------------- 3. TF-IDF + fit train saja
    rule("3. TF-IDF (fit HANYA di train — no transductive leakage)")
    vec = make_vectorizer(args.max_features)
    X_train = vec.fit_transform(tr["Description"])
    X_test = vec.transform(te["Description"])
    print(f"  vocab size: {len(vec.vocabulary_)}")
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
    report["tfidf_vocab_size"] = len(vec.vocabulary_)

    # --------------------------------------------- 4. Logistic Regression (balanced)
    rule(f"4. MODEL {args.model.upper()} (class_weight=balanced, karena imbalance ~40:1)")
    clf = make_model(args.model)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    labels_present = [c for c in order if c in set(y_test)]
    per_class_f1 = f1_score(y_test, y_pred, average=None, labels=labels_present, zero_division=0)

    print(f"\n  accuracy  = {acc:.4f}")
    print(f"  macro-F1  = {macro_f1:.4f}")
    print(f"\n  per-kelas F1:")
    for cls, f1 in zip(labels_present, per_class_f1):
        marker = "" if f1 > 0 else "  <-- NOL, perlu perhatian"
        print(f"    {cls:<10} {f1:.4f}{marker}")

    report["model"] = {
        "type": f"TF-IDF + {type(clf).__name__}(class_weight=balanced)",
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": dict(zip([str(c) for c in labels_present], per_class_f1.tolist())),
    }

    # ----------------------------------------------------- 5. bootstrap CI
    rule("5. BOOTSTRAP CI 95% (1000 resample) untuk macro-F1")
    mean_f1, ci_lo, ci_hi = bootstrap_macro_f1(np.array(y_test), np.array(y_pred))
    print(f"  macro-F1 = {mean_f1:.4f}  CI95%=[{ci_lo:.4f}, {ci_hi:.4f}]")
    report["bootstrap"] = {"mean_macro_f1": mean_f1, "ci95_lo": ci_lo, "ci95_hi": ci_hi}

    # ------------------------------------------------------------ 6. Gate 1b
    rule("6. GATE 1b — apakah model melampaui acuan?")
    GATE_THRESHOLD = 0.25
    passed = macro_f1 >= GATE_THRESHOLD
    # Kriteria kelas minoritas: 'trivial' untuk Severity, kelas termarjinal (count terkecil)
    # untuk target lain seperti Label -- prinsipnya sama, jangan sampai F1 minoritas nol.
    minority_cls = "trivial" if args.target == "Severity" else \
        tr[args.target].value_counts().idxmin()
    minority_f1 = report["model"]["per_class_f1"].get(str(minority_cls), None)
    trivial_ok = minority_f1 is not None and minority_f1 > 0
    print(f"  Kriteria 1: macro-F1 >= {GATE_THRESHOLD} -> {'PASS' if passed else 'GAGAL'} "
          f"(aktual {macro_f1:.4f})")
    print(f"  Kriteria 2: F1 kelas minoritas '{minority_cls}' > 0 -> "
          f"{'PASS' if trivial_ok else 'GAGAL'} (aktual {minority_f1})")
    report["gate_1b"] = {
        "threshold_macro_f1": GATE_THRESHOLD,
        "passed_macro_f1": bool(passed),
        "passed_trivial_f1": bool(trivial_ok),
        "overall_pass": bool(passed and trivial_ok),
    }

    if passed and trivial_ok:
        print("\n  => GATE 1b LOLOS. Lanjut ke arm RQ2 (plus_decon vs raw/full_nodecon).")
    else:
        print("\n  => GATE 1b BELUM LOLOS dengan model paling sederhana.")
        print("     Ini BUKAN kegagalan proyek — dokumentasikan sebagai temuan, coba model")
        print("     lain (SVM, hybrid TF-IDF+embedding) sebelum menyimpulkan RQ tidak feasible.")

    # ----------------------------------------------------------- confusion
    rule("7. CONFUSION MATRIX (ringkas)")
    cm = confusion_matrix(y_test, y_pred, labels=labels_present)
    print(f"  {'':>10}" + "".join(f"{str(c)[:6]:>8}" for c in labels_present))
    for i, cls in enumerate(labels_present):
        print(f"  {str(cls):>10}" + "".join(f"{v:>8}" for v in cm[i]))

    elapsed = time.time() - t0
    report["elapsed_seconds"] = elapsed
    print(f"\n  Selesai dalam {elapsed:.1f} detik")

    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  Laporan disimpan: {args.out}")
    return 0 if (passed and trivial_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
