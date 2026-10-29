"""
task_uji_token_severity.py — Uji khusus: apakah kata severity di teks susulan menyumbang over-estimasi RQ2? (26 Okt 2026)

Latar: bobot LogReg memberi bobot besar pada kata `blocker/critical/major/minor/trivial`, dan sebagian besar kemunculannya
ada di teks susulan (task_asal_token.py). Uji ini mengukur seberapa banyak macro-F1 `ref` bergantung pada kata itu.

Desain (sama dengan `experiment.py --suite rq2_initial`: subset berpasangan utama, train `_orig_idx` <= 28.505,
TF-IDF 20k bigram fit pada train, LogReg C=1 dan LinearSVC C=0,5 bobot kelas seimbang, bootstrap berpasangan 1000x seed 42
dengan indeks resample sama untuk semua lengan). Aturan ditulis SEBELUM dijalankan; tidak diubah setelah hasil terlihat.

Teks susulan = token setelah n token pertama (n = panjang laporan awal, dari bugs*.csv).
Kata yang dibuang: token berbentuk trivial|minor|major|critical|blocker (tanpa beda huruf besar/kecil, tanda baca di tepi
token diabaikan), HANYA di teks susulan. `normal` TIDAK dibuang (kata umum); laporan awal tidak disentuh.

Lengan (semua latih+uji pada teks yang sama kecuali dinyatakan):
  ref              dokumen penuh
  strip_sev        dokumen penuh, kata severity di teks susulan dibuang (train dan test)
  ref_test_strip   latih pada ref, uji pada dokumen test yang kata severity-susulannya dibuang
  control_s        buang JUMLAH token yang sama (per dokumen) tetapi dipilih ACAK dari teks susulan (5 seed)
  initial          laporan awal saja (acuan; over-estimasi = ref - initial)
Perbandingan: strip_sev-ref, control-ref, strip_sev-control, ref_test_strip-ref.
Keluaran: results_uji_token_severity/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment as ex  # noqa: E402

SEV_RE = re.compile(r"[\W_]*(?:trivial|minor|major|critical|blocker)[\W_]*", re.I)
CLS = ex.SEVERITY_ORDER


def split_tail(desc: str, init: str) -> tuple[list[str], list[str]]:
    toks = desc.split()
    n = len(init.split())
    return toks[:n], toks[n:]


def strip_sev_tail(descs, inits) -> tuple[list[str], int]:
    out, removed = [], 0
    for d, i in zip(descs, inits):
        head, tail = split_tail(d, i)
        keep = [t for t in tail if not SEV_RE.fullmatch(t)]
        removed += len(tail) - len(keep)
        out.append(" ".join(head + keep))
    return out, removed


def random_tail_drop(descs, inits, counts, rng) -> list[str]:
    out = []
    for d, i, k in zip(descs, inits, counts):
        head, tail = split_tail(d, i)
        k = min(int(k), len(tail))
        if k > 0:
            drop = set(rng.choice(len(tail), size=k, replace=False).tolist())
            tail = [t for j, t in enumerate(tail) if j not in drop]
        out.append(" ".join(head + tail))
    return out


def per_doc_counts(descs, inits) -> list[int]:
    c = []
    for d, i in zip(descs, inits):
        _, tail = split_tail(d, i)
        c.append(sum(1 for t in tail if SEV_RE.fullmatch(t)))
    return c


def fit_predict_multi(model_name, tr_docs, y_train, te_sets: dict, max_features=20000) -> dict:
    vec = ex.make_vectorizer(max_features)
    X = vec.fit_transform(tr_docs)
    m = ex.make_model(model_name).fit(X, y_train)
    return {k: m.predict(vec.transform(v)) for k, v in te_sets.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(HERE / "archive"))
    ap.add_argument("--pairs", default=str(HERE / "out_laporan_awal" / "pasangan_laporan_awal.csv"))
    ap.add_argument("--out-dir", default=str(HERE / "results_uji_token_severity"))
    ap.add_argument("--models", nargs="+", default=["logreg", "linearsvc"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tr, te, info = ex.build_initial_conditions(Path(args.archive), Path(args.pairs))
    print(info, "| train", len(tr), "| test", len(te))
    tr_strip, rem_tr = strip_sev_tail(tr["Description"], tr["init"])
    te_strip, rem_te = strip_sev_tail(te["Description"], te["init"])
    cnt_tr = per_doc_counts(tr["Description"], tr["init"])
    cnt_te = per_doc_counts(te["Description"], te["init"])
    stats = {"token_dibuang_train": rem_tr, "token_dibuang_test": rem_te,
             "dok_terdampak_train": int(sum(c > 0 for c in cnt_tr)), "dok_terdampak_test": int(sum(c > 0 for c in cnt_te)),
             "pct_dok_terdampak_test": round(100 * sum(c > 0 for c in cnt_te) / len(te), 1)}
    print(stats)

    to_int = {c: i for i, c in enumerate(CLS)}
    y_train = tr["Severity"].values
    y_true = np.array([to_int[v] for v in te["Severity"].values])
    n, k = len(y_true), len(CLS)
    idx = np.random.default_rng(args.boot_seed).integers(0, n, size=(args.n_boot, n))
    full_idx = np.arange(n)[None, :]
    comps = {"strip_sev - ref": ("strip_sev", "ref"), "control - ref": ("control", "ref"),
             "strip_sev - control": ("strip_sev", "control"), "ref_test_strip - ref": ("ref_test_strip", "ref"),
             "initial - ref": ("initial", "ref")}
    rows, report = [], {"info": info, "stats": stats, "models": {}}
    for m in args.models:
        print(f"\n=== MODEL {m} ({time.time() - t0:.0f} dtk)")
        preds = {}
        conv = lambda p: np.array([to_int[v] for v in p])
        p = fit_predict_multi(m, tr["Description"], y_train, {"ref": te["Description"], "ref_test_strip": te_strip})
        preds["ref"], preds["ref_test_strip"] = conv(p["ref"]), conv(p["ref_test_strip"])
        preds["strip_sev"] = conv(fit_predict_multi(m, tr_strip, y_train, {"x": te_strip})["x"])
        preds["initial"] = conv(fit_predict_multi(m, tr["init"], y_train, {"x": te["init"]})["x"])
        for s in range(args.seeds):
            rng = np.random.default_rng(s)
            tr_c = random_tail_drop(tr["Description"], tr["init"], cnt_tr, rng)
            te_c = random_tail_drop(te["Description"], te["init"], cnt_te, rng)
            preds[f"control_{s}"] = conv(fit_predict_multi(m, tr_c, y_train, {"x": te_c})["x"])
        ctrl = [f"control_{s}" for s in range(args.seeds)]
        boot = {a: ex.macro_f1_resamples(y_true, pr, idx, k) for a, pr in preds.items()}
        point = {a: float(ex.macro_f1_resamples(y_true, pr, full_idx, k)[0]) for a, pr in preds.items()}
        boot["control"] = np.mean([boot[c] for c in ctrl], axis=0)
        point["control"] = float(np.mean([point[c] for c in ctrl]))
        print({a: round(point[a], 4) for a in ("ref", "strip_sev", "control", "ref_test_strip", "initial")})
        res = {"point": {a: point[a] for a in ("ref", "strip_sev", "control", "ref_test_strip", "initial")}, "diffs": {}}
        for name, (a, b) in comps.items():
            d = boot[a] - boot[b]
            lo, hi = np.percentile(d, [2.5, 97.5])
            est = point[a] - point[b]
            inc0 = bool(lo <= 0 <= hi)
            res["diffs"][name] = {"estimate": est, "ci95": [float(lo), float(hi)], "ci_includes_zero": inc0}
            rows.append({"model": m, "comparison": name, "estimate": round(est, 4), "ci95_lo": round(float(lo), 4),
                         "ci95_hi": round(float(hi), 4), "ci_includes_zero": inc0})
            print(f"  {name:<22} {est:+.4f} [{lo:+.4f}, {hi:+.4f}] {'melewati 0' if inc0 else 'TIDAK melewati 0'}")
        over = point["ref"] - point["initial"]
        res["porsi_over_estimasi_dari_kata_severity_pct"] = round(100 * (point["ref"] - point["strip_sev"]) / over, 1) if over else None
        report["models"][m] = res
        print("  porsi over-estimasi (ref-initial) yang dijelaskan strip_sev:", res["porsi_over_estimasi_dari_kata_severity_pct"], "%")

    pd.DataFrame(rows).to_csv(out / "tabel_uji_token_severity.csv", index=False)
    report["elapsed_seconds"] = time.time() - t0
    (out / "results_uji_token_severity.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nselesai ({report['elapsed_seconds']:.0f} dtk) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
