"""
task_kata_kunci_k3.py — Pembanding kata kunci K3 dipilih otomatis dari data latih (26 Okt 2026). PASCA-HASIL.

Status: uji sensitivitas yang ditambahkan SETELAH hasil K1/K2 dan analisis token terlihat. K1/K2 (aturan a priori,
task_uji_ketahanan_rq2.py) TIDAK diubah. K3 bukan pengganti K1/K2.

Prosedur (ditetapkan sebelum dijalankan, disetujui peneliti 26 Okt; tidak diubah setelah hasil terlihat):
  1. Data: HANYA train (`_orig_idx` <= 28.505, subset berpasangan utama), HANYA laporan awal (`init`). Test tidak dipakai
     untuk memilih kata.
  2. Kandidat: unigram + bigram (huruf kecil, tokenisasi sama dengan TF-IDF proyek), muncul di >= 20 dokumen train.
  3. Untuk tiap kelas kecuali `normal`: 10 kandidat dengan log-odds ratio tertinggi (kehadiran kata di kelas itu vs
     semua kelas lain, smoothing +0,5).
  4. Aturan: cek berurutan blocker -> critical -> major -> minor -> trivial; dokumen yang memuat salah satu kata kelas
     itu diberi kelas tersebut; tanpa kecocokan -> `normal`.
  5. Uji sekali pada laporan awal test (4.209 dokumen). Bootstrap berpasangan 1000x seed 42 (indeks sama dengan §8f).
Pembanding berpasangan: LogReg/LinearSVC lengan `initial` dan `ref_to_initial` (dilatih ulang, harus mereproduksi §8f),
dan K2 (aturan a priori).

Keluaran: results_kata_kunci_k3/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment as ex  # noqa: E402
import task_uji_ketahanan_rq2 as rob  # noqa: E402

N_KATA, MIN_DOK, SMOOTH = 10, 20, 0.5
URUTAN = ["blocker", "critical", "major", "minor", "trivial"]
CLS, K, TO_INT = rob.CLS, rob.K, rob.TO_INT
ACUAN = {"logreg": {"initial": 0.2752, "ref_to_initial": 0.2724},
         "linearsvc": {"initial": 0.2632, "ref_to_initial": 0.2533}}
ACUAN_K2_INITIAL = 0.2436


def pilih_kata(init_train: pd.Series, y_train: np.ndarray) -> pd.DataFrame:
    cv = CountVectorizer(ngram_range=(1, 2), min_df=MIN_DOK, binary=True)
    X = cv.fit_transform(init_train)
    vocab = np.array(cv.get_feature_names_out())
    rows = []
    for c in URUTAN:
        in_c = y_train == c
        n_c, n_r = int(in_c.sum()), int((~in_c).sum())
        df_c = np.asarray(X[in_c].sum(axis=0)).ravel()
        df_r = np.asarray(X[~in_c].sum(axis=0)).ravel()
        lor = (np.log((df_c + SMOOTH) / (n_c - df_c + SMOOTH))
               - np.log((df_r + SMOOTH) / (n_r - df_r + SMOOTH)))
        for rank, j in enumerate(np.argsort(-lor)[:N_KATA], start=1):
            rows.append({"kelas": c, "rank": rank, "kata": vocab[j], "dok_kelas_train": int(df_c[j]),
                         "dok_total_train": int(df_c[j] + df_r[j]), "log_odds": round(float(lor[j]), 3)})
    return pd.DataFrame(rows), cv.build_analyzer()


def prediksi_k3(docs: pd.Series, kata: pd.DataFrame, analyzer) -> tuple[np.ndarray, float]:
    per_kelas = {c: set(kata.loc[kata["kelas"] == c, "kata"]) for c in URUTAN}
    out = np.full(len(docs), TO_INT["normal"])
    cocok = 0
    for i, t in enumerate(docs):
        f = set(analyzer(t))
        for c in URUTAN:
            if f & per_kelas[c]:
                out[i] = TO_INT[c]
                cocok += 1
                break
    return out, 100 * cocok / len(docs)


def fit_pred(model, tr_docs, y_train, te_docs) -> np.ndarray:
    vec = ex.make_vectorizer(20000)
    m = ex.make_model(model).fit(vec.fit_transform(tr_docs), y_train)
    return np.array([TO_INT[v] for v in m.predict(vec.transform(te_docs))])


def main() -> int:
    t0 = time.time()
    out = HERE / "results_kata_kunci_k3"
    out.mkdir(exist_ok=True)
    tr, te, info = ex.build_initial_conditions(HERE / "archive", HERE / "out_laporan_awal" / "pasangan_laporan_awal.csv")
    print(info, "| train", len(tr), "| test", len(te))
    y_train = tr["Severity"].values
    y_true = np.array([TO_INT[v] for v in te["Severity"].values])
    n = len(y_true)
    idx = np.random.default_rng(42).integers(0, n, size=(1000, n))
    full = np.arange(n)[None, :]

    kata, analyzer = pilih_kata(tr["init"], y_train)
    kata.to_csv(out / "kata_k3.csv", index=False)
    for c in URUTAN:
        print(f"  {c:<9}", ", ".join(kata.loc[kata["kelas"] == c, "kata"]))

    preds, cakupan = {}, {}
    preds["K3"], cakupan["K3"] = prediksi_k3(te["init"], kata, analyzer)
    preds["K2"] = rob.keyword_predict(te["init"], rob.KEYWORD_RULES["kata_kunci_K2"])
    cakupan["K2"] = round(100 * float((preds["K2"] != TO_INT["normal"]).mean()), 1)
    for m in ("logreg", "linearsvc"):
        preds[f"{m}|initial"] = fit_pred(m, tr["init"], y_train, te["init"])
        preds[f"{m}|ref_to_initial"] = fit_pred(m, tr["Description"], y_train, te["init"])
        print(f"  {m} selesai ({time.time() - t0:.0f} dtk)")

    f1b = {a: ex.macro_f1_resamples(y_true, p, idx, K) for a, p in preds.items()}
    f1p = {a: float(ex.macro_f1_resamples(y_true, p, full, K)[0]) for a, p in preds.items()}
    qwb = {a: rob.qwk_resamples(y_true, p, idx) for a, p in preds.items()}
    qwp = {a: float(rob.qwk_resamples(y_true, p, full)[0]) for a, p in preds.items()}

    cek = {"K2_initial": (round(f1p["K2"], 4), ACUAN_K2_INITIAL)}
    for m, d in ACUAN.items():
        for arm, ref in d.items():
            cek[f"{m}|{arm}"] = (round(f1p[f"{m}|{arm}"], 4), ref)
    reproduksi_ok = all(abs(a - b) <= 0.0006 for a, b in cek.values())
    print("  reproduksi:", cek, "OK" if reproduksi_ok else "TIDAK COCOK")

    skor = []
    for a in preds:
        lo, hi = np.percentile(f1b[a], [2.5, 97.5])
        skor.append({"prediktor": a, "macro_f1": round(f1p[a], 4), "ci_lo": round(float(lo), 4),
                     "ci_hi": round(float(hi), 4), "qwk": round(qwp[a], 4), "cakupan_aturan_pct":
                     round(cakupan[a], 1) if a in cakupan else None})
    pd.DataFrame(skor).to_csv(out / "skor_k3.csv", index=False)

    rows = []
    for a in ("logreg|initial", "logreg|ref_to_initial", "linearsvc|initial", "linearsvc|ref_to_initial", "K2"):
        for metrik, b, p in (("macro_f1", f1b, f1p), ("qwk", qwb, qwp)):
            d = b[a] - b["K3"]
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append({"metrik": metrik, "perbandingan": f"{a} - K3", "selisih": round(p[a] - p["K3"], 4),
                         "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                         "ci_melewati_0": bool(lo <= 0 <= hi)})
    sel = pd.DataFrame(rows)
    sel.to_csv(out / "selisih_vs_k3.csv", index=False)
    print(pd.DataFrame(skor).to_string(index=False))
    print(sel.to_string(index=False))

    (out / "ringkasan_k3.json").write_text(json.dumps({
        "status": "PASCA-HASIL (sensitivitas); K1/K2 tidak diubah",
        "prosedur": {"n_kata_per_kelas": N_KATA, "min_dokumen_train": MIN_DOK, "smoothing": SMOOTH,
                     "urutan": URUTAN, "sumber_kata": "train, laporan awal"},
        "info": info, "reproduksi": {k: list(v) for k, v in cek.items()}, "reproduksi_ok": reproduksi_ok,
        "detik": round(time.time() - t0)}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"selesai ({time.time() - t0:.0f} dtk) -> {out}")
    return 0 if reproduksi_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
