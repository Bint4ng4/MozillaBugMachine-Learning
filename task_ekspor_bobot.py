"""
task_ekspor_bobot.py — Ekspor bobot model LogReg (TF-IDF 20k bigram) untuk Gambar 2 Paper 1 (26 Okt 2026).

Memakai komponen yang sama dengan `experiment.py --suite rq2_initial`: subset berpasangan utama, train
`_orig_idx` <= 28.505, TF-IDF fit pada train (kolom `Description` = lengan ref), LogReg C=1 bobot kelas seimbang.
Tidak ada penyetelan. Skrip memeriksa dulu bahwa macro-F1 ref dan ref_to_initial cocok dengan §8f (0,3075 dan 0,2724);
bila tidak cocok, tidak ada yang diekspor.

Keluaran (results_bobot_logreg/):
  bobot_per_kelas.csv   semua token x 6 kelas (koefisien LogReg)
  top30_per_kelas.csv   30 token berkoefisien terbesar per kelas
  top30_global.csv      30 token dengan |koefisien| maksimum antar kelas (+ kelas pemilik)
  ringkasan.json        pemeriksaan reproduksi
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment as ex  # noqa: E402

REF_F1, REF2INIT_F1, TOL = 0.3075, 0.2724, 0.0006


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(HERE / "archive"))
    ap.add_argument("--pairs", default=str(HERE / "out_laporan_awal" / "pasangan_laporan_awal.csv"))
    ap.add_argument("--out-dir", default=str(HERE / "results_bobot_logreg"))
    args = ap.parse_args()

    tr, te, info = ex.build_initial_conditions(Path(args.archive), Path(args.pairs))
    print(info, "| train", len(tr), "| test", len(te))

    vec = ex.make_vectorizer(20000)
    X_tr = vec.fit_transform(tr["Description"])
    model = ex.make_model("logreg").fit(X_tr, tr["Severity"])

    f1_ref = f1_score(te["Severity"], model.predict(vec.transform(te["Description"])), average="macro")
    f1_r2i = f1_score(te["Severity"], model.predict(vec.transform(te["init"])), average="macro")
    print(f"macro-F1 ref={f1_ref:.4f} (acuan {REF_F1}) | ref_to_initial={f1_r2i:.4f} (acuan {REF2INIT_F1})")
    ok = abs(f1_ref - REF_F1) <= TOL and abs(f1_r2i - REF2INIT_F1) <= TOL
    if not ok:
        print("TIDAK COCOK dengan §8f; tidak ada yang diekspor.")
        return 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    classes = list(model.classes_)
    vocab = np.array(vec.get_feature_names_out())
    coef = pd.DataFrame(model.coef_.T, index=vocab, columns=classes)
    coef.index.name = "token"
    coef.to_csv(out / "bobot_per_kelas.csv", float_format="%.5f")

    rows = []
    for c in classes:
        top = coef[c].sort_values(ascending=False).head(30)
        rows += [{"kelas": c, "rank": i + 1, "token": t, "koefisien": round(float(w), 5)}
                 for i, (t, w) in enumerate(top.items())]
    pd.DataFrame(rows).to_csv(out / "top30_per_kelas.csv", index=False)

    best = coef.abs().max(axis=1).sort_values(ascending=False).head(30)
    glob = [{"rank": i + 1, "token": t, "kelas_pemilik": coef.loc[t].abs().idxmax(),
             "koefisien": round(float(coef.loc[t, coef.loc[t].abs().idxmax()]), 5)}
            for i, t in enumerate(best.index)]
    pd.DataFrame(glob).to_csv(out / "top30_global.csv", index=False)

    (out / "ringkasan.json").write_text(json.dumps({
        "train": len(tr), "test": len(te), "n_fitur": int(len(vocab)), "kelas": classes,
        "macro_f1_ref": round(f1_ref, 4), "macro_f1_ref_to_initial": round(f1_r2i, 4), "info": info,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("selesai ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
