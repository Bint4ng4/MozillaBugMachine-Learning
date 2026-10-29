"""
task_perbaiki_gap.py — Menurunkan gap macro-F1 train–val (permintaan user 12 Okt 2026), lanjutan task_kurva_belajar.py.

Data utama: full_initial (laporan awal + cleaning lengkap). Split sama: train_fit < 25.000 | val 25.000–28.505 |
test >= 28.507 (indeks asli). Penyetelan HANYA memakai val; test dipakai sekali di akhir untuk konfigurasi terpilih.

Ruang coba (semua class_weight seimbang, vectorizer di-fit pada train_fit saja):
  fitur    max_features {2.000, 5.000, 20.000} x n-gram {(1,1), (1,2)} x sublinear_tf {False, True}, min_df = 2
  model    LogReg C {0,01; 0,03; 0,1; 0,3; 1} | LinearSVC C {0,003; 0,01; 0,03; 0,1; 0,5} |
           SGD log loss alpha {1e-4; 3e-4; 1e-3; 3e-3; 1e-2} + early stopping (maks 50, patience 5) |
           ComplementNB alpha {0,1; 0,3; 1} | LSA (TruncatedSVD 100/300 dim pada 20k bigram) + LogReg C {0,1; 1; 10}

Aturan pilih (ditetapkan sebelum melihat hasil):
  A  val macro-F1 tertinggi di antara konfigurasi dengan gap <= 5 poin
  B  gap terkecil di antara konfigurasi yang val macro-F1-nya >= (val terbaik - 1 poin)
  Acuan = konfigurasi lama (20k bigram, LogReg C=1 / LinearSVC C=0,5 / SGD alpha 1e-4).

Pakai:  python task_perbaiki_gap.py [--out-dir results_task_20261012_gap]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm as normal_dist  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.naive_bayes import ComplementNB  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import Normalizer  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cleaning as c  # noqa: E402
import task_kurva_belajar as tk  # noqa: E402
from experiment import SEVERITY_ORDER, macro_f1_resamples  # noqa: E402

FEATS = [(mf, ng, sub) for mf in (2000, 5000, 20000) for ng in ((1, 1), (1, 2)) for sub in (False, True)]
BASE_FEAT = (20000, (1, 2), False)
GRID = {"logreg": [0.01, 0.03, 0.1, 0.3, 1.0], "linearsvc": [0.003, 0.01, 0.03, 0.1, 0.5],
        "sgd": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], "cnb": [0.1, 0.3, 1.0]}
BASE_REG = {"logreg": 1.0, "linearsvc": 0.5, "sgd": 1e-4}
LSA_DIMS, LSA_C = (100, 300), (0.1, 1.0, 10.0)
FAMILY_LABEL = {"logreg": "Logistic Regression", "linearsvc": "LinearSVC", "sgd": "SGD (log loss)",
                "cnb": "Complement NB", "lsa_logreg": "LSA + LogReg"}


def mf1(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


def vectorizer(feat):
    mf, ng, sub = feat
    return TfidfVectorizer(max_features=mf, ngram_range=ng, min_df=2, sublinear_tf=sub)


def fit_model(family, reg, Xtr, ytr, Xva, yva, seed=0, max_epochs=50, patience=5):
    """Kembalikan (model terlatih, info tambahan). SGD memakai early stopping di val."""
    if family == "logreg":
        return LogisticRegression(C=reg, max_iter=3000, class_weight="balanced").fit(Xtr, ytr), {}
    if family == "linearsvc":
        return LinearSVC(C=reg, class_weight="balanced", random_state=0).fit(Xtr, ytr), {}
    if family == "cnb":
        return ComplementNB(alpha=reg).fit(Xtr, ytr), {}
    if family == "sgd":
        clf, h, be = tk.train_sgd(Xtr, ytr, Xva, yva, seed, max_epochs, patience, alpha=reg)
        return clf, {"best_epoch": be, "epoch_dijalankan": int(h["epoch"].max()), "history": h}
    raise ValueError(family)


def feat_label(feat):
    mf, ng, sub = feat
    return f"{mf // 1000}k {'uni' if ng == (1, 1) else 'bi'}{' sub' if sub else ''}"


def plot_pareto(res: pd.DataFrame, picks: dict, base_rows: pd.DataFrame, path: Path) -> None:
    fams = list(FAMILY_LABEL)
    fig, ax = plt.subplots(1, len(fams), figsize=(18, 4.4), sharex=True, sharey=True)
    for a, fam in zip(ax, fams):
        d = res[res["family"] == fam]
        a.scatter(d["gap_poin"], d["f1_val"], s=22, color=tk.COL["raw"], alpha=0.65, linewidths=0)
        b = base_rows[base_rows["family"] == fam]
        if len(b):
            a.scatter(b["gap_poin"], b["f1_val"], s=60, facecolors="none", edgecolors=tk.INK, linewidths=1.4)
            a.annotate("acuan lama", (b["gap_poin"].iloc[0], b["f1_val"].iloc[0]), xytext=(6, -10),
                       textcoords="offset points", fontsize=8, color=tk.INK2)
        for name, r in picks.items():
            if r is not None and r["family"] == fam:
                a.scatter([r["gap_poin"]], [r["f1_val"]], s=70, color=tk.COL["full_tailcut"], zorder=3)
                a.annotate(f"pilihan {name}", (r["gap_poin"], r["f1_val"]), xytext=(6, 4),
                           textcoords="offset points", fontsize=8, color=tk.INK)
        a.axvline(5, color=tk.INK2, linewidth=0.8, linestyle=":")
        a.set_title(FAMILY_LABEL[fam])
        a.set_xlabel("gap train − val (poin macro-F1)")
    ax[0].set_ylabel("macro-F1 val")
    fig.suptitle("Gap vs skor val per konfigurasi (full_initial) — garis titik = batas gap 5 poin; "
                 "lingkaran = konfigurasi lama; titik oranye = pilihan A/B", fontsize=11, color=tk.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sgd_compare(hists: dict, path: Path) -> None:
    fig, ax = plt.subplots(2, len(hists), figsize=(5.5 * len(hists), 7), sharey="row", squeeze=False)
    for j, (title, h) in enumerate(hists.items()):
        for i, (metric, ylab) in enumerate((("loss", "log loss (berbobot kelas)"), ("f1", "macro-F1"))):
            a = ax[i, j]
            for part in ("train", "val"):
                a.plot(h["epoch"], h[f"{metric}_{part}"], color=tk.COL[part], linewidth=2, label=part)
                a.annotate(part, (h["epoch"].iloc[-1], h[f"{metric}_{part}"].iloc[-1]), xytext=(4, 0),
                           textcoords="offset points", va="center", fontsize=9, color=tk.INK2)
            a.xaxis.set_major_locator(tk.MaxNLocator(integer=True))
            if i == 0:
                a.set_title(title)
            if j == 0:
                a.set_ylabel(ylab)
            if i == 1:
                a.set_xlabel("epoch")
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.96))
    fig.suptitle("SGD per epoch (seed 0): konfigurasi lama vs konfigurasi SGD terbaik", fontsize=11, color=tk.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="archive")
    ap.add_argument("--out-dir", default="results_task_20261012_gap")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, default=5, help="seed SGD untuk evaluasi akhir")
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    df0 = pd.read_csv(Path(args.archive) / "sev.csv")
    arms = {name: tk.quiet(c.run_config, df0, name, stages)[0] for name, stages in tk.ARMS.items()}
    sp = tk.split3(arms["full_initial"])
    txt = {k: v["Description"].fillna("").astype(str) for k, v in sp.items()}
    y = {k: v["Severity"].values for k, v in sp.items()}
    print(f"full_initial: train_fit {len(y['train'])} | val {len(y['val'])} | test {len(y['test'])}")

    # ----------------------------------------------------------------------------- penyetelan (train_fit + val)
    rows = []
    for feat in FEATS:
        vec = vectorizer(feat)
        Xtr, Xva = vec.fit_transform(txt["train"]), vec.transform(txt["val"])
        for fam, regs in GRID.items():
            for reg in regs:
                t1 = time.time()
                m, info = fit_model(fam, reg, Xtr, y["train"], Xva, y["val"])
                ftr, fva = mf1(y["train"], m.predict(Xtr)), mf1(y["val"], m.predict(Xva))
                rows.append({"family": fam, "fitur": feat_label(feat), "max_features": feat[0], "ngram": str(feat[1]),
                             "sublinear": feat[2], "reg": reg, "f1_train": ftr, "f1_val": fva,
                             "gap_poin": 100 * (ftr - fva), "best_epoch": info.get("best_epoch"),
                             "detik": round(time.time() - t1, 1)})
        print(f"  fitur {feat_label(feat):<12} selesai ({time.time() - t0:.0f} dtk)")
    # LSA + LogReg (fitur padat berdimensi rendah)
    vec = vectorizer(BASE_FEAT)
    Xtr, Xva = vec.fit_transform(txt["train"]), vec.transform(txt["val"])
    for k in LSA_DIMS:
        lsa = make_pipeline(TruncatedSVD(n_components=k, random_state=0), Normalizer(copy=False))
        Ztr, Zva = lsa.fit_transform(Xtr), lsa.transform(Xva)
        for C in LSA_C:
            m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Ztr, y["train"])
            ftr, fva = mf1(y["train"], m.predict(Ztr)), mf1(y["val"], m.predict(Zva))
            rows.append({"family": "lsa_logreg", "fitur": f"20k bi LSA{k}", "max_features": 20000, "ngram": "(1, 2)",
                         "sublinear": False, "reg": C, "lsa_dim": k, "f1_train": ftr, "f1_val": fva,
                         "gap_poin": 100 * (ftr - fva), "best_epoch": None, "detik": None})
    res = pd.DataFrame(rows)
    res.to_csv(out / "penyetelan_val.csv", index=False)

    is_base = (res["fitur"] == feat_label(BASE_FEAT)) & res.apply(
        lambda r: r["family"] in BASE_REG and np.isclose(r["reg"], BASE_REG[r["family"]]), axis=1)
    base_rows = res[is_base]
    best_val = res["f1_val"].max()
    cand_a = res[res["gap_poin"] <= 5]
    pick_a = cand_a.loc[cand_a["f1_val"].idxmax()] if len(cand_a) else None
    cand_b = res[res["f1_val"] >= best_val - 0.01]
    pick_b = cand_b.loc[cand_b["gap_poin"].idxmin()]
    picks = {"A": pick_a, "B": pick_b}
    print("\nAcuan lama (val):")
    print(base_rows[["family", "fitur", "reg", "f1_train", "f1_val", "gap_poin"]].to_string(index=False))
    for k, r in picks.items():
        print(f"Pilihan {k}: " + ("tidak ada konfigurasi dengan gap <= 5 poin" if r is None else
              f"{r['family']} {r['fitur']} reg={r['reg']} | train {r['f1_train']:.3f} val {r['f1_val']:.3f} gap {r['gap_poin']:.1f}"))
    plot_pareto(res, picks, base_rows, out / "pareto_gap_vs_val.png")

    # ----------------------------------------------------------------------------- evaluasi akhir (test, sekali)
    def build(row):
        feat = (int(row["max_features"]), tuple(int(v) for v in row["ngram"].strip("()").split(",")), bool(row["sublinear"]))
        return feat

    final_specs = {f"acuan_{r['family']}": r for _, r in base_rows.iterrows()}
    for k, r in picks.items():
        if r is not None:
            final_specs[f"pilihan_{k}"] = r
    to_int = {k: i for i, k in enumerate(SEVERITY_ORDER)}
    K = len(SEVERITY_ORDER)
    preds, final_rows, sgd_hist = {}, [], {}

    def run_spec(name, r, arm_split):
        trn, val, tst = (arm_split[k]["Description"].fillna("").astype(str) for k in ("train", "val", "test"))
        ytr, yva = arm_split["train"]["Severity"].values, arm_split["val"]["Severity"].values
        feat = build(r)
        vec = vectorizer(feat)
        Xtr, Xva, Xte = vec.fit_transform(trn), vec.transform(val), vec.transform(tst)
        if r["family"] == "lsa_logreg":
            lsa = make_pipeline(TruncatedSVD(n_components=int(r["lsa_dim"]), random_state=0), Normalizer(copy=False))
            Xtr, Xva, Xte = lsa.fit_transform(Xtr), lsa.transform(Xva), lsa.transform(Xte)
            ms = [LogisticRegression(C=r["reg"], max_iter=3000, class_weight="balanced").fit(Xtr, ytr)]
            hist = None
        elif r["family"] == "sgd":
            ms, hist = [], None
            for s in range(args.seeds):
                m, info = fit_model("sgd", r["reg"], Xtr, ytr, Xva, yva, seed=s)
                ms.append(m)
                hist = info["history"] if s == 0 else hist
        else:
            ms, hist = [fit_model(r["family"], r["reg"], Xtr, ytr, Xva, yva)[0]], None
        out_p = [pd.Series(m.predict(Xte), index=arm_split["test"]["_orig_idx"].values) for m in ms]
        trv = [(mf1(ytr, m.predict(Xtr)), mf1(yva, m.predict(Xva))) for m in ms]
        return out_p, trv, hist

    for name, r in final_specs.items():
        P, trv, hist = run_spec(name, r, sp)
        preds[name] = P
        if hist is not None:
            sgd_hist[name] = hist
        yt = y["test"]
        yi = np.array([to_int[v] for v in yt])
        idx = np.random.default_rng(args.boot_seed).integers(0, len(yt), size=(args.n_boot, len(yt)))
        Pi = [np.array([to_int[v] for v in p.values]) for p in P]
        pts = [float(macro_f1_resamples(yi, p, np.arange(len(yt))[None, :], K)[0]) for p in Pi]
        boots = np.mean([macro_f1_resamples(yi, p, idx, K) for p in Pi], axis=0)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        f1c = np.mean([f1_score(yt, p.values, labels=SEVERITY_ORDER, average=None, zero_division=0) for p in P], axis=0)
        final_rows.append({"konfigurasi": name, "family": r["family"], "fitur": r["fitur"], "reg": r["reg"],
                           "f1_train": round(float(np.mean([a for a, _ in trv])), 4),
                           "f1_val": round(float(np.mean([b for _, b in trv])), 4),
                           "f1_test": round(float(np.mean(pts)), 4), "ci95_lo": round(float(lo), 4),
                           "ci95_hi": round(float(hi), 4),
                           "gap_train_val_poin": round(100 * float(np.mean([a - b for a, b in trv])), 1),
                           "gap_train_test_poin": round(100 * float(np.mean([a for a, _ in trv]) - np.mean(pts)), 1),
                           "rentang_seed_poin": round(100 * (max(pts) - min(pts)), 2),
                           **{f"f1_{k}": round(float(v), 4) for k, v in zip(SEVERITY_ORDER, f1c)}})
        print(f"  test {name:<18} macro-F1 {np.mean(pts):.4f} [{lo:.4f}, {hi:.4f}]")
    final = pd.DataFrame(final_rows)
    final.to_csv(out / "evaluasi_test.csv", index=False)

    # pilihan vs acuan keluarga yang sama dan vs acuan SGD lama (bootstrap berpasangan + McNemar)
    yt = y["test"]
    yi = np.array([to_int[v] for v in yt])
    idx = np.random.default_rng(args.boot_seed).integers(0, len(yt), size=(args.n_boot, len(yt)))
    cmp_rows = []
    for k in ("A", "B"):
        if f"pilihan_{k}" not in preds:
            continue
        for ref in [x for x in preds if x.startswith("acuan_")]:
            Pa = [np.array([to_int[v] for v in p.values]) for p in preds[f"pilihan_{k}"]]
            Pb = [np.array([to_int[v] for v in p.values]) for p in preds[ref]]
            ba = np.mean([macro_f1_resamples(yi, p, idx, K) for p in Pa], axis=0)
            bb = np.mean([macro_f1_resamples(yi, p, idx, K) for p in Pb], axis=0)
            lo, hi = np.percentile(ba - bb, [2.5, 97.5])
            ca, cb = Pa[0] == yi, Pb[0] == yi
            b01, b10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
            z = (abs(b01 - b10) - 1) / np.sqrt(b01 + b10) if (b01 + b10) else 0.0
            fa = float(final.loc[final["konfigurasi"] == f"pilihan_{k}", "f1_test"].iloc[0])
            fb = float(final.loc[final["konfigurasi"] == ref, "f1_test"].iloc[0])
            cmp_rows.append({"pembanding": f"pilihan_{k} - {ref}", "selisih_macro_f1": round(fa - fb, 4),
                             "ci95_lo": round(float(lo), 4), "ci95_hi": round(float(hi), 4),
                             "ci_melewati_0": bool(lo <= 0 <= hi), "mcnemar_z": round(float(np.sign(b01 - b10) * z), 3),
                             "mcnemar_p": round(float(2 * (1 - normal_dist.cdf(abs(z)))), 4)})
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(out / "perbandingan_pilihan_vs_acuan.csv", index=False)

    # kebocoran: konfigurasi pilihan pada raw (irisan dokumen test dengan full_initial)
    leak_rows = []
    sp_raw = tk.split3(arms["raw"])
    common = sorted(set(sp_raw["test"]["_orig_idx"]) & set(sp["test"]["_orig_idx"]))
    ytrue = arms["raw"].set_index("_orig_idx").loc[common, "Severity"].values
    for k in ("A", "B"):
        r = picks[k]
        if r is None:
            continue
        P_raw, _, _ = run_spec(f"raw_{k}", r, sp_raw)
        f_raw = np.mean([mf1(ytrue, p.loc[common].values) for p in P_raw])
        f_ini = np.mean([mf1(ytrue, p.loc[common].values) for p in preds[f"pilihan_{k}"]])
        leak_rows.append({"pilihan": k, "n_test_irisan": len(common), "f1_raw": round(float(f_raw), 4),
                          "f1_full_initial": round(float(f_ini), 4), "selisih_poin": round(100 * float(f_ini - f_raw), 2)})
    leak = pd.DataFrame(leak_rows)
    leak.to_csv(out / "kebocoran_raw_vs_initial.csv", index=False)

    if "acuan_sgd" in sgd_hist:
        best_sgd = res[res["family"] == "sgd"].sort_values(["f1_val", "gap_poin"], ascending=[False, True]).iloc[0]
        if f"pilihan_A" in sgd_hist or f"pilihan_B" in sgd_hist:
            comp = {"lama (20k bi, alpha 1e-4)": sgd_hist["acuan_sgd"]}
            for k in ("A", "B"):
                if f"pilihan_{k}" in sgd_hist:
                    comp[f"pilihan {k}"] = sgd_hist[f"pilihan_{k}"]
        else:
            _, _, h = run_spec("sgd_best", best_sgd, sp)
            comp = {"lama (20k bi, alpha 1e-4)": sgd_hist["acuan_sgd"],
                    f"SGD terbaik di val ({best_sgd['fitur']}, alpha {best_sgd['reg']:g})": h}
        plot_sgd_compare(comp, out / "kurva_sgd_lama_vs_baru.png")

    rep = {"detik": round(time.time() - t0, 1), "n_konfigurasi": len(res), "val_terbaik": float(best_val),
           "pilihan": {k: (None if r is None else {kk: (v.item() if hasattr(v, "item") else v)
                                                   for kk, v in r.items() if kk != "best_epoch" or v is not None})
                       for k, r in picks.items()}}
    (out / "hasil.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    top = res.sort_values("f1_val", ascending=False).head(10)
    md = ["# Perbaikan gap train–val (12 Okt 2026)\n",
          f"`task_perbaiki_gap.py`, {len(res)} konfigurasi, {rep['detik']:.0f} dtk. Data: `full_initial`. Penyetelan di val, test sekali.\n",
          "## Acuan lama (val)\n", tk.md_table(base_rows[["family", "fitur", "reg", "f1_train", "f1_val", "gap_poin"]].round(4)), "",
          "## 10 konfigurasi dengan val tertinggi\n",
          tk.md_table(top[["family", "fitur", "reg", "f1_train", "f1_val", "gap_poin"]].round(4)), "",
          "## Pilihan\n",
          *[f"- **{k}**: " + ("tidak ada konfigurasi dengan gap ≤ 5 poin" if r is None else
                             f"{r['family']} · {r['fitur']} · reg {r['reg']:g} → train {r['f1_train']:.3f}, val {r['f1_val']:.3f}, gap {r['gap_poin']:.1f} poin")
            for k, r in picks.items()], "",
          "![pareto](pareto_gap_vs_val.png)\n",
          "## Evaluasi akhir di test\n", tk.md_table(final), "",
          "## Pilihan vs acuan (bootstrap berpasangan, McNemar)\n", tk.md_table(cmp) if len(cmp) else "-", "",
          "## Kebocoran: raw vs full_initial dengan konfigurasi pilihan\n", tk.md_table(leak) if len(leak) else "-", "",
          "![SGD](kurva_sgd_lama_vs_baru.png)\n"]
    (out / "ringkasan.md").write_text("\n".join(md), encoding="utf-8")
    print(final[["konfigurasi", "family", "fitur", "reg", "f1_train", "f1_val", "f1_test", "gap_train_val_poin"]].to_string(index=False))
    print(cmp.to_string(index=False))
    print(leak.to_string(index=False))
    print(f"\nSelesai {rep['detik']:.0f} dtk -> {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
