"""
rq1_struktural.py — RQ1 dan evaluasi dekontaminasi memakai KEBENARAN STRUKTURAL (K-4, 11 Okt 2026).

Kebenaran struktural: untuk dokumen yang berpasangan (tingkat 'utama', kaggle_laporan_awal.py), n_tok_awal token
pertama teks tahap A adalah laporan awal; sisanya komentar susulan. Tidak butuh penilai dan tidak sirkular (batasnya
berasal dari deskripsi pertama Bugzilla, bukan dari leksikon penanda).

Isi:
  1. Prevalensi kontaminasi (dokumen dengan >= T token sesudah laporan awal, T = 1/3/10) + porsi token pasca-laporan.
  2. Penanda leksikal vs kebenaran: seberapa banyak dokumen terkontaminasi yang tertangkap POST_REPORT / _EXT.
  3. Sisa kontaminasi per metode dekontaminasi berbasis awalan (TAILCUT, TAILCUT_EXT, first-100, first-50): token
     pasca-laporan yang masih tersisa dan token laporan awal yang ikut terbuang.
  4. 100 item audit: label tiga penilai AI vs kebenaran struktural.

Pakai:  python rq1_struktural.py [--archive archive] [--pairs out_laporan_awal/pasangan_laporan_awal.csv]
                                 [--out-dir results_laporan_awal] [--min-sisa 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cleaning as c  # noqa: E402
from audit_hitung import wilson  # noqa: E402

AI_SHEETS = {"AI_pembanding_8Okt": "audit_model_pembanding.csv", "AI_lembarA_Haiku": "audit_penilai_A.csv",
             "AI_lembarB_Sonnet": "audit_penilai_B.csv"}
AUDIT_KEY = "audit_langkah3/audit_KUNCI_jangan_dibuka_penilai.csv"


def ntok(s: pd.Series) -> pd.Series:
    return s.str.split().str.len().fillna(0).astype(int)


def prop(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "pct": round(100 * k / n, 2) if n else float("nan"),
            "ci95": [round(100 * lo, 2), round(100 * hi, 2)]}


def method_lengths(text: pd.Series, n_total: pd.Series) -> dict[str, pd.Series]:
    """Panjang awalan yang DISIMPAN tiap metode (semua metode di sini menyimpan awalan dokumen)."""
    return {
        "TAILCUT": ntok(text.map(lambda t: c.tailcut_text(t, c.POST_REPORT))),
        "TAILCUT_EXT": ntok(text.map(lambda t: c.tailcut_text(t, c.POST_REPORT_EXT))),
        "first100": n_total.clip(upper=100),
        "first50": n_total.clip(upper=50),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="archive")
    ap.add_argument("--pairs", default=str(c.LAPORAN_AWAL_CSV))
    ap.add_argument("--out-dir", default="results_laporan_awal")
    ap.add_argument("--min-sisa", type=int, default=3, help="token pasca-laporan minimal agar dihitung terkontaminasi")
    args = ap.parse_args()
    T = args.min_sisa
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    df = c.stage_a_structure(pd.read_csv(Path(args.archive) / "sev.csv"))
    n_awal_map = c.load_initial_pairs(Path(args.pairs))
    sub = df[df["_orig_idx"].isin(n_awal_map.index)].copy()
    sub["n_total"] = ntok(sub["Description"])
    sub["n_awal"] = sub["_orig_idx"].map(n_awal_map).astype(int)
    sub["post"] = (sub["n_total"] - sub["n_awal"]).clip(lower=0)
    N = len(sub)
    rep: dict = {"n_tahap_A": len(df), "n_berpasangan_utama": N, "min_sisa": T}
    print(f"Dokumen tahap A: {len(df)} | berpasangan (utama): {N} ({N / len(df):.1%})")

    # ---------------------------------------------------------------- 1. prevalensi
    rows = []
    for thr in (1, 3, 10):
        r = prop(int((sub["post"] >= thr).sum()), N)
        rows.append({"ambang_token": thr, **{k: v for k, v in r.items() if k != "ci95"},
                     "ci95_lo": r["ci95"][0], "ci95_hi": r["ci95"][1]})
    share = sub["post"] / sub["n_total"].replace(0, np.nan)
    rep["prevalensi"] = rows
    rep["porsi_token_pasca"] = {"median": float(share.median()), "rata2": float(share.mean()),
                                "gabungan": float(sub["post"].sum() / sub["n_total"].sum())}
    pd.DataFrame(rows).to_csv(out / "tabel_rq1_struktural.csv", index=False)
    print("\n1. Prevalensi kontaminasi (kebenaran struktural):")
    for r in rows:
        print(f"   >= {r['ambang_token']:>2} token pasca-laporan: {r['pct']:.2f}% [{r['ci95_lo']:.2f}, {r['ci95_hi']:.2f}]")
    print(f"   porsi token pasca-laporan per dokumen: median {share.median():.2f}, rata-rata {share.mean():.2f}; "
          f"gabungan {rep['porsi_token_pasca']['gabungan']:.2f}")

    # ---------------------------------------------------------------- 2. penanda vs kebenaran
    cont = sub["post"] >= T
    rows = []
    for name, pat in (("POST_REPORT", c.POST_REPORT), ("POST_REPORT_EXT", c.POST_REPORT_EXT)):
        mk = sub["Description"].str.contains(pat)
        r = {"penanda": name, "dok_berpenanda_pct": round(100 * mk.mean(), 2),
             "sensitivitas_pct": prop(int((mk & cont).sum()), int(cont.sum()))["pct"],
             "kontaminasi_bila_berpenanda_pct": prop(int((mk & cont).sum()), int(mk.sum()))["pct"],
             "kontaminasi_bila_tanpa_penanda_pct": prop(int((~mk & cont).sum()), int((~mk).sum()))["pct"],
             "n_tanpa_penanda": int((~mk).sum())}
        rows.append(r)
    rep["penanda_vs_struktur"] = rows
    pd.DataFrame(rows).to_csv(out / "tabel_penanda_vs_struktur.csv", index=False)
    print(f"\n2. Penanda leksikal vs kebenaran (terkontaminasi = >= {T} token pasca-laporan):")
    for r in rows:
        print(f"   {r['penanda']:<16} berpenanda {r['dok_berpenanda_pct']:.1f}% | menangkap {r['sensitivitas_pct']:.1f}% "
              f"dok terkontaminasi | dok TANPA penanda yang terkontaminasi: {r['kontaminasi_bila_tanpa_penanda_pct']:.1f}% "
              f"(n={r['n_tanpa_penanda']})")

    # ---------------------------------------------------------------- 3. sisa per metode
    mk = sub["Description"].str.contains(c.POST_REPORT)
    lens = method_lengths(sub["Description"], sub["n_total"])
    rows = []
    for name, m in lens.items():
        sisa = (m - sub["n_awal"]).clip(lower=0)
        hilang = (sub["n_awal"] - m).clip(lower=0)
        has_post = sub["post"] > 0
        removed_share = (1 - sisa / sub["post"].replace(0, np.nan)).where(has_post)
        for grp, g in (("semua", np.ones(N, bool)), ("berpenanda", mk.values), ("tanpa_penanda", ~mk.values)):
            g = pd.Series(g, index=sub.index)
            r = prop(int(((sisa >= T) & g).sum()), int(g.sum()))
            rows.append({"metode": name, "kelompok": grp, "n": int(g.sum()),
                         "masih_terkontaminasi_pct": r["pct"], "ci95_lo": r["ci95"][0], "ci95_hi": r["ci95"][1],
                         "token_disimpan_median": float(m[g].median()),
                         "sisa_token_pasca_median": float(sisa[g].median()),
                         "porsi_pasca_terbuang_median": float(removed_share[g & has_post].median()),
                         "memotong_laporan_awal_pct": round(100 * float((hilang[g] > 0).mean()), 2),
                         "token_laporan_hilang_median_bila_terpotong": float(hilang[g & (hilang > 0)].median())
                         if bool((g & (hilang > 0)).any()) else 0.0})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "tabel_sisa_per_metode.csv", index=False)
    rep["sisa_per_metode"] = rows
    print(f"\n3. Sisa kontaminasi per metode (masih >= {T} token pasca-laporan setelah metode):")
    for _, r in tab[tab["kelompok"] == "semua"].iterrows():
        print(f"   {r['metode']:<12} masih terkontaminasi {r['masih_terkontaminasi_pct']:.1f}% "
              f"[{r['ci95_lo']:.1f}, {r['ci95_hi']:.1f}] | konten pasca terbuang (median) {r['porsi_pasca_terbuang_median']:.2f} | "
              f"memotong laporan awal {r['memotong_laporan_awal_pct']:.1f}% (median {r['token_laporan_hilang_median_bila_terpotong']:.0f} token)")

    # ---------------------------------------------------------------- 4. audit AI vs struktur
    key = pd.read_csv(AUDIT_KEY, encoding="utf-8-sig")
    tr = df[df["_orig_idx"] <= c.TRAIN_MAX].reset_index(drop=True)
    te = df[df["_orig_idx"] >= c.TEST_MIN].reset_index(drop=True)
    sheet = pd.read_csv(AI_SHEETS["AI_pembanding_8Okt"], encoding="utf-8-sig").set_index("item")
    items, mism = [], 0
    for _, r in key.iterrows():
        src = tr if r["split"] == "train" else te
        row = src.iloc[int(r["row_in_split"])]
        expect = c.tailcut_text(row["Description"]) if r["group"] == "A_dipotong" else row["Description"]
        if " ".join(str(sheet.at[r["item"], "teks"]).split()) != " ".join(expect.split()):
            mism += 1
            continue
        oi = row["_orig_idx"]
        if oi not in n_awal_map.index:
            continue
        sisa = len(expect.split()) - int(n_awal_map[oi])
        items.append({"item": int(r["item"]), "group": r["group"], "sisa_token": sisa, "K_struktur": sisa >= T})
    it = pd.DataFrame(items).set_index("item")
    rep["audit"] = {"item_teks_tidak_cocok": mism, "item_berpasangan": len(it), "per_kelompok": {}, "penilai": {}}
    print(f"\n4. Item audit: {len(it)}/100 berpasangan (teks lembar tidak cocok dengan rekonstruksi: {mism})")
    for g, d in it.groupby("group"):
        rep["audit"]["per_kelompok"][g] = prop(int(d["K_struktur"].sum()), len(d))
        r = rep["audit"]["per_kelompok"][g]
        print(f"   {g}: terkontaminasi menurut struktur {r['pct']:.0f}% [{r['ci95'][0]:.0f}, {r['ci95'][1]:.0f}] (n={r['n']})")
    rows = []
    for name, f in AI_SHEETS.items():
        lab = pd.read_csv(f, encoding="utf-8-sig").set_index("item")["label"].astype(str).str.strip().str.upper()
        j = it.join(lab.rename("label"), how="inner")
        ai_k = (j["label"] == "K").values
        st_k = j["K_struktur"].values
        r = {"penilai": name, "n": len(j), "sepakat_pct": round(100 * float((ai_k == st_k).mean()), 1),
             "kappa": round(float(cohen_kappa_score(st_k, ai_k)), 3) if len(set(st_k)) > 1 else float("nan"),
             "AI_K_struktur_K": int((ai_k & st_k).sum()), "AI_K_struktur_bersih": int((ai_k & ~st_k).sum()),
             "AI_bukanK_struktur_K": int((~ai_k & st_k).sum()), "keduanya_bersih": int((~ai_k & ~st_k).sum())}
        for g in ("A_dipotong", "B_tanpa_penanda"):
            m = (j["group"] == g).values
            r[f"K_AI_pct_{g}"] = round(100 * float(ai_k[m].mean()), 1)
            r[f"K_struktur_pct_{g}"] = round(100 * float(st_k[m].mean()), 1)
        rows.append(r)
        print(f"   {name:<20} sepakat {r['sepakat_pct']:.0f}% | kappa {r['kappa']} | AI K & struktur K {r['AI_K_struktur_K']}"
              f" | AI K tapi bersih {r['AI_K_struktur_bersih']} | AI bukan-K tapi K {r['AI_bukanK_struktur_K']}"
              f" | keduanya bersih {r['keduanya_bersih']}")
    rep["audit"]["penilai"] = rows
    pd.DataFrame(rows).to_csv(out / "tabel_audit_ai_vs_struktur.csv", index=False)
    it.reset_index().to_csv(out / "audit_item_struktur.csv", index=False)

    (out / "rq1_struktural.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nDisimpan di {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
