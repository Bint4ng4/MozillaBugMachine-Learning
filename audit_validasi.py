"""
audit_validasi.py — Validasi manusia atas label audit berbasis AI (runbook langkah 3, opsi 2, 10 Okt 2026).

Latar: 100 item audit (audit_langkah3/) sudah dinilai tiga kali oleh AI (audit_model_pembanding.csv 8 Okt;
audit_penilai_A.csv dan audit_penilai_B.csv 10 Okt, Haiku 5.5 diverifikasi Sonnet 5.5 menurut user). Kesepakatan
antar-AI bukan reliabilitas antar-penilai manusia, jadi dua penilai MANUSIA menilai subset acak berstrata
(default 20 dokumen terpotong + 20 tanpa penanda) untuk memvalidasi label AI.

Desain (anti-bias):
  - Subset diambil acak berstrata dengan seed tetap, urutannya diacak, dan diberi NOMOR BARU (1..n), sehingga
    penilai tidak bisa mencocokkan item dengan lembar AI. Pemetaan ke item asal hanya ada di berkas kunci.
  - Teks yang dinilai sama persis dengan lembar audit asli (teks yang dilihat model).

Pakai:
  python audit_validasi.py siapkan                      # buat lembar P1/P2 + kunci + panduan
  python audit_validasi.py hitung --p1 <csv> [--p2 <csv>]   # bandingkan manusia vs manusia vs AI
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
from audit_hitung import VALID, load_sheet, wilson  # noqa: E402

OUT = Path("audit_langkah3/validasi_manusia")
KEY_ASAL = Path("audit_langkah3/audit_KUNCI_jangan_dibuka_penilai.csv")
SHEET_ASAL = Path("audit_langkah3/audit_penilai_A.csv")        # lembar kosong asli (sumber teks)
AI_SHEETS = {"AI_pembanding_8Okt": "audit_model_pembanding.csv",
             "AI_lembarA_10Okt": "audit_penilai_A.csv",
             "AI_lembarB_10Okt": "audit_penilai_B.csv"}


def pick_subset(key: pd.DataFrame, per_group: int, seed: int) -> pd.DataFrame:
    """Ambil `per_group` item acak per grup, acak urutannya, beri nomor baru 1..n.
    Kembalikan kunci: item (baru), item_asal, group."""
    parts = [g.sample(n=per_group, random_state=seed + i)
             for i, (_, g) in enumerate(sorted(key.groupby("group"), key=lambda kv: kv[0]))]
    sub = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return pd.DataFrame({"item": np.arange(1, len(sub) + 1), "item_asal": sub["item"].values,
                         "group": sub["group"].values})


def gwet_ac1(r1, r2, cats) -> float:
    """Gwet AC1 untuk dua penilai; stabil saat prevalensi satu kategori sangat tinggi (paradoks kappa)."""
    r1, r2 = np.asarray(r1), np.asarray(r2)
    pa = float(np.mean(r1 == r2))
    pi = np.array([(np.mean(r1 == c) + np.mean(r2 == c)) / 2 for c in cats])
    pe = float(np.sum(pi * (1 - pi)) / (len(cats) - 1))
    return (pa - pe) / (1 - pe) if pe < 1 else float("nan")


def pabak(r1, r2, q: int) -> float:
    """Prevalence- and bias-adjusted kappa: (q * kesepakatan - 1) / (q - 1)."""
    return (q * float(np.mean(np.asarray(r1) == np.asarray(r2))) - 1) / (q - 1)


def agreement(r1: pd.Series, r2: pd.Series) -> dict:
    """Ukuran kesepakatan biner (K vs bukan-K) dan 3 kelas (B/R/K) antara dua penilai."""
    b1, b2 = (r1 == "K").map({True: "K", False: "nonK"}), (r2 == "K").map({True: "K", False: "nonK"})
    out = {}
    for name, (x, y, cats) in {"biner_K_vs_bukan": (b1, b2, ["K", "nonK"]),
                               "tiga_kelas": (r1, r2, sorted(VALID))}.items():
        same_const = x.nunique() == 1 and y.nunique() == 1
        kappa = float("nan") if same_const else float(cohen_kappa_score(x, y))
        out[name] = {"kesepakatan": round(float((x == y).mean()), 3), "kappa": round(kappa, 3),
                     "pabak": round(pabak(x, y, len(cats)), 3), "ac1": round(gwet_ac1(x, y, cats), 3)}
    # arah ketidaksepakatan biner: berguna untuk melihat apakah AI cenderung terlalu sering memberi K
    out["r1_K_r2_bukan"] = int(((r1 == "K") & (r2 != "K")).sum())
    out["r1_bukan_r2_K"] = int(((r1 != "K") & (r2 == "K")).sum())
    return out


def cmd_siapkan(args) -> int:
    key = pd.read_csv(args.key_asal, encoding="utf-8-sig")
    sheet = pd.read_csv(args.sheet_asal, encoding="utf-8-sig").set_index("item")
    if sheet["label"].fillna("").astype(str).str.strip().ne("").any():
        print(f"FATAL: {args.sheet_asal} sudah terisi -- sumber teks harus lembar KOSONG asli.")
        return 1
    sub = pick_subset(key, args.per_group, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lembar = pd.DataFrame({"item": sub["item"], "teks": sheet.loc[sub["item_asal"], "teks"].values,
                           "label": "", "kutipan_bukti": "", "catatan": ""})
    for who in ("P1", "P2"):
        lembar.to_csv(out / f"lembar_validasi_{who}.csv", index=False, encoding="utf-8-sig")
    sub.to_csv(out / "KUNCI_validasi_jangan_dibuka_penilai.csv", index=False, encoding="utf-8-sig")
    panduan = Path(args.panduan_asal).read_text(encoding="utf-8")
    panduan = (panduan
               .replace("# Panduan Penilai — Audit Manual Dokumen Bug Report (Langkah 3)",
                        "# Panduan Penilai — Validasi Manusia Audit Dokumen Bug Report (Langkah 3, subset)")
               .replace("`audit_penilai_A.csv` atau `audit_penilai_B.csv`", "`lembar_validasi_P1.csv` atau "
                        "`lembar_validasi_P2.csv`")
               .replace("`audit_KUNCI_jangan_dibuka_penilai.csv`", "`KUNCI_validasi_jangan_dibuka_penilai.csv`")
               .replace("1–2 jam untuk 100 item", f"30–60 menit untuk {len(sub)} item"))
    head, sep, tail = panduan.partition("## Konteks")
    extra = ("5. **Jangan memakai AI** (ChatGPT, Claude, dsb.) dan jangan membuka hasil audit sebelumnya. Penilaianmu\n"
             "   dipakai justru untuk menguji label yang dibuat AI.\n"
             "6. Sebelum mulai, beri tahu peneliti apakah kamu pernah melihat hasil audit atau angka kontaminasi\n"
             "   penelitian ini. Jawaban \"pernah\" tidak membatalkan penilaianmu, tapi wajib dicatat.\n\n")
    panduan = head.rstrip("\n") + "\n" + extra + sep + tail
    panduan = panduan.split("## Setelah selesai")[0] + (
        "## Setelah selesai\n\nKirim lembarmu kembali (tanpa mengubah nama kolom). Penghitungan:\n\n```\n"
        "python audit_validasi.py hitung --p1 audit_langkah3/validasi_manusia/lembar_validasi_P1.csv "
        "--p2 audit_langkah3/validasi_manusia/lembar_validasi_P2.csv\n```\n")
    (out / "PANDUAN_PENILAI_VALIDASI.md").write_text(panduan, encoding="utf-8")
    print(f"Subset {len(sub)} item ({sub['group'].value_counts().to_dict()}; seed {args.seed}) -> {out}/")
    print("Berikan ke penilai HANYA lembar_validasi_P1.csv / _P2.csv dan PANDUAN_PENILAI_VALIDASI.md.")
    return 0


def cmd_hitung(args) -> int:
    out = Path(args.out)
    key = pd.read_csv(out / "KUNCI_validasi_jangan_dibuka_penilai.csv", encoding="utf-8-sig").set_index("item")
    humans = {"P1": load_sheet(Path(args.p1))}
    if args.p2:
        humans["P2"] = load_sheet(Path(args.p2))
    for name, s in humans.items():
        if set(s.index) != set(key.index):
            print(f"FATAL: item {name} tidak cocok dengan kunci validasi.")
            return 1
    asal = key["item_asal"]
    ai = {}
    for name, path in AI_SHEETS.items():
        if Path(path).exists():
            full = load_sheet(Path(path))
            ai[name] = pd.Series(full.loc[asal.values].values, index=key.index, name=name)
    groups = {g: key["group"] == g for g in sorted(key["group"].unique())}
    report: dict = {"n_items": int(len(key)), "per_grup": {g: int(m.sum()) for g, m in groups.items()},
                    "prevalensi": {}, "kesepakatan": {}}

    print(f"Item validasi: {len(key)} {report['per_grup']}\n")
    print("Prevalensi K (interval Wilson 95%):")
    for name, s in {**humans, **ai}.items():
        report["prevalensi"][name] = {}
        line = []
        for g, m in groups.items():
            k, n = int((s[m] == "K").sum()), int(m.sum())
            lo, hi = wilson(k, n)
            report["prevalensi"][name][g] = {"k": k, "n": n, "pct": round(100 * k / n, 1),
                                             "ci95": [round(100 * lo, 1), round(100 * hi, 1)]}
            line.append(f"{g} {100 * k / n:5.1f}% [{100 * lo:.0f}, {100 * hi:.0f}]")
        print(f"  {name:<20} " + " | ".join(line))

    pairs = []
    if len(humans) == 2:
        pairs.append(("P1", "P2"))
    pairs += [(h, a) for h in humans for a in ai]
    print("\nKesepakatan (biner K vs bukan-K; kappa turun tajam bila hampir semua item K -> lihat PABAK/AC1):")
    for r1, r2 in pairs:
        s1 = humans[r1]
        s2 = humans[r2] if r2 in humans else ai[r2]
        res = agreement(s1, s2)
        report["kesepakatan"][f"{r1} vs {r2}"] = res
        b = res["biner_K_vs_bukan"]
        print(f"  {r1} vs {r2:<20} sepakat {100 * b['kesepakatan']:.0f}% | kappa {b['kappa']} | PABAK {b['pabak']} "
              f"| AC1 {b['ac1']} | {r1} K/{r2} bukan: {res['r1_K_r2_bukan']}, {r1} bukan/{r2} K: {res['r1_bukan_r2_K']}")

    if len(humans) == 2:
        tbl = pd.DataFrame({"item": key.index, "item_asal": asal.values, "group": key["group"].values,
                            "P1": humans["P1"].values, "P2": humans["P2"].values,
                            **{a: s.values for a, s in ai.items()}})
        diff = tbl[tbl["P1"] != tbl["P2"]]
        diff.to_csv(out / "validasi_perbedaan_P1_P2.csv", index=False, encoding="utf-8-sig")
        print(f"\n{len(diff)} item beda antara P1 dan P2 -> {out / 'validasi_perbedaan_P1_P2.csv'}")
    (out / "validasi_hasil.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Ringkasan disimpan: {out / 'validasi_hasil.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("siapkan")
    s.add_argument("--key-asal", default=str(KEY_ASAL))
    s.add_argument("--sheet-asal", default=str(SHEET_ASAL))
    s.add_argument("--panduan-asal", default="audit_langkah3/PANDUAN_PENILAI.md")
    s.add_argument("--per-group", type=int, default=20)
    s.add_argument("--seed", type=int, default=20261010)
    s.add_argument("--out", default=str(OUT))
    h = sub.add_parser("hitung")
    h.add_argument("--p1", required=True)
    h.add_argument("--p2")
    h.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    return cmd_siapkan(args) if args.cmd == "siapkan" else cmd_hitung(args)


if __name__ == "__main__":
    sys.exit(main())
