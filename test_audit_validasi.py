"""
test_audit_validasi.py — Tes kecil untuk audit_validasi.py (validasi manusia atas label audit AI).

Jalankan:  python test_audit_validasi.py     (kode keluar 0 bila lulus)
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import audit_validasi as av

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if cond else 'GAGAL'}] {name}" + (f"  -> {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def test_metrics() -> None:
    print("PABAK / Gwet AC1")
    r1 = ["K"] * 19 + ["B"]
    r2 = ["K"] * 20
    # dihitung tangan: pa 0,95; pi_K 0,975, pi_nonK 0,025; pe = 2 * 0,975 * 0,025 = 0,04875
    check("PABAK biner = 2 * 0,95 - 1 = 0,90", abs(av.pabak(r1, r2, 2) - 0.90) < 1e-12)
    check("AC1 biner = (0,95 - 0,04875) / 0,95125", abs(av.gwet_ac1(r1, r2, ["K", "B"]) - 0.90125 / 0.95125) < 1e-12)
    check("AC1 sepakat sempurna = 1", abs(av.gwet_ac1(["K", "B"], ["K", "B"], ["K", "B"]) - 1) < 1e-12)
    res = av.agreement(pd.Series(["K"] * 10), pd.Series(["K"] * 10))
    check("dua penilai sama-sama semua K -> kappa NaN (bukan crash), sepakat 1",
          np.isnan(res["biner_K_vs_bukan"]["kappa"]) and res["biner_K_vs_bukan"]["kesepakatan"] == 1)
    res = av.agreement(pd.Series(list("KKKRB")), pd.Series(list("KKBKB")))
    check("arah beda dihitung: r1 K/r2 bukan = 1, r1 bukan/r2 K = 1",
          res["r1_K_r2_bukan"] == 1 and res["r1_bukan_r2_K"] == 1, str(res))


def test_subset_and_sheets() -> None:
    print("pick_subset / siapkan")
    key = pd.DataFrame({"item": range(1, 101), "group": ["A_dipotong", "B_tanpa_penanda"] * 50})
    sub = av.pick_subset(key, 20, 7)
    check("40 item, 20 per grup", len(sub) == 40 and set(sub["group"].value_counts()) == {20})
    check("nomor baru 1..40 dan item asal unik", list(sub["item"]) == list(range(1, 41))
          and sub["item_asal"].is_unique)
    check("grup di kunci cocok dengan grup asal",
          (key.set_index("item").loc[sub["item_asal"], "group"].values == sub["group"].values).all())
    check("deterministik per seed", av.pick_subset(key, 20, 7).equals(sub))
    check("nomor baru tidak sama dengan nomor asal (urutan diacak)", (sub["item"] != sub["item_asal"]).sum() > 30)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        key.to_csv(td / "kunci.csv", index=False, encoding="utf-8-sig")
        sheet = pd.DataFrame({"item": range(1, 101), "teks": [f"teks dokumen {i}" for i in range(1, 101)],
                              "label": "", "kutipan_bukti": "", "catatan": ""})
        sheet.to_csv(td / "lembar.csv", index=False, encoding="utf-8-sig")
        (td / "panduan.md").write_text(
            "# Panduan Penilai — Audit Manual Dokumen Bug Report (Langkah 3)\n\nlembar `audit_penilai_A.csv` atau "
            "`audit_penilai_B.csv`\n4. Perkiraan waktu: 1–2 jam untuk 100 item.\n\n## Konteks\nisi\n\n"
            "## Setelah selesai\nlama\n", encoding="utf-8")
        args = argparse.Namespace(key_asal=str(td / "kunci.csv"), sheet_asal=str(td / "lembar.csv"),
                                  panduan_asal=str(td / "panduan.md"), per_group=20, seed=7, out=str(td / "v"))
        check("siapkan berhasil", av.cmd_siapkan(args) == 0)
        p1 = pd.read_csv(td / "v" / "lembar_validasi_P1.csv", encoding="utf-8-sig", keep_default_na=False)
        k = pd.read_csv(td / "v" / "KUNCI_validasi_jangan_dibuka_penilai.csv", encoding="utf-8-sig")
        check("lembar tidak membocorkan grup/item asal", list(p1.columns) == ["item", "teks", "label",
                                                                              "kutipan_bukti", "catatan"])
        check("teks lembar = teks item asal", all(p1["teks"] == [f"teks dokumen {i}" for i in k["item_asal"]]))
        check("label kosong", (p1["label"] == "").all())
        pd_txt = (td / "v" / "PANDUAN_PENILAI_VALIDASI.md").read_text(encoding="utf-8")
        check("panduan: larangan AI, nama lembar baru, perintah hitung baru",
              "Jangan memakai AI" in pd_txt and "lembar_validasi_P1.csv" in pd_txt and "audit_validasi.py hitung" in pd_txt
              and "lama" not in pd_txt)
        filled = sheet.assign(label="K")
        filled.to_csv(td / "terisi.csv", index=False, encoding="utf-8-sig")
        check("siapkan menolak lembar sumber yang sudah terisi",
              av.cmd_siapkan(argparse.Namespace(**{**vars(args), "sheet_asal": str(td / "terisi.csv")})) == 1)

        # hitung end-to-end dari folder kerja sementara (lembar AI tidak ada -> hanya manusia)
        lab = np.where(np.arange(40) < 36, "K", "B")
        for who, labels in (("P1", lab), ("P2", np.roll(lab, 1))):
            p1.assign(label=labels).to_csv(td / "v" / f"{who}.csv", index=False, encoding="utf-8-sig")
        cwd = os.getcwd()
        try:
            os.chdir(td)
            rc = av.cmd_hitung(argparse.Namespace(p1="v/P1.csv", p2="v/P2.csv", out="v"))
        finally:
            os.chdir(cwd)
        check("hitung berhasil dan menulis validasi_hasil.json", rc == 0 and (td / "v" / "validasi_hasil.json").exists())


def main() -> int:
    test_metrics()
    test_subset_and_sheets()
    print()
    if FAILED:
        print(f"{len(FAILED)} tes GAGAL: {FAILED}")
        return 1
    print("Semua tes lulus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
