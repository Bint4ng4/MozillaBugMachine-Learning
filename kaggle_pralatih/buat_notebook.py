"""buat_notebook.py — Bungkus rq2_pralatih_kaggle.py menjadi dua notebook: Kaggle (File → Import Notebook) dan
Google Colab (File → Upload notebook).

Sumber kebenaran tetap berkas .py; jalankan ulang skrip ini setiap kali .py diubah.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "rq2_pralatih_kaggle.py").read_text(encoding="utf-8")
setting = "\n".join([
    "import os",
    "# 1) UJI KECIL DULU (±3 menit, 600 dokumen, 1 epoch): hapus tanda # di baris bawah, lalu jalankan semua sel.",
    "#    Kalau berakhir dengan 'Selesai', beri # lagi, lalu jalankan run penuh.",
    '# os.environ["RQ2_SMOKE"] = "1"',
    "# 2) Varian pemotongan opsional — HANYA bila diputuskan (bawaan: head):",
    '# os.environ["RQ2_TRUNC"] = "head_tail"',
])
NB = {
    "rq2_pralatih_kaggle.ipynb": [
        ("markdown", "# RQ2 — uji ketahanan dengan model pralatih (bert-base-uncased) — Kaggle\n\n"
                     "Sebelum menjalankan: **Add Input** → dataset privat berisi `rq2_pralatih.csv`; **Settings** → "
                     "Accelerator *GPU T4 x2* (atau P100) dan **Internet On** (butuh akun terverifikasi nomor HP). "
                     "Uji kecil dulu (sel kedua), lalu **Save Version → Save & Run All (Commit)**. Unduh "
                     "`prediksi_pralatih.csv`, `log_epoch.csv`, `info_run.json` dari tab Output. "
                     "Panduan: `PANDUAN_KAGGLE.md`."),
        ("code", setting),
        ("code", src),
    ],
    "rq2_pralatih_colab.ipynb": [
        ("markdown", "# RQ2 — uji ketahanan dengan model pralatih (bert-base-uncased) — Google Colab\n\n"
                     "Sebelum menjalankan: taruh `rq2_pralatih.csv` di Google Drive, folder `rq2_pralatih/`. "
                     "**Runtime → Change runtime type → T4 GPU**. Jalankan sel mount Drive (izinkan akses), lalu uji "
                     "kecil, lalu run penuh (**Runtime → Run all**). Biarkan tab terbuka. Bila sesi putus, jalankan "
                     "lagi: seed yang selesai dilewati. Hasil: Drive/`rq2_pralatih/hasil/`. "
                     "Panduan: `PANDUAN_KAGGLE.md` (bagian Colab)."),
        ("code", "from google.colab import drive\ndrive.mount('/content/drive')\n"
                 "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else "
                 "'TIDAK ADA — ganti runtime ke T4 GPU')"),
        ("code", setting),
        ("code", src),
    ],
}
for name, cells in NB.items():
    nb = {"cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
                     **({"execution_count": None, "outputs": []} if t == "code" else {})} for t, s in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    (HERE / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{name} ditulis")
