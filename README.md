# MozillaBugMachine-Learning

Audit protocol and experiment pipeline for the pre-thesis **"Post-Report Contamination in Bug
Severity Datasets: A Pre-Training Audit Protocol and the Overestimation of Triage-Time
Performance"**, studying post-report contamination in the Mozilla Bugzilla severity-classification
dataset ("Cleaned Bugzilla", Kaggle) and its effect on reported macro-F1 at triage time.

> This repository holds the detection protocol (marker lexicon, chi-square/Cramér's V, structural
> ground truth from initial bug descriptions) and the ablation pipeline (`cleaning.py`,
> `experiment.py`) used to quantify contamination prevalence (RQ1) and triage-time performance
> overestimation (RQ2). Raw dataset, audit sheets, and intermediate results are not redistributed
> (Kaggle license "Unknown") — scripts to rebuild them from source are provided instead.

## Quick start

```bash
pip install pandas numpy scipy scikit-learn
python cleaning.py --list                 # lihat semua konfigurasi ablasi
python cleaning.py --config full          # jalankan satu konfigurasi
python experiment.py --suite rq2          # jalankan eksperimen RQ2
python rq1_hitung_v.py                    # hitung statistik RQ1 (prevalensi, V Cramér)
```

Dataset (`archive/sev.csv` dan `bugs*.csv`) tidak disertakan di repo ini (lisensi Kaggle
"Unknown") — unduh dari [Cleaned Bugzilla](https://www.kaggle.com/datasets/qicongliu/cleaned-bugzilla)
dan tempatkan di folder `archive/`.

---


### Tentang proyek

Dataset laporan bug untuk klasifikasi tingkat keparahan (*severity*) sering dibangun dengan
menggabungkan seluruh riwayat komentar tiap laporan, sehingga teksnya memuat komentar pengembang
pasca-laporan ("marking verified", "this bug has been marked as a duplicate of", dsb.) yang belum
ada saat laporan baru ditriase. Proyek ini mengusulkan **protokol audit pra-pelatihan** untuk
mendeteksi dan mengukur kontaminasi tersebut (RQ1), serta mengukur seberapa besar kontaminasi itu
melebih-lebihkan kinerja model saat triase (RQ2), menggunakan laporan awal (deskripsi pertama
Bugzilla) sebagai kebenaran struktural.

### Struktur proyek

| Berkas/folder | Isi |
|---|---|
| `cleaning.py` | Pipeline pembersihan `sev.csv`, disusun sebagai tahap-tahap yang bisa diablasikan (lihat `python cleaning.py --list`) |
| `experiment.py` | Eksperimen RQ2: estimasi berlebih macro-F1 (dokumen penuh vs laporan awal), kontrol panjang, kontrol positif sintetis |
| `rq1_struktural.py`, `rq1_hitung_v.py` | Statistik RQ1: prevalensi penanda, kebenaran struktural, uji khi-kuadrat, V Cramér |
| `verify_data.py` | Definisi leksikon penanda (`ANY_MARKER`) dan fungsi pembantu (`cramers_v`) |
| `audit_siapkan.py`, `audit_validasi.py`, `audit_hitung.py` | Penyiapan dan penghitungan audit kontaminasi berbantuan AI/manusia (interval Wilson, kappa, Gwet's AC1, PABAK) |
| `kaggle_laporan_awal.py` | Pemasangan laporan awal (deskripsi pertama Bugzilla) dengan dokumen `sev.csv` |
| `task_*.py` | Skrip analisis pendukung satu-kali (uji ketahanan, perbandingan literatur, kurva belajar, dll.) |
| `test_*.py` | Unit test untuk modul inti |

### Menjalankan pipeline

```bash
python cleaning.py --list                      # daftar 15+ konfigurasi ablasi
python cleaning.py --config plus_tailcut       # satu konfigurasi (keluaran: out_<tanggal>/)
python cleaning.py                             # semua konfigurasi sekaligus

python experiment.py --suite rq2               # RQ2: dokumen penuh vs laporan awal
python experiment.py --suite rq2_sens          # uji sensitivitas ambang token
python experiment.py --suite rq2_posctrl       # kontrol positif sintetis
python experiment.py --suite rq2_initial       # subset berpasangan dengan laporan awal

python rq1_hitung_v.py                         # prevalensi penanda, V Cramér, estimasi prevalensi audit
python rq1_struktural.py                       # kebenaran struktural (laporan awal vs dokumen penuh)
```

Folder keluaran (`out_*/`, `results_*/`) tidak disertakan di repo (bisa dibuat ulang dari skrip di
atas); folder `out/` tanpa tanggal sudah usang, jangan dipakai sebagai acuan.

### Data

- **Set data utama**: `sev.csv` dari rilis Kaggle "Cleaned Bugzilla" (32.428 laporan Mozilla
  Bugzilla, 6 kelas *severity*).
- **Laporan awal**: `bugs*.csv` dari rilis yang sama, berisi deskripsi pertama tiap bug, dipakai
  sebagai kebenaran struktural batas kontaminasi.
- Lisensi dataset tercatat "Unknown" di Kaggle sehingga **tidak didistribusikan ulang** di sini;
  unduh sendiri dari [tautan Kaggle](https://www.kaggle.com/datasets/qicongliu/cleaned-bugzilla)
  dan tempatkan di `archive/`.

### Kebutuhan

```
pandas
numpy
scipy
scikit-learn
```

(Uji ketahanan dengan BERT memakai `transformers`/`torch` secara terpisah — lihat
`kaggle_pralatih/`.)
