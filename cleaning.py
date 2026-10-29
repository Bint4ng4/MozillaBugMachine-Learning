"""
cleaning.py — Pipeline pembersihan sev.csv, disusun sebagai tahap-tahap yang bisa diablasikan.

Setiap tahap adalah fungsi terpisah sehingga bisa dinyalakan/dimatikan satu per satu.
Itulah yang membuat RQ3 (ablasi cleaning) bisa dijawab: jalankan konfigurasi berbeda,
bandingkan macro-F1-nya.

Cara pakai:
    python cleaning.py                      # jalankan semua konfigurasi ablasi
    python cleaning.py --config full        # satu konfigurasi saja
    python cleaning.py --list               # lihat daftar konfigurasi

Keluaran (default --out = out_<YYYYMMDD> hari ini; folder lama out/ basi, jangan dipakai):
    <out>/cleaned_<config>.csv       data bersih per konfigurasi
    <out>/cleaning_analysis.csv      tabel perbandingan antar konfigurasi (untuk Tabel 4)
Paper 1 merujuk out_20261008/ (15 konfigurasi) dan out_20261011/ (3 konfigurasi laporan awal).

PENTING — urutan tahap tidak boleh diubah sembarangan:
  - dedup antar dokumen HARUS terakhir; near-duplicate baru muncul setelah
    boilerplate dan metadata dibuang.
  - dekontaminasi adalah VARIAN EKSPERIMEN, bukan default. RQ2 justru membandingkan
    kondisi dengan dan tanpa tahap ini.

Butuh: pandas, numpy
"""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

SEVERE = {"major", "critical", "blocker"}

# ---------------------------------------------------------------- pola regex

# Penanda pasca-pelaporan — teks yang mustahil ada saat laporan baru dikirim.
# Dihapus per-BARIS, bukan per-dokumen, supaya isi laporan asli tidak ikut terbuang.
#
# SINKRONISASI WAJIB: pattern ini harus tetap identik cakupannya dengan ANY_MARKER
# di verify_data.py. Sempat berbeda (POST_REPORT lupa "Updated by", beberapa varian
# frasa lebih sempit) sehingga pct_contaminated di sini melaporkan under-count vs
# pengukuran verify_data.py pada teks yang sama (0,81% vs 1,96% terukur ulang untuk
# plus_decon, 9 Sep 2026). Lihat MASTER note §12 Catatan Koreksi butir 16.
POST_REPORT = re.compile(
    r"marked as a duplicate(?:\s+of\s+\d+)?"
    r"|Marking\s+(?:Verified|Duplicate|Invalid|WONTFIX|Fixed)"
    r"|WORKSFORME|worksforme|works for me"
    r"|\bRESOLVED\b"
    r"|checked in|fixed in|landed in"
    r"|In\s+reply\s+to\s+comment(?:\s+\d+)?"
    r"|Updated\s+by"
    r"|\bverified\b",
    re.IGNORECASE,
)

# Penanda DIPERLUAS — HANYA untuk analisis sensitivitas (TAILCUT_EXT), BUKAN definisi kontaminasi
# utama (RQ1 tetap memakai POST_REPORT, sinkron dengan verify_data.py). Disusun dari kosakata alur
# kerja Bugzilla (status, resolusi, triase, milestone) setelah audit 8 Okt 2026 menunjukkan
# POST_REPORT melewatkan varian seperti "marking as wontfix", "verifying", "reassigning".
# Sengaja mengutamakan RECALL: positif palsu memotong sebagian laporan asli, dan efek panjangnya
# ditanggung kontrol buang-blok-acak sepanjang sama. JANGAN dipakai untuk mengukur kebersihan
# sampel audit yang ikut membentuk daftar ini (sirkular).
# Sengaja TIDAK dimasukkan: "fixed" tanpa konteks, "duplicate" tanpa "of", milestone telanjang
# ("M13" sering nama build pelapor), dan "PDT" (juga zona waktu di header laporan).
POST_REPORT_EXT = re.compile(
    POST_REPORT.pattern
    + r"|\b(?:re)?assign(?:ed|ing)?\s+(?:to|this|it)\b"
    r"|\bmark(?:ed|ing)\s+(?:(?:this|it)\s+(?:bug\s+)?)?(?:as\s+(?:a\s+)?)?"
    r"(?:verified|fixed|dup\w*|invalid|wontfix|worksforme|resolved|closed|later|remind)\b"
    r"|\bverif(?:y|ying|ies)\b"
    r"|\b(?:wontfix|dupe)\b|\bdup(?:licate)?\s+of\b"
    r"|\breopen(?:ed|ing|s)?\b"
    r"|\btarget\s+milestone\b|\b(?:move|moved|moving)\s+to\s+(?:m\d+|future|later)\b|\bbulk\s+mov\w*"
    r"|\b(?:qa\s+contact|status\s+whiteboard)\b"
    r"|\b(?:chang|sett|rais|lower|bump|upp|increas|decreas)ing\s+(?:the\s+)?"
    r"(?:severity|priority|component|product|milestone|qa\s+contact|status|summary|platform)\b"
    r"|\bseverity\s+(?:to|is\s+now)\b"
    r"|\bcheckin\b|\bcheck\s+in\b|\bpatch\s+(?:checked|landed|committed|reviewed)\b"
    r"|\bconfirm(?:ed|ing)\s+(?:on|with|in|using)\b"
    r"|\bthis\s+bug\s+has\s+been\b",
    re.IGNORECASE,
)

# Boilerplate Bugzilla + PII. Tanda baca sudah hilang, jadi polanya berbasis kata.
BOILERPLATE = re.compile(
    "|".join([
        r"\b(?:Created|Updated|Reported|Modified)\s+by\s+.{0,120}?\s+on\s+"
        r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+.{0,60}?"
        r"\d{1,2}\s+\d{2}\s+\d{2}\s+(?:AM|PM)\s+[A-Z]{2,4}",
        r"\bAdditional\s+Details\b",
        r"\bComment\s+\d+\b",
        r"\bCommit\s+pushed\s+to\s+\S+\s+at\s+.{0,120}",
        # email termutilasi: "nama domain com"
        r"\b[\w\-]+\s+(?:mozilla|netscape|gmail|yahoo|hotmail|redhat|google|apache)"
        r"\s+(?:[a-z]{2,4}\s+){0,3}(?:com|org|net|edu|uk|de)\b",
        r"\bhttps?\s+\S+(?:\s+\S+){0,10}",
        r"\bwww\s+\S+(?:\s+\S+){0,6}",
    ]),
    re.IGNORECASE,
)

# Token bervariasi tinggi yang menggelembungkan vocabulary tanpa memberi sinyal
HIGH_CARDINALITY = [
    (re.compile(r"\b[0-9a-f]{20,}\b"), " "),              # commit SHA
    (re.compile(r"\b0x[0-9a-fA-F]{4,}\b"), " "),          # alamat memori
    (re.compile(r"\b\d{1,3}(?:\s+\d{1,3}){3}\b"), " "),   # IP termutilasi
    (re.compile(r"\b\d+\b"), " "),                        # angka
]

# Batas blok komentar pertama. Dekontaminasi STRUKTURAL memotong di sini —
# jauh lebih tegas daripada menghapus frasa penanda satu per satu, dan jauh lebih
# dekat ke skenario deployment (QA mengirim satu laporan baru, bukan riwayat lengkap).
FIRST_COMMENT_END = re.compile(
    r"\b(?:Updated\s+by|In\s+reply\s+to\s+comment|Comment\s+\d+)\b", re.IGNORECASE
)

# Identitas pelapor/asignee. Pelapor tertentu berkorelasi dengan severity
# (komponen, senioritas) — model bisa belajar "siapa", bukan "apa".
IDENTITY = re.compile(
    r"\b[\w\-]+\s+(?:mozilla|netscape|gmail|yahoo|hotmail|redhat|google|apache|aol|ibm|sun)"
    r"\s+(?:[a-z]{2,4}\s+){0,3}(?:com|org|net|edu|uk|de)\b"
    r"|\b(?:Created|Updated|Reported|Modified|Assigned)\s+(?:by|to)\s+"
    r"(?:[A-Z][a-z]+\s+){1,3}",
)

# Tanggal dan tahun. BAHAYA KHUSUS: split di sini kronologis, sehingga token tahun
# ('1998', '2004') menjadi indikator train/test yang hampir sempurna pada TF-IDF.
#
# SENGAJA PEKA HURUF BESAR/KECIL (tanpa re.IGNORECASE). Versi sebelumnya memakai
# IGNORECASE dengan pola zona waktu r"\b[A-Z]{3}T?\b", yang cocok dengan SETIAP kata tiga
# huruf ("the", "and", "bug") dan kata empat huruf berakhiran t ("that", "text"), serta
# menghapus "may"/"march" sebagai kata biasa (diuji 8 Okt 2026). Di sev.csv penulisan
# tanggal Bugzilla selalu berhuruf kapital ("Tuesday April 7 1998 10 57 25 AM PDT").
# "May"/"March" hanya dihapus bila diikuti angka (tanggal), bukan kata kerja/kata benda.
TIMEZONES = r"(?:PDT|PST|EDT|EST|CDT|CST|MDT|MST|GMT|UTC)"
DATES = re.compile(
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b"
    r"|\b(?:January|February|April|June|July|August|September|October|November|December)\b"
    r"|\b(?:March|May)\b(?=\s+\d)"
    r"|\b(?:19[89]\d|20[0-2]\d)\b"
    r"|\b\d{1,2}\s+\d{2}\s+\d{2}\s+(?:AM|PM)\b"
    rf"|\b{TIMEZONES}\b"
)

# Rentang split kronologis bawaan (sev_train.csv / sev_test.csv; MASTER SS4.7).
# Dipakai oleh split_chronological() dan oleh tahap yang menghitung statistik lintas
# dokumen (G, E2) agar statistik itu TIDAK memakai informasi test (Kapoor & Narayanan 2023,
# L1.2 dan L1.4).
TRAIN_MAX = 28505
TEST_MIN = 28507

NOISE_DOC = re.compile(
    r"^(?:test\d*|testing|ignore(?:\s+this)?|verified(?:\s+\w+)?|"
    r"marking\s+\w+|closed?|fixed|wfm|dupe|duplicate|ok|ugh(?:\s+\w+)*|"
    r"n\s*a|nvm|nevermind)\s*$",
    re.IGNORECASE,
)

_WS = re.compile(r"\s+")


def _norm(t: str) -> str:
    return _WS.sub(" ", t).strip()


# ------------------------------------------------------------------- tahapan

def stage_a_structure(df: pd.DataFrame) -> pd.DataFrame:
    """A. Buang kolom index sampah, baris kosong, normalisasi whitespace, dedup persis.

    Index asli disimpan ke kolom `_orig_idx` SEBELUM kolom sampah dibuang -- dibutuhkan untuk
    split kronologis SETELAH cleaning (lihat split_chronological()). Tanpa ini, index baris
    di file cleaned_*.csv tidak bisa dipetakan balik ke rentang train/test asli
    (0-28505 train, 28507-33024 test, lihat MASTER note SS4.7).
    """
    junk = [c for c in df.columns if c.startswith("Unnamed") or not str(c).strip()]
    if junk and "_orig_idx" not in df.columns:
        df = df.copy()
        df["_orig_idx"] = df[junk[-1]]
    df = df.drop(columns=junk)
    df["Description"] = df["Description"].astype(str)
    df = df[df["Description"].str.strip().ne("")]
    df["Description"] = df["Description"].map(_norm)
    return df.drop_duplicates(subset=["Description"]).reset_index(drop=True)


def split_chronological(df: pd.DataFrame, train_max: int = TRAIN_MAX,
                         test_min: int = TEST_MIN) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split kronologis pada data yang SUDAH dibersihkan, pakai `_orig_idx` dari stage A.

    Rentang default sesuai split bawaan sev_train.csv/sev_test.csv (verify_data.py §6):
    train 0-28505, test 28507-33024, 0 overlap. Baris tanpa _orig_idx (dropped sebelum
    stage A dijalankan) tidak mungkin terjadi karena stage A selalu tahap pertama.
    """
    if "_orig_idx" not in df.columns:
        raise ValueError("Kolom _orig_idx tidak ada -- pastikan stage A dijalankan lebih dulu.")
    train = df[df["_orig_idx"] <= train_max].drop(columns=["_orig_idx"]).reset_index(drop=True)
    test = df[df["_orig_idx"] >= test_min].drop(columns=["_orig_idx"]).reset_index(drop=True)
    return train, test


def stage_decontaminate(df: pd.DataFrame) -> pd.DataFrame:
    """DEKONTAMINASI. Buang segmen pasca-pelaporan (inti RQ2).

    Dihapus per-baris agar isi laporan asli tidak ikut hilang. Karena tahap A sudah
    mengolaps newline, pemotongan dilakukan pada tingkat frasa penanda.
    """
    df = df.copy()
    df["Description"] = df["Description"].map(lambda t: _norm(POST_REPORT.sub(" ", t)))
    return df


def stage_decon_structural(df: pd.DataFrame) -> pd.DataFrame:
    """DEKONTAMINASI STRUKTURAL. Potong dokumen pada blok komentar pertama.

    Jauh lebih tegas daripada penghapusan keyword: menghapus frasa 'Marking Verified'
    menyisakan seluruh riwayat komentar pasca-resolusi, sedangkan pemotongan struktural
    membuang riwayatnya sekaligus. Dokumen turun drastis (~345 -> ~60 token) dan jauh
    lebih dekat ke kondisi deployment nyata.

    Dokumen tanpa penanda batas dibiarkan utuh; laporkan parse rate-nya.
    """
    def cut(t: str) -> str:
        m = FIRST_COMMENT_END.search(t)
        return _norm(t[: m.start()]) if m else t

    df = df.copy()
    matched = df["Description"].str.contains(FIRST_COMMENT_END)
    df["Description"] = df["Description"].map(cut)
    df = df[df["Description"].str.strip().ne("")].reset_index(drop=True)
    print(f"      parse rate (dokumen punya batas): {matched.mean():.1%}")
    if matched.mean() < 0.20:
        print("      PERINGATAN: parse rate < 20%. Pemotongan struktural hanya "
              "mengenai sebagian kecil dokumen — laporkan sebagai keterbatasan, "
              "dan andalkan DECON berbasis keyword sebagai tahap utama.")
    return df


def stage_tailcut(df: pd.DataFrame) -> pd.DataFrame:
    """DEKONTAMINASI EKOR (TAILCUT). Potong dokumen dari kemunculan PERTAMA penanda
    pasca-pelaporan (POST_REPORT) sampai akhir dokumen.

    Dasar: penanda pasca-laporan berada di EKOR dokumen (probe 7 Okt 2026: median posisi
    0,85; 81,8% di paruh akhir), jadi memotong ekor membuang riwayat komentar pasca-resolusi
    sekaligus. Berbeda dengan STRUCT (parse rate hanya 1,6%), TAILCUT mengenai ~86% dokumen.

    Tanpa fallback: dokumen tanpa penanda dibiarkan utuh (~13,5%). Dokumen yang jadi kosong
    atau < 5 token TIDAK dibuang di sini (supaya komposisi test tetap sebanding dengan
    kondisi lain); jumlahnya dilaporkan. Dokumen kosong dibaca sebagai string kosong oleh
    experiment.py.

    CATATAN SIRKULARITAS (T14): pct_contaminated pasca-TAILCUT = 0% secara konstruksi karena
    regex yang sama dipakai untuk memotong dan mengukur. Bukan bukti bahwa kontaminasi
    hilang; perlu audit manual sampel dokumen terpotong. Audit model pembanding 8 Okt 2026
    (provisional, bukan penilai manusia): ~88% dokumen terpotong masih memuat komentar
    pasca-pelaporan -> TAILCUT adalah dekontaminasi PARSIAL.
    """
    return _tailcut(df, POST_REPORT)


def stage_tailcut_ext(df: pd.DataFrame) -> pd.DataFrame:
    """TAILCUT DIPERLUAS (sensitivitas). Sama dengan TAILCUT, tetapi memotong pada penanda pertama
    dari POST_REPORT_EXT (kosakata alur kerja Bugzilla yang lebih luas). Lihat catatan di atas
    POST_REPORT_EXT: mengutamakan recall, bukan definisi kontaminasi utama."""
    return _tailcut(df, POST_REPORT_EXT)


def tailcut_text(t: str, pattern: re.Pattern = POST_REPORT) -> str:
    """Potong satu teks dari kemunculan pertama `pattern` sampai akhir (tanpa fallback)."""
    m = pattern.search(t)
    return _norm(t[: m.start()]) if m else t


# Pasangan archive <-> laporan awal dari dataset Kaggle kedua (bugs*.csv), dibuat oleh kaggle_laporan_awal.py.
LAPORAN_AWAL_CSV = Path("out_laporan_awal/pasangan_laporan_awal.csv")
_PAIRS_CACHE: dict[str, pd.DataFrame] = {}


def load_initial_pairs(path: Path | None = None, tingkat: str = "utama") -> pd.Series:
    """Kembalikan Series orig_idx -> n_tok_awal (jumlah token laporan awal) untuk pasangan tingkat `tingkat`.
    `path` default = LAPORAN_AWAL_CSV yang dibaca saat fungsi dipanggil (bisa diganti dari luar, mis. tes)."""
    path = Path(LAPORAN_AWAL_CSV if path is None else path)
    key = str(path.resolve())
    if key not in _PAIRS_CACHE:
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} tidak ada -- jalankan dulu: python kaggle_laporan_awal.py")
        _PAIRS_CACHE[key] = pd.read_csv(path)
    p = _PAIRS_CACHE[key]
    p = p[p["tingkat"] == tingkat]
    return p.set_index("orig_idx")["n_tok_awal"]


def initial_text(t: str, n_tok: int) -> str:
    """Laporan awal = n token (spasi) pertama teks archive."""
    return " ".join(t.split()[:int(n_tok)])


def stage_initial(df: pd.DataFrame) -> pd.DataFrame:
    """LAPORAN AWAL (INITIAL). Ganti setiap dokumen dengan bagian laporan awalnya saja, yaitu n token pertama
    menurut pasangan dengan bugs*.csv (kaggle_laporan_awal.py, tingkat 'utama').

    Berbeda dengan TAILCUT yang menebak batas dari penanda leksikal, batas di sini berasal dari teks deskripsi
    pertama Bugzilla, jadi seluruh komentar susulan terbuang di mana pun letaknya. Dokumen TANPA pasangan utama
    DIBUANG (konfigurasi ini = subset ~95% dokumen); laporkan sebagai keterbatasan. Butuh `_orig_idx` (tahap A).
    """
    if "_orig_idx" not in df.columns:
        raise ValueError("Tahap INITIAL butuh _orig_idx -- jalankan tahap A lebih dulu.")
    n_awal = load_initial_pairs()
    keep = df["_orig_idx"].isin(n_awal.index)
    df = df[keep].copy()
    n = df["_orig_idx"].map(n_awal)
    before = df["Description"].str.split().str.len()
    df["Description"] = [initial_text(t, k) for t, k in zip(df["Description"], n)]
    after = df["Description"].str.split().str.len()
    print(f"      dokumen berpasangan (utama): {keep.mean():.1%} ({int(keep.sum())}); tanpa pasangan dibuang: "
          f"{int((~keep).sum())}; token dibuang per dokumen (median): {float((before - after).median()):.0f}")
    return df.reset_index(drop=True)


def _tailcut(df: pd.DataFrame, pattern: re.Pattern) -> pd.DataFrame:
    df = df.copy()
    has_marker = df["Description"].str.contains(pattern)
    df["Description"] = df["Description"].map(lambda t: tailcut_text(t, pattern))
    n_tok = df["Description"].str.split().str.len().fillna(0)
    short = n_tok < 5
    print(f"      dokumen dengan penanda (terpotong): {has_marker.mean():.1%}; "
          f"tanpa penanda (utuh): {(~has_marker).mean():.1%}")
    print(f"      jadi kosong: {int((n_tok == 0).sum())}; jadi < 5 token: {int(short.sum())} "
          f"({short.mean():.1%}); sebaran severity: "
          f"{df.loc[short, 'Severity'].value_counts().to_dict()}")
    return df


def stage_h_identity(df: pd.DataFrame) -> pd.DataFrame:
    """H. Buang identitas pelapor/asignee (nama, email termutilasi).

    Mencegah model belajar 'siapa yang melapor' alih-alih 'apa isi bug'.
    """
    df = df.copy()
    df["Description"] = df["Description"].map(lambda t: _norm(IDENTITY.sub(" ", t)))
    return df


def stage_i_dates(df: pd.DataFrame) -> pd.DataFrame:
    """I. Buang tanggal, nama hari/bulan, tahun, dan zona waktu.

    KRITIS di sini: split-nya kronologis, jadi token tahun ('1998', '2004') adalah
    indikator train/test yang hampir sempurna pada TF-IDF. Tanpa tahap ini, model
    bisa mempelajari partisi waktu itu sendiri, bukan severity.
    """
    df = df.copy()
    df["Description"] = df["Description"].map(lambda t: _norm(DATES.sub(" ", t)))
    return df


def stage_c_boilerplate(df: pd.DataFrame) -> pd.DataFrame:
    """C. Buang boilerplate metadata Bugzilla dan PII (nama, email, URL, timestamp)."""
    df = df.copy()
    df["Description"] = df["Description"].map(lambda t: _norm(BOILERPLATE.sub(" ", t)))
    return df


def stage_d_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """D. Normalisasi token bervariasi tinggi (hash, alamat memori, angka).

    Dibuang, bukan diganti placeholder: placeholder seperti <hash> tidak ada di
    vocab.lst sehingga justru menambah OOV.
    """
    def norm(t: str) -> str:
        for pat, repl in HIGH_CARDINALITY:
            t = pat.sub(repl, t)
        return _norm(t)

    df = df.copy()
    df["Description"] = df["Description"].map(norm)
    return df


def stage_e_internal_dedup(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """E1. Buang pengulangan blok n-gram di dalam satu dokumen (artefak quote-reply)."""
    def dedupe(t: str) -> str:
        w = t.split()
        if len(w) < 2 * n:
            return t
        out, seen, i = [], set(), 0
        while i < len(w):
            gram = " ".join(w[i:i + n])
            if len(w[i:i + n]) == n and gram in seen:
                i += n
                continue
            seen.add(gram)
            out.append(w[i])
            i += 1
        return " ".join(out)

    df = df.copy()
    df["Description"] = df["Description"].map(dedupe)
    return df


def stage_f_uninformative(df: pd.DataFrame, min_tokens: int = 5) -> pd.DataFrame:
    """F. Saring dokumen tak informatif ('test', 'Ugh OK', dst)."""
    tok = df["Description"].str.split().str.len()
    keep = (
        tok.ge(min_tokens)
        & ~df["Description"].str.match(NOISE_DOC)
        & df["Description"].str.contains(r"[A-Za-z]{3,}", regex=True)
    )
    removed = df[~keep]
    if len(removed):
        dist = removed["Severity"].value_counts().to_dict()
        print(f"      dibuang {len(removed)} dok; sebaran severity: {dist}")
        # Peringatan kalau filter menggerus kelas minoritas secara tidak proporsional
        for cls in ("blocker", "trivial"):
            n_rm = dist.get(cls, 0)
            n_all = int((df["Severity"] == cls).sum())
            if n_all and n_rm / n_all > 0.05:
                print(f"      PERINGATAN: {n_rm}/{n_all} ({n_rm / n_all:.1%}) kelas "
                      f"'{cls}' terbuang. Pertimbangkan turunkan min_tokens.")
    return df[keep].reset_index(drop=True)


def stage_e2_near_dup(df: pd.DataFrame) -> pd.DataFrame:
    """E2. Dedup antar dokumen. HARUS dijalankan TERAKHIR.

    Near-duplicate baru muncul setelah boilerplate dan metadata dibuang: dokumen
    yang tadinya berbeda hanya di bagian metadata kini menjadi identik.
    """
    key = df["Description"].str.lower().str.replace(r"\s+", " ", regex=True)
    n_before = len(df)

    if "_orig_idx" not in df.columns:
        print("      PERINGATAN: _orig_idx tidak ada; dedup memakai seluruh data (tanpa batas split).")
        df = df[~key.duplicated()].reset_index(drop=True)
        if n_before != len(df):
            print(f"      near-duplicate dibuang: {n_before - len(df)}")
        return df

    # Dedup EKSPLISIT per split (Kapoor & Narayanan 2023, L1.4: duplikat lintas train/test):
    #   - train didedup hanya terhadap dirinya sendiri (test tidak boleh memengaruhi train);
    #   - test didedup terhadap dirinya sendiri DAN terhadap train.
    # Baris di antara TRAIN_MAX dan TEST_MIN dibuang oleh split_chronological(), jadi tidak
    # dipakai sebagai pembanding dan dibiarkan apa adanya.
    is_train = df["_orig_idx"] <= TRAIN_MAX
    is_test = df["_orig_idx"] >= TEST_MIN
    train_keys = set(key[is_train])
    key_tr, key_te = key.where(is_train), key.where(is_test)
    dup_train = is_train & key_tr.duplicated()
    dup_test_self = is_test & key_te.duplicated()
    dup_test_vs_train = is_test & key.isin(train_keys)
    drop = dup_train | dup_test_self | dup_test_vs_train
    df = df[~drop].reset_index(drop=True)
    if drop.any():
        print(f"      near-duplicate dibuang: {int(drop.sum())} "
              f"(train-internal {int(dup_train.sum())}, test-internal {int(dup_test_self.sum())}, "
              f"test=train {int((dup_test_vs_train & ~dup_test_self).sum())})")
    return df


def stage_g_truncate(df: pd.DataFrame, quantile: float = 0.95) -> pd.DataFrame:
    """G. Potong outlier panjang. Dokumen TIDAK dibuang — bug kompleks cenderung severe."""
    tok = df["Description"].str.split().str.len()
    # Batas dihitung dari TRAIN saja, lalu diterapkan ke semua dokumen (Kapoor & Narayanan
    # 2023, L1.2: pre-processing tidak boleh memakai statistik test).
    if "_orig_idx" in df.columns:
        cap = int(tok[df["_orig_idx"] <= TRAIN_MAX].quantile(quantile))
        basis = "train saja"
    else:
        cap = int(tok.quantile(quantile))
        basis = "seluruh data; _orig_idx tidak ada"
    df = df.copy()
    df["n_tokens_original"] = tok           # panjang itu sendiri sering prediktif
    df["Description"] = df["Description"].map(lambda t: " ".join(t.split()[:cap]))
    print(f"      cap = {cap} token (kuantil {quantile}, {basis})")
    return df


# ------------------------------------------------------------ konfigurasi

STAGES = {
    "A": ("struktur dasar", stage_a_structure),
    "DECON": ("dekontaminasi keyword", stage_decontaminate),
    "STRUCT": ("dekontaminasi struktural", stage_decon_structural),
    "TAILCUT": ("dekontaminasi ekor", stage_tailcut),
    "TAILCUT_EXT": ("dekontaminasi ekor diperluas", stage_tailcut_ext),
    "INITIAL": ("laporan awal (pasangan Kaggle)", stage_initial),
    "C": ("boilerplate + PII", stage_c_boilerplate),
    "H": ("buang identitas pelapor", stage_h_identity),
    "I": ("buang tanggal/tahun", stage_i_dates),
    "D": ("normalisasi token", stage_d_tokens),
    "E1": ("dedup internal dokumen", stage_e_internal_dedup),
    "F": ("saring tak informatif", stage_f_uninformative),
    "E2": ("dedup antar dokumen", stage_e2_near_dup),
    "G": ("truncate outlier", stage_g_truncate),
}

# Konfigurasi ablasi. Setiap entri = daftar tahap yang dijalankan berurutan.
CONFIGS: dict[str, list[str]] = {
    # Titik nol: data apa adanya. CATATAN: ini BUKAN "data mentah" —
    # tanda baca sudah dihapus di hulu dan 597 baris sudah dibuang sebelum
    # dataset ini kita terima. Nyatakan itu sebagai threat to construct validity.
    "raw":          [],
    "baseline":     ["A"],
    # --- ablasi satu-tahap: tiap baris menjawab "tahap ini berpengaruh berapa?" (RQ3)
    "plus_decon":   ["A", "DECON"],                                  # <- RQ2 (keyword)
    "plus_struct":  ["A", "STRUCT"],                                 # <- RQ2 (struktural)
    "plus_tailcut": ["A", "TAILCUT"],                                # <- RQ2 (ekor; konfigurasi utama Paper 1)
    "plus_tailcut_ext": ["A", "TAILCUT_EXT"],                        # <- RQ2 sensitivitas (penanda diperluas)
    "plus_initial": ["A", "INITIAL"],                                # <- RQ2 laporan awal (subset berpasangan)
    "plus_boiler":  ["A", "C"],
    "plus_identity": ["A", "H"],                                     # kebocoran identitas
    "plus_dates":   ["A", "I"],                                      # tahun = indikator train/test
    "plus_tokens":  ["A", "D"],
    "plus_dedup":   ["A", "E1", "E2"],
    "plus_filter":  ["A", "F"],
    "plus_trunc":   ["A", "G"],
    # --- konfigurasi lengkap
    "full_nodecon": ["A", "C", "H", "I", "D", "E1", "F", "E2", "G"],           # pembanding RQ2
    "full":         ["A", "DECON", "C", "H", "I", "D", "E1", "F", "E2", "G"],  # pipeline lengkap
    "full_struct":  ["A", "STRUCT", "C", "H", "I", "D", "E1", "F", "E2", "G"], # dekon paling tegas
    "full_tailcut": ["A", "TAILCUT", "C", "H", "I", "D", "E1", "F", "E2", "G"],   # task 11 Okt: cleaning lengkap + TAILCUT
    "full_initial": ["A", "INITIAL", "C", "H", "I", "D", "E1", "F", "E2", "G"],   # cleaning lengkap + laporan awal
}


def run_config(df0: pd.DataFrame, name: str, stages: list[str]) -> tuple[pd.DataFrame, dict]:
    print(f"\n[{name}] tahap: {' -> '.join(stages) if stages else '(tidak ada)'}")
    df = df0.copy()
    if not stages:                                   # 'raw' tetap perlu kolom rapi + _orig_idx
        junk = [c for c in df.columns if c.startswith("Unnamed") or not str(c).strip()]
        if junk:
            df["_orig_idx"] = df[junk[-1]]
        df = df.drop(columns=junk)
        df["Description"] = df["Description"].astype(str)

    for s in stages:
        label, fn = STAGES[s]
        n0, t0 = len(df), df["Description"].str.split().str.len().mean()
        df = fn(df)
        t1 = df["Description"].str.split().str.len().mean()
        print(f"    {s:<6} {label:<32} baris {n0:>6} -> {len(df):<6} "
              f"token/dok {t0:>6.0f} -> {t1:.0f}")

    tok = df["Description"].str.split().str.len()
    counts = df["Severity"].value_counts().to_dict()
    total = len(df)
    top = max(counts, key=counts.get)
    prec = counts[top] / total
    f1_base = (2 * prec / (prec + 1)) / len(counts)

    stats = {
        "config": name,
        "stages": "+".join(stages) if stages else "-",
        "n_rows": total,
        "tokens_mean": round(float(tok.mean()), 1),
        "tokens_median": int(tok.median()),
        "tokens_p99": int(tok.quantile(0.99)),
        "vocab_size": len(set(w for t in df["Description"] for w in t.split())),
        "pct_contaminated": round(
            100 * df["Description"].str.contains(POST_REPORT).mean(), 2
        ),
        "n_trivial": counts.get("trivial", 0),
        "n_blocker": counts.get("blocker", 0),
        "baseline_acc": round(prec, 4),
        "baseline_macro_f1": round(f1_base, 4),
    }
    return df, stats


def merge_analysis(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    """Gabungkan baris baru ke cleaning_analysis.csv yang sudah ada (menimpa config yang sama),
    lalu urutkan sesuai CONFIGS. Mencegah `--config X` menghapus baris konfigurasi lain —
    penyebab tabel harus disambung manual dulu (header korup '2config', 11 Sep 2026)."""
    if path.exists():
        old = pd.read_csv(path)
        if "config" in old.columns:
            new = pd.concat([old[~old["config"].isin(new["config"])], new], ignore_index=True)
    order = {k: i for i, k in enumerate(CONFIGS)}
    return new.sort_values("config", key=lambda s: s.map(order).fillna(len(order))).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="archive")
    # Default folder bertanggal: run polos tidak menulis ke out/ yang basi dan tidak menimpa
    # folder hasil resmi (out_20261008/, out_20261011/) — D-5.
    ap.add_argument("--out", default=f"out_{date.today():%Y%m%d}")
    ap.add_argument("--config", help="jalankan satu konfigurasi saja")
    ap.add_argument("--list", action="store_true", help="tampilkan daftar konfigurasi")
    args = ap.parse_args()

    if args.list:
        print("Konfigurasi tersedia:\n")
        for k, v in CONFIGS.items():
            print(f"  {k:<14} {' -> '.join(v) if v else '(tanpa tahap)'}")
        return 0

    src = Path(args.archive) / "sev.csv"
    if not src.exists():
        print(f"FATAL: {src} tidak ditemukan.")
        return 1

    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    print(f"Memuat {src} ...")
    df0 = pd.read_csv(src)
    print(f"  {len(df0)} baris, kolom: {list(df0.columns)}")

    todo = {args.config: CONFIGS[args.config]} if args.config else CONFIGS
    if args.config and args.config not in CONFIGS:
        print(f"FATAL: konfigurasi '{args.config}' tidak dikenal. Pakai --list.")
        return 1

    rows = []
    for name, stages in todo.items():
        if "INITIAL" in stages and not LAPORAN_AWAL_CSV.exists():
            print(f"\n[{name}] DILEWATI: {LAPORAN_AWAL_CSV} tidak ada (jalankan python kaggle_laporan_awal.py).")
            continue
        df, stats = run_config(df0, name, stages)
        df_out = df.drop(columns=["_orig_idx"]) if "_orig_idx" in df.columns else df
        df_out.to_csv(outdir / f"cleaned_{name}.csv", index=False)

        # Split kronologis SETELAH cleaning (per instruksi asli), pakai _orig_idx dari stage A.
        if "_orig_idx" in df.columns:
            tr, te = split_chronological(df)
            tr.to_csv(outdir / f"cleaned_{name}_train.csv", index=False)
            te.to_csv(outdir / f"cleaned_{name}_test.csv", index=False)
            stats["n_train_split"] = len(tr)
            stats["n_test_split"] = len(te)
        rows.append(stats)

    analysis = merge_analysis(outdir / "cleaning_analysis.csv", pd.DataFrame(rows))
    analysis.to_csv(outdir / "cleaning_analysis.csv", index=False)

    print(f"\n{'=' * 78}\nRINGKASAN (bahan Tabel 4)\n{'=' * 78}")
    print(analysis.to_string(index=False))
    print(f"\nDisimpan di {outdir}/")
    print("\nLangkah berikutnya:")
    print("  1. Bandingkan 'full' vs 'full_nodecon' -> menjawab RQ2")
    print("  2. Bandingkan tiap 'plus_*' vs 'baseline' -> menjawab RQ3")
    print("  3. Perhatikan kolom n_trivial/n_blocker: kalau anjlok, filter terlalu agresif")
    print("  4. cleaned_<config>_train.csv / _test.csv sudah displit kronologis -- pakai")
    print("     langsung di experiment.py --train/--test, jangan split ulang")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
