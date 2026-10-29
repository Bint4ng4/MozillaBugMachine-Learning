"""
rq2_pralatih_kaggle.py — Uji ketahanan RQ2 dengan model pralatih (BERT) di GPU Kaggle (keputusan user 14 Okt 2026).

Tujuan: menguji apakah over-estimasi RQ2 (§8f: skor pada dokumen penuh 3,5–6,2 poin lebih tinggi daripada skor model
yang sama saat dipakai pada laporan awal) juga terjadi pada model yang membaca konteks, bukan hanya pada TF-IDF + model
linear. Ini BUKAN upaya mengejar skor tertinggi; RQ tidak berubah.

Kondisi (sama dengan §8f, dokumen dan urutan test sama):
  ref             latih pada dokumen penuh, uji pada dokumen penuh   (skor yang biasa dilaporkan)
  ref_to_initial  model `ref` yang sama, uji pada laporan awal       (kondisi saat triase)
  initial         latih dan uji pada laporan awal

Protokol: bert-base-uncased (korpus pralatih BooksCorpus + Wikipedia, sehingga laporan bug Mozilla kecil kemungkinannya
ikut dihafal), resep standar tanpa penyetelan: lr 2e-5, batch 16, maks 4 epoch, warmup 6%, weight decay 0,01, fp16,
cross-entropy berbobot kelas seimbang (sama dengan model linear). Latih pada train_fit, pilih epoch terbaik pada val
kronologis (macro-F1), lalu test dipakai sekali. 3 seed. Dokumen > 512 token dipotong: bawaan bagian kepala (dicatat %);
varian opsional head_tail lewat RQ2_TRUNC=head_tail.

Masukan : Kaggle — dataset PRIVAT berisi rq2_pralatih.csv (dari ekspor_data_pralatih.py);
          Colab  — Google Drive/rq2_pralatih/rq2_pralatih.csv (Drive di-mount)
Keluaran: prediksi_pralatih.csv, log_epoch.csv, info_run.json di /kaggle/working (Kaggle) atau
          Drive/rq2_pralatih/hasil (Colab). Seed yang sudah selesai dilewati bila dijalankan ulang.
          Analisis: analisis_pralatih.py (lokal)

Uji kecil (600 dokumen, 128 token, 1 epoch, 1 seed):  RQ2_SMOKE=1  (di sel pengaturan notebook)
"""

import glob
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

SMOKE = os.environ.get("RQ2_SMOKE") == "1"
CFG = {
    "model_name": "bert-base-uncased",   # uji kecil memakai model yang sama: menguji jalur yang persis sama
    "max_len": 128 if SMOKE else 512,
    "batch_size": 16,
    "eval_batch_size": 64,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "epochs": 1 if SMOKE else 4,
    "seeds": [0] if SMOKE else [0, 1, 2],
    "fp16": True,
    "grad_clip": 1.0,
    "truncation": os.environ.get("RQ2_TRUNC", "head"),   # "head" (standar) atau "head_tail" (128 awal + sisa akhir)
}
CLS = ["trivial", "minor", "normal", "major", "critical", "blocker"]
TO_INT = {c: i for i, c in enumerate(CLS)}
K = len(CLS)


COLAB_DIR = Path("/content/drive/MyDrive/rq2_pralatih")      # Google Colab: data & hasil di Google Drive


def find_data() -> Path:
    if os.environ.get("RQ2_DATA"):
        return Path(os.environ["RQ2_DATA"])
    hits = glob.glob("/kaggle/input/**/rq2_pralatih.csv", recursive=True)
    hits += [str(p) for p in (COLAB_DIR / "rq2_pralatih.csv", Path("/content/rq2_pralatih.csv")) if p.exists()]
    if not hits:
        raise SystemExit("rq2_pralatih.csv tidak ditemukan. Kaggle: tambahkan dataset lewat 'Add Input'. "
                         "Colab: taruh di Google Drive/rq2_pralatih/rq2_pralatih.csv dan jalankan sel mount Drive.")
    return Path(hits[0])


def out_dir() -> Path:
    """Kaggle: /kaggle/working. Colab: folder Drive (bertahan bila sesi putus). Lainnya: folder lokal."""
    if Path("/kaggle/working").exists():
        base = Path("/kaggle/working")
    elif COLAB_DIR.parent.exists():
        base = COLAB_DIR / "hasil"
    elif Path("/content").exists():
        base = Path("/content/keluaran_pralatih")
        print("PERINGATAN: Drive tidak ter-mount; hasil hilang bila sesi Colab putus.", flush=True)
    else:
        base = Path("keluaran_pralatih")
    return base / "smoke" if SMOKE else base


OUT = out_dir()
OUT.mkdir(parents=True, exist_ok=True)
if SMOKE:                                    # uji kecil selalu mulai bersih (tidak melanjutkan uji kecil sebelumnya)
    for f in ("prediksi_pralatih.csv", "log_epoch.csv", "info_run.json"):
        (OUT / f).unlink(missing_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = CFG["fp16"] and DEV.type == "cuda"
N_GPU = torch.cuda.device_count()


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def encode(tok, texts) -> tuple[list[list[int]], float]:
    """Tokenisasi penuh lalu potong ke max_len; kembalikan juga % dokumen yang terpotong.

    head      : CLS + token pertama + SEP (standar; ekor dokumen, tempat komentar susulan, ikut terbuang)
    head_tail : 128 token pertama + token terakhir (Sun dkk. 2019) — varian opsional, menyisakan komentar penutup
    """
    ids = tok(list(texts), add_special_tokens=True, truncation=False)["input_ids"]
    L = CFG["max_len"]
    if CFG["truncation"] == "head_tail":
        cut = [x if len(x) <= L else x[:128] + x[-(L - 128):] for x in ids]
    else:
        cut = [x if len(x) <= L else x[:L - 1] + [tok.sep_token_id] for x in ids]
    return cut, 100.0 * float(np.mean([len(x) > L for x in ids]))


def batches(ids, y, bs, pad_id, shuffle, rng):
    n = len(ids)
    if shuffle:   # acak, lalu urutkan panjang di dalam blok besar agar padding hemat; urutan batch diacak lagi
        order = rng.permutation(n)
        blocks = [sorted(order[i:i + bs * 50], key=lambda j: len(ids[j])) for i in range(0, n, bs * 50)]
        groups = [b[i:i + bs] for b in blocks for i in range(0, len(b), bs)]
        rng.shuffle(groups)
    else:
        order = np.argsort([len(x) for x in ids], kind="stable")
        groups = [order[i:i + bs] for i in range(0, n, bs)]
    for g in groups:
        g = np.asarray(g)
        Lb = max(len(ids[j]) for j in g)
        inp = np.full((len(g), Lb), pad_id, dtype=np.int64)
        att = np.zeros((len(g), Lb), dtype=np.int64)
        for r, j in enumerate(g):
            inp[r, :len(ids[j])] = ids[j]
            att[r, :len(ids[j])] = 1
        yield g, torch.from_numpy(inp).to(DEV), torch.from_numpy(att).to(DEV), torch.as_tensor(y[g], device=DEV)


def unwrap(m):
    return m.module if isinstance(m, torch.nn.DataParallel) else m


@torch.no_grad()
def predict(model, ids, y, pad_id, loss_fn):
    model.eval()
    pred = np.empty(len(ids), dtype=np.int64)
    tot, cnt = 0.0, 0
    for g, inp, att, yb in batches(ids, y, CFG["eval_batch_size"], pad_id, False, None):
        with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=USE_AMP):
            logits = model(input_ids=inp, attention_mask=att).logits.float()
        tot += float(loss_fn(logits, yb)) * len(g)
        cnt += len(g)
        pred[g] = logits.argmax(-1).cpu().numpy()
    return pred, tot / cnt


def make_scaler():
    try:
        return torch.amp.GradScaler(DEV.type, enabled=USE_AMP)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=USE_AMP)


def train_condition(seed, cond, tok, data, class_w, log_rows):
    """Latih satu kondisi (teks latih = data[cond]['train']), pilih epoch terbaik di val, kembalikan prediksi test."""
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(CFG["model_name"], num_labels=K).to(DEV)
    if N_GPU > 1:
        model = torch.nn.DataParallel(model)
    no_decay = ("bias", "LayerNorm.weight")
    params = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
               "weight_decay": CFG["weight_decay"]},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=CFG["lr"])
    tr_ids, y_tr = data[cond]["train_ids"], data["y_train"]
    steps = CFG["epochs"] * math.ceil(len(tr_ids) / CFG["batch_size"])
    sched = get_linear_schedule_with_warmup(opt, int(CFG["warmup_ratio"] * steps), steps)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_w)
    scaler = make_scaler()
    rng = np.random.default_rng(seed)
    best = {"f1": -1.0, "epoch": 0, "state": None}
    for ep in range(1, CFG["epochs"] + 1):
        model.train()
        t1, tot, cnt = time.time(), 0.0, 0
        for g, inp, att, yb in batches(tr_ids, y_tr, CFG["batch_size"], tok.pad_token_id, True, rng):
            with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=USE_AMP):
                logits = model(input_ids=inp, attention_mask=att).logits.float()
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss) * len(g)
            cnt += len(g)
        vp, vloss = predict(model, data[cond]["val_ids"], data["y_val"], tok.pad_token_id, loss_fn)
        vf1 = f1_score(data["y_val"], vp, average="macro", zero_division=0)
        log_rows.append({"seed": seed, "kondisi": cond, "epoch": ep, "train_loss": round(tot / cnt, 4),
                         "val_loss": round(vloss, 4), "val_macro_f1": round(float(vf1), 4),
                         "detik": round(time.time() - t1)})
        print(f"  seed {seed} {cond:<8} epoch {ep}: train_loss {tot / cnt:.4f}  val_loss {vloss:.4f}  "
              f"val macro-F1 {vf1:.4f}  ({time.time() - t1:.0f} dtk)", flush=True)
        if vf1 > best["f1"]:
            best = {"f1": float(vf1), "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in unwrap(model).state_dict().items()}}
    unwrap(model).load_state_dict(best["state"])
    tests = {"ref": ["ref", "ref_to_initial"], "init": ["initial"]}[cond]
    out = {}
    for arm in tests:
        ids = data["init" if arm in ("initial", "ref_to_initial") else "ref"]["test_ids"]
        out[arm], _ = predict(model, ids, data["y_test"], tok.pad_token_id, loss_fn)
    del model, opt
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return out, best["epoch"], best["f1"]


def main() -> None:
    t0 = time.time()
    df = pd.read_csv(find_data())
    if SMOKE:
        df = pd.concat([df[df.split == "train_fit"].head(300), df[df.split == "val"].head(100),
                        df[df.split == "test"].head(200)], ignore_index=True)
    sp = {s: df[df.split == s].reset_index(drop=True) for s in ("train_fit", "val", "test")}
    print(f"data: { {s: len(v) for s, v in sp.items()} }; perangkat {DEV} x{max(N_GPU, 1)} "
          f"{torch.cuda.get_device_name(0) if N_GPU else ''}", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(CFG["model_name"])
    except (ValueError, OSError):            # cadangan bila AutoTokenizer gagal menebak kelas tokenizer
        from transformers import BertTokenizer
        tok = BertTokenizer.from_pretrained(CFG["model_name"])
    data = {"y_train": sp["train_fit"]["Severity"].map(TO_INT).values,
            "y_val": sp["val"]["Severity"].map(TO_INT).values,
            "y_test": sp["test"]["Severity"].map(TO_INT).values}
    trunc = {}
    for cond, col in (("ref", "text_ref"), ("init", "text_init")):
        data[cond] = {}
        for s in ("train_fit", "val", "test"):
            ids, pct = encode(tok, sp[s][col].fillna("").astype(str))
            data[cond][{"train_fit": "train_ids", "val": "val_ids", "test": "test_ids"}[s]] = ids
            trunc[f"{col}|{s}"] = round(pct, 1)
    print(f"% dokumen > {CFG['max_len']} token (dipotong): {trunc}  ({time.time() - t0:.0f} dtk)", flush=True)
    counts = np.bincount(data["y_train"], minlength=K)
    class_w = torch.tensor(len(data["y_train"]) / (K * np.maximum(counts, 1)), dtype=torch.float32, device=DEV)

    pred_path, log_path = OUT / "prediksi_pralatih.csv", OUT / "log_epoch.csv"
    prev = glob.glob("/kaggle/input/**/prediksi_pralatih.csv", recursive=True)
    if prev and not pred_path.exists():       # lanjutan: keluaran run sebelumnya dipasang sebagai input
        src = Path(prev[0]).parent
        for f in ("prediksi_pralatih.csv", "log_epoch.csv", "info_run.json"):
            if (src / f).exists():
                (OUT / f).write_bytes((src / f).read_bytes())
        print(f"melanjutkan dari {src}", flush=True)
    done = set(pd.read_csv(pred_path)["seed"].unique()) if pred_path.exists() else set()
    info = {"config": CFG, "smoke": SMOKE, "n": {s: len(v) for s, v in sp.items()}, "persen_dipotong": trunc,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if N_GPU else "cpu", "n_gpu": N_GPU,
            "bobot_kelas": dict(zip(CLS, [round(float(w), 4) for w in class_w.cpu()])), "epoch_terbaik": {}}
    if (OUT / "info_run.json").exists():                      # lanjutan run: pertahankan catatan seed sebelumnya
        info["epoch_terbaik"] = json.loads((OUT / "info_run.json").read_text(encoding="utf-8")).get("epoch_terbaik", {})
    for seed in CFG["seeds"]:
        if seed in done:
            print(f"seed {seed} sudah ada di {pred_path}, dilewati", flush=True)
            continue
        log_rows, preds = [], {}
        for cond in ("ref", "init"):
            p, ep, vf1 = train_condition(seed, cond, tok, data, class_w, log_rows)
            preds.update(p)
            info["epoch_terbaik"][f"{seed}|{cond}"] = {"epoch": ep, "val_macro_f1": round(vf1, 4)}
        rows = [pd.DataFrame({"seed": seed, "lengan": arm, "_orig_idx": sp["test"]["_orig_idx"].values,
                              "y_true": data["y_test"], "y_pred": p}) for arm, p in preds.items()]
        pd.concat(rows).to_csv(pred_path, mode="a", header=not pred_path.exists(), index=False)
        pd.DataFrame(log_rows).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
        for arm, p in preds.items():
            print(f"  seed {seed} test {arm:<15} macro-F1 {f1_score(data['y_test'], p, average='macro'):.4f}", flush=True)
        info["detik"] = round(time.time() - t0)
        (OUT / "info_run.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Selesai {time.time() - t0:.0f} dtk -> {OUT}", flush=True)


main()
