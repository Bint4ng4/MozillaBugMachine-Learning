"""Figure 2 (POC): top-30 weighted tokens of the LogReg model (ref arm), colored by token type.

Input : results_bobot_logreg/asal_token_top30.csv (made by task_asal_token.py)
Output: POC/figure2_top30_tokens.png and .pdf
Color marks token type; the number at the bar end = % of documents that contain the token ONLY in follow-up text.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

SEVERITY = {"blocker", "critical", "major", "minor", "trivial", "to blocker", "as blocker",
            "is blocker", "blocking"}
SYMPTOM = {"crash", "crashes", "crashing"}
LIFECYCLE = {"verif"}

CATEGORIES = {
    "sev": ("Severity word (echoes the label)", "#D55E00"),
    "sym": ("Crash symptom word (valid at triage)", "#0072B2"),
    "life": ("Lifecycle marker (post-report)", "#CC79A7"),
    "other": ("Other token / metadata", "#8C8C8C"),
}


def category(tok: str) -> str:
    if tok in SEVERITY:
        return "sev"
    if tok in SYMPTOM:
        return "sym"
    if tok in LIFECYCLE:
        return "life"
    return "other"


def main() -> int:
    df = pd.read_csv(ROOT / "results_bobot_logreg" / "asal_token_top30.csv")
    assert len(df) == 30, len(df)
    df["cat"] = df["token"].map(category)
    df["pct_followup_only"] = 100 * df["n_hanya_susulan"] / df["n_dok_penuh"]
    df = df.sort_values("koefisien", key=abs, ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.2, 8.0))
    ax.barh(df["token"], df["koefisien"], color=[CATEGORIES[c][1] for c in df["cat"]], height=0.72)
    ax.axvline(0, color="black", lw=0.8)
    xmax = df["koefisien"].abs().max()
    for i, r in df.iterrows():
        off = 0.15 * xmax / 6
        x = r["koefisien"] + (off if r["koefisien"] >= 0 else -off)
        ax.text(x, i, f"{r['pct_followup_only']:.0f}%", va="center",
                ha="left" if r["koefisien"] >= 0 else "right", fontsize=7.5, color="#333333")
    ax.set_xlim(-xmax * 0.4, xmax * 1.12)
    ax.set_xlabel("Logistic Regression coefficient (on the token's owner class)")
    ax.set_title("Top-30 weighted tokens, model trained on full documents", fontsize=11)
    ax.tick_params(axis="y", labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in CATEGORIES.values()]
    ax.legend(handles, [n for n, _ in CATEGORIES.values()], loc="lower right", fontsize=8, frameon=False)
    fig.text(0.01, 0.005,
             "Number at bar end: % of documents containing the token only in follow-up text "
             "(not in the initial report). n = 30,748 paired documents.",
             fontsize=6.8, color="#444444", ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figure2_top30_tokens.{ext}", dpi=200)
    print(df[["token", "koefisien", "cat", "pct_followup_only"]].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
