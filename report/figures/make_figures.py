"""Figures for the report. Numbers are copied from PLAN.md / runs/ (see comments); run from the repo root:

    conda run -n vdk1 python report/figures/make_figures.py
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")
from eval.so_muc_nen import kq_nguong  # noqa: E402

OUT = "report/figures"
C = {"clear": "#0072B2", "amb": "#E69F00", "oos": "#009E73", "grey": "#666666", "red": "#D55E00"}
plt.rcParams.update({"font.size": 8, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False})

# 1) data rounds — clean test (230), mean of 2 seeds unless marked;
#    PLAN.md sections L, M, O, P, Q, U, V; runs/v6/so_v61.txt
ROUNDS = [  # name, clear, ambiguous(39), out-of-scope
    ("R1$^{1}$", 70.3, 59.0, 63.2),
    ("R2", 75.0, 65.4, 61.1),
    ("R3", 76.3, 66.7, 57.2),
    ("R3+desc", 79.7, 70.5, 68.0),
    ("R3+1k/fn", 75.7, 65.4, 69.2),
    ("R4-repl.$^{\\times}$", 75.7, 65.4, 58.6),
    ("R4-add$^{\\times}$", 75.7, 73.1, 59.4),
    ("Contrast.", 78.4, 69.2, 74.8),
    ("Combined", 78.3, 62.8, 73.0),
    ("v5 full", 79.1, 76.9, 75.6),
    ("v5 LoRA", 79.1, 76.9, 83.3),
    ("v6.1 LoRA", 78.4, 75.6, 77.4),  # đếm câu: 116/148, 59/78, 181/234 (bản cũ lấy tb hai % đã làm tròn)
]
fig, ax = plt.subplots(figsize=(6.3, 2.3))
w = 0.27
SERIES = (("clear", "Clear (74)"), ("amb", "Ambiguous (39)"), ("oos", "Out-of-scope (117)"))
for k, (key, lab) in enumerate(SERIES):
    xs = [i + (k - 1) * w for i in range(len(ROUNDS))]
    ax.bar(xs, [r[k + 1] for r in ROUNDS], w, color=C[key], label=lab)
ax.set_xticks(range(len(ROUNDS)), [r[0] for r in ROUNDS], fontsize=7, rotation=25, ha="right")
ax.set_ylim(50, 100)
ax.set_ylabel("Accuracy (%)")
ax.axhline(94.9, color=C["grey"], lw=0.8, ls="--")
ax.text(len(ROUNDS) - 0.6, 96.0, "72B prompted: out-of-scope 94.9", ha="right", fontsize=6.5, color=C["grey"])
ax.legend(ncol=3, frameon=False, loc="upper left", fontsize=7)
fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT}/rounds.pdf")

# 2) noise robustness — runs/stt/on/ket_qua_on.txt (60 utterances, sets C+D, end-to-end correct)
SNR = ["clean", "20", "10", "5", "0"]
NOISE = {
    "In-car (DEMAND TCAR)": ([38, 37, 35, 36, 36], C["clear"], "o"),
    "Brown noise (1/f$^2$)": ([38, 36, 37, 37, 33], C["oos"], "s"),
    "Background speech (FLEURS)": ([38, 34, 27, 16, 13], C["red"], "^"),
}
fig, ax = plt.subplots(figsize=(3.0, 2.0))
for lab, (ys, col, m) in NOISE.items():
    ax.plot(range(5), ys, marker=m, color=col, label=lab, lw=1.2, ms=3.5)
ax.axhline(42, color=C["grey"], lw=0.8, ls="--")
ax.text(4, 43, "typed transcript: 42", ha="right", fontsize=6.5, color=C["grey"])
ax.set_xticks(range(5), SNR)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("End-to-end correct / 60")
ax.set_ylim(0, 48)
ax.legend(frameon=False, fontsize=6.5, loc="lower left")
fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT}/noise.pdf")

# 3) threshold trade-off of the deployed model — runs/gguf/test_v61s42_q4.jsonl
#    (descriptive; the threshold was chosen on the calibration set)
rows = [json.loads(x) for x in open("runs/gguf/test_v61s42_q4.jsonl", encoding="utf-8")]
S = [d for d in rows if d["tap"] == "test_vi_sach_v2"]
R = [d for d in S if d["nhom"] == "ro_rang"]
N = [d for d in S if d["nhom"] == "ngoai_pham_vi"]
ts = [i / 100 for i in range(0, 100)]
cov = [100 * sum(kq_nguong(d, t) == "dung" for d in R) / len(R) for t in ts]
risk = [100 * sum(kq_nguong(d, t) == "goi_bua" for d in N) / len(N) for t in ts]
fig, ax = plt.subplots(figsize=(3.0, 2.0))
ax.plot(risk, cov, color=C["clear"], lw=1.2)
OFFSET = {0.0: (-30, -11), 0.80: (2, -12), 0.95: (5, -9)}
for t, lab in ((0.0, "no threshold"), (0.80, "0.80 (deployed)"), (0.95, "0.95")):
    i = ts.index(t)
    ax.plot(risk[i], cov[i], "o", color=C["red"] if t == 0.80 else C["grey"], ms=3.5, zorder=3)
    ax.annotate(lab, (risk[i], cov[i]), textcoords="offset points", xytext=OFFSET[t], fontsize=6.5)
ax.set_xlabel("Over-triggering on out-of-scope (%)")
ax.set_ylabel("Clear executed (%)")
fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT}/threshold.pdf")
print("ok", [(t, round(risk[ts.index(t)], 1), round(cov[ts.index(t)], 1))
             for t in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95)])
