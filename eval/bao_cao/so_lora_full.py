"""FULL vs LoRA trên dữ liệu v5 — cùng dữ liệu, cùng hạt giống (PLAN.md mục V)."""
import json
import math
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from eval.metrics import cham_mot_cau, doc_output
from gen.records import query_from_call_prompt as q
from schema.loader import CATALOG

TU_CHOI = CATALOG.by_name["irrelevant_function"].nexa_token
ASK = CATALOG.nexa_ask


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def kq_nguong(d, t):
    (t1, p1), (t2, _) = d["top_buoc_1"][0], d["top_buoc_1"][1]
    if t1 in (TU_CHOI, ASK) or p1 >= t:
        return d["ket_qua"]
    qd = "tu_choi" if t2 == TU_CHOI else "hoi_lai"
    if d["nhom"] == "ngoai_pham_vi":
        return "dung" if qd == "tu_choi" else "hoi_lai_khi_phai_tu_choi"
    if d["nhom"] == "ro_rang":
        return "tu_choi_nham" if qd == "tu_choi" else "hoi_lai_thua"
    return "mo_ho_" + qd


def doc(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")]


DEV = doc("data/dev_tay_2/dev_vi_2.jsonl")


def chi_so(test_path, mh4_path):
    R = doc(test_path)
    S = [d for d in R if d["tap"] == "test_vi_sach_v2"]
    o = {}
    for t, nh in ((0.0, ""), (0.90, " +ngưỡng")):
        for n in ("ro_rang", "mo_ho", "ngoai_pham_vi"):
            N = [d for d in S if d["nhom"] == n]
            o[n + nh] = 100 * sum(kq_nguong(d, t) == "dung" for d in N) / len(N)
        o["từ chối nhầm %" + nh] = 100 * sum(kq_nguong(d, t) == "tu_choi_nham" for d in S) / len(S)
    B = [d for d in R if d["tap"] == "test_vi_bo_sung"]
    o["28 câu"] = 100 * sum(d["ket_qua"] == "dung" for d in B) / len(B)
    Dv = {d["cau"]: d["output"] for d in R if d["tap"] == "dev_vi_2"}
    tc = ld = 0
    for r in DEV:
        out = Dv[q(r["prompt"])]
        if r["meta"]["nhom_phu"].startswith("dev_lam_duoc"):
            p, g = doc_output(out), doc_output(r["completion"])
            ld += p.doc_duoc and p.ten_ham[:1] == g.ten_ham[:1]
        else:
            tc += cham_mot_cau(out, r["completion"])[0] == "dung"
    o["dev từ chối /20"], o["dev làm được /10"] = tc, ld
    M = doc(mh4_path)
    o["mo_ho_4"] = 100 * sum(d["ket_qua"] == "dung" for d in M) / len(M)
    dung = {("test", d["cau"]): d["ket_qua"] == "dung" for d in S}
    dung.update({("mh4", d["cau"]): d["ket_qua"] == "dung" for d in M})
    return o, dung


F = {s: chi_so(f"runs/tu_tin/test_v5_{s}_ck2416.jsonl", f"runs/mo_ho_4/v5_{s}.jsonl") for s in ("s42", "s7")}
L = {s: chi_so(f"runs/tu_tin/test_lorav5_{s}.jsonl", f"runs/mo_ho_4/lorav5_{s}.jsonl") for s in ("s42", "s7")}
print(f"{'chỉ số':24} | {'FULL s42 / s7 (tb)':26} | {'LoRA s42 / s7 (tb)':26}")
for k in F["s42"][0]:
    row = []
    for X in (F, L):
        a, b = X["s42"][0][k], X["s7"][0][k]
        row.append(f"{a:5.1f} / {b:5.1f} ({(a + b) / 2:5.1f})")
    print(f"{k:24} | {row[0]:26} | {row[1]:26}")
print("\nMcNemar FULL vs LoRA (cùng hạt giống):")
for s in ("s42", "s7"):
    for tap, ten in (("test", "test sạch 230"), ("mh4", "mo_ho_4 129")):
        f, l_ = F[s][1], L[s][1]
        ks = [k for k in f if k[0] == tap]
        b = sum(f[k] and not l_[k] for k in ks)
        c = sum(l_[k] and not f[k] for k in ks)
        print(f"  {s} {ten:14} full hơn {b:2d}, LoRA hơn {c:2d}, p = {mcnemar(b, c):.3f}")
