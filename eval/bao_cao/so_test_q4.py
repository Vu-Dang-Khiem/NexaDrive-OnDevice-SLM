"""Test MỘT lần: q4_K_M (không ngưỡng / 0,80 đã khoá) vs HF bf16 LoRA s7 (không ngưỡng / 0,85)."""
import math
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from eval.so_muc_nen import doc, kq_nguong

SAI_LEN_XE = {"goi_bua", "sai_ham", "sai_tham_so", "thieu_hoi_lai"}


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


HF = [d for d in doc("runs/tu_tin/test_lorav5_s7.jsonl") if d["tap"] != "dev_vi_2"]
for d in doc("runs/mo_ho_4/lorav5_s7.jsonl"):
    d["tap"] = "test_vi_mo_ho_4"
    HF.append(d)
Q4 = doc("runs/gguf/test_q4_k_m.jsonl")
print("tập trong HF:", sorted({d["tap"] for d in HF}), "· q4:", sorted({d["tap"] for d in Q4}))


def chi_so(R, t):
    o = {}
    S = [d for d in R if d["tap"] == "test_vi_sach_v2"]
    for n in ("ro_rang", "mo_ho", "ngoai_pham_vi"):
        N = [d for d in S if d["nhom"] == n]
        o[n] = 100 * sum(kq_nguong(d, t) == "dung" for d in N) / len(N)
    k = [kq_nguong(d, t) for d in S]
    o["goi_bua"] = k.count("goi_bua")
    o["hoi_thua"] = k.count("hoi_lai_thua")
    o["tc_nham"] = k.count("tu_choi_nham")
    o["sai_len_xe"] = sum(x in SAI_LEN_XE for x in k)
    B = [d for d in R if d["tap"] == "test_vi_bo_sung"]
    o["28 câu"] = 100 * sum(kq_nguong(d, t) == "dung" for d in B) / len(B)
    M = [d for d in R if d["tap"] == "test_vi_mo_ho_4"]
    o["mo_ho_4"] = 100 * sum(kq_nguong(d, t) == "dung" for d in M) / len(M)
    o["_n"] = (len(S), len(B), len(M))
    o["_dung"] = {(d["tap"], d["cau"]): kq_nguong(d, t) == "dung" for d in R}
    return o


CH = [("HF bf16 không ngưỡng", HF, 0.0), ("q4_K_M không ngưỡng", Q4, 0.0),
      ("HF bf16 + 0,85", HF, 0.85), ("q4_K_M + 0,80 (KHOÁ)", Q4, 0.80), ("q4_K_M + 0,85 (đối chiếu)", Q4, 0.85)]
KQ = {}
print(f"{'':28} | rõ ràng | mơ hồ | ngoài pv | gọi bừa | hỏi thừa | TC nhầm | sai lên xe | 28 câu | mo_ho_4")
for ten, R, t in CH:
    o = chi_so(R, t)
    KQ[ten] = o
    print(f"{ten:28} | {o['ro_rang']:7.1f} | {o['mo_ho']:5.1f} | {o['ngoai_pham_vi']:8.1f} | {o['goi_bua']:7d} | "
          f"{o['hoi_thua']:8d} | {o['tc_nham']:7d} | {o['sai_len_xe']:10d} | {o['28 câu']:6.1f} | {o['mo_ho_4']:7.1f}")
print("số câu (sạch, bổ sung, mo_ho_4):", KQ["q4_K_M không ngưỡng"]["_n"], KQ["HF bf16 không ngưỡng"]["_n"])
for a, b in (("HF bf16 không ngưỡng", "q4_K_M không ngưỡng"), ("HF bf16 + 0,85", "q4_K_M + 0,80 (KHOÁ)")):
    x, y = KQ[a]["_dung"], KQ[b]["_dung"]
    for tap in ("test_vi_sach_v2", "test_vi_mo_ho_4"):
        ks = [k for k in x if k[0] == tap and k in y]
        bb = sum(x[k] and not y[k] for k in ks)
        cc = sum(y[k] and not x[k] for k in ks)
        print(f"McNemar {tap}: '{a}' hơn {bb}, '{b}' hơn {cc}, p = {mcnemar(bb, cc):.3f} (ghép {len(ks)})")
