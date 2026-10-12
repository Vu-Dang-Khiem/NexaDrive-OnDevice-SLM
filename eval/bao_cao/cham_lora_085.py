"""Chấm test MỘT lần với ngưỡng LoRA đã khoá 0,85 (PLAN.md cuối mục V). So với LoRA không ngưỡng, LoRA 0,90, full + 0,90."""
import json
import math
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from schema.loader import CATALOG

TU_CHOI = CATALOG.by_name["irrelevant_function"].nexa_token
ASK = CATALOG.nexa_ask


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


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


SAI_LEN_XE = {"goi_bua", "sai_ham", "sai_tham_so", "thieu_hoi_lai"}


def chi_so(path, t):
    S = [json.loads(x) for x in open(path, encoding="utf-8")]
    S = [d for d in S if d["tap"] == "test_vi_sach_v2"]
    k = {d["cau"]: (d["nhom"], kq_nguong(d, t)) for d in S}
    o = {n: 100 * sum(x == "dung" for m, x in k.values() if m == n) / sum(m == n for m, _ in k.values())
         for n in ("ro_rang", "mo_ho", "ngoai_pham_vi")}
    o["goi_bua"] = sum(x == "goi_bua" for _, x in k.values())
    o["tc_nham"] = sum(x == "tu_choi_nham" for _, x in k.values())
    o["hl_thua"] = sum(x == "hoi_lai_thua" for _, x in k.values())
    o["sai_len_xe"] = sum(x in SAI_LEN_XE for _, x in k.values())
    return o, {c: x == "dung" for c, (_, x) in k.items()}


CH = [("LoRA không ngưỡng", "lorav5", 0.0), ("LoRA + 0,90 (ngưỡng của full)", "lorav5", 0.90),
      ("LoRA + 0,85 (KHOÁ)", "lorav5", 0.85), ("Full v5 không ngưỡng", "v5_{s}_ck2416", 0.0),
      ("Full v5 + 0,90", "v5_{s}_ck2416", 0.90)]
KQ = {}
print(f"{'':32} | {'rõ ràng':>14} | {'mơ hồ':>14} | {'ngoài pv':>14} | gọi bừa | TC nhầm | hỏi thừa | sai lên xe")
for ten, mau, t in CH:
    r = {}
    for s in ("s42", "s7"):
        p = f"runs/tu_tin/test_{mau.format(s=s)}_{s}.jsonl" if mau == "lorav5" else f"runs/tu_tin/test_{mau.format(s=s)}.jsonl"
        r[s] = chi_so(p, t)
    KQ[ten] = r
    a, b = r["s42"][0], r["s7"][0]
    f = lambda key: f"{a[key]:4.1f}/{b[key]:4.1f} ({(a[key] + b[key]) / 2:4.1f})"
    g = lambda key: f"{a[key]:3d}·{b[key]:3d}"
    print(f"{ten:32} | {f('ro_rang')} | {f('mo_ho')} | {f('ngoai_pham_vi')} | {g('goi_bua')} | {g('tc_nham')} | {g('hl_thua')} | {g('sai_len_xe')}")

print("\nMcNemar test sạch 230 (từng hạt giống):")
for a_, b_ in (("LoRA không ngưỡng", "LoRA + 0,85 (KHOÁ)"), ("Full v5 + 0,90", "LoRA + 0,85 (KHOÁ)")):
    for s in ("s42", "s7"):
        x, y = KQ[a_][s][1], KQ[b_][s][1]
        b = sum(x[c] and not y[c] for c in x)
        c = sum(y[c] and not x[c] for c in x)
        print(f"  {s}: '{a_}' hơn {b:2d}, '{b_}' hơn {c:2d}, p = {mcnemar(b, c):.3f}")
