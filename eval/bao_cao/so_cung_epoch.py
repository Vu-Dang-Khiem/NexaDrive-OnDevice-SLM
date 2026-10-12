"""Full vs LoRA ở CÙNG epoch (2 và 3), từng hạt giống — loại trừ khác biệt do chọn checkpoint khác epoch."""
import json
import math
import sys

sys.stdout.reconfigure(encoding="utf-8")


def doc(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")]


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


F = {  # (hạt giống, epoch) -> (file test, file mo_ho_4)
    ("full", "s42", 2): ("runs/tu_tin/test_v5_s42_ck2416.jsonl", "runs/mo_ho_4/v5_s42.jsonl"),
    ("full", "s7", 2): ("runs/tu_tin/test_v5_s7_ck2416.jsonl", "runs/mo_ho_4/v5_s7.jsonl"),
    ("full", "s42", 3): ("runs/cung_epoch/test_full_s42_ck3624.jsonl", "runs/cung_epoch/mh4_full_s42_ck3624.jsonl"),
    ("full", "s7", 3): ("runs/cung_epoch/test_full_s7_ck3624.jsonl", "runs/cung_epoch/mh4_full_s7_ck3624.jsonl"),
    ("lora", "s42", 2): ("runs/cung_epoch/test_lora_s42_ck2416.jsonl", "runs/cung_epoch/mh4_lora_s42_ck2416.jsonl"),
    ("lora", "s7", 2): ("runs/tu_tin/test_lorav5_s7.jsonl", "runs/mo_ho_4/lorav5_s7.jsonl"),
    ("lora", "s42", 3): ("runs/tu_tin/test_lorav5_s42.jsonl", "runs/mo_ho_4/lorav5_s42.jsonl"),
    ("lora", "s7", 3): ("runs/cung_epoch/test_lora_s7_ck3624.jsonl", "runs/cung_epoch/mh4_lora_s7_ck3624.jsonl"),
}
KQ = {}
for k, (ft, fm) in F.items():
    S = [d for d in doc(ft) if d["tap"] == "test_vi_sach_v2"]
    M = doc(fm)
    o = {n: 100 * sum(d["ket_qua"] == "dung" for d in S if d["nhom"] == n) / sum(d["nhom"] == n for d in S)
         for n in ("ro_rang", "mo_ho", "ngoai_pham_vi")}
    o["goi_bua"] = sum(d["ket_qua"] == "goi_bua" for d in S)
    o["tu_choi_nham"] = sum(d["ket_qua"] == "tu_choi_nham" for d in S)
    o["mo_ho_4"] = 100 * sum(d["ket_qua"] == "dung" for d in M) / len(M)
    o["_dung"] = {d["cau"]: d["ket_qua"] == "dung" for d in S}
    KQ[k] = o
print(f"{'':16} | {'rõ ràng':>8} {'mơ hồ39':>8} {'ngoài pv':>9} {'gọi bừa':>8} {'TC nhầm':>8} {'mo_ho_4':>8}")
for ep in (2, 3):
    for nh in ("full", "lora"):
        for s in ("s42", "s7"):
            o = KQ[(nh, s, ep)]
            print(f"{nh} {s:3} epoch {ep}  | {o['ro_rang']:8.1f} {o['mo_ho']:8.1f} {o['ngoai_pham_vi']:9.1f} "
                  f"{o['goi_bua']:8d} {o['tu_choi_nham']:8d} {o['mo_ho_4']:8.1f}")
print("\nMcNemar test sạch 230, full vs LoRA CÙNG epoch, cùng hạt giống:")
for ep in (2, 3):
    for s in ("s42", "s7"):
        f, l_ = KQ[("full", s, ep)]["_dung"], KQ[("lora", s, ep)]["_dung"]
        b = sum(f[c] and not l_[c] for c in f)
        c = sum(l_[c] and not f[c] for c in f)
        print(f"  epoch {ep} {s:3}: full hơn {b:2d}, LoRA hơn {c:2d}, p = {mcnemar(b, c):.3f}")
