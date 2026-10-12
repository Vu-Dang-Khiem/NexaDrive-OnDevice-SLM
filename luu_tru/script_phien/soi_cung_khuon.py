import json
import random
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
R = [json.loads(x) for x in open("data/processed/v5/v5_hoi_lai_hc_chi_tiet.jsonl", encoding="utf-8")]
ck = [r for r in R if r["loai_vi"] == "cung_khuon"]
khac = [r for r in R if r["loai_vi"] != "cung_khuon" and r["loai_vi"] not in ("loc_code", "tieng_anh", "qua_ngan", "ra_lenh", "trung")]


def tu(s):
    return set(re.findall(r"\w+", s.lower()))


T = [(r, tu(r["query"])) for r in khac]
random.Random(3).shuffle(ck)
bins = {"<0.35": 0, "0.35-0.5": 0, "0.5-0.7": 0, ">=0.7": 0}
mau = []
for r in ck[:400]:
    a = tu(r["query"])
    best, bq = 0.0, ""
    for x, b in T:
        if x["fn_dich"] != r["fn_dich"]:
            continue
        j = len(a & b) / len(a | b)
        if j > best:
            best, bq = j, x["query"]
    k = "<0.35" if best < 0.35 else "0.35-0.5" if best < 0.5 else "0.5-0.7" if best < 0.7 else ">=0.7"
    bins[k] += 1
    if len(mau) < 12:
        mau.append((best, r["query"], bq))
print("400 câu 'cùng khuôn' ngẫu nhiên — độ giống CAO NHẤT với câu đã giữ (cùng hàm):", bins)
for j, a, b in mau:
    print(f"\n {j:.2f}\n   bị loại: {a}\n   giống  : {b}")
