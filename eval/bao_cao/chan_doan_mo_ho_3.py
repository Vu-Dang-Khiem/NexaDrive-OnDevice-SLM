import collections
import json
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from eval.metrics import cham_mot_cau
from gen.records import query_from_call_prompt as q


def du(p):
    return {json.loads(x)["prompt"]: json.loads(x)["output"] for x in open(p, encoding="utf-8")}


R = [json.loads(x) for x in open("data/processed/final/test_vi_mo_ho_3.jsonl", encoding="utf-8")]
M = {"s42": du("runs/mo_ho_3/du_doan_v4dcs42_test_vi_mo_ho_3.jsonl"),
     "s7": du("runs/mo_ho_3/du_doan_v4dcs7_test_vi_mo_ho_3.jsonl"),
     "72B": du("runs/mo_ho_3/du_doan_bl_sip_72b_test_vi_mo_ho_3.jsonl")}
for ten, D in M.items():
    c = collections.defaultdict(collections.Counter)
    for r in R:
        nhom = "ngắn (≤10 chữ)" if len(q(r["prompt"]).split()) <= 10 else "dài (>10 chữ)"
        c[nhom][cham_mot_cau(D[r["prompt"]], r["completion"])[0]] += 1
    for nhom in ("ngắn (≤10 chữ)", "dài (>10 chữ)"):
        n = sum(c[nhom].values())
        print(f"{ten:4} {nhom:15} n={n:3d} đúng {100*c[nhom]['dung']/n:5.1f}% | từ chối nhầm {c[nhom]['tu_choi_nham']:2d}"
              f" | sai hàm {c[nhom]['sai_ham']:2d} | thiếu hỏi lại {c[nhom]['thieu_hoi_lai']:2d}")
# quạt gió: model chọn gì
print("\nquạt gió — hàm model chọn (s42 / s7):")
for r in R:
    if r["meta"]["nhom_phu"] == "mh3_set_fan_speed":
        print("  ", M["s42"][r["prompt"]][:45], " | ", M["s7"][r["prompt"]][:45])
