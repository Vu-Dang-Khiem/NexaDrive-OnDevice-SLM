import json
import sys
from collections import Counter

sys.stdout = open(sys.argv[1], "w", encoding="utf-8")
sys.path.insert(0, ".")
from eval.metrics import cham_mot_cau
from gen.records import query_from_call_prompt as q


def doc(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")]


def du(p):
    return {r["prompt"]: r["output"] for r in doc(p)}


T = doc("data/processed/final/test_vi_bo_sung.jsonl")
M = {"v3_42": "runs/student-full-v3", "v3_7": "runs/student-full-v3-s7",
     "dc_42": "runs/student-v4dc-s42", "dc_7": "runs/student-v4dc-s7"}
P = {k: du(f"{v}/du_doan_test_vi_bo_sung.jsonl") for k, v in M.items()}
for k in M:
    c = Counter(cham_mot_cau(P[k][r["prompt"]], r["completion"])[0] for r in T)
    print(k, dict(c))
print()
for r in T:
    kq = {k: cham_mot_cau(P[k][r["prompt"]], r["completion"])[0] for k in M}
    if kq["dc_7"] != "dung" or kq["dc_42"] != "dung":
        print(q(r["prompt"])[:70], "| cần", r["completion"][:45])
        for k in M:
            print(f"    {k:6} {kq[k]:13} {P[k][r['prompt']][:55]}")
