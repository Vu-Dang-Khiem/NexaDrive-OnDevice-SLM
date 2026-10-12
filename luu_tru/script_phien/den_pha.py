import json
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
from gen.records import query_from_call_prompt as q

r = [json.loads(x) for x in open("data/processed/final_v3/train_vi.jsonl", encoding="utf-8")]
c = Counter()
vd = {}
for x in r:
    if x["meta"]["fn"] != "set_headlights" or x["meta"]["type"] != "call":
        continue
    cau = q(x["prompt"]).lower()
    if re.search(r"bật đèn pha", cau) and not re.search(r"xa|gần|cốt|tự động|auto|tắt|đỗ", cau):
        m = re.search(r'mode="(\w+)"', x["completion"]).group(1)
        c[m] += 1
        vd.setdefault(m, cau)
print("train v3 — 'bật đèn pha' không nói chế độ:", dict(c))
print(vd)
