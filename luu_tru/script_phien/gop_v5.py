import collections
import csv
import json
import random
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from gen.records import query_from_call_prompt as q

D = "data/processed/v5/"
rows = []
for f in ("cham/v5_hoi_lai_hc_giu.jsonl", "cham/v5cuu_hoi_lai_hc_giu.jsonl"):
    rows += [json.loads(x) for x in open(D + f, encoding="utf-8")]
print("tổng giữ:", len(rows))
c = collections.Counter(r["fn_dich"] for r in rows)
print("theo hàm đích:", dict(sorted(c.items())))
dai = [len(q(r["prompt"]).split()) for r in rows]
print("độ dài: >10 chữ:", sum(x > 10 for x in dai), "| trung vị", sorted(dai)[len(dai) // 2])
print("2 cách xử lý:", sum(r["completion"].count("<nexa_") >= 4 for r in rows))
with open(D + "v5_hoi_lai_hc_gop.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        r = {k: v for k, v in r.items() if k != "fn_dich"}
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# 50 câu kiểm tay: bốc phân tầng theo hàm (~4 câu/hàm), seed 2070
from schema.call_format import parse_calls, render_readable

rng = random.Random(2070)
theo = collections.defaultdict(list)
for r in rows:
    theo[r["fn_dich"]].append(r)
mau = []
for h in sorted(theo):
    mau += rng.sample(theo[h], 4)
mau = mau[:52]
rng.shuffle(mau)
with open(D + "kiem_tay_50.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "cau", "tro_ly_se_hoi_lai_ve", "dung_hay_sai", "ghi_chu"])
    for i, r in enumerate(mau, 1):
        calls = parse_calls(r["completion"].replace("<|im_end|>", ""))
        w.writerow([i, q(r["prompt"]), " ; ".join(render_readable(c.name, dict(c.args)) for c in calls), "", ""])
print("đã ghi kiem_tay_50.csv:", len(mau), "câu")
