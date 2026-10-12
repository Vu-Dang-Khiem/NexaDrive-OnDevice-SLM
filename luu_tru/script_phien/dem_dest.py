import csv
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
RONG = 'destination=""'
for r in csv.DictReader(open("data/processed/v5/kiem_tay_50.csv", encoding="utf-8-sig")):
    if RONG in r["tro_ly_se_hoi_lai_ve"]:
        print(r["id"], "|", r["cau"], "|", r["tro_ly_se_hoi_lai_ve"])
R = [json.loads(x) for x in open("data/processed/v5/v5_hoi_lai_hc_gop.jsonl", encoding="utf-8")]
print("v5 mới:", sum(RONG.replace('"', '\\"') in json.dumps(r) or RONG in r["completion"] for r in R), "/",
      sum("<nexa_21>" in r["completion"] for r in R), "câu có dẫn đường")
T = [json.loads(x) for x in open("data/processed/final_v4_doi_chung/train_vi.jsonl", encoding="utf-8")]
print("train cũ (đối chứng):", sum(RONG in r["completion"] for r in T), "/", sum("<nexa_21>" in r["completion"] for r in T))
for r in R:
    if RONG in r["completion"]:
        print("   ví dụ:", r["prompt"].split("Query: ")[-1][:90] if "Query: " in r["prompt"] else r["prompt"][-120:])
        break
