import csv
import sys

sys.stdout.reconfigure(encoding="utf-8")
R = list(csv.DictReader(open("data/mo_ho_4/2_da_gan_nhan_130_cau.csv", encoding="utf-8-sig")))
print("số dòng:", len(R))
for r in R:
    if r["id"].strip() in ("6", "14", "59", "60", "70", "120", "130"):
        print(f"{r['id']:>3} | {r['cau']}\n      {r['nhan']} | {r['ghi_chu']}")
print("còn high_beam:", [r["id"] for r in R if "high_beam" in r["nhan"]])
print("set_ac off:", [r["id"] for r in R if 'set_ac(state="off")' in r["nhan"]])
