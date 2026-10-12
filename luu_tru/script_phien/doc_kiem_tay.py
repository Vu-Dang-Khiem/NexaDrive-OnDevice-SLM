import collections
import csv
import math
import sys

sys.stdout.reconfigure(encoding="utf-8")
R = list(csv.DictReader(open("data/processed/v5/kiem_tay_50.csv", encoding="utf-8-sig")))
print("cột:", list(R[0].keys()), "| số dòng:", len(R))


def chuan(s):
    s = s.strip().lower()
    if s.startswith("đ") or s in ("dung", "d", "ok", "true", "1"):
        return "dung"
    if s.startswith("s") or s in ("0", "false"):
        return "sai"
    if s.startswith("p"):
        return "phan_van"
    return "trong" if not s else "khac:" + s


c = collections.Counter(chuan(r["dung_hay_sai"]) for r in R)
print("kết quả:", dict(c))
n = sum(c.values()) - c.get("trong", 0)
k = c.get("dung", 0)
z = 1.96
p = k / n
d = 1 + z * z / n
t = (p + z * z / (2 * n)) / d
h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
print(f"đúng {k}/{n} = {100*p:.1f}% (KTC95 {100*(t-h):.1f}–{100*(t+h):.1f}), phân vân tính là KHÔNG đúng")
print("\n=== sai / phân vân ===")
for r in R:
    if chuan(r["dung_hay_sai"]) != "dung":
        print(f"  {r['id']:>2} [{chuan(r['dung_hay_sai'])}] {r['cau']}\n      -> {r['tro_ly_se_hoi_lai_ve']} | {r['ghi_chu']}")
