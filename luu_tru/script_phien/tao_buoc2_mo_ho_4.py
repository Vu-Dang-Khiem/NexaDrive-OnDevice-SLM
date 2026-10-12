import csv
import sys

sys.stdout.reconfigure(encoding="utf-8")
D = "data/mo_ho_4/"
rows = list(csv.DictReader(open(D + "1_viet_cau_130_cau.csv", encoding="utf-8-sig")))
r55 = next(r for r in rows if r["id"].strip() == "55")
print("câu 55:", r55["cau"])
assert len(rows) == 130 and all(r["cau"].strip() for r in rows)
with open(D + "2_cho_gan_nhan.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "cau", "ghi_chu"])
    for r in sorted(rows, key=lambda r: int(r["id"])):
        w.writerow([r["id"].strip(), r["cau"].strip(), ""])
print("đã tạo 2_cho_gan_nhan.csv (130 câu, không có cột chức năng)")
