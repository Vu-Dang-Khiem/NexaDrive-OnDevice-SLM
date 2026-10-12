import csv

D = "data/mo_ho_3/"
rows = list(csv.DictReader(open(D + "2_gan_nhan_ban_sua.csv", encoding="utf-8-sig")))
ham = {r["id"]: r["ham_dich"] for r in csv.DictReader(open(D + "khoa_id.csv", encoding="utf-8-sig"))}
assert len(rows) == 130
with open(D + "2_gan_nhan_da_duyet.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "cau", "nhan", "ghi_chu", "nhom_phu"])
    w.writeheader()
    for r in rows:
        w.writerow({"id": r["id"], "cau": r["cau"], "nhan": r["nhan"], "ghi_chu": r["ghi_chu"],
                    "nhom_phu": "mh3_" + ham[r["id"]]})
print("đã ghi 2_gan_nhan_da_duyet.csv")
