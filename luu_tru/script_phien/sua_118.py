import csv

D = "data/mo_ho_3/"
moi = {r["id"]: r for r in csv.DictReader(open(D + "1b_viet_lai_da_viet.csv", encoding="utf-8-sig"))}
cau = moi["118"]["cau"].strip()
rows = list(csv.DictReader(open(D + "2_gan_nhan_ban_sua.csv", encoding="utf-8-sig")))
for r in rows:
    if r["id"] == "118":
        r["cau"] = cau
        r["nhan"] = "ask: lock_doors()"
        r["ghi_chu"] = ("Sửa câu sau khi gán nhãn (2026-10-03): 'quên mất đóng cửa xe' dễ hiểu thành đóng cánh cửa "
                        "-> 'không nhớ đã khóa xe chưa'; nhãn theo đề xuất, người dùng duyệt.")
with open(D + "2_gan_nhan_ban_sua.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "cau", "nhan", "ghi_chu"])
    w.writeheader()
    w.writerows(rows)
print("đã sửa 118:", cau)
