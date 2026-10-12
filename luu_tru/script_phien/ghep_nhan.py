import csv

D = "data/mo_ho_3/"
goc = list(csv.DictReader(open(D + "2_gan_nhan_id_1_130.csv", encoding="utf-8-sig")))
sua = {r["id"]: r for r in csv.DictReader(open(D + "2b_sua_25_nhan.csv", encoding="utf-8-sig"))}
assert len(sua) == 25
ra = []
for r in goc:
    s = sua.get(r["id"])
    if s:
        assert s["cau"].strip() == r["cau"].strip(), r["id"]
        r = s
    ra.append(r)
with open(D + "2_gan_nhan_ban_sua.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "cau", "nhan", "ghi_chu"])
    w.writeheader()
    for r in ra:
        w.writerow({k: r.get(k, "") for k in ["id", "cau", "nhan", "ghi_chu"]})
print("ghép xong", len(ra))
