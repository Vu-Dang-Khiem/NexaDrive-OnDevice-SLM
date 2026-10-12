import csv

D = "data/mo_ho_4/"
rows = list(csv.DictReader(open(D + "2_da_gan_nhan_130_cau.csv", encoding="utf-8-sig")))
ham = {r["id"]: r["ham_dich"] for r in csv.DictReader(open(D + "khoa_id.csv", encoding="utf-8-sig"))}
LOAI = {"120"}  # đèn đã bật, muốn nhìn xa — chỉ đèn xa mới hợp, mà hệ thống không gợi ý đèn xa (PLAN.md mục U)
with open(D + "2_gan_nhan_da_duyet.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "cau", "nhan", "ghi_chu", "nhom_phu"])
    w.writeheader()
    n = 0
    for r in rows:
        i = r["id"].strip()
        if i in LOAI:
            continue
        w.writerow({"id": i, "cau": r["cau"].strip(), "nhan": r["nhan"].strip(), "ghi_chu": r["ghi_chu"],
                    "nhom_phu": "mh4_" + ham[i]})
        n += 1
print("đã ghi", n, "câu")
