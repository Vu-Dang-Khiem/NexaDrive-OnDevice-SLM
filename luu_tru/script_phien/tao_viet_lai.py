import csv

IDS = {"5", "36", "55", "70", "77", "81", "83", "95", "97", "107", "111", "113", "115", "118", "122", "129"}
D = "data/mo_ho_3/"
goc = list(csv.DictReader(open(D + "1_viet_cau.csv", encoding="utf-8-sig")))
with open(D + "1b_viet_lai.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "chuc_nang", "cau", "ghi_chu"])
    for r in goc:
        if r["id"] in IDS:
            w.writerow([r["id"], r["chuc_nang"], "", ""])
print("xong")
