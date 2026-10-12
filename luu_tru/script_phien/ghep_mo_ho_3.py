import csv
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
D = "data/mo_ho_3/"
chinh = list(csv.DictReader(open(D + "1_viet_cau_da_viet.csv", encoding="utf-8-sig")))
moi = {r["id"]: r for r in csv.DictReader(open(D + "1b_viet_lai_da_viet.csv", encoding="utf-8-sig"))}
ham = {r["id"]: r["ham_dich"] for r in csv.DictReader(open(D + "khoa_id.csv", encoding="utf-8-sig"))}
ra = [moi.get(r["id"], r) for r in chinh]
assert len(ra) == 130 and sum(r["id"] in moi for r in ra) == 16
with open(D + "1_viet_cau_hop_nhat.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "chuc_nang", "cau", "ghi_chu"])
    w.writeheader()
    for r in ra:
        w.writerow({k: r[k] for k in ["id", "chuc_nang", "cau", "ghi_chu"]})
# file cho bước 2: bỏ cột chuc_nang
with open(D + "2_cho_gan_nhan.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "cau", "ghi_chu"])
    for r in ra:
        w.writerow([r["id"], r["cau"], ""])

CAM = r"\b(bật|mở|đóng|khoá|khóa|hạ|nâng|tìm|chỉnh|tăng|giảm|dẫn đường|tắt)\b"
print("động từ sai bảo:", [(r["id"], re.findall(CAM, r["cau"].lower())) for r in ra if re.findall(CAM, r["cau"].lower())])


def tu(s):
    return set(re.findall(r"\w+", s.lower()))


print("cặp gần trùng (Jaccard ≥ 0,25) có dính câu mới:")
for i in range(len(ra)):
    for j in range(i + 1, len(ra)):
        if ra[i]["id"] not in moi and ra[j]["id"] not in moi:
            continue
        a, b = tu(ra[i]["cau"]), tu(ra[j]["cau"])
        jac = len(a & b) / len(a | b)
        if jac >= 0.25:
            print(f"  {jac:.2f} [{ham[ra[i]['id']]}] {ra[i]['id']}: {ra[i]['cau']}\n       [{ham[ra[j]['id']]}] {ra[j]['id']}: {ra[j]['cau']}")
do_dai = [len(r["cau"].split()) for r in ra]
print("độ dài: ≤8:", sum(x <= 8 for x in do_dai), "| 9–14:", sum(9 <= x <= 14 for x in do_dai), "| ≥15:", sum(x >= 15 for x in do_dai))
