import csv
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
D = "data/mo_ho_3/"
ra = list(csv.DictReader(open(D + "1_viet_cau_hop_nhat.csv", encoding="utf-8-sig")))
ham = {r["id"]: r["ham_dich"] for r in csv.DictReader(open(D + "khoa_id.csv", encoding="utf-8-sig"))}
print("số câu:", len(ra), "| trống:", [r["id"] for r in ra if not r["cau"].strip()])
for r in ra:
    if r["id"] in ("95", "97", "115", "118"):
        print(f"  {r['id']} [{ham[r['id']]}] {r['cau']}")
CAM = r"\b(bật|mở|đóng|khoá|khóa|hạ|nâng|tìm|chỉnh|tăng|giảm|dẫn đường|tắt)\b"
print("động từ sai bảo:", [(r["id"], re.findall(CAM, r["cau"].lower())) for r in ra if re.findall(CAM, r["cau"].lower())])


def tu(s):
    return set(re.findall(r"\w+", s.lower()))


print("cặp gần trùng (Jaccard ≥ 0,25), toàn bộ:")
for i in range(len(ra)):
    for j in range(i + 1, len(ra)):
        a, b = tu(ra[i]["cau"]), tu(ra[j]["cau"])
        jac = len(a & b) / len(a | b)
        if jac >= 0.25:
            print(f"  {jac:.2f} {ra[i]['id']}: {ra[i]['cau']}\n       {ra[j]['id']}: {ra[j]['cau']}")
with open(D + "2_cho_gan_nhan.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "cau", "ghi_chu"])
    for r in ra:
        w.writerow([r["id"], r["cau"].strip(), ""])
print("đã tạo lại 2_cho_gan_nhan.csv")
