import collections
import csv
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
D = "data/mo_ho_3/"
goc = {r["id"]: r["chuc_nang"] for r in csv.DictReader(open(D + "1_viet_cau.csv", encoding="utf-8-sig"))}
ham = {r["id"]: r["ham_dich"] for r in csv.DictReader(open(D + "khoa_id.csv", encoding="utf-8-sig"))}
rows = list(csv.DictReader(open(D + "1_viet_cau_da_viet.csv", encoding="utf-8-sig")))
print("số dòng:", len(rows), "| cột:", list(rows[0].keys()))
ids = [r["id"].strip() for r in rows]
print("id trùng:", [i for i, c in collections.Counter(ids).items() if c > 1])
print("id thiếu:", sorted(set(goc) - set(ids), key=int))
print("chuc_nang bị đổi:", [r["id"] for r in rows if r["chuc_nang"].strip() != goc.get(r["id"].strip(), "").strip()])
trong = [r["id"] for r in rows if not r["cau"].strip()]
print("câu trống:", trong)

CAM = r"\b(bật|mở|đóng|khoá|khóa|hạ|nâng|tìm|chỉnh|tăng|giảm|dẫn đường|tắt)\b"
print("\n-- có động từ sai bảo (cần xem có phải ra lệnh không):")
for r in rows:
    m = re.findall(CAM, r["cau"].lower())
    if m:
        print(f"  {r['id']:>3} [{ham[r['id']]}] {r['cau']}  <- {m}")

print("\n-- có chữ Latin dạng tiếng Anh:")
for r in rows:
    w = [x for x in re.findall(r"[A-Za-z]{3,}", r["cau"]) if not re.search(r"[À-ỹ]", x)]
    if w:
        print(f"  {r['id']:>3} {r['cau']}  <- {w}")

do_dai = [len(r["cau"].split()) for r in rows if r["cau"].strip()]
print("\nđộ dài (chữ):", "≤8:", sum(x <= 8 for x in do_dai), "| 9–11:", sum(9 <= x <= 11 for x in do_dai), "| ≥12:", sum(x >= 12 for x in do_dai))


def tu(s):
    return set(re.findall(r"\w+", s.lower()))


print("\n-- cặp gần trùng (Jaccard ≥ 0,3):")
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        a, b = tu(rows[i]["cau"]), tu(rows[j]["cau"])
        if a and b and len(a & b) / len(a | b) >= 0.3:
            print(f"  {rows[i]['id']} ~ {rows[j]['id']}: {rows[i]['cau']} || {rows[j]['cau']}")

print("\n==== TOÀN BỘ CÂU THEO HÀM ====")
for h in sorted(set(ham.values())):
    print(f"\n## {h}")
    for r in sorted(rows, key=lambda r: int(r["id"])):
        if ham[r["id"]] == h:
            print(f"  {r['id']:>3} ({len(r['cau'].split()):2d}) {r['cau']}")
