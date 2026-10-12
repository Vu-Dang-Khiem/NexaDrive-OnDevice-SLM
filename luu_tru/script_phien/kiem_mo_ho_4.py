import collections
import csv
import json
import re
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from gen.loc import tu_tieng_anh
from gen.records import query_from_call_prompt as q

D = "data/mo_ho_4/"
goc = {r["id"]: r["chuc_nang"] for r in csv.DictReader(open(D + "1_viet_cau.csv", encoding="utf-8-sig"))}
ham = {r["id"]: r["ham_dich"] for r in csv.DictReader(open(D + "khoa_id.csv", encoding="utf-8-sig"))}
rows = list(csv.DictReader(open(D + "1_viet_cau_130_cau.csv", encoding="utf-8-sig")))
ids = [r["id"].strip() for r in rows]
print("số dòng:", len(rows), "| cột:", list(rows[0].keys()))
print("id trùng:", [i for i, c in collections.Counter(ids).items() if c > 1], "| id thiếu:", sorted(set(goc) - set(ids), key=int))
print("chuc_nang bị đổi:", [r["id"] for r in rows if r["chuc_nang"].strip() != goc.get(r["id"].strip(), "").strip()])
print("câu trống:", [r["id"] for r in rows if not r["cau"].strip()])
CAM = r"\b(bật|mở|đóng|khoá|khóa|hạ|nâng|tìm|chỉnh|tăng|giảm|dẫn đường|tắt)\b"
print("có động từ sai bảo:", [(r["id"], re.findall(CAM, r["cau"].lower())) for r in rows if re.findall(CAM, r["cau"].lower())])
print("có tiếng Anh:", [(r["id"], tu_tieng_anh(r["cau"])) for r in rows if tu_tieng_anh(r["cau"])])
dd = [len(r["cau"].split()) for r in rows if r["cau"].strip()]
print("độ dài: ≤8:", sum(x <= 8 for x in dd), "| 9–14:", sum(9 <= x <= 14 for x in dd), "| ≥15:", sum(x >= 15 for x in dd))


def tu(s):
    return set(re.findall(r"\w+", s.lower()))


print("\ncặp gần trùng trong bộ (≥0,3):")
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        a, b = tu(rows[i]["cau"]), tu(rows[j]["cau"])
        if a and b and len(a & b) / len(a | b) >= 0.3:
            print(f"  {len(a & b) / len(a | b):.2f} {rows[i]['id']}: {rows[i]['cau']}\n       {rows[j]['id']}: {rows[j]['cau']}")
m3 = [q(json.loads(x)["prompt"]) for x in open("data/processed/final/test_vi_mo_ho_3.jsonl", encoding="utf-8")]
print("\ngần trùng với mo_ho_3 (≥0,4):")
for r in rows:
    a = tu(r["cau"])
    for c in m3:
        b = tu(c)
        if a and len(a & b) / len(a | b) >= 0.4:
            print(f"  {r['id']}: {r['cau']}\n     ~ {c}")
print("\n==== TOÀN BỘ THEO HÀM ====")
for h in sorted(set(ham.values())):
    print(f"\n## {h}")
    for r in sorted(rows, key=lambda r: int(r["id"])):
        if ham[r["id"].strip()] == h:
            print(f"  {r['id']:>3} ({len(r['cau'].split()):2d}) {r['cau']}")
