import collections
import csv
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from eval.chuyen_test import doc_nhan, dong_thanh_row

D = "data/mo_ho_3/"
F = sys.argv[1] if len(sys.argv) > 1 else "2_gan_nhan_id_1_130.csv"
ham = {r["id"]: r["ham_dich"] for r in csv.DictReader(open(D + "khoa_id.csv", encoding="utf-8-sig"))}
cau_goc = {r["id"]: r["cau"].strip() for r in csv.DictReader(open(D + "2_cho_gan_nhan.csv", encoding="utf-8-sig"))}
rows = list(csv.DictReader(open(D + F, encoding="utf-8-sig")))
print("file:", F, "| số dòng:", len(rows), "| cột:", list(rows[0].keys()))
print("id thiếu:", sorted(set(cau_goc) - {r["id"].strip() for r in rows}, key=int))
print("câu bị đổi so với file đưa đi:", [r["id"] for r in rows if r["cau"].strip() != cau_goc.get(r["id"].strip())])

loi, khong_ask, ngoai, lech, dem = [], [], [], [], collections.Counter()
for r in rows:
    i = r["id"].strip()
    try:
        dong_thanh_row(r["cau"], r["nhan"])
        calls, ask = doc_nhan(r["nhan"])
    except Exception as e:  # noqa: BLE001
        loi.append((i, r["nhan"], str(e)))
        continue
    ten = [t for t, _ in calls]
    if ten[0] == "irrelevant_function":
        ngoai.append(i)
        dem["ngoai_pham_vi"] += 1
    elif not ask:
        khong_ask.append(i)
        dem["ra_lenh"] += 1
    else:
        dem["ask"] += 1
        if ham[i] not in ten:
            lech.append(i)
        dem["2_cach" if len(ten) > 1 else "1_cach"] += 1

print("\nthống kê:", dict(dem))
print("\n== LỖI CÚ PHÁP / SCHEMA:", len(loi))
for i, n, e in loi:
    print(f"  {i}: {n}  -> {e}")
by = {r["id"].strip(): r for r in rows}
for ten_nhom, ds in (("KHÔNG có ask (ra lệnh)", khong_ask), ("NGOÀI PHẠM VI", ngoai), ("ask nhưng KHÔNG chứa hàm đích", lech)):
    print(f"\n== {ten_nhom}: {len(ds)}")
    for i in ds:
        r = by[i]
        print(f"  {i} [đích {ham[i]}] {r['cau']}\n      nhãn: {r['nhan']} | {r.get('ghi_chu', '')}")
print("\n== nhãn ask theo hàm đầu tiên:")
c = collections.Counter()
for r in rows:
    try:
        calls, ask = doc_nhan(r["nhan"])
        if ask:
            c[calls[0][0]] += 1
    except Exception:  # noqa: BLE001
        pass
print(" ", dict(c.most_common()))
