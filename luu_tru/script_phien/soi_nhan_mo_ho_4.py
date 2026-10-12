import csv
import sys

sys.stdout.reconfigure(encoding="utf-8")
D = "data/mo_ho_4/"
R = list(csv.DictReader(open(D + "2_da_gan_nhan_130_cau.csv", encoding="utf-8-sig")))
ham = {r["id"]: r["ham_dich"] for r in csv.DictReader(open(D + "khoa_id.csv", encoding="utf-8-sig"))}
for r in sorted(R, key=lambda r: (ham[r["id"].strip()], int(r["id"]))):
    n = r["nhan"]
    can_xem = (
        "set_fan_speed" in n or "set_headlights" in n or ";" in n or "get_" in n
        or ("set_volume" in n) or ("set_ac" in n and any(k in r["cau"].lower() for k in ("lạnh", "rét", "buốt")))
    )
    if can_xem:
        print(f"{r['id']:>3} [{ham[r['id'].strip()]}] {r['cau']}\n      {n}  | {r['ghi_chu']}")
