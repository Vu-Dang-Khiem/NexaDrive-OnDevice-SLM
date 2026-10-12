"""Chọn ngưỡng riêng cho LoRA theo LUẬT ĐÃ KHOÁ (PLAN.md cuối mục V) — chỉ val_vi + dev_vi_2, không test.
Kiểm trước: chạy trên bản đối chứng phải ra lại bảng mục S (không ngưỡng 1.513 / 10; 0,90: 1.500 / 6)."""
import json
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from schema.loader import CATALOG

TU_CHOI = CATALOG.by_name["irrelevant_function"].nexa_token
ASK = CATALOG.nexa_ask
LUOI = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97)


def kq_nguong(d, t):
    (t1, p1), (t2, _) = d["top_buoc_1"][0], d["top_buoc_1"][1]
    if t1 in (TU_CHOI, ASK) or p1 >= t:
        return d["ket_qua"]
    qd = "tu_choi" if t2 == TU_CHOI else "hoi_lai"
    if d["nhom"] == "ngoai_pham_vi":
        return "dung" if qd == "tu_choi" else "hoi_lai_khi_phai_tu_choi"
    if d["nhom"] == "ro_rang":
        return "tu_choi_nham" if qd == "tu_choi" else "hoi_lai_thua"
    return "mo_ho_" + qd


def bang(path):
    R = [json.loads(x) for x in open(path, encoding="utf-8")]
    R = [d for d in R if d["tap"] in ("val_vi", "dev_vi_2")]
    print(f"\n{path}  ({len(R)} câu val+dev)")
    print(f"{'ngưỡng':>8} | {'gọi bừa':>7} | {'rõ ràng đúng':>12} | {'giảm %':>6} | {'từ chối nhầm':>12} | {'hỏi lại thừa':>12} | {'mơ hồ đúng':>10}")
    goc = None
    kq = []
    for t in (0.0,) + LUOI:
        k = [(d["nhom"], kq_nguong(d, t)) for d in R]
        gb = sum(n == "ngoai_pham_vi" and x == "goi_bua" for n, x in k)
        rr = sum(n == "ro_rang" and x == "dung" for n, x in k)
        tcn = sum(x == "tu_choi_nham" for _, x in k)
        hlt = sum(x == "hoi_lai_thua" for _, x in k)
        mh = sum(n == "mo_ho" and x == "dung" for n, x in k)
        goc = goc or (gb, rr)
        giam = 100 * (goc[1] - rr) / goc[1]
        kq.append((t, gb, rr, giam))
        print(f"{t if t else 'không':>8} | {gb:7d} | {rr:12d} | {giam:6.2f} | {tcn:12d} | {hlt:12d} | {mh:10d}")
    hop = [t for t, gb, rr, giam in kq if t and giam <= 1.0 and gb <= goc[0] - 1]
    print("  -> THEO LUẬT KHOÁ:", f"ngưỡng {max(hop)}" if hop else "KHÔNG dùng ngưỡng")


for p in sys.argv[1:]:
    bang(p)
