"""Cách 2 (người dùng chọn 2026-10-06): NGƯỠNG CÓ CAM KẾT THỐNG KÊ thay cho luật ±1% tự đặt.

Luật KHOÁ TRƯỚC (PLAN.md): tập hiệu chỉnh = 194 câu NGOÀI PHẠM VI do ChatGPT viết + người dùng duyệt (khác tập test, đã
bỏ câu trùng test / train). Với từng model, lưới ngưỡng 0,00–0,99; tại mỗi ngưỡng đếm k / n câu bị làm bừa (luật áp ngưỡng
`kq_nguong` như runtime), cận trên Clopper–Pearson MỘT PHÍA 90%; chọn ngưỡng THẤP NHẤT có cận trên ≤ 10% (thấp nhất =
model tự làm nhiều lệnh nhất). Rồi áp lên test đã đóng băng để báo: làm bừa thật + lệnh đúng ngay / sau ≤ 1 xác nhận.

    python -m eval.nguong_cam_ket
"""

from __future__ import annotations

import math
import sys
from typing import Any

from eval.nguong_v6 import tai
from eval.so_muc_nen import kq_nguong

MUC_TIEU = 0.10  # làm bừa ≤ 10% (người dùng chọn)
DO_TIN = 0.90  # cận trên một phía 90%
LUOI = [i / 100 for i in range(100)]
MODEL = ("v5s42", "v5s7", "v61s42", "v61s7")


def can_tren(k: int, n: int, do_tin: float = DO_TIN) -> float:
    """Cận trên Clopper–Pearson một phía: p nhỏ nhất mà P(X ≤ k | n, p) ≤ 1 − do_tin (chia đôi)."""
    if k >= n:
        return 1.0

    def cdf(p: float) -> float:
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))

    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if cdf(mid) > 1 - do_tin:
            lo = mid
        else:
            hi = mid
    return hi


def lam_bua(R: list[dict[str, Any]], t: float) -> int:
    return sum(kq_nguong(d, t) == "goi_bua" for d in R)


def chon(R: list[dict[str, Any]]) -> tuple[float | None, int, float]:
    for t in LUOI:  # tăng dần -> ngưỡng đầu tiên đạt là ngưỡng THẤP NHẤT
        k = lam_bua(R, t)
        if can_tren(k, len(R)) <= MUC_TIEU:
            return t, k, can_tren(k, len(R))
    return None, lam_bua(R, LUOI[-1]), can_tren(lam_bua(R, LUOI[-1]), len(R))


def ty(ds: list[bool]) -> float:
    return 100 * sum(ds) / len(ds) if ds else float("nan")


def main() -> None:
    nguong: dict[str, float | None] = {}
    print(f"== HIỆU CHỈNH (mục tiêu làm bừa ≤ {MUC_TIEU:.0%}, cận trên {DO_TIN:.0%}) ==")
    for m in MODEL:
        R = tai(f"tu_tin_hc_{m}.jsonl", "final_v6")
        k0 = lam_bua(R, 0.0)
        t, k, ct = chon(R)
        nguong[m] = t
        print(f"  {m:7s} n={len(R)} | không ngưỡng: làm bừa {k0} ({100 * k0 / len(R):.1f}%, cận trên "
              f"{100 * can_tren(k0, len(R)):.1f}%) | NGƯỠNG CAM KẾT = {t} (làm bừa {k}, cận trên {100 * ct:.1f}%)")
    print("\n== ÁP LÊN TEST ĐÃ ĐÓNG BĂNG (mỗi model ngưỡng cam kết của nó) ==")
    for m in MODEL:
        t = nguong[m]
        if t is None:
            print(f"  {m}: KHÔNG cam kết được ở lưới ngưỡng")
            continue
        R = tai(f"tu_tin_test_{m}.jsonl", "final_v6")
        npv = [d for d in R if d["tap"] == "test_vi_sach_v2" and d["nhom"] == "ngoai_pham_vi"]
        rr = [d for d in R if d["tap"] == "test_vi_sach_v2" and d["nhom"] == "ro_rang"]
        ab = [d for d in R if d["tap"] == "test_v6_mot_luot" and d["nhom_phu"] in ("v6_A", "v6_B")]
        hl = [d for d in R if d["tap"] in ("test_v6_hai_luot", "test_v6_hai_luot_khong_nho")]
        ngay = lambda ds: ty([kq_nguong(d, t) == "dung" for d in ds])  # noqa: E731
        xn = lambda ds: ty([kq_nguong(d, t) == "dung" or (kq_nguong(d, t) == "hoi_lai_thua"  # noqa: E731
                                                          and d["ket_qua"] == "dung") for d in ds])
        print(f"  {m:7s} t={t:.2f} | test ngoài pv làm bừa {100 - ty([kq_nguong(d, t) != 'goi_bua' for d in npv]):4.1f}%"
              f" | rõ ràng đúng ngay {ngay(rr):4.1f} (≤1 xác nhận {xn(rr):4.1f}) | A+B {ngay(ab):4.1f} ({xn(ab):4.1f})"
              f" | hai lượt (không luật) {ngay(hl):4.1f}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
