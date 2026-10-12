"""Cách 1 (người dùng chọn 2026-10-06): so v5 vs v6.1 KHÔNG cần chọn ngưỡng — quét MỌI ngưỡng (đường rủi ro – độ phủ).

Người dùng phản biện: luật chọn ngưỡng (rõ ràng giảm ≤ 1%, gọi bừa giảm ≥ 1) là tự đặt, không có tiêu chí chứng minh. Cách
chuẩn trong "dự đoán có quyền từ chối" (selective prediction; Geifman & El-Yaniv 2017 — cần kiểm lại trích dẫn) là báo CẢ
đường cong thay vì một điểm:

  (a) Đánh đổi lợi – hại theo từng ngưỡng t (luật áp ngưỡng `kq_nguong` như runtime):
        lợi = % câu RÕ RÀNG làm đúng ngay · hại = % câu NGOÀI PHẠM VI bị làm bừa
      So ở CÙNG mức hại: model nào làm được nhiều lệnh đúng hơn thì tốt hơn, bất kể chọn ngưỡng nào.
  (b) AURC chuẩn: độ tin = xác suất token đầu; ngưỡng càng cao càng nhiều câu "nhường" (hỏi lại); rủi ro = % sai trong các
      câu model tự quyết; AURC = diện tích dưới đường rủi ro theo độ phủ (càng THẤP càng tốt).

Dữ liệu: độ tự tin đã đo trên test (`runs/v6/tu_tin/tu_tin_test_*.jsonl`, eval/nguong_v6.py). Mô tả, KHÔNG chọn ngưỡng trên test.

    python -m eval.rui_ro_do_phu
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from eval.so_muc_nen import ASK, TU_CHOI, kq_nguong

R = Path("runs/v6/tu_tin")
LUOI = [i / 100 for i in range(0, 100)] + [0.995, 0.999, 1.0]
TAP_CU = ("test_vi_sach_v2", "test_vi_bo_sung", "test_vi_mo_ho_4")


def doc(ten: str) -> list[dict[str, Any]]:
    return [json.loads(x) for x in open(R / f"tu_tin_test_{ten}.jsonl", encoding="utf-8")]


def diem(ds: list[dict[str, Any]], t: float) -> tuple[float, float]:
    """(lợi %, hại %) ở ngưỡng t trên tập test cũ: lợi = rõ ràng đúng ngay, hại = ngoài phạm vi bị gọi bừa."""
    rr = [d for d in ds if d["nhom"] == "ro_rang"]
    npv = [d for d in ds if d["nhom"] == "ngoai_pham_vi"]
    loi = 100 * sum(kq_nguong(d, t) == "dung" for d in rr) / len(rr)
    hai = 100 * sum(kq_nguong(d, t) == "goi_bua" for d in npv) / len(npv)
    return loi, hai


def loi_o_muc_hai(ds: list[dict[str, Any]], hai_toi_da: float) -> float:
    """Lợi lớn nhất đạt được mà hại ≤ hai_toi_da (chọn ngưỡng TỐT NHẤT cho từng model — so công bằng)."""
    tot = [loi for loi, hai in (diem(ds, t) for t in LUOI) if hai <= hai_toi_da + 1e-9]
    return max(tot) if tot else float("nan")


def aurc(ds: list[dict[str, Any]]) -> float:
    """AURC: xếp câu theo độ tin giảm dần; ở độ phủ k/n, rủi ro = % sai trong k câu tin nhất. Câu model tự mở đầu
    bằng từ chối / hỏi lại tính độ tin như mọi câu (đó cũng là quyết định của model)."""
    xep = sorted(ds, key=lambda d: -d["p_buoc_1"])
    sai = 0
    tong = 0.0
    for k, d in enumerate(xep, 1):
        sai += d["ket_qua"] != "dung"
        tong += sai / k
    return 100 * tong / len(xep)


def main() -> None:
    du = {f"{m}{s}": doc(f"{m}{s}") for m in ("v5", "v61") for s in ("s42", "s7")}
    cu = {k: [d for d in v if d["tap"] in TAP_CU] for k, v in du.items()}
    print("(a) ĐÁNH ĐỔI LỢI – HẠI trên test cũ (rõ ràng 74 + ngoài pv 117 của test_vi_sach_v2; mo_ho_4 / bổ sung không có nhóm này)")
    print("    lợi tốt nhất của từng model khi hại (gọi bừa) không quá mức X — mỗi model tự chọn ngưỡng tốt nhất cho nó:")
    print("    mức hại X  |  v5 s42  v5 s7  (tb) |  v6.1 s42  v6.1 s7  (tb)")
    for x in (2, 4, 6, 8, 10, 12, 15, 20, 25):
        a = [loi_o_muc_hai(cu[k], x) for k in ("v5s42", "v5s7")]
        b = [loi_o_muc_hai(cu[k], x) for k in ("v61s42", "v61s7")]
        print(f"    ≤ {x:4.1f}%    |  {a[0]:5.1f}  {a[1]:5.1f}  ({sum(a) / 2:5.1f}) |  {b[0]:5.1f}   {b[1]:5.1f}   ({sum(b) / 2:5.1f})")
    print("\n    vài điểm trên đường (ngưỡng: lợi / hại):")
    for t in (0.0, 0.5, 0.8, 0.85, 0.9, 0.95, 0.99):
        dong = "  ".join(f"{k} {diem(cu[k], t)[0]:4.1f}/{diem(cu[k], t)[1]:4.1f}" for k in du)
        print(f"    t={t:4.2f}: {dong}")
    print("\n(b) AURC (%, càng thấp càng tốt):")
    for ten, chon in (("test cũ", lambda d: d["tap"] in TAP_CU), ("toàn bộ test (cũ + v6 + nhiễu)", lambda d: True)):
        g = {k: aurc([d for d in v if chon(d)]) for k, v in du.items()}
        print(f"    {ten:30s} v5 {g['v5s42']:5.2f} / {g['v5s7']:5.2f} (tb {(g['v5s42'] + g['v5s7']) / 2:5.2f})"
              f"  |  v6.1 {g['v61s42']:5.2f} / {g['v61s7']:5.2f} (tb {(g['v61s42'] + g['v61s7']) / 2:5.2f})")
    _ = (ASK, TU_CHOI)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
