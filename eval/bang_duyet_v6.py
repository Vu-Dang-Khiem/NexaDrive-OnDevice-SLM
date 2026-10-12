"""Tập test v6 — tạo BẢNG DUYỆT dễ đọc cho người dùng: mỗi nhãn dịch sang tiếng Việt ("hạ kính ghế phụ") để
người duyệt chỉ cần đọc câu + việc trợ lý sẽ làm, rồi đánh dấu dòng SAI. Không sửa nhãn nào.

    python -m eval.bang_duyet_v6 --vao data/test_v6/2_da_gan_nhan_210.csv   # -> data/test_v6/3_bang_duyet.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from eval.chuyen_test import doc_nhan
from eval.kiem_test_v6 import doc_csv
from serve.tra_loi import hanh_dong

# hàm mà câu tiếng Việt của `hanh_dong` không nói hết tham số -> ghi thêm các tham số này cho người duyệt
_THEM_CHI_TIET = {
    "set_volume": ("set_value",), "play_media": ("source", "query"), "operate_sunroof": ("intensity",),
    "get_tire_pressure": ("wheel",), "get_odometer": ("unit",), "get_range_estimate": ("trip_km",),
    "get_weather": ("location", "day"), "find_nearby": ("radius_km",), "navigate_to": ("avoid",),
    "unlock_doors": ("door",), "set_defrost": ("zone",),
}


def _mot_viec(t: str, a: dict, cau: str) -> str:
    if t == "irrelevant_function":
        return "TỪ CHỐI (xe không làm được / không liên quan)"
    them = [f"{k}={a[k]}" for k in _THEM_CHI_TIET.get(t, ()) if k in a]
    return hanh_dong(t, a, cau) + (f" [{', '.join(them)}]" if them else "")


def doc_viec(nhan: str, cau: str) -> str:
    """Nhãn -> cụm việc tiếng Việt; nhiều lời gọi nối bằng " + "."""
    if not nhan.strip():
        return ""
    calls, hoi = doc_nhan(nhan)
    viec = " + ".join(_mot_viec(t, a, cau) for t, a in calls)
    return f"HỎI LẠI: {viec}" if hoi else viec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vao", type=Path, required=True)
    ap.add_argument("--ra", type=Path, default=Path("data/test_v6/3_bang_duyet.csv"))
    a = ap.parse_args()
    khoa = {r["id"]: r for r in doc_csv(Path("data/test_v6/khoa_id.csv"))}
    with open(a.ra, "w", encoding="utf-8-sig", newline="") as f:
        cot = ["id", "phan", "cau_luot_1", "tro_ly_lam_luot_1", "cau", "tro_ly_se_lam", "SAI", "sua_thanh",
               "nhan_goc"]
        w = csv.DictWriter(f, fieldnames=cot)
        w.writeheader()
        for r in doc_csv(a.vao):
            w.writerow({
                "id": r["id"], "phan": khoa[r["id"]]["phan"], "cau_luot_1": r.get("luot_1", ""),
                "tro_ly_lam_luot_1": doc_viec(r.get("nhan_luot_1", ""), r.get("luot_1", "")),
                "cau": r["cau"], "tro_ly_se_lam": doc_viec(r["nhan"], r["cau"]),
                "SAI": "", "sua_thanh": "", "nhan_goc": r["nhan"],
            })
    print(f"đã tạo {a.ra}")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
