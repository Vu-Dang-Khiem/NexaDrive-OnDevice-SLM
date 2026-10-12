"""Tập đánh giá GRPO — bước 1 -> bước 2: kiểm file câu ChatGPT viết (người dùng đã sửa tay) rồi tạo file cho
bước gán nhãn (bỏ cột chức năng / yêu cầu / phần — người gán nhãn không được thấy "đáp án định sẵn").
Cảnh báo (không chặn): câu phần M có động từ ra lệnh — phần M phải là than phiền / bóng gió, không ra lệnh.

    python -m eval.kiem_test_grpo --vao data/test_grpo/1_viet_cau_200_cau.csv
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from eval.kiem_test_v6 import doc_csv, tao_file_gan_nhan

_RA_LENH = re.compile(r"^(bật|tắt|mở|đóng|hạ|nâng|kéo|khoá|khóa|tăng|giảm|vặn|chỉnh|cho|đặt|tìm|dẫn|chỉ|"
                      r"phát)\b|\b(giúp|hộ|giùm|dùm)\s+(tôi|em|anh|chị|mình|bác|cô)\b|\bđi\s*[.!]*$")


def kiem(goc: list[dict[str, str]], da_viet: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    """(lỗi phải sửa, cảnh báo nên xem)."""
    loi: list[str] = []
    canh_bao: list[str] = []
    theo_id = {r["id"]: r for r in da_viet}
    if len(da_viet) != len(goc) or set(theo_id) != {r["id"] for r in goc}:
        thieu = sorted({r["id"] for r in goc} - set(theo_id))[:10]
        loi.append(f"số dòng / id lệch: {len(da_viet)} dòng, thiếu {thieu}")
    for g in goc:
        r = theo_id.get(g["id"])
        if r is None:
            continue
        for cot in ("phan", "chuc_nang", "yeu_cau"):
            if r.get(cot, "") != g[cot]:
                loi.append(f"id {g['id']}: cột {cot} bị đổi")
        if not r.get("cau"):
            loi.append(f"id {g['id']}: thiếu câu")
        if g["phan"] == "H" and not r.get("luot_1"):
            loi.append(f"id {g['id']} (H): thiếu câu lượt 1")
        if g["phan"] != "H" and r.get("luot_1"):
            loi.append(f"id {g['id']}: phần {g['phan']} mà có câu lượt 1")
        if g["phan"] == "M" and _RA_LENH.search(r.get("cau", "").lower()):
            canh_bao.append(f"id {g['id']} (M) nghe như RA LỆNH: \"{r['cau']}\"")
    dem = Counter(" ".join(r.get("cau", "").lower().split()) for r in da_viet if r.get("cau"))
    for c, n in dem.items():
        if n > 1:
            loi.append(f"câu trùng {n} lần: \"{c}\"")
    return loi, canh_bao


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vao", type=Path, required=True)
    ap.add_argument("--goc", type=Path, default=Path("data/test_grpo/1_viet_cau.csv"))
    ap.add_argument("--ra", type=Path, default=Path("data/test_grpo/2_cho_gan_nhan.csv"))
    a = ap.parse_args()
    goc, da_viet = doc_csv(a.goc), doc_csv(a.vao)
    loi, canh_bao = kiem(goc, da_viet)
    print(f"{len(da_viet)} dòng; theo phần: {dict(Counter(r.get('phan') for r in da_viet))}")
    print("\n".join(loi) if loi else "không có lỗi")
    if canh_bao:
        print("CẢNH BÁO (nên xem lại):\n" + "\n".join(canh_bao))
    if not loi:
        tao_file_gan_nhan(da_viet, a.ra)
        print(f"đã tạo {a.ra}")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
