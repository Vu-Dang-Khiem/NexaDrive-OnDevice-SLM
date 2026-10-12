"""Chuyển tập test NGƯỜI VIẾT TAY (CSV) sang JSONL để chấm điểm — chạy local, miễn phí.

Tập test là thước đo cuối cùng của cả đồ án, nên KHÔNG bao giờ được sinh bằng model:
data train do teacher Qwen sinh, nếu test cũng vậy thì chỉ đo được "student có bắt chước
teacher giống không", không đo được "agent có dùng được ngoài đời không".

Người viết chỉ cần điền 1 file CSV 3 cột (xem data/test_tay/test_vi.csv):

    cau                              | nhan                                   | ghi_chu
    Mở cốp xe                        | open_trunk()                           |
    Bật điều hòa 26 độ ghế lái       | set_ac(state="on", temperature=26, zone="driver") |
    Nóng quá                         | ask: set_ac(state="on")                | câu mơ hồ -> hỏi lại
    Kể chuyện cười đi                | irrelevant_function()                  |

Script này kiểm từng dòng (tên hàm có thật, tham số đúng kiểu, đúng enum/khoảng giá trị,
đủ tham số bắt buộc) rồi in ra lỗi kèm SỐ DÒNG để sửa, và chỉ ghi file khi không còn lỗi.

    python -m eval.chuyen_test --input data/test_tay/test_vi.csv \\
                               --output data/processed/final/test_vi.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from gen.records import IM_END
from gen.teacher_prompts import TeacherResponseError, _kiem_tra_tham_so
from schema.call_format import CallParseError, parse_readable, render_completion
from schema.loader import CATALOG

TIEN_TO_HOI_LAI = "ask:"


def doc_nhan(nhan: str) -> tuple[list[tuple[str, dict[str, Any]]], bool]:
    """Đọc ô `nhan` -> (danh sách lời gọi, có phải kiểu hỏi lại không).

    Kiểm luôn tham số theo schema, nên người viết tay gõ sai enum hay quá khoảng giá trị
    là biết ngay, không lọt vào tập test.
    """
    s = nhan.strip()
    la_hoi_lai = s.lower().startswith(TIEN_TO_HOI_LAI)
    if la_hoi_lai:
        s = s[len(TIEN_TO_HOI_LAI) :].strip()
    calls = parse_readable(s)
    da_kiem = [(ten, _kiem_tra_tham_so(args, CATALOG.by_name[ten])) for ten, args in calls]
    return da_kiem, la_hoi_lai


def dong_thanh_row(cau: str, nhan: str, lang: str = "vi", nhom_phu: str = "") -> dict[str, Any]:
    """`nhom_phu` = nhãn loại câu, để lúc chấm tách được nhóm KHÓ với nhóm DỄ.

    Cần thiết vì điểm gộp của nhóm từ chối che mất điểm yếu thật: model từ chối tốt
    câu tán gẫu nhưng kém hẳn với yêu cầu ĐÚNG chủ đề xe mà xe không hỗ trợ
    ("bật sưởi ghế") — đúng điểm yếu của functional token mà Huawei 2026 đo được.
    """
    calls, la_hoi_lai = doc_nhan(nhan)
    if la_hoi_lai and any(ten == "irrelevant_function" for ten, _ in calls):
        raise CallParseError("câu ngoài phạm vi thì không dùng tiền tố 'ask:'")
    if la_hoi_lai and all(CATALOG.by_name[ten].chi_doc for ten, _ in calls):
        # Luật "chỉ đọc không hỏi lại" — giữ tập test khớp đúng quy ước của data train
        raise CallParseError(
            f"hàm chỉ đọc ({', '.join(t for t, _ in calls)}) thì làm luôn, bỏ tiền tố 'ask:'"
        )
    return {
        "prompt": CATALOG.render_call_prompt(cau),
        "completion": render_completion(calls, ask=la_hoi_lai) + IM_END,
        "meta": {
            "lang": lang,
            "type": "ask" if la_hoi_lai else "call",
            "fn": calls[0][0],
            "fns": [ten for ten, _ in calls],
            "nguon": "nguoi_viet_tay",
            "nhom_phu": nhom_phu.strip() or "khac",
        },
    }


def doc_csv(path: Path, lang: str = "vi") -> tuple[list[dict[str, Any]], list[str]]:
    """Trả về (rows hợp lệ, danh sách lỗi kèm số dòng)."""
    rows: list[dict[str, Any]] = []
    loi: list[str] = []
    da_thay: dict[str, int] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for so_dong, muc in enumerate(csv.DictReader(f), start=2):  # dòng 1 là tiêu đề
            cau = (muc.get("cau") or "").strip()
            nhan = (muc.get("nhan") or "").strip()
            if not cau and not nhan:
                continue
            if not cau or not nhan:
                loi.append(f"dòng {so_dong}: thiếu cột 'cau' hoặc 'nhan'")
                continue
            khoa = " ".join(cau.split()).lower()
            if khoa in da_thay:
                loi.append(f"dòng {so_dong}: câu trùng với dòng {da_thay[khoa]}")
                continue
            da_thay[khoa] = so_dong
            try:
                rows.append(dong_thanh_row(cau, nhan, lang, muc.get("nhom_phu") or ""))
            except (CallParseError, TeacherResponseError, KeyError) as e:
                loi.append(f"dòng {so_dong}: {e}")
    return rows, loi


def thong_ke(rows: list[dict[str, Any]]) -> dict[str, Any]:
    theo_nhom = {"ro_rang": 0, "mo_ho": 0, "ngoai_pham_vi": 0}
    theo_ham: dict[str, int] = {}
    for r in rows:
        if r["meta"]["fn"] == "irrelevant_function":
            theo_nhom["ngoai_pham_vi"] += 1
        elif r["meta"]["type"] == "ask":
            theo_nhom["mo_ho"] += 1
        else:
            theo_nhom["ro_rang"] += 1
        theo_ham[r["meta"]["fn"]] = theo_ham.get(r["meta"]["fn"], 0) + 1
    thieu = [f.name for f in CATALOG.functions if f.name not in theo_ham]
    theo_nhom_phu: dict[str, int] = {}
    for r in rows:
        khoa = r["meta"].get("nhom_phu", "khac")
        theo_nhom_phu[khoa] = theo_nhom_phu.get(khoa, 0) + 1
    return {
        "tong": len(rows),
        "theo_nhom": theo_nhom,
        "theo_nhom_phu": dict(sorted(theo_nhom_phu.items())),
        "theo_ham": dict(sorted(theo_ham.items())),
        "ham_chua_co_cau_test": thieu,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Kiểm và chuyển tập test viết tay sang JSONL")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--lang", default="vi")
    args = ap.parse_args()

    rows, loi = doc_csv(args.input, args.lang)
    if loi:
        print(f"CÓ {len(loi)} LỖI — sửa file CSV rồi chạy lại, chưa ghi gì cả:\n")
        for x in loi:
            print(f"  {x}")
        raise SystemExit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"OK: {len(rows)} câu -> {args.output}\n")
    print(json.dumps(thong_ke(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
