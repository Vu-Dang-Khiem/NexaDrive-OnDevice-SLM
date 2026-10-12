"""Tập test v6 — kiểm TỰ ĐỘNG nhãn ChatGPT gán (trước khi người dùng duyệt tay). Chỉ ĐÁNH DẤU dòng đáng ngờ để
người dùng soi kỹ; không tự sửa nhãn nào (sửa là việc của người duyệt).

    python -m eval.kiem_nhan_v6 --vao data/test_v6/2_da_gan_nhan_210.csv   # -> data/test_v6/2_can_xem.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from eval.chuyen_test import doc_nhan
from eval.kiem_test_v6 import doc_csv
from gen.teacher_prompts import TeacherResponseError
from gen.v6 import ap_quy_uoc
from schema.call_format import CallParseError

CUNG_VIEC = {"open_trunk": {"open_trunk", "close_trunk"}, "close_trunk": {"open_trunk", "close_trunk"},
             "lock_doors": {"lock_doors", "unlock_doors"}, "unlock_doors": {"lock_doors", "unlock_doors"}}
# phần C: tham số phải có theo yêu cầu đề (khớp theo chữ trong cột yeu_cau)
KY_VONG_C = [
    ("MỘT KHOẢNG", lambda a: "delta" in a, "phải dùng delta (tăng / giảm một khoảng)"),
    ("KHÔNG nói số", lambda a: a.get("delta") in (-2, 2), "không nói số -> delta=-2 hoặc 2"),
    ("VỀ một nhiệt độ", lambda a: "temperature" in a and "delta" not in a, "về N độ -> temperature"),
    ("sưởi ấm", lambda a: a.get("mode") == "heat", 'sưởi -> mode="heat"'),
    ("làm lạnh", lambda a: a.get("mode") == "cool", 'làm lạnh -> mode="cool"'),
    ("tự động", lambda a: a.get("mode") == "auto", 'tự động -> mode="auto"'),
    ("CẢ HAI kính phía sau", lambda a: a.get("window") == "rear", 'cả hai kính sau -> window="rear"'),
    ("CẢ HAI kính phía trước", lambda a: a.get("window") == "front", 'cả hai kính trước -> window="front"'),
]


def _doc(nhan: str) -> tuple[list[tuple[str, dict[str, Any]]], bool, str]:
    try:
        calls, hoi = doc_nhan(nhan)
        return calls, hoi, ""
    except (CallParseError, TeacherResponseError, KeyError, ValueError) as e:
        return [], False, f"nhãn không đọc được / sai schema: {e}"


def kiem_dong(r: dict[str, str], de: dict[str, str], khoa: dict[str, str]) -> list[str]:
    van_de: list[str] = []
    phan, ham_dich, kieu = khoa["phan"], khoa["ham_dich"], khoa["kieu"]
    calls, hoi, loi = _doc(r.get("nhan", ""))
    if loi:
        return [loi]
    if not calls:
        return ["thiếu nhãn"]
    if hoi:
        van_de.append("có 'ask:' — tập này toàn câu ra lệnh thẳng, kiểm lại có nên hỏi lại không")
    ten, args = calls[0]
    # hai lượt: con số có thể nằm ở lượt 1 ("Chỉnh bên lái xuống 22 độ" >> "Bên phụ để giống thế") -> xét
    # cả hai câu
    ngu_canh = f"{r.get('luot_1', '')} {r['cau']}" if phan == "D" else r["cau"]
    for t, a in calls:
        sua = ap_quy_uoc(t, ngu_canh, a)
        if sua is None:
            van_de.append(f"{t}: vi phạm quy ước nhiệt độ (temperature / delta)")
        elif sua != a:
            van_de.append(f"{t}: theo quy ước nên là {sua}")
    if phan in ("A", "B", "C") and ten != ham_dich:
        van_de.append(f"khác chức năng của đề ({ham_dich}) — đúng nếu câu ChatGPT viết là việc khác")
    if phan == "C":
        for chu, dk, mo_ta in KY_VONG_C:
            if chu in de.get("yeu_cau", "") and not dk(args):
                van_de.append(f"phần C: {mo_ta}")
    if phan == "D":
        c1, _, loi1 = _doc(r.get("nhan_luot_1", ""))
        if loi1 or not c1:
            van_de.append(f"lượt 1: {loi1 or 'thiếu nhan_luot_1'}")
        elif c1[0][0] != ham_dich:
            van_de.append(f"lượt 1 là {c1[0][0]}, đề yêu cầu {ham_dich}")
        cung = CUNG_VIEC.get(ham_dich, {ham_dich})
        if kieu in ("doi_chieu", "them_nua", "bo_phan_khac") and ten not in cung:
            van_de.append(f"lượt 2 kiểu '{kieu}' phải cùng việc với lượt 1 ({ham_dich}), nhãn lại là {ten}")
        if kieu == "ngoai_pham_vi" and ten != "irrelevant_function":
            van_de.append("lượt 2 kiểu ngoài phạm vi mà nhãn không phải irrelevant_function — kiểm lại")
        if kieu == "lenh_moi" and (ten == "irrelevant_function" or ten in cung):
            van_de.append("lượt 2 kiểu lệnh mới mà nhãn trùng việc lượt 1 hoặc từ chối")
    return van_de


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vao", type=Path, required=True)
    ap.add_argument("--ra", type=Path, default=Path("data/test_v6/2_can_xem.csv"))
    a = ap.parse_args()
    nhan = doc_csv(a.vao)
    de = {r["id"]: r for r in doc_csv(Path("data/test_v6/1_viet_cau.csv"))}
    khoa = {r["id"]: r for r in doc_csv(Path("data/test_v6/khoa_id.csv"))}
    can_xem = []
    for r in nhan:
        if (vd := kiem_dong(r, de[r["id"]], khoa[r["id"]])):
            can_xem.append({"id": r["id"], "phan": khoa[r["id"]]["phan"], "luot_1": r.get("luot_1", ""),
                            "nhan_luot_1": r.get("nhan_luot_1", ""), "cau": r["cau"], "nhan": r["nhan"],
                            "van_de": " | ".join(vd)})
    with open(a.ra, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "phan", "luot_1", "nhan_luot_1", "cau", "nhan", "van_de"])
        w.writeheader()
        w.writerows(can_xem)
    print(f"{len(nhan)} dòng; cần xem kỹ {len(can_xem)} dòng -> {a.ra}")
    print("theo phần:", dict(Counter(x["phan"] for x in can_xem)))
    for x in can_xem:
        truoc = x["luot_1"] + " >> " if x["luot_1"] else ""
        print(f"  [{x['id']:>3} {x['phan']}] {truoc}{x['cau']}  =>  {x['nhan']}")
        print(f"        {x['van_de']}")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
