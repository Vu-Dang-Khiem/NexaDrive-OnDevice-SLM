"""Tập đánh giá GRPO — bước 2 -> bước 3: kiểm TỰ ĐỘNG nhãn ChatGPT gán + tạo BẢNG DUYỆT tiếng Việt cho
người dùng. Không tự sửa nhãn nào (sửa là việc của người duyệt). Một bảng duy nhất: mọi dòng + cột `van_de`
(dòng có vấn đề cần soi kỹ, dòng khác đọc lướt).

Kỳ vọng theo phần (khoá đề, `khoa_id.csv`): R làm luôn đúng chức năng đề · M HỎI LẠI (`ask:`) đúng chức năng
đề · N từ chối · H lượt 1 đúng chức năng đề; lượt 2 "nối tiếp" cùng việc lượt 1, "bóng gió khác" là hỏi lại
việc KHÁC, "ngoài phạm vi" là từ chối. Quy ước đèn pha / cốt KHÔNG áp cho nhãn hỏi lại (câu than phiền nhắc
"xa" để tả vấn đề — cùng lý do `serve/quy_uoc.py`).

    python -m eval.duyet_test_grpo --vao data/test_grpo/2_da_gan_nhan_1_200.csv   # -> 3_bang_duyet.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from eval.bang_duyet_v6 import doc_viec
from eval.kiem_nhan_v6 import CUNG_VIEC, _doc
from eval.kiem_test_v6 import doc_csv
from gen.v6 import ap_quy_uoc
from schema.loader import CATALOG

TU_CHOI = "irrelevant_function"
THU_MUC = Path("data/test_grpo")


def kiem_dong(r: dict[str, str], khoa: dict[str, str]) -> list[str]:
    phan, ham_dich, kieu = khoa["phan"], khoa["ham_dich"], khoa["kieu"]
    calls, hoi, loi = _doc(r.get("nhan", ""))
    if loi:
        return [loi]
    if not calls:
        return ["thiếu nhãn"]
    vd: list[str] = []
    ten = calls[0][0]
    if hoi and all(CATALOG.by_name[t].chi_doc for t, _ in calls):
        vd.append("hàm chỉ đọc (xăng, lốp, thời tiết…) thì làm luôn, bỏ 'ask:' (quy ước dữ liệu train)")
    ngu_canh = f"{r.get('luot_1', '')} {r['cau']}" if phan == "H" else r["cau"]
    for t, a in calls:
        if hoi and t == "set_headlights":
            continue
        sua = ap_quy_uoc(t, ngu_canh, a)
        if sua is None:
            vd.append(f"{t}: vi phạm quy ước nhiệt độ (temperature / delta)")
        elif sua != a:
            vd.append(f"{t}: theo quy ước nên là {sua}")
    if phan == "R":
        if hoi:
            vd.append("câu ra lệnh thẳng mà nhãn HỎI LẠI")
        if ten == TU_CHOI:
            vd.append("câu ra lệnh thẳng mà nhãn TỪ CHỐI")
        elif ten not in CUNG_VIEC.get(ham_dich, {ham_dich}):
            vd.append(f"khác chức năng đề ({ham_dich}) — đúng nếu câu ChatGPT viết là việc khác")
    elif phan == "M":
        if ten == TU_CHOI:
            vd.append("câu than phiền / bóng gió mà nhãn TỪ CHỐI — đề muốn hỏi lại")
        elif not hoi:
            vd.append("câu than phiền / bóng gió mà nhãn LÀM LUÔN — đề muốn HỎI LẠI (ask:)")
        if ten != TU_CHOI and ham_dich not in {t for t, _ in calls}:
            vd.append(f"đề xuất khác chức năng đề ({ham_dich})")
    elif phan == "N":
        if ten != TU_CHOI:
            vd.append("câu ngoài phạm vi mà nhãn KHÔNG từ chối — kiểm lại xe có làm được không")
    else:  # H
        c1, _, loi1 = _doc(r.get("nhan_luot_1", ""))
        if loi1 or not c1:
            vd.append(f"lượt 1: {loi1 or 'thiếu nhan_luot_1'}")
        elif c1[0][0] not in CUNG_VIEC.get(ham_dich, {ham_dich}):
            vd.append(f"lượt 1 là {c1[0][0]}, đề yêu cầu {ham_dich}")
        if kieu == "noi_tiep" and ten not in CUNG_VIEC.get(ham_dich, {ham_dich}):
            vd.append(f"lượt 2 nối tiếp phải cùng việc lượt 1 ({ham_dich}), nhãn lại là {ten}")
        if kieu == "bong_gio_khac":
            if ten == TU_CHOI or not hoi:
                vd.append("lượt 2 bóng gió việc khác — đề muốn HỎI LẠI (ask:)")
            if ten in CUNG_VIEC.get(ham_dich, {ham_dich}):
                vd.append("lượt 2 bóng gió việc KHÁC mà nhãn lại cùng việc lượt 1")
        if kieu == "ngoai_pham_vi" and ten != TU_CHOI:
            vd.append("lượt 2 ngoài phạm vi mà nhãn không phải từ chối")
    return vd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vao", type=Path, required=True)
    ap.add_argument("--ra", type=Path, default=THU_MUC / "3_bang_duyet.csv")
    a = ap.parse_args()
    khoa = {r["id"]: r for r in doc_csv(THU_MUC / "khoa_id.csv")}
    rows = doc_csv(a.vao)
    if {r["id"] for r in rows} != set(khoa):
        raise SystemExit(f"id lệch so với đề: {len(rows)} dòng")
    cot = ["id", "phan", "van_de", "cau_luot_1", "tro_ly_lam_luot_1", "cau", "tro_ly_se_lam", "SAI",
           "sua_thanh", "nhan_goc", "ghi_chu_chatgpt"]
    so_van_de: Counter[str] = Counter()
    with open(a.ra, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cot)
        w.writeheader()
        for r in sorted(rows, key=lambda x: int(x["id"])):
            k = khoa[r["id"]]
            vd = kiem_dong(r, k)
            if vd:
                so_van_de[k["phan"]] += 1
            try:
                viec = doc_viec(r["nhan"], r["cau"])
                viec1 = doc_viec(r.get("nhan_luot_1", ""), r.get("luot_1", ""))
            except Exception as e:  # nhãn sai schema -> vẫn ghi dòng để người dùng sửa
                viec, viec1 = f"(không đọc được nhãn: {e})", ""
            w.writerow({"id": r["id"], "phan": k["phan"], "van_de": " | ".join(vd),
                        "cau_luot_1": r.get("luot_1", ""), "tro_ly_lam_luot_1": viec1, "cau": r["cau"],
                        "tro_ly_se_lam": viec, "SAI": "", "sua_thanh": "", "nhan_goc": r["nhan"],
                        "ghi_chu_chatgpt": r.get("ghi_chu", "")})
    print(f"{len(rows)} dòng -> {a.ra}; dòng có vấn đề theo phần: {dict(so_van_de)} "
          f"(tổng {sum(so_van_de.values())})")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
