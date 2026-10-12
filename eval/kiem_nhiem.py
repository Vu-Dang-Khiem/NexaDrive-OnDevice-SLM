"""Kiểm nhiễm chéo: câu trong tập test có xuất hiện trong train/val không.

VÌ SAO CẦN. Tập test là 296 câu người viết tay, còn train/val do teacher sinh — nghe thì
tưởng không thể trùng. Nhưng lệnh xe thường rất ngắn và ai cũng nói giống nhau ("còn bao
nhiêu xăng", "mở khóa tất cả các cửa"), nên teacher sinh ra đúng câu đó là chuyện bình
thường. Đo thật trên vòng 1: **26/296 câu test trùng Y HỆT một câu trong train**, thêm 10
câu nữa trùng với val — tổng 34 câu (11,5%).

Hậu quả lên điểm, đo trên chính checkpoint vòng 1:

    34 câu ĐÃ THẤY khi train : nhóm rõ ràng 83,9%
    262 câu CHƯA TỪNG THẤY   : nhóm rõ ràng 67,0%
    gộp chung (đã báo cáo)   : nhóm rõ ràng 71,2%   <- bị thổi lên 4,2 điểm

Con số đem vào luận văn phải là con số trên phần CHƯA TỪNG THẤY.

Vì sao khâu chia train/val không chặn được: `gen/chia_tap.py` khử trùng lặp GIỮA train và
val (cả hai cùng từ một kho), còn tập test đi đường khác — người viết tay, nhập từ CSV —
nên chưa bao giờ được đối chiếu với train. Đây là chỗ hở của quy trình, không phải lỗi
của ai.

Dùng lại `chuan_hoa` + `nhom_gan_trung` của gen/chia_tap.py để bắt cả câu GẦN trùng
("mở cốp xe" vs "mở cốp xe."), không chỉ trùng tuyệt đối.

    python -m eval.kiem_nhiem --test data/processed/final/test_vi.jsonl \\
        --nguon data/processed/final/train_vi.jsonl data/processed/final/val_vi.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gen.chia_tap import chuan_hoa, nhom_gan_trung
from gen.records import query_from_call_prompt


def _cau_cua(row: dict[str, Any]) -> str | None:
    try:
        return query_from_call_prompt(row["prompt"])
    except ValueError:
        return None


def tim_nhiem(
    test: list[dict[str, Any]], nguon: list[dict[str, Any]], gan_trung: bool = True
) -> dict[str, Any]:
    """Trả về danh sách câu test bị nhiễm + chỉ số (theo vị trí trong `test`)."""
    cau_test = [_cau_cua(r) for r in test]
    cau_nguon = [c for c in (_cau_cua(r) for r in nguon) if c]

    trung_tuyet_doi = {chuan_hoa(c) for c in cau_nguon}
    bi_nhiem: dict[int, str] = {}
    for i, c in enumerate(cau_test):
        if c and chuan_hoa(c) in trung_tuyet_doi:
            bi_nhiem[i] = "trung_y_het"

    if gan_trung:
        # Gộp chung rồi nhóm: câu test nào rơi vào cùng nhóm với một câu nguồn là gần trùng.
        hop = [c or "" for c in cau_test] + cau_nguon
        ma = nhom_gan_trung(hop)
        ma_nguon = set(ma[len(cau_test) :])
        for i in range(len(cau_test)):
            if i not in bi_nhiem and cau_test[i] and ma[i] in ma_nguon:
                bi_nhiem[i] = "gan_trung"

    return {
        "so_cau_test": len(test),
        "so_bi_nhiem": len(bi_nhiem),
        "ty_le_%": round(len(bi_nhiem) / len(test) * 100, 1) if test else 0.0,
        "chi_so": sorted(bi_nhiem),
        "kieu": bi_nhiem,
        "vi_du": [
            {"cau": cau_test[i], "kieu": bi_nhiem[i], "fn": test[i]["meta"]["fn"]}
            for i in sorted(bi_nhiem)[:15]
        ],
    }


def tach_sach(
    test: list[dict[str, Any]], nhiem: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Trả về (phần chưa từng thấy, phần đã thấy khi train)."""
    ban = set(nhiem["chi_so"])
    sach = [r for i, r in enumerate(test) if i not in ban]
    da_thay = [r for i, r in enumerate(test) if i in ban]
    return sach, da_thay


def _doc(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Kiểm câu test có lọt vào train/val không")
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--nguon", required=True, type=Path, nargs="+", help="train và/hoặc val")
    ap.add_argument("--ghi-sach", type=Path, default=None, help="ghi phần chưa từng thấy ra đây")
    ap.add_argument("--bo-gan-trung", action="store_true", help="chỉ bắt câu trùng y hệt")
    args = ap.parse_args()

    test = _doc(args.test)
    nguon: list[dict[str, Any]] = []
    for p in args.nguon:
        nguon.extend(_doc(p))

    kq = tim_nhiem(test, nguon, gan_trung=not args.bo_gan_trung)
    print(f"tập test : {kq['so_cau_test']} câu")
    print(f"bị nhiễm : {kq['so_bi_nhiem']} câu ({kq['ty_le_%']}%)  <- ĐÃ xuất hiện khi train")
    if kq["vi_du"]:
        print("\nví dụ:")
        for v in kq["vi_du"]:
            print(f"   [{v['kieu']:11}] {v['fn']:20} {v['cau'][:50]}")
    if kq["so_bi_nhiem"]:
        print(
            "\nKhông xoá câu nào — chấm TÁCH LÀM HAI và lấy phần 'chưa từng thấy' làm số\n"
            "chính thức. Xoá câu người viết tay thì mất luôn thông tin, mà giữ lẫn vào thì\n"
            "thổi điểm lên."
        )

    if args.ghi_sach:
        sach, _ = tach_sach(test, kq)
        args.ghi_sach.parent.mkdir(parents=True, exist_ok=True)
        with open(args.ghi_sach, "w", encoding="utf-8") as f:
            for r in sach:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nĐã ghi {len(sach)} câu sạch -> {args.ghi_sach}")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
