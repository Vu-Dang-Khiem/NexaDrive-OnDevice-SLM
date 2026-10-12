"""Dựng bộ dữ liệu train/val VÒNG 3 (+ hai biến thể thí nghiệm) — chạy ở máy, miễn phí, tái lập được.

    python -m gen.dung_vong_3                       # vòng 3: vòng 2 + sửa nhãn bằng code -> final_v3
    python -m gen.dung_vong_3 --mo-ta               # + mô tả hàm trong đáp án (Octopus v2) -> final_v3_mota
    python -m gen.dung_vong_3 --per-fn 1000         # quy mô 1.000 mẫu/hàm như Octopus v2 -> final_v3_1000

Vòng 3 = y hệt vòng 2 (cùng nguồn, cùng lọc, cùng hạt giống chia/bốc) + MỘT thay đổi: sửa nhãn có hệ thống
bằng code (`gen/sua_nhan.py`). Mỗi biến thể thí nghiệm chỉ đổi đúng MỘT biến so với vòng 3:
  --mo-ta     : lấy NGUYÊN train/val vòng 3, chỉ nối mô tả hàm vào đáp án (`gen/mo_ta_dap_an.py`).
  --per-fn N  : trần N mẫu/hàm thay vì 300 (dữ liệu sẵn có trong kho, không sinh thêm).
"""

from __future__ import annotations

import argparse
import json

from gen.dung_vong_2 import PER_FN, P, doc, dung, ghi
from gen.mo_ta_dap_an import them_mo_ta_rows
from gen.records import datasheet
from gen.sua_nhan import sua_rows

RA = P / "final_v3"


def dung_mo_ta() -> dict:
    """Biến thể mô tả hàm: đọc train/val vòng 3 ĐÃ DỰNG, không dựng lại — bảo đảm cùng mẫu, cùng thứ tự."""
    ra = P / "final_v3_mota"
    bc = {}
    for tap in ("train_vi", "val_vi", "val_mau"):
        rows = them_mo_ta_rows(doc(RA / f"{tap}.jsonl"))
        ghi(ra / f"{tap}.jsonl", rows)
        bc[tap] = {"tong": len(rows), "co_mo_ta": sum(1 for r in rows if r["meta"].get("mo_ta_dap_an"))}
    bc["datasheet_train"] = datasheet(doc(ra / "train_vi.jsonl"))
    (ra / "bao_cao_dung.json").write_text(json.dumps(bc, ensure_ascii=False, indent=2), encoding="utf-8")
    return bc


def main() -> None:
    ap = argparse.ArgumentParser(description="Dựng dữ liệu vòng 3 và các biến thể thí nghiệm")
    ap.add_argument("--mo-ta", action="store_true", help="nối mô tả hàm vào đáp án (Octopus v2)")
    ap.add_argument("--per-fn", type=int, default=PER_FN, help="trần mẫu mỗi hàm khi bốc cân bằng")
    args = ap.parse_args()

    if args.mo_ta:
        bc = dung_mo_ta()
        print({k: v for k, v in bc.items() if not k.startswith("datasheet")})
        return

    ra = RA if args.per_fn == PER_FN else P / f"final_v3_{args.per_fn}"
    bc = dung(ra=ra, buoc_sua=sua_rows, per_fn=args.per_fn)
    for k, v in bc.items():
        if k.startswith("datasheet"):
            continue
        if k == "4_sua_nhan":
            print(f"{k}: {v['truoc']} -> {v['sau']}")
            for tap in ("trong_train", "trong_val"):
                for ly_do, n in v[tap].items():
                    print(f"   {tap:12} {ly_do:42} {n}")
            continue
        print(f"{k}: {v}")
    print(f"\ntrain: {bc['datasheet_train']['tong']} mẫu | val: {bc['datasheet_val']['tong']} mẫu -> {ra}")
    print("theo loại:", bc["datasheet_train"]["theo_loai"])


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
