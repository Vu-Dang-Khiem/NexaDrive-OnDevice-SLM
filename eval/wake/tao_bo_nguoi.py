"""Wake-word — danh sách người dùng tự nói "hey nexa" (PLAN.md Phase 6b). KHOÁ trước khi đo / train. Ghi
bằng đúng công cụ của bộ B / C:

    python -m eval.wake.tao_bo_nguoi --bo bo_wake          # bản 1: 40 lần ĐỂ ĐO (đã ghi 2026-10-08)
    python -m eval.wake.tao_bo_nguoi --bo bo_wake_train    # bản 2: 60 lần ĐỂ TRAIN (45 train / 15 val)
    python -m eval.wake.tao_bo_nguoi --bo bo_wake_2        # bản 2: 40 lần MỚI ĐỂ ĐO
    conda run -n giong_noi python -m eval.stt.ghi_am --nguoi khiem --bo <bộ>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BINH = "Hey Nexa  (nói bình thường, ngồi trước laptop)"
NHANH = "Hey Nexa  (nói NHANH)"
NHO = "Hey Nexa  (nói NHỎ)"
XA = "Hey Nexa  (ngồi XA laptop ~1–1,5 m)"
OI = "Hey Nexa ơi"
LIEN_1 = ["mở cốp", "bật điều hoà", "hạ kính ghế phụ", "nóng quá", "tắt nhạc", "xăng còn bao nhiêu"]
LIEN_2 = ["đóng cốp lại", "tăng âm lượng", "bật đèn khẩn cấp", "lạnh quá", "dẫn đường về nhà", "khoá cửa xe"]
BO = {
    "bo_wake": [(10, BINH), (6, NHANH), (6, NHO), (6, XA), (6, OI)]
    + [(1, f"Hey Nexa {c}  (nói LIỀN, không ngừng)") for c in LIEN_1],
    "bo_wake_train": [(15, BINH), (12, NHANH), (12, NHO), (12, XA), (9, OI)],
    "bo_wake_2": [(10, BINH), (6, NHANH), (6, NHO), (6, XA), (6, OI)]
    + [(1, f"Hey Nexa {c}  (nói LIỀN, không ngừng)") for c in LIEN_2],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bo", choices=list(BO), required=True)
    bo = ap.parse_args().bo
    ra = Path("data/stt") / bo / "danh_sach.jsonl"
    if ra.exists():
        raise SystemExit(f"{ra} đã có — danh sách đã khoá, không tạo lại")
    if bo == "bo_wake_train":
        # chia THEO TỪNG KIỂU NÓI (mỗi kiểu: lần thứ 4, 8, 12… -> val) rồi xếp XEN KẼ các kiểu (lần lượt bình
        # thường, nhanh, nhỏ, xa, ơi…) để người ghi không đọc 15 lần y hệt liền nhau
        nhom = [[(c, "val" if (j + 1) % 4 == 0 else "train") for j in range(n)] for n, c in BO[bo]]
        ds = [
            nhom[t][j] for j in range(max(len(g) for g in nhom)) for t in range(len(nhom)) if j < len(nhom[t])
        ]
    else:
        ds = [(c, "test") for n, c in BO[bo] for _ in range(n)]
    ra.parent.mkdir(parents=True, exist_ok=True)
    with ra.open("w", encoding="utf-8") as f:
        for k, (c, tach) in enumerate(ds, 1):
            f.write(json.dumps({"tep": f"w{k:02d}.wav", "cau": c, "tach": tach}, ensure_ascii=False) + "\n")
    print(len(ds), "lần ->", ra)


if __name__ == "__main__":
    main()
