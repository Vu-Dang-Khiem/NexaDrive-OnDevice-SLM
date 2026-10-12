"""Dựng bộ dữ liệu train/val VÒNG 2 — một lệnh, chạy ở máy, miễn phí, tái lập được.

    python -m gen.dung_vong_2

Các bước (mỗi bước ứng với một quyết định đã ghi trong PLAN.md):

  1. Gộp: kho vòng 1 (`full2/train_vi_sach.jsonl`, đã thay câu từ chối lượt 2) + bổ sung vòng 2
     (câu dương từ đồng nghĩa + câu từ chối dạng mệnh lệnh) + mẫu hỏi lại ĐÃ QUA QUY TRÌNH CHẤM MỚI
     (`ask_cham_moi/*_giu.jsonl`). KHÔNG lấy file train vòng 1: nó chứa sẵn 803 mẫu hỏi lại cũ, có
     nhãn sai kiểu "chỗ đỗ không an toàn -> đèn khẩn cấp".
  2. Bỏ câu "Điếc tai với tiếng ồn bên ngoài/ngoài kia" -> chỉnh âm lượng: cả hai giám khảo cùng sai,
     người gán là đóng kính. Câu "tiếng ồn này" (không rõ nguồn) giữ lại.
  3. Lọc bằng code (`gen/loc.py`): chữ lạ, lộ tên hàm, câu lai tiếng Anh, tham số sai, số bịa.
  4. Đánh dấu `meta.uu_tien` cho câu dương vòng 2 -> khâu bốc cân bằng giữ chúng TRƯỚC
     (không thì chỉ ~1/9 lọt vào, bản sửa khoá từ vựng mất tác dụng).
  5. Chia train/val THEO NHÓM câu gần trùng (hàm băm cố định), rồi bốc cân bằng train:
     300 mẫu/hàm như vòng 1 (giữ nguyên để so sánh được), trần mẫu hỏi lại mỗi hàm, giữ hết câu
     từ chối khó, dương:âm 1:1.
  6. Kiểm nhiễm: bỏ mọi mẫu train/val trùng / gần trùng tập test chính thức (230 câu sạch + 28 câu
     bổ sung).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from eval.kiem_nhiem import tim_nhiem
from gen.chia_tap import chia_theo_nhom
from gen.loc import loc_rows
from gen.records import datasheet, query_from_call_prompt
from gen.subsample import subsample

P = Path("data/processed")
NGUON_V1 = P / "full2/train_vi_sach.jsonl"
NGUON_V2 = P / "v2/v2_bo_sung_vi.jsonl"
NGUON_ASK = [
    P / "v2/ask_cham_moi/v1_ask_sach_giu.jsonl",
    P / "v2/ask_cham_moi/v2_ask_train_vi_giu.jsonl",
    # sinh bổ sung cho 6 hàm mỏng với hướng dẫn tình huống (PLAN.md VÒNG 2 — I)
    P / "v3/ask_cham_moi/v3_ask_train_vi_giu.jsonl",
]
TAP_TEST = [P / "final/test_vi_sach_v2.jsonl", P / "final/test_vi_bo_sung.jsonl"]
RA = P / "final_v2"

PER_FN = 300  # giữ như vòng 1 để so sánh chỉ khác DỮ LIỆU, không khác quy mô
TRAN_ASK_MOI_HAM = 80  # set_ac có 387 mẫu hỏi lại -> cắt; hàm khác đều dưới mức này
SEED = 42

_OI_BEN_NGOAI = re.compile(r"bên ngoài|ngoài kia", re.IGNORECASE)


def doc(p: Path) -> list[dict[str, Any]]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def ghi(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def bo_on_ben_ngoai(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    giu, bo = [], 0
    for r in rows:
        if r["meta"]["type"] == "ask" and r["meta"]["fn"] == "set_volume" and _OI_BEN_NGOAI.search(
            query_from_call_prompt(r["prompt"])
        ):
            bo += 1
            continue
        giu.append(r)
    return giu, bo


def bo_nhiem(rows: list[dict[str, Any]], test: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Bỏ mẫu train/val trùng hoặc gần trùng câu test chính thức."""
    kq = tim_nhiem(rows, test)
    ban = set(kq["chi_so"])
    return [r for i, r in enumerate(rows) if i not in ban], len(ban)


def dung(
    ty_le_val: float = 0.1,
    ra: Path = RA,
    buoc_sua: Callable[[list[dict[str, Any]]], tuple[list[dict[str, Any]], dict[str, Any]]] | None = None,
    per_fn: int = PER_FN,
) -> dict[str, Any]:
    """`buoc_sua` (vòng 3): sửa nhãn bằng code SAU khi chia (val) và SAU khi bốc cân bằng (train).

    Chia và bốc đều dùng MỘT bộ sinh ngẫu nhiên chung cho mọi (hàm, loại). Sửa trước -> số mẫu
    của vài hàm đổi -> bộ sinh xê dịch -> train/val đổi ở CẢ những hàm không bị sửa (đã đo: sửa
    trước khi chia làm val 3.423 -> 3.284; sửa trước khi bốc làm 3.383 mẫu train khác vòng 2).
    Sửa sau cùng -> vòng 3 là ĐÚNG các mẫu của vòng 2, chỉ khác nhãn đã sửa và mẫu bị bỏ.
    """
    bao_cao: dict[str, Any] = {}

    v1 = doc(NGUON_V1)
    v2 = doc(NGUON_V2)
    for r in v2:
        if r["meta"]["fn"] != "irrelevant_function":
            r["meta"]["uu_tien"] = True  # câu dương vòng 2 (từ đồng nghĩa)
        r["meta"]["vong"] = 2
    ask: list[dict[str, Any]] = []
    for p in NGUON_ASK:
        ask += doc(p)
    ask, bo_on = bo_on_ben_ngoai(ask)
    bao_cao["1_gop"] = {"kho_v1": len(v1), "bo_sung_v2": len(v2), "hoi_lai_da_cham": len(ask) + bo_on}
    bao_cao["2_bo_on_ben_ngoai"] = bo_on

    kho = v1 + v2 + ask
    kho, bi_loc = loc_rows(kho)
    bao_cao["3_loc_code"] = {"con": len(kho), "bi_loai": len(bi_loc),
                             "ly_do": dict(Counter(r["ly_do_loai"].split(":")[0] for r in bi_loc))}
    train_day, val = chia_theo_nhom(kho, ty_le_val=ty_le_val, seed=SEED)
    train = subsample(train_day, per_fn=per_fn, seed=SEED, tran_ask_moi_ham=TRAN_ASK_MOI_HAM)
    bao_cao["5_chia"] = {"train_day": len(train_day), "train_can_bang": len(train), "val": len(val)}
    if buoc_sua is not None:
        truoc = {"train": len(train), "val": len(val)}
        train, bc_train = buoc_sua(train)
        val, bc_val = buoc_sua(val)
        bao_cao["4_sua_nhan"] = {"truoc": truoc, "sau": {"train": len(train), "val": len(val)},
                                 "trong_train": bc_train["dem"], "trong_val": bc_val["dem"],
                                 "vi_du": {**bc_val["vi_du"], **bc_train["vi_du"]}}

    test: list[dict[str, Any]] = []
    for p in TAP_TEST:
        test += doc(p)
    train, nhiem_train = bo_nhiem(train, test)
    val, nhiem_val = bo_nhiem(val, test)
    bao_cao["6_bo_nhiem_test"] = {"train": nhiem_train, "val": nhiem_val}

    ghi(ra / "train_vi.jsonl", train)
    ghi(ra / "val_vi.jsonl", val)
    bao_cao["datasheet_train"] = datasheet(train)
    bao_cao["datasheet_val"] = datasheet(val)
    bao_cao["train_vong_2"] = {
        "cau_duong_vong_2": sum(1 for r in train if r["meta"].get("uu_tien")),
        "tu_choi_kho": sum(1 for r in train if r["meta"].get("nhom_phu") == "tu_choi_kho"),
        "hoi_lai": sum(1 for r in train if r["meta"]["type"] == "ask"),
    }
    (ra / "bao_cao_dung.json").write_text(json.dumps(bao_cao, ensure_ascii=False, indent=2), encoding="utf-8")
    return bao_cao


def main() -> None:
    ap = argparse.ArgumentParser(description="Dựng bộ dữ liệu train/val vòng 2")
    ap.add_argument("--ty-le-val", type=float, default=0.1)
    args = ap.parse_args()
    bc = dung(args.ty_le_val)
    for k, v in bc.items():
        if not k.startswith("datasheet"):
            print(f"{k}: {v}")
    print(f"\ntrain: {bc['datasheet_train']['tong']} mẫu | val: {bc['datasheet_val']['tong']} mẫu -> {RA}")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
