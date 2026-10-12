"""Bốc mẫu con từ kho dữ liệu lớn — logic thuần, chạy local, không tốn GPU.

Ý tưởng (đã chốt): sinh MỘT LẦN thật nhiều (vd 1000 câu/hàm), sau đó muốn train
với bao nhiêu mẫu thì bốc ra bấy nhiêu — miễn phí, tức thì, không phải chạy lại
teacher lần nào nữa.

Quan trọng cho phần thí nghiệm của đồ án: khi so sánh 100 / 500 / 1000 mẫu (bảng
Table 1 của paper), các mức đều bốc từ CÙNG một kho -> chỉ khác đúng 1 biến là số
lượng. Nếu mỗi mức sinh data riêng thì không biết chênh lệch độ chính xác là do số
lượng hay do 2 đợt data khác nhau.

    python -m gen.subsample --input data/processed/train_vi.jsonl \\
                            --per-fn 300 --output data/processed/train_vi_300.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from gen.records import datasheet


def doc_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ghi_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


NEGATIVE_FN = "irrelevant_function"

# Nhãn `meta.nhom_phu` của mẫu từ chối dạng MỆNH LỆNH (gen/modal_app.py::sinh_vong_2).
# Luôn giữ hết loại này khi bốc — chúng là thuốc chữa lỗi gọi bừa, mà nếu bốc ngẫu nhiên
# chung với ~8.000 mẫu từ chối dễ thì chỉ còn lại một phần nhỏ.
TU_CHOI_KHO = "tu_choi_kho"


def _boc_uu_tien(ds: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    """Bốc n mẫu, lấy mẫu có `meta.uu_tien` TRƯỚC, phần còn lại mới bốc ngẫu nhiên.

    Vì sao: vòng 2 sinh ~124 câu/hàm để sửa khoá từ vựng ("hạ kính"), trong khi kho vòng 1 có
    ~1.000 câu/hàm. Bốc ngẫu nhiên 300 câu thì chỉ ~1/9 số câu vòng 2 lọt vào — control_window
    còn ~30 câu có chữ "kính" trên 300, bản sửa gần như mất tác dụng. Không có mẫu nào được đánh
    dấu thì hành vi y hệt trước (tái lập được vòng 1).
    """
    if n <= 0:
        return []
    uu = [r for r in ds if r["meta"].get("uu_tien")]
    con = [r for r in ds if not r["meta"].get("uu_tien")]
    if len(uu) >= n:
        return rng.sample(uu, n)
    return uu + rng.sample(con, n - len(uu))


def can_bang_ask(
    asks: list[dict[str, Any]], tran_moi_ham: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Cắt mỗi hàm được đề xuất xuống tối đa `tran_moi_ham` mẫu ask.

    Vì sao: vòng 1 có 297/803 mẫu ask đề xuất `set_ac` (37%). Model học thành "bí thì
    đề nghị bật điều hoà": trong 13 câu mơ hồ chọn sai hàm trên tập test, 8 câu nó ra
    `set_ac`.

    Sinh đều theo hàm KHÔNG đủ. Đo ở vòng 2: chỉ sinh cho 7 hàm, KHÔNG có set_ac, mà vẫn
    ra 170/517 mẫu set_ac. Nguyên nhân nằm ở KHÂU SINH chứ không ở khâu gán nhãn: bảo
    teacher viết câu than phiền cho `set_fan_speed`, nó trôi sang than nóng lạnh chung
    chung ("Nóng quá, mồ hôi ướt hết áo") — và khâu gán nhãn xếp ĐÚNG những câu đó về điều
    hoà. Hàm có hàng xóm gần bị hút mạnh nhất: set_fan_speed chỉ giữ được 14%, control_window
    13%; close_trunk (không có hàng xóm) giữ 90%. Nên phải cắt SAU khi gán nhãn, ở đây.
    """
    theo_ham: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in asks:
        theo_ham[r["meta"]["fn"]].append(r)
    ra: list[dict[str, Any]] = []
    for fn in sorted(theo_ham):
        ds = theo_ham[fn]
        ra.extend(rng.sample(ds, min(len(ds), tran_moi_ham)))
    return ra


def subsample(
    rows: list[dict[str, Any]],
    per_fn: int,
    seed: int = 42,
    giu_ti_le_reply: bool = True,
    so_negative: int | None = None,
    so_ask: int | None = None,
    tran_ask_moi_ham: int | None = None,
) -> list[dict[str, Any]]:
    """Bốc tối đa `per_fn` mẫu cho mỗi hàm THẬT, giữ nguyên tỉ lệ loại call/reply.

    - Bốc theo từng hàm (không bốc ngẫu nhiên toàn cục) để mọi hàm đều có đủ mẫu,
      tránh trường hợp hàm hiếm bị bốc trượt hoàn toàn.
    - `irrelevant_function` KHÔNG bị giới hạn bởi `per_fn`: nó là 1 nhãn đứng cho
      cả nhóm câu ngoài phạm vi, cần cỡ ngang TỔNG positive chứ không phải ngang
      1 hàm. Nếu cắt nó xuống `per_fn` thì tỉ lệ positive:negative thành 24:1.
      Mặc định `so_negative` = tổng positive đã bốc (giữ tỉ lệ 1:1 như paper).
    - Mẫu "ask" (câu ngầm hiểu, nhãn <nexa_ask>) là nhóm nhỏ riêng: mặc định giữ HẾT
      (`so_ask`), không cắt theo `per_fn`; được tính là positive khi tính tỉ lệ 1:1.
    - seed cố định -> chạy lại ra đúng tập cũ, tái lập được cho luận văn.
    """
    rng = random.Random(seed)

    # gom theo (hàm, loại) để giữ được tỉ lệ call/reply trong từng hàm
    nhom: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        nhom[(r["meta"]["fn"], r["meta"]["type"])].append(r)

    ket_qua: list[dict[str, Any]] = []
    ten_ham = sorted({fn for fn, _ in nhom if fn != NEGATIVE_FN})
    for fn in ten_ham:
        calls = nhom.get((fn, "call"), [])
        replies = nhom.get((fn, "reply"), [])

        if giu_ti_le_reply and calls and replies:
            # giữ đúng tỉ lệ reply/call vốn có của hàm này
            ti_le = len(replies) / (len(calls) + len(replies))
            n_reply = min(len(replies), round(per_fn * ti_le))
            n_call = min(len(calls), per_fn - n_reply)
        else:
            n_call = min(len(calls), per_fn)
            n_reply = min(len(replies), max(0, per_fn - n_call))

        ket_qua.extend(_boc_uu_tien(calls, n_call, rng))
        ket_qua.extend(_boc_uu_tien(replies, n_reply, rng))

    # ask: bốc tối đa so_ask (mặc định giữ hết) — không có dòng này thì mẫu ask bị rơi mất
    # âm thầm, vì vòng lặp phía trên chỉ lấy loại call/reply
    asks = [r for r in rows if r["meta"]["type"] == "ask"]
    if tran_ask_moi_ham is not None:
        asks = can_bang_ask(asks, tran_ask_moi_ham, rng)
    n_ask = len(asks) if so_ask is None else min(len(asks), so_ask)
    ket_qua.extend(rng.sample(asks, n_ask) if n_ask else [])

    # negative: bốc chung cả call lẫn reply (nếu có) -> tỉ lệ loại tự xấp xỉ theo kho.
    # Mẫu từ chối KHÓ (dạng mệnh lệnh) được giữ hết trước, phần còn lại mới bốc ngẫu nhiên.
    negatives = nhom.get((NEGATIVE_FN, "call"), []) + nhom.get((NEGATIVE_FN, "reply"), [])
    kho = [r for r in negatives if r["meta"].get("nhom_phu") == TU_CHOI_KHO]
    de = [r for r in negatives if r["meta"].get("nhom_phu") != TU_CHOI_KHO]
    muon = len(ket_qua) if so_negative is None else so_negative
    n_kho = min(len(kho), muon)
    n_de = min(len(de), max(0, muon - n_kho))
    ket_qua.extend(rng.sample(kho, n_kho) if n_kho else [])
    ket_qua.extend(rng.sample(de, n_de) if n_de else [])

    rng.shuffle(ket_qua)
    return ket_qua


def main() -> None:
    ap = argparse.ArgumentParser(description="Bốc mẫu con từ kho dữ liệu lớn")
    ap.add_argument("--input", required=True, type=Path, help="file JSONL kho đầy đủ")
    ap.add_argument("--per-fn", required=True, type=int, help="số mẫu tối đa mỗi hàm")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42, help="cố định để tái lập được")
    ap.add_argument(
        "--so-negative",
        type=int,
        default=None,
        help="số mẫu irrelevant_function (mặc định = tổng positive, tỉ lệ 1:1)",
    )
    ap.add_argument("--so-ask", type=int, default=None, help="số mẫu câu ngầm hiểu (mặc định giữ hết)")
    ap.add_argument("--tran-ask-moi-ham", type=int, default=None, help="trần mẫu ask cho mỗi hàm đề xuất")
    args = ap.parse_args()

    rows = doc_jsonl(args.input)
    con = subsample(
        rows,
        per_fn=args.per_fn,
        seed=args.seed,
        so_negative=args.so_negative,
        so_ask=args.so_ask,
        tran_ask_moi_ham=args.tran_ask_moi_ham,
    )
    ghi_jsonl(args.output, con)

    print(f"kho gốc : {len(rows):>6} mẫu")
    n_neg = sum(1 for r in con if r["meta"]["fn"] == NEGATIVE_FN)
    print(f"bốc ra  : {len(con):>6} mẫu (tối đa {args.per_fn}/hàm thật, seed={args.seed})")
    print(f"pos:neg : {len(con) - n_neg}:{n_neg}")
    print(f"ghi vào : {args.output}")
    print(json.dumps(datasheet(con), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
