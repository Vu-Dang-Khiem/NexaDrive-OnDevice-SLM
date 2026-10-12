"""Bốc 400 câu hỏi lại để NGƯỜI gán nhãn — đáp án để chấm ngược giám khảo (PLAN.md E0b).

Vì sao cần: bộ hiệu chỉnh 16 câu chỉ đủ làm cổng chặn; đọc tay 30 mẫu chỉ đủ tìm kiểu lỗi.
Muốn nói "giám khảo / quy trình đúng bao nhiêu %" trong luận văn thì cần đáp án do người gán,
đủ nhiều. 400 câu -> khoảng tin cậy 95% rộng ~7 điểm, và mỗi hàm ~30 câu nên nói được riêng
cho từng hàm.

Ba nguyên tắc của file này:
  1. BỐC PHÂN TẦNG theo hàm trong nhãn — bốc ngẫu nhiên thì `set_ac` chiếm 1/3, hàm nhỏ
     như `lock_doors` gần như vắng mặt, trong khi đó mới là chỗ cần đo.
  2. GIẤU mọi thứ model đã nói: người gán chỉ thấy câu, không thấy nhãn, không thấy lựa chọn
     của Qwen/Llama — thấy rồi thì dễ bị dẫn theo (đúng lý do của Zheng 2023).
  3. TRỘN thứ tự các hàm để người gán không quen tay theo một hàm.

Khoá (câu -> nhãn gốc, nguồn) ghi ra file riêng, chỉ dùng lúc chấm.

    python -m eval.boc_nhan_tay_ask --n 400
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from gen.records import query_from_call_prompt

NGUON = ("data/processed/v2/v1_ask_sach.jsonl", "data/processed/v2/v2_ask_train_vi.jsonl")


def chia_quota(kich_thuoc: dict[str, int], tong: int) -> dict[str, int]:
    """Chia đều `tong` cho các tầng; tầng nào ít hơn phần chia thì lấy hết, phần dư dồn cho
    các tầng còn lại (kiểu đổ nước). Kết quả: hàm nhỏ có mặt đủ, không tầng nào vượt quá kho."""
    quota = dict.fromkeys(kich_thuoc, 0)
    con = dict(kich_thuoc)
    can = min(tong, sum(con.values()))
    while can > 0 and con:
        phan = max(1, can // len(con))
        for k in sorted(con, key=lambda x: con[x]):
            lay = min(phan, con[k], can)
            quota[k] += lay
            con[k] -= lay
            can -= lay
            if can == 0:
                break
        con = {k: v for k, v in con.items() if v > 0}
    return quota


def main() -> None:
    ap = argparse.ArgumentParser(description="Bốc câu hỏi lại cho người gán nhãn")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", type=Path, default=Path("data/nhan_tay"))
    args = ap.parse_args()

    theo_ham: dict[str, list[dict]] = defaultdict(list)
    da_co: set[str] = set()
    for p in NGUON:
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                cau = query_from_call_prompt(r["prompt"])
                if cau in da_co:
                    continue
                da_co.add(cau)
                theo_ham[r["meta"]["fn"]].append(
                    {"cau": cau, "nhan_goc": r["completion"].replace("<|im_end|>", ""), "nguon": Path(p).stem}
                )

    rng = random.Random(args.seed)
    quota = chia_quota({k: len(v) for k, v in theo_ham.items()}, args.n)
    bo = []
    for fn, n in quota.items():
        bo.extend(rng.sample(theo_ham[fn], n))
    rng.shuffle(bo)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # utf-8-sig để Excel mở đúng dấu tiếng Việt
    with open(args.out_dir / "ask_400.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stt", "cau", "loai_cau", "ham_1", "ham_2", "ghi_chu"])
        for i, x in enumerate(bo, 1):
            w.writerow([i, x["cau"], "", "", "", ""])
    with open(args.out_dir / "ask_400_khoa.jsonl", "w", encoding="utf-8") as f:
        for i, x in enumerate(bo, 1):
            f.write(json.dumps({"stt": i, **x}, ensure_ascii=False) + "\n")

    print(f"Đã bốc {len(bo)} câu -> {args.out_dir / 'ask_400.csv'}")
    for fn, n in sorted(quota.items(), key=lambda x: -x[1]):
        print(f"   {fn:20} {n:>3} / {len(theo_ham[fn])}")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
