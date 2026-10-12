"""Bốc mẫu PHÂN TẦNG từ tập train để đem đi thẩm định chéo — chạy local, miễn phí.

Vì sao cần thẩm định chéo: teacher Qwen 72B vừa SINH data vừa TỰ CHẤM data của chính
nó, nên con số "giữ lại 83%" không thể dùng làm bằng chứng chất lượng — model dễ dãi
với văn phong của chính mình. Jafarzadeh Asl et al. 2026 (§A.7) xử lý bằng cách chấm
bằng model KHÁC HỌ, và quan trọng hơn: chỉ coi kết quả đó là **chẩn đoán**, không tự
động xoá mẫu, vì khi đối chiếu với người thì model chấm cũng sai khá nhiều.

Bốc phân tầng (mỗi hàm × mỗi loại một ít) chứ không bốc ngẫu nhiên toàn cục, để hàm
hiếm và loại hiếm (ask, reply) vẫn có mặt trong mẫu thẩm định.

    python -m eval.boc_mau_tham_dinh --input data/processed/final/train_vi.jsonl \\
                                     --n 500 --output data/processed/tham_dinh/mau.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def boc_phan_tang(rows: list[dict[str, Any]], n: int, seed: int = 42) -> list[dict[str, Any]]:
    """Chia rows theo (hàm, loại) rồi lấy đều mỗi tầng, phần dư rải ngẫu nhiên."""
    rng = random.Random(seed)
    tang: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        tang[(r["meta"]["fn"], r["meta"]["type"])].append(r)

    ket_qua: list[dict[str, Any]] = []
    khoa = sorted(tang)
    moi_tang = max(1, n // max(len(khoa), 1))
    for k in khoa:
        ds = tang[k]
        ket_qua.extend(rng.sample(ds, min(moi_tang, len(ds))))

    # thiếu thì bù ngẫu nhiên từ phần chưa lấy; thừa thì cắt bớt
    if len(ket_qua) < n:
        da_lay = {id(r) for r in ket_qua}
        con_lai = [r for r in rows if id(r) not in da_lay]
        ket_qua.extend(rng.sample(con_lai, min(n - len(ket_qua), len(con_lai))))
    rng.shuffle(ket_qua)
    return ket_qua[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="Bốc mẫu phân tầng để thẩm định chéo")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    mau = boc_phan_tang(rows, args.n, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in mau:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    theo_loai: dict[str, int] = {}
    theo_ham: dict[str, int] = {}
    for r in mau:
        theo_loai[r["meta"]["type"]] = theo_loai.get(r["meta"]["type"], 0) + 1
        theo_ham[r["meta"]["fn"]] = theo_ham.get(r["meta"]["fn"], 0) + 1
    print(f"kho    : {len(rows)} mẫu")
    print(f"bốc ra : {len(mau)} mẫu -> {args.output}")
    print(f"theo loại: {dict(sorted(theo_loai.items()))}")
    print(f"số hàm có mặt: {len(theo_ham)}/25")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
