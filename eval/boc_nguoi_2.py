"""Bốc ~100 câu trong 400 câu đã gán tay để NGƯỜI THỨ HAI gán độc lập -> mốc người↔người.

Vì sao cần: bài toán hỏi lại vốn mơ hồ (Qwen và Llama tự chọn chỉ trùng nhau 81%). Nếu chỉ
có nhãn của một người thì khi thấy "giám khảo khớp người 80%", không biết 20% lệch là do giám
khảo sai hay do câu vốn ai trả lời cũng khác. Zheng et al. 2023 giải bằng cách đo cả mức
người đồng ý với người (81%) rồi so giám khảo với mốc đó (85%). Xem PLAN.md mục E0b.

Bốc ĐỀU theo hàm của nhãn gốc (như file 400 câu) để hàm nào cũng có mặt. File ra KHÔNG có
nhãn của người thứ nhất — người thứ hai phải gán mà không nhìn thấy.

    python -m eval.boc_nguoi_2 --n 100
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from schema.loader import CATALOG

KHOA = Path("data/nhan_tay/ask_400_khoa.jsonl")
RA = Path("data/nguoi_2/2_gan_nhan_100_cau.csv")


def ham_chinh(nhan: str) -> str:
    m = re.search(r"<nexa_(\d+)>", nhan)
    ten = {f.index: f.name for f in CATALOG.functions}
    return ten[int(m.group(1))] if m else "?"


def boc(n: int, seed: int = 11) -> list[dict]:
    rows = [json.loads(x) for x in KHOA.read_text(encoding="utf-8").splitlines() if x.strip()]
    tang: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        tang[ham_chinh(r["nhan_goc"])].append(r)
    rng = random.Random(seed)
    for ds in tang.values():
        rng.shuffle(ds)
    # chia lượt vòng tròn giữa các tầng cho tới đủ n -> tầng nhỏ vẫn có mặt
    ra: list[dict] = []
    i = 0
    while len(ra) < n and any(i < len(ds) for ds in tang.values()):
        for k in sorted(tang):
            if i < len(tang[k]) and len(ra) < n:
                ra.append(tang[k][i])
        i += 1
    rng.shuffle(ra)
    return ra


def main() -> None:
    ap = argparse.ArgumentParser(description="Bốc câu cho người gán thứ hai")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    ra = boc(args.n)
    RA.parent.mkdir(parents=True, exist_ok=True)
    with open(RA, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stt", "cau", "loai_cau", "ham_1", "ham_2", "ghi_chu"])
        for r in ra:
            w.writerow([r["stt"], r["cau"], "", "", "", ""])
    dem: dict[str, int] = defaultdict(int)
    for r in ra:
        dem[ham_chinh(r["nhan_goc"])] += 1
    print(f"Đã bốc {len(ra)} câu -> {RA}")
    print("   theo hàm (nhãn gốc):", dict(sorted(dem.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
