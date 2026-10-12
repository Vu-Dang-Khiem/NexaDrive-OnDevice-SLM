"""Gộp các kho dữ liệu, chia train/val THEO NHÓM câu gần trùng, rồi cân bằng train.

Vì sao phải chia theo nhóm (bài học từ Jafarzadeh Asl et al. 2026, §A.6):
data negative sinh ra từ cùng một prompt lặp lại hàng trăm lô nên có rất nhiều câu
gần giống nhau:

    "tự lái xe đến nhà tôi"
    "tự lái xe đến nhà tôi đi"
    "bạn có thể tự lái xe đến nhà tôi được không"

Chia ngẫu nhiên thì câu 1 vào train, câu 2 vào val -> model chỉ cần NHỚ câu 1 là trả
lời đúng câu 2, điểm val đẹp giả tạo. Gom chúng vào 1 nhóm và ép cả nhóm nằm cùng một
bên thì điểm val mới phản ánh khả năng xử lý câu MỚI.

Tập test KHÔNG sinh ở đây — test là tập người viết tay (xem PLAN.md Phase 3).

    python -m gen.chia_tap --inputs data/processed/full2/train_vi_sach.jsonl \\
                                    data/processed/ask/ask_train_vi_sach.jsonl \\
                           --out-dir data/processed/final --per-fn 300
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from gen.records import datasheet, query_from_call_prompt
from gen.subsample import subsample

SO_HAM_BAM = 20  # số hàm băm MinHash
SO_HANG = 4  # mỗi dải gồm 4 hàm băm -> 5 dải; cặp giống ~80% mới hay đụng nhau
NGUONG_GIONG = 0.8  # Jaccard tối thiểu để coi là cùng một nhóm

_DAU_CAU = re.compile(r"[^\w\s]", re.UNICODE)


def chuan_hoa(text: str) -> str:
    """Bỏ dấu câu, gộp khoảng trắng, về chữ thường — để so sánh câu."""
    t = unicodedata.normalize("NFC", text).lower()
    return " ".join(_DAU_CAU.sub(" ", t).split())


def _bam(so_thu_tu: int, s: str) -> int:
    """Hàm băm CỐ ĐỊNH giữa các lần chạy.

    Trước đây dùng `hash()` của Python — hàm này bị xáo ngẫu nhiên mỗi tiến trình
    (PYTHONHASHSEED), nên cách gom câu gần trùng đổi theo từng lần chạy: chia train/val
    không tái lập được, và số câu test "sạch" nhảy giữa 228 và 229 tuỳ lần đo. Lộ ra vì
    `test_ca_nhom_gan_trung_nam_cung_mot_ben` lúc đỏ lúc xanh.
    """
    return zlib.crc32(f"{so_thu_tu}\x00{s}".encode())


def _shingle(text: str, k: int = 5) -> set[str]:
    t = chuan_hoa(text)
    if len(t) <= k:
        return {t}
    return {t[i : i + k] for i in range(len(t) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    chung = len(a & b)
    return chung / (len(a) + len(b) - chung)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.cha = list(range(n))

    def tim(self, x: int) -> int:
        while self.cha[x] != x:
            self.cha[x] = self.cha[self.cha[x]]
            x = self.cha[x]
        return x

    def hop(self, a: int, b: int) -> None:
        ra, rb = self.tim(a), self.tim(b)
        if ra != rb:
            self.cha[rb] = ra


def nhom_gan_trung(cau: list[str]) -> list[int]:
    """Gán mỗi câu một mã nhóm; câu gần trùng nhau nhận cùng mã.

    Dùng MinHash + chia dải (LSH) để không phải so từng cặp một: 27k câu mà so đôi một
    là ~370 triệu phép so, quá chậm. LSH chỉ đưa các câu CÓ KHẢ NĂNG giống vào cùng rổ,
    sau đó mới kiểm Jaccard thật trong rổ.
    """
    n = len(cau)
    uf = _UnionFind(n)
    shingles = [_shingle(c) for c in cau]

    # câu chuẩn hoá giống hệt -> gộp ngay
    theo_chuan: dict[str, int] = {}
    for i, c in enumerate(cau):
        k = chuan_hoa(c)
        if k in theo_chuan:
            uf.hop(theo_chuan[k], i)
        else:
            theo_chuan[k] = i

    # Biến thể kiểu thêm đuôi ("... đi", "... nhé") có Jaccard hơi thấp nhưng vẫn là cùng
    # một câu. Sắp xếp chuỗi chuẩn hoá thì các biến thể này nằm CẠNH NHAU (tiền tố giống
    # nhau), nên chỉ cần so hàng xóm là gộp được, rẻ hơn so từng cặp.
    thu_tu = sorted(range(n), key=lambda i: chuan_hoa(cau[i]))
    for x, y in zip(thu_tu, thu_tu[1:], strict=False):
        a, b = chuan_hoa(cau[x]), chuan_hoa(cau[y])
        ngan, dai = (a, b) if len(a) <= len(b) else (b, a)
        if dai.startswith(ngan) and len(ngan) / max(len(dai), 1) >= 0.7:
            uf.hop(x, y)

    chu_ky = [[min(_bam(h, s) for s in sh) if sh else 0 for h in range(SO_HAM_BAM)] for sh in shingles]

    for dai in range(SO_HAM_BAM // SO_HANG):
        ro: dict[tuple, list[int]] = defaultdict(list)
        for i, ck in enumerate(chu_ky):
            khoa = (dai, *ck[dai * SO_HANG : (dai + 1) * SO_HANG])
            ro[khoa].append(i)
        for ds in ro.values():
            if len(ds) < 2 or len(ds) > 200:  # rổ quá to thì bỏ qua, tránh nổ thời gian
                continue
            for idx_a in range(len(ds)):
                for idx_b in range(idx_a + 1, len(ds)):
                    a, b = ds[idx_a], ds[idx_b]
                    if uf.tim(a) != uf.tim(b) and _jaccard(shingles[a], shingles[b]) >= NGUONG_GIONG:
                        uf.hop(a, b)

    return [uf.tim(i) for i in range(n)]


def chia_theo_nhom(
    rows: list[dict[str, Any]], ty_le_val: float = 0.1, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Chia train/val sao cho cả nhóm câu gần trùng nằm cùng một bên.

    Chia theo từng (hàm, loại mẫu) để val có đủ mọi hàm, không bị trống hàm hiếm.
    """
    rng = random.Random(seed)
    ma_nhom = nhom_gan_trung([query_from_call_prompt(r["prompt"]) for r in rows])

    # gom các dòng theo (hàm, loại) -> nhóm
    theo_tang: dict[tuple[str, str], dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r, g in zip(rows, ma_nhom, strict=True):
        theo_tang[(r["meta"]["fn"], r["meta"]["type"])][g].append(r)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for nhom_theo_ma in theo_tang.values():
        cac_nhom = list(nhom_theo_ma.values())
        rng.shuffle(cac_nhom)
        can_val = round(sum(len(x) for x in cac_nhom) * ty_le_val)
        dem = 0
        for nhom in cac_nhom:
            if dem < can_val and len(cac_nhom) > 1:
                val.extend(nhom)
                dem += len(nhom)
            else:
                train.extend(nhom)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser(description="Gộp, chia train/val theo nhóm, cân bằng train")
    ap.add_argument("--inputs", required=True, nargs="+", type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--per-fn", type=int, default=300, help="trần số mẫu mỗi hàm thật khi cân bằng")
    ap.add_argument("--so-ask", type=int, default=None, help="trần số mẫu câu mơ hồ")
    ap.add_argument(
        "--tran-ask-moi-ham", type=int, default=None, help="trần mẫu ask cho MỖI hàm được đề xuất"
    )
    ap.add_argument("--ty-le-val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for p in args.inputs:
        with open(p, encoding="utf-8") as f:
            phan = [json.loads(line) for line in f if line.strip()]
        print(f"đọc {len(phan):>6} mẫu từ {p}")
        rows.extend(phan)

    train_day, val = chia_theo_nhom(rows, ty_le_val=args.ty_le_val, seed=args.seed)
    train = subsample(
        train_day,
        per_fn=args.per_fn,
        seed=args.seed,
        so_ask=args.so_ask,
        tran_ask_moi_ham=args.tran_ask_moi_ham,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ten, phan in (("train_vi", train), ("val_vi", val)):
        p = args.out_dir / f"{ten}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in phan:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n=== {ten}: {len(phan)} mẫu -> {p}")
        print(json.dumps(datasheet(phan), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()

