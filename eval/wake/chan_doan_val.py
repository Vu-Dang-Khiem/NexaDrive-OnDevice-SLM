"""Wake-word — CHẨN ĐOÁN trên VAL (được phép: val dùng để phát triển; FLEURS test + bản ghi người dùng không
đụng). Chấm TỪNG đoạn âm của val riêng lẻ (đệm 1,2 s im trước, 0,5 s sau), in các đoạn có điểm cao nhất
theo loại.

    python -m eval.wake.chan_doan_val
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict

import numpy as np

from serve.danh_thuc import DacTrung, diem, nap_mo_hinh
from train.danh_thuc import BO_DAU, FLEURS_DEV, SEED, SR, TTS, doc_fleurs, doc_wav, sang_int16


def main() -> None:
    goi = nap_mo_hinh()
    mo_hinh, nguong = goi["mo_hinh"], goi["nguong"]
    dt = DacTrung()
    man = [json.loads(x) for x in open(TTS / "manifest.jsonl", encoding="utf-8")]
    doan = [
        (d["loai"], d["chu"], doc_wav(TTS / d["tep"]))
        for d in man
        if d["tach"] == "val" and d["loai"] in ("gan_giong", "lenh")
    ]
    fl = doc_fleurs(FLEURS_DEV)
    ten = sorted(fl)
    random.Random(SEED).shuffle(ten)
    tsv = {}
    for dong in open("data/wake/fleurs/dev.tsv", encoding="utf-8"):
        cot = dong.rstrip("\n").split("\t")
        if len(cot) > 2:
            tsv[cot[1]] = cot[2]
    doan += [("fleurs", tsv.get(t, t), fl[t]) for t in ten[: int(0.3 * len(ten))]]
    theo_loai: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for loai, chu, a in doan:
        x = np.concatenate([np.zeros(int(1.2 * SR), np.float32), a, np.zeros(int(0.5 * SR), np.float32)])
        theo_loai[loai].append((float(diem(mo_hinh, dt.chuoi(sang_int16(x))[BO_DAU:]).max()), chu))
    print(f"ngưỡng đang lưu {nguong}")
    for loai, ds in theo_loai.items():
        ds.sort(reverse=True)
        cao = sum(d >= 0.9 for d, _ in ds)
        print(f"\n== {loai}: {len(ds)} đoạn, {cao} đoạn có điểm ≥ 0,9")
        for d, chu in ds[:12]:
            print(f"   {d:.3f}  {chu[:90]}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
