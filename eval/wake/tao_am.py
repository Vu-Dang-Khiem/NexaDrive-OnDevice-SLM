"""Wake-word — bước 2: VieNeu đọc các câu ở `data/wake/cau.json` (PLAN.md Phase 6b). Chạy trong env
`giong_noi`:

    python -m eval.wake.tao_am

Chia THEO GIỌNG (khoá trước khi train): 25 giọng xáo seed -> 5 giọng VAL (chọn ngưỡng), 20 giọng TRAIN.
VieNeu lấy mẫu ngẫu nhiên (nhiệt 0,8) nên đọc lại một câu ra bản khác -> mỗi (giọng, cách viết) đọc 3 lần
(train) / 2 lần (val). `tu_thuc` (câu trả lời của trợ lý) đọc bằng giọng loa thật "Trúc Ly" — CHỈ để đo,
không train. Ghi wav 16 kHz mono vào `data/wake/tts/` + `manifest.jsonl` (loai, tach, chu, giong). Chạy
lại thì bỏ qua file đã có.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

CAU = Path("data/wake/cau.json")
RA = Path("data/wake/tts")
SEED = 2026_10_08
GIONG_LOA = "Trúc Ly"


def ke_hoach(cau: dict[str, list[str]], giong: list[str]) -> list[dict[str, str]]:
    xao = list(giong)
    random.Random(SEED).shuffle(xao)
    val, train = xao[:5], xao[5:]
    so = {g: i for i, g in enumerate(giong)}  # tên file dùng số thứ tự giọng (tên giọng có dấu, dấu cách)
    ds: list[dict[str, str]] = []
    for tach, cac_giong, lan in (("train", train, 3), ("val", val, 2)):
        for g in cac_giong:
            for k, chu in enumerate(cau["duong"]):
                ds += [
                    {
                        "loai": "duong",
                        "tach": tach,
                        "chu": chu,
                        "giong": g,
                        "id": f"duong_{tach}_g{so[g]}_{k}_{n}",
                    }
                    for n in range(lan)
                ]
    for k, chu in enumerate(cau["gan_giong"]):
        for n in range(4):
            g = train[(k + n) % len(train)]
            ds.append(
                {"loai": "gan_giong", "tach": "train", "chu": chu, "giong": g, "id": f"gan_train_{k}_{n}"}
            )
        ds.append(
            {"loai": "gan_giong", "tach": "val", "chu": chu, "giong": val[k % len(val)], "id": f"gan_val_{k}"}
        )
    for k, chu in enumerate(cau["lenh"]):
        tach = "val" if k % 5 == 0 else "train"
        g = (val if tach == "val" else train)[k % (5 if tach == "val" else len(train))]
        ds.append({"loai": "lenh", "tach": tach, "chu": chu, "giong": g, "id": f"lenh_{tach}_{k}"})
    ds += [
        {"loai": "tu_thuc", "tach": "test", "chu": chu, "giong": GIONG_LOA, "id": f"tu_thuc_{k}"}
        for k, chu in enumerate(cau["tu_thuc"])
    ]
    return ds


def main() -> None:
    import soundfile as sf
    import soxr
    from vieneu import Vieneu

    tts = Vieneu()
    giong = [ten for _, ten in tts.list_preset_voices()]
    ds = ke_hoach(json.loads(CAU.read_text(encoding="utf-8")), giong)
    RA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with (RA / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for i, d in enumerate(ds, 1):
            tep = RA / f"{d['id']}.wav"
            if not tep.exists():
                a = tts.infer(d["chu"], voice=d["giong"])
                sf.write(str(tep), soxr.resample(a, tts.sample_rate, 16000), 16000, subtype="PCM_16")
            f.write(json.dumps({**d, "tep": tep.name}, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"  {i}/{len(ds)} ({time.time() - t0:.0f}s)", flush=True)
    print("xong", len(ds), "file ->", RA)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
