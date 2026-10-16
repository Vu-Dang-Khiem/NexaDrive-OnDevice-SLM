"""ĐO TTS — tốc độ, đo riêng khi máy rảnh (lần đọc 163 câu cho RTF ~1,0, chậm gấp đôi lần đo 2026-10-05).
Mỗi câu (lấy đều 20 câu trong bộ câu trả lời) đo hai cách như demo dùng: `infer` cả câu (RTF) và
`infer_stream` (thời gian tới đoạn tiếng đầu, tổng thời gian tạo). Chạy trong env `giong_noi`:

    conda run --no-capture-output -n giong_noi python -m eval.tts.toc_do
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from serve.giong_noi import GIONG

VAO = Path("runs/tts/cau_tra_loi.jsonl")
RA = Path("runs/tts/toc_do.txt")


def main() -> None:
    from vieneu import Vieneu

    rows = [json.loads(x) for x in VAO.open(encoding="utf-8")]
    rows = rows[:: max(1, len(rows) // 20)][:20]
    tts = Vieneu()
    for cau in ("Đã bật điều hoà 22 độ.", "Xăng còn 62%."):  # chạy nóng
        tts.infer(cau, voice=GIONG)
    rtf, dau, rtf_dong = [], [], []
    for r in rows:
        t = time.perf_counter()
        am = np.asarray(tts.infer(r["text"], voice=GIONG)).reshape(-1)
        giay = len(am) / tts.sample_rate
        rtf.append((time.perf_counter() - t) / giay)
        t = time.perf_counter()
        dau_tien, mau = None, 0
        for doan in tts.infer_stream(r["text"], voice=GIONG):
            if dau_tien is None:
                dau_tien = time.perf_counter() - t
            mau += np.asarray(doan).reshape(-1).size
        dau.append(dau_tien * 1000)
        rtf_dong.append((time.perf_counter() - t) / (mau / tts.sample_rate))
    ra = [
        f"Tốc độ VieNeu (giọng {GIONG}, CPU), {len(rows)} câu, máy rảnh:",
        f"  cả câu (infer):       RTF p50 {statistics.median(rtf):.2f} · lớn nhất {max(rtf):.2f}",
        f"  từng đoạn (stream):   tiếng đầu p50 {statistics.median(dau):.0f} ms · lớn nhất "
        f"{max(dau):.0f} ms · RTF p50 {statistics.median(rtf_dong):.2f} · lớn nhất {max(rtf_dong):.2f}",
    ]
    RA.write_text("\n".join(ra) + "\n", encoding="utf-8")
    print("\n".join(ra))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
