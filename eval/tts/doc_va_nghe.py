"""ĐO TTS — bước 2: VieNeu đọc từng câu trả lời (giọng "Trúc Ly", ĐÚNG như demo:
`tts.infer(text, voice=GIONG)`, chữ gửi nguyên văn, không đổi số trước), đổi 48 kHz -> 16 kHz (như âm thanh
STT nhận), rồi hai STT nghe lại.

Giám khảo CHÍNH = PhoWhisper-small GỐC: CHƯA từng nghe VieNeu (bản fine-tune đã học 23 giọng VieNeu, có
"Trúc Ly" -> chấm bằng nó sẽ đẹp giả). Giám khảo phụ = Zipformer-30M (kiến trúc khác). Kết quả chỉ để MÔ TẢ,
không chọn gì theo số này. Chạy trong env `giong_noi`:

    conda run --no-capture-output -n giong_noi python -m eval.tts.doc_va_nghe
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from eval.stt.chay_stt import nap
from serve.giong_noi import GIONG

VAO = Path("runs/tts/cau_tra_loi.jsonl")
WAV = Path("runs/tts/wav")
RA = Path("runs/tts/nghe_lai.jsonl")


def main() -> None:
    from vieneu import Vieneu

    rows = [json.loads(x) for x in VAO.open(encoding="utf-8")]
    tts = Vieneu()
    for cau in ("Đã bật điều hoà 22 độ.", "Xăng còn 62%."):  # chạy nóng (câu đầu sau khởi động tạo chậm)
        tts.infer(cau, voice=GIONG)
    WAV.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for k, r in enumerate(rows, 1):
        t = time.perf_counter()
        am = np.asarray(tts.infer(r["text"], voice=GIONG), dtype=np.float32).reshape(-1)
        r["tao_ms"] = round((time.perf_counter() - t) * 1000, 1)
        r["giay"] = round(len(am) / tts.sample_rate, 2)
        assert tts.sample_rate == 48000, tts.sample_rate
        sf.write(WAV / f"{r['id']}.wav", resample_poly(am, 1, 3).astype(np.float32), 16000)
        if k % 40 == 0:
            print(f"  TTS {k}/{len(rows)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    for ten in ("phowhisper-small", "zipformer"):
        nghe = nap(ten)
        nghe(str(WAV / f"{rows[0]['id']}.wav"))  # chạy nóng
        for r in rows:
            r[f"chu_{ten}"] = nghe(str(WAV / f"{r['id']}.wav"))
        print(f"  {ten} xong ({time.perf_counter() - t0:.0f}s)", flush=True)
    with RA.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("ghi", RA)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
