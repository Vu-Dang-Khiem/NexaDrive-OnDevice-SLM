"""Bộ A (giọng máy, so STT sơ bộ — PLAN.md Phase 6 (3)): VieNeu-TTS đọc toàn bộ test sạch 230 câu, mỗi câu
một giọng, chia đều 25 giọng sẵn (xáo seed 2026). Ghi wav 16 kHz mono + manifest. Chạy trong env `giong_noi`:

    conda run -n giong_noi python -m eval.stt.tao_bo_a
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

VAO = Path("data/processed/final/test_vi_sach_v2.jsonl")
RA = Path("data/stt/bo_a")


def main() -> None:
    import soundfile as sf
    import soxr
    from vieneu import Vieneu

    from gen.records import query_from_call_prompt

    rows = [json.loads(x) for x in VAO.open(encoding="utf-8")]
    tts = Vieneu()
    giong = [ten for _, ten in tts.list_preset_voices()]
    thu_tu = list(range(len(rows)))
    random.Random(2026).shuffle(thu_tu)
    gan = {i: giong[k % len(giong)] for k, i in enumerate(thu_tu)}
    RA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with (RA / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            cau = query_from_call_prompt(r["prompt"])
            tep = RA / f"a{i:03d}.wav"
            if not tep.exists():
                a = tts.infer(cau, voice=gan[i])
                a16 = soxr.resample(a, tts.sample_rate, 16000)
                sf.write(str(tep), a16, 16000, subtype="PCM_16")
            dong = {"id": f"a{i:03d}", "tep": tep.name, "cau": cau, "giong": gan[i],
                    "prompt": r["prompt"], "completion": r["completion"]}
            f.write(json.dumps(dong, ensure_ascii=False) + "\n")
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    print("xong", len(rows), "câu ->", RA)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
