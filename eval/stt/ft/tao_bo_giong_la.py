"""Bộ G — GIỌNG LẠ (PLAN.md "FINE-TUNE STT", phép kiểm thay thế khi lumi-record-smh chưa được duyệt, 2026-10-13):
60 câu lệnh xe từ tập test đóng băng (STT fine-tune chưa từng thấy: TTS train + bộ FT đã kiểm nhiễm với test), đọc
bằng HAI giọng VieNeu ĐỂ DÀNH (không dùng lúc train: `tao_tts.GIU_LAI`), mỗi giọng 30 câu. Xáo seed 2035, tỉ lệ
bộ B: 30 rõ ràng · 15 mơ hồ · 15 ngoài phạm vi. Bỏ câu có chữ viết tắt Latin / số dạng lạ (như lúc tạo TTS train).

    conda run --no-capture-output -n giong_noi python -m eval.stt.ft.tao_bo_giong_la
Ra: data/stt/bo_g/{manifest.jsonl, g*.wav} theo định dạng eval.stt.chay_stt.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

from eval.stt.ft.tao_tts import GIU_LAI, VIET_TAT, so_sang_chu

RA = Path("data/stt/bo_g")


def main() -> None:
    import soundfile as sf
    import soxr
    from vieneu import Vieneu

    from eval.metrics import doc_output, nhom_cua_dap_an
    from eval.stt.chon_bo_c import NGUON
    from gen.records import query_from_call_prompt

    if (RA / "manifest.jsonl").exists():
        raise SystemExit("bộ G đã có — đã khoá")
    rows = []
    for ten, p in NGUON.items():
        for x in p.open(encoding="utf-8"):
            r = json.loads(x)
            cau = query_from_call_prompt(r["prompt"])
            noi = so_sang_chu(cau)
            if "\n" in cau or VIET_TAT.search(cau) or re.search(r"\d", noi):
                continue
            rows.append({"cau": cau, "noi": noi, "prompt": r["prompt"], "completion": r["completion"],
                         "nguon": ten, "nhom": nhom_cua_dap_an(doc_output(r["completion"]))})
    random.Random(2035).shuffle(rows)
    chon = [r for k, n in (("ro_rang", 30), ("mo_ho", 15), ("ngoai_pham_vi", 15))
            for r in [x for x in rows if x["nhom"] == k][:n]]
    giong = sorted(GIU_LAI)
    tts = Vieneu()
    RA.mkdir(parents=True, exist_ok=True)
    with (RA / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(chon):
            g = giong[i % 2]
            am = tts.infer(r["noi"], voice=g)
            sf.write(str(RA / f"g{i:02d}.wav"), soxr.resample(am, tts.sample_rate, 16000), 16000, subtype="PCM_16")
            f.write(json.dumps({"id": f"g{i:02d}", "tep": f"g{i:02d}.wav", "cau": r["cau"], "giong": g,
                                "nhom": r["nhom"], "prompt": r["prompt"], "completion": r["completion"]},
                               ensure_ascii=False) + "\n")
    print("xong", len(chon), "câu ·", giong, "->", RA)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
