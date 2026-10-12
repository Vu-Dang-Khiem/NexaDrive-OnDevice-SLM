"""Gom dữ liệu fine-tune STT vào data/stt/ft_len/ đúng cấu trúc train/modal_stt_ft.py đọc, rồi tải lên volume
(PLAN.md "FINE-TUNE STT"). Nhãn kiểu PhoWhisper viết: chữ thường, số viết bằng chữ, có dấu câu.

  fleurs/  FLEURS tiếng Việt phần TRAIN (bỏ câu có chữ số — FLEURS ghi số bằng chữ số, khác PhoWhisper)
  on/      4 kênh ồn xe DEMAND TCAR (16 kHz)
  that/    bộ FT giọng thật (data/stt/bo_ft/khiem) — nhãn = so_sang_chu(câu)
  (tts/ dùng thẳng data/stt/ft_tts — không chép)

    conda run -n vdk1 python -m eval.stt.ft.chuan_bi
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

from eval.stt.ft.tao_tts import so_sang_chu

RA = Path("data/stt/ft_len")


def fleurs() -> int:
    d = RA / "fleurs"
    (d / "wav").mkdir(parents=True, exist_ok=True)
    nhan = {}
    with open("data/wake/fleurs/train.tsv", encoding="utf-8") as f:
        for dong in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(dong) > 2 and not re.search(r"\d", dong[2]):
                nhan[dong[1]] = dong[2].strip().lower()
    n = 0
    with (
        tarfile.open("data/wake/fleurs/train.tar.gz") as t,
        (d / "manifest.jsonl").open("w", encoding="utf-8") as f,
    ):
        for m in t:
            ten = Path(m.name).name
            if not m.isfile() or ten not in nhan:
                continue
            (d / "wav" / ten).write_bytes(t.extractfile(m).read())
            f.write(json.dumps({"tep": f"wav/{ten}", "nhan": nhan[ten]}, ensure_ascii=False) + "\n")
            n += 1
    return n


def on() -> int:
    d = RA / "on"
    d.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile("data/stt/on_nen/demand/TCAR_16k.zip") as z:
        for k in ("01", "05", "09", "13"):
            (d / f"tcar_ch{k}.wav").write_bytes(z.read(f"TCAR/ch{k}.wav"))
    return 4


def that() -> int:
    vao = Path("data/stt/bo_ft/khiem")
    if not (vao / "manifest.jsonl").exists():
        return 0
    d = RA / "that"
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    with (d / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for x in (vao / "manifest.jsonl").open(encoding="utf-8"):
            r = json.loads(x)
            shutil.copy(vao / r["tep"], d / r["tep"])
            f.write(
                json.dumps(
                    {"tep": r["tep"], "nhan": so_sang_chu(r["cau"]).lower().rstrip(" .!?") + "."},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    return n


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("fleurs", fleurs(), "· on", on(), "· that", that())
