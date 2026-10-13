"""Tải các checkpoint fine-tune STT (CTranslate2 int8) từ volume về models/stt/, kiểm KÍCH THƯỚC từng file và tải
lại tới khi khớp (đường tải Modal -> máy này hay cắt cụt âm thầm). PLAN.md "FINE-TUNE STT".

    conda run -n vdk1 python -m eval.stt.ft.tai_ckpt A      # -> models/stt/phowhisper-small-ftA/ep{1,2,3}-ct2-int8
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal


def main(ban: str) -> None:
    vol = modal.Volume.from_name("nexa-drive")
    goc = f"models/stt/phowhisper-small-ft{ban}"
    for e in vol.listdir(f"/{goc}"):
        if not e.path.endswith("-ct2-int8"):
            continue
        ra = Path(e.path)
        ra.mkdir(parents=True, exist_ok=True)
        for f in vol.listdir(f"/{e.path}"):
            dich = ra / Path(f.path).name
            if dich.exists() and dich.stat().st_size == f.size:
                continue
            for _ in range(6):
                with dich.open("wb") as g:
                    for khoi in vol.read_file(f.path):
                        g.write(khoi)
                if dich.stat().st_size == f.size:
                    break
            else:
                raise RuntimeError(f"tải {f.path} không đủ sau 6 lần")
        print("ok", ra, sum(p.stat().st_size for p in ra.iterdir()) // 2**20, "MB")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1])
