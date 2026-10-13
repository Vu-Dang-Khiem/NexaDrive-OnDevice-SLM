"""So kích thước TỪNG file trên volume `nexa-drive` với bản trên máy (đường tải Modal từng cắt cụt file
âm thầm — PLAN.md mục S). In file thiếu / lệch; tải lại bằng `modal volume put --force`.

    conda run -n vdk1 python -m eval.stt.ft.kiem_tai_len
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

CAP = [
    ("data/stt/ft_tts", "/data/stt_ft/tts"),
    ("data/stt/ft_len/fleurs", "/data/stt_ft/fleurs"),
    ("data/stt/ft_len/on", "/data/stt_ft/on"),
    ("data/stt/ft_len/that", "/data/stt_ft/that"),
]


def main() -> None:
    vol = modal.Volume.from_name("nexa-drive")
    loi = 0
    for may, xa in CAP:
        goc = Path(may)
        if not goc.exists():
            print(f"(bỏ qua {may}: chưa có trên máy)")
            continue
        tren_may = {p.relative_to(goc).as_posix(): p.stat().st_size for p in goc.rglob("*") if p.is_file()}
        try:
            tren_vol = {
                e.path.removeprefix(xa.lstrip("/") + "/"): e.size
                for e in vol.listdir(xa, recursive=True)
                if e.type == modal.volume.FileEntryType.FILE
            }
        except Exception as e:  # noqa: BLE001 — thư mục chưa có trên volume
            print(f"{xa}: chưa có trên volume ({e})")
            loi += len(tren_may)
            continue
        thieu = [k for k in tren_may if k not in tren_vol]
        lech = [k for k in tren_may if k in tren_vol and tren_vol[k] != tren_may[k]]
        loi += len(thieu) + len(lech)
        print(
            f"{xa}: máy {len(tren_may)} file · volume {len(tren_vol)} · thiếu {len(thieu)} · lệch {len(lech)}"
            + (f" · vd {(thieu + lech)[:3]}" if thieu or lech else "")
        )
    print("==> KHỚP HẾT" if loi == 0 else f"==> CÓ {loi} FILE THIẾU / LỆCH")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
