"""Ghi âm bộ B (giọng THẬT — PLAN.md Phase 6 (3)): hiện lần lượt 40 câu đã khoá, người đọc nói vào micro
laptop. Lưu wav 16 kHz mono vào `data/stt/bo_b/<nguoi>/` + manifest. Câu đã ghi thì bỏ qua (chạy lại để ghi
tiếp).

    conda run -n giong_noi python -m eval.stt.ghi_am --nguoi khiem
    python -m eval.stt.ghi_am --nguoi khiem --nghe [--tu b20]   # nghe lại cả bộ, r = ghi lại câu hỏng
    python -m eval.stt.ghi_am --nguoi khiem --bo bo_c            # bộ C (câu mới, chon_bo_c.py)

Mỗi câu: Enter = bắt đầu nói · Enter lần nữa = dừng · rồi Enter = câu tiếp · r = ghi lại · n = nghe lại ·
q = thoát.
Đọc TỰ NHIÊN như đang nói với trợ lý trên xe (không cần đọc chậm / rõ từng chữ); đọc vấp thì bấm r.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
from pathlib import Path

SR = 16000


def ghi() -> object:
    import numpy as np
    import sounddevice as sd

    q: queue.Queue = queue.Queue()
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=lambda d, *_: q.put(d.copy())):
        input("   🎙  ĐANG GHI... nói xong bấm Enter")
    khoi = []
    while not q.empty():
        khoi.append(q.get())
    return np.concatenate(khoi)[:, 0] if khoi else np.zeros(0, dtype="float32")


def nghe_lai(ra: Path, rows: list[dict], tu: str | None) -> None:
    """Nghe lại các câu ĐÃ ghi (kèm chữ), ghi lại câu nào hỏng. Nghe xong thì quay về ghi câu còn thiếu."""
    import sounddevice as sd
    import soundfile as sf

    bat_dau = next((k for k, r in enumerate(rows) if r["tep"].startswith(tu)), 0) if tu else 0
    for k, r in enumerate(rows[bat_dau:], bat_dau + 1):
        tep = ra / r["tep"]
        if not tep.exists():
            continue
        am, sr = sf.read(str(tep), dtype="float32")
        while True:
            print(f"[{k}/{len(rows)}] {tep.stem}  « {r['cau']} »  ({len(am) / sr:.1f} s)")
            sd.play(am, sr)
            sd.wait()
            lenh = input("   Enter = câu tiếp · n = nghe lại · r = ghi lại · q = thoát: ").strip().lower()
            if lenh == "n":
                continue
            if lenh == "q":
                return
            if lenh == "r":
                moi = ghi()
                if len(moi) < SR * 0.3:
                    print("   quá ngắn — giữ bản cũ")
                    continue
                sf.write(str(tep), moi, SR, subtype="PCM_16")
                am, sr = moi, SR
                print("   đã thay bản ghi mới — nghe lại:")
                continue
            break


def main() -> None:
    import sounddevice as sd
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--nguoi", required=True, help="tên người đọc (không dấu), vd khiem")
    ap.add_argument("--nghe", action="store_true", help="nghe lại các câu đã ghi (r = ghi lại câu hỏng)")
    ap.add_argument("--tu", help="với --nghe: bắt đầu từ câu này, vd b20")
    ap.add_argument("--bo", default="bo_b", help="bộ câu trong data/stt/: bo_b (mặc định) hoặc bo_c")
    a = ap.parse_args()
    ra = Path("data/stt") / a.bo / a.nguoi
    ra.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (Path("data/stt") / a.bo / "danh_sach.jsonl").open(encoding="utf-8")]
    if a.nghe:
        nghe_lai(ra, rows, a.tu)
        print("\nNghe xong.")
    print(f"Người đọc: {a.nguoi} · {len(rows)} câu · lưu ở {ra}\n")
    for k, r in enumerate(rows, 1):
        tep = ra / r["tep"]
        if tep.exists():
            continue
        while True:
            print(f"[{k}/{len(rows)}]  « {r['cau']} »")
            lenh = input("   Enter = bắt đầu nói · q = thoát: ").strip().lower()
            if lenh == "q":
                print("Đã dừng — chạy lại lệnh để ghi tiếp từ câu chưa ghi.")
                return
            am = ghi()
            print(f"   ghi được {len(am) / SR:.1f} s")
            while True:
                tiep = input("   Enter = lưu, sang câu tiếp · r = ghi lại · n = nghe lại: ").strip().lower()
                if tiep == "n":
                    sd.play(am, SR)
                    sd.wait()
                    continue
                break
            if tiep == "r" or len(am) < SR * 0.3:
                if len(am) < SR * 0.3:
                    print("   quá ngắn — ghi lại")
                continue
            sf.write(str(tep), am, SR, subtype="PCM_16")
            break
    with (ra / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            if (ra / r["tep"]).exists():
                f.write(json.dumps({**r, "nguoi": a.nguoi}, ensure_ascii=False) + "\n")
    print(f"\nXong. Đã ghi {sum((ra / r['tep']).exists() for r in rows)}/{len(rows)} câu.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
