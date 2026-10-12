"""Chạy một STT trên một bộ âm thanh (manifest.jsonl) — CÙNG điều kiện cho mọi ứng viên (PLAN.md Phase 6
(3)): CPU, 8 luồng, giải mã tham lam (greedy), tiếng Việt, mô hình nén int8. Ghi chữ STT gốc, chữ sau đổi
số, thời gian. Chạy trong env `giong_noi`:

    conda run -n giong_noi python -m eval.stt.chay_stt --stt parakeet --bo data/stt/bo_a \
        --ra runs/stt/bo_a_parakeet.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

from serve.so_chu import doi_so

LUONG = 8
# GỢI Ý TỪ VỰNG cho Whisper (`initial_prompt`) — PLAN.md Phase 6 "cách 1": dựng CHỈ từ thứ đã có TRƯỚC khi
# soi lỗi bộ B / C: danh sách khả năng `serve.tra_loi.KHA_NANG` + các đơn vị của `serve.so_chu`. Không thêm từ
# nhặt từ lỗi đã thấy.
GOI_Y = ("Lệnh cho trợ lý trên xe ô tô: điều hoà, quạt gió, sấy kính, khoá cửa, cốp, kính, cửa sổ trời, đèn, "
         "âm lượng, nhạc, dẫn đường, tìm chỗ gần đây, xem thời tiết và kiểm tra tình trạng xe; "
         "độ, mức, nấc, phần trăm, km.")
MO_HINH = {
    "parakeet": "models/stt/parakeet-ctc-0.6b-vi-onnx",
    "phowhisper-small": "models/stt/phowhisper-small-ct2-int8",
    "phowhisper-medium": "models/stt/phowhisper-medium-ct2-int8",
}


def nap(stt: str, beam: int = 1, goi_y: bool = False) -> Callable[[str], str]:
    if stt == "parakeet":
        if beam != 1 or goi_y:
            raise SystemExit("--beam / --goi-y chỉ áp cho PhoWhisper (Parakeet CTC không có prompt)")
        import onnx_asr
        import onnxruntime as rt

        so = rt.SessionOptions()
        so.intra_op_num_threads = LUONG
        m = onnx_asr.load_model("nemo-conformer-ctc", MO_HINH[stt], quantization="int8", sess_options=so)
        return lambda tep: m.recognize(tep)
    from faster_whisper import WhisperModel

    m = WhisperModel(MO_HINH[stt], device="cpu", compute_type="int8", cpu_threads=LUONG)

    def nghe(tep: str) -> str:
        doan, _ = m.transcribe(tep, language="vi", beam_size=beam, without_timestamps=True,
                               condition_on_previous_text=False, vad_filter=False,
                               initial_prompt=GOI_Y if goi_y else None)
        return " ".join(d.text.strip() for d in doan).strip()

    return nghe


def main() -> None:
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--stt", choices=list(MO_HINH), required=True)
    ap.add_argument("--bo", required=True, help="thư mục có manifest.jsonl + wav")
    ap.add_argument("--ra", required=True)
    ap.add_argument("--beam", type=int, default=1, help="beam search (mặc định 1 = greedy, cách đo đã khoá)")
    ap.add_argument("--goi-y", action="store_true", help="gợi ý từ vựng xe cho Whisper (initial_prompt)")
    a = ap.parse_args()
    bo = Path(a.bo)
    rows = [json.loads(x) for x in (bo / "manifest.jsonl").open(encoding="utf-8")]
    t0 = time.perf_counter()
    nghe = nap(a.stt, a.beam, a.goi_y)
    print(f"nạp {a.stt}: {time.perf_counter() - t0:.1f} s", flush=True)
    nghe(str(bo / rows[0]["tep"]))  # chạy nóng 1 lần, không tính giờ
    Path(a.ra).parent.mkdir(parents=True, exist_ok=True)
    with open(a.ra, "w", encoding="utf-8") as f:
        for k, r in enumerate(rows, 1):
            tep = str(bo / r["tep"])
            t = time.perf_counter()
            chu = nghe(tep)
            ms = (time.perf_counter() - t) * 1000
            f.write(json.dumps({**{x: r[x] for x in ("id", "cau", "prompt", "completion")},
                                "stt": a.stt, "chu_stt": chu, "chu_doi_so": doi_so(chu), "ms": round(ms, 1),
                                "giay_am_thanh": round(sf.info(tep).duration, 2)}, ensure_ascii=False) + "\n")
            if k % 25 == 0:
                print(f"  {k}/{len(rows)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print("ghi", a.ra)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
