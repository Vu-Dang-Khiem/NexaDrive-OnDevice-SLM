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
    # hynt/Zipformer-30M-RNNT-6000h — CC BY-NC-ND 4.0 (phi thương mại, không phân phối bản sửa)
    "zipformer": "models/stt/zipformer-30m-rnnt-6000h",
}
# Từ khoá hotword của Zipformer (--goi-y): CHỈ lấy từ GOI_Y ở trên (có trước khi soi lỗi)
TU_KHOA = [t.strip() for t in
           GOI_Y.split(":", 1)[1].replace(";", ",").replace(" và ", ", ").rstrip(".").split(",")]


def _thay(s: str, bang: dict[str, str]) -> str:
    for a, b in bang.items():
        s = s.replace(a, b)
    return s


def _nap_zipformer(beam: int, goi_y: bool) -> Callable[[str], str]:
    import sentencepiece as spm
    import sherpa_onnx
    import soundfile as sf

    d = Path(MO_HINH["zipformer"])
    sp = spm.SentencePieceProcessor(model_file=str(d / "bpe.model"))
    so = range(sp.get_piece_size())
    if not (d / "tokens.txt").exists():  # repo không kèm tokens.txt -> dựng từ bpe.model (id -> mảnh)
        (d / "tokens.txt").write_text("".join(f"{sp.id_to_piece(i)} {i}\n" for i in so), encoding="utf-8")
    them: dict = {}
    if goi_y:  # hotword cần beam search + bảng từ BPE (mảnh \t điểm)
        (d / "bpe.vocab").write_text("".join(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n" for i in so),
                                     encoding="utf-8")
        # model viết dấu kiểu cũ ("hòa", "khóa") -> thêm cả cách viết đó (chỉ đổi chính tả)
        cu = {"oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa", "oè": "òe", "oé": "óe",
              "uỳ": "ùy", "uý": "úy"}
        tu = list(dict.fromkeys(x for t in TU_KHOA if t for x in (t.lower(), _thay(t.lower(), cu))))
        (d / "tu_khoa.txt").write_text("".join(f"{t}\n" for t in tu), encoding="utf-8")
        them = {"hotwords_file": str(d / "tu_khoa.txt"), "hotwords_score": 1.5, "modeling_unit": "bpe",
                "bpe_vocab": str(d / "bpe.vocab")}
    nhan = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(d / "encoder-epoch-20-avg-10.int8.onnx"),
        decoder=str(d / "decoder-epoch-20-avg-10.int8.onnx"),
        joiner=str(d / "joiner-epoch-20-avg-10.int8.onnx"), tokens=str(d / "tokens.txt"), num_threads=LUONG,
        decoding_method="modified_beam_search" if (beam > 1 or goi_y) else "greedy_search",
        max_active_paths=max(beam, 4), **them)

    def nghe(tep: str) -> str:
        am, sr = sf.read(tep, dtype="float32")
        s = nhan.create_stream()
        s.accept_waveform(sr, am)
        nhan.decode_stream(s)
        return s.result.text.strip().lower()

    return nghe


def nap(stt: str, beam: int = 1, goi_y: bool = False) -> Callable[[str], str]:
    if stt == "zipformer":
        return _nap_zipformer(beam, goi_y)
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
