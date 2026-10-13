"""Dịch vụ GIỌNG NÓI (PLAN.md Phase 6) — chạy riêng trong env `giong_noi` (STT/TTS không cài vào `vdk1`):

    conda run -n giong_noi uvicorn serve.giong_noi:tao_app_that --factory --port 8001

  POST /stt   thân = file âm thanh (webm/opus từ MediaRecorder của trình duyệt, hoặc wav)
              -> {"text": chữ đã chuẩn hoá đưa cho trợ lý, "chu_stt": chữ STT gốc, "ms": thời gian nhận dạng}
  POST /tts   {"text": "..."} -> audio/wav (giọng VieNeu "Trúc Ly", người dùng chốt 2026-10-05)
  POST /tts_stream  {"text": "..."} -> âm thanh THÔ PCM 16-bit mono, gửi TỪNG ĐOẠN ngay khi VieNeu
              tạo xong (`infer_stream`); tần số lấy mẫu ở header `X-Tan-So`. Giao diện phát đoạn đầu trong
              khi đoạn sau còn đang tạo -> nghe tiếng sau ~0,2–0,3 s thay vì chờ cả câu (~1 s). Câu mới tới
              thì câu cũ dừng tạo.

  WS   /wake  wake-word "hey nexa" (Phase 6b): trình duyệt gửi liên tục khung âm thanh int16 16 kHz
              (nhị phân) + tin {"type": "tam_dung" | "tiep_tuc" | "nghe_tiep"} (tạm dừng khi loa đang đọc /
              đang bấm mic; "nghe_tiep" = sau khi đọc xong câu trả lời, nghe câu lệnh TIẾP THEO ~6 s mà KHÔNG
              cần gọi lại "hey nexa"). Server gửi {"type": "thuc"} khi nghe thấy, {"type": "lenh", "text",
              "chu_stt", "ms", "tiep"} khi đã gom + nhận dạng xong câu lệnh, {"type": "chi_goi"} khi chỉ gọi
              "hey nexa" mà không kèm lệnh (giao diện chào rồi gửi "nghe_tiep"), {"type": "het_tiep"} khi nghe
              tiếp mà không ai nói (quay về chờ "hey nexa").

STT = PhoWhisper-small int8 (chọn theo luật đã khoá trên bộ B giọng thật), CÙNG cách gọi lúc đo
(`eval/stt/chay_stt.nap`: CPU 8 luồng, int8). Mặc định dùng bản FINE-TUNE trên câu lệnh xe (`STT_FT`) nếu có
trên máy; NEXA_STT=goc để về bản gốc. Beam / gợi ý từ vựng để mặc định = cấu hình đã đo (greedy,
không gợi ý); đổi qua NEXA_STT_BEAM, NEXA_STT_GOI_Y=1 khi bộ D chốt biến thể khác. Chữ STT -> `chuan_cau_stt`
(đổi số chữ -> chữ số, bỏ dấu câu cuối, hoa chữ đầu) — đúng bước đã dùng khi chấm cả chuỗi. Đường
wake-word cắt thêm đuôi cụm gọi ("nét xa ơi…") ở đầu chữ trước (`serve.danh_thuc.bo_cum_goi`).

Backend agent (serve/app.py) KHÔNG đổi: giao diện gửi chữ nhận được qua WebSocket như câu gõ tay
(wake-word thì kèm `nguon="wake_word"` để luật an toàn biết lệnh có thể không phải của tài xế).
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from serve.danh_thuc import KHUNG, ThuLenh, bo_cum_goi, bo_dem_truoc
from serve.so_chu import chuan_cau_stt

GIONG = "Trúc Ly"
TOI_DA_BYTE = 10_000_000  # ~ vài phút âm thanh nén — lệnh trên xe chỉ vài giây
CHO_SAU_GOI = 25  # thức rồi mà 2 s không ai nói -> coi là CHỈ GỌI "hey nexa" (chào, rồi nghe tiếp)
# Nghe câu lệnh tiếp theo không cần gọi lại: chờ người nói tối đa 10 s (trước 6 s — người dùng thử 2026-10-13
# thấy ngắn, chưa kịp nghĩ lệnh kế tiếp đã phải gọi lại; trợ lý thật để ~5–8 s).
CHO_NGHE_TIEP = 125
# Tạm dừng (loa đang đọc) tối đa 30 s rồi TỰ nghe lại: giao diện lỡ không gửi "tiep_tuc" / "nghe_tiep" (tab
# treo, lỗi phát tiếng…) thì trợ lý không được "điếc" mãi. Câu trả lời dài nhất đọc ~10 s.
TAM_DUNG_TOI_DA = 16000 * 30


class YeuCauDoc(BaseModel):
    text: str


def nghe_pcm(nghe: Callable[[str], str], am: np.ndarray) -> tuple[str, int]:
    """STT cho âm thanh int16 16 kHz trong bộ nhớ (ghi wav tạm vì `nghe` nhận đường dẫn file)."""
    import wave

    with tempfile.TemporaryDirectory() as d:
        tep = Path(d) / "lenh.wav"
        with wave.open(str(tep), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(am.astype("<i2").tobytes())
        t0 = time.perf_counter()
        chu = nghe(str(tep)).strip()
        return chu, round((time.perf_counter() - t0) * 1000)


def luu_lan_nghe(thu_muc: Path, duoi: str, am: bytes, chu: str, nguon: str) -> None:
    """CHẨN ĐOÁN (bật bằng NEXA_LUU_GIONG=<thư mục>): lưu âm thanh dịch vụ NHẬN ĐƯỢC + chữ STT ra, để so đường
    demo (trình duyệt) với đường đã đo (ghi âm thô). Chỉ ghi trên máy."""
    thu_muc.mkdir(parents=True, exist_ok=True)
    ten = time.strftime("%Y%m%d_%H%M%S") + f"_{nguon}{duoi}"
    (thu_muc / ten).write_bytes(am)
    with (thu_muc / "nhat_ky.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"tep": ten, "nguon": nguon, "chu_stt": chu}, ensure_ascii=False) + "\n")


def pcm16(am: np.ndarray) -> bytes:
    """Âm thanh float [-1, 1] -> PCM 16-bit little-endian (cắt phần vượt biên thay vì để tràn số)."""
    return (np.clip(am, -1.0, 1.0) * 32767).astype("<i2").tobytes()


class DocTungDoan:
    """Đọc từng đoạn: `sinh(chữ)` trả các mảng âm thanh float theo thứ tự tạo (VieNeu `infer_stream`).

    Mỗi lần gọi -> (tần số lấy mẫu, bộ lặp bytes PCM). Model chỉ tạo MỘT câu một lúc (khoá); có câu mới
    thì câu đang tạo dở dừng ở đoạn kế tiếp — người lái đã sang câu khác, đọc tiếp câu cũ chỉ làm câu mới
    phải chờ."""

    def __init__(self, sinh: Callable[[str], Iterable[np.ndarray]], tan_so: int) -> None:
        self.sinh, self.tan_so = sinh, tan_so
        self._khoa = threading.Lock()
        self._moi_nhat = 0

    def __call__(self, text: str) -> tuple[int, Iterator[bytes]]:
        self._moi_nhat += 1
        return self.tan_so, self._dong(text, self._moi_nhat)

    def _dong(self, text: str, so: int) -> Iterator[bytes]:
        with self._khoa:
            for am in self.sinh(text):
                if so != self._moi_nhat:
                    return
                if len(am):
                    yield pcm16(am)


def _wav(am: np.ndarray) -> bytes:
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(am.astype("<i2").tobytes())
    return buf.getvalue()


def tao_app(nghe: Callable[[str], str], doc: Callable[[str], bytes],
            tao_danh_thuc: Callable[[], tuple[Any, Any]] | None = None,
            luu: Path | None = None,
            doc_dong: Callable[[str], tuple[int, Iterator[bytes]]] | None = None) -> FastAPI:
    """`nghe(đường dẫn file) -> chữ`, `doc(chữ) -> bytes wav`, `tao_danh_thuc() -> (bộ báo thức, VAD)`
    — tách ra để test không cần model thật. Không có `tao_danh_thuc` thì không có /wake. `luu`: thư mục lưu
    âm thanh + chữ mỗi lần nhận dạng (chẩn đoán). `doc_dong(chữ) -> (tần số, các đoạn PCM)` cho /tts_stream
    (không có thì /tts_stream báo 404, giao diện dùng /tts)."""
    app = FastAPI(title="Nexa Drive — giọng nói")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Tan-So"],  # trình duyệt chỉ đọc được header lạ khi được phép
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/stt")
    async def stt(req: Request) -> dict[str, object]:
        am = await req.body()
        if not am:
            raise HTTPException(400, "không có âm thanh")
        if len(am) > TOI_DA_BYTE:
            raise HTTPException(413, "âm thanh quá dài")
        duoi = ".webm" if "webm" in req.headers.get("content-type", "") else ".wav"
        with tempfile.TemporaryDirectory() as d:
            tep = Path(d) / f"lenh{duoi}"
            tep.write_bytes(am)
            t0 = time.perf_counter()
            chu = nghe(str(tep)).strip()
            ms = round((time.perf_counter() - t0) * 1000)
        if luu is not None:
            luu_lan_nghe(luu, duoi, am, chu, "mic")
        return {"text": chuan_cau_stt(chu) if chu else "", "chu_stt": chu, "ms": ms}

    @app.post("/tts")
    async def tts(yc: YeuCauDoc) -> Response:
        if not yc.text.strip():
            raise HTTPException(400, "không có chữ để đọc")
        return Response(content=doc(yc.text), media_type="audio/wav")

    @app.post("/tts_stream")
    async def tts_stream(yc: YeuCauDoc) -> StreamingResponse:
        if doc_dong is None:
            raise HTTPException(404, "chưa bật đọc từng đoạn")
        if not yc.text.strip():
            raise HTTPException(400, "không có chữ để đọc")
        tan_so, dong = doc_dong(yc.text)
        # bộ lặp thường (không async) -> Starlette chạy ở luồng phụ, gửi từng đoạn ngay khi có
        return StreamingResponse(dong, media_type="application/octet-stream",
                                 headers={"X-Tan-So": str(tan_so)})

    if tao_danh_thuc is not None:

        @app.websocket("/wake")
        async def wake(ws: WebSocket) -> None:
            await ws.accept()
            bo, vad = tao_danh_thuc()  # mỗi kết nối một bộ đệm riêng
            du = np.zeros(0, np.int16)
            truoc = bo_dem_truoc()
            thu: ThuLenh | None = None
            tiep = False  # đang NGHE TIẾP (không cần gọi lại) chứ không phải vừa nghe thấy "hey nexa"
            tam_dung = False
            da_dung = 0  # số mẫu âm thanh đã bỏ qua trong lần tạm dừng này
            try:
                while True:
                    tin = await ws.receive()
                    if tin.get("type") == "websocket.disconnect":
                        break
                    if tin.get("text"):
                        loai = json.loads(tin["text"]).get("type")
                        if loai == "nghe_tiep":  # đọc xong câu trả lời -> nghe lệnh kế tiếp, không cần gọi
                            tam_dung, da_dung, du = False, 0, np.zeros(0, np.int16)
                            truoc.clear()
                            bo.lam_moi()
                            thu, tiep = ThuLenh([], vad, cho_noi_khung=CHO_NGHE_TIEP), True
                            continue
                        # chỉ xử lý khi ĐỔI trạng thái — "tiep_tuc" thừa (vd dừng loa đúng lúc vừa thức) không
                        # được xoá câu lệnh đang gom
                        if (loai == "tam_dung") != tam_dung and loai in ("tam_dung", "tiep_tuc"):
                            tam_dung, da_dung = loai == "tam_dung", 0
                            thu, tiep, du = None, False, np.zeros(0, np.int16)
                            truoc.clear()
                            bo.lam_moi()
                        continue
                    if not tin.get("bytes"):
                        continue
                    if tam_dung:
                        da_dung += len(tin["bytes"]) // 2
                        if da_dung < TAM_DUNG_TOI_DA:
                            continue
                        tam_dung, da_dung = False, 0  # quá lâu không được báo nghe lại -> tự nghe lại
                        bo.lam_moi()
                        continue
                    du = np.concatenate([du, np.frombuffer(tin["bytes"], dtype="<i2")])
                    while len(du) >= KHUNG:
                        khung, du = du[:KHUNG].copy(), du[KHUNG:]
                        if thu is None:
                            _, thuc = bo.nghe(khung)
                            truoc.append(khung)
                            if thuc:
                                thu, tiep = ThuLenh(list(truoc), vad, cho_noi_khung=CHO_SAU_GOI), False
                                await ws.send_json({"type": "thuc"})
                            continue
                        ket = thu.them(khung)
                        if ket is None:
                            continue
                        am, thu, la_tiep = thu.am_thanh(), None, tiep
                        truoc.clear()
                        if ket == "huy":
                            await ws.send_json({"type": "het_tiep" if la_tiep else "chi_goi"})
                        else:
                            chu, ms = await run_in_threadpool(nghe_pcm, nghe, am)
                            if luu is not None:
                                luu_lan_nghe(luu, ".wav", _wav(am), chu, "wake")
                            con = bo_cum_goi(chu)
                            # vừa gọi mà sau cụm gọi không còn lệnh (chỉ "hey nexa", đuôi "xa") -> chào
                            if not la_tiep and len(con.split()) < 2:
                                await ws.send_json({"type": "chi_goi", "chu_stt": chu})
                            elif la_tiep and not con:
                                await ws.send_json({"type": "het_tiep"})
                            else:
                                await ws.send_json({"type": "lenh", "text": chuan_cau_stt(con),
                                                    "chu_stt": chu, "ms": ms, "tiep": la_tiep})
                        bo.lam_moi()
                        du = np.zeros(0, np.int16)  # bỏ âm thanh dồn lại trong lúc nhận dạng
            except WebSocketDisconnect:
                pass

    return app


# PhoWhisper-small FINE-TUNE trên câu lệnh xe (bản B epoch 3: TTS nhiều giọng + FLEURS + 12 phút giọng
# thật; PLAN.md "FINE-TUNE STT", người dùng đồng ý thay 2026-10-13). Bộ E: 24 -> 28 / 40, FLEURS không kém.
STT_FT = Path("models/stt/phowhisper-small-ftB/ep3-ct2-int8")


def chon_stt(bien: str | None, co_ft: bool) -> str | None:
    """Thư mục model STT theo `NEXA_STT`: mặc định bản fine-tune NẾU có trên máy; `goc` = PhoWhisper-small gốc
    (None = bản gốc `eval.stt.chay_stt.MO_HINH`). Thiếu bản fine-tune thì tự về bản gốc — demo vẫn chạy."""
    return None if bien == "goc" or not co_ft else str(STT_FT)


def tao_app_that() -> FastAPI:
    """App với model thật — gọi qua `uvicorn serve.giong_noi:tao_app_that --factory` (nạp model ~10 s)."""
    import soundfile as sf
    from vieneu import Vieneu

    from eval.stt.chay_stt import nap

    beam, goi_y = int(os.environ.get("NEXA_STT_BEAM", "1")), os.environ.get("NEXA_STT_GOI_Y") == "1"
    thu_muc = chon_stt(os.environ.get("NEXA_STT"), STT_FT.exists())
    print(f"[giong_noi] STT: {thu_muc or 'PhoWhisper-small gốc'}", flush=True)
    nghe = nap("phowhisper-small", beam, goi_y, thu_muc)
    tts = Vieneu()

    def doc(text: str) -> bytes:
        am = tts.infer(text, voice=GIONG)
        buf = io.BytesIO()
        sf.write(buf, am, tts.sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def tao_danh_thuc() -> tuple[Any, Any]:
        from openwakeword.vad import VAD

        from serve.danh_thuc import THU_MUC, BoDanhThuc

        return BoDanhThuc(), VAD(model_path=str(THU_MUC / "silero_vad.onnx"))

    from serve.danh_thuc import MO_HINH

    # chưa train wake-word thì không có /wake; NEXA_LUU_GIONG=<thư mục> bật lưu âm thanh + chữ để chẩn đoán
    luu = Path(os.environ["NEXA_LUU_GIONG"]) if os.environ.get("NEXA_LUU_GIONG") else None
    doc_dong = DocTungDoan(lambda text: tts.infer_stream(text, voice=GIONG), tts.sample_rate)
    return tao_app(nghe, doc, tao_danh_thuc if MO_HINH.exists() else None, luu, doc_dong)
