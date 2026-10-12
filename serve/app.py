"""Backend Nexa Drive — FastAPI + WebSocket.

Mỗi kết nối WebSocket = một phiên agent riêng (một xe mô phỏng + lời gọi đang chờ riêng). Luồng xử lý một câu
nằm ở `serve/tac_tu.py`; model đi qua cổng chung `serve/mo_hinh.py`. Chọn model bằng biến môi trường:
  NEXA_MO_HINH=gia   (mặc định) model giả luật từ khoá — chạy thử luồng / test, không cần llama.cpp;
  NEXA_MO_HINH=gguf  student THẬT: v6.1 s42 GGUF q4_K_M qua `llama-server` trên CPU, ngưỡng 0,80.
Câu trả lời (lượt 2): NEXA_LUOT_2=mau (mặc định — câu mẫu, đúng số, tức thì)
                      | model (model viết, qua lưới an toàn).
Bộ nhớ lượt trước:    NEXA_NHO=model (mặc định khi dùng GGUF mặc định — model v6.1 đã học ngữ cảnh)
                      | luat (luật `serve/bo_nho.py`; mặc định khi đặt NEXA_GGUF khác, vd model v5).
Model khác bản mặc định: NEXA_GGUF=<đường dẫn .gguf> (+ NEXA_NGUONG=<ngưỡng của bản đó>, NEXA_NHO).

Chạy local:
    uvicorn serve.app:app --port 8000                                   # model giả
    $env:NEXA_MO_HINH="gguf"; uvicorn serve.app:app --port 8000          # model thật (PowerShell)
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from schema.loader import CATALOG
from serve import tra_loi
from serve.contracts import AssistantTurn, ClientMessage, DieuKhienXe, ErrorEvent, StateEvent
from serve.mo_hinh import MoHinh, MoHinhGia
from serve.tac_tu import TacTu

# model dùng chung + máy chủ llama (nếu có), tạo một lần khi app khởi động
_TRANG_THAI: dict[str, object] = {}


@asynccontextmanager
async def _vong_doi(_: FastAPI) -> AsyncIterator[None]:
    if os.environ.get("NEXA_MO_HINH", "gia") == "gguf":
        from serve.mo_hinh_gguf import MayChuLlama, MoHinhGGUF

        # NEXA_GGUF: đường dẫn GGUF khác bản mặc định (vd model v5); NEXA_NHO=model: model đã học ngữ cảnh
        # lượt trước (v6 trở đi) -> đưa lời gọi lượt trước vào prompt, tắt luật bộ nhớ
        gguf = os.environ.get("NEXA_GGUF")
        # ~2 s; dừng hẳn nếu thiếu file GGUF / không bật được
        may = (MayChuLlama(gguf) if gguf else MayChuLlama()).bat()
        # bản mặc định (v6.1) học ngữ cảnh -> "model"; model khác chưa chắc -> "luat" trừ khi đặt NEXA_NHO
        nho = os.environ.get("NEXA_NHO", "luat" if gguf else "model") == "model"
        # NEXA_NGUONG: ngưỡng tin cậy riêng của model (chọn trên val theo luật mục V cho từng bản GGUF)
        kw = {"nguong": float(os.environ["NEXA_NGUONG"])} if os.environ.get("NEXA_NGUONG") else {}
        _TRANG_THAI.update(may=may, mo_hinh=MoHinhGGUF(url=may.url, nho_luot_truoc=nho, **kw))
    try:
        yield
    finally:
        may = _TRANG_THAI.pop("may", None)
        if may is not None:
            may.tat()  # type: ignore[attr-defined]
        _TRANG_THAI.pop("mo_hinh", None)


app = FastAPI(title="Nexa Drive", version="0.3.0", lifespan=_vong_doi)

# Dev: cho phép Vite dev server (cổng 5173) gọi sang.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def tao_mo_hinh() -> MoHinh:
    """Chỗ DUY NHẤT chọn model: student GGUF nếu đã bật lúc khởi động, không thì model giả."""
    m = _TRANG_THAI.get("mo_hinh")
    return m if m is not None else MoHinhGia()  # type: ignore[return-value]


@app.get("/health")
async def health() -> dict[str, object]:
    """Kiểm tra nhanh backend sống và catalog load được."""
    return {
        "status": "ok",
        "functions": len(CATALOG.functions),
        "special_tokens": len(CATALOG.special_tokens),
        "mo_hinh": type(tao_mo_hinh()).__name__,
    }


@app.get("/schema/actions")
async def scene_actions() -> dict[str, object]:
    """Cấu hình scene 3D suy ra từ functions.yaml (frontend cũng có bản tĩnh)."""
    return CATALOG.scene_actions()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    luot_2 = "model" if os.environ.get("NEXA_LUOT_2", "mau") == "model" else "mau"
    tac_tu = TacTu(tao_mo_hinh(), luot_2=luot_2)
    await ws.send_json(StateEvent(vehicle=tac_tu.xe.anh_chup()).model_dump())
    try:
        while True:
            raw = await ws.receive_json()
            try:
                if isinstance(raw, dict) and raw.get("type") == "dieu_khien_xe":
                    dk = DieuKhienXe.model_validate(raw)
                    bat_dau_chay = dk.dang_chay is True and not tac_tu.xe.s.dang_chay
                    if dk.dang_chay is not None:
                        tac_tu.xe.s.dang_chay = dk.dang_chay
                    if dk.khoa_tre_em is not None:
                        tac_tu.xe.s.khoa_tre_em = dk.khoa_tre_em
                    await ws.send_json(StateEvent(vehicle=tac_tu.xe.anh_chup()).model_dump())
                    if bat_dau_chay and (cb := tra_loi.canh_bao_khi_chay(tac_tu.xe.s)):
                        await ws.send_json(AssistantTurn(text=cb, loai="tu_choi").model_dump())
                    continue
                msg = ClientMessage.model_validate(raw)
            except ValidationError as exc:  # payload sai định dạng -> báo lỗi, giữ kết nối
                await ws.send_json(ErrorEvent(message="payload không hợp lệ", detail=str(exc)).model_dump())
                continue
            if not (msg.text or "").strip():
                await ws.send_json(
                    ErrorEvent(message="chưa có nội dung (nhận giọng nói chưa hỗ trợ)").model_dump()
                )
                continue
            # model thật mất ~0,3–2 s / lượt -> chạy ngoài vòng sự kiện để WebSocket không bị treo
            for su_kien in await run_in_threadpool(tac_tu.xu_ly, msg.text or "", msg.nguon):
                await ws.send_json(su_kien.model_dump())
    except WebSocketDisconnect:
        pass
