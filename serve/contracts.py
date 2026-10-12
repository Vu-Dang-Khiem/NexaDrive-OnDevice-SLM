"""Hợp đồng dữ liệu giữa web client và backend agent (WebSocket /ws).

Client -> server:  ClientMessage (câu nói) | DieuKhienXe (công tắc demo "Đang chạy", "Khoá trẻ em")
Server -> client:  một hoặc nhiều ServerEvent (assistant_turn | clarify | error | state)

Scene 3D được điều khiển hoàn toàn bằng `ui_actions` trong assistant_turn, nên
frontend không cần biết ngữ nghĩa của hàm — chỉ cần biết id camera/animation/hud,
vốn sinh ra từ schema/functions.yaml qua web/src/sceneActions.json.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Lang = Literal["vi", "en", "auto"]


# ── client -> server ────────────────────────────────────────────────────────
class ClientMessage(BaseModel):
    type: Literal["user_message"] = "user_message"
    text: str | None = None
    audio_b64: str | None = Field(default=None, description="WAV/OGG base64 nếu client tự thu giọng nói")
    lang: Lang = "auto"
    session_id: str | None = None
    # Lệnh đến từ nút bấm hay từ khoá đánh thức — luật an toàn bắt hàm nhạy cảm từ wake-word phải xác nhận
    nguon: Literal["nut", "wake_word"] = "nut"


class DieuKhienXe(BaseModel):
    """Công tắc demo trên UI: "Đỗ xe / Đang chạy", "Khoá trẻ em" — luật an toàn đọc hai cờ này."""

    type: Literal["dieu_khien_xe"] = "dieu_khien_xe"
    dang_chay: bool | None = None
    khoa_tre_em: bool | None = None


# ── server -> client ────────────────────────────────────────────────────────
class CallPayload(BaseModel):
    name: str
    arguments: dict[str, Any] = {}
    nexa: str


class UiActionPayload(BaseModel):
    camera: str
    animation: str | None = None
    hud: dict[str, Any] = {}


class AssistantTurn(BaseModel):
    """Một lượt trả lời hoàn chỉnh: lời gọi hàm + kết quả + câu trả lời + hành động 3D."""

    type: Literal["assistant_turn"] = "assistant_turn"
    text: str
    lang: Lang = "vi"
    # thuc_thi: đã làm (có thể kèm lời giải thích luật an toàn) · tu_choi: ngoài khả năng / luật an toàn chặn
    # · chua_hieu: model sinh lời gọi hỏng hoặc sai giá trị · huy: người lái huỷ lời gọi đang chờ
    # · tro_chuyen: câu xã giao / tán gẫu (chào, cảm ơn, hỏi danh tính...) — đáp câu mẫu, không đụng xe
    loai: Literal["thuc_thi", "tu_choi", "chua_hieu", "huy", "tro_chuyen"] = "thuc_thi"
    calls: list[CallPayload] = []
    results: list[dict[str, Any]] = []
    ui_actions: list[UiActionPayload] = []
    latency_ms: float | None = None


class ClarifyTurn(BaseModel):
    """Phát ra THAY VÌ thực thi — agent hỏi lại người lái.

    ly_do: de_xuat (câu mơ hồ, model chỉ đề xuất) · thieu_tham_so · an_toan (luật yêu cầu xác nhận)
    · chon (đề xuất có 2 lựa chọn mà người lái chỉ nói "có")."""

    type: Literal["clarify"] = "clarify"
    text: str
    lang: Lang = "vi"
    ly_do: Literal["de_xuat", "thieu_tham_so", "an_toan", "chon"] = "de_xuat"
    function: str = ""
    functions: list[str] = []
    missing: list[str] = []


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str
    detail: str | None = None


class StateEvent(BaseModel):
    """Ảnh chụp toàn bộ trạng thái xe, đẩy khi vừa kết nối và sau mỗi lần thay đổi."""

    type: Literal["state"] = "state"
    vehicle: dict[str, Any]


ServerEvent = AssistantTurn | ClarifyTurn | ErrorEvent | StateEvent
