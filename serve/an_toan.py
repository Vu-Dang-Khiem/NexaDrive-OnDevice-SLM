"""Guardrail 2 — LUẬT AN TOÀN: đứng SAU validator, TRƯỚC khi đụng vào xe (PLAN.md Phase 5).

Nguyên tắc "model lo hiểu, code lo an toàn": model không bao giờ tự quyết chuyện an toàn. Luật đọc từ
trường `an_toan` trong schema/functions.yaml (một nguồn duy nhất) và từ trạng thái xe lúc đó:

  - `dang_chay`   xe đang chạy (công tắc demo "Đỗ xe / Đang chạy")
  - `khoa_tre_em` khoá trẻ em đang bật
  - `wake_word`   lệnh đến từ từ khoá đánh thức — ai ngồi gần cũng nói được, nên hàm nhạy cảm phải xác nhận

Một luật có thể ghép nhiều điều kiện (`khi: [khoa_tre_em, wake_word]`) — chỉ áp khi TẤT CẢ cùng đúng.
Luật xét theo thứ tự, luật khớp ĐẦU TIÊN quyết định.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from schema.loader import CATALOG
from serve.xe_mo_phong import TrangThaiXe

XuLy = Literal["cho_phep", "tu_choi", "hoi_lai", "gioi_han"]


@dataclass(frozen=True)
class QuyetDinh:
    xu_ly: XuLy
    args: dict[str, Any]  # tham số sau khi áp luật (khác đầu vào khi `gioi_han`)
    ly_do: str = ""


def _gia_tri(ten: str, args: dict[str, Any], tham_so: str) -> Any:
    """Giá trị tham số, kể cả khi model không ghi mà dùng mặc định trong schema."""
    if tham_so in args:
        return args[tham_so]
    p = next(p for p in CATALOG.by_name[ten].params if p.name == tham_so)
    return p.default


def xet(ten: str, args: dict[str, Any], xe: TrangThaiXe, nguon: str = "nut") -> QuyetDinh:
    dieu_kien = {"dang_chay": xe.dang_chay, "khoa_tre_em": xe.khoa_tre_em, "wake_word": nguon == "wake_word"}
    for q in CATALOG.by_name[ten].an_toan:
        if not all(dieu_kien[k] for k in q.khi):
            continue
        if any(_gia_tri(ten, args, k) not in cho_phep for k, cho_phep in q.neu_tham_so.items()):
            continue
        if q.xu_ly == "gioi_han":
            assert q.tham_so is not None and q.toi_da is not None  # loader đã kiểm
            hien_tai = _gia_tri(ten, args, q.tham_so)
            if hien_tai is None or hien_tai <= q.toi_da:
                continue  # đã trong giới hạn -> luật không cần can thiệp
            return QuyetDinh("gioi_han", {**args, q.tham_so: type(hien_tai)(q.toi_da)}, q.ly_do)
        return QuyetDinh(q.xu_ly, dict(args), q.ly_do)  # type: ignore[arg-type]
    return QuyetDinh("cho_phep", dict(args))
