"""Guardrail 1 — VALIDATOR: kiểm tham số model sinh ra TRƯỚC khi đụng vào xe (PLAN.md Phase 5).

Model 1,5B đôi khi sinh tham số ngoài schema — đã thấy thật trên tập test: `operate_sunroof(action="fly")`,
`set_hazard_lights(level=3)`. Không bao giờ đưa những lời gọi đó xuống xe. Ba cách phản ứng khác nhau:

  - giá trị SAI (ngoài danh sách cho phép / ngoài khoảng) -> KHÔNG thực thi, báo chưa hiểu;
  - THIẾU tham số bắt buộc -> HỎI LẠI người lái ("hạ kính cửa nào?");
  - tham số LẠ (schema không có) -> bỏ qua, ghi lại để theo dõi.

Ép kiểu và kiểm khoảng dùng lại đúng hàm của dây chuyền sinh dữ liệu (`gen/teacher_prompts.py`) —
nhãn train và lời gọi lúc chạy được kiểm bằng CÙNG một luật.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gen.teacher_prompts import TeacherResponseError, _check_rang_buoc, _coerce_to_declared_type
from schema.loader import CATALOG

# `set_volume` / `set_fan_speed` (v6) không có tham số bắt buộc nhưng phải có ĐÚNG MỘT trong hai (mô tả trong
# schema)
THIEU_MUC_AM_LUONG = "muc_am_luong"
THIEU_MUC_QUAT = "muc_quat"
_MOT_TRONG_HAI = {
    "set_volume": (("set_value", "delta"), THIEU_MUC_AM_LUONG),
    "set_fan_speed": (("level", "delta"), THIEU_MUC_QUAT),
}


@dataclass
class KetQuaKiemTra:
    args: dict[str, Any]
    thieu: list[str] = field(default_factory=list)  # tham số bắt buộc còn thiếu -> hỏi lại
    loi: list[str] = field(default_factory=list)  # giá trị sai -> không thực thi
    bo_qua: list[str] = field(default_factory=list)  # tham số lạ đã bỏ

    @property
    def hop_le(self) -> bool:
        return not self.thieu and not self.loi


def kiem_tra(ten: str, args: dict[str, Any]) -> KetQuaKiemTra:
    spec = CATALOG.by_name.get(ten)
    if spec is None:
        return KetQuaKiemTra(args={}, loi=[f"không có hàm tên {ten!r}"])
    theo_ten = {p.name: p for p in spec.params}
    kq = KetQuaKiemTra(args={})
    sai: set[str] = set()
    for k, v in args.items():
        if k not in theo_ten:
            kq.bo_qua.append(k)
            continue
        # Chuỗi rỗng = CHƯA BIẾT, không phải giá trị: dữ liệu hỏi lại dùng `navigate_to(destination="")`
        # cho câu "sợ lạc quá" (gợi ý dẫn đường, chưa biết đi đâu). Coi như thiếu -> hỏi "đi đâu?".
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        p = theo_ten[k]
        try:
            gia_tri = _coerce_to_declared_type(v, p)
            _check_rang_buoc(gia_tri, p)
        except TeacherResponseError as e:
            kq.loi.append(str(e))
            sai.add(k)
            continue
        kq.args[k] = gia_tri
    # tham số đã SAI thì không hỏi lại — cả lời gọi đã hỏng, hỏi thêm chỉ làm rối
    kq.thieu = [p.name for p in spec.params if p.required and p.name not in kq.args and p.name not in sai]
    if ten in _MOT_TRONG_HAI:
        (a, b), ma_thieu = _MOT_TRONG_HAI[ten]
        co = [k for k in (a, b) if k in kq.args]
        if len(co) == 2:
            kq.loi.append(f"{ten} chỉ dùng một trong {a} / {b}")
        elif not co and not kq.loi:
            kq.thieu.append(ma_thieu)
    if ten == "set_ac" and "temperature" in kq.args and "delta" in kq.args:
        kq.loi.append("set_ac chỉ dùng một trong temperature / delta")
    return kq
