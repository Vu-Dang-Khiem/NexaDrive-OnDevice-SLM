# ruff: noqa: E501  — file chứa nguyên văn prompt tiếng Việt; ngắt dòng giữa câu sẽ làm đổi chính prompt
"""BASELINE "mô tả hàm trong prompt" (Schema-in-Prompt, SIP) — PLAN.md Phase 3, chốt 2026-10-02.

Câu hỏi cần trả lời: "làm agent thì chỉ cần prompt, sao phải train?". Baseline đưa câu người dùng + MÔ TẢ 24 HÀM
vào prompt, model trả JSON. Chạy trên:
  (a) teacher Qwen2.5-72B (model lớn, KHÔNG train)  — mốc "model lớn + prompt";
  (b) Qwen2.5-1.5B-Instruct GỐC (chưa train)          — đúng cách "chỉ dùng prompt" trên chính model nhỏ của bài.

Công bằng với student:
  - CÙNG chính sách hành vi như dữ liệu train: làm luôn / hỏi lại khi chỉ than thở / từ chối việc không có; hàm
    chỉ đọc không hỏi lại; không đề xuất 4 hàm nhạy cảm; chỉ điền tham số được nói rõ.
  - Output JSON được CHUYỂN sang `<nexa_i>(...)<nexa_end>` để chấm bằng NGUYÊN bộ chấm (`eval/metrics.py`, luật C).
    JSON hỏng / tên hàm lạ -> không đọc được (tính sai) — như student sinh sai định dạng.
  - Ví dụ mẫu (few-shot) lấy từ câu mồi trong schema + 1 câu từ chối viết riêng — có test chặn trùng tập test.
"""

from __future__ import annotations

import json
from typing import Any

from gen.teacher_prompts import TeacherResponseError, _extract_json
from schema.call_format import render_completion
from schema.loader import CATALOG

TU_CHOI = "irrelevant_function"
HAM_NHAY_CAM = sorted(f.name for f in CATALOG.functions if f.nhay_cam)
HAM_CHI_DOC = sorted(f.name for f in CATALOG.functions if f.chi_doc)

# Few-shot: một ví dụ cho mỗi hành vi. Câu lấy từ câu mồi schema (đã có test chặn trùng tập test) + 1 câu từ chối.
VI_DU = [
    (
        "Bật điều hòa 22 độ ghế lái",
        {
            "hanh_dong": "goi",
            "loi_goi": [{"ham": "set_ac", "tham_so": {"state": "on", "temperature": 22, "zone": "driver"}}],
        },
    ),
    (
        "Trong xe hầm hơi quá chừng",
        {
            "hanh_dong": "hoi_lai",
            "loi_goi": [
                {"ham": "set_ac", "tham_so": {"state": "on"}},
                {"ham": "control_window", "tham_so": {"window": "all", "position": "down"}},
            ],
        },
    ),
    ("Bật chế độ massage ghế sau giúp tôi", {"hanh_dong": "tu_choi", "loi_goi": []}),
]


def danh_sach_ham() -> str:
    return "\n\n".join(f.teacher_doc() for f in CATALOG.functions if f.name != TU_CHOI)


def build_sip_prompt(cau: str, luot_truoc: tuple[str, dict[str, Any]] | None = None) -> str:
    """Prompt SIP dạng ChatML (cả teacher lẫn Qwen 1,5B gốc đều là Qwen-Instruct).

    `luot_truoc` = (hàm, tham số) đã thực hiện ở lượt trước — tập hai lượt (bổ sung 2026-10-12). Baseline thấy ĐÚNG
    thông tin student thấy (lời gọi lượt trước, không có câu nói lượt trước). Không có `luot_truoc` -> prompt Y HỆT
    bản cũ (số baseline cũ chạy lại vẫn khớp)."""
    vi_du = "\n".join(f'Câu: "{c}"\nTrả lời: {json.dumps(d, ensure_ascii=False)}' for c, d in VI_DU)
    luat_luot_truoc, dong_luot_truoc = "", ""
    if luot_truoc is not None:
        luat_luot_truoc = (
            '\n- Nếu có dòng "Lượt trước", câu hiện tại có thể nói tiếp việc vừa làm (đổi chiều, làm thêm, áp dụng cho'
            " bộ phận khác) — hiểu câu theo lượt trước; câu không liên quan thì xử lý như câu mới."
        )
        ham, tham_so = luot_truoc
        dong_luot_truoc = f"Lượt trước (đã thực hiện): {ham}({json.dumps(tham_so, ensure_ascii=False)})\n"
    he_thong = f"""Bạn là tác tử điều khiển xe ô tô bằng giọng nói. Bạn CHỈ làm được các việc qua những hàm sau:

{danh_sach_ham()}

Với mỗi câu của người lái, chọn MỘT trong ba hành động:
- "goi": câu nhờ RÕ RÀNG một việc làm được -> gọi hàm (1 hoặc vài lời gọi), chỉ điền tham số mà câu nói rõ; không bịa.
- "hoi_lai": câu chỉ than thở / kể tình huống mà một hàm có thể giúp -> đề xuất 1–2 lời gọi để hỏi lại người lái.
  Hàm chỉ đọc ({", ".join(HAM_CHI_DOC)}) thì không hỏi lại, gọi luôn. Không bao giờ đề xuất {", ".join(HAM_NHAY_CAM)}.
- "tu_choi": việc không có hàm nào làm được, tán gẫu, hỏi kiến thức chung.{luat_luot_truoc}

Chỉ trả về đúng một JSON, không thêm chữ nào:
{{"hanh_dong": "goi" | "hoi_lai" | "tu_choi", "loi_goi": [{{"ham": "tên_hàm", "tham_so": {{...}}}}]}}

Ví dụ:
{vi_du}"""
    return (
        f"<|im_start|>system\n{he_thong}<|im_end|>\n"
        f'<|im_start|>user\n{dong_luot_truoc}Câu: "{cau.strip()}"<|im_end|>\n<|im_start|>assistant\n'
    )


def sang_nexa(text: str) -> str:
    """JSON của baseline -> chuỗi `<nexa_i>(...)<nexa_end>` cho bộ chấm. Hỏng -> "" (bộ chấm tính không đọc được).

    Khoan dung như với student: bỏ tham số lạ / giá trị null; KHÔNG sửa giá trị sai (bộ chấm tự tính sai tham số).
    """
    try:
        d = _extract_json(text)
    except TeacherResponseError:
        return ""
    if not isinstance(d, dict):
        return ""
    hanh_dong = d.get("hanh_dong")
    if hanh_dong == "tu_choi":
        return render_completion([(TU_CHOI, {})])
    if hanh_dong not in ("goi", "hoi_lai"):
        return ""
    calls: list[tuple[str, dict[str, Any]]] = []
    for lg in d.get("loi_goi") or []:
        if not isinstance(lg, dict):
            return ""
        spec = CATALOG.by_name.get(str(lg.get("ham")))
        if spec is None or spec.name == TU_CHOI:
            return ""
        ten_tham_so = {p.name for p in spec.params}
        args = {k: v for k, v in (lg.get("tham_so") or {}).items() if k in ten_tham_so and v is not None}
        calls.append((spec.name, args))
    if not calls:
        return ""
    return render_completion(calls, ask=hanh_dong == "hoi_lai")
