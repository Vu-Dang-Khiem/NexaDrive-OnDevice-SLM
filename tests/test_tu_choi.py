"""Test phần sinh lại câu trả lời TỪ CHỐI cho nhóm ngoài phạm vi.

Lỗi đang sửa: 652/1128 mẫu reply của nhóm này trả lời THẲNG câu hỏi ngoài phạm vi
("máy ảnh nào tốt nhất?" -> tư vấn máy ảnh) thay vì từ chối. Người dùng chỉ nghe lượt 2
nên toàn bộ việc dạy từ chối ở lượt 1 thành vô nghĩa.
"""

from gen.records import CallRecord, ReplyRecord, negative_record
from gen.teacher_prompts import build_refusal_reply_prompt
from gen.thay_tu_choi import la_reply_tu_choi, thay


# ── prompt từ chối ──────────────────────────────────────────────────────────
def test_prompt_cam_tra_loi_noi_dung():
    prompt = build_refusal_reply_prompt("máy ảnh nào tốt nhất hiện nay?")
    assert "KHÔNG trả lời nội dung câu hỏi" in prompt
    assert "máy ảnh nào tốt nhất hiện nay?" in prompt


def test_prompt_neu_pham_vi_ho_tro_de_tu_choi_co_thong_tin():
    prompt = build_refusal_reply_prompt("kể chuyện cười đi")
    assert "điều hoà" in prompt and "dẫn đường" in prompt


def test_prompt_yeu_cau_da_dang_cach_noi():
    """Data từ chối rất dễ rập khuôn 1 câu -> model học thành vẹt."""
    assert "đa dạng cách diễn đạt" in build_refusal_reply_prompt("giá vàng hôm nay?")


def test_prompt_khong_lo_functional_token():
    assert "<nexa_" not in build_refusal_reply_prompt("thủ đô nước Pháp là gì?")


# ── thay mẫu cũ bằng mẫu mới ────────────────────────────────────────────────
def _reply_tu_choi(cau: str, tra_loi: str) -> dict:
    return ReplyRecord(cau, "irrelevant_function", {}, {"handled": True}, tra_loi).to_row()


def test_nhan_dien_dung_mau_reply_cua_nhom_tu_choi():
    assert la_reply_tu_choi(_reply_tu_choi("kể chuyện cười", "Xin lỗi, tôi không hỗ trợ."))
    assert not la_reply_tu_choi(negative_record("kể chuyện cười").to_row())  # mẫu call
    assert not la_reply_tu_choi(
        ReplyRecord("còn xăng?", "get_fuel_level", {}, {"fuel_percent": 42}, "Còn 42%.").to_row()
    )


def test_thay_bo_mau_cu_va_them_mau_moi():
    goc = [
        CallRecord("mở cốp xe", "open_trunk", {}).to_row(),
        negative_record("kể chuyện cười").to_row(),
        _reply_tu_choi("máy ảnh nào tốt nhất?", "Hiện nay có nhiều máy ảnh tốt..."),  # mẫu HỎNG
    ]
    moi = [_reply_tu_choi("máy ảnh nào tốt nhất?", "Xin lỗi, tôi chỉ hỗ trợ việc trên xe thôi.")]
    ket_qua, da_bo, da_them = thay(goc, moi)

    assert da_bo == 1 and da_them == 1
    assert len(ket_qua) == 3
    tra_loi = [r["completion"] for r in ket_qua if la_reply_tu_choi(r)]
    assert "Xin lỗi" in tra_loi[0]
    assert "máy ảnh tốt" not in tra_loi[0]


def test_thay_khong_dung_den_mau_khac():
    goc = [CallRecord("mở cốp xe", "open_trunk", {}).to_row()]
    ket_qua, da_bo, da_them = thay(goc, [])
    assert ket_qua == goc and da_bo == 0 and da_them == 0
