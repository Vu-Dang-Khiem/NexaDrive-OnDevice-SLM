"""Thí nghiệm mô tả hàm trong đáp án: phần lời gọi phải giữ NGUYÊN, chấm điểm phải ra y hệt."""

from eval.metrics import cham_mot_cau
from eval.sinh_du_doan import cat_output
from gen.mo_ta_dap_an import TIEN_TO, them_mo_ta, them_mo_ta_rows
from gen.records import CallRecord, ReplyRecord, SuggestRecord, negative_record


def test_call_duoc_noi_mo_ta_sau_nexa_end():
    r = CallRecord("Bật điều hoà 22 độ", "set_ac", {"state": "on", "temperature": 22}).to_row()
    m = them_mo_ta(r)
    assert m["completion"].startswith(r["completion"].replace("<|im_end|>", "") + TIEN_TO)
    assert "điều hoà" in m["completion"].lower() and m["completion"].endswith("<|im_end|>")
    assert m["prompt"] == r["prompt"] and m["meta"]["mo_ta_dap_an"] is True


def test_cat_output_bo_phan_mo_ta_nen_cham_diem_khong_doi():
    """Model sinh '...<nexa_end>\\n\\nMô tả hàm: ...' -> sau khi cắt phải chấm ĐÚNG như đáp án gốc."""
    r = CallRecord("Mở cốp sau", "open_trunk").to_row()
    sinh = them_mo_ta(r)["completion"]
    assert cham_mot_cau(cat_output(sinh), r["completion"])[0] == "dung"


def test_ask_nhieu_ham_co_mo_ta_moi_ham():
    r = SuggestRecord("Trong xe bí quá", (("set_ac", {"state": "on"}),
                                          ("control_window", {"window": "all", "position": "down"}))).to_row()
    c = them_mo_ta(r)["completion"]
    assert c.startswith("<nexa_ask>") and "set_ac:" in c and "control_window:" in c


def test_tu_choi_cung_co_mo_ta():
    c = them_mo_ta(negative_record("Kể chuyện cười đi").to_row())["completion"]
    assert "không liên quan" in c


def test_reply_giu_nguyen():
    r = ReplyRecord("Mở cốp", "open_trunk", {}, {"trunk_open": True}, "Đã mở cốp.").to_row()
    assert them_mo_ta_rows([r]) == [r]
