"""Test ghép mẫu training (gen/records.py) — logic thuần, không cần GPU."""

import pytest

from gen.records import (
    CallRecord,
    ReplyRecord,
    SuggestRecord,
    datasheet,
    negative_record,
    query_from_call_prompt,
)
from schema.call_format import parse_calls
from schema.loader import CATALOG


# ── mẫu loại A (dạy gọi hàm) ────────────────────────────────────────────────
def test_call_record_ghep_dung_prompt_va_completion():
    rec = CallRecord("Bật điều hòa 22 độ ghế lái", "set_ac", {"state": "on", "temperature": 22})
    row = rec.to_row()

    # prompt phải là Prompt A, kết thúc đúng chỗ model bắt đầu sinh
    assert row["prompt"].rstrip("\n").endswith("<|im_start|>assistant")
    assert "Bật điều hòa 22 độ ghế lái" in row["prompt"]
    assert "<nexa_" not in row["prompt"]  # không lộ token trong phần model nhìn thấy

    # completion phải là lời gọi + token dừng
    assert row["completion"] == '<nexa_6>(state="on", temperature=22)<nexa_end><|im_end|>'
    assert row["meta"] == {"lang": "vi", "type": "call", "fn": "set_ac"}


def test_call_record_completion_parse_lai_duoc():
    """Nhãn sinh ra phải đọc ngược lại được — nếu không, lúc chạy thật sẽ hỏng."""
    rec = CallRecord("mở cốp", "open_trunk", {})
    (call,) = parse_calls(rec.to_row()["completion"])
    assert call.name == "open_trunk"
    assert call.args == {}


def test_negative_record_luon_la_irrelevant_function():
    row = negative_record("thủ đô nước Pháp là gì?").to_row()
    assert row["completion"] == "<nexa_24>()<nexa_end><|im_end|>"
    assert row["meta"]["fn"] == "irrelevant_function"


# ── mẫu loại ask (câu ngầm hiểu -> đề xuất rồi hỏi lại) ─────────────────────
def test_suggest_record_ghep_dung_completion_co_token_hoi_lai():
    rec = SuggestRecord(
        "Trong xe bí quá",
        (("set_ac", {"state": "on"}), ("control_window", {"window": "all", "position": "down"})),
    )
    row = rec.to_row()
    assert row["prompt"] == CATALOG.render_call_prompt("Trong xe bí quá")  # cùng Prompt A
    assert row["completion"] == (
        '<nexa_ask><nexa_6>(state="on");<nexa_15>(window="all", position="down")<nexa_end><|im_end|>'
    )
    assert row["meta"] == {
        "lang": "vi",
        "type": "ask",
        "fn": "set_ac",
        "fns": ["set_ac", "control_window"],
    }


def test_suggest_record_parse_lai_ra_de_xuat():
    row = SuggestRecord("Ôi nóng quá", (("set_ac", {"state": "on"}),)).to_row()
    (call,) = parse_calls(row["completion"])
    assert call.name == "set_ac"
    assert call.suggested is True


def test_suggest_record_rong_bao_loi():
    with pytest.raises(ValueError):
        SuggestRecord("Ôi nóng quá", ())


def test_lay_lai_cau_nguoi_dung_tu_prompt_a():
    row = negative_record("hôm nay trời nóng quá!").to_row()
    assert query_from_call_prompt(row["prompt"]) == "hôm nay trời nóng quá!"


def test_lay_lai_cau_tu_prompt_hong_thi_bao_loi():
    with pytest.raises(ValueError):
        query_from_call_prompt("không phải ChatML")


# ── mẫu loại B (dạy trả lời) ────────────────────────────────────────────────
def test_reply_record_ghep_dung_prompt_co_ket_qua_ham():
    rec = ReplyRecord(
        query="còn bao nhiêu xăng?",
        fn_name="get_fuel_level",
        args={},
        result={"fuel_percent": 42, "range_km": 176.4},
        reply="Xăng còn 42%, đi được khoảng 176 km.",
    )
    row = rec.to_row()

    assert "<nexa_0>()<nexa_end>" in row["prompt"]  # lượt 1 nằm trong context
    assert '"fuel_percent": 42' in row["prompt"]  # kết quả hàm có trong context
    assert "<|im_start|>tool" in row["prompt"]
    assert row["prompt"].rstrip("\n").endswith("<|im_start|>assistant")
    assert row["completion"] == "Xăng còn 42%, đi được khoảng 176 km.<|im_end|>"
    assert row["meta"]["type"] == "reply"


def test_reply_record_cat_khoang_trang_thua():
    rec = ReplyRecord("mở cốp", "open_trunk", {}, {"trunk_open": True}, "  Đã mở cốp.  ")
    assert rec.to_row()["completion"] == "Đã mở cốp.<|im_end|>"


# ── datasheet (kiểm tra cân bằng dataset trước khi train) ───────────────────
def test_datasheet_dem_dung_theo_loai_va_ham():
    rows = [
        CallRecord("q1", "set_ac", {"state": "on"}).to_row(),
        CallRecord("q2", "open_trunk", {}).to_row(),
        ReplyRecord("q3", "open_trunk", {}, {"trunk_open": True}, "Đã mở cốp.").to_row(),
    ]
    d = datasheet(rows)
    assert d["tong"] == 3
    assert d["theo_loai"] == {"call": 2, "reply": 1}
    assert d["theo_ham"] == {"open_trunk": 2, "set_ac": 1}
    assert d["theo_ngon_ngu"] == {"vi": 3}
    assert d["ti_le_reply"] == pytest.approx(0.333, abs=0.01)


def test_datasheet_chi_ra_ham_chua_co_mau_nao():
    """Cảnh báo sớm nếu 1 hàm không có mẫu nào — model sẽ không học được hàm đó."""
    rows = [CallRecord("q", "set_ac", {"state": "on"}).to_row()]
    d = datasheet(rows)
    assert "open_trunk" in d["ham_chua_co_mau"]
    assert len(d["ham_chua_co_mau"]) == len(CATALOG.functions) - 1
