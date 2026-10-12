"""Kiểm tra tính toàn vẹn của catalog hàm và định dạng chuỗi lời gọi."""

import pytest

from schema.call_format import CallParseError, parse_calls, render_completion
from schema.loader import CATALOG


# ── catalog ─────────────────────────────────────────────────────────────────
def test_index_lien_tuc_va_khong_trung_ten():
    indices = [f.index for f in CATALOG.functions]
    assert indices == list(range(len(indices)))
    assert len({f.name for f in CATALOG.functions}) == len(CATALOG.functions)


def test_so_token_dac_biet_bang_so_ham_cong_token_ket_thuc_va_token_hoi_lai():
    # token mới nối vào CUỐI — không làm xê dịch id các token cũ
    assert CATALOG.special_tokens[-2:] == [CATALOG.nexa_end, CATALOG.nexa_ask]
    assert len(CATALOG.special_tokens) == len(CATALOG.functions) + 2
    assert len(set(CATALOG.special_tokens)) == len(CATALOG.special_tokens)


def test_implicit_examples_chi_o_ham_hop_ly():
    co = {f.name for f in CATALOG.functions if f.implicit_examples}
    assert "set_ac" in co and "control_window" in co
    # hàm nhạy cảm không được sinh câu "gợi ý" (PLAN.md: luật an toàn)
    for nhay_cam in ("unlock_doors", "open_trunk", "open_charge_port", "open_fuel_flap"):
        assert nhay_cam not in co
    assert "irrelevant_function" not in co


def test_moi_ham_co_vi_du_va_ui_action_hop_le():
    cams, huds = set(CATALOG.camera_presets), set(CATALOG.hud_widgets)
    for f in CATALOG.functions:
        assert f.examples, f"{f.name} chưa có câu ví dụ"
        assert f.ui_action.camera in cams
        assert f.ui_action.hud.get("widget", "none") in huds


def test_json_schema_danh_dau_dung_tham_so_bat_buoc():
    schema = CATALOG.json_schema("set_ac")
    assert schema["required"] == ["state"]
    assert schema["properties"]["temperature"]["minimum"] == 16
    assert schema["properties"]["temperature"]["maximum"] == 30
    assert schema["properties"]["zone"]["enum"] == ["driver", "passenger", "rear", "all"]


def test_co_ham_fallback_ngoai_pham_vi():
    assert "irrelevant_function" in CATALOG.by_name


# ── prompt (ChatML, vì student là bản Instruct) ─────────────────────────────
def test_prompt_a_khong_lo_schema_va_ket_thuc_dung_cho():
    prompt = CATALOG.render_call_prompt("mở cốp xe")
    assert "def " not in prompt and "<nexa_" not in prompt  # không lộ mô tả hàm
    assert prompt.startswith("<|im_start|>system")
    assert prompt.rstrip("\n").endswith("<|im_start|>assistant")
    assert "mở cốp xe" in prompt


def test_prompt_b_chua_ket_qua_ham_va_ket_thuc_dung_cho():
    prompt = CATALOG.render_reply_prompt(
        "còn bao nhiêu xăng?",
        "<nexa_0>()<nexa_end>",
        {"fuel_percent": 42, "range_km": 380},
    )
    assert "<nexa_0>()" in prompt
    assert '"fuel_percent": 42' in prompt
    assert "<|im_start|>tool" in prompt
    assert prompt.rstrip("\n").endswith("<|im_start|>assistant")


def test_prompt_b_giu_nguyen_tieng_viet_khong_escape_unicode():
    prompt = CATALOG.render_reply_prompt("thời tiết?", "<nexa_23>()<nexa_end>", {"description": "mưa nhẹ"})
    assert "mưa nhẹ" in prompt


# ── render / parse chuỗi lời gọi ────────────────────────────────────────────
@pytest.mark.parametrize(
    "name,args",
    [
        ("get_fuel_level", {}),
        ("set_ac", {"state": "on", "temperature": 22, "zone": "driver"}),
        ("navigate_to", {"destination": "Sân bay Nội Bài", "avoid": "tolls"}),
        ("set_volume", {"delta": -3}),
        ("control_window", {"window": "front_right", "position": "down", "amount": 50}),
    ],
)
def test_render_roi_parse_lai_ra_dung_ban_dau(name, args):
    completion = render_completion([(name, args)])
    assert completion.endswith(CATALOG.nexa_end)
    (call,) = parse_calls(completion)
    assert call.name == name
    assert call.args == args
    assert not call.warnings


@pytest.mark.parametrize(
    "mo_ta,name,args",
    [
        ("dấu ; trong chuỗi", "navigate_to", {"destination": "Hà Nội; Việt Nam"}),
        ("dấu , trong chuỗi", "navigate_to", {"destination": "Hà Nội, Việt Nam"}),
        ("dấu = trong chuỗi", "play_media", {"query": "1+1=2"}),
        ("dấu ngoặc đơn", "play_media", {"query": "Bài hát (remix)"}),
        ("dấu nháy kép bên trong", "play_media", {"query": 'bài "hot" nhất'}),
        ("số âm", "set_volume", {"delta": -3}),
        ("số 0", "set_fan_speed", {"level": 0}),
        ("chuỗi nhìn giống số", "play_media", {"query": "22"}),
        ("chuỗi nhìn giống bool", "play_media", {"query": "true"}),
    ],
)
def test_ky_tu_dac_biet_trong_chuoi_van_roundtrip_dung(mo_ta, name, args):
    """Ký tự ; , = ( ) " trong giá trị chuỗi không được làm parser cắt nhầm chỗ."""
    (call,) = parse_calls(render_completion([(name, args)]))
    assert call.args == args, mo_ta
    assert not call.warnings


def test_parse_tham_so_theo_vi_tri_map_dung_ten():
    (call,) = parse_calls('<nexa_6>("on", 24, "rear")<nexa_end>')
    assert call.name == "set_ac"
    assert call.args == {"state": "on", "temperature": 24, "zone": "rear"}


def test_parse_nhieu_ham_song_song():
    calls = parse_calls('<nexa_11>();<nexa_6>(state="on", temperature=20)<nexa_end>')
    assert [c.name for c in calls] == ["open_trunk", "set_ac"]


def test_thieu_token_ket_thuc_van_parse_duoc_nhung_co_canh_bao():
    (call,) = parse_calls("<nexa_0>()")
    assert call.name == "get_fuel_level"
    assert any("thiếu" in w for w in call.warnings)


def test_bo_duoi_chatml_neu_model_sinh_kem():
    (call,) = parse_calls("<nexa_11>()<nexa_end><|im_end|>")
    assert call.name == "open_trunk"
    assert not call.warnings


def test_loi_goi_binh_thuong_khong_phai_de_xuat():
    (call,) = parse_calls('<nexa_6>(state="on")<nexa_end>')
    assert call.suggested is False


def test_de_xuat_render_roi_parse_lai_duoc():
    completion = render_completion([("set_ac", {"state": "on"})], ask=True)
    assert completion == '<nexa_ask><nexa_6>(state="on")<nexa_end>'
    (call,) = parse_calls(completion + "<|im_end|>")
    assert call.name == "set_ac"
    assert call.args == {"state": "on"}
    assert call.suggested is True
    assert not call.warnings


def test_de_xuat_2_ham_song_song_deu_danh_dau_de_xuat():
    calls = parse_calls(
        '<nexa_ask><nexa_6>(state="on");<nexa_15>(window="all", position="down")<nexa_end>'
    )
    assert [c.name for c in calls] == ["set_ac", "control_window"]
    assert all(c.suggested for c in calls)


def test_chuoi_rac_thi_bao_loi():
    with pytest.raises(CallParseError):
        parse_calls("Tôi nghĩ bạn nên mở cốp xe")


def test_functional_token_khong_ton_tai_thi_bao_loi():
    with pytest.raises(CallParseError):
        parse_calls("<nexa_999>()<nexa_end>")
