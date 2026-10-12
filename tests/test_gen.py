"""Test prompt + parser của gen/ bằng dữ liệu GIẢ LẬP (hardcode) — đóng vai
"câu trả lời mà teacher có thể trả về", KHÔNG gọi model thật, không cần GPU/Modal.

Mỗi test: tự bịa 1 chuỗi kiểu model hay trả lời -> đưa qua parser -> kiểm tra
parser đọc đúng ra dữ liệu mong muốn. Khi chạy thật trên Modal gặp kiểu lạ chưa
lường tới, thêm test hardcode mới cho kiểu đó rồi sửa parser (quy trình lặp lại).
"""

import json

import pytest

from gen.simulate import all_functions_have_simulator, simulate_result
from gen.teacher_prompts import (
    TeacherResponseError,
    build_ask_label_prompt,
    build_call_gen_prompt,
    build_implicit_query_prompt,
    build_negative_query_prompt,
    build_query_gen_prompt,
    build_reply_gen_prompt,
    build_verify_ask_prompt,
    build_verify_negative_prompt,
    build_verify_prompt,
    de_xuat_hop_le,
    parse_ask_label_response,
    parse_call_gen_response,
    parse_query_gen_response,
    parse_reply_gen_response,
    parse_verify_response,
    tham_so_duoc_noi_ro,
)
from schema.call_format import render_completion
from schema.loader import CATALOG


# ── simulate.py ──────────────────────────────────────────────────────────────
def test_moi_ham_trong_catalog_deu_co_simulator():
    assert all_functions_have_simulator() == []


def test_simulator_tra_dung_cac_khoa_da_khai_trong_returns():
    """Kết quả mô phỏng phải khớp đúng `returns` khai trong functions.yaml —
    lệch là dữ liệu loại B sẽ dạy model nói về field không tồn tại."""
    for f in CATALOG.functions:
        thuc_te = set(simulate_result(f.name, {}, seed=1).keys())
        khai_bao = set(f.returns.keys())
        assert thuc_te == khai_bao, f"{f.name}: thiếu={khai_bao - thuc_te}, thừa={thuc_te - khai_bao}"


def test_simulate_cung_seed_ra_cung_ket_qua():
    a = simulate_result("get_fuel_level", {}, seed=42)
    b = simulate_result("get_fuel_level", {}, seed=42)
    assert a == b


def test_simulate_set_ac_echo_dung_tham_so_da_truyen():
    r = simulate_result("set_ac", {"state": "on", "temperature": 22, "zone": "driver"}, seed=1)
    assert r == {"ac_on": True, "temperature": 22, "zone": "driver", "mode": "auto"}  # v6: thêm `mode`


def test_simulate_ham_khong_ton_tai_bao_loi():
    with pytest.raises(KeyError):
        simulate_result("ham_khong_ton_tai", {}, seed=1)


# ── 1. parse_query_gen_response — GIẢ LẬP câu trả lời của teacher ───────────
def test_parse_query_json_sach():
    gia_lap = '{"queries": ["Bật điều hòa lên", "Điều hòa 22 độ ghế lái"]}'
    assert parse_query_gen_response(gia_lap) == ["Bật điều hòa lên", "Điều hòa 22 độ ghế lái"]


def test_parse_query_boc_trong_markdown_fence():
    # Model rất hay trả lời kiểu này — bọc JSON trong ```json ... ```
    gia_lap = """Đây là các câu tôi tạo ra:
    ```json
    {"queries": ["Mở cốp xe", "Cho tôi mở cốp"]}
    ```
    Hy vọng giúp ích được cho bạn."""
    assert parse_query_gen_response(gia_lap) == ["Mở cốp xe", "Cho tôi mở cốp"]


def test_parse_query_bo_qua_chuoi_rong():
    gia_lap = '{"queries": ["Bật đèn pha", "  ", ""]}'
    assert parse_query_gen_response(gia_lap) == ["Bật đèn pha"]


def test_parse_query_thieu_khoa_bao_loi_ro_rang():
    gia_lap = '{"cau_hoi": ["Bật điều hòa lên"]}'  # đặt sai tên khoá
    with pytest.raises(TeacherResponseError):
        parse_query_gen_response(gia_lap)


def test_parse_query_khong_phai_json_bao_loi():
    gia_lap = "Xin lỗi, tôi không thể tạo câu hỏi cho yêu cầu này."
    with pytest.raises(TeacherResponseError):
        parse_query_gen_response(gia_lap)


# ── vớt vát khi output bị cắt giữa chừng (LỖI THẬT đã gặp ở lần chạy 1000/hàm) ──
def test_parse_query_bi_cat_giua_chung_van_vot_duoc_cau_hoan_chinh():
    """Chạm max_tokens -> JSON thiếu dấu đóng -> json.loads hỏng.
    Không được vứt cả lô: các câu hoàn chỉnh phía trước vẫn phải lấy được."""
    bi_cat = '{"queries": ["Bật điều hòa lên", "Cho tôi mát chút", "Chỉnh nhiệt độ xuống 20 đ'
    ket_qua = parse_query_gen_response(bi_cat)
    assert ket_qua == ["Bật điều hòa lên", "Cho tôi mát chút"]  # câu dở dang bị loại


def test_parse_query_bi_cat_giu_duoc_nhieu_cau():
    """Mô phỏng đúng tình huống thật: sinh được vài trăm câu rồi mới bị cắt."""
    cau = [f"câu hỏi mẫu số {i}" for i in range(250)]
    day_du = '{"queries": [' + ", ".join(f'"{c}"' for c in cau)
    bi_cat = day_du + ', "câu cuối đang viết dở'  # cắt giữa chừng
    ket_qua = parse_query_gen_response(bi_cat)
    assert len(ket_qua) == 250
    assert ket_qua[0] == "câu hỏi mẫu số 0"
    assert ket_qua[-1] == "câu hỏi mẫu số 249"


def test_parse_query_bi_cat_van_giai_escape_dung():
    bi_cat = '{"queries": ["nói \\"xin chào\\" đi", "câu dở'
    assert parse_query_gen_response(bi_cat) == ['nói "xin chào" đi']


def test_parse_query_cat_ngay_tu_dau_khong_vot_duoc_gi_thi_bao_loi():
    """Nếu cắt trước cả khi có câu nào hoàn chỉnh -> không vớt được -> vẫn phải báo lỗi."""
    bi_cat = '{"queries": ["câu đầu tiên chưa viết xong'
    with pytest.raises(TeacherResponseError):
        parse_query_gen_response(bi_cat)


# ── build_query_gen_prompt ──────────────────────────────────────────────────
def test_query_prompt_co_du_thong_tin_teacher_can():
    fn = CATALOG.by_name["set_ac"]
    prompt = build_query_gen_prompt(fn, n=7)
    assert "7 câu" in prompt  # đúng số lượng yêu cầu
    assert "set_ac" in prompt  # có chữ ký hàm
    assert "một trong: on, off" in prompt  # enum đúng (không phải True/False — lỗi YAML đã sửa)
    assert "16..30" in prompt  # khoảng giá trị
    assert fn.examples[0] in prompt  # có câu mẫu để mồi
    assert '{"queries"' in prompt  # có yêu cầu định dạng trả về


def test_query_prompt_khong_lo_functional_token_cho_teacher():
    """Teacher không biết <nexa_i> — lộ ra chỉ làm nó rối, sinh data sai."""
    for fn in CATALOG.functions:
        assert "<nexa_" not in build_query_gen_prompt(fn, n=3)


@pytest.mark.parametrize("lang,mong_doi", [("vi", "tiếng Việt"), ("en", "English")])
def test_query_prompt_doi_ngon_ngu(lang, mong_doi):
    prompt = build_query_gen_prompt(CATALOG.by_name["open_trunk"], n=3, lang=lang)
    assert mong_doi in prompt


# ── build_call_gen_prompt ───────────────────────────────────────────────────
def test_call_prompt_co_cau_hoi_va_mo_ta_ham():
    fn = CATALOG.by_name["set_ac"]
    prompt = build_call_gen_prompt("Bật điều hòa 22 độ ghế lái", fn)
    assert "Bật điều hòa 22 độ ghế lái" in prompt
    assert "set_ac" in prompt
    assert "driver, passenger, rear, all" in prompt  # enum zone để teacher chọn đúng
    assert '{"arguments"' in prompt
    assert "đừng tự bịa giá trị mặc định" in prompt  # chặn teacher tự điền default


def test_call_prompt_khong_lo_functional_token():
    fn = CATALOG.by_name["set_ac"]
    assert "<nexa_" not in build_call_gen_prompt("bật điều hoà", fn)


# ── 2. build_negative_query_prompt ──────────────────────────────────────────
def test_negative_prompt_khong_nhac_ten_ham_cu_the():
    prompt = build_negative_query_prompt(5)
    for name in ["get_fuel_level", "set_ac", "<nexa_"]:
        assert name not in prompt


# ── 3. parse_call_gen_response — GIẢ LẬP câu trả lời của teacher ────────────
def test_parse_call_json_sach():
    fn = CATALOG.by_name["set_ac"]
    gia_lap = '{"arguments": {"state": "on", "temperature": 22, "zone": "driver"}}'
    assert parse_call_gen_response(gia_lap, fn) == {"state": "on", "temperature": 22, "zone": "driver"}


def test_parse_call_loc_bo_tham_so_khong_thuoc_schema():
    # Model đôi khi "sáng tạo" thêm tham số không có trong schema -> phải lọc bỏ
    fn = CATALOG.by_name["set_ac"]
    gia_lap = '{"arguments": {"state": "on", "mau_sac": "do"}}'
    assert parse_call_gen_response(gia_lap, fn) == {"state": "on"}


def test_parse_call_loc_bo_gia_tri_null():
    fn = CATALOG.by_name["set_ac"]
    gia_lap = '{"arguments": {"state": "on", "temperature": null}}'
    assert parse_call_gen_response(gia_lap, fn) == {"state": "on"}


def test_parse_call_ham_khong_tham_so_tra_arguments_rong():
    fn = CATALOG.by_name["get_fuel_level"]
    gia_lap = '{"arguments": {}}'
    assert parse_call_gen_response(gia_lap, fn) == {}


# ── ép kiểu: LLM rất hay trả số dạng chuỗi, phải quy về đúng kiểu schema ────
@pytest.mark.parametrize(
    "gia_tri_teacher_tra",
    [22, "22", 22.0, "22.0", " 22 "],
)
def test_parse_call_ep_moi_kieu_so_ve_cung_1_int(gia_tri_teacher_tra):
    fn = CATALOG.by_name["set_ac"]
    gia_lap = json.dumps({"arguments": {"state": "on", "temperature": gia_tri_teacher_tra}})
    args = parse_call_gen_response(gia_lap, fn)
    assert args["temperature"] == 22
    assert isinstance(args["temperature"], int)
    # quan trọng nhất: nhãn render ra phải giống hệt nhau dù teacher trả kiểu nào
    assert render_completion([("set_ac", args)]) == '<nexa_6>(state="on", temperature=22)<nexa_end>'


def test_parse_call_number_ve_float():
    fn = CATALOG.by_name["find_nearby"]
    args = parse_call_gen_response('{"arguments": {"category": "gas_station", "radius_km": "10"}}', fn)
    assert args["radius_km"] == 10.0
    assert isinstance(args["radius_km"], float)


def test_parse_call_gia_tri_khong_ep_kieu_duoc_thi_bao_loi():
    fn = CATALOG.by_name["set_ac"]
    with pytest.raises(TeacherResponseError, match="temperature"):
        parse_call_gen_response('{"arguments": {"state": "on", "temperature": "hai mươi hai"}}', fn)


def test_parse_call_bool_cho_tham_so_so_thi_bao_loi():
    # bool là subclass của int trong Python — dễ lọt nếu không chặn riêng
    fn = CATALOG.by_name["set_ac"]
    with pytest.raises(TeacherResponseError):
        parse_call_gen_response('{"arguments": {"state": "on", "temperature": true}}', fn)


def test_parse_call_bool_cho_tham_so_chuoi_thi_bao_loi():
    fn = CATALOG.by_name["set_ac"]
    with pytest.raises(TeacherResponseError):
        parse_call_gen_response('{"arguments": {"state": true}}', fn)


# ── ràng buộc enum/min/max — 3 vi phạm THẬT gặp khi chạy 50 câu/hàm ─────────
def test_parse_call_gia_tri_ngoai_enum_thi_loai_mau():
    """Đã gặp thật: teacher trả unit='m' (mét) trong khi enum chỉ có km/mi."""
    fn = CATALOG.by_name["get_odometer"]
    with pytest.raises(TeacherResponseError, match="enum"):
        parse_call_gen_response('{"arguments": {"unit": "m"}}', fn)


def test_parse_call_gop_2_gia_tri_enum_thanh_1_chuoi_thi_loai():
    """Đã gặp thật (7 mẫu): teacher gộp 'tolls, highways' khi người dùng muốn tránh cả hai."""
    fn = CATALOG.by_name["navigate_to"]
    with pytest.raises(TeacherResponseError, match="enum"):
        parse_call_gen_response(
            '{"arguments": {"destination": "Nội Bài", "avoid": "tolls, highways"}}', fn
        )


def test_parse_call_gia_tri_duoi_min_thi_loai_mau():
    """Đã gặp thật (8 mẫu): teacher trả day=-1 cho 'thời tiết hôm qua', schema chỉ cho 0..6.
    Phải loại CẢ MẪU — nếu chỉ bỏ `day` thì nhãn thành thời tiết HÔM NAY, sai ý câu hỏi."""
    fn = CATALOG.by_name["get_weather"]
    with pytest.raises(TeacherResponseError, match="min"):
        parse_call_gen_response('{"arguments": {"day": -1}}', fn)


def test_parse_call_gia_tri_tren_max_thi_loai_mau():
    fn = CATALOG.by_name["set_ac"]
    with pytest.raises(TeacherResponseError, match="max"):
        parse_call_gen_response('{"arguments": {"state": "on", "temperature": 99}}', fn)


def test_parse_call_gia_tri_dung_enum_va_trong_khoang_thi_giu():
    fn = CATALOG.by_name["get_weather"]
    assert parse_call_gen_response('{"arguments": {"day": 3}}', fn) == {"day": 3}


def test_parse_call_thieu_tham_so_bat_buoc_thi_loai_mau():
    """set_ac bắt buộc có `state` — thiếu thì lời gọi vô nghĩa (bật hay tắt?)."""
    fn = CATALOG.by_name["set_ac"]
    with pytest.raises(TeacherResponseError, match="bắt buộc"):
        parse_call_gen_response('{"arguments": {"temperature": 22}}', fn)


# ── 4. build/parse reply — GIẢ LẬP câu trả lời của teacher ──────────────────
def test_reply_prompt_co_chi_dan_ngon_ngu_va_khong_bia_so():
    prompt = build_reply_gen_prompt("còn bao nhiêu xăng?", "<nexa_0>()<nexa_end>", {"fuel_percent": 42})
    assert "tiếng Việt" in prompt
    assert "không bịa thêm" in prompt
    assert prompt.rstrip("\n").endswith("<|im_start|>assistant")


def test_parse_reply_cat_dung_doi_chatml():
    gia_lap = "Xăng còn 42%, đi được khoảng 176 km.<|im_end|>"
    assert parse_reply_gen_response(gia_lap) == "Xăng còn 42%, đi được khoảng 176 km."


def test_parse_reply_rong_bao_loi():
    with pytest.raises(TeacherResponseError):
        parse_reply_gen_response("   <|im_end|>")


# ── 5. build/parse verify — GIẢ LẬP câu trả lời của teacher ─────────────────
def test_parse_verify_hop_le():
    gia_lap = '{"valid": true, "reason": "đúng ý, đủ tham số"}'
    r = parse_verify_response(gia_lap)
    assert r.valid is True
    assert "đủ tham số" in r.reason


def test_parse_verify_khong_hop_le():
    gia_lap = '{"valid": false, "reason": "thiếu tham số zone"}'
    r = parse_verify_response(gia_lap)
    assert r.valid is False


def test_verify_prompt_khong_co_reply_thi_khong_hoi_ve_bia_so():
    prompt = build_verify_prompt("mở cốp xe", CATALOG.by_name["open_trunk"], {})
    assert "bịa số" not in prompt


def test_verify_prompt_co_reply_thi_hoi_ve_bia_so():
    prompt = build_verify_prompt(
        "còn bao nhiêu xăng?",
        CATALOG.by_name["get_fuel_level"],
        {},
        reply="Xăng còn 42%.",
        result={"fuel_percent": 42},
    )
    assert "bịa số" in prompt
    assert "42" in prompt


# ── 2 lỗi thật đã gặp khi chạy 50 câu/hàm — khoá lại để không tái phạm ──────
def test_verify_prompt_dung_ten_ham_doc_duoc_khong_dung_nexa_token():
    """LỖI ĐÃ GẶP: đưa `<nexa_0>()` cho teacher chấm -> teacher không hiểu token lạ,
    loại sạch mẫu của hàm không tham số (0-11% giữ lại, vs 88-100% ở hàm có tham số)."""
    prompt = build_verify_prompt("còn bao nhiêu xăng?", CATALOG.by_name["get_fuel_level"], {})
    assert "<nexa_" not in prompt, "không được lộ functional token cho teacher"
    assert "get_fuel_level()" in prompt, "phải dùng tên hàm đọc được"


def test_verify_prompt_noi_ro_ham_khong_tham_so_goi_rong_la_dung():
    """Chặn đúng nguyên nhân teacher loại mẫu hàm 0 tham số."""
    prompt = build_verify_prompt("mở cốp xe", CATALOG.by_name["open_trunk"], {})
    assert "không có tham số nào" in prompt
    assert "open_trunk()` cũng là ĐÚNG" in prompt


def test_verify_prompt_noi_ro_tham_so_tuy_chon_bo_trong_la_dung():
    """LỖI ĐÃ GẶP với giám khảo Llama: chê `get_odometer()` là 'thiếu tham số',
    trong khi `unit` vốn tuỳ chọn -> phải nói rõ bắt buộc/tuỳ chọn trong prompt."""
    prompt = build_verify_prompt("Xe đi được bao nhiêu km rồi?", CATALOG.by_name["get_odometer"], {})
    assert "unit (tuỳ chọn" in prompt
    assert "bỏ trống là ĐÚNG" in prompt


def test_verify_prompt_danh_dau_tham_so_bat_buoc():
    prompt = build_verify_prompt("bật điều hoà", CATALOG.by_name["set_ac"], {"state": "on"})
    assert "state (BẮT BUỘC)" in prompt
    assert "temperature (tuỳ chọn)" in prompt


def test_verify_ask_prompt_neu_ro_quy_tac_phan_biet():
    """Giám khảo khác họ không biết luật của dự án -> phải ghi thẳng vào prompt,
    không thì nó áp tiêu chuẩn riêng ('khát quá, cần vào quán cà phê' = ra lệnh thẳng)."""
    prompt = build_verify_ask_prompt("Khát nước quá", [("find_nearby", {"category": "coffee"})])
    assert "YÊU CẦU THẲNG" in prompt
    assert "MÔ TẢ TÌNH HUỐNG" in prompt
    assert "đừng dùng tiêu chuẩn riêng của bạn" in prompt


def test_verify_prompt_co_mo_ta_ham_de_teacher_biet_ham_lam_gi():
    prompt = build_verify_prompt("khoá xe lại", CATALOG.by_name["lock_doors"], {})
    assert "Khoá toàn bộ cửa xe" in prompt


def test_verify_negative_prompt_hoi_dung_cau_hoi():
    """LỖI ĐÃ GẶP: hỏi 'lời gọi hàm có đúng ý câu hỏi không' cho câu ngoài phạm vi
    -> irrelevant_function() không bao giờ 'trả lời' được câu hỏi -> loại 95%.
    Phải hỏi: 'câu này có THẬT SỰ ngoài phạm vi không?'"""
    prompt = build_verify_negative_prompt("thủ đô nước Pháp là gì?")
    assert "NGOÀI PHẠM VI" in prompt
    assert "thủ đô nước Pháp là gì?" in prompt
    assert "<nexa_" not in prompt
    # phải liệt kê phạm vi hỗ trợ để teacher có căn cứ đối chiếu
    assert "điều hoà" in prompt and "dẫn đường" in prompt


# ── 6. câu ngầm hiểu -> đề xuất rồi hỏi lại (<nexa_ask>) ────────────────────
def test_implicit_prompt_cam_dong_tu_ra_lenh_va_co_vi_du():
    fn = CATALOG.by_name["set_ac"]
    prompt = build_implicit_query_prompt(fn, n=50)
    assert "50 câu" in prompt
    assert "KHÔNG dùng động từ ra lệnh" in prompt
    assert fn.implicit_examples[0] in prompt
    assert "<nexa_" not in prompt


def test_ask_label_prompt_liet_ke_moi_ham_that_khong_lo_token():
    prompt = build_ask_label_prompt("Ôi nóng quá")
    assert '"Ôi nóng quá"' in prompt
    for f in CATALOG.functions:
        if f.name != "irrelevant_function":
            assert f"- {f.name}(" in prompt, f.name
    assert "- irrelevant_function(" not in prompt  # "ngoài phạm vi" là 1 loại riêng
    assert "<nexa_" not in prompt
    assert "state∈{on|off}" in prompt  # enum rút gọn để teacher điền đúng


def test_ask_label_prompt_du_ngan_cho_context_teacher():
    """Catalog đầy đủ phải vừa context 4096 token của teacher, chừa chỗ cho output.
    Tiếng Việt ~1 token / 2-3 ký tự -> giữ dưới 7000 ký tự là an toàn."""
    assert len(build_ask_label_prompt("x")) < 7000


def test_parse_ask_label_ngam_hieu_2_de_xuat():
    gia_lap = json.dumps(
        {
            "loai": "ngam_hieu",
            "de_xuat": [
                {"ten_ham": "set_ac", "tham_so": {"state": "on"}},
                {"ten_ham": "control_window", "tham_so": {"window": "all", "position": "down"}},
            ],
            "ly_do": "than bí, có thể bật điều hoà hoặc hạ kính",
        },
        ensure_ascii=False,
    )
    lab = parse_ask_label_response(gia_lap)
    assert lab.loai == "ngam_hieu"
    assert lab.calls == [
        ("set_ac", {"state": "on"}),
        ("control_window", {"window": "all", "position": "down"}),
    ]


def test_parse_ask_label_ngoai_pham_vi_khong_co_de_xuat():
    lab = parse_ask_label_response('{"loai": "ngoai_pham_vi", "de_xuat": [], "ly_do": "tán gẫu"}')
    assert lab.loai == "ngoai_pham_vi"
    assert lab.calls == []


def test_parse_ask_label_ro_rang_chi_giu_1_de_xuat():
    gia_lap = json.dumps(
        {
            "loai": "ro_rang",
            "de_xuat": [
                {"ten_ham": "control_window", "tham_so": {"window": "all", "position": "down"}},
                {"ten_ham": "set_ac", "tham_so": {"state": "on"}},
            ],
        }
    )
    assert len(parse_ask_label_response(gia_lap).calls) == 1


def test_parse_ask_label_cat_bot_khi_de_xuat_qua_2():
    de_xuat = [
        {"ten_ham": "set_ac", "tham_so": {"state": "on"}},
        {"ten_ham": "set_fan_speed", "tham_so": {"level": 5}},
        {"ten_ham": "control_window", "tham_so": {"window": "all", "position": "down"}},
    ]
    lab = parse_ask_label_response(json.dumps({"loai": "ngam_hieu", "de_xuat": de_xuat}))
    assert [n for n, _ in lab.calls] == ["set_ac", "set_fan_speed"]


def test_parse_ask_label_bo_rieng_de_xuat_hong_giu_de_xuat_tot():
    de_xuat = [
        {"ten_ham": "ham_bia", "tham_so": {}},  # tên lạ
        {"ten_ham": "set_ac", "tham_so": {"state": "on", "temperature": 99}},  # ngoài khoảng
        {"ten_ham": "irrelevant_function", "tham_so": {}},  # không phải đề xuất
        {"ten_ham": "control_window", "tham_so": {"window": "all", "position": "down"}},
    ]
    lab = parse_ask_label_response(json.dumps({"loai": "ngam_hieu", "de_xuat": de_xuat}))
    assert lab.calls == [("control_window", {"window": "all", "position": "down"})]


def test_parse_ask_label_ngam_hieu_khong_con_de_xuat_nao_thi_bao_loi():
    """Thiếu tham số bắt buộc (control_window cần window + position) -> không dùng được."""
    de_xuat = [{"ten_ham": "control_window", "tham_so": {"window": "all"}}]
    with pytest.raises(TeacherResponseError):
        parse_ask_label_response(json.dumps({"loai": "ngam_hieu", "de_xuat": de_xuat}))


def test_de_xuat_bo_tham_so_tuy_chon_khong_giu_so_teacher_bia():
    """LỖI ĐÃ GẶP ở đợt chạy thử: câu 'nóng quá' không nhắc số nào, teacher vẫn trả
    temperature=22, zone='all'. Giữ lại thì nhãn mâu thuẫn nhau và dạy model đoán bừa."""
    gia_lap = json.dumps(
        {
            "loai": "ngam_hieu",
            "de_xuat": [{"ten_ham": "set_ac", "tham_so": {"state": "on", "temperature": 22, "zone": "all"}}],
        }
    )
    lab = parse_ask_label_response(gia_lap)
    assert lab.calls == [("set_ac", {"state": "on"})]  # chỉ còn tham số BẮT BUỘC


def test_de_xuat_giu_tham_so_cau_da_noi_ro():
    """Quy ước sửa lại: bỏ tham số phải BỊA, nhưng GIỮ tham số câu nói rõ.
    LỖI ĐÃ GẶP: 'Kính sau đọng hơi' -> set_defrost(state='on') -> runtime sấy nhầm kính trước."""
    gia_lap = json.dumps(
        {"loai": "ngam_hieu", "de_xuat": [{"ten_ham": "set_defrost", "tham_so": {"state": "on"}}]}
    )
    lab = parse_ask_label_response(gia_lap, "Kính sau đọng hơi nước nhiều quá")
    assert lab.calls == [("set_defrost", {"state": "on", "zone": "rear"})]


def test_de_xuat_khong_noi_ro_thi_khong_them_tham_so():
    gia_lap = json.dumps(
        {"loai": "ngam_hieu", "de_xuat": [{"ten_ham": "set_defrost", "tham_so": {"state": "on"}}]}
    )
    assert parse_ask_label_response(gia_lap, "Chẳng nhìn thấy gì cả").calls == [
        ("set_defrost", {"state": "on"})
    ]


@pytest.mark.parametrize(
    "cau,mong_doi",
    [
        ("Kính trước mờ hết rồi", "front"),
        ("Kính hậu đọng hơi nước", "rear"),
        ("Cả hai kính đều mờ", "both"),
    ],
)
def test_tu_khoa_vi_tri_say_kinh(cau, mong_doi):
    assert tham_so_duoc_noi_ro("set_defrost", cau) == {"zone": mong_doi}


def test_de_xuat_sai_ngu_canh_bi_loai():
    """LỖI ĐÃ GẶP: 'Ngồi ở phía sau, kính mờ quá' -> operate_sunroof(action='open').
    Kính mờ là việc của sấy kính, mở cửa sổ trời không giải quyết gì."""
    assert de_xuat_hop_le("operate_sunroof", {"action": "open"}, "Ngồi ở phía sau, kính mờ quá") is False
    assert de_xuat_hop_le("set_defrost", {"state": "on"}, "Ngồi ở phía sau, kính mờ quá") is True


def test_de_xuat_sunroof_van_hop_le_khi_cau_khong_noi_ve_kinh_mo():
    assert de_xuat_hop_le("operate_sunroof", {"action": "open"}, "Muốn ngắm trời một chút") is True


def test_cau_ro_rang_van_giu_nguyen_tham_so_tuy_chon():
    """Luật bỏ tham số chỉ áp dụng cho câu mơ hồ — 'bật điều hoà 26 độ' phải giữ 26."""
    gia_lap = json.dumps(
        {
            "loai": "ro_rang",
            "de_xuat": [{"ten_ham": "set_ac", "tham_so": {"state": "on", "temperature": 26}}],
        }
    )
    assert parse_ask_label_response(gia_lap).calls == [("set_ac", {"state": "on", "temperature": 26})]


def test_de_xuat_khong_duoc_chua_ham_nhay_cam():
    """LỖI ĐÃ GẶP: 'bạn có thể mở cửa hàng khi tôi đang xem phim không' -> unlock_doors."""
    de_xuat = [
        {"ten_ham": "unlock_doors", "tham_so": {}},
        {"ten_ham": "open_trunk", "tham_so": {}},
        {"ten_ham": "set_ac", "tham_so": {"state": "on"}},
    ]
    lab = parse_ask_label_response(json.dumps({"loai": "ngam_hieu", "de_xuat": de_xuat}))
    assert [n for n, _ in lab.calls] == ["set_ac"]


def test_de_xuat_toan_ham_nhay_cam_thi_loai_ca_mau():
    de_xuat = [{"ten_ham": "unlock_doors", "tham_so": {}}]
    with pytest.raises(TeacherResponseError):
        parse_ask_label_response(json.dumps({"loai": "ngam_hieu", "de_xuat": de_xuat}))


def test_ask_label_prompt_uu_tien_mot_de_xuat():
    prompt = build_ask_label_prompt("nóng quá")
    assert "thường chỉ 1 chức năng" in prompt
    assert "HAI CÁCH" in prompt


def test_parse_ask_label_loai_la_thi_bao_loi():
    with pytest.raises(TeacherResponseError):
        parse_ask_label_response('{"loai": "mo_ho", "de_xuat": []}')


def test_verify_ask_prompt_dung_ten_doc_duoc_va_hoi_dung_y():
    prompt = build_verify_ask_prompt("Trời bắt đầu sập tối rồi", [("set_headlights", {"mode": "auto"})])
    assert 'set_headlights(mode="auto")' in prompt
    assert "<nexa_" not in prompt
    assert "HỎI LẠI" in prompt
    assert "hỏi lại là SAI" in prompt  # câu ra lệnh thẳng thì không được hỏi lại
