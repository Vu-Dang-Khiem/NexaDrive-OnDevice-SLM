"""Bước 4 (PLAN.md mục W): cổng model GGUF thật, luật ngưỡng 0,80, "bật sưởi", câu hỏi lại kèm trạng thái xe,
lọc tiếng Anh ở lượt 2. Không cần llama.cpp: tokenizer + máy chủ đều giả."""

import math

import pytest

from schema.call_format import parse_calls, render_completion
from schema.loader import CATALOG
from serve.contracts import AssistantTurn, ClarifyTurn
from serve.kiem_tra import kiem_tra
from serve.mo_hinh import MoHinhGia
from serve.mo_hinh_gguf import MoHinhGGUF, ap_nguong
from serve.tac_tu import TacTu
from serve.tra_loi import canh_bao_khi_chay, hanh_dong

TU_CHOI = CATALOG.by_name["irrelevant_function"].nexa_token
AC = CATALOG.by_name["set_ac"].nexa_token
KINH = CATALOG.by_name["control_window"].nexa_token
END, ASK = CATALOG.nexa_end, CATALOG.nexa_ask
OUT_KINH = f'{KINH}(window="all", position="up"){END}'


# ── luật ngưỡng (hàm thuần) ───────────────────────────────────────────────
def test_id_nexa_end_dung_thu_tu_token_moi():
    # 27 token mới thêm sau 151665 token gốc của Qwen2.5 theo thứ tự CATALOG.special_tokens
    # (train/add_tokens.py); llama-server dùng id này làm token KẾT THÚC (serve/mo_hinh_gguf.MayChuLlama)
    from serve.mo_hinh_gguf import ID_NEXA_END

    assert ID_NEXA_END == 151665 + CATALOG.special_tokens.index(CATALOG.nexa_end)


def test_tu_tin_cao_giu_nguyen():
    assert ap_nguong(OUT_KINH, [(KINH, 0.95), (TU_CHOI, 0.03)], 0.80) == (OUT_KINH, "giu")


def test_duoi_nguong_hang_2_tu_choi_thi_tu_choi():
    out, ly_do = ap_nguong(OUT_KINH, [(KINH, 0.60), (TU_CHOI, 0.35)], 0.80)
    assert parse_calls(out)[0].name == "irrelevant_function" and ly_do == "nguong_tu_choi"


def test_duoi_nguong_hang_2_khac_thi_hoi_lai_DU_THAM_SO():
    out, ly_do = ap_nguong(OUT_KINH, [(KINH, 0.60), (AC, 0.30)], 0.80)
    c = parse_calls(out)
    assert ly_do == "nguong_hoi_lai" and c[0].suggested
    assert (c[0].name, dict(c[0].args)) == ("control_window", {"window": "all", "position": "up"})


def test_token_hang_1_la_tu_choi_hay_hoi_lai_thi_lam_theo_model():
    tc = render_completion([("irrelevant_function", {})])
    assert ap_nguong(tc, [(TU_CHOI, 0.5), (KINH, 0.4)], 0.80)[0] == tc
    hl = render_completion([("set_ac", {"state": "on"})], ask=True)
    assert ap_nguong(hl, [(ASK, 0.5), (AC, 0.4)], 0.80)[0] == hl


def test_output_hong_duoi_nguong_de_router_bao_chua_hieu():
    assert ap_nguong("rác", [(KINH, 0.5), (AC, 0.4)], 0.80) == ("rác", "giu")


def test_duoi_nguong_tham_so_la_thi_bo_khong_vo():
    """Đo độ chịu ồn 2026-10-08: model sinh `find_nearby(query=…)` (tham số không có trong schema) dưới
    ngưỡng -> render câu hỏi lại từng ném KeyError làm hỏng cả lượt."""
    tim = CATALOG.by_name["find_nearby"].nexa_token
    vao = f'{tim}(category="gas_station", query="cây xăng"){END}'
    out, ly_do = ap_nguong(vao, [(tim, 0.5), (AC, 0.4)], 0.80)
    c = parse_calls(out)
    assert ly_do == "nguong_hoi_lai" and c[0].suggested and dict(c[0].args) == {"category": "gas_station"}


# ── MoHinhGGUF với tokenizer + máy chủ giả ───────────────────────────────
class TokGia:
    BANG = {1: KINH, 2: '(window="all", position="up")', 3: END, 4: "<|im_end|>", 5: TU_CHOI, 6: AC,
            7: "Đã nâng hết kính.", 8: "()"}

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [len(text)]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.BANG[i] for i in ids)

    def convert_ids_to_tokens(self, i):
        return self.BANG[i]


def _phan_hoi(ids, top):
    return {"tokens": ids, "completion_probabilities": [
        {"id": top[0][0], "logprob": math.log(top[0][1]),
         "top_logprobs": [{"id": i, "logprob": math.log(p)} for i, p in top]}]}


def test_goi_ham_gui_id_token_va_ap_nguong():
    gui = []

    def http(body):
        gui.append(body)
        return _phan_hoi([1, 2, 3, 4], [(1, 0.62), (6, 0.30)])

    m = MoHinhGGUF(goi_http=http, tokenizer=TokGia(), nguong=0.80)
    out = m.goi_ham("Gió lùa quá")
    assert parse_calls(out)[0].suggested and m.lan_cuoi.ly_do == "nguong_hoi_lai"
    assert abs(m.lan_cuoi.p - 0.62) < 1e-9 and m.lan_cuoi.output_goc == OUT_KINH
    assert isinstance(gui[0]["prompt"], list) and gui[0]["cache_prompt"] is False and gui[0]["n_probs"] == 5


def test_tra_loi_cat_o_im_end():
    m = MoHinhGGUF(goi_http=lambda b: {"tokens": [7, 4, 7]}, tokenizer=TokGia())
    assert m.tra_loi("Đóng kính", OUT_KINH, {"window": "all"}) == "Đã nâng hết kính."


# ── router: "bật sưởi", trạng thái xe, lọc tiếng Anh ─────────────────────
def test_than_lanh_khong_nhac_dieu_hoa_hoi_bat_suoi_roi_bao_bat_suoi():
    de_xuat = render_completion([("set_ac", {"state": "on"})], ask=True)
    tt = TacTu(MoHinhGia(bang={"Trong xe lạnh quá": de_xuat}, cau_tra_loi={"set_ac": "Đã bật điều hoà."}),
               luot_2="model")  # kể cả khi bật model lượt 2, câu than lạnh vẫn báo "sưởi"
    hoi = tt.xu_ly("Trong xe lạnh quá")[0]
    assert isinstance(hoi, ClarifyTurn) and hoi.text == "Bạn có muốn mình bật sưởi cho ấm không?"
    tl = tt.xu_ly("có")[0]
    assert isinstance(tl, AssistantTurn) and "sưởi" in tl.text and tt.xe.s.ac_on


def test_than_lanh_co_nhac_dieu_hoa_van_noi_tat_dieu_hoa():
    de_xuat = render_completion([("set_ac", {"state": "off"})], ask=True)
    tt = TacTu(MoHinhGia(bang={"Điều hoà lạnh buốt": de_xuat}))
    assert "tắt điều hoà" in tt.xu_ly("Điều hoà lạnh buốt")[0].text


def test_cau_hoi_lai_kem_trang_thai_kinh():
    tt = TacTu(MoHinhGia(bang={"Gió lùa quá": render_completion(
        [("control_window", {"window": "all", "position": "up"})], ask=True)}))
    tt.xe.s.windows["front_right"] = 50
    hoi = tt.xu_ly("Gió lùa quá")[0]
    assert hoi.text.startswith("Đang mở: kính ghế phụ 50%.") and "nâng tất cả kính" in hoi.text


def test_luot_2_lan_tieng_anh_thi_dung_cau_mau():
    tt = TacTu(MoHinhGia(cau_tra_loi={"get_fuel_level": "Còn 62% xăng, không cần refuel ngay."}),
               luot_2="model")
    tl = tt.xu_ly("còn bao nhiêu xăng")[0]
    assert "refuel" not in tl.text and tl.text.startswith("Xăng còn")


def test_luot_2_don_vi_do_khong_bi_coi_la_tieng_anh():
    kiem = TacTu._cau_dung_duoc
    assert kiem("Nhiệt độ khoảng 24°C, mưa 13 mm.", {"temperature": 24, "rain_mm": 13}, "thời tiết")
    assert kiem("Lốp trước trái 31 psi.", {"psi": 31}, "lốp")
    assert not kiem("Đang phát nhạc. Enjoy!", {"now_playing": "nhạc"}, "mở nhạc")


@pytest.mark.parametrize("cau, ra_model, dau, sau", [
    ("giảm nhiệt độ điều hòa", render_completion([("set_ac", {"state": "off"})]), 24, 22),  # model đoán TẮT
    ("hạ nhiệt độ điều hoà xuống", render_completion([("set_ac", {"state": "on"})]), 24, 22),
    # model không chắc -> từ chối
    ("chỉnh điều hòa mát hơn", render_completion([("irrelevant_function", {})]), 22, 20),
    ("tăng nhiệt độ điều hòa", render_completion([("set_ac", {"state": "on"})]), 24, 26),
    ("cho ấm hơn chút", render_completion([("set_ac", {"state": "on"})], ask=True), 29, 30),  # chạm trần 30
])
def test_chinh_nhiet_tuong_doi_khong_so(cau, ra_model, dau, sau):
    tt = TacTu(MoHinhGia(bang={cau: ra_model}))
    tt.xe.s.ac_temperature = dau
    tl = tt.xu_ly(cau)[0]
    assert tt.xe.s.ac_on and tt.xe.s.ac_temperature == sau
    assert str(sau) in tl.text and ("giảm" in tl.text) == (sau < dau)


def test_chinh_nhiet_co_so_thi_de_model_lam():
    cau = "giảm nhiệt độ điều hòa xuống 20 độ"
    tt = TacTu(MoHinhGia(bang={cau: render_completion([("set_ac", {"state": "on", "temperature": 20})])}))
    tt.xu_ly(cau)
    assert tt.xe.s.ac_temperature == 20


def test_model_dua_delta_thi_luat_khong_de_len():
    cau = "giảm nhiệt độ điều hòa"
    tt = TacTu(MoHinhGia(bang={cau: render_completion([("set_ac", {"state": "on", "delta": -3})])}))
    tl = tt.xu_ly(cau)[0]
    # câu không nói số -> quy ước gán nhãn ±2 áp lại lúc chạy (serve/quy_uoc.py): model ra −3 -> −2, 24 -> 22
    assert tt.xe.s.ac_temperature == 22 and tl.text == "Đã giảm nhiệt độ điều hoà xuống 22 độ."


def test_delta_kep_trong_16_30_va_hoi_lai_noi_ro_muc():
    tt = TacTu(MoHinhGia(bang={"tăng điều hoà thêm 10 độ": render_completion(
        [("set_ac", {"state": "on", "delta": 10})])}))
    tt.xe.s.ac_temperature = 26
    tl = tt.xu_ly("tăng điều hoà thêm 10 độ")[0]
    assert tt.xe.s.ac_temperature == 30 and tl.text.endswith("30 độ (cao nhất).")
    assert hanh_dong("set_ac", {"state": "on", "delta": -2}) == "giảm nhiệt độ điều hoà 2 độ"


def test_set_ac_khong_cho_ca_temperature_lan_delta():
    assert not kiem_tra("set_ac", {"state": "on", "temperature": 20, "delta": -2}).hop_le
    assert kiem_tra("set_ac", {"state": "on", "delta": -2}).hop_le
    assert not kiem_tra("set_ac", {"state": "on", "delta": -20}).hop_le  # ngoài −14…14


@pytest.mark.parametrize("cau", ["bật điều hòa lạnh", "bật điều hoà lạnh", "bật điều hoa lạnh"])
def test_bat_dieu_hoa_lanh_khong_bi_noi_la_bat_suoi(cau):
    # "hòa" (dấu trên o) từng không khớp luật nhắc điều hoà -> báo "Đã bật sưởi" (người dùng gặp 2026-10-06)
    tt = TacTu(MoHinhGia(bang={cau: render_completion([("set_ac", {"state": "on"})])}))
    tl = tt.xu_ly(cau)[0]
    assert "sưởi" not in tl.text and tl.text.startswith("Đã bật điều hoà")


@pytest.mark.parametrize("cau, de_xuat", [
    ("Nhạc trong xe to vãi", "giảm âm lượng 3 mức"),
    ("Chẳng nghe rõ ca sĩ hát gì cả", "tăng âm lượng 3 mức"),
])
def test_than_am_luong_khong_muc_thi_de_xuat_theo_huong(cau, de_xuat):
    tt = TacTu(MoHinhGia(bang={cau: render_completion([("set_volume", {})], ask=True)}))
    tt.xe.s.volume = 20
    hoi = tt.xu_ly(cau)[0]
    assert isinstance(hoi, ClarifyTurn) and de_xuat in hoi.text
    tt.xu_ly("có")
    assert tt.xe.s.volume == (17 if "giảm" in de_xuat else 23)


def test_than_am_luong_khong_ro_huong_van_bao_chua_ro():
    tt = TacTu(MoHinhGia(bang={"âm thanh kỳ ghê": render_completion([("set_volume", {})], ask=True)}))
    assert tt.xu_ly("âm thanh kỳ ghê")[0].text == "Bạn muốn mình làm gì cụ thể ạ?"


def test_canh_bao_khi_chuyen_sang_chay_ma_cop_dang_mo():
    tt = TacTu(MoHinhGia())
    assert canh_bao_khi_chay(tt.xe.s) == ""
    tt.xe.s.trunk_open, tt.xe.s.doors_open = True, ["driver"]
    assert canh_bao_khi_chay(tt.xe.s).startswith("Cảnh báo: cốp sau, cửa ghế lái đang mở.")


def test_cau_khac_khong_bi_luat_nhiet_dong_vao():
    tt = TacTu(MoHinhGia())
    tt.xu_ly("giảm âm lượng")
    assert tt.xe.s.ac_on is False


def test_mo_hinh_gia_nhan_ca_hai_cach_bo_dau_dieu_hoa():
    # người dùng gõ "điều hòa" (dấu trên o) bị model giả từ chối vì luật chỉ nhận "điều hoà" (2026-10-06)
    for cau in ("bật điều hòa 22 độ", "bật điều hoà 22 độ", "bật điều hoa 22 độ"):
        assert parse_calls(MoHinhGia().goi_ham(cau))[0].name == "set_ac"


def test_luot_2_tu_tieng_anh_co_trong_ket_qua_van_giu():
    tt = TacTu(MoHinhGia(cau_tra_loi={"play_media": "Đang phát Shape of You cho bạn."}), luot_2="model")
    tl = tt.xu_ly("mở nhạc Shape of You")[0]
    assert tl.text == "Đang phát Shape of You cho bạn."
