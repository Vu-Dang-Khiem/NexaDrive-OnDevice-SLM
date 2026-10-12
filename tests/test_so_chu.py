"""`serve/so_chu.py`: STT viết số bằng chữ -> chữ số như câu train (PLAN.md Phase 6)."""

import pytest

from serve.so_chu import doc_cum, doi_so


@pytest.mark.parametrize("cum, so", [
    ("hai mươi hai", 22), ("hai mươi", 20), ("mười lăm", 15), ("mười", 10), ("hai mươi mốt", 21),
    ("hai mươi tư", 24), ("năm mươi", 50), ("một trăm", 100), ("một trăm linh năm", 105),
    ("hai trăm sáu mươi", 260), ("ba nghìn", 3000), ("hai nghìn không trăm hai mươi sáu", 2026),
    ("chín mươi chín", 99),
])
def test_doc_cum(cum, so):
    assert doc_cum(cum.split()) == so


@pytest.mark.parametrize("vao, ra", [
    ("Bật điều hoà hai mươi hai độ cho ghế lái", "Bật điều hoà 22 độ cho ghế lái"),
    ("Hạ kính ghế phụ năm mươi phần trăm", "Hạ kính ghế phụ 50%"),
    ("Chỉnh quạt lên mức ba", "Chỉnh quạt lên mức 3"),
    ("Còn bảy phẩy ba ki lô mét nữa", "Còn 7,3 km nữa"),
    ("Đi được khoảng hai trăm sáu mươi phẩy bốn ki lô mét.", "Đi được khoảng 260,4 km."),
    ("Mất chừng mười hai phút.", "Mất chừng 12 phút."),
    ("Tăng âm lượng thêm hai nấc", "Tăng âm lượng thêm 2 nấc"),
    ("Nhiệt độ hai mươi lăm độ nhé", "Nhiệt độ 25 độ nhé"),
])
def test_doi_so_cau_lenh(vao, ra):
    assert doi_so(vao) == ra


@pytest.mark.parametrize("cau", [
    "Năm nay trời nóng quá",          # "năm" = năm tháng
    "Gọi cho ba tôi",                   # "ba" = bố
    "Hạ kính xuống một chút",           # "một chút"
    "Mở cốp sau giúp mình",
    "Bạn có muốn tôi bật sưởi cho ấm không?",
])
def test_khong_doi_tu_thuong(cau):
    assert doi_so(cau) == cau


@pytest.mark.parametrize("vao, ra", [
    ("chạy thêm năm mươi km nữa có đủ không?", "Chạy thêm 50 km nữa có đủ không"),
    ("Kiểm tra xem có cửa nào chưa đóng chặt không.", "Kiểm tra xem có cửa nào chưa đóng chặt không"),
    ("mở cốp sau giúp mình...", "Mở cốp sau giúp mình"),
    ("Đang mở: kính ghế phụ", "Đang mở: kính ghế phụ"),
])
def test_chuan_cau_stt(vao, ra):
    from serve.so_chu import chuan_cau_stt

    assert chuan_cau_stt(vao) == ra


# ── lỗi tìm thấy khi soi bộ B giọng thật (2026-10-07, PLAN.md Phase 6 "KẾT QUẢ BỘ B") ──────────────────
@pytest.mark.parametrize("vao, ra", [
    ("Bật điều hòa hai hai độ hộ cái", "Bật điều hòa 22 độ hộ cái"),     # nói tắt 22 — từng ra 4
    ("Chỉnh điều hòa ghế phụ hai năm độ", "Chỉnh điều hòa ghế phụ 25 độ"),  # từng ra 7
    ("Bật điều hòa hai lăm độ", "Bật điều hòa 25 độ"),
    ("Đặt nhiệt độ hai mốt độ", "Đặt nhiệt độ 21 độ"),
    ("Quạt lên mức ba tư", "Quạt lên mức 34"),
])
def test_so_noi_tat_hai_chu_so(vao, ra):
    assert doi_so(vao) == ra


@pytest.mark.parametrize("cau", [
    "Chỉ đường tới Vincom bà Triệu tránh đường cao tốc",  # tên đường — từng ra "bà 1000000"
    "Không biết pin có đủ chạy về không",                 # "không" phủ định — từng ra "về 0"
    "Giảm nhiệt độ xuống không",
    "Đi hai ba ngày nữa",                                  # hai chữ số rời không cạnh đơn vị: giữ nguyên
])
def test_khong_doi_ten_rieng_va_tu_phu_dinh(cau):
    assert doi_so(cau) == cau


@pytest.mark.parametrize("vao, ra", [
    ("Nhiệt độ hai mươi lăm độ không?", "Nhiệt độ 25 độ không?"),
    ("Còn hai mươi không?", "Còn 20 không?"),         # "không" cuối câu là tiểu từ hỏi, không nhập vào số
    ("Quạt mức không", "Quạt mức 0"),
    ("Tra đường về quận một", "Tra đường về quận 1"),
    # bản sửa đầu (luật "trăm / triệu đứng đầu = tên riêng" quá rộng) làm hỏng câu này
    ("Chạy về quê chắc phải hơn trăm cây số", "Chạy về quê chắc phải hơn 100 km"),
])
def test_khong_cuoi_cau_va_so_sau_quan(vao, ra):
    assert doi_so(vao) == ra


@pytest.mark.parametrize("cum", ["hai hai", "hai mươi trăm", "mười lăm hai"])
def test_doc_cum_khong_hop_le(cum):
    assert doc_cum(cum.split()) is None


def test_giu_nguyen_chu_so_san_co():
    assert doi_so("Bật điều hoà 22 độ") == "Bật điều hoà 22 độ"
