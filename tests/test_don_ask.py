"""Test áp chính sách đề xuất lên mẫu ask đã sinh (gen/don_ask.py)."""

from gen.don_ask import don_row, don_rows
from gen.records import CallRecord, SuggestRecord
from gen.teacher_prompts import ap_dung_chinh_sach_de_xuat
from schema.call_format import parse_calls


def test_de_xuat_pha_xa_bi_ha_xuong_chieu_gan():
    """97/123 mẫu đèn trong đợt sinh thật gợi ý high_beam cho câu 'trời tối quá'."""
    row = SuggestRecord(
        "Trời tối quá, không nhìn rõ đường", (("set_headlights", {"mode": "high_beam"}),)
    ).to_row()
    moi, doi = don_row(row)
    assert doi is True
    (call,) = parse_calls(moi["completion"])
    assert call.args == {"mode": "low_beam"}
    assert call.suggested is True


def test_cau_nguoi_dung_khong_bi_doi():
    row = SuggestRecord("Tối om om, không thấy gì", (("set_headlights", {"mode": "high_beam"}),)).to_row()
    moi, _ = don_row(row)
    assert moi["prompt"] == row["prompt"]
    assert moi["meta"] == row["meta"]


def test_de_xuat_khac_giu_nguyen():
    row = SuggestRecord("Nóng quá", (("set_ac", {"state": "on"}),)).to_row()
    moi, doi = don_row(row)
    assert doi is False
    assert moi["completion"] == row["completion"]


def test_giu_khoa_phu_nhu_nguon():
    row = {**SuggestRecord("Nóng quá", (("set_ac", {"state": "on"}),)).to_row(), "nguon": "sinh_moi"}
    moi, _ = don_row(row)
    assert moi["nguon"] == "sinh_moi"


def test_de_xuat_2_ham_van_don_duoc():
    row = SuggestRecord(
        "Trời tối và bí quá",
        (
            ("set_headlights", {"mode": "high_beam"}),
            ("control_window", {"window": "all", "position": "down"}),
        ),
    ).to_row()
    moi, doi = don_row(row)
    assert doi is True
    calls = parse_calls(moi["completion"])
    assert [c.name for c in calls] == ["set_headlights", "control_window"]
    assert calls[0].args == {"mode": "low_beam"}
    assert calls[1].args == {"window": "all", "position": "down"}


def test_dem_dung_so_nhan_da_sua():
    rows = [
        SuggestRecord("tối quá", (("set_headlights", {"mode": "high_beam"}),)).to_row(),
        SuggestRecord("nóng quá", (("set_ac", {"state": "on"}),)).to_row(),
    ]
    _, n_doi, n_bo = don_rows(rows)
    assert n_doi == 1
    assert n_bo == 0


def test_de_xuat_vo_ly_thi_bo_ca_mau():
    """Đã gặp thật: 'Nắng nhẹ nhàng lắm' -> set_headlights(mode='off')."""
    rows = [SuggestRecord("Nắng nhẹ nhàng lắm", (("set_headlights", {"mode": "off"}),)).to_row()]
    con, _, n_bo = don_rows(rows)
    assert con == []
    assert n_bo == 1


def test_chinh_sach_khong_dung_cho_lenh_ro_rang():
    """Câu 'bật pha xa lên' là lệnh rõ ràng -> nhãn call thường, không qua chính sách này."""
    row = CallRecord("Bật đèn pha xa lên", "set_headlights", {"mode": "high_beam"}).to_row()
    assert 'mode="high_beam"' in row["completion"]
    # hàm chính sách chỉ được gọi ở nhánh đề xuất
    assert ap_dung_chinh_sach_de_xuat("set_headlights", {"mode": "high_beam"}) == {"mode": "low_beam"}
