"""`train/grpo.py` — điểm thưởng (khoá trước khi chạy) + chọn câu luyện. Thuần Python, không cần GPU / TRL."""

import pytest

from train.grpo import CauHinhChon, chon_cau, chon_kho, diem_thuong, ha_chu_dau, loai_ket_cuc, ung_vien

MO_COP = "<nexa_11>()<nexa_end><|im_end|>"
TU_CHOI = "<nexa_24>()<nexa_end><|im_end|>"
HOI_CUA_SO_TROI = '<nexa_ask><nexa_16>(action="open")<nexa_end><|im_end|>'
AC_22 = '<nexa_6>(state="on", temperature=22)<nexa_end><|im_end|>'


# phần thừa sau <nexa_end> bị cắt · gọi bừa ngoài phạm vi / làm luôn câu đáng hỏi / sai tham số khi làm = −5
# · đề xuất (hỏi lại) sai chưa đụng xe = −0,5
@pytest.mark.parametrize("output, dap_an, loai, diem", [
    ("<nexa_11>()<nexa_end>", MO_COP, "dung", 1.0),
    ("<nexa_11>()<nexa_end>\n\nMô tả hàm: …", MO_COP, "dung", 1.0),
    ("<nexa_ask><nexa_11>()<nexa_end>", MO_COP, "hoi_lai_thua", -0.2),
    ("<nexa_24>()<nexa_end>", MO_COP, "tu_choi_nham", -0.5),
    ("<nexa_11>()<nexa_end>", TU_CHOI, "lam_sai_len_xe", -5.0),
    ('<nexa_16>(action="open")<nexa_end>', HOI_CUA_SO_TROI, "lam_sai_len_xe", -5.0),
    ('<nexa_6>(state="on", temperature=25)<nexa_end>', AC_22, "lam_sai_len_xe", -5.0),
    ("<nexa_ask><nexa_7>(delta=-1)<nexa_end>", HOI_CUA_SO_TROI, "de_xuat_sai", -0.5),
    ("<nexa_ask><nexa_16>(action=\"open\")<nexa_end>", TU_CHOI, "de_xuat_sai", -0.5),
    ("rác không phải lời gọi", MO_COP, "khong_parse", -1.0),
])
def test_diem_thuong(output, dap_an, loai, diem):
    assert loai_ket_cuc(output, dap_an) == loai and diem_thuong(output, dap_an) == diem


def test_thu_tu_phat_theo_hau_qua():
    # làm sai lên xe < hỏng < từ chối nhầm = đề xuất sai < hỏi lại thừa < đúng
    from train.grpo import THUONG as t

    assert t["lam_sai_len_xe"] < t["khong_parse"] < t["tu_choi_nham"] <= t["de_xuat_sai"]
    assert t["de_xuat_sai"] < t["hoi_lai_thua"] < t["dung"]


_SYS = "<|im_start|>system\nBạn là tác tử điều khiển xe ô tô.<|im_end|>\n<|im_start|>user\n"
P1 = _SYS + "Mở cốp<|im_end|>\n<|im_start|>assistant\n"
P2 = _SYS + "Lượt trước: <nexa_11>()\nĐóng đi<|im_end|>\n<|im_start|>assistant\n"


def test_ha_chu_dau_chi_doi_cau_nguoi_dung():
    assert "user\nmở cốp<|im_end|>" in ha_chu_dau(P1)
    assert "Lượt trước: <nexa_11>()\nđóng đi<|im_end|>" in ha_chu_dau(P2)


def _row(prompt, completion):
    return {"prompt": prompt, "completion": completion, "meta": {}}


def test_chon_cau_theo_ty_le_bo_reply_va_tat_dinh():
    rows = ([_row(P1, MO_COP)] * 400 + [_row(P2, MO_COP)] * 100 + [_row(P1, TU_CHOI)] * 400
            + [_row(P1, HOI_CUA_SO_TROI)] * 400 + [_row(P1, "Đã mở cốp sau.<|im_end|>")] * 50)
    cfg = CauHinhChon(so_cau=300)
    a, b = chon_cau(rows, cfg), chon_cau(rows, cfg)
    assert a == b and len(a) == 300
    dem = {n: sum(c["nhom"] == n for c in a) for n in ("mo_ho", "ngoai_pham_vi", "ro_rang")}
    assert dem == {"mo_ho": 105, "ngoai_pham_vi": 90, "ro_rang": 105}
    assert sum(c["hai_luot"] for c in a if c["nhom"] == "ro_rang") == 26  # ≥ 25% câu hai lượt khi có đủ
    assert 0.1 < sum(c["chu_thuong"] for c in a) / 300 < 0.4
    assert all(not c["dap_an"].startswith("Đã") for c in a)  # mẫu lượt 2 bị bỏ


def test_ung_vien_them_ban_chu_thuong_va_bo_reply():
    rows = [_row(P1, MO_COP), _row(P2, MO_COP), _row(P1, "Đã mở cốp sau.<|im_end|>")]
    uv = ung_vien(rows)
    assert len(uv) == 4 and sum(u["chu_thuong"] for u in uv) == 2
    assert {u["prompt"] for u in uv if u["chu_thuong"]} == {ha_chu_dau(P1), ha_chu_dau(P2)}


def test_chon_kho_giu_cau_phan_van_cong_it_cau_chac():
    ds = [{"i": i} for i in range(10)]
    p = [0.5, 0.99, 0.6, 0.99, 0.94, 0.99, 0.99, 0.99, 0.7, 0.99]
    chon = chon_kho(ds, p, so_kho=3, so_de=2, nguong=0.95, seed=1)
    kho = {d["i"] for d in chon if p[d["i"]] < 0.95}
    assert len(chon) == 5 and len(kho) == 3 and kho <= {0, 2, 4, 8}
    assert chon == chon_kho(ds, p, so_kho=3, so_de=2, nguong=0.95, seed=1)
