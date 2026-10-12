"""Test gộp/chia tập theo nhóm câu gần trùng (gen/chia_tap.py)."""

import random

from gen.chia_tap import chia_theo_nhom, chuan_hoa, nhom_gan_trung
from gen.records import CallRecord, SuggestRecord, negative_record


def test_chuan_hoa_bo_dau_cau_va_chu_hoa():
    assert chuan_hoa("Mở cốp xe!") == chuan_hoa("mở cốp   xe")


def test_cau_gan_trung_vao_cung_nhom():
    """Đúng kiểu trùng lặp có thật trong data negative."""
    cau = [
        "tự lái xe đến nhà tôi",
        "tự lái xe đến nhà tôi đi",
        "Tự lái xe đến nhà tôi.",
        "thủ đô nước Pháp là gì",
    ]
    ma = nhom_gan_trung(cau)
    assert ma[0] == ma[1] == ma[2]
    assert ma[3] != ma[0]


def test_cau_khac_han_thi_khac_nhom():
    ma = nhom_gan_trung(["mở cốp xe", "bật điều hoà 22 độ ghế lái", "thời tiết hôm nay thế nào"])
    assert len(set(ma)) == 3


_TU = (
    "hôm nay trời mưa nắng gió bạn tôi anh chị đi về nhà quán xe máy bay tàu thuyền "
    "cơm phở bánh mì cà phê trà sữa sách vở bút mực bàn ghế cửa sổ điện thoại đồng hồ "
    "chó mèo chim cá hoa lá cành núi sông biển đảo phố huyện tỉnh làng xóm chợ trường"
).split()


def _cau_khac_nhau(n: int, seed: int) -> list[str]:
    """Sinh n câu KHÁC HẲN nhau (không gần trùng) để test phần chia tập."""
    rng = random.Random(seed)
    ket_qua, da_co = [], set()
    while len(ket_qua) < n:
        cau = " ".join(rng.sample(_TU, 6))
        if cau not in da_co:
            da_co.add(cau)
            ket_qua.append(cau)
    return ket_qua


def _kho() -> list[dict]:
    rows = []
    for i, cau in enumerate(_cau_khac_nhau(120, seed=11)):
        if i % 3 == 0:
            rows.append(CallRecord(cau, "open_trunk", {}).to_row())
        elif i % 3 == 1:
            rows.append(CallRecord(cau, "set_ac", {"state": "on"}).to_row())
        else:
            rows.append(negative_record(cau).to_row())
    return rows


def test_chia_khong_lam_mat_mau():
    train, val = chia_theo_nhom(_kho(), ty_le_val=0.2, seed=1)
    assert len(train) + len(val) == 120
    assert val  # có mẫu val thật sự


def test_ca_nhom_gan_trung_nam_cung_mot_ben():
    """Ràng buộc quan trọng nhất: không được để 1 câu ở train, câu gần trùng ở val."""
    bien_the = [
        "tự lái xe đến nhà tôi",
        "tự lái xe đến nhà tôi đi",
        "tự lái xe đến nhà tôi nhé",
        "Tự lái xe đến nhà tôi!",
    ]
    rows = [negative_record(c).to_row() for c in bien_the]
    rows += [negative_record(c).to_row() for c in _cau_khac_nhau(30, seed=3)]
    train, val = chia_theo_nhom(rows, ty_le_val=0.3, seed=7)

    def co(tap, cau):
        return any(cau in r["prompt"] for r in tap)

    ben_train = [c for c in bien_the if co(train, c)]
    ben_val = [c for c in bien_the if co(val, c)]
    assert not (ben_train and ben_val), f"nhóm bị xé đôi: train={ben_train}, val={ben_val}"


def test_val_co_du_moi_ham():
    train, val = chia_theo_nhom(_kho(), ty_le_val=0.2, seed=1)
    ham_val = {r["meta"]["fn"] for r in val}
    assert {"open_trunk", "set_ac", "irrelevant_function"} <= ham_val


def test_cung_seed_ra_cung_ket_qua():
    a = chia_theo_nhom(_kho(), seed=5)[1]
    b = chia_theo_nhom(_kho(), seed=5)[1]
    assert [r["prompt"] for r in a] == [r["prompt"] for r in b]


def test_mau_ask_cung_duoc_chia():
    rows = _kho() + [
        SuggestRecord(c, (("set_ac", {"state": "on"}),)).to_row() for c in _cau_khac_nhau(20, seed=5)
    ]
    train, val = chia_theo_nhom(rows, ty_le_val=0.2, seed=1)
    assert any(r["meta"]["type"] == "ask" for r in train)
    assert any(r["meta"]["type"] == "ask" for r in val)


def test_gom_nhom_giong_het_nhau_giua_cac_lan_chay():
    """hash() của Python bị xáo theo PYTHONHASHSEED -> chia tập từng không tái lập được."""
    import os
    import subprocess
    import sys

    ma = (
        "from gen.chia_tap import nhom_gan_trung;"
        "c=['mở cốp xe giúp tôi','mở cốp xe giúp tôi với','bật điều hoà','bật điều hoà lên nhé',"
        "'kể chuyện cười đi','kể một câu chuyện cười đi'];print(nhom_gan_trung(c))"
    )
    ra = set()
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        ra.add(subprocess.run([sys.executable, "-c", ma], capture_output=True, text=True, env=env).stdout)
    assert len(ra) == 1, ra
