"""Test bốc mẫu con — logic thuần, không cần GPU."""

from gen.records import CallRecord, ReplyRecord
from gen.subsample import subsample


def _kho(so_call_moi_ham: int = 100, so_reply_moi_ham: int = 20) -> list[dict]:
    """Kho giả lập: 3 hàm, mỗi hàm có sẵn số call/reply cho trước."""
    rows = []
    for fn, args in [("set_ac", {"state": "on"}), ("open_trunk", {}), ("lock_doors", {})]:
        for i in range(so_call_moi_ham):
            rows.append(CallRecord(f"câu hỏi {fn} {i}", fn, args).to_row())
        for i in range(so_reply_moi_ham):
            rows.append(ReplyRecord(f"câu hỏi {fn} {i}", fn, args, {"ok": True}, f"trả lời {i}").to_row())
    return rows


def test_boc_dung_so_luong_moi_ham():
    con = subsample(_kho(), per_fn=30, seed=1)
    theo_ham: dict[str, int] = {}
    for r in con:
        theo_ham[r["meta"]["fn"]] = theo_ham.get(r["meta"]["fn"], 0) + 1
    assert set(theo_ham) == {"set_ac", "open_trunk", "lock_doors"}
    for n in theo_ham.values():
        assert n == 30


def test_moi_ham_deu_co_mau_khong_bi_boc_truot():
    """Bốc theo từng hàm (không bốc ngẫu nhiên toàn cục) nên hàm hiếm vẫn có mẫu."""
    kho = _kho(so_call_moi_ham=100, so_reply_moi_ham=20)
    # thêm 1 hàm chỉ có 3 mẫu — bốc toàn cục rất dễ trượt hàm này
    kho += [CallRecord(f"hiếm {i}", "operate_sunroof", {"action": "open"}).to_row() for i in range(3)]
    con = subsample(kho, per_fn=50, seed=1)
    assert any(r["meta"]["fn"] == "operate_sunroof" for r in con)


def test_giu_ti_le_reply_gan_dung():
    """Kho có ~17% reply -> tập bốc ra cũng phải xấp xỉ vậy, không lệch hẳn."""
    con = subsample(_kho(so_call_moi_ham=100, so_reply_moi_ham=20), per_fn=60, seed=1)
    n_reply = sum(1 for r in con if r["meta"]["type"] == "reply")
    ti_le = n_reply / len(con)
    assert 0.10 < ti_le < 0.25


def test_cung_seed_ra_cung_ket_qua():
    """Tái lập được — quan trọng để viết vào luận văn."""
    a = subsample(_kho(), per_fn=20, seed=7)
    b = subsample(_kho(), per_fn=20, seed=7)
    assert [r["completion"] for r in a] == [r["completion"] for r in b]


def test_seed_khac_ra_tap_khac():
    a = subsample(_kho(), per_fn=20, seed=1)
    b = subsample(_kho(), per_fn=20, seed=2)
    assert [r["completion"] for r in a] != [r["completion"] for r in b]


def _kho_co_negative(so_negative: int) -> list[dict]:
    kho = _kho(so_call_moi_ham=100, so_reply_moi_ham=20)  # 3 hàm thật
    kho += [CallRecord(f"tán gẫu {i}", "irrelevant_function", {}).to_row() for i in range(so_negative)]
    return kho


def test_negative_khong_bi_cat_theo_per_fn_ma_giu_ti_le_1_1():
    """Lỗi đã suýt mắc: cắt irrelevant_function xuống per_fn -> tỉ lệ pos:neg thành 3:1
    (với 3 hàm) hay 24:1 (với 24 hàm thật)."""
    con = subsample(_kho_co_negative(1000), per_fn=30, seed=1)
    n_neg = sum(1 for r in con if r["meta"]["fn"] == "irrelevant_function")
    n_pos = len(con) - n_neg
    assert n_pos == 3 * 30
    assert n_neg == n_pos  # mặc định 1:1


def test_so_negative_chi_dinh_ro():
    con = subsample(_kho_co_negative(1000), per_fn=30, seed=1, so_negative=45)
    assert sum(1 for r in con if r["meta"]["fn"] == "irrelevant_function") == 45


def test_negative_it_hon_yeu_cau_thi_lay_het():
    con = subsample(_kho_co_negative(10), per_fn=30, seed=1)
    assert sum(1 for r in con if r["meta"]["fn"] == "irrelevant_function") == 10


def test_mau_ask_duoc_giu_het_va_tinh_la_positive():
    """Lỗi dễ mắc: vòng lặp theo hàm chỉ lấy call/reply -> mẫu ask rơi mất âm thầm."""
    from gen.records import SuggestRecord

    kho = _kho_co_negative(1000)
    kho += [SuggestRecord(f"nóng quá {i}", (("set_ac", {"state": "on"}),)).to_row() for i in range(10)]
    con = subsample(kho, per_fn=30, seed=1)
    n_ask = sum(1 for r in con if r["meta"]["type"] == "ask")
    n_neg = sum(1 for r in con if r["meta"]["fn"] == "irrelevant_function")
    assert n_ask == 10
    assert n_neg == 3 * 30 + 10  # 1:1 tính cả mẫu ask


def test_so_ask_chi_dinh_ro():
    from gen.records import SuggestRecord

    kho = _kho() + [SuggestRecord(f"nóng {i}", (("set_ac", {"state": "on"}),)).to_row() for i in range(10)]
    con = subsample(kho, per_fn=30, seed=1, so_ask=4)
    assert sum(1 for r in con if r["meta"]["type"] == "ask") == 4


def test_xin_nhieu_hon_kho_co_thi_lay_het_khong_bao_loi():
    con = subsample(_kho(so_call_moi_ham=5, so_reply_moi_ham=2), per_fn=1000, seed=1)
    assert len(con) == 3 * (5 + 2)  # 3 hàm × (5 call + 2 reply)
