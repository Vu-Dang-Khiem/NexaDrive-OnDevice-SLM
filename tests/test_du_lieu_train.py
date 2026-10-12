"""Test chuẩn bị dữ liệu train (train/du_lieu.py) — quan trọng nhất là MASK LOSS."""

import pytest

from gen.records import CallRecord, negative_record
from train.du_lieu import BO_QUA, GhepLo, dem_token_hoc, tao_mau, thong_ke_do_dai

torch = pytest.importorskip("torch")


class TokenizerGia:
    """Tokenizer giả: mỗi KÝ TỰ là 1 token (id = mã ký tự). Đủ để kiểm logic mask."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) for c in text]


TK = TokenizerGia()


# ── mask loss ───────────────────────────────────────────────────────────────
def test_phan_prompt_bi_mask_phan_completion_thi_khong():
    row = {"prompt": "abc", "completion": "XY"}
    mau = tao_mau(TK, row)
    assert mau["labels"][:3] == [BO_QUA, BO_QUA, BO_QUA]
    assert mau["labels"][3:] == [ord("X"), ord("Y")]


def test_input_ids_la_prompt_noi_completion():
    mau = tao_mau(TK, {"prompt": "ab", "completion": "cd"})
    assert mau["input_ids"] == [ord(c) for c in "abcd"]


def test_dem_dung_so_token_duoc_hoc():
    mau = tao_mau(TK, {"prompt": "dài dòng lắm", "completion": "abc"})
    assert dem_token_hoc(mau) == 3


def test_attention_mask_phu_toan_bo_chuoi_that():
    mau = tao_mau(TK, {"prompt": "ab", "completion": "cde"})
    assert mau["attention_mask"] == [1] * 5


def test_cat_bot_khi_qua_dai():
    mau = tao_mau(TK, {"prompt": "a" * 10, "completion": "b" * 10}, do_dai_toi_da=12)
    assert len(mau["input_ids"]) == 12
    assert len(mau["labels"]) == 12


def test_mau_that_tu_repo_mask_dung():
    """Kiểm trên đúng định dạng dòng dữ liệu thật."""
    row = CallRecord("Mở cốp xe", "open_trunk", {}).to_row()
    mau = tao_mau(TK, row)
    so_ky_tu_completion = len(row["completion"])
    assert dem_token_hoc(mau) == so_ky_tu_completion
    assert all(x == BO_QUA for x in mau["labels"][:-so_ky_tu_completion])


# ── ghép lô ─────────────────────────────────────────────────────────────────
def test_ghep_lo_dem_ve_cung_do_dai():
    lo = GhepLo(pad_token_id=0)([tao_mau(TK, {"prompt": "a", "completion": "b"}),
                                 tao_mau(TK, {"prompt": "abc", "completion": "de"})])
    assert lo["input_ids"].shape == (2, 5)
    assert lo["labels"].shape == (2, 5)


def test_phan_dem_khong_tinh_loss_va_khong_duoc_nhin():
    lo = GhepLo(pad_token_id=0)([tao_mau(TK, {"prompt": "a", "completion": "b"}),
                                 tao_mau(TK, {"prompt": "abc", "completion": "de"})])
    ngan = lo["labels"][0]
    assert ngan[-3:].tolist() == [BO_QUA] * 3  # phần đệm
    assert lo["attention_mask"][0].tolist() == [1, 1, 0, 0, 0]


def test_ghep_lo_dung_pad_token_da_chi_dinh():
    lo = GhepLo(pad_token_id=7)([tao_mau(TK, {"prompt": "a", "completion": "b"}),
                                 tao_mau(TK, {"prompt": "abcd", "completion": "e"})])
    assert lo["input_ids"][0][-1].item() == 7


# ── thống kê độ dài ─────────────────────────────────────────────────────────
def test_thong_ke_do_dai_chay_duoc():
    rows = [CallRecord(f"mở cốp {i}", "open_trunk", {}).to_row() for i in range(10)]
    rows += [negative_record(f"tán gẫu {i}").to_row() for i in range(10)]
    tk = thong_ke_do_dai(TK, rows)
    assert tk["so_mau_kiem"] == 20
    assert tk["dai_nhat"] >= tk["p95"] >= 0
    assert tk["token_tinh_loss_trung_binh"] > 0
