"""Test bước thêm functional token (train/add_tokens.py).

Dùng model Qwen2 TÍ HON dựng tại chỗ (không tải mạng, chạy trong ~1 giây) để kiểm đúng
phần dễ sai nhất: nới lm_head và khởi tạo vector token mới.
"""

import pytest

torch = pytest.importorskip("torch")
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from gen.records import CallRecord, ReplyRecord  # noqa: E402
from schema.loader import CATALOG  # noqa: E402
from train.add_tokens import (  # noqa: E402
    danh_sach_token_moi,
    kiem_ranh_gioi,
    mo_rong_model,
    vector_khoi_tao,
)

VOCAB_CU = 64
HIDDEN = 16


def _model_ti_hon(tie: bool = False) -> Qwen2ForCausalLM:
    cfg = Qwen2Config(
        vocab_size=VOCAB_CU,
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=tie,
    )
    return Qwen2ForCausalLM(cfg)


# ── danh sách token ─────────────────────────────────────────────────────────
def test_lay_dung_27_token_tu_schema():
    ds = danh_sach_token_moi()
    assert len(ds) == len(CATALOG.functions) + 2  # 25 hàm + nexa_end + nexa_ask
    assert ds[0] == "<nexa_0>" and ds[-1] == CATALOG.nexa_ask


def test_khong_co_token_trung_nhau():
    ds = danh_sach_token_moi()
    assert len(set(ds)) == len(ds)


# ── khởi tạo vector ─────────────────────────────────────────────────────────
def test_vector_moi_nam_gan_trung_binh_vector_cu():
    cu = torch.randn(100, 8) * 3 + 5
    day_du = torch.cat([cu, torch.zeros(5, 8)])
    moi = vector_khoi_tao(day_du, so_token_moi=5)
    # mỗi vector mới phải gần trung bình vector cũ, lệch trong khoảng nhiễu nhỏ
    assert torch.allclose(moi.mean(dim=0), cu.mean(dim=0), atol=0.5)


def test_cac_token_moi_khong_giong_het_nhau():
    """Giống hệt nhau thì gradient cũng giống nhau -> 27 token khó tách ra được."""
    day_du = torch.cat([torch.randn(100, 8), torch.zeros(5, 8)])
    moi = vector_khoi_tao(day_du, so_token_moi=5)
    assert not torch.allclose(moi[0], moi[1])


def test_cung_seed_ra_cung_vector():
    day_du = torch.cat([torch.randn(100, 8), torch.zeros(3, 8)])
    a = vector_khoi_tao(day_du, 3, seed=7)
    b = vector_khoi_tao(day_du, 3, seed=7)
    assert torch.equal(a, b)


# ── mở rộng model ───────────────────────────────────────────────────────────
def test_mo_rong_ca_embedding_lan_lm_head():
    model = _model_ti_hon()
    mo_rong_model(model, so_token_moi=27, vocab_moi=VOCAB_CU + 27)
    assert model.get_input_embeddings().weight.shape[0] == VOCAB_CU + 27
    assert model.get_output_embeddings().weight.shape[0] == VOCAB_CU + 27


def test_khong_lam_hong_vector_token_cu():
    model = _model_ti_hon()
    cu = model.get_input_embeddings().weight[:VOCAB_CU].clone()
    mo_rong_model(model, 27, VOCAB_CU + 27)
    assert torch.equal(model.get_input_embeddings().weight[:VOCAB_CU], cu)


def test_vector_token_moi_khac_0():
    """Để nguyên 0 thì model không phân biệt nổi 27 token mới."""
    model = _model_ti_hon()
    mo_rong_model(model, 27, VOCAB_CU + 27)
    moi = model.get_input_embeddings().weight[-27:]
    assert not torch.allclose(moi, torch.zeros_like(moi))


def test_model_dung_chung_trong_so_van_chay_duoc():
    """tie_word_embeddings=True: lm_head DÙNG CHUNG trọng số với embedding."""
    model = _model_ti_hon(tie=True)
    mo_rong_model(model, 27, VOCAB_CU + 27)
    vao = model.get_input_embeddings().weight
    ra = model.get_output_embeddings().weight
    assert vao.shape[0] == ra.shape[0] == VOCAB_CU + 27
    assert torch.equal(vao[-27:], ra[-27:])


# ── kiểm ranh giới prompt|completion ────────────────────────────────────────
class _TokenizerGhep:
    """Tokenizer giả CÓ ghép token qua ranh giới: '\\nX' thành 1 token."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids, i = [], 0
        while i < len(text):
            if text[i : i + 2] == "\nX":
                ids.append(9999)
                i += 2
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids


class _TokenizerThuong:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) for c in text]


def test_ranh_gioi_khong_lech_thi_bao_0():
    rows = [CallRecord("Mở cốp xe", "open_trunk", {}).to_row()]
    assert kiem_ranh_gioi(_TokenizerThuong(), rows)["so_mau_lech"] == 0


def test_phat_hien_duoc_ranh_gioi_bi_ghep():
    """prompt kết thúc bằng '\\n', completion mở đầu bằng 'X' -> tokenizer ghép lại."""
    rows = [{"prompt": "abc\n", "completion": "Xyz", "meta": {"type": "reply"}}]
    kq = kiem_ranh_gioi(_TokenizerGhep(), rows)
    assert kq["so_mau_lech"] == 1
    assert kq["vi_du_lech"][0]["loai"] == "reply"


def test_bao_cao_kem_ngu_canh_de_soi():
    rows = [{"prompt": "xin chao\n", "completion": "Xin loi", "meta": {"type": "reply"}}]
    vi_du = kiem_ranh_gioi(_TokenizerGhep(), rows)["vi_du_lech"][0]
    assert vi_du["cuoi_prompt"].endswith("\n")
    assert vi_du["dau_completion"].startswith("X")


def test_mau_reply_that_cung_kiem_duoc():
    rows = [ReplyRecord("còn xăng?", "get_fuel_level", {}, {"fuel_percent": 42}, "Xăng còn 42%.").to_row()]
    kq = kiem_ranh_gioi(_TokenizerThuong(), rows)
    assert kq["so_mau_kiem"] == 1


def test_model_sinh_duoc_token_moi_sau_khi_mo_rong():
    """Kiểm điều quan trọng nhất: lớp ra có đủ ô cho token mới -> sinh ra được."""
    model = _model_ti_hon()
    mo_rong_model(model, 27, VOCAB_CU + 27)
    out = model(input_ids=torch.tensor([[1, 2, 3]]))
    assert out.logits.shape[-1] == VOCAB_CU + 27
