"""Test nhánh LoRA — chạy trên model Qwen2 tí hon ở CPU, không cần GPU, không tải mạng.

Kiểm đúng 3 thứ dễ hỏng mà log không báo:
  1. Cấu hình LoRA có đúng như paper §4.2 không (r=16, α=32, 6 module, KHÔNG gate_proj)
  2. `embed_tokens` / `lm_head` có thật sự được train không — nếu không thì 27 functional
     token vĩnh viễn giữ giá trị khởi tạo mà loss vẫn giảm đẹp
  3. Hàm kiểm ở (2) có thật sự phát hiện được khi token KHÔNG đổi không
"""

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from train.lora_finetune import (  # noqa: E402
    LOP_PHAI_TRAIN,
    MODULE_LORA,
    CauHinhLora,
    anh_chup_token_moi,
    bo_lora,
    dem_tham_so,
    kiem_token_da_hoc,
)

VOCAB = 64
SO_TOKEN_MOI = 27


def _model_ti_hon() -> Qwen2ForCausalLM:
    cfg = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(cfg)


# ── cấu hình bám paper ──────────────────────────────────────────────────────
def test_cau_hinh_dung_paper():
    """Paper §4.2: rank 16, alpha 32."""
    cfg = CauHinhLora()
    assert cfg.r == 16
    assert cfg.alpha == 32


def test_dung_6_module_cua_paper():
    assert MODULE_LORA == ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj")


def test_khong_co_gate_proj():
    """Qwen2.5 CÓ gate_proj nhưng paper không liệt kê — giữ đúng paper để còn đối chiếu."""
    assert "gate_proj" not in MODULE_LORA


def test_bat_buoc_train_hai_lop_chua_functional_token():
    assert LOP_PHAI_TRAIN == ("embed_tokens", "lm_head")


# ── gắn adapter ─────────────────────────────────────────────────────────────
def test_chi_train_mot_phan_nho_tham_so():
    model = bo_lora(_model_ti_hon(), CauHinhLora())
    tk = dem_tham_so(model)
    assert 0 < tk["duoc_train"] < tk["tong"]


def test_embed_tokens_va_lm_head_duoc_train():
    """Rủi ro số 1: quên modules_to_save -> 27 token mới không bao giờ học được."""
    model = bo_lora(_model_ti_hon(), CauHinhLora())
    ten_hoc = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("embed_tokens" in n and "modules_to_save" in n for n in ten_hoc)
    assert any("lm_head" in n and "modules_to_save" in n for n in ten_hoc)


def test_trong_so_goc_bi_dong_bang():
    model = bo_lora(_model_ti_hon(), CauHinhLora())
    goc = [
        p
        for n, p in model.named_parameters()
        if "lora_" not in n and "modules_to_save" not in n
    ]
    assert goc and all(not p.requires_grad for p in goc)


# ── kiểm token mới có học được không ────────────────────────────────────────
def test_phat_hien_khi_token_moi_khong_doi():
    """Không train gì cả -> phải báo 0 token đổi."""
    model = bo_lora(_model_ti_hon(), CauHinhLora())
    truoc = anh_chup_token_moi(model, SO_TOKEN_MOI)
    kq = kiem_token_da_hoc(model, truoc, SO_TOKEN_MOI)
    assert kq["so_token_da_doi"] == 0


def test_phat_hien_khi_token_moi_da_doi():
    model = bo_lora(_model_ti_hon(), CauHinhLora())
    truoc = anh_chup_token_moi(model, SO_TOKEN_MOI)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "embed_tokens" in n and "modules_to_save" in n:
                p[-SO_TOKEN_MOI:] += 0.5
    kq = kiem_token_da_hoc(model, truoc, SO_TOKEN_MOI)
    assert kq["so_token_da_doi"] == SO_TOKEN_MOI
    assert kq["lech_lon_nhat"] == pytest.approx(0.5, abs=1e-4)


def test_gradient_that_su_chay_toi_token_moi():
    """Kiểm tận gốc: chạy 1 bước lan ngược, embedding token mới phải có gradient khác 0."""
    model = bo_lora(_model_ti_hon(), CauHinhLora())
    ids = torch.randint(0, VOCAB, (2, 8))
    model(input_ids=ids, labels=ids).loss.backward()
    emb = [
        p
        for n, p in model.named_parameters()
        if "embed_tokens" in n and "modules_to_save" in n
    ]
    assert emb and emb[0].grad is not None
    assert emb[0].grad.abs().sum() > 0


# ── trọng số DÙNG CHUNG (Qwen2.5-1.5B: tie_word_embeddings=True) ─────────────────────────────
def _model_dung_chung() -> Qwen2ForCausalLM:
    cfg = Qwen2Config(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(cfg)


def test_thao_rang_buoc_tach_hai_lop_ma_khong_doi_hanh_vi():
    from train.lora_finetune import thao_rang_buoc

    m = _model_dung_chung()
    assert m.lm_head.weight.data_ptr() == m.get_input_embeddings().weight.data_ptr()
    ids = torch.randint(0, VOCAB, (1, 6))
    truoc = m(input_ids=ids).logits
    thao_rang_buoc(m)
    assert m.lm_head.weight.data_ptr() != m.get_input_embeddings().weight.data_ptr()
    assert m.config.tie_word_embeddings is False
    assert torch.allclose(truoc, m(input_ids=ids).logits)


def test_luu_va_nap_lai_giu_lop_dau_ra_da_train(tmp_path):
    """Lỗi tìm ra khi đọc code: model dùng chung trọng số + modules_to_save -> sau gộp/lưu có thể mất bản
    đã train của lớp đầu ra. Kiểm: train (giả) lm_head của token mới, gộp, lưu, nạp lại -> vẫn còn."""
    from transformers import AutoModelForCausalLM

    model = bo_lora(_model_dung_chung(), CauHinhLora())
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lm_head" in n and "modules_to_save" in n:
                p[-SO_TOKEN_MOI:] += 0.7
    gop = model.merge_and_unload()
    gop.config.tie_word_embeddings = False
    gop.save_pretrained(tmp_path)
    nap = AutoModelForCausalLM.from_pretrained(tmp_path)
    assert torch.allclose(nap.lm_head.weight, gop.lm_head.weight)
    assert not torch.allclose(nap.lm_head.weight, nap.get_input_embeddings().weight)


def test_gop_checkpoint_bao_loi_khi_lop_dau_ra_khong_doi(tmp_path):
    """Checkpoint mà token mới chưa học gì ở lớp đầu ra -> gop_checkpoint phải từ chối."""
    from train.lora_finetune import gop_checkpoint

    goc = _model_dung_chung()
    goc.save_pretrained(tmp_path / "goc")
    model = bo_lora(Qwen2ForCausalLM.from_pretrained(tmp_path / "goc"), CauHinhLora())
    model.save_pretrained(tmp_path / "ck")  # adapter chưa train gì
    with pytest.raises(RuntimeError, match="KHÔNG đổi"):
        gop_checkpoint(str(tmp_path / "goc"), str(tmp_path / "ck"), str(tmp_path / "ra"))


def test_gop_adapter_ra_model_thuong():
    """Sau khi gộp, model phải nạp/chạy được như model bình thường (để chấm + quantize)."""
    model = bo_lora(_model_ti_hon(), CauHinhLora())
    gop = model.merge_and_unload()
    out = gop(input_ids=torch.randint(0, VOCAB, (1, 5)))
    assert out.logits.shape[-1] == VOCAB
    assert not any("lora_" in n for n, _ in gop.named_parameters())
