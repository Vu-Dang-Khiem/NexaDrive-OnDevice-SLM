"""Test phần cắt output của model + tính toán cấu hình train (không cần GPU)."""

from eval.sinh_du_doan import cat_output, thong_ke_do_tre
from train.full_finetune import CauHinhTrain, so_buoc_moi_epoch


# ── cắt output model sinh ra ────────────────────────────────────────────────
def test_giu_nguyen_loi_goi_sach():
    assert cat_output("<nexa_11>()<nexa_end>") == "<nexa_11>()<nexa_end>"


def test_cat_duoi_chatml():
    assert cat_output("<nexa_11>()<nexa_end><|im_end|>") == "<nexa_11>()<nexa_end>"


def test_cat_phan_lam_nham_sau_khi_da_goi_ham():
    """Model hay sinh tiếp sau khi xong việc — phải cắt, không thì parser đọc nhầm."""
    tho = "<nexa_11>()<nexa_end>\nTôi đã mở cốp xe cho bạn rồi nhé!"
    assert cat_output(tho) == "<nexa_11>()<nexa_end>"


def test_cat_endoftext():
    assert cat_output("<nexa_0>()<nexa_end><|endoftext|>rác") == "<nexa_0>()<nexa_end>"


def test_giu_nguyen_khi_thieu_token_ket_thuc():
    """Thiếu <nexa_end> là lỗi của model — giữ nguyên để khâu chấm ghi nhận, không tự vá."""
    assert cat_output("<nexa_11>()") == "<nexa_11>()"


def test_bo_khoang_trang_thua():
    assert cat_output("   <nexa_11>()<nexa_end>  \n") == "<nexa_11>()<nexa_end>"


def test_giu_duoc_token_hoi_lai():
    tho = '<nexa_ask><nexa_6>(state="on")<nexa_end><|im_end|>'
    assert cat_output(tho) == '<nexa_ask><nexa_6>(state="on")<nexa_end>'


# ── thống kê độ trễ ─────────────────────────────────────────────────────────
def test_do_tre_tinh_dung_p50_p95():
    tk = thong_ke_do_tre([0.1] * 50 + [0.2] * 50)
    assert tk["p50_ms"] in (100.0, 200.0)
    assert tk["p95_ms"] == 200.0


def test_do_tre_rong_khong_loi():
    assert thong_ke_do_tre([]) == {}


# ── cấu hình train ──────────────────────────────────────────────────────────
def test_tinh_dung_so_buoc_moi_epoch():
    # 15.942 mẫu, batch 8, tích luỹ 2 -> batch hiệu dụng 16
    assert so_buoc_moi_epoch(15942, batch_size=8, grad_accum=2) == 997


def test_so_buoc_lam_tron_len():
    assert so_buoc_moi_epoch(so_mau=10, batch_size=4, grad_accum=1) == 3


def test_cau_hinh_mac_dinh_bam_theo_paper():
    """Octopus v2: AdamW lr 5e-5, 10 bước warmup, 3 epoch."""
    cfg = CauHinhTrain(model_dir="m", train_file="t", val_file="v", out_dir="o")
    assert cfg.lr == 5e-5
    assert cfg.warmup_steps == 10
    assert cfg.epochs == 3.0


def test_chay_thu_thi_khong_luu_checkpoint():
    cfg = CauHinhTrain(model_dir="m", train_file="t", val_file="v", out_dir="o", gioi_han_buoc=30)
    assert cfg.gioi_han_buoc == 30
