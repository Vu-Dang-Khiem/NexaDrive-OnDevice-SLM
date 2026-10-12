"""WER/CER của bộ chấm cả chuỗi giọng nói (eval/stt/cham_chuoi.py)."""

from eval.stt.cham_chuoi import chuan, wer_cer


def test_chuan_hoa_so_va_dau_cau():
    assert chuan("Bật điều hoà hai mươi hai độ!") == chuan("bật điều hoà 22 độ")
    assert chuan("Hạ kính 50%") == chuan("hạ kính năm mươi phần trăm")


def test_wer_cer():
    assert wer_cer("Mở cốp sau", "mở cốp sau.") == (0.0, 0.0)
    w, c = wer_cer("mở cốp sau", "mở cốc sau")
    assert abs(w - 1 / 3) < 1e-9 and 0 < c < w
