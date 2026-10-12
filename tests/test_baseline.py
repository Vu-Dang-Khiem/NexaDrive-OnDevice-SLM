"""Baseline SIP: prompt đủ 24 hàm + chính sách, chuyển JSON -> nexa đúng, ví dụ mẫu không trùng tập test."""

import json
from pathlib import Path

import pytest

from eval.baseline import VI_DU, build_sip_prompt, sang_nexa
from eval.metrics import cham_mot_cau
from schema.loader import CATALOG


def test_prompt_co_du_24_ham_va_ba_hanh_dong():
    p = build_sip_prompt("Mở cốp sau")
    for f in CATALOG.functions:
        if f.name != "irrelevant_function":
            assert f"def {f.name}(" in p
    assert all(h in p for h in ('"goi"', '"hoi_lai"', '"tu_choi"'))
    assert p.endswith("<|im_start|>assistant\n")


def test_khong_co_luot_truoc_thi_prompt_nhu_cu():
    p = build_sip_prompt("Mở cốp sau")
    assert "Lượt trước" not in p
    assert '<|im_start|>user\nCâu: "Mở cốp sau"<|im_end|>' in p


def test_luot_truoc_dua_loi_goi_nhu_student_thay():
    truoc = ("control_window", {"window": "front_right", "position": "down"})
    p = build_sip_prompt("Bên trái cũng vậy nha", truoc)
    assert '\nLượt trước (đã thực hiện): control_window({"window": "front_right", "position": "down"})\n' in p
    assert 'Câu: "Bên trái cũng vậy nha"<|im_end|>' in p
    assert 'Nếu có dòng "Lượt trước"' in p
    assert p.endswith("<|im_start|>assistant\n")


@pytest.mark.parametrize(
    "json_ra,nexa",
    [
        ('{"hanh_dong": "goi", "loi_goi": [{"ham": "open_trunk", "tham_so": {}}]}', "<nexa_11>()<nexa_end>"),
        ('{"hanh_dong": "tu_choi", "loi_goi": []}', "<nexa_24>()<nexa_end>"),
        (
            '{"hanh_dong": "hoi_lai", "loi_goi": [{"ham": "set_ac", "tham_so": {"state": "on"}}]}',
            '<nexa_ask><nexa_6>(state="on")<nexa_end>',
        ),
        # tham số lạ bị bỏ, null bị bỏ
        (
            '{"hanh_dong": "goi", "loi_goi": [{"ham": "set_volume", '
            '"tham_so": {"delta": 2, "x": 1, "set_value": null}}]}',
            "<nexa_19>(delta=2)<nexa_end>",
        ),
    ],
)
def test_chuyen_json_sang_nexa(json_ra, nexa):
    assert sang_nexa(json_ra) == nexa


@pytest.mark.parametrize(
    "hong",
    [
        "không phải json",
        '{"hanh_dong": "bay"}',
        '{"hanh_dong": "goi", "loi_goi": [{"ham": "fly_car"}]}',
        '{"hanh_dong": "goi", "loi_goi": []}',
    ],
)
def test_output_hong_thanh_khong_doc_duoc(hong):
    assert sang_nexa(hong) == ""
    assert cham_mot_cau(sang_nexa(hong), "<nexa_11>()<nexa_end>")[0] == "khong_parse"


def test_cham_duoc_bang_bo_cham_cu():
    ra = sang_nexa('{"hanh_dong": "goi", "loi_goi": [{"ham": "open_trunk", "tham_so": {}}]}')
    assert cham_mot_cau(ra, "<nexa_11>()<nexa_end><|im_end|>")[0] == "dung"


def test_vi_du_mau_khong_trung_tap_test_va_dev():
    from eval.kiem_nhiem import tim_nhiem
    from gen.records import negative_record

    tap = []
    for p in (
        "data/processed/final/test_vi.jsonl",
        "data/processed/final/test_vi_bo_sung.jsonl",
        "data/dev_tay_2/dev_vi_2.jsonl",
    ):
        if Path(p).exists():
            tap += [json.loads(x) for x in open(p, encoding="utf-8")]
    if not tap:
        pytest.skip("không có tập test ở máy này")
    vi_du = [negative_record(c).to_row() for c, _ in VI_DU]
    assert tim_nhiem(vi_du, tap)["so_bi_nhiem"] == 0
