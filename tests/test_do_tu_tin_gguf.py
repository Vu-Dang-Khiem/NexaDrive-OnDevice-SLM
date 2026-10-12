"""`eval/do_tu_tin_gguf.py`: phản hồi llama-server -> bản ghi chấm (PLAN.md mục W, bước 3)."""

import json
import math

import pytest

from eval.do_tu_tin_gguf import doc_tap, dong_ket_qua
from schema.loader import CATALOG

TU_CHOI = CATALOG.by_name["irrelevant_function"].nexa_token
HAM_0 = CATALOG.functions[0].nexa_token
PROMPT = (
    "<|im_start|>system\nBạn là tác tử điều khiển xe ô tô.<|im_end|>\n"
    "<|im_start|>user\nKể chuyện cười đi<|im_end|>\n<|im_start|>assistant\n"
)


class TokGia:
    """Tokenizer giả: id -> chuỗi; giải mã GIỮ token đặc biệt (như HF với skip_special_tokens=False)."""

    BANG = {1: TU_CHOI, 2: "()", 3: CATALOG.nexa_end, 4: "<|im_end|>", 5: CATALOG.nexa_ask, 6: HAM_0,
            7: " Hy"}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.BANG[i] for i in ids)

    def convert_ids_to_tokens(self, i):
        return self.BANG[i]


def _buoc(id_chon, top):
    return {"id": id_chon, "logprob": math.log(top[0][1]),
            "top_logprobs": [{"id": i, "logprob": math.log(p)} for i, p in top]}


def _phan_hoi(ids, cac_buoc):
    # `content` của llama.cpp ẩn token CONTROL -> cố ý để "()" để chắc code KHÔNG đọc trường này
    return {"content": "()", "tokens": ids, "completion_probabilities": cac_buoc,
            "timings": {"prompt_ms": 265.07, "predicted_ms": 94.79, "predicted_n": len(ids)}}


def test_tu_choi_dung_va_doc_tu_id_token_khong_doc_content():
    row = {"prompt": PROMPT, "completion": f"{TU_CHOI}(){CATALOG.nexa_end}<|im_end|>",
           "meta": {"nhom_phu": "oos"}}
    buoc_1 = _buoc(1, [(1, 0.9927), (6, 0.0018), (7, 0.0004)])
    ph = _phan_hoi([1, 2, 3, 4], [buoc_1] + [_buoc(2, [(2, 1.0)])] * 3)
    d = dong_ket_qua(row, ph, TokGia())
    assert d["output"] == f"{TU_CHOI}(){CATALOG.nexa_end}"
    assert d["ket_qua"] == "dung" and d["nhom"] == "ngoai_pham_vi"
    assert d["cau"] == "Kể chuyện cười đi"
    assert d["p_buoc_1"] == pytest.approx(0.9927, abs=1e-5)
    assert d["top_buoc_1"][1] == (HAM_0, pytest.approx(0.0018, abs=1e-5))
    assert d["ms_doc_prompt"] == 265.1 and d["ms_sinh"] == 94.8


def test_hoi_lai_ghi_them_buoc_2():
    row = {"prompt": PROMPT, "completion": f"{TU_CHOI}(){CATALOG.nexa_end}<|im_end|>"}
    ph = _phan_hoi([5, 6, 2, 3, 4], [_buoc(5, [(5, 0.7), (1, 0.2)]), _buoc(6, [(6, 0.6), (1, 0.3)])]
                   + [_buoc(2, [(2, 1.0)])] * 3)
    d = dong_ket_qua(row, ph, TokGia())
    assert d["p_buoc_2"] == pytest.approx(0.6, abs=1e-5)
    assert d["top_buoc_2"][0][0] == HAM_0


def test_khong_doc_tap_test_khi_chua_khoa(tmp_path):
    p = tmp_path / "test_vi_sach_v2.jsonl"
    dong = {"prompt": PROMPT, "completion": "x", "meta": {"type": "call"}}
    p.write_text(json.dumps(dong) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        doc_tap([str(p)], cho_phep_test=False)
    assert len(doc_tap([str(p)], cho_phep_test=True)) == 1


def test_chi_lay_mau_luot_1(tmp_path):
    p = tmp_path / "val_vi.jsonl"
    dong = [{"prompt": PROMPT, "completion": "x", "meta": {"type": t}} for t in ("call", "ask", "reply")]
    p.write_text("\n".join(json.dumps(d) for d in dong) + "\n", encoding="utf-8")
    rows = doc_tap([str(p)], cho_phep_test=False)
    assert [r["meta"]["type"] for r in rows] == ["call", "ask"] and rows[0]["meta"]["tap"] == "val_vi"
