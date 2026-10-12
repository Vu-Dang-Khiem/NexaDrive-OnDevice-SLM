"""Student THẬT sau cổng model: GGUF q4_K_M chạy bằng llama.cpp (`llama-server`) trên CPU
(PLAN.md mục W, bước 4).

Cùng giao diện `MoHinh` với model giả (`serve/mo_hinh.py`) -> router không phải sửa. Ba điểm đã kiểm
ở bước 2–3:
  • prompt gửi dạng ID TOKEN do tokenizer HF cắt (`models/gguf/tokenizer`, trùng `llama-tokenize`);
  • output đọc từ ID token rồi giải mã bằng tokenizer HF — trường chữ `content` của llama.cpp ẨN token
    CONTROL, tức mất đúng `<nexa_i>`;
  • NGƯỠNG độ tự tin ở token quyết định (luật mục S, ngưỡng 0,80 khoá cho q4): hạng 1 là một hàm mà p < ngưỡng
    -> hạng 2 là token từ chối thì TỪ CHỐI, còn lại thì HỎI LẠI đề xuất ĐỦ lời gọi hạng 1 (kèm tham số) để
    người lái chỉ cần nói "có". Token hạng 1 là từ chối / hỏi lại -> làm theo model.
"""

from __future__ import annotations

import json
import math
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.sinh_du_doan import cat_output
from schema.call_format import CallParseError, parse_calls, render_completion
from schema.loader import CATALOG

GGUF_MAC_DINH = "models/gguf/lora-v61-s42-q4_k_m.gguf"  # bản triển khai v6.1 s42 ck4192 (v5: lora-v5-s7-…)
TOKENIZER_MAC_DINH = "models/gguf/tokenizer"
NGUONG_MAC_DINH = 0.80  # v6.1 q4: chọn trên tập hiệu chỉnh 194 câu (v5 q4 cũng 0,80 theo luật mục W)
SERVER = Path.home() / "llama-bin" / "llama-server.exe"
ID_NEXA_END = 151690  # id của <nexa_end> trong tokenizer (27 token mới: 151665–151691), kiểm ở test

TU_CHOI = CATALOG.by_name["irrelevant_function"].nexa_token
ASK = CATALOG.nexa_ask
IM_END = "<|im_end|>"

GoiHttp = Callable[[dict[str, Any]], dict[str, Any]]


def ap_nguong(output: str, top: list[tuple[str, float]], nguong: float) -> tuple[str, str]:
    """Áp luật ngưỡng lên output lượt 1 -> (output mới, lý do). Hàm thuần, test được không cần model."""
    (t1, p1), t2 = top[0], (top[1][0] if len(top) > 1 else None)
    if t1 in (TU_CHOI, ASK) or p1 >= nguong:
        return output, "giu"
    if t2 == TU_CHOI:
        return render_completion([("irrelevant_function", {})]), "nguong_tu_choi"
    try:
        calls = parse_calls(output)
    except CallParseError:
        return output, "giu"  # output hỏng -> để router báo "chưa hiểu"
    if any(c.name not in CATALOG.by_name for c in calls):
        return output, "giu"
    # model đôi khi sinh tham số KHÔNG có trong schema (đo độ chịu ồn 2026-10-08: `find_nearby(query=…)`)
    # — viết lại thành câu hỏi lại thì bỏ tham số lạ (validator phía sau cũng bỏ), không để render báo lỗi
    # làm hỏng cả lượt
    hop_le = []
    for c in calls:
        co = {p.name for p in CATALOG.by_name[c.name].params}
        hop_le.append((c.name, {k: v for k, v in c.args.items() if k in co}))
    return render_completion(hop_le, ask=True), "nguong_hoi_lai"


@dataclass
class LanGoi:
    """Thông tin lần gọi gần nhất — để ghi log / hiện ở khung debug của demo."""

    output_goc: str = ""
    output: str = ""
    p: float = 0.0
    top: list[tuple[str, float]] = field(default_factory=list)
    ly_do: str = ""
    ms: float = 0.0


class MoHinhGGUF:
    def __init__(
        self,
        url: str = "http://127.0.0.1:8090",
        tokenizer_dir: str = TOKENIZER_MAC_DINH,
        nguong: float = NGUONG_MAC_DINH,
        goi_http: GoiHttp | None = None,
        tokenizer: Any = None,
        nho_luot_truoc: bool = False,
    ) -> None:
        # True CHỈ với model đã train trên dữ liệu hai lượt (v6) — model v5 chưa từng thấy dòng "Lượt trước:"
        self.nho_luot_truoc = nho_luot_truoc
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.tok = tokenizer
        self.url = url
        self.nguong = nguong
        self._goi = goi_http or self._goi_that
        self.lan_cuoi = LanGoi()
        self._khoa = threading.Lock()  # nhiều phiên WebSocket dùng chung một model / một khe llama-server

    def _goi_that(self, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(f"{self.url}/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    def _sinh(self, prompt: str, n_predict: int, n_probs: int) -> dict[str, Any]:
        ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        # cache_prompt=False: giữ đúng điều kiện đã đo / chấm ở bước 3 (prompt ngắn, lợi ích cache nhỏ)
        return self._goi({"prompt": ids, "n_predict": n_predict, "temperature": 0, "n_probs": n_probs,
                          "return_tokens": True, "cache_prompt": False})

    # ── lượt 1 ────────────────────────────────────────────────────────────
    def goi_ham(self, cau: str, luot_truoc: str | None = None) -> str:
        with self._khoa:
            t0 = time.perf_counter()
            truoc = luot_truoc if self.nho_luot_truoc else None
            ph = self._sinh(CATALOG.render_call_prompt(cau, truoc), 64, 5)
            goc = cat_output(self.tok.decode(ph["tokens"], skip_special_tokens=False))
            buoc = ph["completion_probabilities"][0]
            top = [(self.tok.convert_ids_to_tokens(int(t["id"])), math.exp(t["logprob"]))
                   for t in buoc["top_logprobs"]]
            out, ly_do = ap_nguong(goc, top, self.nguong)
            self.lan_cuoi = LanGoi(goc, out, math.exp(buoc["logprob"]), top, ly_do,
                                   round((time.perf_counter() - t0) * 1000, 1))
            return out

    # ── lượt 2 ────────────────────────────────────────────────────────────
    def tra_loi(self, cau: str, loi_goi: str, ket_qua: dict[str, Any]) -> str | None:
        with self._khoa:
            ph = self._sinh(CATALOG.render_reply_prompt(cau, loi_goi, ket_qua), 96, 0)
        text = self.tok.decode(ph["tokens"], skip_special_tokens=False).split(IM_END, 1)[0].strip()
        return text or None


class MayChuLlama:
    """Bật / tắt `llama-server` cho model GGUF (dùng kèm `with`)."""

    def __init__(self, gguf: str = GGUF_MAC_DINH, port: int = 8090, luong: int = 8) -> None:
        self.gguf, self.port, self.luong = gguf, port, luong
        self.p: subprocess.Popen[bytes] | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def bat(self) -> MayChuLlama:
        if not Path(self.gguf).exists():
            raise FileNotFoundError(f"không thấy model GGUF: {self.gguf}")
        # <nexa_end> = token KẾT THÚC: không có nó, câu từ chối phân vân bị model viết tiếp "Mô tả hàm: …"
        # (dấu vết thí nghiệm mô tả hàm trong đáp án) tới đủ n_predict rồi mới bị cắt -> chậm 5–8 lần. Chuỗi
        # stop không ăn vì <nexa_end> là token đặc biệt. Đã kiểm 2026-10-07: 400 câu val + 8 câu demo output
        # Y HỆT, tổng thời gian 540 s -> 249 s; lượt 2 vẫn dừng ở <|im_end|> (llama.cpp tự coi là kết thúc).
        lenh = [str(SERVER), "-m", self.gguf, "-c", "2048", "-t", str(self.luong), "-np", "1",
                "--port", str(self.port), "--log-disable",
                "--override-kv", f"tokenizer.ggml.eos_token_id=int:{ID_NEXA_END}"]
        self.p = subprocess.Popen(lenh, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(120):
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=2) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        return self
            except Exception:
                pass
            time.sleep(1)
        self.tat()
        raise RuntimeError("llama-server không lên được")

    def tat(self) -> None:
        if self.p is not None:
            self.p.terminate()
            self.p.wait(timeout=10)
            self.p = None

    def __enter__(self) -> MayChuLlama:
        return self.bat()

    def __exit__(self, *_: object) -> None:
        self.tat()
