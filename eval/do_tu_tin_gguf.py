"""Đo ĐỘ TỰ TIN + chấm bản GGUF (llama.cpp) — cùng định dạng đầu ra với `eval/do_tu_tin.py` (PLAN.md mục W).

Khác bản HF ở hai chỗ, đều đã kiểm (mục W, bước 2–3):
  • prompt gửi lên `llama-server` dưới dạng ID TOKEN do tokenizer HF cắt (đã kiểm 30/30 prompt trùng
    `llama-tokenize`) — tránh mọi khác biệt khi server tự cắt chữ;
  • output lấy từ ID token server trả về rồi giải mã bằng tokenizer HF: phần chữ (`content`) của
    llama.cpp ẨN token loại CONTROL, tức mất đúng `<nexa_i>` cần chấm.
Xác suất = exp(logprob) server trả (trước khi lấy mẫu; greedy). Ghi kèm thời gian từng câu.

CHỈ chạy val / dev khi đang chọn mức nén / ngưỡng; test chấm MỘT lần sau khi đã khoá.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from eval.metrics import cham_mot_cau, doc_output, nhom_cua_dap_an
from eval.sinh_du_doan import cat_output
from gen.records import query_from_call_prompt
from schema.loader import CATALOG

SERVER = Path.home() / "llama-bin" / "llama-server.exe"
TOKENIZER_MAC_DINH = "models/gguf/tokenizer"  # trùng từng byte tokenizer models/lora-v5-s7-ck2416 (đã xoá)


def _top(buoc: dict[str, Any], tokenizer: Any) -> list[tuple[str, float]]:
    return [
        (tokenizer.convert_ids_to_tokens(int(t["id"])), round(math.exp(t["logprob"]), 5))
        for t in buoc["top_logprobs"]
    ]


def dong_ket_qua(row: dict[str, Any], phan_hoi: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Một câu: phản hồi `/completion` của llama-server -> bản ghi giống `eval/do_tu_tin.py`."""
    ids = phan_hoi["tokens"]
    out = cat_output(tokenizer.decode(ids, skip_special_tokens=False))
    xs = phan_hoi["completion_probabilities"]
    dong: dict[str, Any] = {
        "cau": query_from_call_prompt(row["prompt"]),
        "nhom": nhom_cua_dap_an(doc_output(row["completion"])),
        "nhom_phu": row.get("meta", {}).get("nhom_phu"),
        "dap_an": row["completion"],
        "output": out,
        "ket_qua": cham_mot_cau(out, row["completion"])[0],
        "p_buoc_1": round(math.exp(xs[0]["logprob"]), 5),
        "top_buoc_1": _top(xs[0], tokenizer),
    }
    if tokenizer.convert_ids_to_tokens(int(ids[0])) == CATALOG.nexa_ask and len(xs) > 1:
        dong["p_buoc_2"] = round(math.exp(xs[1]["logprob"]), 5)
        dong["top_buoc_2"] = _top(xs[1], tokenizer)
    t = phan_hoi.get("timings", {})
    dong["ms_doc_prompt"] = round(t.get("prompt_ms", 0.0), 1)
    dong["ms_sinh"] = round(t.get("predicted_ms", 0.0), 1)
    dong["so_token_sinh"] = t.get("predicted_n", len(ids))
    return dong


def _goi(port: int, ids: list[int]) -> dict[str, Any]:
    body = {"prompt": ids, "n_predict": 64, "temperature": 0, "n_probs": 5, "return_tokens": True,
            "cache_prompt": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def _bat_server(gguf: str, port: int, luong: int) -> subprocess.Popen[bytes]:
    p = subprocess.Popen([str(SERVER), "-m", gguf, "-c", "2048", "-t", str(luong), "-np", "1",
                          "--port", str(port), "--log-disable"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return p
        except Exception:
            pass
        time.sleep(1)
    p.kill()
    raise SystemExit(f"llama-server không lên được với {gguf}")


def doc_tap(tap: list[str], cho_phep_test: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in tap:
        if "test" in Path(p).name and not cho_phep_test:
            raise SystemExit(f"Không đo trên tập test khi đang chọn mức nén / ngưỡng: {p}")
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("meta", {}).get("type") in ("call", "ask"):  # chỉ lượt 1
                    r.setdefault("meta", {})["tap"] = Path(p).stem
                    rows.append(r)
    return rows


def chay(gguf: str, tap: list[str], out_file: str, tokenizer_dir: str = TOKENIZER_MAC_DINH,
         port: int = 8089, luong: int = 8, cho_phep_test: bool = False) -> None:
    from transformers import AutoTokenizer

    rows = doc_tap(tap, cho_phep_test)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    server = _bat_server(gguf, port, luong)
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            for i, r in enumerate(rows, 1):
                ids = tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
                d = dong_ket_qua(r, _goi(port, ids), tokenizer)
                d["tap"] = r["meta"]["tap"]
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                if i % 200 == 0 or i == len(rows):
                    print(f"  đã đo {i}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    finally:
        server.terminate()
    print(f"ghi vào {out_file}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Đo độ tự tin + chấm bản GGUF qua llama-server")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tap", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tokenizer", default=TOKENIZER_MAC_DINH)
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--luong", type=int, default=8, help="số luồng CPU cho llama-server")
    ap.add_argument("--cho-phep-test", action="store_true", help="chỉ sau khi mức nén + ngưỡng đã khoá")
    a = ap.parse_args()
    chay(a.gguf, a.tap, a.output, a.tokenizer, a.port, a.luong, a.cho_phep_test)


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
