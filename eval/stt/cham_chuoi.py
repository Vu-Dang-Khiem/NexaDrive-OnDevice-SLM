"""Chấm CẢ CHUỖI giọng nói (PLAN.md Phase 6 (3)): chữ STT (đã đổi số) -> student TRIỂN KHAI (GGUF q4_K_M
+ ngưỡng 0,80, qua đúng cổng `serve/mo_hinh_gguf.MoHinhGGUF`) -> chấm bằng `eval/metrics.cham_mot_cau` với
đáp án đóng băng. Mốc "chữ gốc" đi qua CÙNG đường (cùng cổng model, cùng luật ngưỡng) để so công bằng.
Kèm WER / CER sau chuẩn hoá. Chạy trong env `vdk1` (cần llama-server):

    conda run -n vdk1 python -m eval.stt.cham_chuoi runs/stt/bo_a_parakeet.jsonl \
        runs/stt/bo_a_phowhisper-small.jsonl
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from eval.metrics import cham_mot_cau, doc_output, nhom_cua_dap_an
from serve.so_chu import chuan_cau_stt, doi_so

# True: chữ STT -> `chuan_cau_stt` (đổi số + bỏ dấu câu cuối + hoa chữ đầu) trước student — thêm sau chẩn đoán
# bộ A (PLAN.md Phase 6), KHOÁ trước khi đo bộ B. False: chỉ đổi số (cách đo bộ A lần đầu).
CHUAN_HOA = "--chi-doi-so" not in sys.argv
# model student: mặc định bản GGUF mặc định của runtime (v5); bộ B chấm bằng BẢN TRIỂN KHAI hiện tại
# (`--gguf=models/gguf/lora-v61-s42-q4_k_m.gguf`, PLAN.md Phase 7 "NGƯỜI DÙNG CHỐT")
GGUF = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--gguf=")), None)
DUOI = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--duoi=")), "")  # hậu tố tên file kết quả


def chuan(s: str) -> list[str]:
    """Chuẩn hoá để tính WER/CER: NFC, thường hoá, đổi số chữ -> chữ số, bỏ dấu câu."""
    s = unicodedata.normalize("NFC", doi_so(s)).lower().replace("%", " phần trăm ")
    return re.sub(r"[^\w\s,]", " ", s).replace(",", " , ").split()


def _sua(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[len(b)]


def wer_cer(goc: str, stt: str) -> tuple[float, float]:
    a, b = chuan(goc), chuan(stt)
    return _sua(a, b) / max(1, len(a)), _sua(list(" ".join(a)), list(" ".join(b))) / max(1, len(" ".join(a)))


def main(tep_stt: list[str]) -> None:
    from serve.mo_hinh_gguf import MayChuLlama, MoHinhGGUF

    bo = {Path(p).stem: [json.loads(x) for x in open(p, encoding="utf-8")] for p in tep_stt}
    goc = next(iter(bo.values()))
    with (MayChuLlama(GGUF, port=8091) if GGUF else MayChuLlama()) as may:  # 8090 có thể đang bị demo giữ
        m = MoHinhGGUF(url=may.url)

        def cham(cau: str, completion: str) -> str:
            return cham_mot_cau(m.goi_ham(cau), completion)[0]

        kq_goc = {r["id"]: cham(r["cau"], r["completion"]) for r in goc}
        bang = {}
        for ten, rows in bo.items():
            ds = []
            for r in rows:
                w, c = wer_cer(r["cau"], r["chu_stt"])
                vao = chuan_cau_stt(r["chu_stt"]) if CHUAN_HOA else r["chu_doi_so"]
                ds.append({**r, "vao_student": vao, "ket_qua": cham(vao, r["completion"]),
                           "ket_qua_chu_goc": kq_goc[r["id"]],
                           "nhom": nhom_cua_dap_an(doc_output(r["completion"])), "wer": w, "cer": c})
            ra = Path(tep_stt[0]).parent / f"cham_{ten}{'' if CHUAN_HOA else '_chi_doi_so'}{DUOI}.jsonl"
            ra.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in ds), encoding="utf-8")
            bang[ten] = ds
    n = len(goc)
    dung_goc = sum(v == "dung" for v in kq_goc.values())
    print(f"MỐC chữ gốc qua cùng đường: {dung_goc}/{n} đúng ({100 * dung_goc / n:.1f}%)")
    for ten, ds in bang.items():
        dung = sum(d["ket_qua"] == "dung" for d in ds)
        mat = sum(d["ket_qua_chu_goc"] == "dung" and d["ket_qua"] != "dung" for d in ds)
        duoc = sum(d["ket_qua_chu_goc"] != "dung" and d["ket_qua"] == "dung" for d in ds)
        theo_nhom = {}
        for nh in ("ro_rang", "mo_ho", "ngoai_pham_vi"):
            N = [d for d in ds if d["nhom"] == nh]
            theo_nhom[nh] = f"{sum(d['ket_qua'] == 'dung' for d in N)}/{len(N)}"
        ms = sorted(d["ms"] for d in ds)
        rtf = statistics.mean(d["ms"] / 1000 / d["giay_am_thanh"] for d in ds)
        wer = 100 * statistics.mean(d["wer"] for d in ds)
        cer = 100 * statistics.mean(d["cer"] for d in ds)
        print(f"\n== {ten}\n  cả chuỗi đúng {dung}/{n} ({100 * dung / n:.1f}%) · mất so với chữ gốc {mat}"
              f" · được thêm {duoc} · theo nhóm {theo_nhom}")
        p50, p95 = ms[n // 2], ms[int(0.95 * n)]
        print(f"  WER {wer:.1f}% · CER {cer:.1f}% · STT p50 {p50:.0f} ms · p95 {p95:.0f} ms"
              f" · RTF tb {rtf:.2f}")
        print("  kiểu sai:", dict(Counter(d["ket_qua"] for d in ds if d["ket_qua"] != "dung").most_common(6)))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main([a for a in sys.argv[1:] if not a.startswith("--")])
