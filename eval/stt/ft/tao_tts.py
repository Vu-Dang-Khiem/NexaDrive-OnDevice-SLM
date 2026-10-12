"""Dữ liệu TTS cho FINE-TUNE STT (PLAN.md "FINE-TUNE STT TRÊN CÂU LỆNH XE", 2026-10-12) — chạy trên máy, $0.

Câu: lệnh từ `final_v6_1/train_vi` (dữ liệu TRAIN của model gọi hàm — đã kiểm nhiễm với test), bỏ câu
nhiễu / hai lượt / lượt trước thừa / có chữ viết tắt Latin (FM, USB…: không chắc STT viết thế nào); kiểm nhiễm
lại với mọi tập test và bộ ghi âm B/C/D/E. Số đổi sang CHỮ ("22 độ" -> "hai mươi hai độ") để TTS đọc đúng và
nhãn khớp kiểu PhoWhisper viết. Giọng: 23 / 25 giọng VieNeu (GIỮ LẠI 2 giọng chỉ để kiểm). Âm thanh
SẠCH 16 kHz — làm bẩn ngẫu nhiên lúc train.

    conda run --no-capture-output -n giong_noi python -m eval.stt.ft.tao_tts [--so-cau 5000]
Ra: data/stt/ft_tts/{wav/*.wav, manifest.jsonl}; chạy lại thì tiếp từ câu chưa tạo.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from serve.so_chu import doi_so

RA = Path("data/stt/ft_tts")
GIU_LAI = {"Ngọc Trân", "Minh Triết"}  # giọng KHÔNG dùng để train (Trung · nữ, Nam · nam) — chỉ để kiểm
VIET_TAT = re.compile(r"\b[A-Za-z]{2,}\b|[A-Z]/[A-Z]")  # FM, USB, ATM, A/C, odo… -> bỏ câu
SO = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def doc_so(n: int) -> str:
    """Số nguyên 0–999 999 -> chữ kiểu đọc thường (21 -> "hai mươi mốt", 105 -> "một trăm linh năm")."""
    if n < 10:
        return SO[n]
    if n < 100:
        c, d = divmod(n, 10)
        dau = "mười" if c == 1 else f"{SO[c]} mươi"
        if d == 0:
            return dau
        duoi = {1: "mốt" if c > 1 else "một", 5: "lăm", 4: "tư" if c > 1 else "bốn"}.get(d, SO[d])
        return f"{dau} {duoi}"
    if n < 1000:
        t, r = divmod(n, 100)
        return f"{SO[t]} trăm" + ("" if r == 0 else f" linh {SO[r]}" if r < 10 else f" {doc_so(r)}")
    ng, r = divmod(n, 1000)
    if r == 0:
        duoi = ""
    elif r >= 100:
        duoi = f" {doc_so(r)}"
    elif r >= 10:
        duoi = f" không trăm {doc_so(r)}"
    else:
        duoi = f" không trăm linh {SO[r]}"
    return f"{doc_so(ng)} nghìn{duoi}"


def so_sang_chu(cau: str) -> str:
    """Đổi số + đơn vị trong câu sang chữ để đọc: "5km" -> "năm ki lô mét", "30%" -> "ba mươi phần trăm",
    "92.5" / "92,5" -> "chín mươi hai phẩy năm"."""
    def thay(m: re.Match) -> str:
        nguyen, le, dv = m.group(1), m.group(2), (m.group(3) or "").lower()
        chu = doc_so(int(nguyen)) + (" phẩy " + " ".join(SO[int(c)] for c in le) if le else "")
        return chu + {"%": " phần trăm", "km": " ki lô mét", "°c": " độ", "°": " độ"}.get(dv, "")
    return re.sub(r"(\d{1,6})(?:[.,](\d{1,2}))?(?:\s*(%|km|°C|°c|°)(?!\w))?", thay, cau)


def chon_cau(n: int) -> list[dict]:
    from eval.kiem_nhiem import tim_nhiem
    from eval.metrics import doc_output, nhom_cua_dap_an
    from gen.records import query_from_call_prompt

    def doc(p: Path) -> list[dict]:
        return [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]

    rows = []
    for r in doc(Path("data/processed/final_v6_1/train_vi.jsonl")):
        m = r["meta"]
        bo_v6 = str(m.get("v6", "")).startswith(("nhieu", "luot"))
        if m["type"] not in ("call", "ask") or "fn_1" in m or bo_v6:
            continue
        cau = query_from_call_prompt(r["prompt"])
        if "\n" in cau or VIET_TAT.search(cau) or not 1 <= len(cau.split()) <= 25:
            continue
        noi = so_sang_chu(cau)
        if re.search(r"\d", noi):
            continue  # số dạng lạ (giờ 7:30, dải 3-4…) -> bỏ
        rows.append({"cau": cau, "noi": noi, "prompt": r["prompt"], "fn": m["fn"], "meta": {"fn": m["fn"]},
                     "nhom": nhom_cua_dap_an(doc_output(r["completion"]))})
    cam = [r for p in sorted(Path("data/processed/final").glob("test_*.jsonl")) for r in doc(p)]
    cam += [r for b in "bcde" for r in doc(Path(f"data/stt/bo_{b}/danh_sach.jsonl"))]
    nhiem = set(tim_nhiem(rows, cam)["chi_so"])
    rows = list({r["cau"]: r for i, r in enumerate(rows) if i not in nhiem}.values())
    random.Random(2032).shuffle(rows)
    # cân theo hàm: chia đều cho mỗi (nhóm, hàm), hết suất thì lấy phần còn lại theo thứ tự đã xáo
    theo = defaultdict(list)
    for r in rows:
        theo[(r["nhom"], r["fn"])].append(r)
    suat = max(1, n // len(theo))
    chon = [r for ds in theo.values() for r in ds[:suat]]
    da = {id(r) for r in chon}
    chon += [r for r in rows if id(r) not in da][: max(0, n - len(chon))]
    return chon[:n]


def main() -> None:
    import soundfile as sf
    import soxr
    from vieneu import Vieneu

    ap = argparse.ArgumentParser()
    ap.add_argument("--so-cau", type=int, default=5000)  # không dùng "--n": `conda run` hiểu nhầm
    a = ap.parse_args()
    (RA / "wav").mkdir(parents=True, exist_ok=True)
    tep_cau = RA / "cau.jsonl"
    if not tep_cau.exists():  # khoá danh sách câu ở lần chạy đầu
        tep_cau.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chon_cau(a.so_cau)),
                           encoding="utf-8")
    cau = [json.loads(x) for x in tep_cau.open(encoding="utf-8")]
    tts = Vieneu()
    giong = [ten for _, ten in tts.list_preset_voices() if ten not in GIU_LAI]
    rng = random.Random(2033)
    gan = [giong[rng.randrange(len(giong))] for _ in cau]
    t0 = time.time()
    with (RA / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(cau):
            tep = RA / "wav" / f"t{i:05d}.wav"
            if not tep.exists():
                am = tts.infer(r["noi"], voice=gan[i])
                sf.write(str(tep), soxr.resample(am, tts.sample_rate, 16000), 16000, subtype="PCM_16")
            nhan = r["noi"].lower().rstrip(" .!?") + "."  # kiểu PhoWhisper viết: chữ thường, chấm cuối câu
            assert doi_so(nhan) is not None
            f.write(json.dumps({"tep": f"wav/{tep.name}", "nhan": nhan, "giong": gan[i], "cau": r["cau"],
                                "nhom": r["nhom"], "fn": r["fn"]}, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(cau)} ({time.time() - t0:.0f}s)", flush=True)
    print("xong", len(cau), "câu ->", RA)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
