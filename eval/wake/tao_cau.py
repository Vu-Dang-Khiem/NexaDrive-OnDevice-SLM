"""Wake-word — bước 1: danh sách CÂU để VieNeu đọc (PLAN.md Phase 6b, "Wake-word — KẾ HOẠCH"). Chạy
trong env `vdk1` (cần schema / runtime):

    python -m eval.wake.tao_cau

Ghi `data/wake/cau.json`:
  duong      cách viết "hey nexa" kiểu người Việt đọc (VieNeu đọc ra "hay nét xa", "hây nét xa"... — đã thử)
  gan_giong  câu / từ GẦN giống (dễ báo nhầm) — có "Nexa", "NexaAI" vì loa của trợ lý tự nói tên mình
  lenh       câu lệnh xe lấy từ dữ liệu TRAIN v6.1 (lượt 1) — câu nói ngay sau "hey nexa", không được báo
  tu_thuc    câu trả lời của trợ lý (câu mẫu + câu đáp xã giao) — để đo "tự đánh thức" (tiêu chí c),
             KHÔNG train
"""

from __future__ import annotations

import json
import random
from pathlib import Path

RA = Path("data/wake/cau.json")
SEED = 2026_10_08
DUONG = ["Hey Nexa", "Hây Néc xa", "Hây Nếch xa", "Hê Néc xa", "Hây Nếch-xa", "Hey Nexa ơi"]
GAN_GIONG = [
    "Nexa",
    "NexaAI",
    "Mình là NexaAI",
    "Chào bạn, mình là NexaAI",
    "Nexa à",
    "Hây",
    "Hey",
    "Hay đấy",
    "Hay nhỉ",
    "Hay quá",
    "Hay là thôi",
    "Hây da",
    "Hết xăng",
    "Hết xăng rồi",
    "Nét xa",
    "Nếch",
    "Hãy nói xa hơn",
    "Hai xe",
    "Hai nét",
    "Né xa ra",
    "Hình như xa lắm",
    "Hey Siri",
    "Hey Google",
    "OK Google",
    "Alexa",
    "Này",
    "Ê",
    "Nè",
    "Hây nha",
    "Hết sạch rồi",
    "Nét xa quá",
    "Hai năm sau",
    "Hay nói xạo",
    "Hấp xả",
    "Nhà xa quá",
    "Đi xa nha",
    "Nghe xa xa",
    "Hay nghe nhạc",
    "Hãy nghe này",
    "Ê nghe nè",
]


def main() -> None:
    from gen.records import query_from_call_prompt
    from schema.loader import CATALOG
    from serve import tra_loi
    from serve.contracts import AssistantTurn, ClarifyTurn
    from serve.mo_hinh import MoHinhGia
    from serve.tac_tu import TacTu

    rows = [json.loads(x) for x in open("data/processed/final_v6_1/train_vi.jsonl", encoding="utf-8")]
    lenh = sorted(
        {
            query_from_call_prompt(r["prompt"])
            for r in rows
            if r["completion"].lstrip().startswith("<nexa") and "Lượt trước" not in r["prompt"]
        }
    )
    random.Random(SEED).shuffle(lenh)

    tu_thuc = {tra_loi.TU_CHOI, tra_loi.CHUA_HIEU, tra_loi.CHUA_RO, tra_loi.HUY}
    for _, _, cau_dap in tra_loi._NGOAI_PHAM_VI:
        tu_thuc.update(cau_dap)
    for f in CATALOG.functions:  # câu trả lời / hỏi lại mẫu cho từng hàm (model giả + xe mô phỏng)
        for vd in f.examples:
            for s in TacTu(MoHinhGia()).xu_ly(vd):
                if isinstance(s, (AssistantTurn, ClarifyTurn)) and s.text.strip():
                    tu_thuc.add(s.text.strip())
    RA.parent.mkdir(parents=True, exist_ok=True)
    RA.write_text(
        json.dumps(
            {"duong": DUONG, "gan_giong": GAN_GIONG, "lenh": lenh[:300], "tu_thuc": sorted(tu_thuc)},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    so_lenh = min(300, len(lenh))
    print(f"dương {len(DUONG)} · gần giống {len(GAN_GIONG)} · lệnh {so_lenh} · tự thức {len(tu_thuc)}")


if __name__ == "__main__":
    main()
