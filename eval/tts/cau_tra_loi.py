"""ĐO TTS — bước 1: dựng bộ câu TRỢ LÝ ĐỌC THẬT (PLAN.md "Đo độ rõ TTS"). Mỗi câu lệnh của các tập test
một lượt đi qua ĐÚNG luồng chạy thật (`serve.tac_tu.TacTu`, câu trả lời bằng câu mẫu như demo), model là
`MoHinhGia` ép trả ĐÁP ÁN ĐÚNG đóng băng -> chữ trợ lý nói ra khi model làm đúng. Lấy mọi lượt có chữ
(thực thi, hỏi lại, từ chối) + vài câu xã giao, bỏ trùng. Không đụng âm thanh — chạy trong env `vdk1`:

    conda run -n vdk1 python -m eval.tts.cau_tra_loi
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from serve.mo_hinh import MoHinhGia
from serve.tac_tu import TacTu

TAP = ["test_vi_sach_v2", "test_vi_bo_sung", "test_vi_mo_ho_4", "test_v6_mot_luot"]
XA_GIAO = ["Xin chào", "Bạn là ai", "Bạn làm được gì", "Cảm ơn nhé"]
RA = Path("runs/tts/cau_tra_loi.jsonl")
_USER = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)


def main() -> None:
    gap: dict[str, dict] = {}
    for tap in TAP:
        for dong in Path(f"data/processed/final/{tap}.jsonl").open(encoding="utf-8"):
            r = json.loads(dong)
            cau = _USER.search(r["prompt"]).group(1).strip()
            tac_tu = TacTu(MoHinhGia(bang={cau: r["completion"].replace("<|im_end|>", "")}))
            for su_kien in tac_tu.xu_ly(cau):
                chu = getattr(su_kien, "text", None)
                if chu and chu not in gap:
                    gap[chu] = {"tap": tap, "cau_lenh": cau, "loai": type(su_kien).__name__}
    for cau in XA_GIAO:
        for su_kien in TacTu(MoHinhGia()).xu_ly(cau):
            chu = getattr(su_kien, "text", None)
            if chu and chu not in gap:
                gap[chu] = {"tap": "xa_giao", "cau_lenh": cau, "loai": type(su_kien).__name__}
    RA.parent.mkdir(parents=True, exist_ok=True)
    with RA.open("w", encoding="utf-8") as f:
        for i, (chu, m) in enumerate(gap.items()):
            f.write(json.dumps({"id": f"t{i:03d}", "text": chu, **m}, ensure_ascii=False) + "\n")
    print(f"{len(gap)} câu trả lời khác nhau -> {RA}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
