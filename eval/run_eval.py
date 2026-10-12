"""In bảng kết quả từ tập test + file dự đoán của model — chạy local, miễn phí.

Tách làm 2 bước (chấm ở đây, sinh dự đoán ở nơi khác) để:
  • chấm lại được nhiều lần mà không cần bật GPU;
  • so nhiều model chỉ bằng cách đổi file dự đoán;
  • và quan trọng nhất: logic chấm test được bằng dữ liệu hardcode (tests/test_metrics.py).

File dự đoán là JSONL, mỗi dòng {"prompt": ..., "output": ...} — đúng `prompt` trong tập
test và `output` là chuỗi thô model sinh ra.

    python -m eval.run_eval --test data/processed/final/test_vi.jsonl \\
                            --du-doan runs/student-full/du_doan.jsonl \\
                            --cau-sai runs/student-full/cau_sai.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.metrics import cau_sai, cham_toan_bo

# Ngưỡng đạt đã chốt (PLAN.md Phase 4). Đặt TRƯỚC khi train để tránh chạy xong mới
# chọn con số đẹp nhất mà khoe.
NGUONG = {
    "ro_rang": 95.0,
    "ngoai_pham_vi": 90.0,
    "mo_ho": 70.0,
    "tu_choi_nham_toi_da": 5.0,
    # Lượt 2 (câu trả lời tự nhiên). Bịa số trên xe là lỗi AN TOÀN, không phải lỗi văn
    # phong — "xăng còn 60%" trong khi thực tế 12% có thể khiến người ta hết xăng giữa
    # đường. Nên ngưỡng đặt sát 0 chứ không nới như các nhóm khác.
    "bia_so_toi_da": 1.0,
}


def doc_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _bang(tieu_de: str, bang: dict[str, dict[str, Any]]) -> str:
    dong = [f"\n{tieu_de}", f"  {'nhóm':28} {'số câu':>7} {'đúng %':>8}"]
    for k, v in bang.items():
        dong.append(f"  {k:28} {v['so_cau']:>7} {v['dung_%']:>8.1f}")
    return "\n".join(dong)


def in_bao_cao(bc: dict[str, Any]) -> None:
    print(f"\nĐã chấm {bc['so_cau_da_cham']} câu" + (
        f" (THIẾU {bc['thieu_du_doan']} câu không có dự đoán)" if bc["thieu_du_doan"] else ""
    ))
    print(_bang("=== THEO NHÓM (đây là bảng chính, KHÔNG gộp thành 1 số) ===", bc["theo_nhom"]))
    print(_bang("=== NHÓM PHỤ (tách câu từ chối khó / dễ) ===", bc["theo_nhom_phu"]))
    print(_bang("=== THEO HÀM ===", bc["theo_ham"]))

    print("\n=== KIỂU SAI ===")
    for kieu, n in bc["loi_theo_kieu"].items():
        print(f"  {kieu:16} {n:>5}")
    print("\n=== TỈ LỆ LỖI (%) ===")
    for ten, v in bc["ty_le_loi_%"].items():
        print(f"  {ten:16} {v:>6.1f}")
    print(f"\n=== THAM SỐ === P={bc['tham_so']['precision_%']}%  "
          f"R={bc['tham_so']['recall_%']}%  F1={bc['tham_so']['f1_%']}%")

    if bc["hay_nham_nhat"]:
        print("\n=== HAY NHẦM NHẤT (sửa data ở đúng các cặp này) ===")
        for x in bc["hay_nham_nhat"]:
            print(f"  {x['dap_an']:22} -> {x['model_chon']:22} {x['so_lan']:>4} lần")


def kiem_nguong(bc: dict[str, Any]) -> bool:
    """In đạt/chưa đạt theo ngưỡng đã chốt trước. Trả về True nếu đạt hết."""
    nhom = bc["theo_nhom"]
    kiem = [
        ("rõ ràng", nhom.get("ro_rang", {}).get("dung_%", 0), NGUONG["ro_rang"], "≥"),
        ("ngoài phạm vi", nhom.get("ngoai_pham_vi", {}).get("dung_%", 0), NGUONG["ngoai_pham_vi"], "≥"),
        ("mơ hồ", nhom.get("mo_ho", {}).get("dung_%", 0), NGUONG["mo_ho"], "≥"),
        ("từ chối nhầm", bc["ty_le_loi_%"]["tu_choi_nham"], NGUONG["tu_choi_nham_toi_da"], "≤"),
    ]
    # Chỉ kiểm lượt 2 khi tập đang chấm CÓ mẫu reply (tập test viết tay thì không có).
    if "tra_loi" in nhom:
        kiem.append(("bịa số (lượt 2)", bc["ty_le_loi_%"].get("bia_so", 0.0),
                     NGUONG["bia_so_toi_da"], "≤"))
    print("\n=== ĐỐI CHIẾU NGƯỠNG ĐÃ CHỐT TRƯỚC ===")
    dat_het = True
    for ten, thuc_te, nguong, dau in kiem:
        dat = thuc_te >= nguong if dau == "≥" else thuc_te <= nguong
        dat_het &= dat
        print(f"  {ten:16} {thuc_te:>6.1f}%  (cần {dau} {nguong}%)  {'ĐẠT' if dat else 'CHƯA ĐẠT'}")
    return dat_het


def main() -> None:
    ap = argparse.ArgumentParser(description="Chấm điểm model trên tập test")
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--du-doan", required=True, type=Path, help="JSONL {prompt, output}")
    ap.add_argument("--cau-sai", type=Path, default=None, help="ghi danh sách câu sai ra đây")
    ap.add_argument("--json", type=Path, default=None, help="ghi báo cáo dạng JSON ra đây")
    ap.add_argument("--luat", choices=["C", "cu"], default="C",
                    help="C: bỏ tham số = mặc định, câu mơ hồ chỉ chấm hàm + hướng (PLAN.md); cu: so khít")
    args = ap.parse_args()

    dap_an = doc_jsonl(args.test)
    du_doan = {r["prompt"]: r["output"] for r in doc_jsonl(args.du_doan)}

    bc = cham_toan_bo(dap_an, du_doan, args.luat)
    print(f"Luật chấm: {args.luat}")
    in_bao_cao(bc)
    kiem_nguong(bc)

    if args.cau_sai:
        ds = cau_sai(dap_an, du_doan, args.luat)
        args.cau_sai.parent.mkdir(parents=True, exist_ok=True)
        with open(args.cau_sai, "w", encoding="utf-8") as f:
            for r in ds:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nĐã ghi {len(ds)} câu sai -> {args.cau_sai} (đọc tay để biết sửa data ở đâu)")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(bc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Đã ghi báo cáo -> {args.json}")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
