"""Thêm functional token vào tokenizer + mở rộng model — bước ĐẦU TIÊN của Phase 4.

Vì sao cần: Qwen chưa từng thấy chuỗi `<nexa_11>`. Nếu không thêm vào từ điển, tokenizer
sẽ cắt nó thành nhiều mảnh vụn (`<`, `nexa`, `_`, `11`, `>`) — mỗi mảnh vô nghĩa, và việc
"chọn hàm" biến thành việc sinh đúng 5 mảnh liên tiếp thay vì phân loại 1 token. Toàn bộ
lợi thế của functional token (Octopus v2) nằm ở chỗ CHỌN HÀM = SINH 1 TOKEN.

Ba việc phải làm cùng nhau:
  1. Thêm 27 token vào tokenizer (25 hàm + `<nexa_end>` + `<nexa_ask>`)
  2. Nới lớp embedding VÀ lớp đầu ra (lm_head) thêm 27 ô
  3. Khởi tạo vector cho token mới = TRUNG BÌNH vector cũ + nhiễu nhỏ

Vì sao khởi tạo bằng trung bình chứ không để ngẫu nhiên: vector ngẫu nhiên nằm rất xa
vùng vector thật, model phải tốn nhiều bước chỉ để kéo nó về "vùng có nghĩa", train chậm
và dễ loạn ở những bước đầu. Trung bình đặt token mới ngay giữa bản đồ, quá trình train
chỉ cần đẩy nó về đúng chỗ. Nhiễu nhỏ để 27 token không khởi đầu giống hệt nhau (giống
hệt nhau thì gradient cũng giống nhau, chúng khó tách ra).

    python -m train.add_tokens --model models/Qwen2.5-1.5B-Instruct --output models/student-base
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from schema.loader import CATALOG

TY_LE_NHIEU = 0.02  # nhiễu = 2% độ lệch chuẩn của embedding cũ


def danh_sach_token_moi() -> list[str]:
    """27 token lấy thẳng từ schema — không hardcode ở đây để không bao giờ lệch nhau."""
    return CATALOG.special_tokens


def vector_khoi_tao(embeddings: Any, so_token_moi: int, seed: int = 42) -> Any:
    """Sinh ma trận (so_token_moi × hidden_size) = trung bình vector cũ + nhiễu nhỏ."""
    import torch

    cu = embeddings[:-so_token_moi] if so_token_moi else embeddings
    trung_binh = cu.mean(dim=0, keepdim=True).to(torch.float32)
    do_lech = cu.to(torch.float32).std()
    g = torch.Generator(device="cpu").manual_seed(seed)
    nhieu = torch.randn(so_token_moi, cu.shape[1], generator=g) * (do_lech * TY_LE_NHIEU)
    return (trung_binh + nhieu).to(embeddings.dtype)


def mo_rong_model(model: Any, so_token_moi: int, vocab_moi: int, seed: int = 42) -> None:
    """Nới embedding + lm_head lên `vocab_moi`, điền vector khởi tạo cho phần mới.

    Phải xử lý CẢ HAI: lớp vào (embedding) và lớp ra (lm_head). Quên lm_head thì model
    đọc được token mới nhưng không bao giờ sinh ra được nó.
    """
    import torch

    model.resize_token_embeddings(vocab_moi)
    with torch.no_grad():
        vao = model.get_input_embeddings().weight
        vao[-so_token_moi:] = vector_khoi_tao(vao.data, so_token_moi, seed)

        ra = model.get_output_embeddings()
        # Nhiều model buộc lm_head dùng chung trọng số với embedding (tied weights) —
        # lúc đó gán 1 lần là đủ, gán thêm chỉ ghi đè lên chính nó.
        if ra is not None and ra.weight.data_ptr() != vao.data_ptr():
            ra.weight[-so_token_moi:] = vector_khoi_tao(ra.weight.data, so_token_moi, seed + 1)


def them_token(model_dir: str | Path, out_dir: str | Path, seed: int = 42) -> dict[str, Any]:
    """Nạp model gốc -> thêm token -> lưu ra `out_dir`. Chạy được trên CPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token_moi = danh_sach_token_moi()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir))

    vocab_cu = len(tokenizer)
    so_them = tokenizer.add_special_tokens({"additional_special_tokens": token_moi})
    vocab_moi = len(tokenizer)
    if so_them != len(token_moi):
        raise RuntimeError(f"chỉ thêm được {so_them}/{len(token_moi)} token — model gốc đã có sẵn?")

    mo_rong_model(model, so_them, vocab_moi, seed)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(out_dir))
    model.save_pretrained(str(out_dir))

    # Kiểm ngay: mỗi token mới phải được cắt thành ĐÚNG 1 mảnh
    loi = [t for t in token_moi if len(tokenizer.encode(t, add_special_tokens=False)) != 1]
    if loi:
        raise RuntimeError(f"token bị cắt thành nhiều mảnh: {loi}")

    return {
        "vocab_cu": vocab_cu,
        "vocab_moi": vocab_moi,
        "so_token_them": so_them,
        "token": token_moi,
        "luu_tai": str(out_dir),
    }


def kiem_ranh_gioi(tokenizer: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Kiểm chỗ NỐI giữa prompt và completion có bị tokenizer ghép token không.

    Cách chuẩn bị data của dự án mã hoá RIÊNG 2 phần rồi nối:
        input_ids = encode(prompt) + encode(completion)
    Cách này khớp đúng với lúc chạy thật (lúc đó model chỉ có prompt, cũng được mã hoá
    riêng). Nhưng nếu tokenizer vốn sẽ GHÉP token cuối của prompt với token đầu của
    completion khi mã hoá cả chuỗi, thì dãy id lúc train khác dãy id của cùng đoạn chữ
    đó viết liền — cần BIẾT điều đó, không phải để sửa mà để không ngạc nhiên về sau.

    Mẫu `call`/`ask` an toàn tuyệt đối vì completion mở đầu bằng special token
    (`<nexa_i>`) — special token không bao giờ bị ghép. Chỉ mẫu `reply` (mở đầu bằng
    chữ thường) mới cần soi.
    """
    lech: list[dict[str, str]] = []
    for r in rows:
        rieng = tokenizer.encode(r["prompt"], add_special_tokens=False) + tokenizer.encode(
            r["completion"], add_special_tokens=False
        )
        chung = tokenizer.encode(r["prompt"] + r["completion"], add_special_tokens=False)
        if rieng != chung:
            lech.append(
                {
                    "loai": r.get("meta", {}).get("type", "?"),
                    "cuoi_prompt": r["prompt"][-20:],
                    "dau_completion": r["completion"][:20],
                }
            )
    return {
        "so_mau_kiem": len(rows),
        "so_mau_lech": len(lech),
        "vi_du_lech": lech[:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Thêm functional token vào model gốc")
    ap.add_argument("--model", required=True, help="thư mục model gốc (Qwen2.5-1.5B-Instruct)")
    ap.add_argument("--output", required=True, help="thư mục lưu model đã thêm token")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kiem-data", default=None, help="file JSONL để kiểm ranh giới token + độ dài")
    ap.add_argument("--so-mau-kiem", type=int, default=200)
    args = ap.parse_args()

    bao_cao = them_token(args.model, args.output, args.seed)
    print(f"vocab: {bao_cao['vocab_cu']} -> {bao_cao['vocab_moi']} (+{bao_cao['so_token_them']})")
    print(f"token mới: {', '.join(bao_cao['token'][:3])} ... {bao_cao['token'][-1]}")
    print(f"đã lưu: {bao_cao['luu_tai']}")

    if args.kiem_data:
        import json as _json

        from transformers import AutoTokenizer

        from train.du_lieu import doc_jsonl, thong_ke_do_dai

        tok = AutoTokenizer.from_pretrained(args.output)
        rows = doc_jsonl(args.kiem_data)[: args.so_mau_kiem]

        print("\n── độ dài mẫu (để chọn do_dai_toi_da, và để NHÌN LÀ BIẾT mask có hỏng không) ──")
        print(_json.dumps(thong_ke_do_dai(tok, rows), ensure_ascii=False, indent=2))

        print("\n── ranh giới prompt|completion ──")
        kq = kiem_ranh_gioi(tok, rows)
        print(_json.dumps(kq, ensure_ascii=False, indent=2))
        if kq["so_mau_lech"]:
            print(
                f"  LƯU Ý: {kq['so_mau_lech']}/{kq['so_mau_kiem']} mẫu có ranh giới bị ghép token.\n"
                "  Không phải lỗi (cách mã hoá riêng vẫn khớp lúc chạy thật), nhưng ghi nhận lại."
            )


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
