"""So SHA256 model tải về với mã băm tính trên Modal (train/modal_train.py::ma_bam)."""
import hashlib
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
MODAL = {
    "added_tokens.json": "33f58e9c41cc48dc0d0ecbf80ed97f34b68a607fa6dc25f6afb96e7a70ff8385",
    "chat_template.jinja": "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f",
    "config.json": "7b46493c31c2dc9f55eff0d04e5ecf4641e2497a4cb00dc771f4685690260094",
    "generation_config.json": "cd3e781699b69eb78534f464b0d57590b9babf52d8123d71eb99f001eb5e8a00",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors": "759b366cd23c78f16f259328d15fe71a44b368d583bdcd0b87954301ed6174dc",
    "special_tokens_map.json": "5426528f14be743f9886061382b5fafdb883c8f231d53524bae7adf0d8f716f5",
    "tokenizer.json": "1ea1531147ab4af8cd27f9f7a14fe97e72065f5d26a024119ba47d88a4a4212d",
    "tokenizer_config.json": "bb1b82f9428f826beb524c902b318fcaa08e7b0ecde20e3554a15493130af147",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
d = Path(sys.argv[1])
loi = 0
for ten, bam in MODAL.items():
    p = d / ten
    if not p.exists():
        print(f"THIẾU  {ten}")
        loi += 1
        continue
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for k in iter(lambda: f.read(1 << 24), b""):
            h.update(k)
    ok = h.hexdigest() == bam
    loi += not ok
    print(f"{'KHỚP ' if ok else 'LỆCH '} {ten} ({p.stat().st_size:,} byte)")
print("==> TẤT CẢ KHỚP" if loi == 0 else f"==> {loi} FILE LỖI — KHÔNG DÙNG")
