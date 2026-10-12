"""Tải TUẦN TỰ một file từ Modal Volume (SDK read_file), băm trong lúc tải; chỉ thay file đích khi mã băm khớp."""
import hashlib
import os
import sys
import time

import modal

sys.stdout.reconfigure(encoding="utf-8")
nguon, dich, bam_dung = sys.argv[1], sys.argv[2], sys.argv[3]
vol = modal.Volume.from_name("nexa-drive")
for lan in range(1, 6):
    tam = dich + ".part"
    h, n, t0 = hashlib.sha256(), 0, time.time()
    try:
        with open(tam, "wb") as f:
            for khoi in vol.read_file(nguon):
                f.write(khoi)
                h.update(khoi)
                n += len(khoi)
                if n // (256 << 20) != (n - len(khoi)) // (256 << 20):
                    print(f"  {n / 2**30:.2f} GB ({n / 2**20 / (time.time() - t0):.1f} MB/s)", flush=True)
    except Exception as e:  # mạng đứt -> thử lại từ đầu
        print(f"lần {lan}: lỗi {type(e).__name__}: {e} — thử lại", flush=True)
        time.sleep(20)
        continue
    if h.hexdigest() == bam_dung:
        os.replace(tam, dich)
        print(f"KHỚP mã băm ({n:,} byte) -> {dich}")
        sys.exit(0)
    print(f"lần {lan}: LỆCH mã băm ({n:,} byte) — thử lại", flush=True)
print("THẤT BẠI sau 5 lần")
sys.exit(1)
