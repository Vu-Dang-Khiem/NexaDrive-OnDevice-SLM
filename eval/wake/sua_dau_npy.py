"""Đặt lại phần đầu (header) .npy cho đoạn ĐẦU đã tải của đặc trưng ACAV100M (openWakeWord, 17 GB — chỉ tải
500.000 cửa sổ đầu bằng HTTP Range). Header gốc ghi 5.625.000 hàng -> sửa thành số hàng thật, giữ đúng 128
byte.

    python -m eval.wake.sua_dau_npy data/wake/owf/acav_500k.npy.part data/wake/owf/acav_500k.npy
"""

from __future__ import annotations

import os
import sys

DONG = 16 * 96 * 2  # một cửa sổ (16, 96) float16


def main() -> None:
    vao, ra = sys.argv[1], sys.argv[2]
    co = os.path.getsize(vao)
    with open(vao, "r+b") as f:
        dau = f.read(128)
        assert dau.startswith(b"\x93NUMPY") and b"'<f2'" in dau, "không phải file đặc trưng float16 mong đợi"
        n, du = divmod(co - 128, DONG)
        assert du == 0, f"tải thiếu: {du} byte lẻ"
        moi = f"{{'descr': '<f2', 'fortran_order': False, 'shape': ({n}, 16, 96), }}".encode()
        moi = b"\x93NUMPY\x01\x00" + (118).to_bytes(2, "little") + moi.ljust(117) + b"\n"
        assert len(moi) == 128
        f.seek(0)
        f.write(moi)
    os.replace(vao, ra)
    print(n, "cửa sổ ->", ra)


if __name__ == "__main__":
    main()
