# eval/bao_cao — script tái tạo các bảng kết quả trong PLAN.md

Mọi script ĐỌC kết quả đã có trong `runs/` (dự đoán, xác suất) và `data/` (tập đóng băng) — **không chạy model,
không tốn tiền**. Chạy từ thư mục gốc dự án, env `vdk1`:

```
conda run -n vdk1 python eval/bao_cao/<ten>.py              # in bảng ra màn hình
conda run -n vdk1 python eval/bao_cao/so_dc.py ra.txt       # 3 script cũ (so_dc, so_cuoi, cham_bl) ghi ra file
```

Chuyển từ thư mục tạm của phiên làm việc vào dự án ngày 2026-10-05 (PLAN.md mục W). Khi chuyển đã chạy lại cả
23 script: tất cả chạy được; 11 script có file kết quả lưu sẵn trong `runs/` cho ra **trùng từng dòng**. Logic giữ
nguyên — chỉ sửa tự động import thừa / thứ tự import, và `kiem_token_gguf.py` (trỏ sang bản q4 vì f16 đã xoá;
sửa lỗi ghi file `\r\n`).

| Script | Bảng / kết luận (mục PLAN.md) | Đầu ra đã lưu |
|---|---|---|
| `so_dc.py`, `so_cuoi.py` | So các vòng v3 / v4 thay / v4 thêm / đối chứng / cấu hình cuối trên test sạch, 28 câu, dev (mục P, Q) | — |
| `cham_bl.py` | Baseline: 72B SIP, 1,5B SIP, student — theo nhóm (Phase 3, bảng kết quả tổng) | — |
| `mcnemar_72b.py` | McNemar student (đối chứng) vs 72B SIP từng hạt giống (bảng kết quả tổng) | `runs/mcnemar_72b.txt` |
| `phan_tich_tu_tin.py`, `mo_phong_nguong.py` | Phân bố xác suất token đầu + mô phỏng ngưỡng trên val/dev, bản đối chứng (mục S) | — |
| `cham_test_nguong.py` | Test với ngưỡng 0,90 đã khoá, bản đối chứng (mục S) | — |
| `cham_mo_ho_3.py`, `chan_doan_mo_ho_3.py` | Bộ mơ hồ 130 câu: student vs 72B + chẩn đoán (mục T) | — |
| `chon_ck_v5.py`, `so_v5.py` | Chọn checkpoint v5 + "không được tụt" so với đối chứng (mục U) | `runs/chon_ck_v5.txt`, `runs/so_v5.txt` |
| `cham_mo_ho_4.py` | mo_ho_4: v5 vs đối chứng vs 72B, McNemar (mục U) | `runs/mo_ho_4/ket_qua.txt` |
| `chung_minh_tied.py` | Chứng minh lỗi trọng số dùng chung khi gộp LoRA (mục V) | — |
| `chon_ck_lora.py`, `so_lora_full.py` | Chọn checkpoint LoRA; full vs LoRA + McNemar (mục V) | `runs/chon_ck_lora.txt`, `runs/so_lora_full.txt` |
| `so_cung_epoch.py` | Full vs LoRA cùng epoch (mục V) | `runs/so_cung_epoch.txt` |
| `chon_ck_lr2.py`, `so_ba_nhanh.py` | Cách B: full lr 2e-5; bảng ba nhánh (mục V) | `runs/chon_ck_lr2.txt`, `runs/so_ba_nhanh.txt` |
| `chon_nguong_lora.py` | Chọn ngưỡng riêng LoRA trên val/dev → 0,85 (mục V) | — |
| `cham_lora_085.py` | Test LoRA với ngưỡng 0,85 đã khoá (mục V) | `runs/so_lora_nguong_085.txt` |
| `tach_ro_rang.py` | Tách 74 câu rõ ràng: đúng ngay / hỏi lại đúng việc / làm sai (mục V) | — |
| `kiem_token_gguf.py` | 27 functional token + cắt token trong GGUF trùng HF (mục W) | — |
| `so_test_q4.py` | Test GGUF q4 + 0,80 vs HF (mục W) | `runs/gguf/so_test_q4.txt` |

Công cụ chính thức (có test) nằm ở `eval/`: `do_tu_tin.py`, `do_tu_tin_gguf.py`, `so_muc_nen.py`, `metrics.py`.
Script dựng dữ liệu một lần (mo_ho_3/4, v5, kiểm tay...) lưu nguyên trạng ở `luu_tru/script_phien/`.
