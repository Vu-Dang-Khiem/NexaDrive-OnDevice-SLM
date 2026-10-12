# luu_tru/script_phien — script một lần, lưu nguyên trạng

Chép NGUYÊN TRẠNG (2026-10-05) từ thư mục tạm của các phiên làm việc 2026-10-01 → 10-05, để giữ dấu vết cách dựng
dữ liệu và các lần kiểm tra lặt vặt. **Không bảo trì, không lint** (`pyproject.toml` loại thư mục này). Kết quả của
chúng đã ghi trong PLAN.md; dữ liệu chúng tạo ra đã đóng băng (mã băm trong `tests/test_luat_c.py`). Có thể không còn
chạy được (đường dẫn tạm, file đã xoá). Script tái tạo BẢNG KẾT QUẢ nằm ở `eval/bao_cao/`.

| Nhóm | Script |
|---|---|
| Bộ mơ hồ 3 (mục T) | `tao_mo_ho_3`, `kiem_mo_ho_3`, `tao_viet_lai`, `ghep_mo_ho_3`, `kiem_cuoi_mo_ho_3`, `kiem_nhan_mo_ho_3`, `ghep_nhan`, `sua_118`, `chot_mo_ho_3`, `dem_mo_ho` |
| Bộ mơ hồ 4 (mục U) | `tao_mo_ho_4`, `kiem_mo_ho_4`, `tao_buoc2_mo_ho_4`, `kiem_nhan_mo_ho_4`, `soi_nhan_mo_ho_4`, `xem_dong_sua`, `chot_mo_ho_4` |
| Dữ liệu v5 (mục U) | `dem_train`, `doc_v5thu`, `doc_v5thu2`, `soi_cung_khuon`, `gop_v5`, `dem_dest`, `doc_kiem_tay`, `dem_loi_he_thong`, `do_dai_v5` |
| Vòng 4 / đối chứng (mục P, Q) | `chon_dc`, `soi_dc`, `chon_cuoi`, `soi_cuoi`, `den_pha`, `do_dai` |
| Ngưỡng — thử nghiệm (mục S) | `xem_tap`, `soi_sai`, `thu_che_do`, `thu_tok` |
| Tải model + GGUF (mục W) | `so_bam` (so SHA256 với Modal), `tai_tuan_tu` (tải tuần tự + băm — **cách tải file lớn nên dùng**), `thu_tren_may`, `so_token`, `thu_server` |
