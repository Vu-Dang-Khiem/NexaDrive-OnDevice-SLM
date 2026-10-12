# Bộ câu mơ hồ thứ 3 — cách làm (đọc hết trước khi bắt đầu)

Mục đích: thêm 130 câu mơ hồ vào tập đánh giá để kiểm định "student hỏi lại tốt hơn 72B" cho chắc
(PLAN.md, BẢNG KẾT QUẢ TỔNG — 67 câu hiện có quá ít). Làm HAI bước, mỗi bước một cuộc trò chuyện
ChatGPT MỚI. **Không** đưa ChatGPT bất kỳ file nào khác trong dự án.

---

## Bước 1 — Sinh câu (cuộc trò chuyện mới, chỉ dán đoạn dưới + file `1_viet_cau.csv`)

Dán nguyên văn đoạn trong khung, rồi dán file CSV theo từng lô khoảng 26 dòng (5 lô). Mỗi lô nhắc lại:
"Tiếp lô sau, đừng lặp ý hay khuôn câu của các lô trước."

```
Mình đang làm một trợ lý giọng nói trên xe ô tô. Mình cần bạn viết câu tiếng Việt mà người lái
hoặc người ngồi trong xe có thể buột miệng nói.

Với mỗi dòng trong file, viết MỘT câu vào cột "cau". Câu đó phải khiến một trợ lý thông minh nghe
xong hiểu rằng người nói đang cần chức năng ghi ở cột "chuc_nang" — nhưng vì câu không nói thẳng,
trợ lý sẽ HỎI LẠI kiểu "Bạn có muốn tôi ... không?".

Quy tắc quan trọng nhất: KHÔNG ra lệnh thẳng. Chỉ than thở, kể tình huống, kể cảm giác, nói bâng
quơ, cảm thán. Không dùng các động từ sai bảo như "bật", "mở", "đóng", "khoá", "hạ", "nâng", "tìm",
"chỉnh", "tăng", "giảm", "dẫn đường"; không nhắc thẳng tên chức năng.

Thêm:
- Mỗi câu một tình huống khác nhau; đừng viết các câu chỉ đổi vài chữ.
- Trộn độ dài: khoảng một nửa câu ngắn, cộc lốc (3–8 chữ); một nửa câu dài hơn, kể hoàn cảnh
  (12–30 chữ), ví dụ đang đi đâu, với ai, thời tiết, cảm giác.
- Viết như người thật nói: có thể có từ đệm, nói tắt, không cần đúng ngữ pháp. Chỉ tiếng Việt.
- Câu vẫn phải đủ manh mối để một người bình thường đoán ra nhu cầu đó.
- Câu nào bạn thấy khó hoặc dễ bị hiểu sang chức năng khác, ghi lý do vào cột "ghi_chu".

Trả lại đúng định dạng CSV với 4 cột: id, chuc_nang, cau, ghi_chu. Giữ nguyên id và chuc_nang.
```

Xong 5 lô → ghép lại, lưu `1_viet_cau_da_viet.csv` (CSV UTF-8) trong thư mục này.

**Bạn tự đọc lại trước khi sang bước 2:** câu nào lỡ ra lệnh thẳng, trùng ý câu khác, hoặc có từ
tiếng Anh thì sửa tay hoặc xoá chữ trong cột `cau` (để trống) và ghi lý do vào `ghi_chu`.

---

## Bước 2 — Gán nhãn (cuộc trò chuyện MỚI, khác bước 1)

Đưa ChatGPT file `../test_tay/HUONG_DAN.md` (bảng tra hàm) + `HUONG_DAN_gan_nhan.md` + file
`1_viet_cau_da_viet.csv` **đã bỏ cột `chuc_nang`** (để nhãn theo đúng câu, không theo chức năng mình
nhắm tới). Bảng tra hàm có câu ví dụ — vì vậy nó chỉ được đưa ở bước 2, KHÔNG bao giờ ở bước 1.

Sau đó **bạn duyệt và sửa từng nhãn** như các bộ trước, lưu `2_gan_nhan_da_duyet.csv` với các cột
`id, cau, nhan, ghi_chu`.

---

## Bước 3 — Gửi lại cho Claude

Claude sẽ: chuyển sang jsonl bằng `eval/chuyen_test.py`; kiểm lỗi schema; **kiểm trùng** với toàn bộ
dữ liệu train/val/dev/test (`eval/kiem_nhiem.py`); báo số câu theo hàm; **đóng băng** bằng mã băm.
Chỉ sau đó mới chấm student s42, s7 và 72B — một lần, theo cách kiểm đã ghi khoá trong PLAN.md.

**Không** chạy câu nào qua model trước khi đóng băng.
