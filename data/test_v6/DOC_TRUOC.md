# Tập test v6 — đọc hết trước khi bắt đầu

Mục đích: đo đúng các lỗi bạn gặp khi thử demo (2026-10-06) — "bật pha" không hiểu, "kính ghế phụ" bị từ chối,
"mở cốp" rồi "đóng đi" không hiểu, chỉnh nhiệt độ không có số, sưởi / làm lạnh, hai kính sau. Tập này làm và
**đóng băng TRƯỚC** khi sinh dữ liệu train v6 (Claude không được nhìn dữ liệu train mới rồi mới đặt đề).

**Không** đưa ChatGPT file nào khác ngoài file ghi trong từng bước. **Không** mở file test / train cũ khi làm.

`1_viet_cau.csv` có 210 dòng, 4 phần:

| Phần | Số dòng | Viết gì |
|---|---|---|
| A | 72 | lệnh **ngắn 1–4 chữ**, đủ 24 chức năng (mỗi chức năng 3 câu) |
| B | 48 | lệnh bình thường nhưng **gọi theo cách khác** (gọi tắt, từ địa phương, khẩu ngữ) — 24 chức năng × 2 |
| C | 30 | tính năng mới: chỉnh nhiệt **tương đối**, **sưởi / làm lạnh / tự động**, **cả hai kính** trước / sau |
| D | 60 | **hai lượt**: cột `luot_1` là câu lượt 1, cột `cau` là câu lượt 2 nói tiếp |

---

## Bước 1 — Viết câu (cuộc trò chuyện ChatGPT MỚI, chỉ đính kèm `1_viet_cau.csv`)

Tin nhắn 1 (dán nguyên văn):

```
File 1_viet_cau.csv đính kèm có 210 dòng. Mình đang làm một trợ lý giọng nói trên xe ô tô. Với mỗi dòng,
viết câu tiếng Việt mà người lái / người ngồi trong xe THẬT SỰ hay nói với trợ lý, đúng chức năng ở cột
"chuc_nang" và đúng yêu cầu ở cột "yeu_cau".

- Phần A, B, C: viết MỘT câu vào cột "cau". Đây là câu RA LỆNH / HỎI THẲNG (trợ lý làm luôn được).
- Phần D: viết HAI câu. Cột "luot_1" = câu lượt đầu, ra lệnh thẳng đúng chức năng ở cột "chuc_nang"
  (sau chữ "lượt 1:"). Cột "cau" = câu lượt hai nói NGAY SAU đó, đúng yêu cầu ở cột "yeu_cau".

Quy tắc:
- Viết ĐÚNG như người Việt nói trong xe: câu đời thường, có thể cộc lốc, nói tắt, có từ đệm (đi, nhé, cái,
  hộ, giùm...). Không viết kiểu văn viết hay câu mẫu sách giáo khoa.
- Đa dạng thật sự: đổi động từ, đổi cách gọi bộ phận, đổi giọng Bắc / Trung / Nam. Không lặp khuôn câu.
- Viết tiếng Việt CÓ DẤU, đúng chính tả (lỗi gõ sẽ có bộ riêng).
- Phần A: đúng 1–4 chữ.
- Phần D: câu lượt 2 phải nghe tự nhiên khi đứng ngay sau câu lượt 1 — nếu yêu cầu là "không nhắc tên bộ
  phận" thì đừng nhắc lại.
- Đọc lại từng câu và tự hỏi: "Có người thật nào nói đúng câu này trong xe không?" — nếu không, viết lại.

Trả lại CSV với đủ các cột: id, phan, chuc_nang, yeu_cau, luot_1, cau, ghi_chu. Giữ nguyên id, phan,
chuc_nang, yeu_cau. Làm theo lô: lần này chỉ làm id 1 đến 35.
```

Tin nhắn 2–6: `Tiếp id 36 đến 70, cùng quy tắc. Đừng lặp ý hay khuôn câu của các lô trước.`
(rồi 71–105, 106–140, 141–175, 176–210).

Ghép 6 lô → lưu **một file** `1_viet_cau_da_viet.csv` (nhớ Ctrl+S). Đọc lại: câu nào không tự nhiên, sai chức
năng, phần A quá 4 chữ, hay trùng ý câu khác thì **tự sửa tay** (không nhờ ChatGPT viết lại riêng).

Xong bước 1 → nhắn Claude. Claude tạo `2_cho_gan_nhan.csv` (bỏ cột chức năng / yêu cầu).

---

## Bước 2 — Gán nhãn (cuộc trò chuyện ChatGPT KHÁC)

Đính kèm **`2_cho_gan_nhan.csv` + `BANG_TRA_HAM.md`** (cùng thư mục này), dán prompt trong
`HUONG_DAN_gan_nhan.md`, làm theo lô 35 câu. Rồi bạn duyệt từng nhãn, lưu `2_gan_nhan_da_duyet.csv`.

## Bước 3 — Gửi Claude

Claude chuyển sang jsonl, kiểm schema, kiểm trùng với MỌI dữ liệu train, đóng băng (mã băm) — rồi mới sinh dữ
liệu train v6. Bộ "không dấu / lỗi gõ" Claude sinh bằng code từ chính tập đã đóng băng, bạn không phải làm.
