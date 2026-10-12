# Bộ câu mơ hồ thứ 4 — đánh giá vòng 5 (đọc hết trước khi bắt đầu)

Mục đích: bộ mơ hồ MỚI để đánh giá model v5 (PLAN.md mục U — tiêu chí đã khoá). Bộ mo_ho_3 đã bị chấm một
lần nên không dùng để kết luận vòng này. Làm y hệt mo_ho_3, rút kinh nghiệm từ lần đó.

**Không** đưa ChatGPT file nào khác. **Không** mở file của mo_ho_3 khi làm bộ này.

---

## Bước 1 — Sinh câu (cuộc trò chuyện ChatGPT MỚI, chỉ đính kèm `1_viet_cau.csv`)

Tin nhắn 1 (dán nguyên văn):

```
File 1_viet_cau.csv đính kèm có 130 dòng. Mình đang làm một trợ lý giọng nói trên xe ô tô. Với mỗi
dòng, viết MỘT câu tiếng Việt vào cột "cau" — câu mà người ngồi trong xe thật sự hay buột miệng
nói, khiến một trợ lý thông minh hiểu họ đang cần chức năng ở cột "chuc_nang", nhưng vì không nói
thẳng nên trợ lý sẽ hỏi lại "Bạn có muốn tôi ... không?".

Quy tắc:
- KHÔNG ra lệnh thẳng: không dùng "bật", "mở", "đóng", "khoá", "hạ", "nâng", "tìm", "chỉnh",
  "tăng", "giảm", "dẫn đường"; không nhắc thẳng tên chức năng. Chỉ than, kể, cảm thán.
- Quan trọng nhất: viết ĐÚNG như người Việt nói chuyện với người ngồi cạnh trong xe — câu đời
  thường, dễ hiểu ngay, có thể có từ đệm. Không viết kiểu văn viết, kiểu thơ hay kiểu tít báo;
  không nén chữ; không dùng tình huống hiếm gặp.
- Trộn tự nhiên câu ngắn và câu kể hoàn cảnh dài hơn — độ dài tuỳ câu, miễn tự nhiên.
- Mỗi câu một tình huống khác hẳn nhau; không viết các câu chỉ đổi vài chữ.
- Đọc lại từng câu và tự hỏi: "Có người thật nào nói đúng câu này trong xe không?" — nếu không,
  viết lại.

Trả lại CSV với 4 cột: id, chuc_nang, cau, ghi_chu. Giữ nguyên id và chuc_nang.
Làm theo lô: lần này chỉ làm id 1 đến 26.
```

Tin nhắn 2–5: `Tiếp id 27 đến 52, cùng quy tắc. Đừng lặp ý hay khuôn câu của các lô trước.`
(rồi 53–78, 79–104, 105–130).

Ghép 5 lô → lưu **một file** `1_viet_cau_da_viet.csv` (nhớ Ctrl+S). Đọc lại: câu nào không tự nhiên, ra
lệnh thẳng hay trùng ý câu khác thì **tự sửa tay** (không nhờ ChatGPT viết lại riêng — lần trước làm vậy
sinh ra câu trùng ý).

---

## Bước 2 — Gán nhãn (cuộc trò chuyện ChatGPT KHÁC)

Claude sẽ tạo file `2_cho_gan_nhan.csv` (đã bỏ cột chức năng). Đính kèm **`2_cho_gan_nhan.csv` +
`BANG_TRA_HAM.md`** (cùng thư mục này), dán prompt trong `HUONG_DAN_gan_nhan.md`, làm theo lô 45 câu.
Nhắc ChatGPT thêm: **tham số BẮT BUỘC phải có đủ** — quạt gió luôn có `level` (gió mạnh / ồn → 2, yếu → 6),
đèn pha luôn có `mode` (tối → low_beam), kính cửa sổ luôn có `window` (không rõ → "all").

Rồi bạn duyệt từng nhãn, lưu `2_gan_nhan_da_duyet.csv` (cột id, cau, nhan, ghi_chu).

## Bước 3 — Gửi Claude: chuyển jsonl, kiểm schema, kiểm trùng với MỌI dữ liệu (cả train v5 mới), đóng băng,
rồi mới chấm (một lần).
