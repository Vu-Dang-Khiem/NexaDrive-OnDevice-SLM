# Gán nhãn tập đánh giá GRPO (đưa kèm `BANG_TRA_HAM.md`)

Mở cuộc trò chuyện ChatGPT **KHÁC** với lúc viết câu. Đính kèm **hai file**: `BANG_TRA_HAM.md` và
`2_cho_gan_nhan.csv` (cột `id, luot_1, cau, nhan_luot_1, nhan, ghi_chu`). Dán đoạn dưới:

```
Đây là các câu người ngồi trong xe nói với trợ lý giọng nói. Trợ lý có các chức năng trong bảng tra hàm
đính kèm. Với mỗi dòng, ghi vào cột "nhan" trợ lý NÊN làm gì với câu ở cột "cau", đúng cách ghi trong bảng
tra hàm:

1. Câu RA LỆNH / HỎI THẲNG (trợ lý làm luôn được): ghi lời gọi hàm, ví dụ dạng set_ac(state="on", temperature=22).
2. Câu KHÔNG ra lệnh — than phiền, nói bóng gió, kể hoàn cảnh — mà trợ lý nên HỎI LẠI để xác nhận: ghi
   "ask: " rồi lời gọi hàm trợ lý sẽ đề xuất, ví dụ dạng: ask: set_ac(state="on")
   Nếu có HAI cách hợp lý, ghi cả hai ngăn bằng dấu ";", cách hợp lý hơn để trước. Không ghi quá hai cách.
3. Câu không liên quan chức năng nào, tán gẫu, hoặc nhờ việc xe KHÔNG làm được: ghi irrelevant_function().
4. Chỉ ghi tham số mà câu có nhắc tới hoặc suy ra chắc chắn (vd than nóng -> state="on", không tự thêm nhiệt
   độ). Giá trị tham số phải nằm trong danh sách hợp lệ của bảng tra. Tham số BẮT BUỘC phải có đủ.
5. Đọc kỹ mục "Cặp hàm dễ nhầm" trong bảng tra — đặc biệt: nhiệt độ "về / xuống N độ" dùng temperature,
   "tăng / giảm N độ" dùng delta, không nói số thì delta=-2 hoặc delta=2; sưởi / làm lạnh / tự động dùng mode;
   cả hai kính sau / trước dùng window="rear" / "front"; quạt tăng / giảm tương đối dùng delta.
6. Câu cần làm HAI việc: ghi hai lời gọi, ngăn bằng dấu ";".
7. Dòng có cột "luot_1" (hội thoại hai lượt): ghi thêm cột "nhan_luot_1" = lời gọi cho câu lượt 1. Cột "nhan"
   = cho câu lượt 2, HIỂU THEO NGỮ CẢNH lượt 1 (vd lượt 1 "mở cốp", lượt 2 "đóng đi" -> close_trunk()). Nếu
   lượt 2 là chuyện KHÁC không liên quan lượt 1 thì ghi theo đúng lượt 2 (có thể là ask: ... hoặc
   irrelevant_function()).
8. Phân vân thì vẫn ghi nhãn bạn thấy hợp lý nhất và ghi lý do vào cột ghi_chu.

Trả lại CSV với các cột: id, luot_1, cau, nhan_luot_1, nhan, ghi_chu. Giữ nguyên id, luot_1, cau.
Làm theo lô: lần này chỉ làm id 1 đến 50.
```

Tin nhắn 2–4: `Tiếp id 51 đến 100, cùng quy tắc.` (rồi 101–150, 151–200).

Ghép 4 lô → lưu `2_da_gan_nhan_200.csv` trong thư mục này (nhớ Ctrl+S) → nhắn Claude. Claude kiểm nhãn bằng code
(đúng tên hàm, tham số, giá trị) và tạo **bảng duyệt** để bạn duyệt từng nhãn.

## Khi bạn duyệt (bước sau)

- Mỗi nhãn đọc lại theo câu: người bình thường nói câu này có muốn đúng việc đó không?
- Câu than phiền / bóng gió: phải là `ask: …` (hỏi lại), **không** phải lời gọi làm luôn.
- Đèn khẩn cấp chỉ khi **xe mình** gặp sự cố / dừng gấp — thấy xe khác nháy đèn thì không.
- Quạt gió: câu nói về gió mạnh / yếu / ồn -> `set_fan_speed`, không phải `set_ac`.
- Hai lượt: nhãn lượt 2 phải đúng VỚI ngữ cảnh lượt 1 (vd lượt 1 sấy kính trước, lượt 2 "cả phía sau nữa"
  -> sấy kính sau).
- Không mở các file test / train khác khi duyệt.
