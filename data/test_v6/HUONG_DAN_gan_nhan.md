# Gán nhãn tập test v6 (đưa kèm `BANG_TRA_HAM.md`)

Dán đoạn dưới cho ChatGPT, kèm hai file: `BANG_TRA_HAM.md` và `2_cho_gan_nhan.csv`
(cột `id, luot_1, cau, nhan_luot_1, nhan, ghi_chu`).

```
Đây là các câu người ngồi trong xe nói với trợ lý giọng nói. Trợ lý có các chức năng trong bảng tra hàm
đính kèm. Với mỗi dòng, ghi vào cột "nhan" lời gọi hàm trợ lý NÊN làm cho câu ở cột "cau", đúng cách ghi
trong bảng tra hàm, ví dụ dạng: set_ac(state="on", temperature=22)

1. Chỉ ghi tham số mà câu có nhắc tới hoặc suy ra chắc chắn. Giá trị tham số phải nằm trong danh sách hợp lệ
   của bảng tra. Tham số BẮT BUỘC phải có đủ.
2. Đọc kỹ mục "Cặp hàm dễ nhầm" trong bảng tra — đặc biệt: nhiệt độ "về / xuống N độ" dùng temperature,
   "tăng / giảm N độ" dùng delta, không nói số thì delta=-2 hoặc delta=2; sưởi / làm lạnh / tự động dùng mode;
   cả hai kính sau / trước dùng window="rear" / "front".
3. Câu cần làm HAI việc: ghi hai lời gọi, ngăn bằng dấu ";".
4. Câu không liên quan chức năng nào hoặc xe không làm được: ghi irrelevant_function().
5. Dòng có cột "luot_1" (hội thoại hai lượt): ghi thêm cột "nhan_luot_1" = lời gọi cho câu lượt 1. Cột "nhan"
   = lời gọi cho câu lượt 2, HIỂU THEO NGỮ CẢNH lượt 1 (vd lượt 1 "mở cốp", lượt 2 "đóng đi" -> close_trunk()).
   Nếu lượt 2 là lệnh mới không liên quan thì ghi theo đúng lượt 2.
6. Phân vân thì vẫn ghi nhãn bạn thấy hợp lý nhất và ghi lý do vào cột ghi_chu.

Trả lại CSV với các cột: id, luot_1, cau, nhan_luot_1, nhan, ghi_chu. Giữ nguyên id, luot_1, cau.
```

## Khi bạn duyệt (sau ChatGPT)

- Mỗi nhãn đọc lại theo câu: người bình thường nói câu này có muốn đúng việc đó không?
- Phần A (câu ngắn): "bật pha" là đèn pha; "pha xa" / "pha gần" là đổi chế độ; "tắt quạt" là quạt gió mức 0.
- Nhiệt độ: "xuống 20 độ" -> `temperature=20`; "giảm 2 độ" -> `delta=-2`; "mát hơn chút" -> `delta=-2`.
- Hai lượt: nhãn lượt 2 phải đúng VỚI ngữ cảnh lượt 1 (vd lượt 1 hạ kính ghế phụ, lượt 2 "cả bên kia nữa"
  -> hạ kính ghế lái).
- Câu `irrelevant_function()` **giữ lại** trong file (không xoá).
- Không mở các file test / train khác khi duyệt.
