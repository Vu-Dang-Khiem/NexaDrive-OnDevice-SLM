# Gán nhãn bộ câu mơ hồ (đưa kèm bảng tra hàm `HUONG_DAN.md`)

Dán đoạn dưới cho ChatGPT, kèm hai file: bảng tra hàm và CSV câu (cột `id, cau, ghi_chu`).

```
Đây là các câu người ngồi trong xe có thể nói với trợ lý giọng nói. Trợ lý có các chức năng trong
bảng tra hàm đính kèm. Với mỗi câu, ghi vào cột "nhan" trợ lý NÊN làm gì, theo đúng cách ghi trong
bảng tra hàm:

1. Câu không nói thẳng, trợ lý nên HỎI LẠI để xác nhận: ghi "ask: " rồi lời gọi hàm trợ lý sẽ đề
   xuất. Ví dụ dạng: ask: set_ac(state="on")
2. Nếu có HAI cách hợp lý (ví dụ nóng → điều hoà hoặc hạ kính), ghi cả hai, ngăn bằng dấu ";",
   cách hợp lý hơn để trước. Không ghi quá hai cách.
3. Chỉ ghi tham số mà câu có nhắc tới hoặc suy ra chắc chắn (vd "nóng quá" → state="on", không tự
   thêm nhiệt độ). Giá trị tham số phải nằm trong danh sách hợp lệ của bảng tra.
4. Nếu câu thật ra là RA LỆNH THẲNG (trợ lý làm luôn được, không cần hỏi): ghi lời gọi bình thường,
   không có "ask: ", và ghi "ra_lenh" vào cột ghi_chu.
5. Nếu câu không liên quan tới chức năng nào: ghi irrelevant_function() và ghi lý do vào ghi_chu.
6. Phân vân thì vẫn ghi nhãn bạn thấy hợp lý nhất và ghi lý do vào ghi_chu.

Trả lại CSV với các cột: id, cau, nhan, ghi_chu. Giữ nguyên id và câu.
```

## Khi bạn duyệt (sau ChatGPT)

- Mỗi nhãn đọc lại theo câu: người bình thường nghe câu này có muốn đúng việc đó không?
- Đèn khẩn cấp chỉ khi **xe mình** gặp sự cố / dừng gấp — thấy xe khác nháy đèn thì không.
- Quạt gió: câu nói về gió mạnh / yếu / ồn → `set_fan_speed`, không phải `set_ac`.
- Câu `ra_lenh` hoặc `irrelevant_function()` **giữ lại** trong file (không xoá) — Claude sẽ tách ra
  báo riêng, không tính vào nhóm mơ hồ.
- Không mở các file test / train khác khi duyệt.
