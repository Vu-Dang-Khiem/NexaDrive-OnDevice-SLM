# Hướng dẫn kiểm `kiem_200.csv`

**Mục đích:** đo chất lượng phần dữ liệu train chưa ai kiểm tay (gọi hàm, trả lời lượt 2, từ chối) —
để có con số so được với APIGen (95,3%) và Huawei 2026 (100%).

Mỗi dòng đã có sẵn câu và đáp án. Bạn **chỉ đánh giá đáp án có đúng không**, không cần tự nghĩ lại.

**Mở kèm `bang_tra_ham.md`** (cùng thư mục): ý nghĩa từng hàm, nghĩa tiếng Việt của từng giá trị tham số, giá trị
mặc định khi bỏ trống, và bảng **các cặp hàm dễ nhầm** — đọc bảng đó trước khi bắt đầu.

## Điền cột `danh_gia`

| Giá trị | Khi nào |
|---|---|
| `dung` | đáp án làm đúng việc người dùng muốn |
| `sai` | sai hàm, sai tham số, bịa tham số, hoặc lẽ ra phải khác (từ chối / hỏi lại / gọi hàm) |
| `phan_van` | có thể hiểu hai cách, không chắc |

Với `sai` và `phan_van`: ghi ngắn vào `ghi_chu` vì sao, và đáp án đúng theo bạn (nếu có).

## Mỗi loại kiểm gì

**`gọi hàm`** — cột `dap_an` là lời gọi hàm.
- Đúng hàm chưa? Ví dụ "Xăng còn đủ để đi thêm 175km không?" hỏi về **một chuyến đi cụ thể** — nên là
  `get_range_estimate(trip_km=175)` hơn là chỉ `get_fuel_level()`.
- Tham số có khớp câu nói không? Có bịa thêm chi tiết câu không nhắc tới không?
- Tham số tuỳ chọn **bỏ trống là bình thường** — chỉ tính sai khi thiếu thứ người dùng ĐÃ nói.

**`trả lời lượt 2`** — xe đã chạy hàm, `ket_qua_ham` là kết quả, `cau_tra_loi` là câu trợ lý nói lại.
- Câu trả lời có khớp kết quả không? **Có bịa số nào không có trong kết quả không?**
- Có trả lời đúng điều người dùng hỏi không?

**`từ chối`** — đáp án là "TỪ CHỐI". Kiểm: xe **thật sự không làm được** việc này bằng các chức năng có
trong danh sách không (xem `bang_tra_ham.md`)?
- Nếu thật ra có hàm làm được → `sai` (ví dụ "bật đèn pha cao" là `set_headlights`, không được từ chối).
- Nếu câu thật ra là than phiền mơ hồ, nên hỏi lại → `sai`, ghi chú "nên hỏi lại".

## Mẹo

Làm liền một lượt được (~30–40 phút). Lưu dạng **CSV UTF-8**, đóng Excel trước khi báo xong.
