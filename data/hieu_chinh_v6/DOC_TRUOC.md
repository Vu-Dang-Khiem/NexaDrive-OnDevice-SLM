# Tập hiệu chỉnh ngưỡng (cách 2) — đọc hết trước khi bắt đầu

Mục đích: chọn ngưỡng tin cậy CÓ CAM KẾT THỐNG KÊ — "câu ngoài phạm vi bị làm bừa ≤ 10%" (với độ tin 90%). Tập này KHÔNG
phải tập test, và KHÔNG được lấy câu từ tập test. Chỉ gồm câu NGOÀI PHẠM VI (trợ lý phải từ chối), nên bạn **không phải gán
nhãn** — chỉ duyệt xem câu đúng là việc trợ lý KHÔNG làm được.

`1_viet_cau.csv` có 200 dòng, 5 loại:

| Loại | Số dòng | Viết gì |
|---|---|---|
| ngan_tinh_nang_xe | 50 | lệnh NGẮN 1–5 chữ cho tính năng xe có nhưng trợ lý không làm (vd kiểu gập gương, sưởi ghế) |
| tinh_nang_xe | 40 | lệnh bình thường cho tính năng xe có nhưng trợ lý không làm |
| viec_ngoai_xe | 40 | nhờ việc không liên quan xe |
| tan_gau | 30 | nói chuyện phiếm, cảm thán |
| ke_hoan_canh | 40 | kể hoàn cảnh rồi nhờ việc trợ lý không làm được |

---

## Bước 1 — Viết câu (cuộc trò chuyện ChatGPT MỚI, chỉ đính kèm `1_viet_cau.csv`)

Tin nhắn 1 (dán nguyên văn):

```
File 1_viet_cau.csv đính kèm có 200 dòng. Mình đang làm một trợ lý giọng nói trên xe ô tô. Trợ lý này CHỈ làm được
các việc sau:
- xem xăng, pin, áp suất lốp, số km đã đi, tình trạng xe, quãng đường còn đi được
- bật/tắt/chỉnh nhiệt độ điều hoà (cả sưởi ấm, làm lạnh), quạt gió điều hoà, sấy kính
- khoá/mở khoá cửa, mở/đóng cốp sau, mở nắp sạc, mở nắp bình xăng
- hạ/nâng kính cửa sổ, mở/đóng/hé cửa sổ trời
- đèn pha (chiếu xa, chiếu gần, tự động, tắt), đèn khẩn cấp
- chỉnh âm lượng, mở nhạc/đài
- dẫn đường tới một nơi, tìm chỗ gần đây (cây xăng, trạm sạc, bãi đỗ, quán ăn, bệnh viện, ATM, cà phê), xem thời tiết

Với mỗi dòng, viết MỘT câu tiếng Việt vào cột "cau" — câu người trong xe thật sự hay nói, đúng yêu cầu ở cột
"yeu_cau". MỌI câu phải là việc trợ lý KHÔNG làm được (ngoài danh sách trên) hoặc chỉ là nói chuyện phiếm.

Quy tắc:
- Viết đúng như người Việt nói trong xe: đời thường, có thể cộc lốc, nói tắt, có từ đệm, giọng Bắc/Trung/Nam.
- Đa dạng thật sự, không lặp khuôn câu. Viết có dấu, đúng chính tả.
- Đặc biệt với các loại "tinh_nang_xe": chọn các tính năng xe THẬT có (gương, ghế, gạt mưa, đèn trong xe, cửa gió, chế
  độ lái, camera, phanh tay, cốp trước, cửa sổ trời rèm che, chuyển bài hát, gọi điện...), viết sao cho nghe GIỐNG lệnh
  bình thường — đó chính là chỗ dễ nhầm.
- Không viết câu mà trợ lý làm được (ví dụ "hạ kính", "bật điều hoà", "mở cốp sau" là SAI).

Trả lại CSV với các cột: id, loai, yeu_cau, cau, ghi_chu. Giữ nguyên id, loai, yeu_cau.
Làm theo lô: lần này chỉ làm id 1 đến 40.
```

Tin nhắn 2–5: `Tiếp id 41 đến 80, cùng quy tắc. Đừng lặp ý hay khuôn câu của các lô trước.`
(rồi 81–120, 121–160, 161–200).

## Bước 2 — Bạn duyệt

Ghép 5 lô thành một file `2_da_viet_200.csv` (Ctrl+S). Đọc từng câu và tự hỏi: **"trợ lý này có làm được việc này không?"**
- Làm được (dù chỉ một phần, vd "bật sưởi lên cho ấm") -> **xoá dòng đó** hoặc ghi `bo` vào cột ghi_chu.
- Câu không tự nhiên / trùng ý câu khác -> tự sửa tay.
Xong nhắn Claude. Claude sẽ đóng băng tập này, đo độ tự tin của v5 và v6.1 trên đó (~$0,3), rồi tính ngưỡng theo cam kết.
