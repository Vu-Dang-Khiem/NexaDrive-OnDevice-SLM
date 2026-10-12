# Hướng dẫn gán nhãn `2_gan_nhan_100_cau.csv`

**Mục đích:** xem hai người khác nhau hiểu cùng một câu có giống nhau không. Không có đáp án đúng sai —
cứ gán theo ý bạn.

**Gán độc lập:** đừng hỏi hay xem cách người khác đã gán.

## Mỗi câu điền 3 cột (cột `ghi_chu` tuỳ ý)

### `loai_cau` — chọn một trong ba

| Giá trị | Khi nào | Ví dụ |
|---|---|---|
| `tinh_huong` | than thở, kể cảm giác, mô tả tình huống — KHÔNG ra lệnh | "Nóng như cái lò trong này" |
| `ra_lenh` | yêu cầu thẳng một việc (có động từ ra lệnh, hoặc hỏi thẳng số liệu) | "Bật điều hoà lên", "Mở cửa sổ tí được không?" |
| `khong_ro` | câu lủng củng, vô nghĩa, hoặc chẳng liên quan gì tới xe | "Treo đầu –and nhạc nào" |

### `ham_1` — nếu được GỢI Ý một việc để giúp người nói, bạn gợi ý hàm nào?

Ghi đúng **tên hàm**, không cần tham số. Danh sách ở cuối file này.

- Câu `ra_lenh`: vẫn ghi hàm người đó muốn gọi.
- Câu `khong_ro`: để trống.
- Không có hàm nào hợp: ghi `khong_co`.

### `ham_2` — CHỈ điền khi có hàm thứ hai hợp lý NGANG BẰNG `ham_1`

"Trong xe bí quá" → `ham_1 = set_ac`, `ham_2 = control_window` (cả hai đều là cách xử lý bình thường).
Nếu hàm thứ hai chỉ "cũng giúp được chút" thì để trống.

### `ghi_chu` — tuỳ ý

Ghi khi phân vân, ví dụ "không rõ ồn trong hay ngoài xe".

## Luật đã chốt của dự án (để nhãn nhất quán)

- **4 hàm nhạy cảm** — `unlock_doors`, `open_trunk`, `open_charge_port`, `open_fuel_flap` — agent không được
  tự gợi ý. Nếu bạn thấy đó là cách hợp lý nhất thì vẫn ghi, nhưng thêm `ghi_chu: nhạy cảm`.
- **Đèn khẩn cấp** (`set_hazard_lights`) chỉ dùng khi xe dừng/hỏng trên đường — không dùng để chống trộm.
- **Hàm chỉ đọc** (7 hàm `get_...`) — ghi bình thường nếu thấy hợp, code sẽ tự xử lý luật.

## Danh sách hàm

| Tên hàm | Làm gì |
|---|---|
| `get_fuel_level` | xem mức xăng |
| `get_battery_range` | xem pin còn bao nhiêu, đi được bao xa |
| `get_tire_pressure` | xem áp suất lốp |
| `get_vehicle_status` | xem tổng quan xe (cửa, cốp, kính, đèn...) |
| `get_odometer` | xem tổng số km đã đi |
| `get_range_estimate` | còn đi được bao xa, đủ cho chuyến đi không |
| `get_weather` | xem thời tiết |
| `set_ac` | bật/tắt điều hoà, đặt nhiệt độ |
| `set_fan_speed` | chỉnh tốc độ quạt gió |
| `set_defrost` | sấy kính (làm tan sương/hơi nước) |
| `lock_doors` | khoá cửa |
| `unlock_doors` | mở khoá cửa ⚠️ nhạy cảm |
| `open_trunk` | mở cốp ⚠️ nhạy cảm |
| `close_trunk` | đóng cốp |
| `open_charge_port` | mở nắp sạc ⚠️ nhạy cảm |
| `open_fuel_flap` | mở nắp bình xăng ⚠️ nhạy cảm |
| `control_window` | hạ/nâng kính cửa sổ |
| `operate_sunroof` | mở/đóng/hé cửa sổ trời |
| `set_headlights` | đèn pha / đèn cốt / tự động |
| `set_hazard_lights` | đèn khẩn cấp (xi nhan đôi) |
| `set_volume` | chỉnh âm lượng |
| `play_media` | phát nhạc / radio |
| `navigate_to` | dẫn đường tới một nơi |
| `find_nearby` | tìm địa điểm gần đây (cây xăng, trạm sạc, quán ăn, cà phê, bãi đỗ...) |

## Mẹo

- Làm từng đợt 50–100 câu, nghỉ giữa chừng — mệt thì dễ gán vội.
- Mở bằng Excel được; lưu lại dạng **CSV UTF-8**. Đóng Excel trước khi báo xong (Excel khoá file khi đang mở).
