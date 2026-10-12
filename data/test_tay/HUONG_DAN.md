# Bảng tra hàm — dùng khi viết tập test tay

Cách dùng: viết câu vào cột `cau` trong `test_vi.csv`. Cột `nhan` ghi theo mẫu ở đây.

| Kiểu câu | Cách ghi cột `nhan` | Ví dụ |
|---|---|---|
| Ra lệnh rõ ràng | `tên_hàm(tham_số=giá_trị)` | `set_ac(state="on", temperature=26)` |
| Mơ hồ, agent nên hỏi lại | thêm `ask: ` ở đầu | `ask: set_ac(state="on")` |
| Mơ hồ, có 2 cách xử lý | ngăn bằng dấu `;` | `ask: set_ac(state="on"); operate_sunroof(action="open")` |
| Ngoài phạm vi | luôn là | `irrelevant_function()` |

Quy tắc: **chỉ ghi tham số mà câu nói có nhắc tới**. "Bật điều hòa" thì chỉ ghi
`set_ac(state="on")`, đừng tự thêm nhiệt độ.

---


## Tra cứu trạng thái xe

### `get_fuel_level`
Báo lượng xăng còn lại theo phần trăm dung tích bình, quãng đường ước tính đi được, và cho biết đã xuống dưới ngưỡng cần đổ xăng hay chưa.

*Không có tham số.*

Câu ví dụ:
- "Còn bao nhiêu xăng, đã đến mức phải đổ chưa?"
- "Kiểm tra tình trạng xăng"
- "Xăng còn nhiều không em?"

### `get_battery_range`
Báo mức pin cao áp theo phần trăm và quãng đường chạy điện ước tính còn lại.

*Không có tham số.*

Câu ví dụ:
- "Pin xe còn bao nhiêu phần trăm?"
- "Pin còn đi được bao xa nữa?"

### `get_tire_pressure`
Báo áp suất lốp theo đơn vị PSI. Nếu chỉ định một bánh thì chỉ báo bánh đó, không thì báo cả bốn bánh và cảnh báo bánh nào ngoài ngưỡng an toàn.

| Tham số | Bắt buộc | Giá trị hợp lệ |
|---|---|---|
| `wheel` | không | `front_left`, `front_right`, `rear_left`, `rear_right` |

Câu ví dụ:
- "Kiểm tra áp suất lốp"
- "Lốp trước bên phải có non không?"

### `get_vehicle_status`
Trả về tổng quan trạng thái xe: khoá/mở cửa, cốp, cửa sổ, đèn, điều hoà, phần trăm xăng và pin.

*Không có tham số.*

Câu ví dụ:
- "Tình trạng xe hiện tại thế nào?"
- "Cho tôi xem tổng quan tình trạng xe"

### `get_odometer`
Báo tổng quãng đường xe đã đi.

| Tham số | Bắt buộc | Giá trị hợp lệ |
|---|---|---|
| `unit` | không | `km`, `mi` |

Câu ví dụ:
- "Xe đã đi được bao nhiêu km?"
- "Số công-tơ-mét hiện tại là bao nhiêu?"

### `get_range_estimate`
Tổng quãng đường còn đi được (gộp xăng và/hoặc pin), kèm trả lời có/không cho việc chuyến đi với quãng đường cho trước có tới nơi mà không cần tiếp nhiên liệu hay không.

| Tham số | Bắt buộc | Giá trị hợp lệ |
|---|---|---|
| `trip_km` | không | số từ 0 đến None |

Câu ví dụ:
- "Đi 200km nữa có phải đổ xăng dọc đường không?"
- "Xe còn đi được bao xa nữa?"

## Điều hoà

### `set_ac`
Bật hoặc tắt điều hoà, có thể đặt nhiệt độ mục tiêu cho một vùng. Nhiệt độ tính theo độ C, trong khoảng 16–30.

| Tham số | Bắt buộc | Giá trị hợp lệ |
|---|---|---|
| `state` | **có** | `on`, `off` |
| `temperature` | không | số từ 16 đến 30 |
| `zone` | không | `driver`, `passenger`, `rear`, `all` |

Câu ví dụ:
- "Bật điều hòa 22 độ ghế lái"
- "Bật điều hoà lên"
- "Chỉnh điều hoà 24 độ cho cả xe"

Câu mơ hồ (ghi kèm `ask:`):
- "Ôi trong xe nóng quá"
- "Lạnh run cả người"

### `set_fan_speed`
Đặt tốc độ quạt gió điều hoà theo thang 0–7 (0 là tắt).

| Tham số | Bắt buộc | Giá trị hợp lệ |
|---|---|---|
| `level` | **có** | số từ 0 đến 7 |

Câu ví dụ:
- "Tăng quạt gió lên mức 5"
- "Bật quạt gió mạnh nhất"

Câu mơ hồ (ghi kèm `ask:`):
- "Gió yếu quá chả thấy mát gì"
- "Quạt thổi ù ù ồn quá"

### `set_defrost`
Bật hoặc tắt sấy kính cho vùng chỉ định.

| Tham số | Bắt buộc | Giá trị hợp lệ |
|---|---|---|
| `state` | **có** | `on`, `off` |
| `zone` | không | `front`, `rear`, `both` |

Câu ví dụ:
- "Bật sấy kính sau"
- "Kính mờ quá, bật sấy kính trước đi"

Câu mơ hồ (ghi kèm `ask:`):
- "Mờ hết cả kính, chả nhìn thấy đường"
- "Nhìn gương chiếu hậu toàn hơi nước"

## Cửa, cốp, nắp

### `lock_doors`
Khoá toàn bộ cửa xe.

*Không có tham số.*

Câu ví dụ:
- "Khóa hết cửa lại"
- "Khoá xe giúp tôi"

### `unlock_doors`
Mở khoá toàn bộ cửa, hoặc chỉ một cửa nếu được chỉ định.

| Tham số | Bắt buộc | Giá trị hợp lệ |
|---|---|---|
| `door` | không | `driver`, `passenger`, `rear_left`, `rear_right`, `all` |

Câu ví dụ:
- "Mở khóa cửa"
- "Chỉ mở khoá cửa ghế lái thôi"

### `open_trunk`
Mở cốp sau xe.

*Không có tham số.*

Câu ví dụ:
- "Mở cốp xe"
- "Mở cốp sau giúp tôi"

### `close_trunk`
Đóng cốp sau xe (loại cốp điện).

*Không có tham số.*

Câu ví dụ:
- "Đóng cốp lại"
- "Đóng cốp sau đi"

### `open_charge_port`
Mở nắp cổng sạc điện.

*Không có tham số.*

Câu ví dụ:
- "Mở nắp sạc"
- "Mở cổng sạc điện"

### `open_fuel_flap`
Mở nắp bình xăng.

*Không có tham số.*

Câu ví dụ:
- "Mở nắp bình xăng"
- "Sắp đổ xăng rồi, mở nắp bình đi"

## Cửa sổ, cửa sổ trời

### `control_window`
Hạ hoặc nâng cửa sổ, có thể chỉ định mức di chuyển theo phần trăm (mặc định là mở/đóng hoàn toàn).

| Tham số | Bắt buộc | Giá trị hợp lệ |
|---|---|---|
| `window` | **có** | `front_left`, `front_right`, `rear_left`, `rear_right`, `all` |
| `position` | **có** | `up`, `down` |
| `amount` | không | số từ 0 đến 100 |

Câu ví dụ:
- "Hạ cửa sổ ghế phụ xuống một nửa"
- "Đóng hết cửa sổ lại"

Câu mơ hồ (ghi kèm `ask:`):
- "Trong xe bí quá"
- "Có mùi gì khó chịu thế nhỉ"

### `operate_sunroof`
Mở, đóng hoặc hé cửa sổ trời, có thể chỉ định mức độ theo phần trăm.

| Tham số | Bắt buộc | Giá trị hợp lệ |
|---|---|---|
| `action` | **có** | `open`, `close`, `tilt` |
| `intensity` | không | số từ 0 đến 100 |

Câu ví dụ:
- "Hé cửa sổ trời"
- "Mở cửa sổ trời một nửa"

Câu mơ hồ (ghi kèm `ask:`):
- "Trời đêm nay nhiều sao đẹp ghê"
- "Muốn hít chút không khí trời quá"

## Đèn

### `set_headlights`
Đặt chế độ đèn pha.

| Tham số | Bắt buộc | Giá trị hợp lệ |
|---|---|---|
| `mode` | **có** | `off`, `auto`, `low_beam`, `high_beam`, `parking` |

Câu ví dụ:
- "Bật đèn pha"
- "Chuyển đèn sang chế độ tự động"

Câu mơ hồ (ghi kèm `ask:`):
- "Đường tối om chẳng thấy gì"
- "Trời bắt đầu sập tối rồi"

### `set_hazard_lights`
Bật hoặc tắt đèn cảnh báo nguy hiểm.

| Tham số | Bắt buộc | Giá trị hợp lệ |
|---|---|---|
| `state` | **có** | `on`, `off` |

Câu ví dụ:
- "Bật đèn khẩn cấp"
- "Tắt đèn cảnh báo đi"

## Nhạc, âm lượng

### `set_volume`
Đặt âm lượng về một giá trị tuyệt đối (0–40) hoặc tăng/giảm theo mức chênh. Chỉ dùng một trong hai tham số set_value / delta.

| Tham số | Bắt buộc | Giá trị hợp lệ |
|---|---|---|
| `set_value` | không | số từ 0 đến 40 |
| `delta` | không | số từ -40 đến 40 |

Câu ví dụ:
- "Tăng âm lượng lên 3 nấc"
- "Đặt âm lượng mức 15"

Câu mơ hồ (ghi kèm `ask:`):
- "Nhạc to nhức hết cả đầu"
- "Chẳng nghe rõ ca sĩ hát gì cả"

### `play_media`
Phát nhạc/đài từ nguồn chỉ định, có thể kèm từ khoá tìm kiếm.

| Tham số | Bắt buộc | Giá trị hợp lệ |
|---|---|---|
| `source` | không | `radio`, `bluetooth`, `usb`, `streaming` |
| `query` | không | chữ tự do |

Câu ví dụ:
- "Mở nhạc của Sơn Tùng"
- "Bật radio lên"

Câu mơ hồ (ghi kèm `ask:`):
- "Lái đường dài buồn ngủ quá"
- "Trong xe im lặng chán ghê"

## Dẫn đường

### `navigate_to`
Bắt đầu dẫn đường tới điểm đến; trả về khoảng cách và thời gian dự kiến. Có thể yêu cầu tránh trạm thu phí hoặc cao tốc.

| Tham số | Bắt buộc | Giá trị hợp lệ |
|---|---|---|
| `destination` | **có** | chữ tự do |
| `avoid` | không | `none`, `tolls`, `highways` |

Câu ví dụ:
- "Tra map cho tôi đến sân bay Nội Bài"
- "Dẫn đường tới bệnh viện gần nhất, tránh trạm thu phí"

### `find_nearby`
Tìm địa điểm theo nhóm quanh vị trí hiện tại.

| Tham số | Bắt buộc | Giá trị hợp lệ |
|---|---|---|
| `category` | **có** | `gas_station`, `charging_station`, `parking`, `restaurant`, `hospital`, `atm`, `coffee` |
| `radius_km` | không | số từ 0.5 đến 25 |

Câu ví dụ:
- "Tìm cây xăng gần đây"
- "Có trạm sạc nào trong bán kính 10km không?"

Câu mơ hồ (ghi kèm `ask:`):
- "Đói bụng quá rồi"
- "Mắt díp lại, cần chút cà phê"

## Thông tin ngoài

### `get_weather`
Thời tiết tại một địa điểm (mặc định là vị trí hiện tại của xe) cho ngày cách hôm nay bao nhiêu ngày (0 là hôm nay).

| Tham số | Bắt buộc | Giá trị hợp lệ |
|---|---|---|
| `location` | không | chữ tự do |
| `day` | không | số từ 0 đến 6 |

Câu ví dụ:
- "Thời tiết hôm nay thế nào?"
- "Mai Đà Nẵng có mưa không?"

## Ngoài phạm vi

### `irrelevant_function`
Được gọi khi câu của người dùng không liên quan tới bất kỳ chức năng nào của xe (tán gẫu, hỏi kiến thức chung, hoặc yêu cầu xe không hỗ trợ như tự lái). Lượt 2 vẫn trả lời tự nhiên: từ chối lịch sự và nói rõ phạm vi hỗ trợ.

*Không có tham số.*

Câu ví dụ:
- "Kể cho tôi một câu chuyện cười"
- "Tự lái đưa tôi về nhà đi"
- "Thủ đô nước Pháp là gì?"
