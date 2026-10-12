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

## Cặp hàm dễ nhầm — đọc trước khi kiểm

| Câu nói về | Hàm đúng | Hay bị nhầm thành |
|---|---|---|
| còn bao nhiêu xăng / pin **nói chung** | `get_fuel_level` / `get_battery_range` | — |
| đi **một quãng cụ thể** ("đi thêm 175km") | `get_range_estimate(trip_km=175)` | `get_fuel_level()` |
| **hạ / nâng kính** (kính hai bên) | `control_window` | `set_defrost` (sấy kính) |
| **sấy / sưởi / khử sương** kính | `set_defrost` | `control_window` |
| đèn **pha / cốt** (chiếu sáng) | `set_headlights` | `set_hazard_lights` |
| đèn **khẩn cấp / xi nhan đôi** — xe MÌNH gặp sự cố | `set_hazard_lights` | `set_headlights` |
| đèn / sự cố của **xe khác** | không làm gì (`irrelevant_function`) | `set_hazard_lights` |
| **gió** mạnh / yếu / ồn (điều hoà đang chạy) | `set_fan_speed` | `set_ac` |
| **nóng / lạnh** | `set_ac` | `set_fan_speed` |
| đến **một nơi có tên** | `navigate_to(destination=...)` | `find_nearby` |
| tìm **một chỗ bất kỳ gần đây** (cây xăng, quán ăn...) | `find_nearby(category=...)` | `navigate_to` |
| âm lượng **về một mức** ("mức 15") | `set_volume(set_value=15)` | `set_volume(delta=...)` |
| âm lượng **thêm / bớt** ("to thêm 3") | `set_volume(delta=3)` | `set_volume(set_value=...)` |

**Tham số tuỳ chọn bỏ trống là bình thường** — xe dùng giá trị mặc định (cột "Mặc định" bên dưới). Chỉ tính
là sai khi thiếu thứ người dùng **đã nói**, hoặc bịa thêm thứ người dùng **không nói**.

---


## Tra cứu trạng thái xe

### `get_fuel_level`
Báo lượng xăng còn lại theo phần trăm dung tích bình, quãng đường ước tính đi được, và cho biết đã xuống dưới ngưỡng cần đổ xăng hay chưa.

*Không có tham số.*

Câu ví dụ:
- "Còn bao nhiêu xăng, đã đến mức phải đổ chưa?"
- "Kiểm tra tình trạng nhiên liệu"
- "Bình xăng còn được mấy phần rồi?"

### `get_battery_range`
Báo mức pin cao áp theo phần trăm và quãng đường chạy điện ước tính còn lại.

*Không có tham số.*

Câu ví dụ:
- "Pin cao áp đang ở mức mấy phần trăm vậy?"
- "Với lượng pin này chạy thêm được mấy cây số?"

### `get_tire_pressure`
Báo áp suất lốp theo đơn vị PSI. Nếu chỉ định một bánh thì chỉ báo bánh đó, không thì báo cả bốn bánh và cảnh báo bánh nào ngoài ngưỡng an toàn.

| Tham số | Bắt buộc | Giá trị hợp lệ | Mặc định (khi bỏ trống) |
|---|---|---|---|
| `wheel` | không | `front_left` trước trái (ghế lái), `front_right` trước phải (ghế phụ), `rear_left` sau trái, `rear_right` sau phải | tuỳ xe |

Câu ví dụ:
- "Kiểm tra áp suất lốp"
- "Lốp trước bên phải có non không?"

### `get_vehicle_status`
Trả về tổng quan trạng thái xe: khoá/mở cửa, cốp, cửa sổ, đèn, điều hoà, phần trăm xăng và pin.

*Không có tham số.*

Câu ví dụ:
- "Tổng quan xe lúc này ra sao, có gì bất thường không?"
- "Cho tôi xem tổng quan tình trạng xe"

### `get_odometer`
Báo tổng quãng đường xe đã đi.

| Tham số | Bắt buộc | Giá trị hợp lệ | Mặc định (khi bỏ trống) |
|---|---|---|---|
| `unit` | không | `km` ki-lô-mét, `mi` dặm | `km` |

Câu ví dụ:
- "Xe đã đi được bao nhiêu km?"
- "Đồng hồ tổng quãng đường đang chỉ bao nhiêu?"

### `get_range_estimate`
Tổng quãng đường còn đi được (gộp xăng và/hoặc pin), kèm trả lời có/không cho việc chuyến đi với quãng đường cho trước có tới nơi mà không cần tiếp nhiên liệu hay không.

| Tham số | Bắt buộc | Giá trị hợp lệ | Mặc định (khi bỏ trống) |
|---|---|---|---|
| `trip_km` | không | số ≥ 0 | tuỳ xe |

Câu ví dụ:
- "Chuyến 350 cây nữa thì có cần dừng tiếp nhiên liệu không?"
- "Xe còn đi được bao xa nữa?"

## Điều hoà

### `set_ac`
Bật hoặc tắt điều hoà, có thể đặt nhiệt độ mục tiêu cho một vùng. Nhiệt độ tính theo độ C, trong khoảng 16–30.

| Tham số | Bắt buộc | Giá trị hợp lệ | Mặc định (khi bỏ trống) |
|---|---|---|---|
| `state` | **có** | `on` bật, `off` tắt | — |
| `temperature` | không | số từ 16 đến 30 | tuỳ xe |
| `zone` | không | `driver` ghế lái, `passenger` ghế phụ, `rear` hàng sau, `all` tất cả | `all` |

Câu ví dụ:
- "Bật điều hòa 22 độ ghế lái"
- "Bật máy lạnh lên"
- "Chỉnh điều hoà 24 độ cho cả xe"

Câu mơ hồ (ghi kèm `ask:`):
- "Ôi trong xe nóng quá"
- "Buốt hết cả người vì lạnh"

### `set_fan_speed`
Đặt tốc độ quạt gió điều hoà theo thang 0–7 (0 là tắt).

| Tham số | Bắt buộc | Giá trị hợp lệ | Mặc định (khi bỏ trống) |
|---|---|---|---|
| `level` | **có** | số từ 0 đến 7 | — |

Câu ví dụ:
- "Tăng quạt gió lên mức 5"
- "Bật quạt gió mạnh nhất"
- "Cho sức gió về mức 2 thôi"

Câu mơ hồ (ghi kèm `ask:`):
- "Gió yếu quá chả thấy mát gì"
- "Quạt thổi ù ù ồn quá"

### `set_defrost`
Bật hoặc tắt sấy kính cho vùng chỉ định.

| Tham số | Bắt buộc | Giá trị hợp lệ | Mặc định (khi bỏ trống) |
|---|---|---|---|
| `state` | **có** | `on` bật, `off` tắt | — |
| `zone` | không | `front` trước, `rear` hàng sau, `both` cả trước và sau | `front` |

Câu ví dụ:
- "Bật sấy kính sau"
- "Kính mờ quá, bật sấy kính trước đi"
- "Cho sưởi kính hậu lên"

Câu mơ hồ (ghi kèm `ask:`):
- "Mờ hết cả kính, chả nhìn thấy đường"
- "Nhìn gương chiếu hậu toàn hơi nước"

## Cửa, cốp, nắp

### `lock_doors`
Khoá toàn bộ cửa xe — khi rời xe, hoặc khi thấy chỗ đỗ/khu vực xung quanh không an toàn.

*Không có tham số.*

Câu ví dụ:
- "Khóa hết cửa lại"
- "Khoá xe giúp tôi"
- "Cài khoá tất cả cửa đi"

Câu mơ hồ (ghi kèm `ask:`):
- "Đỗ chỗ này trông vắng vẻ quá"
- "Khu này nghe nói hay bị mất đồ trong xe"

### `unlock_doors`
Mở khoá toàn bộ cửa, hoặc chỉ một cửa nếu được chỉ định.

| Tham số | Bắt buộc | Giá trị hợp lệ | Mặc định (khi bỏ trống) |
|---|---|---|---|
| `door` | không | `driver` ghế lái, `passenger` ghế phụ, `rear_left` sau trái, `rear_right` sau phải, `all` tất cả | `all` |

Câu ví dụ:
- "Mở khóa cửa"
- "Chỉ mở khoá cửa ghế lái thôi"
- "Nhả khoá xe ra"

### `open_trunk`
Mở cốp sau xe.

*Không có tham số.*

Câu ví dụ:
- "Bật cốp giúp tôi cất vali"
- "Mở cốp sau giúp tôi"
- "Cho tôi mở khoang để hành lý phía sau"

### `close_trunk`
Đóng cốp sau xe (loại cốp điện).

*Không có tham số.*

Câu ví dụ:
- "Đóng cốp lại"
- "Hạ cốp phía sau xuống đi"
- "Hạ khoang hành lý xuống"

Câu mơ hồ (ghi kèm `ask:`):
- "Chất xong hết đồ rồi đấy"
- "Sắp chạy rồi mà đằng sau vẫn còn hở"

### `open_charge_port`
Mở nắp cổng sạc điện.

*Không có tham số.*

Câu ví dụ:
- "Mở nắp sạc"
- "Mở cổng sạc điện"
- "Bật cửa sạc lên để cắm điện"

### `open_fuel_flap`
Mở nắp bình xăng.

*Không có tham số.*

Câu ví dụ:
- "Bật nắp bình xăng lên cho nhân viên bơm"
- "Chuẩn bị đổ xăng, mở chỗ đổ xăng ra giúp"
- "Bật nắp nhiên liệu giúp cái"

## Cửa sổ, cửa sổ trời

### `control_window`
Hạ hoặc nâng cửa sổ, có thể chỉ định mức di chuyển theo phần trăm (mặc định là mở/đóng hoàn toàn).

| Tham số | Bắt buộc | Giá trị hợp lệ | Mặc định (khi bỏ trống) |
|---|---|---|---|
| `window` | **có** | `front_left` trước trái (ghế lái), `front_right` trước phải (ghế phụ), `rear_left` sau trái, `rear_right` sau phải, `all` tất cả | — |
| `position` | **có** | `up` nâng lên (đóng), `down` hạ xuống (mở) | — |
| `amount` | không | số từ 0 đến 100 | `100` |

Câu ví dụ:
- "Hạ cửa sổ ghế phụ xuống một nửa"
- "Nâng kính phía ghế lái lên hết"
- "Đóng hết cửa kính lại"

Câu mơ hồ (ghi kèm `ask:`):
- "Trong xe hầm hơi quá chừng"
- "Có mùi gì khó chịu thế nhỉ"

### `operate_sunroof`
Mở, đóng hoặc hé cửa sổ trời, có thể chỉ định mức độ theo phần trăm.

| Tham số | Bắt buộc | Giá trị hợp lệ | Mặc định (khi bỏ trống) |
|---|---|---|---|
| `action` | **có** | `open` mở, `close` đóng, `tilt` hé | — |
| `intensity` | không | số từ 0 đến 100 | tuỳ xe |

Câu ví dụ:
- "Hé cửa sổ trời"
- "Mở cửa nóc một nửa"
- "Đóng cửa trời lại đi"

Câu mơ hồ (ghi kèm `ask:`):
- "Trời đêm nay nhiều sao đẹp ghê"
- "Muốn hít chút không khí trời quá"

## Đèn

### `set_headlights`
Đặt chế độ đèn pha.

| Tham số | Bắt buộc | Giá trị hợp lệ | Mặc định (khi bỏ trống) |
|---|---|---|---|
| `mode` | **có** | `off` tắt, `auto` tự động, `low_beam` đèn cốt (chiếu gần), `high_beam` đèn pha (chiếu xa), `parking` đèn đỗ xe | — |

Câu ví dụ:
- "Bật đèn pha"
- "Để đèn trước ở chế độ auto"
- "Cho đèn chiếu gần thôi, đừng chói người ta"

Câu mơ hồ (ghi kèm `ask:`):
- "Đường tối om chẳng thấy gì"
- "Trời bắt đầu sập tối rồi"

### `set_hazard_lights`
Bật hoặc tắt đèn cảnh báo nguy hiểm (hai bên xi nhan nháy cùng lúc). Dùng khi xe dừng đột ngột, hỏng hóc hoặc gặp sự cố trên đường để cảnh báo xe khác — không phải để chống trộm hay khi thấy chỗ đỗ không an toàn.

| Tham số | Bắt buộc | Giá trị hợp lệ | Mặc định (khi bỏ trống) |
|---|---|---|---|
| `state` | **có** | `on` bật, `off` tắt | — |

Câu ví dụ:
- "Bật đèn khẩn cấp"
- "Thôi, tắt xi nhan đôi được rồi"
- "Bật xi nhan đôi lên"

Câu mơ hồ (ghi kèm `ask:`):
- "Xe tự nhiên khựng lại phải tấp vào lề"
- "Đang phải dừng ở làn khẩn cấp đây"

## Nhạc, âm lượng

### `set_volume`
Đặt âm lượng về một giá trị tuyệt đối (0–40) hoặc tăng/giảm theo mức chênh. Chỉ dùng một trong hai tham số set_value / delta.

| Tham số | Bắt buộc | Giá trị hợp lệ | Mặc định (khi bỏ trống) |
|---|---|---|---|
| `set_value` | không | số từ 0 đến 40 | tuỳ xe |
| `delta` | không | số từ -40 đến 40 | tuỳ xe |

Câu ví dụ:
- "Cho âm lượng to thêm 3 mức"
- "Đặt âm lượng mức 15"
- "Vặn nhỏ tiếng xuống"

Câu mơ hồ (ghi kèm `ask:`):
- "Nhạc to nhức hết cả đầu"
- "Chẳng nghe rõ ca sĩ hát gì cả"

### `play_media`
Phát nhạc/đài từ nguồn chỉ định, có thể kèm từ khoá tìm kiếm.

| Tham số | Bắt buộc | Giá trị hợp lệ | Mặc định (khi bỏ trống) |
|---|---|---|---|
| `source` | không | `radio` đài FM/AM, `bluetooth` điện thoại qua Bluetooth, `usb` USB, `streaming` nhạc trực tuyến | `streaming` |
| `query` | không | chữ tự do | tuỳ xe |

Câu ví dụ:
- "Mở nhạc của Sơn Tùng"
- "Bật radio lên"

Câu mơ hồ (ghi kèm `ask:`):
- "Lái đường dài buồn ngủ quá"
- "Trong xe im lặng chán ghê"

## Dẫn đường

### `navigate_to`
Bắt đầu dẫn đường tới điểm đến; trả về khoảng cách và thời gian dự kiến. Có thể yêu cầu tránh trạm thu phí hoặc cao tốc.

| Tham số | Bắt buộc | Giá trị hợp lệ | Mặc định (khi bỏ trống) |
|---|---|---|---|
| `destination` | **có** | chữ tự do | — |
| `avoid` | không | `none` không tránh, `tolls` tránh trạm thu phí, `highways` tránh cao tốc | `none` |

Câu ví dụ:
- "Tra map cho tôi đến sân bay Nội Bài"
- "Dẫn đường tới bệnh viện gần nhất, tránh trạm thu phí"

Câu mơ hồ (ghi kèm `ask:`):
- "Lạc mất đường rồi chẳng biết lối nào"
- "Chỗ này lạ hoắc, chưa tới bao giờ"

### `find_nearby`
Tìm địa điểm theo nhóm quanh vị trí hiện tại.

| Tham số | Bắt buộc | Giá trị hợp lệ | Mặc định (khi bỏ trống) |
|---|---|---|---|
| `category` | **có** | `gas_station` cây xăng, `charging_station` trạm sạc, `parking` bãi đỗ xe, `restaurant` quán ăn, `hospital` bệnh viện, `atm` ATM, `coffee` quán cà phê | — |
| `radius_km` | không | số từ 0.5 đến 25 | `5` |

Câu ví dụ:
- "Tìm cây xăng gần đây"
- "Có trạm sạc nào trong bán kính 10km không?"

Câu mơ hồ (ghi kèm `ask:`):
- "Đói bụng quá rồi"
- "Mắt díp lại, cần chút cà phê"

## Thông tin ngoài

### `get_weather`
Thời tiết tại một địa điểm (mặc định là vị trí hiện tại của xe) cho ngày cách hôm nay bao nhiêu ngày (0 là hôm nay).

| Tham số | Bắt buộc | Giá trị hợp lệ | Mặc định (khi bỏ trống) |
|---|---|---|---|
| `location` | không | chữ tự do | tuỳ xe |
| `day` | không | số từ 0 đến 6 | `0` |

Câu ví dụ:
- "Hôm nay ngoài trời có nắng gắt không?"
- "Ngày mốt ở Huế trời ra sao?"

## Ngoài phạm vi

### `irrelevant_function`
Được gọi khi câu của người dùng không liên quan tới bất kỳ chức năng nào của xe (tán gẫu, hỏi kiến thức chung, hoặc yêu cầu xe không hỗ trợ như tự lái). Lượt 2 vẫn trả lời tự nhiên: từ chối lịch sự và nói rõ phạm vi hỗ trợ.

*Không có tham số.*

Câu ví dụ:
- "Giải giúp tôi bài toán phương trình bậc hai"
- "Đặt vé máy bay đi Đà Lạt cho tôi"
- "Thủ đô nước Pháp là gì?"
