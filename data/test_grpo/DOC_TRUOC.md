# Tập đánh giá GRPO — đọc hết trước khi bắt đầu

Mục đích: đo GRPO có làm model **cân nhắc tốt hơn** không — câu than phiền / nói bóng gió nên **hỏi lại**, câu
ngoài phạm vi nên **từ chối**, lượt trước không được làm model hiểu sai câu sau — và câu lệnh rõ ràng **không bị
tụt**. Tập này làm và **đóng băng TRƯỚC** khi chạy GRPO. Tập test cũ đã dùng nhiều lần nên không dùng để kết
luận GRPO nữa.

**Không** đưa ChatGPT file nào khác ngoài file ghi trong từng bước. **Không** mở file test / train cũ khi làm.

`1_viet_cau.csv` có 200 dòng, 4 phần (thứ tự dòng đã xáo trộn):

| Phần | Số dòng | Viết gì |
|---|---|---|
| R | 48 | lệnh **nói thẳng** cho đủ 24 chức năng |
| M | 64 | **than phiền / nói bóng gió / kể hoàn cảnh** — không ra lệnh, trợ lý nên hỏi lại |
| N | 48 | **ngoài phạm vi**: tán gẫu, việc ngoài xe, tính năng xe trợ lý không làm được |
| H | 40 | **hai lượt**: cột `luot_1` là câu lượt 1, cột `cau` là câu lượt 2 |

---

## Bước 1 — Viết câu (cuộc trò chuyện ChatGPT MỚI, chỉ đính kèm `1_viet_cau.csv`)

Tin nhắn 1 (dán nguyên văn):

```
File 1_viet_cau.csv đính kèm có 200 dòng. Mình đang làm một trợ lý giọng nói trên xe ô tô. Với mỗi dòng,
viết câu tiếng Việt mà người lái / người ngồi trong xe THẬT SỰ hay nói, đúng chức năng ở cột "chuc_nang" và
đúng yêu cầu ở cột "yeu_cau".

- Phần R, M, N: viết MỘT câu vào cột "cau".
- Phần R: câu RA LỆNH / HỎI THẲNG (trợ lý làm luôn được).
- Phần M: câu KHÔNG ra lệnh — chỉ than phiền / nói bóng gió / kể chuyện — nhưng người nghe hiểu nên dùng chức
  năng ở cột "chuc_nang". Tuyệt đối không viết thành câu ra lệnh.
- Phần N: câu mà trợ lý KHÔNG làm được hoặc chỉ là tán gẫu, đúng yêu cầu ở cột "yeu_cau".
- Phần H: viết HAI câu. Cột "luot_1" = câu lượt đầu, ra lệnh thẳng đúng chức năng ở cột "chuc_nang" (sau
  chữ "lượt 1:"). Cột "cau" = câu lượt hai nói NGAY SAU đó, đúng yêu cầu ở cột "yeu_cau".

Quy tắc:
- Viết ĐÚNG như người Việt nói trong xe: câu đời thường, có thể cộc lốc, nói tắt, có từ đệm (đi, nhé, cái,
  hộ, giùm...). Không viết kiểu văn viết hay câu mẫu sách giáo khoa.
- Đa dạng thật sự: đổi động từ, đổi cách gọi bộ phận, đổi giọng Bắc / Trung / Nam. Không lặp khuôn câu.
- Viết tiếng Việt CÓ DẤU, đúng chính tả, viết hoa chữ đầu câu.
- Đọc lại từng câu và tự hỏi: "Có người thật nào nói đúng câu này trong xe không?" — nếu không, viết lại.

Trả lại CSV với đủ các cột: id, phan, chuc_nang, yeu_cau, luot_1, cau, ghi_chu. Giữ nguyên id, phan,
chuc_nang, yeu_cau. Làm theo lô: lần này chỉ làm id 1 đến 40.
```

Tin nhắn 2–5: `Tiếp id 41 đến 80, cùng quy tắc. Đừng lặp ý hay khuôn câu của các lô trước.`
(rồi 81–120, 121–160, 161–200).

Ghép 5 lô → lưu **một file** `1_viet_cau_da_viet.csv` (nhớ Ctrl+S). Đọc lại: câu nào không tự nhiên, sai
chức năng, phần M lỡ viết thành câu ra lệnh, hay trùng ý câu khác thì **tự sửa tay**.

Xong bước 1 → nhắn Claude. Claude tạo `2_cho_gan_nhan.csv` (bỏ cột chức năng / yêu cầu) + hướng dẫn gán nhãn.

---

## Bước 2 — Gán nhãn (cuộc trò chuyện ChatGPT KHÁC) — Claude gửi hướng dẫn sau bước 1

Đính kèm `2_cho_gan_nhan.csv` + `BANG_TRA_HAM.md`, ChatGPT gán nhãn, rồi **bạn duyệt từng nhãn**.

## Bước 3 — Gửi Claude

Claude chuyển sang jsonl, kiểm schema, **kiểm trùng với MỌI dữ liệu train**, đóng băng (mã băm), tạo bản
"chữ đầu viết thường" bằng code — rồi mới chạy GRPO.
