# Quy trình làm một dự án Agent / Chatbot dùng LLM

Ghi lại từ kinh nghiệm dự án Nexa Drive (agent điều khiển xe, 2026). Không riêng cho dự án này —
dùng làm khung cho các dự án agent / chatbot về sau. Mỗi bước có một ví dụ thật đã xảy ra.

---

## 1. Khác biệt cốt lõi so với ML truyền thống

**ML truyền thống** — đường thẳng:

```
dữ liệu (có sẵn) → xử lý → trực quan → train → eval → so sánh → xong
```

Dữ liệu thường có sẵn, nhãn đáng tin. Việc chính là chọn và chỉnh model.

**Agent / chatbot LLM** — vòng lặp:

```
định nghĩa hành vi → bộ đánh giá → thử không train ─┐
                                                    ▼
                       ┌──▶ dữ liệu → train → đánh giá → đọc lỗi ──┐
                       └───────────────── sửa dữ liệu ◀────────────┘
                                                    ▼
                             runtime + guardrail → thử người thật → giám sát
```

Dữ liệu **do mình tạo** (thường bằng một LLM khác) nên có thể sai theo cách không ngờ. Việc chính
là **đo cho đúng** và **sửa dữ liệu**, không phải chỉnh model. Phần lớn thời gian nằm ở vòng lặp giữa.

| | ML truyền thống | Agent / chatbot LLM |
|---|---|---|
| Dữ liệu | có sẵn, nhãn tin được | tự tạo, nhãn có thể sai |
| Thứ quan trọng nhất | model | bộ đánh giá và dữ liệu |
| Trình tự | đường thẳng | vòng lặp |
| Đo | một con số | theo từng nhóm tình huống |
| Tiêu chí | độ chính xác | độ chính xác + an toàn + độ trễ |
| An toàn | ít khi đặt ra | guardrail bằng code, tách khỏi model |
| Tìm lỗi | nhìn metric | đọc câu sai, xem vết |

---

## 2. Tám bước

### Bước 0 — Định nghĩa bằng HÀNH VI, không phải bằng model

Trả lời trước khi viết code:
- Agent **làm được gì**: danh sách công cụ/hàm, tham số, **nguồn chính thống** (đừng tự bịa).
- Các **loại tình huống** và hành vi đúng cho từng loại: rõ ràng / mơ hồ / ngoài phạm vi / nguy hiểm.
- Lỗi nào **nặng** (làm bậy, bịa số liệu), lỗi nào **nhẹ** (hỏi lại thừa).

> Ví dụ: 24 hàm lấy từ Android Automotive `VehiclePropertyIds`; câu mơ hồ thì đề xuất rồi hỏi lại
> (`<nexa_ask>`); nguyên tắc "thà từ chối nhầm còn hơn làm bậy".

### Bước 1 — Viết bộ đánh giá TRƯỚC khi có dữ liệu train

- Tập test **người viết**, tách hẳn khỏi nguồn sinh dữ liệu, **đóng băng** (ghi mã băm vào test tự động).
- Chấm **theo từng nhóm**, không gộp một con số (model từ chối tất cả vẫn được ~44% nếu gộp).
- **Chốt ngưỡng đạt trước** khi thấy kết quả.

> Ví dụ: val do teacher sinh cho **97,8%**, test người viết chỉ **66%**. Không có test viết tay thì
> đã tưởng model gần xong.

### Bước 2 — Thử KHÔNG train trước

Model có sẵn + đưa danh sách hàm vào prompt, đo trên bộ đánh giá. Nhiều khi thế là đủ. Chỉ fine-tune
khi có lý do cụ thể: chạy trên thiết bị, cần nhanh, cần rẻ, hoặc prompt không đạt. Kết quả bước này
là **mốc so sánh** bắt buộc: "fine-tune hơn không fine-tune bao nhiêu?".

### Bước 3 — Dữ liệu: sinh, lọc, và NGHI NGỜ nó

- Sinh bằng model lớn, nhưng **lọc bằng code mọi thứ code kiểm được** (tên hàm, enum, khoảng giá trị,
  chữ lạ, số bịa).
- Giám khảo LLM **phải được hiệu chỉnh** bằng câu có đáp án, đủ hai chiều (nên giữ / nên loại), và đo
  bằng **nhãn người** — không tin phần trăm đồng ý giữa hai máy.
- **Rà lệch trước khi train**: từ vựng (một hàm chỉ dùng một từ?), dạng câu (mệnh lệnh chỉ có ở mẫu
  dương?), phân bố hàm.
- **Kiểm nhiễm**: câu test có lọt vào dữ liệu train không — kể cả qua câu ví dụ mồi đưa cho teacher.

> Ví dụ đã gặp:
> - 267/267 câu "hạ kính" chỉ dùng chữ "cửa sổ", còn "kính" thuộc 100% về sấy kính → người nói "hạ
>   kính" thì model gọi sấy kính (đúng 8%).
> - Mẫu từ chối gần như không có câu mệnh lệnh → model học tắt "cứ ra lệnh là làm" → gọi bừa 17%.
> - Giám khảo gật đầu 517/518 mẫu, kể cả nhãn sai rõ ràng.
> - 66/296 câu test lọt vào dữ liệu train, thổi điểm lên ~5 điểm.

### Bước 4 — Train nhỏ, nhanh, nhiều vòng

Mỗi vòng rẻ để lặp được nhiều lần. Chọn checkpoint bằng **val**, không bao giờ bằng test.

### Bước 5 — Đánh giá và ĐỌC LỖI — trái tim của vòng lặp

Con số cho biết **có** lỗi; đọc từng câu sai mới biết **vì sao**. Chẩn đoán dựa trên dữ liệu train
và val — **không vá theo câu test** (vá theo test thì điểm vòng sau đẹp giả). Rồi quay lại bước 3.
Thường 3–5 vòng.

### Bước 6 — Runtime và guardrail

**Model lo hiểu, code lo an toàn.** Những gì model không nhìn thấy (xe đang chạy, có trẻ em, quyền
người dùng) thì code kiểm lúc chạy, không dạy model.

> Ví dụ: "Mở cốp" — nhãn vẫn là mở cốp; code từ chối khi xe đang chạy. Khoá trẻ em bật mà lệnh hạ kính
> sau đến từ "hey nexa" (ai cũng nói được) — code hỏi xác nhận; bấm mic (tài xế) thì làm luôn.

### Bước 7 — Đánh giá TRỌN CHUỖI và thử người thật

Agent có nhiều khâu (nghe → hiểu → gọi hàm → nhận kết quả → trả lời → hành động). Đúng từng khâu chưa
chắc cả chuỗi đúng. Xem **vết** từng lượt, đo độ trễ, cho người thật dùng — họ luôn nói những câu không
ai nghĩ tới.

### Bước 8 — Giám sát sau khi chạy thật

Ghi lại các lượt dùng; câu agent xử lý sai thì **thêm vào tập test**. Tập test lớn dần theo thực tế.

---

## 3. Những nguyên tắc rút ra (dán lên tường)

1. **Đo trước, sửa sau.** Không có thước đo đúng thì mọi cải tiến đều có thể là ảo.
2. **Test tách khỏi nguồn sinh dữ liệu, đóng băng.** Lý tưởng là câu người dùng thật; tối thiểu là một model
   KHÁC teacher sinh, không được xem dữ liệu train, người duyệt từng câu — và ghi đúng nguồn gốc (bài này: ChatGPT
   sinh + người duyệt, xem PLAN mục R; trước đó ghi nhầm là "người viết").
3. **Thứ gì code kiểm được thì không nhờ model chấm.**
4. **Hai máy đồng ý không có nghĩa là đúng.** Đo giám khảo bằng nhãn người, báo Cohen's kappa, và
   so với mức **người đồng ý với người**.
5. **Chia dữ liệu dev / giữ lại TRƯỚC khi nhìn lỗi.** Nhìn lỗi rồi mới chia thì phần "giữ lại" không
   còn sạch.
6. **Luôn kiểm bắt oan ở dữ liệu chưa từng nhìn.** Luật lọc trông hoàn hảo trên tập đã biết vẫn có thể
   bắt oan hàng loạt ngoài đó.
7. **Không bao giờ báo một con số gộp.**
8. **Model lo hiểu, code lo an toàn.**
9. **Sửa giám khảo cho hiểu đúng luật là được; vặn prompt tới khi ra số đẹp là không được.**
   Lỗi chỉ đổi chỗ sau hai lần sửa prompt → vấn đề ở KIỂU câu hỏi, không ở câu chữ.
10. **Ghi lại mọi thay đổi kèm bằng chứng** (vấn đề → bằng chứng → sửa → kết quả), kể cả những lần sai.
11. **Điểm tăng chưa chắc là tăng thật.** Kèm khoảng tin cậy (Wilson) cho mỗi tỉ lệ; so hai model trên
    cùng bộ câu thì dùng kiểm định McNemar (chỉ đếm câu đổi kết quả). p ≥ 0,05 → ghi "chưa khẳng
    định được", không ghi "đã cải thiện". McNemar chỉ tính nhiễu do chọn câu test, KHÔNG tính nhiễu do
    train lại: train cùng dữ liệu hai lần (khác thứ tự xáo trộn) có thể đổi kết quả hàng chục câu. So hai
    bộ dữ liệu thì train ≥2–3 hạt giống mỗi bên, báo trung bình + khoảng dao động.
13. **Dữ liệu "khó" phải đi theo cặp đối chứng.** Thêm một kiểu câu mới chỉ vào MỘT phía nhãn (vd câu kể hoàn
    cảnh chỉ có ở mẫu từ chối) thì model học tắt theo hình thức câu, không theo nghĩa. Kiểu câu mới phải có ở CẢ
    HAI phía nhãn; và tập dev phải có câu làm được cùng kiểu để bắt được loại lỗi này.
12. **Mỗi vòng chỉ một cụm thay đổi, mọi thứ khác giữ y hệt** (kể cả hạt giống ngẫu nhiên và bộ mẫu).
    Đổi hai thứ cùng lúc thì không biết cái nào có tác dụng.
14. **Ngưỡng / tham số sau train chọn trên val–dev, ghi khoá TRƯỚC khi chấm test, chấm test một lần.** Đo phân bố
    độ tự tin của câu đúng và câu sai trước: tách nhau mới dùng ngưỡng; chồng nhau thì ngưỡng không cứu được
    (model sai mà vẫn tự tin) và đó cũng là một kết quả phải ghi.
15. **Đọc kết quả theo HẬU QUẢ, không chỉ tổng điểm.** Một thay đổi có thể không tăng tổng điểm mà vẫn đáng dùng
    nếu nó đổi lỗi nặng (làm sai lên xe) lấy lỗi nhẹ (hỏi lại thừa). Đếm riêng số lần hành động sai.
16. **File lớn tải từ máy chủ phải kiểm trước khi dùng** (mã băm, hoặc chạy thử vài câu so với kết quả trên máy
    chủ). Trình tải song song tạo file đủ kích thước ngay từ đầu — kích thước đúng không có nghĩa là tải xong.

---

## 4. Tổ chức code và notebook

```
gen/ train/ eval/   ← code thật, có test tự động
job trên GPU thuê   ← sinh dữ liệu, train, chạy eval hàng loạt (chạy nền, tắt máy không mất)
notebooks/          ← xem, thử, phân tích — mỗi file một việc, chỉ đọc kết quả đã lưu
```

Không dồn pipeline nặng vào notebook: chạy nhầm một ô là mất tiền GPU, mất kết nối là mất job, và
chạy ô không theo thứ tự thì kết quả không tái lập được.

| Notebook | Thuộc bước | Làm gì |
|---|---|---|
| `01_du_lieu` | 3 | phân bố, rà lệch, so các vòng |
| `02_thu_model` | 2, 5 | gõ câu thử, xem model phản ứng |
| `03_danh_gia` | 1, 5 | điểm theo nhóm, so các phiên bản |
| `04_phan_tich_loi` | 5 | duyệt câu sai theo kiểu sai |
| `05_vet_agent` | 7 | xem trọn một lượt từ câu nói tới hành động |
| `00_tong_quan` | tổng kết | kể câu chuyện cho báo cáo / slide |

---

## 5. Tài liệu nên đọc

| Chủ đề | Tài liệu |
|---|---|
| Functional token, model nhỏ gọi hàm | Octopus v2 — arXiv 2404.01744 |
| Kiểm dữ liệu gọi hàm theo tầng | APIGen — arXiv 2406.18518; ToolACE — arXiv 2409.00920 |
| Giám khảo LLM và thiên kiến | Zheng et al. 2023 — arXiv 2306.05685; Thakur et al. 2024 — arXiv 2406.12624 |
| Nhiều giám khảo khác họ | Verga et al. 2024 — arXiv 2404.18796 |
| Tìm nhãn sai bằng model | Northcutt et al. — Confident Learning, arXiv 1911.00068 |
