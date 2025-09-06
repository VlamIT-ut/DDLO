## Giới thiệu
Dự án này mô phỏng **bài toán tối ưu offloading đa người dùng – đa tác vụ** bằng cách sử dụng **Memory DNN**.  
Mục tiêu: huấn luyện một mạng nơ-ron để quyết định nên xử lý tác vụ **tại local device** hay **offload lên server** nhằm giảm **độ trễ** và **năng lượng tiêu thụ**.

Hệ thống gồm 2 thành phần chính:

1. **Memory**  
   - Bộ nhớ replay lưu lại dữ liệu `(h, m)` và huấn luyện các mạng con (ensemble).  
   - Kiểm chứng bằng cách sinh dữ liệu ngẫu nhiên → thu được biểu đồ loss.  

2. **MUMT (Multi-User Multi-Task environment)**  
   - Môi trường mô phỏng hệ thống thực tế.  
   - Nhận input là **channel state + task size**, output là **chi phí Q-value** gồm năng lượng + độ trễ.  
   - Giúp đánh giá chất lượng quyết định offloading bằng **gain ratio**.
3. ** Cài đặt các thư viện cần thiết**
   - Sử dụng câu lệnh "pip install -r requirements.txt"
