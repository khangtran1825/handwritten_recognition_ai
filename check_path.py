import pandas as pd
import os

# Đường dẫn file đầu vào và đầu ra
txt_file = "gt_test.txt"
csv_output = "models/model_demo/val.csv"

# Kiểm tra nếu file txt tồn tại
if os.path.exists(txt_file):
    # Đọc file txt với dấu phân cách là Tab (\t)
    # Vì file của bạn không có dòng tiêu đề (header), chúng ta sẽ tự đặt tên cột
    df = pd.read_csv(txt_file, sep='\t', header=None, names=['image_path', 'label'])

    # Tiền xử lý: Nếu bạn muốn đường dẫn ảnh có thư mục con giống file val.csv cũ
    # Nhưng vì bạn đang để hết ảnh vào 1 folder, ta chỉ cần giữ nguyên tên file
    # df['image_path'] = df['image_path'] 

    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(csv_output), exist_ok=True)

    # Xuất ra file CSV (không ghi chỉ số dòng)
    df.to_csv(csv_output, index=False)

    print(f"✅ Đã chuyển đổi thành công! File lưu tại: {csv_output}")
    print(df.head())  # Hiển thị thử 5 dòng đầu
else:
    print(f"❌ Không tìm thấy file {txt_file} để chuyển đổi.")