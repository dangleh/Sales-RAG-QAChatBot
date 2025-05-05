import pandas as pd

# Sample media production service quote
data = {
    "STT": [1, 2, 3, 4, 5],
    "Tên Dịch Vụ": [
        "Quay phim sự kiện",
        "Dịch vụ dựng phim",
        "Chỉnh màu (Color Grading)",
        "Thiết kế Motion Graphics",
        "Lồng tiếng & Hậu kỳ âm thanh"
    ],
    "Mô Tả Ngắn": [
        "Quay phim Full HD, 2 cameramen, 4 giờ",
        "Dựng, cắt ghép, hiệu ứng cơ bản, 5-7 phút video",
        "Tối ưu ánh sáng, tone màu cinematic cho video",
        "Thiết kế intro/outro, đồ họa chuyển động ngắn",
        "Chỉnh âm, mix nhạc, lồng tiếng cho video"
    ],
    "Đơn Vị Tính": ["Gói", "Gói", "Gói", "Gói", "Gói"],
    "Số Lượng": [1, 1, 1, 1, 1],
    "Đơn Giá (₫)": [5000000, 3000000, 2000000, 1500000, 1000000],
}

df = pd.DataFrame(data)

# Save to CSV
file_path = "./service_quote_media.csv"
df.to_csv(file_path, index=False)

import csv
docs = []
with open("service_quote_media.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Gom thông tin từng dịch vụ thành tập text
        text = (f"STT: {row['STT']}\n"
                f"Dịch vụ: {row['Tên Dịch Vụ']}\n"
                f"Mô tả: {row['Mô Tả Ngắn']}\n"
                f"Số lượng: {row['Số Lượng']}\n"
                f"Đơn giá: {row['Đơn Giá (₫)']}\n")
        docs.append(text)