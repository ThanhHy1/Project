import pandas as pd
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
ratings = pd.read_csv("./data/ratings.csv")

# Chia Train 80%, phần còn lại 20%
train_data, temp_data = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

# Chia 20% còn lại thành Validation 10% và Test 10%
val_data, test_data = train_test_split(
    temp_data,
    test_size=0.5,
    random_state=42
)

# Hiển thị kết quả theo dạng bảng dễ nhìn hơn
train_count = len(train_data)
val_count = len(val_data)
test_count = len(test_data)
total_count = len(ratings)

rows = [
    ["Train", f"{train_count:,}", f"{train_count / total_count * 100:.2f}%"],
    ["Validation", f"{val_count:,}", f"{val_count / total_count * 100:.2f}%"],
    ["Test", f"{test_count:,}", f"{test_count / total_count * 100:.2f}%"],
    ["Tổng", f"{total_count:,}", "100.00%"],
]

header = ["Tập dữ liệu", "Số Rating", "Tỷ lệ"]
col_widths = [
    max(len(str(row[0])) for row in [header] + rows),
    max(len(str(row[1])) for row in [header] + rows),
    max(len(str(row[2])) for row in [header] + rows),
]

def format_row(row):
    return " | ".join(str(value).ljust(col_widths[i]) for i, value in enumerate(row))

print("===== TRAIN / VALIDATION / TEST =====")
print(format_row(header))
print("-" * (sum(col_widths) + 6))
for row in rows:
    print(format_row(row))