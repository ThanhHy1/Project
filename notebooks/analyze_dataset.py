import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. Đọc dữ liệu
# =========================

ratings = pd.read_csv("./data/ratings.csv")
movies = pd.read_csv("./data/movies.csv")

# =========================
# 2. Thông tin tổng quan
# =========================

print("===== DATASET OVERVIEW =====")

print("Số lượng User:", ratings["userId"].nunique())
print("Số lượng Movie trong ratings:", ratings["movieId"].nunique())
print("Số lượng Movie trong movies:", movies["movieId"].nunique())
print("Số lượng Rating:", len(ratings))

print("\nKhoảng Rating:")
print("Min:", ratings["rating"].min())
print("Max:", ratings["rating"].max())

print("Rating trung bình:", round(ratings["rating"].mean(), 3))

num_users = ratings["userId"].nunique()
num_movies = ratings["movieId"].nunique()
num_ratings = len(ratings)

total_possible = num_users * num_movies

sparsity = 1 - (num_ratings / total_possible)

print("\n===== MATRIX SPARSITY =====")
print("Tổng số tương tác có thể có:", total_possible)
print("Số Rating thực tế:", num_ratings)
print("Độ thưa dữ liệu:", round(sparsity * 100, 2), "%")
print("Độ đầy đủ dữ liệu:", round((1 - sparsity) * 100, 2), "%")


# =========================
# 3. Thông tin ratings.csv
# =========================

print("\n===== RATINGS.CSV =====")

print("Các cột:")
print(ratings.columns.tolist())

print("\n5 dòng đầu:")
print(ratings.head())

print("\nKiểu dữ liệu:")
print(ratings.dtypes)


# =========================
# 4. Thông tin movies.csv
# =========================

print("\n===== MOVIES.CSV =====")

print("Các cột:")
print(movies.columns.tolist())

print("\n5 dòng đầu:")
print(movies.head())

print("\nKiểu dữ liệu:")
print(movies.dtypes)


# =========================
# 5. Số rating trung bình / User
# =========================

ratings_per_user = ratings.groupby("userId").size()

print("\n===== USER STATISTICS =====")

print("Rating trung bình / User:",
      round(ratings_per_user.mean(), 2))

print("Rating ít nhất / User:",
      ratings_per_user.min())

print("Rating nhiều nhất / User:",
      ratings_per_user.max())

print("\nThống kê số lượng Rating theo User:")
print(ratings_per_user.describe())

plt.figure(1, figsize=(8, 5))
plt.hist(ratings_per_user, bins=30, color="skyblue", edgecolor="black")
plt.title("Phân bố số lượng Rating theo User")
plt.xlabel("Số lượng Rating")
plt.ylabel("Số lượng User")
plt.tight_layout()
plt.show(block=False)
plt.pause(0.001)

# =========================
# 6. Số rating trung bình / Movie
# =========================

ratings_per_movie = ratings.groupby("movieId").size()

print("\n===== MOVIE STATISTICS =====")

print("Rating trung bình / Movie:",
      round(ratings_per_movie.mean(), 2))

print("Rating ít nhất / Movie:",
      ratings_per_movie.min())

print("Rating nhiều nhất / Movie:",
      ratings_per_movie.max())

plt.figure(3, figsize=(8, 5))
plt.hist(ratings_per_movie, bins=30, color="lightgreen", edgecolor="black")
plt.title("Phân bố số lượng Rating theo Movie")
plt.xlabel("Số lượng Rating")
plt.ylabel("Số lượng Movie")
plt.tight_layout()
plt.show(block=False)
plt.pause(0.001)

# =========================
# 7. Kiểm tra dữ liệu thiếu
# =========================

print("\n===== MISSING VALUES =====")

print("Ratings:")
print(ratings.isnull().sum())

print("\nMovies:")
print(movies.isnull().sum())


# =========================
# 8. Kiểm tra dữ liệu trùng
# =========================

print("\n===== DUPLICATES =====")

print("Rating trùng:", ratings.duplicated().sum())
print("Movie trùng:", movies.duplicated().sum())
top_movies = (
    ratings.groupby("movieId")
    .agg(
        rating_count=("rating", "count"),
        rating_mean=("rating", "mean")
    )
    .sort_values("rating_count", ascending=False)
    .head(10)
    .reset_index()
)

top_movies = top_movies.merge(
    movies[["movieId", "title"]],
    on="movieId",
    how="left"
)

top_movies = top_movies[
    ["movieId", "title", "rating_count", "rating_mean"]
]

print(top_movies)
rating_distribution = (
    ratings["rating"]
    .value_counts()
    .sort_index()
)

print(rating_distribution)

plt.figure(2, figsize=(8, 5))
plt.bar(rating_distribution.index.astype(str), rating_distribution.values, color="salmon", edgecolor="black")
plt.title("Phân bố Rating")
plt.xlabel("Rating")
plt.ylabel("Số lượng Rating")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show(block=False)
plt.pause(0.001)

input("Nhấn Enter để đóng 3 cửa sổ biểu đồ...")