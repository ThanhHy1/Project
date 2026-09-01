import pandas as pd


def build_popularity_model(ratings, min_ratings=50, use_filtered_mean_for_C=False):
    required_cols = {"userId", "movieId", "rating"}
    if not required_cols.issubset(ratings.columns):
        missing = required_cols - set(ratings.columns)
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")
    if ratings.empty:
        raise ValueError("ratings rỗng, không thể xây dựng mô hình popularity")

    # Tổng số rating của toàn bộ dataset
    total_mean = ratings["rating"].mean()

    # Thống kê rating theo từng phim
    movie_stats = (
        ratings.groupby("movieId")["rating"]
        .agg(["count", "mean"])
        .reset_index()
    )

    movie_stats.columns = [
        "movieId",
        "rating_count",
        "rating_mean"
    ]

    # Chỉ giữ những phim có đủ số lượng rating
    popularity = movie_stats[
        movie_stats["rating_count"] >= min_ratings
    ].copy()

    # Weighted Rating
    # C: rating trung bình toàn bộ dataset
    # m: số rating tối thiểu
    # v: số rating của phim
    # R: rating trung bình của phim
    if use_filtered_mean_for_C:
        C = popularity["rating_mean"].mean()   # chỉ tính trên phim đã lọc
    else:
        C = total_mean
    m = min_ratings
    v = popularity["rating_count"]
    R = popularity["rating_mean"]

    popularity["score"] = (
        (v / (v + m)) * R
        + (m / (v + m)) * C
    )

    # Sắp xếp theo độ phổ biến
    popularity = popularity.sort_values(
        by="score",
        ascending=False
    ).reset_index(drop=True)

    return popularity


def get_top_k(popularity, k=10):
    """
    Lấy danh sách Top-K phim phổ biến nhất.
    """

    return popularity.head(k)


if __name__ == "__main__":

    # Đọc dữ liệu rating
    ratings = pd.read_csv("./data/ratings.csv")

    # Xây dựng Popularity Baseline
    popularity = build_popularity_model(ratings, min_ratings=50, use_filtered_mean_for_C=True)

    # Lấy Top-10
    top_10 = get_top_k(popularity, k=10)

    print("\nTop 10 phim phổ biến nhất:")
    print(
        top_10[
            ["movieId", "rating_count", "rating_mean", "score"]
        ].to_string(index=False)
    )