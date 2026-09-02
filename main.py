"""
main.py

Script demo (entry point) của hệ thống gợi ý phim, minh họa việc huấn
luyện một hoặc nhiều mô hình gợi ý (Popularity Baseline, User-based CF,
Item-based CF, Matrix Factorization) trên tập dữ liệu MovieLens và sinh
danh sách Top-K phim được gợi ý cho một user cụ thể (Chương 4 - Xây dựng
hệ thống của file lý thuyết).

Khác với evaluate.py (dùng để đánh giá và so sánh định lượng các mô hình
bằng RMSE/Precision@K/Recall@K trên toàn bộ tập Test), main.py tập trung
vào việc DEMO trực quan: với một user_id do người dùng chỉ định, hệ
thống hiển thị:
    1. Một số phim mà user đó đã đánh giá cao trong tập Train (để tham
       khảo "gu" xem phim của user).
    2. Danh sách Top-K phim được (các) mô hình gợi ý cho user đó (kèm
       tên phim, tra cứu từ data/movies.csv).

Quy trình dữ liệu (đọc, chia Train/Validation/Test 80/10/10 với
random_state=42) và cách khởi tạo các mô hình được tái sử dụng trực tiếp
từ evaluate.py để đảm bảo nhất quán giữa demo và đánh giá.

Cách chạy:
    python main.py                                   # demo mặc định
    python main.py --model mf --user 1 --top_k 10
    python main.py --model all --user 42 --top_k 5
    python main.py --list-users                       # xem một số userId hợp lệ
"""

import argparse
import os
import sys
from typing import Dict, List

import pandas as pd

# --------------------------------------------------------------------- #
# Thiết lập đường dẫn để import models/, utils/ và các hàm dùng chung
# trong evaluate.py (load_ratings, split_data, get_models_to_evaluate, ...)
# --------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

for _path in (PROJECT_ROOT, MODELS_DIR, UTILS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from base import BaseRecommender  # noqa: E402
from popularity_baseline import PopularityRecommender  # noqa: E402
from user_based_CF import UserBasedRecommender  # noqa: E402
from item_based_CF import ItemBasedRecommender  # noqa: E402
from matrix_factorization import MatrixFactorizationRecommender  # noqa: E402

from evaluate import (  # noqa: E402
    RANDOM_STATE,
    load_ratings,
    split_data,
)


# --------------------------------------------------------------------- #
# Đăng ký các mô hình có thể chọn qua tham số dòng lệnh --model
# --------------------------------------------------------------------- #
MODEL_REGISTRY: Dict[str, str] = {
    "popularity": "Popularity Baseline",
    "user_cf": "User-based CF",
    "item_cf": "Item-based CF",
    "mf": "Matrix Factorization (SVD)",
}


def build_model(model_key: str) -> BaseRecommender:
    """
    Khởi tạo một instance mô hình (chưa huấn luyện) tương ứng với
    `model_key`.

    Args:
        model_key (str): Một trong các khóa của MODEL_REGISTRY
            ('popularity', 'user_cf', 'item_cf', 'mf').

    Returns:
        BaseRecommender: Instance mô hình gợi ý tương ứng, cấu hình
            tham số mặc định giống với evaluate.py để đảm bảo nhất quán.

    Raises:
        ValueError: Nếu `model_key` không hợp lệ.
    """
    if model_key == "popularity":
        return PopularityRecommender()
    if model_key == "user_cf":
        return UserBasedRecommender(k_neighbors=20)
    if model_key == "item_cf":
        return ItemBasedRecommender(k_neighbors=20)
    if model_key == "mf":
        return MatrixFactorizationRecommender(
            n_factors=30,
            n_epochs=20,
            learning_rate=0.01,
            regularization=0.05,
            random_state=RANDOM_STATE,
        )
    raise ValueError(
        f"model_key không hợp lệ: '{model_key}'. "
        f"Các giá trị hợp lệ: {list(MODEL_REGISTRY.keys()) + ['all']}"
    )


def load_movie_titles(data_dir: str = DATA_DIR) -> pd.Series:
    """
    Đọc data/movies.csv và trả về ánh xạ movieId -> title, dùng để hiển
    thị tên phim thay vì chỉ hiển thị movieId.

    Args:
        data_dir (str): Đường dẫn thư mục chứa dữ liệu.

    Returns:
        pd.Series: Series với index là movieId, giá trị là title.
    """
    movies_path = os.path.join(data_dir, "movies.csv")
    movies = pd.read_csv(movies_path)
    return movies.set_index("movieId")["title"]


def movie_title(movie_id: int, movie_titles: pd.Series) -> str:
    """
    Tra cứu tên phim từ movieId, trả về nhãn dự phòng nếu không tìm thấy.

    Args:
        movie_id (int): ID phim cần tra cứu.
        movie_titles (pd.Series): Ánh xạ movieId -> title (từ
            `load_movie_titles`).

    Returns:
        str: Tên phim, hoặc "Movie <movieId> (không rõ tên)" nếu không
            tìm thấy trong data/movies.csv.
    """
    if movie_id in movie_titles.index:
        return str(movie_titles.loc[movie_id])
    return f"Movie {movie_id} (không rõ tên)"


def print_user_history(
    user_id: int,
    train_data: pd.DataFrame,
    movie_titles: pd.Series,
    n_movies: int = 5,
) -> None:
    """
    In ra một số phim user đã đánh giá cao nhất trong tập Train, giúp
    người xem demo có ngữ cảnh về "gu" xem phim của user trước khi xem
    danh sách gợi ý.

    Args:
        user_id (int): ID user cần hiển thị lịch sử.
        train_data (pd.DataFrame): Tập dữ liệu Train.
        movie_titles (pd.Series): Ánh xạ movieId -> title.
        n_movies (int, optional): Số phim hiển thị. Mặc định là 5.

    Returns:
        None
    """
    user_history = train_data[train_data["userId"] == user_id]

    if user_history.empty:
        print(
            f"User {user_id} không có dữ liệu trong tập Train "
            f"(cold-start user)."
        )
        return

    top_rated = user_history.sort_values("rating", ascending=False).head(n_movies)
    print(f"Một số phim User {user_id} đã đánh giá cao (trong tập Train):")
    for _, row in top_rated.iterrows():
        title = movie_title(int(row["movieId"]), movie_titles)
        print(f"    - {title}  (rating: {row['rating']})")


def print_recommendations(
    model_name: str,
    model: BaseRecommender,
    user_id: int,
    top_k: int,
    movie_titles: pd.Series,
) -> None:
    """
    Sinh và in ra danh sách Top-K phim được gợi ý cho user_id bởi một mô
    hình đã huấn luyện.

    Args:
        model_name (str): Tên hiển thị của mô hình.
        model (BaseRecommender): Mô hình đã được huấn luyện (đã gọi fit()).
        user_id (int): ID user cần gợi ý.
        top_k (int): Số lượng phim được gợi ý.
        movie_titles (pd.Series): Ánh xạ movieId -> title.

    Returns:
        None
    """
    recommended_ids = model.recommend(user_id, top_k=top_k)

    print(f"\n--- Gợi ý từ mô hình: {model_name} ---")
    if not recommended_ids:
        print("    (Không có gợi ý nào được sinh ra.)")
        return

    for rank, movie_id in enumerate(recommended_ids, start=1):
        title = movie_title(movie_id, movie_titles)
        predicted_rating = model.predict(user_id, movie_id)
        print(f"    {rank:2d}. {title}  (dự đoán rating: {predicted_rating:.2f})")


def list_sample_users(train_data: pd.DataFrame, n: int = 10) -> None:
    """
    In ra một số userId hợp lệ (kèm số lượng phim đã đánh giá trong tập
    Train), giúp người dùng chọn user_id để chạy demo.

    Args:
        train_data (pd.DataFrame): Tập dữ liệu Train.
        n (int, optional): Số user hiển thị. Mặc định là 10.

    Returns:
        None
    """
    rating_counts = (
        train_data.groupby("userId")["movieId"].count().sort_values(ascending=False)
    )
    print(f"Một số userId hợp lệ (Top {n} theo số lượng rating trong Train):")
    for user_id, count in rating_counts.head(n).items():
        print(f"    - userId={user_id}: {count} rating")


def parse_args() -> argparse.Namespace:
    """
    Phân tích tham số dòng lệnh cho main.py.

    Returns:
        argparse.Namespace: Đối tượng chứa các tham số đã parse
            (model, user, top_k, history, list_users).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Demo hệ thống gợi ý phim: huấn luyện mô hình và sinh danh "
            "sách Top-K phim được gợi ý cho một user."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help="Mô hình dùng để gợi ý: popularity | user_cf | item_cf | mf | all (mặc định: all).",
    )
    parser.add_argument(
        "--user",
        type=int,
        default=1,
        help="userId cần sinh gợi ý (mặc định: 1).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Số lượng phim được gợi ý (mặc định: 10).",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=5,
        help="Số phim hiển thị trong lịch sử đánh giá của user (mặc định: 5).",
    )
    parser.add_argument(
        "--list-users",
        action="store_true",
        help="Chỉ hiển thị một số userId hợp lệ rồi thoát (không chạy demo).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Hàm chính: đọc dữ liệu, chia Train/Validation/Test, huấn luyện (các)
    mô hình được chọn qua tham số dòng lệnh, rồi in ra lịch sử đánh giá
    và danh sách Top-K phim được gợi ý cho user được chỉ định.

    Returns:
        None
    """
    args = parse_args()

    print("===== ĐỌC VÀ CHIA DỮ LIỆU =====")
    ratings = load_ratings()
    movie_titles = load_movie_titles()
    train_data, val_data, test_data = split_data(ratings, random_state=RANDOM_STATE)
    print(
        f"Train: {len(train_data):,} | Validation: {len(val_data):,} | "
        f"Test: {len(test_data):,}"
    )

    if args.list_users:
        list_sample_users(train_data, n=10)
        return

    print(f"\n===== DEMO GỢI Ý PHIM CHO USER {args.user} =====")
    print_user_history(args.user, train_data, movie_titles, n_movies=args.history)

    model_keys: List[str] = (
        list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    )

    for model_key in model_keys:
        model_name = MODEL_REGISTRY[model_key]
        model = build_model(model_key)

        print(f"\n>>> Đang huấn luyện mô hình: {model_name} ...")
        model.fit(train_data)

        print_recommendations(model_name, model, args.user, args.top_k, movie_titles)


if __name__ == "__main__":
    main()