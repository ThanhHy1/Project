"""
popularity.py

Module triển khai `PopularityRecommender` - một mô hình gợi ý đơn giản
dựa trên độ phổ biến (popularity-based recommender). Mô hình này gợi ý
các item được nhiều người dùng đánh giá nhất (số lượt rating cao nhất),
không cá nhân hóa theo sở thích riêng của từng user.

Đây thường được dùng làm baseline (mô hình nền) để so sánh với các
mô hình phức tạp hơn như Collaborative Filtering, Matrix Factorization.
"""

from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """
    Mô hình gợi ý dựa trên độ phổ biến (popularity-based).

    Độ phổ biến của một item được xác định bằng số lượt đánh giá (rating count)
    mà item đó nhận được trong tập huấn luyện. Các item có nhiều lượt đánh giá
    nhất sẽ được ưu tiên gợi ý cho mọi user (ngoại trừ những item user đó
    đã từng đánh giá).

    Attributes:
        popularity_ranking (Optional[pd.DataFrame]): Bảng xếp hạng các item
            theo độ phổ biến (số lượt rating) giảm dần, gồm các cột
            'movieId', 'rating_count', 'rating_mean'.
        item_avg_ratings (Optional[pd.Series]): Rating trung bình của từng
            'movieId', dùng để phục vụ cho hàm `predict`.
        global_mean_rating (float): Rating trung bình của toàn bộ hệ thống,
            dùng làm giá trị dự đoán mặc định khi item chưa từng xuất hiện.
        user_seen_items (Dict[int, Set[int]]): Ánh xạ từ `userId` sang tập hợp
            các `movieId` mà user đó đã đánh giá, dùng để loại bỏ item đã xem
            khi sinh danh sách gợi ý.
    """

    def __init__(self) -> None:
        """
        Khởi tạo các biến lưu trữ danh sách phim phổ biến và rating trung bình.

        Returns:
            None
        """
        super().__init__()
        self.popularity_ranking: Optional[pd.DataFrame] = None
        self.item_avg_ratings: Optional[pd.Series] = None
        self.global_mean_rating: float = 0.0
        self.user_seen_items: Dict[int, Set[int]] = {}

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Huấn luyện mô hình dựa trên dữ liệu rating.

        Quá trình huấn luyện gồm:
            1. Đếm số lượt đánh giá (popularity) và tính rating trung bình
               cho mỗi 'movieId'.
            2. Sắp xếp danh sách item theo số lượt đánh giá giảm dần.
            3. Lưu lại danh sách các item mà mỗi user đã từng đánh giá,
               phục vụ cho việc lọc khi gợi ý.

        Args:
            train_data (pd.DataFrame): DataFrame gồm các cột bắt buộc
                'userId', 'movieId', 'rating'.

        Returns:
            None

        Raises:
            ValueError: Nếu `train_data` thiếu một trong các cột bắt buộc
                hoặc rỗng.
        """
        required_columns = {"userId", "movieId", "rating"}
        if not required_columns.issubset(train_data.columns):
            raise ValueError(
                f"train_data phải chứa các cột {required_columns}, "
                f"nhận được các cột: {list(train_data.columns)}"
            )
        if train_data.empty:
            raise ValueError("train_data không được rỗng.")

        # Tính độ phổ biến (count) và rating trung bình (mean) cho mỗi movieId
        item_stats = (
            train_data.groupby("movieId")["rating"]
            .agg(rating_count="count", rating_mean="mean")
            .reset_index()
        )

        # Sắp xếp theo số lượt đánh giá giảm dần
        self.popularity_ranking = item_stats.sort_values(
            by="rating_count", ascending=False
        ).reset_index(drop=True)

        # Lưu rating trung bình từng item (dùng cho predict)
        self.item_avg_ratings = item_stats.set_index("movieId")["rating_mean"]

        # Rating trung bình toàn hệ thống (dùng khi item chưa từng xuất hiện)
        self.global_mean_rating = float(train_data["rating"].mean())

        # Lưu danh sách item mỗi user đã đánh giá (để loại bỏ khi gợi ý)
        self.user_seen_items = (
            train_data.groupby("userId")["movieId"].apply(set).to_dict()
        )

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Dự đoán rating cho một cặp (user_id, item_id).

        Vì đây là mô hình popularity-based (không cá nhân hóa), giá trị
        dự đoán chỉ phụ thuộc vào `item_id`, không phụ thuộc vào `user_id`.

        Args:
            user_id (int): ID người dùng (không ảnh hưởng đến kết quả dự đoán,
                được giữ lại để tuân thủ giao diện chung `BaseRecommender`).
            item_id (int): ID của item cần dự đoán rating.

        Returns:
            float: Rating trung bình của `item_id` trong tập huấn luyện.
                Nếu `item_id` chưa từng xuất hiện trong tập huấn luyện,
                trả về rating trung bình của toàn bộ hệ thống.

        Raises:
            RuntimeError: Nếu mô hình chưa được huấn luyện (chưa gọi `fit`).
        """
        if self.item_avg_ratings is None:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")

        if item_id in self.item_avg_ratings.index:
            return float(self.item_avg_ratings.loc[item_id])

        return self.global_mean_rating

    def recommend(self, user_id: int, top_k: int = 10) -> List[int]:
        """
        Sinh danh sách top-K item phổ biến nhất, loại bỏ các item user đã xem.

        Args:
            user_id (int): ID của người dùng cần gợi ý. Nếu user_id chưa từng
                xuất hiện trong tập huấn luyện, hệ thống sẽ gợi ý top-K item
                phổ biến nhất mà không cần lọc.
            top_k (int, optional): Số lượng item được gợi ý. Mặc định là 10.

        Returns:
            List[int]: Danh sách 'movieId' của top-K item phổ biến nhất còn
                lại (chưa được user đánh giá), sắp xếp theo độ phổ biến
                giảm dần. Có thể trả về ít hơn `top_k` phần tử nếu không đủ
                item phù hợp.

        Raises:
            RuntimeError: Nếu mô hình chưa được huấn luyện (chưa gọi `fit`).
        """
        if self.popularity_ranking is None:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")

        if top_k <= 0:
            return []

        seen_items = self.user_seen_items.get(user_id, set())

        ranked_items = self.popularity_ranking["movieId"].tolist()
        recommendations = [
            movie_id for movie_id in ranked_items if movie_id not in seen_items
        ]

        return recommendations[:top_k]


if __name__ == "__main__":
    # Ví dụ nhanh để kiểm tra mô hình
    data = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3, 3],
            "movieId": [10, 20, 10, 30, 10, 20, 40],
            "rating": [5, 4, 3, 5, 4, 2, 5],
        }
    )

    model = PopularityRecommender()
    model.fit(data)

    print("Popularity ranking:\n", model.popularity_ranking)
    print("Predict (user=1, item=10):", model.predict(1, 10))
    print("Predict (user=1, item=999 - chưa từng xuất hiện):", model.predict(1, 999))
    print("Recommend (user=1, top_k=3):", model.recommend(1, top_k=3))
    print("Recommend (user=99 - user mới, top_k=3):", model.recommend(99, top_k=3))