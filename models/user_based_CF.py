"""
user_based.py

Module triển khai `UserBasedRecommender` - mô hình gợi ý theo phương pháp
User-based Collaborative Filtering (CF). Ý tưởng cốt lõi: những user có
"khẩu vị" (pattern rating) giống nhau trong quá khứ sẽ có xu hướng thích
những item giống nhau trong tương lai.

Thuật toán:
    1. Xây dựng ma trận User-Item từ dữ liệu rating.
    2. Áp dụng Mean-Centering (trừ đi rating trung bình của từng user) để
       loại bỏ độ lệch (bias) trong thói quen chấm điểm giữa các user.
    3. Tính độ tương tự (similarity) giữa các user bằng Cosine Similarity.
    4. Dự đoán rating của một user cho một item dựa trên trung bình có
       trọng số (weighted average) từ k user láng giềng gần nhất (k-NN)
       đã từng đánh giá item đó.
"""

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from base import BaseRecommender


class UserBasedRecommender(BaseRecommender):
    """
    Mô hình gợi ý User-based Collaborative Filtering (k-NN trên user).

    Attributes:
        k_neighbors (int): Số lượng user láng giềng gần nhất được dùng để
            dự đoán rating.
        user_item_matrix (Optional[pd.DataFrame]): Ma trận User-Item gốc
            (index = userId, columns = movieId, values = rating, NaN nếu
            user chưa đánh giá item đó).
        user_means (Optional[pd.Series]): Rating trung bình của từng user
            (chỉ tính trên các item user đó đã đánh giá).
        user_similarity_matrix (Optional[pd.DataFrame]): Ma trận độ tương tự
            cosine giữa các user (index = columns = userId).
        item_avg_ratings (Optional[pd.Series]): Rating trung bình của từng
            item, dùng cho fallback khi không đủ dữ liệu láng giềng.
        global_mean_rating (float): Rating trung bình toàn hệ thống, dùng
            làm fallback cuối cùng.
        user_seen_items (Dict[int, Set[int]]): Ánh xạ userId -> tập hợp các
            movieId user đó đã đánh giá.
    """

    def __init__(self, k_neighbors: int = 20) -> None:
        """
        Khởi tạo mô hình User-based Collaborative Filtering.

        Args:
            k_neighbors (int, optional): Số lượng user láng giềng gần nhất
                được sử dụng khi dự đoán rating. Mặc định là 20.

        Returns:
            None
        """
        super().__init__(k_neighbors=k_neighbors)
        self.k_neighbors: int = k_neighbors

        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.user_means: Optional[pd.Series] = None
        self.user_similarity_matrix: Optional[pd.DataFrame] = None
        self.item_avg_ratings: Optional[pd.Series] = None
        self.global_mean_rating: float = 0.0
        self.user_seen_items: Dict[int, Set[int]] = {}

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Huấn luyện mô hình từ dữ liệu rating.

        Các bước thực hiện:
            1. Xây dựng ma trận User-Item (pivot table).
            2. Tính rating trung bình của mỗi user (bỏ qua các ô NaN).
            3. Mean-centering: trừ mỗi rating cho trung bình của user đó,
               các ô chưa đánh giá được điền 0 để phục vụ tính cosine similarity.
            4. Tính ma trận độ tương tự giữa các user bằng cosine_similarity.
            5. Lưu lại rating trung bình từng item và danh sách item mỗi
               user đã xem, phục vụ cho predict/recommend.

        Args:
            train_data (pd.DataFrame): DataFrame gồm các cột bắt buộc
                'userId', 'movieId', 'rating'.

        Returns:
            None

        Raises:
            ValueError: Nếu `train_data` thiếu cột bắt buộc hoặc rỗng.
        """
        required_columns = {"userId", "movieId", "rating"}
        if not required_columns.issubset(train_data.columns):
            raise ValueError(
                f"train_data phải chứa các cột {required_columns}, "
                f"nhận được các cột: {list(train_data.columns)}"
            )
        if train_data.empty:
            raise ValueError("train_data không được rỗng.")

        # 1. Xây dựng ma trận User-Item (giữ NaN cho ô chưa đánh giá)
        self.user_item_matrix = train_data.pivot_table(
            index="userId", columns="movieId", values="rating"
        )

        # 2. Rating trung bình của mỗi user (chỉ tính trên item đã đánh giá)
        self.user_means = self.user_item_matrix.mean(axis=1, skipna=True)

        # 3. Mean-centering, điền 0 cho ô chưa đánh giá để tính similarity
        centered_matrix = self.user_item_matrix.sub(self.user_means, axis=0)
        centered_matrix_filled = centered_matrix.fillna(0.0)

        # 4. Tính ma trận độ tương tự cosine giữa các user
        similarity_values = cosine_similarity(centered_matrix_filled.values)
        self.user_similarity_matrix = pd.DataFrame(
            similarity_values,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )

        # 5. Rating trung bình từng item + rating trung bình toàn hệ thống
        self.item_avg_ratings = self.user_item_matrix.mean(axis=0, skipna=True)
        self.global_mean_rating = float(train_data["rating"].mean())

        # Danh sách item mỗi user đã đánh giá (để lọc khi gợi ý)
        self.user_seen_items = (
            train_data.groupby("userId")["movieId"].apply(set).to_dict()
        )

    def _fallback_rating(self, item_id: int) -> float:
        """
        Trả về giá trị dự đoán fallback khi không đủ dữ liệu để dự đoán
        bằng láng giềng (user hoặc item không tồn tại, hoặc không có
        láng giềng phù hợp).

        Args:
            item_id (int): ID của item cần lấy rating fallback.

        Returns:
            float: Rating trung bình của item nếu item tồn tại trong tập
                huấn luyện, ngược lại trả về rating trung bình toàn hệ thống.
        """
        if (
            self.item_avg_ratings is not None
            and item_id in self.item_avg_ratings.index
            and not np.isnan(self.item_avg_ratings.loc[item_id])
        ):
            return float(self.item_avg_ratings.loc[item_id])
        return self.global_mean_rating

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Dự đoán rating của `user_id` cho `item_id` bằng k-NN weighted average.

        Công thức:
            pred = mean(user) + [ sum(sim_i * (rating_i - mean_i)) ]
                                  / [ sum(|sim_i|) ]
            với i chạy trên k user láng giềng gần nhất (theo cosine similarity)
            đã từng đánh giá item_id.

        Args:
            user_id (int): ID user cần dự đoán.
            item_id (int): ID item cần dự đoán.

        Returns:
            float: Rating dự đoán. Nếu user_id hoặc item_id chưa từng xuất
                hiện trong tập huấn luyện, hoặc không tìm được láng giềng
                hợp lệ (tổng trọng số bằng 0), trả về giá trị fallback
                (rating trung bình của item, hoặc trung bình toàn hệ thống).

        Raises:
            RuntimeError: Nếu mô hình chưa được huấn luyện (chưa gọi fit()).
        """
        if self.user_item_matrix is None or self.user_similarity_matrix is None:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")

        # User hoặc item chưa từng xuất hiện trong tập huấn luyện -> fallback
        if (
            user_id not in self.user_item_matrix.index
            or item_id not in self.user_item_matrix.columns
        ):
            return self._fallback_rating(item_id)

        # Lấy rating của tất cả user đã từng đánh giá item_id (bỏ NaN)
        item_ratings = self.user_item_matrix[item_id].dropna()
        # Loại bỏ chính user_id (không tự làm láng giềng của chính mình)
        item_ratings = item_ratings.drop(index=user_id, errors="ignore")

        if item_ratings.empty:
            return self._fallback_rating(item_id)

        # Lấy độ tương tự giữa user_id và các user đã đánh giá item_id
        similarities = self.user_similarity_matrix.loc[user_id, item_ratings.index]

        # Chọn top k_neighbors láng giềng có độ tương tự cao nhất
        top_neighbors = similarities.sort_values(ascending=False).head(
            self.k_neighbors
        )

        weights = top_neighbors.values
        neighbor_ids = top_neighbors.index
        neighbor_ratings = item_ratings.loc[neighbor_ids].values
        neighbor_means = self.user_means.loc[neighbor_ids].values

        denominator = np.sum(np.abs(weights))

        # Xử lý chia cho 0: không có láng giềng nào có độ tương tự khác 0
        if denominator == 0.0:
            return self._fallback_rating(item_id)

        numerator = np.sum(weights * (neighbor_ratings - neighbor_means))
        predicted_rating = float(self.user_means.loc[user_id]) + (
            numerator / denominator
        )

        return float(predicted_rating)

    def recommend(self, user_id: int, top_k: int = 10) -> List[int]:
        """
        Sinh danh sách top-K item được gợi ý cho một user.

        Với mỗi item mà user_id chưa đánh giá, mô hình dùng `predict()` để
        ước lượng rating, sau đó sắp xếp giảm dần theo rating dự đoán và
        trả về top_k item có điểm cao nhất.

        Args:
            user_id (int): ID user cần gợi ý.
            top_k (int, optional): Số lượng item được gợi ý. Mặc định là 10.

        Returns:
            List[int]: Danh sách movieId của top_k item được gợi ý, sắp xếp
                theo điểm dự đoán giảm dần. Nếu user_id chưa từng xuất hiện
                trong tập huấn luyện (cold-start), trả về top_k item có
                rating trung bình cao nhất (fallback dựa trên độ phổ biến).

        Raises:
            RuntimeError: Nếu mô hình chưa được huấn luyện (chưa gọi fit()).
        """
        if self.user_item_matrix is None:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")

        if top_k <= 0:
            return []

        all_items = self.user_item_matrix.columns.tolist()
        seen_items = self.user_seen_items.get(user_id, set())
        candidate_items = [item for item in all_items if item not in seen_items]

        # Cold-start: user chưa từng xuất hiện -> fallback theo rating trung bình
        if user_id not in self.user_item_matrix.index:
            fallback_ranking = self.item_avg_ratings.loc[candidate_items].sort_values(
                ascending=False
            )
            return fallback_ranking.head(top_k).index.tolist()

        # Dự đoán rating cho từng item chưa xem
        predicted_scores = {
            item_id: self.predict(user_id, item_id) for item_id in candidate_items
        }

        ranked_items = sorted(
            predicted_scores.items(), key=lambda pair: pair[1], reverse=True
        )

        return [item_id for item_id, _ in ranked_items[:top_k]]


if __name__ == "__main__":
    # Ví dụ nhanh để kiểm tra mô hình
    data = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
            "movieId": [10, 20, 30, 10, 20, 10, 30, 40, 20, 40],
            "rating": [5, 4, 3, 5, 5, 2, 4, 5, 3, 4],
        }
    )

    model = UserBasedRecommender(k_neighbors=2)
    model.fit(data)

    print("Predict (user=1, item=40):", model.predict(1, 40))
    print("Predict (user=99 - user mới, item=10):", model.predict(99, 10))
    print("Predict (user=1, item=999 - item mới):", model.predict(1, 999))
    print("Recommend (user=1, top_k=3):", model.recommend(1, top_k=3))
    print("Recommend (user=99 - user mới, top_k=3):", model.recommend(99, top_k=3))