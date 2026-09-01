"""
item_based.py

Module triển khai `ItemBasedRecommender` - mô hình gợi ý theo phương pháp
Item-based Collaborative Filtering (CF). Ý tưởng cốt lõi: một user có xu
hướng thích những item "giống" (tương tự về pattern rating) với những item
mà chính user đó đã từng thích trong quá khứ.

Thuật toán:
    1. Xây dựng ma trận Item-User từ dữ liệu rating.
    2. Tính độ tương tự (similarity) giữa các item bằng Cosine Similarity
       (dựa trên vector rating mà các user đã chấm cho từng item).
    3. Dự đoán rating của user cho một item dựa trên trung bình có trọng số
       (weighted average) giữa độ tương tự của item đó với các item mà
       user đã từng đánh giá (chỉ lấy k item tương tự nhất - k-NN).
"""

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from base import BaseRecommender


class ItemBasedRecommender(BaseRecommender):
    """
    Mô hình gợi ý Item-based Collaborative Filtering (k-NN trên item).

    Attributes:
        k_neighbors (int): Số lượng item láng giềng (item tương tự nhất)
            được dùng để tính điểm dự đoán.
        item_user_matrix (Optional[pd.DataFrame]): Ma trận Item-User gốc
            (index = movieId, columns = userId, values = rating, NaN nếu
            user chưa đánh giá item đó).
        item_similarity_matrix (Optional[pd.DataFrame]): Ma trận độ tương
            tự cosine giữa các item (index = columns = movieId).
        item_avg_ratings (Optional[pd.Series]): Rating trung bình của từng
            item, dùng cho fallback.
        global_mean_rating (float): Rating trung bình toàn hệ thống, dùng
            làm fallback cuối cùng.
        user_rated_items (Dict[int, pd.Series]): Ánh xạ userId -> Series
            (index = movieId, value = rating) chứa toàn bộ item mà user đó
            đã đánh giá, phục vụ cho predict/recommend.
    """

    def __init__(self, k_neighbors: int = 20) -> None:
        """
        Khởi tạo mô hình Item-based Collaborative Filtering.

        Args:
            k_neighbors (int, optional): Số lượng item tương tự nhất được
                sử dụng khi dự đoán rating. Mặc định là 20.

        Returns:
            None
        """
        super().__init__(k_neighbors=k_neighbors)
        self.k_neighbors: int = k_neighbors

        self.item_user_matrix: Optional[pd.DataFrame] = None
        self.item_similarity_matrix: Optional[pd.DataFrame] = None
        self.item_avg_ratings: Optional[pd.Series] = None
        self.global_mean_rating: float = 0.0
        self.user_rated_items: Dict[int, pd.Series] = {}

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Huấn luyện mô hình từ dữ liệu rating.

        Các bước thực hiện:
            1. Xây dựng ma trận Item-User (pivot table).
            2. Điền các ô chưa đánh giá bằng 0 để phục vụ tính cosine
               similarity giữa các item.
            3. Tính ma trận độ tương tự giữa các item bằng cosine_similarity.
            4. Lưu lại rating trung bình từng item, rating trung bình toàn
               hệ thống, và lịch sử đánh giá của từng user.

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

        # 1. Xây dựng ma trận Item-User (giữ NaN cho ô chưa đánh giá)
        self.item_user_matrix = train_data.pivot_table(
            index="movieId", columns="userId", values="rating"
        )

        # 2. Điền 0 cho ô chưa đánh giá để tính cosine similarity giữa item
        item_user_filled = self.item_user_matrix.fillna(0.0)

        # 3. Tính ma trận độ tương tự cosine giữa các item
        similarity_values = cosine_similarity(item_user_filled.values)
        self.item_similarity_matrix = pd.DataFrame(
            similarity_values,
            index=self.item_user_matrix.index,
            columns=self.item_user_matrix.index,
        )

        # 4. Rating trung bình từng item + rating trung bình toàn hệ thống
        self.item_avg_ratings = self.item_user_matrix.mean(axis=1, skipna=True)
        self.global_mean_rating = float(train_data["rating"].mean())

        # Lịch sử đánh giá của từng user: userId -> Series(movieId -> rating)
        self.user_rated_items = {
            user_id: group.set_index("movieId")["rating"]
            for user_id, group in train_data.groupby("userId")
        }

    def _fallback_rating(self, item_id: int) -> float:
        """
        Trả về giá trị dự đoán fallback khi không đủ dữ liệu để dự đoán
        bằng láng giềng item (user/item không tồn tại, hoặc không có
        item tương tự phù hợp).

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
        Dự đoán rating $\\hat{r}_{u,i}$ của `user_id` cho `item_id` bằng
        k-NN weighted average trên các item tương tự.

        Công thức:
            pred = [ sum(sim(i, j) * r_{u,j}) ] / [ sum(|sim(i, j)|) ]
            với j chạy trên k item tương tự nhất với item i = item_id
            trong số các item mà user_id đã từng đánh giá.

        Args:
            user_id (int): ID user cần dự đoán.
            item_id (int): ID item cần dự đoán.

        Returns:
            float: Rating dự đoán. Nếu user_id hoặc item_id chưa từng xuất
                hiện trong tập huấn luyện, hoặc không tìm được item tương
                tự hợp lệ (tổng trọng số bằng 0), trả về giá trị fallback
                (rating trung bình của item, hoặc trung bình toàn hệ thống).

        Raises:
            RuntimeError: Nếu mô hình chưa được huấn luyện (chưa gọi fit()).
        """
        if self.item_similarity_matrix is None or self.item_user_matrix is None:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")

        # Item hoặc user chưa từng xuất hiện trong tập huấn luyện -> fallback
        if (
            item_id not in self.item_similarity_matrix.index
            or user_id not in self.user_rated_items
        ):
            return self._fallback_rating(item_id)

        rated_items = self.user_rated_items[user_id]
        # Loại bỏ chính item_id (không tự làm láng giềng của chính mình)
        rated_items = rated_items.drop(index=item_id, errors="ignore")

        if rated_items.empty:
            return self._fallback_rating(item_id)

        # Lấy độ tương tự giữa item_id và các item mà user đã đánh giá
        similarities = self.item_similarity_matrix.loc[item_id, rated_items.index]

        # Chọn top k_neighbors item tương tự nhất
        top_neighbors = similarities.sort_values(ascending=False).head(
            self.k_neighbors
        )

        weights = top_neighbors.values
        neighbor_ratings = rated_items.loc[top_neighbors.index].values

        denominator = np.sum(np.abs(weights))

        # Xử lý chia cho 0: không có item nào có độ tương tự khác 0
        if denominator == 0.0:
            return self._fallback_rating(item_id)

        predicted_rating = float(np.sum(weights * neighbor_ratings) / denominator)

        return predicted_rating

    def recommend(self, user_id: int, top_k: int = 10) -> List[int]:
        """
        Sinh danh sách top-K item được gợi ý cho một user.

        Với mỗi item mà user_id chưa đánh giá, mô hình dùng `predict()` để
        ước lượng rating dựa trên độ tương tự với các item user đã đánh
        giá, sau đó sắp xếp giảm dần theo rating dự đoán và trả về top_k
        item có điểm cao nhất.

        Args:
            user_id (int): ID user cần gợi ý.
            top_k (int, optional): Số lượng item được gợi ý. Mặc định là 10.

        Returns:
            List[int]: Danh sách movieId của top_k item được gợi ý, sắp xếp
                theo điểm dự đoán giảm dần. Nếu user_id chưa từng xuất hiện
                trong tập huấn luyện (cold-start), trả về top_k item có
                rating trung bình cao nhất (fallback theo độ phổ biến).

        Raises:
            RuntimeError: Nếu mô hình chưa được huấn luyện (chưa gọi fit()).
        """
        if self.item_user_matrix is None:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")

        if top_k <= 0:
            return []

        all_items = self.item_user_matrix.index.tolist()

        if user_id in self.user_rated_items:
            seen_items: Set[int] = set(self.user_rated_items[user_id].index)
        else:
            seen_items = set()

        candidate_items = [item for item in all_items if item not in seen_items]

        # Cold-start: user chưa từng xuất hiện -> fallback theo rating trung bình
        if user_id not in self.user_rated_items:
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

    model = ItemBasedRecommender(k_neighbors=2)
    model.fit(data)

    print("Predict (user=1, item=40):", model.predict(1, 40))
    print("Predict (user=99 - user mới, item=10):", model.predict(99, 10))
    print("Predict (user=1, item=999 - item mới):", model.predict(1, 999))
    print("Recommend (user=1, top_k=3):", model.recommend(1, top_k=3))
    print("Recommend (user=99 - user mới, top_k=3):", model.recommend(99, top_k=3))