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

        # --- Cache dạng NumPy (hiệu năng) ---
        # Tương tự UserBasedRecommender: lưu lại đúng nội dung của
        # item_similarity_matrix / item_avg_ratings / user_rated_items dưới
        # dạng mảng NumPy thuần để predict()/recommend() tránh chi phí rất
        # lớn của việc tra cứu bằng pandas.loc trong vòng lặp Python. Kết
        # quả tính toán tương đương về mặt toán học với cách làm bằng
        # pandas ở trên.
        self._item_id_to_idx: Dict[int, int] = {}
        self._item_ids_arr: np.ndarray = np.array([])
        self._sim_np: np.ndarray = np.array([])
        self._item_avg_np: np.ndarray = np.array([])
        self._user_rated_idx: Dict[int, np.ndarray] = {}
        self._user_rated_ratings: Dict[int, np.ndarray] = {}

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

        # 5. Xây dựng cache NumPy để tăng tốc predict()/recommend() (xem
        # giải thích ở phần khai báo thuộc tính __init__). Dữ liệu ở đây
        # được suy ra trực tiếp từ các cấu trúc pandas đã tính ở trên,
        # không làm thay đổi kết quả, chỉ giúp tính nhanh hơn.
        self._item_id_to_idx = {
            iid: idx for idx, iid in enumerate(self.item_user_matrix.index)
        }
        self._item_ids_arr = self.item_user_matrix.index.to_numpy()
        self._sim_np = self.item_similarity_matrix.to_numpy(dtype=np.float64)
        self._item_avg_np = self.item_avg_ratings.to_numpy(dtype=np.float64)

        self._user_rated_idx = {}
        self._user_rated_ratings = {}
        for user_id, series in self.user_rated_items.items():
            idxs = series.index.map(self._item_id_to_idx).to_numpy(dtype=np.int64)
            self._user_rated_idx[user_id] = idxs
            self._user_rated_ratings[user_id] = series.to_numpy(dtype=np.float64)

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

    def _predict_by_index(
        self, item_idx: int, rated_idx: np.ndarray, rated_ratings: np.ndarray
    ) -> Optional[float]:
        """
        Phiên bản NumPy thuần của công thức dự đoán k-NN weighted average
        trên item, làm việc trực tiếp trên chỉ số (index) nội bộ. Đây là
        phần lõi tính toán được cả `predict()` và `recommend()` dùng
        chung, cài đặt ĐÚNG công thức toán học như mô tả trong `predict()`
        nhưng tránh chi phí tra cứu (indexing) rất lớn của pandas khi phải
        gọi lặp lại hàng nghìn lần (trong `recommend()`).

        Args:
            item_idx (int): Chỉ số nội bộ của item cần dự đoán.
            rated_idx (np.ndarray): Chỉ số nội bộ các item mà user đã đánh
                giá (đã bao gồm hoặc chưa loại trừ item_idx).
            rated_ratings (np.ndarray): Rating tương ứng với `rated_idx`.

        Returns:
            Optional[float]: Rating dự đoán, hoặc None nếu không có item
                tương tự hợp lệ (cần dùng giá trị fallback).
        """
        # Loại bỏ chính item_idx (không tự làm láng giềng của chính mình)
        keep_mask = rated_idx != item_idx
        if not np.all(keep_mask):
            rated_idx = rated_idx[keep_mask]
            rated_ratings = rated_ratings[keep_mask]

        if rated_idx.size == 0:
            return None

        sims = self._sim_np[item_idx, rated_idx]

        if rated_idx.size > self.k_neighbors:
            top_pos = np.argpartition(-sims, self.k_neighbors)[: self.k_neighbors]
            sims = sims[top_pos]
            rated_ratings = rated_ratings[top_pos]

        denominator = np.sum(np.abs(sims))
        if denominator == 0.0:
            return None

        return float(np.sum(sims * rated_ratings) / denominator)

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Dự đoán rating $\\hat{r}_{u,i}$ của `user_id` cho `item_id` bằng
        k-NN weighted average trên các item tương tự.

        Công thức:
            pred = [ sum(sim(i, j) * r_{u,j}) ] / [ sum(|sim(i, j)|) ]
            với j chạy trên k item tương tự nhất với item i = item_id
            trong số các item mà user_id đã từng đánh giá.

        Ghi chú hiệu năng: phần tính toán thực sự được thực hiện bằng
        NumPy thuần trong `_predict_by_index()` (xem ở trên) để tránh chi
        phí lớn của việc tra cứu bằng pandas.loc/sort_values khi hàm này
        được gọi lặp lại rất nhiều lần (ví dụ trong `recommend()`). Công
        thức và kết quả trả về hoàn toàn tương đương với cách tính bằng
        pandas.

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

        item_idx = self._item_id_to_idx.get(item_id)

        # Item hoặc user chưa từng xuất hiện trong tập huấn luyện -> fallback
        if item_idx is None or user_id not in self._user_rated_idx:
            return self._fallback_rating(item_id)

        rated_idx = self._user_rated_idx[user_id]
        rated_ratings = self._user_rated_ratings[user_id]

        predicted_rating = self._predict_by_index(item_idx, rated_idx, rated_ratings)
        if predicted_rating is None:
            return self._fallback_rating(item_id)

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

        # Cold-start: user chưa từng xuất hiện -> fallback theo rating trung bình
        if user_id not in self.user_rated_items:
            seen_items: Set[int] = set()
            candidate_items = [
                item for item in self.item_user_matrix.index if item not in seen_items
            ]
            fallback_ranking = self.item_avg_ratings.loc[candidate_items].sort_values(
                ascending=False
            )
            return fallback_ranking.head(top_k).index.tolist()

        seen_items = set(self.user_rated_items[user_id].index)
        candidate_mask = np.array(
            [item_id not in seen_items for item_id in self._item_ids_arr]
        )
        candidate_indices = np.flatnonzero(candidate_mask)

        rated_idx = self._user_rated_idx[user_id]
        rated_ratings = self._user_rated_ratings[user_id]

        # Dự đoán rating cho từng item chưa xem (dùng công thức tương
        # đương predict(), nhưng tính trực tiếp bằng NumPy để nhanh hơn
        # nhiều lần so với gọi self.predict() theo kiểu pandas).
        scores = np.empty(candidate_indices.size, dtype=np.float64)
        for pos, item_idx in enumerate(candidate_indices):
            predicted = self._predict_by_index(int(item_idx), rated_idx, rated_ratings)
            if predicted is None:
                predicted = self._item_avg_np[item_idx]
            scores[pos] = predicted

        k = min(top_k, scores.size)
        if k == 0:
            return []
        # Sắp xếp ổn định (stable) giảm dần theo score - tương đương hành
        # vi sorted() (ổn định) trên predicted_scores.items() của cách
        # làm gốc: khi có nhiều item cùng điểm dự đoán, thứ tự giữa các
        # item bằng điểm được giữ theo đúng thứ tự candidate_indices ban
        # đầu (tăng dần theo movieId).
        order = np.argsort(-scores, kind="stable")[:k]
        top_item_indices = candidate_indices[order]

        return [self._item_ids_arr[idx].item() for idx in top_item_indices]


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