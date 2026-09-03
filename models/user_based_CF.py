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

        # --- Cache dạng NumPy (hiệu năng) ---
        # Các cấu trúc dữ liệu bên dưới lưu lại đúng nội dung của
        # user_item_matrix / user_similarity_matrix / user_means dưới dạng
        # mảng NumPy thuần (thay vì pandas.DataFrame/Series), giúp predict()
        # và đặc biệt là recommend() (vốn gọi predict() cho hàng nghìn item
        # mỗi user) tránh được chi phí overhead rất lớn của việc tra cứu
        # (indexing) trên pandas trong vòng lặp Python. Kết quả tính toán
        # là TƯƠNG ĐƯƠNG về mặt toán học với cách làm bằng pandas ở trên,
        # chỉ khác cách hiện thực để chạy nhanh hơn.
        self._user_id_to_idx: Dict[int, int] = {}
        self._item_id_to_idx: Dict[int, int] = {}
        self._item_ids_arr: np.ndarray = np.array([])
        self._ratings_filled: np.ndarray = np.array([])  # NaN -> 0.0
        self._rated_mask: np.ndarray = np.array([])       # True nếu đã rating
        self._user_means_np: np.ndarray = np.array([])
        self._sim_np: np.ndarray = np.array([])
        self._item_avg_np: np.ndarray = np.array([])

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

        # 6. Xây dựng cache NumPy để tăng tốc predict()/recommend() (xem
        # giải thích ở phần khai báo thuộc tính __init__). Dữ liệu ở đây
        # được suy ra trực tiếp từ các cấu trúc pandas đã tính ở trên,
        # không làm thay đổi kết quả, chỉ giúp tính nhanh hơn.
        self._user_id_to_idx = {
            uid: idx for idx, uid in enumerate(self.user_item_matrix.index)
        }
        self._item_id_to_idx = {
            iid: idx for idx, iid in enumerate(self.user_item_matrix.columns)
        }
        self._item_ids_arr = self.user_item_matrix.columns.to_numpy()
        raw_values = self.user_item_matrix.to_numpy(dtype=np.float64)
        self._rated_mask = ~np.isnan(raw_values)
        self._ratings_filled = np.nan_to_num(raw_values, nan=0.0)
        self._user_means_np = self.user_means.to_numpy(dtype=np.float64)
        self._sim_np = self.user_similarity_matrix.to_numpy(dtype=np.float64)
        self._item_avg_np = self.item_avg_ratings.to_numpy(dtype=np.float64)

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

    def _predict_by_index(self, user_idx: int, item_idx: int) -> Optional[float]:
        """
        Phiên bản NumPy thuần của công thức dự đoán k-NN weighted average,
        làm việc trực tiếp trên chỉ số (index) nội bộ thay vì userId/movieId
        gốc. Đây là phần lõi tính toán được cả `predict()` và `recommend()`
        dùng chung, cài đặt ĐÚNG công thức toán học như mô tả trong
        `predict()` nhưng tránh chi phí tra cứu (indexing) rất lớn của
        pandas khi phải gọi lặp lại hàng nghìn lần (trong `recommend()`).

        Args:
            user_idx (int): Chỉ số nội bộ của user (theo self._user_id_to_idx).
            item_idx (int): Chỉ số nội bộ của item (theo self._item_id_to_idx).

        Returns:
            Optional[float]: Rating dự đoán, hoặc None nếu không có láng
                giềng hợp lệ (cần dùng giá trị fallback).
        """
        mask = self._rated_mask[:, item_idx]
        # Loại bỏ chính user (không tự làm láng giềng của chính mình)
        if mask[user_idx]:
            mask = mask.copy()
            mask[user_idx] = False

        rater_indices = np.flatnonzero(mask)
        if rater_indices.size == 0:
            return None

        sims = self._sim_np[user_idx, rater_indices]

        if rater_indices.size > self.k_neighbors:
            top_pos = np.argpartition(-sims, self.k_neighbors)[: self.k_neighbors]
            sims = sims[top_pos]
            rater_indices = rater_indices[top_pos]

        denominator = np.sum(np.abs(sims))
        if denominator == 0.0:
            return None

        neighbor_ratings = self._ratings_filled[rater_indices, item_idx]
        neighbor_means = self._user_means_np[rater_indices]
        numerator = np.sum(sims * (neighbor_ratings - neighbor_means))

        return float(self._user_means_np[user_idx] + numerator / denominator)

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Dự đoán rating của `user_id` cho `item_id` bằng k-NN weighted average.

        Công thức:
            pred = mean(user) + [ sum(sim_i * (rating_i - mean_i)) ]
                                  / [ sum(|sim_i|) ]
            với i chạy trên k user láng giềng gần nhất (theo cosine similarity)
            đã từng đánh giá item_id.

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
                hiện trong tập huấn luyện, hoặc không tìm được láng giềng
                hợp lệ (tổng trọng số bằng 0), trả về giá trị fallback
                (rating trung bình của item, hoặc trung bình toàn hệ thống).

        Raises:
            RuntimeError: Nếu mô hình chưa được huấn luyện (chưa gọi fit()).
        """
        if self.user_item_matrix is None or self.user_similarity_matrix is None:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")

        user_idx = self._user_id_to_idx.get(user_id)
        item_idx = self._item_id_to_idx.get(item_id)

        # User hoặc item chưa từng xuất hiện trong tập huấn luyện -> fallback
        if user_idx is None or item_idx is None:
            return self._fallback_rating(item_id)

        predicted_rating = self._predict_by_index(user_idx, item_idx)
        if predicted_rating is None:
            return self._fallback_rating(item_id)

        return predicted_rating

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

        seen_items = self.user_seen_items.get(user_id, set())
        user_idx = self._user_id_to_idx.get(user_id)

        # Cold-start: user chưa từng xuất hiện -> fallback theo rating trung bình
        if user_idx is None:
            candidate_items = [
                item for item in self.user_item_matrix.columns if item not in seen_items
            ]
            fallback_ranking = self.item_avg_ratings.loc[candidate_items].sort_values(
                ascending=False
            )
            return fallback_ranking.head(top_k).index.tolist()

        # Chỉ số các item ứng viên (chưa được user xem) - làm việc trên
        # index NumPy thay vì list/dict pandas để tránh overhead khi lặp
        # qua hàng nghìn item (xem _predict_by_index()).
        candidate_mask = np.array(
            [item_id not in seen_items for item_id in self._item_ids_arr]
        )
        candidate_indices = np.flatnonzero(candidate_mask)

        # Dự đoán rating cho từng item chưa xem (dùng công thức tương
        # đương predict(), nhưng tính trực tiếp bằng NumPy để nhanh hơn
        # nhiều lần so với gọi self.predict() theo kiểu pandas).
        scores = np.empty(candidate_indices.size, dtype=np.float64)
        for pos, item_idx in enumerate(candidate_indices):
            predicted = self._predict_by_index(user_idx, int(item_idx))
            if predicted is None:
                predicted = self._item_avg_np[item_idx]
            scores[pos] = predicted

        k = min(top_k, scores.size)
        if k == 0:
            return []
        # Sắp xếp ổn định (stable) giảm dần theo score - tương đương hành vi
        # sorted() (ổn định) trên predicted_scores.items() của cách làm
        # gốc: khi có nhiều item cùng điểm dự đoán (ví dụ đều rơi vào
        # fallback), thứ tự giữa các item bằng điểm được giữ theo đúng thứ
        # tự candidate_indices ban đầu (tăng dần theo movieId).
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

    model = UserBasedRecommender(k_neighbors=2)
    model.fit(data)

    print("Predict (user=1, item=40):", model.predict(1, 40))
    print("Predict (user=99 - user mới, item=10):", model.predict(99, 10))
    print("Predict (user=1, item=999 - item mới):", model.predict(1, 999))
    print("Recommend (user=1, top_k=3):", model.recommend(1, top_k=3))
    print("Recommend (user=99 - user mới, top_k=3):", model.recommend(99, top_k=3))