"""
matrix_factorization.py

Module triển khai `MatrixFactorizationRecommender` - mô hình gợi ý theo
phương pháp Matrix Factorization (MF), cụ thể là biến thể SVD (Singular
Value Decomposition) được huấn luyện bằng Stochastic Gradient Descent
(SGD), tương tự cách tiếp cận của thư viện Surprise.

Ý tưởng cốt lõi: phân rã ma trận User-Item (thường rất thưa) thành hai
ma trận có số chiều thấp hơn P (user factors) và Q (item factors), sao
cho tích P.Q^T xấp xỉ ma trận rating ban đầu:

    R ≈ P Q^T

Công thức dự đoán (có bổ sung bias):

    r_hat_ui = mu + b_u + b_i + q_i^T p_u

Trong đó:
    - mu   : rating trung bình toàn bộ dữ liệu huấn luyện.
    - b_u  : bias của user u.
    - b_i  : bias của item i.
    - p_u  : vector yếu tố tiềm ẩn (latent factor) của user u.
    - q_i  : vector yếu tố tiềm ẩn (latent factor) của item i.

Hàm mất mát (có Regularization) cần tối ưu:

    L = sum_{(u,i) in K} (r_ui - r_hat_ui)^2
        + lambda * (b_u^2 + b_i^2 + ||p_u||^2 + ||q_i||^2)

Các tham số p_u, q_i, b_u, b_i được cập nhật bằng SGD theo từng rating
quan sát được trong tập huấn luyện, lặp qua nhiều epoch cho đến khi hội
tụ (hoặc đạt số epoch tối đa).
"""

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from base import BaseRecommender


class MatrixFactorizationRecommender(BaseRecommender):
    """
    Mô hình gợi ý Matrix Factorization (SVD, huấn luyện bằng SGD).

    Attributes:
        n_factors (int): Số lượng yếu tố tiềm ẩn (latent factors) k dùng để
            biểu diễn user và item.
        n_epochs (int): Số epoch huấn luyện (số lần lặp qua toàn bộ tập
            rating huấn luyện).
        learning_rate (float): Tốc độ học (learning rate) dùng khi cập nhật
            tham số bằng SGD.
        regularization (float): Hệ số Regularization (lambda), giúp kiểm
            soát độ lớn tham số và hạn chế overfitting.
        random_state (Optional[int]): Seed cho bộ sinh số ngẫu nhiên, đảm
            bảo kết quả khởi tạo và huấn luyện có thể tái lập.
        global_mean (float): Rating trung bình toàn bộ dữ liệu huấn luyện
            (mu), dùng làm thành phần bias chung trong công thức dự đoán.
        user_factors (Optional[np.ndarray]): Ma trận P, kích thước
            (số user, n_factors), mỗi hàng là vector p_u của một user.
        item_factors (Optional[np.ndarray]): Ma trận Q, kích thước
            (số item, n_factors), mỗi hàng là vector q_i của một item.
        user_biases (Optional[np.ndarray]): Mảng bias b_u của từng user.
        item_biases (Optional[np.ndarray]): Mảng bias b_i của từng item.
        user_id_to_index (Dict[int, int]): Ánh xạ từ userId gốc sang chỉ
            số hàng trong user_factors/user_biases.
        item_id_to_index (Dict[int, int]): Ánh xạ từ movieId gốc sang chỉ
            số hàng trong item_factors/item_biases.
        user_seen_items (Dict[int, Set[int]]): Ánh xạ userId -> tập hợp
            movieId mà user đó đã đánh giá, dùng để lọc khi gợi ý.
        train_rmse_history (List[float]): RMSE trên tập huấn luyện sau
            mỗi epoch, phục vụ theo dõi quá trình hội tụ.
        min_rating (float): Giá trị rating nhỏ nhất quan sát được trong
            tập huấn luyện, dùng để giới hạn (clip) giá trị dự đoán.
        max_rating (float): Giá trị rating lớn nhất quan sát được trong
            tập huấn luyện, dùng để giới hạn (clip) giá trị dự đoán.
    """

    def __init__(
        self,
        n_factors: int = 20,
        n_epochs: int = 20,
        learning_rate: float = 0.005,
        regularization: float = 0.02,
        random_state: Optional[int] = 42,
    ) -> None:
        """
        Khởi tạo mô hình Matrix Factorization.

        Args:
            n_factors (int, optional): Số lượng yếu tố tiềm ẩn (k). Mặc
                định là 20.
            n_epochs (int, optional): Số epoch huấn luyện. Mặc định là 20.
            learning_rate (float, optional): Tốc độ học cho SGD. Mặc định
                là 0.005.
            regularization (float, optional): Hệ số Regularization (lambda).
                Mặc định là 0.02.
            random_state (Optional[int], optional): Seed cho bộ sinh số
                ngẫu nhiên, đảm bảo khả năng tái lập. Mặc định là 42.

        Returns:
            None
        """
        super().__init__(
            n_factors=n_factors,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            regularization=regularization,
            random_state=random_state,
        )
        self.n_factors: int = n_factors
        self.n_epochs: int = n_epochs
        self.learning_rate: float = learning_rate
        self.regularization: float = regularization
        self.random_state: Optional[int] = random_state

        self.global_mean: float = 0.0
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_biases: Optional[np.ndarray] = None
        self.item_biases: Optional[np.ndarray] = None

        self.user_id_to_index: Dict[int, int] = {}
        self.item_id_to_index: Dict[int, int] = {}
        self.item_ids: List[int] = []

        self.user_seen_items: Dict[int, Set[int]] = {}
        self.train_rmse_history: List[float] = []

        self.min_rating: float = 0.5
        self.max_rating: float = 5.0

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Huấn luyện mô hình Matrix Factorization bằng Stochastic Gradient
        Descent (SGD) trên dữ liệu rating.

        Các bước thực hiện:
            1. Xây dựng ánh xạ userId/movieId sang chỉ số nội bộ (index)
               để có thể lưu trữ các vector yếu tố tiềm ẩn trong ma trận
               numpy.
            2. Tính rating trung bình toàn bộ dữ liệu (global_mean = mu).
            3. Khởi tạo ngẫu nhiên (phân phối chuẩn, độ lệch chuẩn nhỏ)
               các ma trận P, Q và khởi tạo bias b_u, b_i bằng 0.
            4. Với mỗi epoch, xáo trộn (shuffle) thứ tự các rating và với
               từng rating (u, i, r) thực hiện:
                   - Tính rating dự đoán: r_hat = mu + b_u + b_i + q_i^T p_u
                   - Tính sai số: e = r - r_hat
                   - Cập nhật tham số theo hướng giảm gradient của hàm mất
                     mát có Regularization:
                         b_u <- b_u + lr * (e - lambda * b_u)
                         b_i <- b_i + lr * (e - lambda * b_i)
                         p_u <- p_u + lr * (e * q_i - lambda * p_u)
                         q_i <- q_i + lr * (e * p_u - lambda * q_i)
            5. Sau mỗi epoch, tính RMSE trên tập huấn luyện để theo dõi
               quá trình hội tụ (lưu vào `train_rmse_history`).
            6. Lưu danh sách item mỗi user đã đánh giá, phục vụ việc lọc
               khi sinh danh sách gợi ý.

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

        rng = np.random.default_rng(self.random_state)

        # 1. Ánh xạ userId / movieId gốc sang chỉ số nội bộ liên tục 0..n-1
        unique_user_ids = train_data["userId"].unique()
        unique_item_ids = train_data["movieId"].unique()

        self.user_id_to_index = {
            user_id: idx for idx, user_id in enumerate(unique_user_ids)
        }
        self.item_id_to_index = {
            item_id: idx for idx, item_id in enumerate(unique_item_ids)
        }
        self.item_ids = list(unique_item_ids)

        n_users = len(unique_user_ids)
        n_items = len(unique_item_ids)

        # 2. Rating trung bình toàn bộ dữ liệu huấn luyện (mu)
        self.global_mean = float(train_data["rating"].mean())
        self.min_rating = float(train_data["rating"].min())
        self.max_rating = float(train_data["rating"].max())

        # 3. Khởi tạo ngẫu nhiên P, Q (phân phối chuẩn, std nhỏ) và bias = 0
        self.user_factors = rng.normal(
            loc=0.0, scale=0.1, size=(n_users, self.n_factors)
        )
        self.item_factors = rng.normal(
            loc=0.0, scale=0.1, size=(n_items, self.n_factors)
        )
        self.user_biases = np.zeros(n_users, dtype=np.float64)
        self.item_biases = np.zeros(n_items, dtype=np.float64)

        # Chuẩn bị mảng chỉ số user/item và rating để lặp SGD
        user_indices = train_data["userId"].map(self.user_id_to_index).to_numpy()
        item_indices = train_data["movieId"].map(self.item_id_to_index).to_numpy()
        ratings = train_data["rating"].to_numpy(dtype=np.float64)

        n_samples = len(ratings)
        sample_order = np.arange(n_samples)

        self.train_rmse_history = []

        # 4-5. Vòng lặp huấn luyện SGD qua nhiều epoch
        for epoch in range(self.n_epochs):
            rng.shuffle(sample_order)
            squared_error_sum = 0.0

            for sample_idx in sample_order:
                u = user_indices[sample_idx]
                i = item_indices[sample_idx]
                r_ui = ratings[sample_idx]

                pred = (
                    self.global_mean
                    + self.user_biases[u]
                    + self.item_biases[i]
                    + np.dot(self.item_factors[i], self.user_factors[u])
                )
                error = r_ui - pred
                squared_error_sum += error ** 2

                # Lưu lại vector cũ của p_u, q_i trước khi cập nhật, để
                # dùng chung cho cả hai công thức cập nhật (tránh dùng giá
                # trị đã bị thay đổi giữa chừng)
                p_u_old = self.user_factors[u].copy()
                q_i_old = self.item_factors[i].copy()

                self.user_biases[u] += self.learning_rate * (
                    error - self.regularization * self.user_biases[u]
                )
                self.item_biases[i] += self.learning_rate * (
                    error - self.regularization * self.item_biases[i]
                )
                self.user_factors[u] += self.learning_rate * (
                    error * q_i_old - self.regularization * p_u_old
                )
                self.item_factors[i] += self.learning_rate * (
                    error * p_u_old - self.regularization * q_i_old
                )

            epoch_rmse = float(np.sqrt(squared_error_sum / n_samples))
            self.train_rmse_history.append(epoch_rmse)

        # 6. Lưu danh sách item mỗi user đã đánh giá (để lọc khi gợi ý)
        self.user_seen_items = (
            train_data.groupby("userId")["movieId"].apply(set).to_dict()
        )

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Dự đoán rating của `user_id` cho `item_id`.

        Công thức: r_hat_ui = mu + b_u + b_i + q_i^T p_u

        Nếu user hoặc item chưa từng xuất hiện trong tập huấn luyện (cold
        start), thành phần bias/latent factor tương ứng được coi là 0,
        khi đó dự đoán suy biến về mu (+ bias của phía còn lại nếu có).
        Giá trị dự đoán cuối cùng được giới hạn (clip) trong khoảng
        [min_rating, max_rating] quan sát được từ tập huấn luyện.

        Args:
            user_id (int): ID người dùng cần dự đoán.
            item_id (int): ID item cần dự đoán.

        Returns:
            float: Rating dự đoán, đã được giới hạn trong khoảng rating
                hợp lệ của tập huấn luyện.

        Raises:
            RuntimeError: Nếu mô hình chưa được huấn luyện (chưa gọi fit()).
        """
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")

        user_idx = self.user_id_to_index.get(user_id)
        item_idx = self.item_id_to_index.get(item_id)

        pred = self.global_mean
        if user_idx is not None:
            pred += self.user_biases[user_idx]
        if item_idx is not None:
            pred += self.item_biases[item_idx]
        if user_idx is not None and item_idx is not None:
            pred += float(
                np.dot(self.item_factors[item_idx], self.user_factors[user_idx])
            )

        # Giới hạn dự đoán trong khoảng rating hợp lệ
        pred = max(self.min_rating, min(self.max_rating, pred))

        return float(pred)

    def recommend(self, user_id: int, top_k: int = 10) -> List[int]:
        """
        Sinh danh sách top-K item được gợi ý cho một user.

        Với user đã tồn tại trong tập huấn luyện, mô hình tính điểm dự
        đoán cho toàn bộ item chưa được đánh giá bằng phép nhân ma trận
        (vector hóa để tăng tốc độ), sau đó sắp xếp giảm dần và trả về
        top_k item có điểm cao nhất.

        Với user mới (cold-start, chưa từng xuất hiện trong tập huấn
        luyện), mô hình không có vector yếu tố tiềm ẩn cho user này nên
        trả về top_k item có bias (độ phổ biến/chất lượng trung bình)
        cao nhất như một fallback.

        Args:
            user_id (int): ID user cần gợi ý.
            top_k (int, optional): Số lượng item được gợi ý. Mặc định là 10.

        Returns:
            List[int]: Danh sách movieId của top_k item được gợi ý, sắp
                xếp theo điểm dự đoán giảm dần.

        Raises:
            RuntimeError: Nếu mô hình chưa được huấn luyện (chưa gọi fit()).
        """
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")

        if top_k <= 0:
            return []

        seen_items = self.user_seen_items.get(user_id, set())
        user_idx = self.user_id_to_index.get(user_id)

        candidate_ids = [
            item_id for item_id in self.item_ids if item_id not in seen_items
        ]
        candidate_indices = np.array(
            [self.item_id_to_index[i] for i in candidate_ids]
        )

        if user_idx is None:
            # Cold-start: fallback theo bias item (tương tự độ phổ biến)
            scores = self.global_mean + self.item_biases[candidate_indices]
        else:
            # Vector hóa: tính r_hat cho toàn bộ item ứng viên cùng lúc
            scores = (
                self.global_mean
                + self.user_biases[user_idx]
                + self.item_biases[candidate_indices]
                + self.item_factors[candidate_indices] @ self.user_factors[user_idx]
            )

        ranked_order = np.argsort(-scores)[:top_k]
        return [candidate_ids[idx] for idx in ranked_order]


if __name__ == "__main__":
    # Ví dụ nhanh để kiểm tra mô hình
    data = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5],
            "movieId": [10, 20, 30, 10, 20, 10, 30, 40, 20, 40, 30, 40],
            "rating": [5, 4, 3, 5, 5, 2, 4, 5, 3, 4, 4, 5],
        }
    )

    model = MatrixFactorizationRecommender(
        n_factors=4, n_epochs=50, learning_rate=0.05, regularization=0.02
    )
    model.fit(data)

    print("Train RMSE qua các epoch cuối:", model.train_rmse_history[-5:])
    print("Predict (user=1, item=40):", model.predict(1, 40))
    print("Predict (user=99 - user mới, item=10):", model.predict(99, 10))
    print("Predict (user=1, item=999 - item mới):", model.predict(1, 999))
    print("Recommend (user=1, top_k=3):", model.recommend(1, top_k=3))
    print("Recommend (user=99 - user mới, top_k=3):", model.recommend(99, top_k=3))