"""
base.py

Module định nghĩa lớp trừu tượng (abstract base class) `BaseRecommender`,
đóng vai trò là bộ khung (framework) nền tảng cho tất cả các mô hình
trong hệ thống gợi ý (recommender system).

Mọi mô hình cụ thể (ví dụ: Collaborative Filtering, Matrix Factorization,
Content-Based, ...) đều phải kế thừa từ `BaseRecommender` và triển khai
đầy đủ các phương thức trừu tượng: `fit`, `predict`, `recommend`.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseRecommender(ABC):
    """
    Lớp trừu tượng làm nền tảng cho tất cả các mô hình gợi ý.

    Class này định nghĩa giao diện (interface) chung mà mọi mô hình con
    (subclass) bắt buộc phải triển khai, bao gồm việc huấn luyện mô hình,
    dự đoán rating và sinh danh sách gợi ý.

    Attributes:
        config (Dict[str, Any]): Từ điển lưu trữ các tham số cấu hình
            chung được truyền vào lúc khởi tạo mô hình.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Khởi tạo các tham số cấu hình chung cho mô hình gợi ý.

        Args:
            **kwargs (Any): Các tham số cấu hình tùy ý (ví dụ: learning_rate,
                n_factors, n_epochs, random_state, ...). Toàn bộ tham số này
                sẽ được lưu vào thuộc tính `self.config` để các lớp con
                có thể truy cập và sử dụng khi cần.

        Returns:
            None
        """
        self.config: Dict[str, Any] = dict(kwargs)

    @abstractmethod
    def fit(self, train_data: Any) -> None:
        """
        Huấn luyện mô hình gợi ý dựa trên dữ liệu huấn luyện.

        Đây là phương thức trừu tượng, bắt buộc mọi lớp con phải triển khai
        logic huấn luyện cụ thể (ví dụ: học ma trận latent factor, tính toán
        độ tương đồng, huấn luyện mạng neural, ...).

        Args:
            train_data (Any): Dữ liệu huấn luyện, có thể là DataFrame,
                ma trận numpy, hoặc bất kỳ cấu trúc dữ liệu nào phù hợp
                với mô hình cụ thể (ví dụ: tập các tuple (user_id, item_id, rating)).

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Dự đoán điểm rating (mức độ ưa thích) của một user cho một item cụ thể.

        Args:
            user_id (int): ID của người dùng cần dự đoán.
            item_id (int): ID của item cần dự đoán.

        Returns:
            float: Giá trị rating dự đoán (ví dụ: thang điểm 1.0 - 5.0).
        """
        raise NotImplementedError

    @abstractmethod
    def recommend(self, user_id: int, top_k: int = 10) -> List[int]:
        """
        Sinh danh sách gợi ý gồm top-K item phù hợp nhất cho một user.

        Args:
            user_id (int): ID của người dùng cần gợi ý.
            top_k (int, optional): Số lượng item được gợi ý trả về.
                Mặc định là 10.

        Returns:
            List[int]: Danh sách ID của top-K item được gợi ý, sắp xếp
                theo thứ tự mức độ phù hợp giảm dần.
        """
        raise NotImplementedError