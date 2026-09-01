"""
metrics.py

Module chứa các hàm đánh giá (evaluation metrics) cho hệ thống gợi ý
(recommender system), bao gồm:
    - RMSE (Root Mean Square Error)
    - Precision@K
    - Recall@K
"""

from typing import List, Sequence, Union
import numpy as np

Number = Union[int, float]


def calculate_rmse(y_true: Union[List[Number], np.ndarray],
                    y_pred: Union[List[Number], np.ndarray]) -> float:
    """
    Tính Root Mean Square Error (RMSE) giữa rating thực tế và rating dự đoán.

    Công thức:
        RMSE = sqrt( (1/n) * sum( (y_true_i - y_pred_i)^2 ) )

    Args:
        y_true (Union[List[Number], np.ndarray]): Mảng/list các giá trị rating thực tế.
        y_pred (Union[List[Number], np.ndarray]): Mảng/list các giá trị rating dự đoán.
            Phải có cùng độ dài với y_true.

    Returns:
        float: Giá trị RMSE. Giá trị càng nhỏ, mô hình dự đoán càng chính xác.

    Raises:
        ValueError: Nếu y_true và y_pred không cùng kích thước.

    Example:
        >>> calculate_rmse([3, 4, 5], [2.5, 4.2, 4.8])
        0.3162...
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true và y_pred phải có cùng kích thước, "
            f"nhận được {y_true_arr.shape} và {y_pred_arr.shape}."
        )

    squared_errors = (y_true_arr - y_pred_arr) ** 2
    rmse = np.sqrt(np.mean(squared_errors))
    return float(rmse)


def precision_at_k(actual_items: Sequence,
                    recommended_items: Sequence,
                    k: int) -> float:
    """
    Tính Precision@K: tỷ lệ item liên quan (relevant) trong top-K item được gợi ý.

    Công thức:
        Precision@K = (Số item giao nhau giữa actual_items và top-K recommended_items) / K

    Args:
        actual_items (Sequence): Danh sách/tập hợp các item thực sự liên quan
            (ví dụ: item người dùng đã thích/mua/tương tác).
        recommended_items (Sequence): Danh sách các item được hệ thống gợi ý,
            đã được sắp xếp theo thứ tự ưu tiên giảm dần.
        k (int): Số lượng item đầu tiên trong recommended_items được xét đến.

    Returns:
        float: Giá trị Precision@K, nằm trong khoảng [0.0, 1.0].
            Trả về 0.0 nếu k <= 0.

    Example:
        >>> precision_at_k({1, 2, 3}, [1, 4, 2, 5, 6], k=3)
        0.6666666666666666
    """
    if k <= 0:
        return 0.0

    top_k_items = recommended_items[:k]
    actual_set = set(actual_items)

    relevant_count = len(set(top_k_items) & actual_set)

    return relevant_count / k


def recall_at_k(actual_items: Sequence,
                 recommended_items: Sequence,
                 k: int) -> float:
    """
    Tính Recall@K: tỷ lệ item liên quan được tìm thấy trong top-K item gợi ý
    so với tổng số item liên quan thực tế.

    Công thức:
        Recall@K = (Số item giao nhau giữa actual_items và top-K recommended_items)
                   / (Tổng số lượng actual_items)

    Args:
        actual_items (Sequence): Danh sách/tập hợp các item thực sự liên quan.
        recommended_items (Sequence): Danh sách các item được hệ thống gợi ý,
            đã được sắp xếp theo thứ tự ưu tiên giảm dần.
        k (int): Số lượng item đầu tiên trong recommended_items được xét đến.

    Returns:
        float: Giá trị Recall@K, nằm trong khoảng [0.0, 1.0].
            Trả về 0.0 nếu actual_items rỗng (tránh lỗi chia cho 0) hoặc k <= 0.

    Example:
        >>> recall_at_k({1, 2, 3}, [1, 4, 2, 5, 6], k=3)
        0.6666666666666666
    """
    if not actual_items or k <= 0:
        return 0.0

    top_k_items = recommended_items[:k]
    actual_set = set(actual_items)

    relevant_count = len(set(top_k_items) & actual_set)

    return relevant_count / len(actual_set)


if __name__ == "__main__":
    # Ví dụ nhanh để kiểm tra các hàm
    y_true = [3, 4, 5, 2]
    y_pred = [2.5, 4.2, 4.8, 3.0]
    print("RMSE:", calculate_rmse(y_true, y_pred))

    actual = {1, 2, 3, 4}
    recommended = [5, 1, 2, 6, 3, 7]
    k = 5
    print(f"Precision@{k}:", precision_at_k(actual, recommended, k))
    print(f"Recall@{k}:", recall_at_k(actual, recommended, k))