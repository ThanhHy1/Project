"""
evaluate.py

Script huấn luyện và đánh giá, so sánh các mô hình gợi ý phim đã triển
khai trong đề tài:
    - Popularity Baseline                  (models/popularity_baseline.py)
    - User-based Collaborative Filtering   (models/user_based_CF.py)
    - Item-based Collaborative Filtering   (models/item_based_CF.py)
    - Matrix Factorization (SVD, SGD)      (models/matrix_factorization.py)

Phương pháp đánh giá tuân theo đúng nội dung trình bày trong file lý
thuyết (Cơ sở lý thuyết - Chương 2 và Chương 3):

    1. Dữ liệu rating (data/ratings.csv) được chia thành ba tập Train /
       Validation / Test theo tỷ lệ 80% - 10% - 10%, sử dụng
       train_test_split hai bước với random_state = 42 để đảm bảo có
       thể tái lập (mục 3.5.2). Tập Train dùng để huấn luyện mô hình,
       tập Test dùng để đánh giá kết quả cuối cùng (mục 3.5.3).

    2. Mỗi mô hình được đánh giá bằng ba chỉ số (mục 2.5):
        - RMSE: đo sai lệch giữa rating dự đoán và rating thực tế, tính
          trên TOÀN BỘ tập Test (mục 2.5.1).
        - Precision@K: tỷ lệ item "phù hợp" (relevant) trong Top-K item
          được gợi ý cho mỗi user (mục 2.5.2).
        - Recall@K: tỷ lệ item "phù hợp" của user được tìm thấy trong
          Top-K item được gợi ý (mục 2.5.3).

       Một item trong tập Test được coi là "phù hợp" (relevant) với một
       user nếu rating thực tế của user đó cho item >= RELEVANCE_THRESHOLD
       (mặc định 4.0, theo thang rating 0.5 - 5.0 của MovieLens).

    3. Ghi chú về hiệu năng: hàm recommend() của User-based CF và
       Item-based CF (đã triển khai trong models/) duyệt qua toàn bộ
       item chưa được đánh giá và gọi predict() cho từng item để xếp
       hạng Top-K, nên chi phí tính cho MỖI user là khá lớn (~9.000 item
       x thời gian predict). Vì vậy, Precision@K/Recall@K được tính trên
       một MẪU NGẪU NHIÊN gồm RANKING_SAMPLE_SIZE user trong tập Test
       (mặc định 30 user, random_state cố định để tái lập) thay vì toàn
       bộ ~600 user - đây là cách làm phổ biến trong đánh giá hệ thống
       gợi ý khi việc xếp hạng toàn bộ catalog cho mọi user không khả
       thi về mặt thời gian chạy. RMSE vẫn được tính trên toàn bộ tập
       Test vì predict() cho một cặp (user, item) có chi phí thấp.
       Có thể tăng RANKING_SAMPLE_SIZE (hoặc đặt None để dùng toàn bộ
       user) nếu chấp nhận thời gian chạy lâu hơn.

    4. Kết quả của bốn mô hình được tổng hợp thành một bảng so sánh và
       lưu ra file CSV, phục vụ trình bày trong Chương 4 của báo cáo.

Cách chạy:
    python evaluate.py
"""

import os
import sys
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------- #
# Thiết lập đường dẫn để có thể import các module trong models/ và
# utils/. Các file trong models/ dùng import dạng "from base import
# BaseRecommender" (import tương đối theo kiểu script), nên cần thêm
# thư mục models/ và utils/ vào sys.path trước khi import.
# --------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

for _path in (MODELS_DIR, UTILS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from base import BaseRecommender  # noqa: E402
from popularity_baseline import PopularityRecommender  # noqa: E402
from user_based_CF import UserBasedRecommender  # noqa: E402
from item_based_CF import ItemBasedRecommender  # noqa: E402
from matrix_factorization import MatrixFactorizationRecommender  # noqa: E402
from metrics import calculate_rmse, precision_at_k, recall_at_k  # noqa: E402


# --------------------------------------------------------------------- #
# Cấu hình thực nghiệm
# --------------------------------------------------------------------- #
RANDOM_STATE = 42
RELEVANCE_THRESHOLD = 4.0          # rating >= ngưỡng này được coi là "phù hợp"
TOP_K_LIST = [5, 10]               # các giá trị K dùng để tính Precision@K, Recall@K
RANKING_SAMPLE_SIZE = 30           # số user lấy mẫu để tính Precision@K/Recall@K
                                    # (đặt None để đánh giá trên toàn bộ user của tập Test)


def load_ratings(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Đọc dữ liệu rating từ data/ratings.csv.

    Args:
        data_dir (str): Đường dẫn thư mục chứa dữ liệu.

    Returns:
        pd.DataFrame: DataFrame gồm các cột 'userId', 'movieId', 'rating',
            'timestamp'.
    """
    ratings_path = os.path.join(data_dir, "ratings.csv")
    return pd.read_csv(ratings_path)


def split_data(
    ratings: pd.DataFrame, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chia dữ liệu rating thành Train / Validation / Test theo tỷ lệ
    80% - 10% - 10%, đúng phương pháp mô tả tại mục 3.5.2 của file lý
    thuyết (train_test_split hai bước, random_state cố định để tái lập).

    Args:
        ratings (pd.DataFrame): Toàn bộ dữ liệu rating.
        random_state (int, optional): Seed cho việc chia dữ liệu.
            Mặc định là 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_data,
            val_data, test_data).
    """
    train_data, temp_data = train_test_split(
        ratings, test_size=0.2, random_state=random_state
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=random_state
    )
    return train_data, val_data, test_data


def build_relevant_items(
    test_data: pd.DataFrame, threshold: float = RELEVANCE_THRESHOLD
) -> Dict[int, Set[int]]:
    """
    Xây dựng tập item "phù hợp" (relevant) của mỗi user trong tập Test,
    dùng làm ground-truth để tính Precision@K / Recall@K (mục 2.5.2,
    2.5.3).

    Args:
        test_data (pd.DataFrame): Tập dữ liệu Test.
        threshold (float, optional): Ngưỡng rating để coi một item là
            "phù hợp". Mặc định là 4.0.

    Returns:
        Dict[int, Set[int]]: Ánh xạ userId -> tập hợp movieId "phù hợp"
            trong tập Test.
    """
    relevant = (
        test_data[test_data["rating"] >= threshold]
        .groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )
    return relevant


def sample_users_for_ranking(
    relevant_items: Dict[int, Set[int]],
    sample_size: "int | None" = RANKING_SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> Dict[int, Set[int]]:
    """
    Lấy mẫu ngẫu nhiên (cố định seed) một tập con user từ `relevant_items`
    để tính Precision@K/Recall@K, giúp việc đánh giá khả thi về mặt thời
    gian chạy đối với các mô hình có recommend() không được vector hóa
    (User-based CF, Item-based CF).

    Args:
        relevant_items (Dict[int, Set[int]]): Toàn bộ ground-truth
            userId -> tập item phù hợp trong tập Test.
        sample_size (int | None, optional): Số lượng user cần lấy mẫu.
            Nếu None hoặc lớn hơn số user hiện có, trả về toàn bộ
            `relevant_items`. Mặc định là RANKING_SAMPLE_SIZE.
        random_state (int, optional): Seed đảm bảo kết quả lấy mẫu có
            thể tái lập.

    Returns:
        Dict[int, Set[int]]: Tập con userId -> tập item phù hợp, dùng để
            đánh giá ranking.
    """
    if sample_size is None or sample_size >= len(relevant_items):
        return relevant_items

    rng = np.random.default_rng(random_state)
    all_user_ids = np.array(list(relevant_items.keys()))
    sampled_ids = rng.choice(all_user_ids, size=sample_size, replace=False)
    return {user_id: relevant_items[user_id] for user_id in sampled_ids}


def evaluate_rmse(model: BaseRecommender, test_data: pd.DataFrame) -> float:
    """
    Tính RMSE của mô hình trên toàn bộ tập Test (mục 2.5.1).

    Args:
        model (BaseRecommender): Mô hình đã được huấn luyện (đã gọi fit()).
        test_data (pd.DataFrame): Tập dữ liệu Test, gồm 'userId', 'movieId',
            'rating'.

    Returns:
        float: Giá trị RMSE trên tập Test.
    """
    y_true = test_data["rating"].to_numpy()
    y_pred = np.array(
        [
            model.predict(int(user_id), int(item_id))
            for user_id, item_id in zip(test_data["userId"], test_data["movieId"])
        ]
    )
    return calculate_rmse(y_true, y_pred)


def evaluate_ranking(
    model: BaseRecommender,
    relevant_items_sample: Dict[int, Set[int]],
    k_list: List[int] = TOP_K_LIST,
) -> Dict[int, Dict[str, float]]:
    """
    Tính Precision@K và Recall@K trung bình trên tập user đã lấy mẫu
    (mục 2.5.2, 2.5.3).

    Với mỗi user, hệ thống sinh danh sách Top-K gợi ý bằng model.recommend()
    (K = max(k_list) để tái sử dụng cho mọi giá trị K nhỏ hơn), sau đó so
    sánh với tập item "phù hợp" của user đó trong tập Test.

    Args:
        model (BaseRecommender): Mô hình đã được huấn luyện.
        relevant_items_sample (Dict[int, Set[int]]): Ánh xạ userId -> tập
            item "phù hợp" trong tập Test (mẫu từ `sample_users_for_ranking`).
        k_list (List[int], optional): Danh sách các giá trị K cần tính.
            Mặc định là TOP_K_LIST ([5, 10]).

    Returns:
        Dict[int, Dict[str, float]]: Với mỗi K trong k_list, trả về
            {"precision": giá trị Precision@K trung bình,
             "recall": giá trị Recall@K trung bình}.
    """
    max_k = max(k_list)
    per_k_scores: Dict[int, Dict[str, List[float]]] = {
        k: {"precision": [], "recall": []} for k in k_list
    }

    for user_id, actual_items in relevant_items_sample.items():
        try:
            recommended_items = model.recommend(user_id, top_k=max_k)
        except RuntimeError:
            # Mô hình chưa được huấn luyện - bỏ qua (không nên xảy ra)
            continue

        for k in k_list:
            per_k_scores[k]["precision"].append(
                precision_at_k(actual_items, recommended_items, k)
            )
            per_k_scores[k]["recall"].append(
                recall_at_k(actual_items, recommended_items, k)
            )

    summary: Dict[int, Dict[str, float]] = {}
    for k in k_list:
        precisions = per_k_scores[k]["precision"]
        recalls = per_k_scores[k]["recall"]
        summary[k] = {
            "precision": float(np.mean(precisions)) if precisions else 0.0,
            "recall": float(np.mean(recalls)) if recalls else 0.0,
        }
    return summary


def evaluate_model(
    name: str,
    model: BaseRecommender,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    relevant_items_sample: Dict[int, Set[int]],
    k_list: List[int] = TOP_K_LIST,
) -> Dict[str, float]:
    """
    Huấn luyện một mô hình trên tập Train và đánh giá trên tập Test bằng
    cả ba chỉ số RMSE, Precision@K, Recall@K.

    Args:
        name (str): Tên mô hình (dùng để hiển thị trong bảng kết quả).
        model (BaseRecommender): Instance của mô hình cần đánh giá (chưa
            được huấn luyện).
        train_data (pd.DataFrame): Tập dữ liệu Train.
        test_data (pd.DataFrame): Tập dữ liệu Test (dùng cho RMSE, tính
            trên toàn bộ).
        relevant_items_sample (Dict[int, Set[int]]): Ground-truth item
            phù hợp của mẫu user dùng để tính Precision@K/Recall@K.
        k_list (List[int], optional): Danh sách K cần đánh giá.

    Returns:
        Dict[str, float]: Từ điển kết quả gồm 'model', 'rmse',
            'train_time_sec' và 'precision@K'/'recall@K' cho mỗi K.
    """
    print(f"\n>>> Đang huấn luyện mô hình: {name} ...")
    sys.stdout.flush()
    start_time = time.time()
    model.fit(train_data)
    train_time = time.time() - start_time
    print(f"    Huấn luyện xong sau {train_time:.2f} giây.")

    print("    Đang tính RMSE trên toàn bộ tập Test ...")
    sys.stdout.flush()
    rmse_start = time.time()
    rmse = evaluate_rmse(model, test_data)
    print(f"    RMSE = {rmse:.4f} (tính trong {time.time() - rmse_start:.2f}s)")

    print(
        f"    Đang tính Precision@K / Recall@K trên mẫu "
        f"{len(relevant_items_sample)} user "
    )
    sys.stdout.flush()
    rank_start = time.time()
    ranking_scores = evaluate_ranking(model, relevant_items_sample, k_list)
    print(f"    Hoàn tất trong {time.time() - rank_start:.2f}s")

    result: Dict[str, float] = {
        "model": name,
        "rmse": rmse,
        "train_time_sec": train_time,
    }
    for k in k_list:
        result[f"precision@{k}"] = ranking_scores[k]["precision"]
        result[f"recall@{k}"] = ranking_scores[k]["recall"]

    # In ngay kết quả của model này (không đợi chạy xong hết cả 4 model
    # mới thấy số, vì các model CF có thể chạy khá lâu).
    print(f"\n    ===== Kết quả - {name} =====")
    print(f"    RMSE            : {rmse:.4f}")
    for k in k_list:
        print(
            f"    Precision@{k:<2} / Recall@{k:<2} : "
            f"{result[f'precision@{k}']:.4f} / {result[f'recall@{k}']:.4f}"
        )
    sys.stdout.flush()

    return result


def print_results_table(
    results: List[Dict[str, float]], k_list: List[int] = TOP_K_LIST
) -> None:
    """
    In bảng so sánh kết quả của các mô hình ra console theo định dạng dễ
    đọc, tương tự cách trình bày bảng thống kê trong file lý thuyết.

    Args:
        results (List[Dict[str, float]]): Danh sách kết quả trả về từ
            `evaluate_model` của từng mô hình.
        k_list (List[int], optional): Danh sách K đã đánh giá.

    Returns:
        None
    """
    headers = ["Model", "RMSE", "Train time (s)"]
    for k in k_list:
        headers.append(f"Precision@{k}")
        headers.append(f"Recall@{k}")

    rows = []
    for res in results:
        row = [
            res["model"],
            f"{res['rmse']:.4f}",
            f"{res['train_time_sec']:.2f}",
        ]
        for k in k_list:
            row.append(f"{res[f'precision@{k}']:.4f}")
            row.append(f"{res[f'recall@{k}']:.4f}")
        rows.append(row)

    col_widths = [
        max(len(str(row[col_idx])) for row in [headers] + rows)
        for col_idx in range(len(headers))
    ]

    def format_row(row: List[str]) -> str:
        return " | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row))

    print("\n===== BẢNG SO SÁNH KẾT QUẢ CÁC MÔ HÌNH =====")
    print(format_row(headers))
    print("-" * (sum(col_widths) + 3 * (len(headers) - 1)))
    for row in rows:
        print(format_row(row))


def save_results(results: List[Dict[str, float]], output_path: str) -> None:
    """
    Lưu bảng kết quả so sánh ra file CSV để phục vụ đưa vào báo cáo
    (Chương 4 - Xây dựng hệ thống / phần đánh giá).

    Args:
        results (List[Dict[str, float]]): Danh sách kết quả của các mô hình.
        output_path (str): Đường dẫn file CSV đầu ra.

    Returns:
        None
    """
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nĐã lưu bảng kết quả vào: {output_path}")


def get_models_to_evaluate() -> List[Tuple[str, BaseRecommender]]:
    """
    Khởi tạo danh sách các mô hình cần huấn luyện và đánh giá - đúng ba
    phương pháp được trình bày trong đề tài (mục 1.2, 1.3 của file lý
    thuyết): Popularity Baseline, Collaborative Filtering (User-based +
    Item-based) và Matrix Factorization.

    Returns:
        List[Tuple[str, BaseRecommender]]: Danh sách (tên mô hình,
            instance mô hình chưa huấn luyện).
    """
    return [
        ("Popularity Baseline", PopularityRecommender()),
        ("User-based CF", UserBasedRecommender(k_neighbors=20)),
        ("Item-based CF", ItemBasedRecommender(k_neighbors=20)),
        (
            "Matrix Factorization (SVD)",
            MatrixFactorizationRecommender(
                n_factors=30,
                n_epochs=20,
                learning_rate=0.01,
                regularization=0.05,
                random_state=RANDOM_STATE,
            ),
        ),
    ]


def main() -> None:
    """
    Hàm chính: đọc dữ liệu, chia Train/Validation/Test, huấn luyện và
    đánh giá lần lượt bốn mô hình (Popularity Baseline, User-based CF,
    Item-based CF, Matrix Factorization), sau đó in và lưu bảng so sánh
    kết quả.

    Returns:
        None
    """
    print("===== ĐỌC VÀ CHIA DỮ LIỆU =====")
    ratings = load_ratings()
    train_data, val_data, test_data = split_data(ratings, random_state=RANDOM_STATE)

    total = len(ratings)
    print(f"Tổng số Rating   : {total:,}")
    print(f"Train ({len(train_data) / total:.0%})      : {len(train_data):,}")
    print(f"Validation ({len(val_data) / total:.0%}) : {len(val_data):,}")
    print(f"Test ({len(test_data) / total:.0%})       : {len(test_data):,}")

    relevant_items = build_relevant_items(test_data, threshold=RELEVANCE_THRESHOLD)
    relevant_items_sample = sample_users_for_ranking(
        relevant_items, sample_size=RANKING_SAMPLE_SIZE, random_state=RANDOM_STATE
    )
    print(
        f"\nSố user có item 'phù hợp' (rating >= {RELEVANCE_THRESHOLD}) "
        f"trong tập Test: {len(relevant_items)}"
    )
    print(
        f"Số user được lấy mẫu để tính Precision@K/Recall@K: "
        f"{len(relevant_items_sample)}"
    )

    models_to_evaluate = get_models_to_evaluate()

    all_results: List[Dict[str, float]] = []
    for name, model in models_to_evaluate:
        result = evaluate_model(
            name, model, train_data, test_data, relevant_items_sample, k_list=TOP_K_LIST
        )
        all_results.append(result)

    print_results_table(all_results, k_list=TOP_K_LIST)

    output_path = os.path.join(PROJECT_ROOT, "evaluation_results.csv")
    save_results(all_results, output_path)


if __name__ == "__main__":
    main()  