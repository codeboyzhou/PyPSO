from typing import Callable

import numpy as np


def normalize(vectors: np.ndarray) -> np.ndarray:
    """
    对输入的向量进行归一化处理，使每列的值在0到1之间。

    Args:
        vectors (np.ndarray): 输入的向量，形状为 (n, m)

    Returns:
        np.ndarray: 归一化后的向量，形状与输入相同
    """
    min_values = np.min(vectors, axis=0)
    max_values = np.max(vectors, axis=0)
    normalized_vectors = (vectors - min_values) / (max_values - min_values + 1e-6)  # 防止除0
    return normalized_vectors


def binarize(vectors: np.ndarray, condition: Callable[[np.ndarray], bool]) -> np.ndarray:
    """
    将输入向量中的每个元素根据给定的条件进行二值化处理，如果满足条件则为1，否则为0

    Args:
        vectors (np.ndarray): 输入的向量，形状为 (n, m)
        condition (Callable[[np.ndarray | [np.ndarray, ...]], bool]): 用于判断每个元素是否满足条件的函数

    Returns:
        np.ndarray: 二值化后的向量，形状为 (n, m)，每个元素为1或0，表示是否满足条件
    """
    return np.array([1 if condition(x) else 0 for x in vectors])


def cos_angles(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    """
    计算向量之间的余弦夹角

    Args:
        vector1 (np.ndarray): 输入的向量1，形状为 (n, m)，其中 n 是向量数量，m 是向量维度
        vector2 (np.ndarray): 输入的向量2，形状为 (n, m)，其中 n 是向量数量，m 是向量维度

    Returns:
        np.ndarray: 每对向量之间的余弦夹角，形状为 (n, n)
    """
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-6)  # 防止除0
