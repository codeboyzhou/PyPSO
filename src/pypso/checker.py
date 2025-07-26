def is_converged(values: list[float], consecutive: int = 10, threshold: float = 1e-6) -> bool:
    """
    检查 values 的值是否已经趋于收敛

    Args:
        values (list[float]): 待检查的数据值
        consecutive (int): 允许数据差值满足收敛阈值的连续次数
        threshold (float): 数据差值阈值

    Returns:
        True: values 已经收敛
        False: values 暂未收敛
    """
    length = len(values)

    if length < consecutive:
        return False

    consecutive_count = 0

    for i in range(1, length):
        diff = abs(values[i] - values[i - 1])
        if diff < threshold:
            consecutive_count += 1
            if consecutive_count >= consecutive - 1:
                return True
        else:
            consecutive_count = 0

    return False
