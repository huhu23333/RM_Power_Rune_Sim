import numpy as np
from typing import Tuple

def compute_bbox(xs: np.ndarray, ys: np.ndarray, expand: int = 0) -> Tuple[int, int, int, int]:
    """
    根据关键点计算包围框，并向四周扩展指定像素。

    参数:
        xs: 关键点 x 坐标数组
        ys: 关键点 y 坐标数组
        expand: 向四周扩展的像素数

    返回:
        (x1, y1, x2, y2): 包围框左上角和右下角坐标（整数）
    """
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    x1 = int(np.min(xs))
    y1 = int(np.min(ys))
    x2 = int(np.max(xs))
    y2 = int(np.max(ys))
    # 扩展
    x1 -= expand
    y1 -= expand
    x2 += expand
    y2 += expand
    return (x1, y1, x2, y2) 
