import numpy as np
import cv2
from typing import List, Tuple

# ------------------------------------------------------------
# 辅助函数：绘制关键点（仅圆点）
# ------------------------------------------------------------

def draw_keypoints_opencv(image_bgr: np.ndarray,
                          keypoints_groups: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]],
                          point_color: Tuple[int, int, int] = (0, 255, 0),
                          radius: int = 5,
                          thickness: int = -1):
    """
    在 BGR 图像上绘制关键点（仅圆点，不带文字）。
    keypoints_groups: 格式与 render() 返回的第二个元素相同。
    """
    for obj_type, indices, xs, ys in keypoints_groups:
        for x, y in zip(xs, ys):
            cx, cy = int(round(x)), int(round(y))
            # 检查是否在图像范围内
            if 0 <= cx < image_bgr.shape[1] and 0 <= cy < image_bgr.shape[0]:
                cv2.circle(image_bgr, (cx, cy), radius, point_color, thickness)
 
