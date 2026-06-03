import numpy as np
import cv2
from typing import List, Tuple
from keypoint_utils import compute_bbox

# ------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------

def draw_keypoints_opencv(image_bgr: np.ndarray,
                          keypoints_groups: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]],
                          radius: int = 5,
                          thickness: int = -1,
                          bbox_expand: int = 30,
                          bbox_thickness: int = 2):
    """
    在 BGR 图像上绘制关键点（圆点）及物体的包围框。
    对不同物体采用不同的关键点颜色，并绘制其包围框（基于所有关键点的最小包围矩形，向四周扩展固定范围）。

    参数:
        image_bgr: BGR 图像（会被原地修改）
        keypoints_groups: 与 render() 返回的第二个元素格式相同，每个元素为 (obj_type, indices, xs, ys)
        radius: 关键点圆点半径
        thickness: 圆点填充厚度（-1 表示填充）
        bbox_expand: 包围框向四周扩展的像素数
        bbox_thickness: 包围框线条粗细
    """
    # 预定义颜色列表（BGR 格式），循环使用
    color_palette = [
        (0, 255, 0),     # 绿色
        (0, 0, 255),     # 红色
        (255, 0, 0),     # 蓝色
        (0, 255, 255),   # 黄色
        (255, 0, 255),   # 品红
        (255, 255, 0),   # 青色
        (128, 0, 255),   # 橙红
        (255, 128, 0),   # 天蓝
        (0, 128, 255),   # 草绿
        (128, 255, 0),   # 黄绿
    ]

    for idx, (obj_type, indices, xs, ys) in enumerate(keypoints_groups):
        # 为该物体选择颜色（基于物体索引循环取色）
        color = color_palette[idx % len(color_palette)]

        # 绘制包围框
        x1, y1, x2, y2 = compute_bbox(xs, ys, bbox_expand)
        # 截断到图像范围
        height, width = image_bgr.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)
        if x1 < x2 and y1 < y2:
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, bbox_thickness)

        # 绘制关键点
        for x, y in zip(xs, ys):
            cx, cy = int(round(x)), int(round(y))
            if 0 <= cx < image_bgr.shape[1] and 0 <= cy < image_bgr.shape[0]:
                cv2.circle(image_bgr, (cx, cy), radius, color, thickness)
 
def blend_with_color_background(rgba: np.ndarray, bg_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    将 RGBA 图像与指定颜色的 RGB 背景进行 alpha 混合。
    bg_color: (B, G, R) 格式的元组，值域 0-255。
    返回 BGR 格式的图像（适合 OpenCV 显示）。
    """
    if rgba.shape[2] != 4:
        raise ValueError("Input image must be RGBA")
    # 分离通道
    r, g, b, a = cv2.split(rgba)
    alpha = a.astype(np.float32) / 255.0
    # 背景图像（纯色）
    bg = np.full((rgba.shape[0], rgba.shape[1], 3), bg_color, dtype=np.uint8)
    # 混合公式：result = foreground * alpha + background * (1 - alpha)
    for c in range(3):
        bg[:, :, c] = (bg[:, :, c] * (1 - alpha)).astype(np.uint8)
    fg = cv2.merge([b, g, r])  # OpenCV 是 BGR 顺序，注意这里直接使用 BGR
    fg = (fg * alpha[..., np.newaxis]).astype(np.uint8)
    result = cv2.add(fg, bg)
    return result 
