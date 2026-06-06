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
                          bbox_expand: int = 0,
                          bbox_thickness: int = 2,
                          font_scale: float = 0.5,
                          text_thickness: int = 1):
    """
    在 BGR 图像上绘制关键点（圆点）及物体的包围框。
    对不同物体采用不同的关键点颜色，并绘制其包围框（基于所有关键点的最小包围矩形，向四周扩展固定范围）。
    新增功能：
        - 在每个包围框左上角绘制物体类型（obj_type）
        - 在每个关键点附近绘制其索引（indices）

    参数:
        image_bgr: BGR 图像（会被原地修改）
        keypoints_groups: 与 render() 返回的第二个元素格式相同，每个元素为 (obj_type, indices, xs, ys)
        radius: 关键点圆点半径
        thickness: 圆点填充厚度（-1 表示填充）
        bbox_expand: 包围框向四周扩展的像素数
        bbox_thickness: 包围框线条粗细
        font_scale: 文本字体大小
        text_thickness: 文本线条粗细
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

    for idx, (obj_type, indices, xs, ys, valids, occludeds) in enumerate(keypoints_groups):
        # 为该物体选择颜色（基于物体索引循环取色）
        color = color_palette[idx % len(color_palette)]

        # 绘制包围框
        x1, y1, x2, y2 = compute_bbox(xs, ys, bbox_expand)
        height, width = image_bgr.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)
        if x1 < x2 and y1 < y2:
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, bbox_thickness)

            # ---- 1. 绘制物体类型文本 ----
            # 将 obj_type 转为字符串（假设可能是 int 或 str）
            type_str = str(obj_type)
            # 文本位置：框的左上角上方或内部，避免超出图像边界
            text_x = x1
            text_y = y1 - 5 if y1 - 5 > 0 else y1 + 15
            cv2.putText(image_bgr, type_str, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness)

        # 绘制关键点及其索引
        for i, (x, y, valid, occluded) in enumerate(zip(xs, ys, valids, occludeds)):
            cx, cy = int(round(x)), int(round(y))
            if 0 <= cx < image_bgr.shape[1] and 0 <= cy < image_bgr.shape[0]:
                use_color = color if (valid == 1 and occluded == 0) else (255, 255, 255)
                cv2.circle(image_bgr, (cx, cy), radius, use_color, thickness)

                # ---- 2. 绘制关键点索引 ----
                # 获取当前关键点的索引值（来自 indices 列表）
                idx_val = indices[i] if i < len(indices) else i  # 后备方案
                idx_str = str(idx_val)
                # 文本偏移量（避免覆盖圆点）
                offset = radius + 3
                text_cx = cx + offset
                text_cy = cy - offset
                # 确保文本在图像内
                if text_cx > image_bgr.shape[1]:
                    text_cx = cx - offset - 5
                if text_cy < 0:
                    text_cy = cy + offset + 5
                # 使用黑色背景轮廓提高可读性（可选）
                cv2.putText(image_bgr, idx_str, (text_cx, text_cy),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness + 1)
                cv2.putText(image_bgr, idx_str, (text_cx, text_cy),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, use_color, text_thickness)
 
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
