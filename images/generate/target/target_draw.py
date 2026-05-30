"""
极坐标绘图工具 - 使用OpenCV在极坐标下指定区域绘制颜色

功能：
- 支持指定图像大小和缩放比例
- 自动以图像中心作为原点
- 可在极坐标下指定区域 ([theta_min, theta_max][r_min, r_max]) 绘制颜色
- 默认为BGRA图像
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List


def create_canvas(
    width: int = 800,
    height: int = 800,
    dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """
    创建空白BGRA画布。
    """
    return np.zeros((height, width, 4), dtype=dtype)


def get_center(image: np.ndarray) -> Tuple[int, int]:
    """
    获取图像中心坐标。

    Parameters
    ----------
    image : np.ndarray
        图像数组。

    Returns
    -------
    Tuple[int, int]
        (cx, cy) 中心坐标。
    """
    h, w = image.shape[:2]
    return w // 2, h // 2


def polar_to_pixel(
    cx: float, cy: float,
    r: float, theta: float,
    scale: float
) -> Tuple[float, float]:
    """
    将极坐标 (r, theta) 转换为像素坐标 (x, y)。
    """
    x = cx + r * scale * np.cos(theta)
    y = cy + r * scale * np.sin(theta)
    return x, y


def polar_draw(
    image: np.ndarray,
    color: Tuple[int, int, int, int],
    theta_range: Tuple[float, float],
    r_range: Tuple[float, float],
    scale: float = 1.0,
    fill_value: Optional[int] = None,
) -> np.ndarray:
    """
    在极坐标下指定区域 ([theta_min, theta_max][r_min, r_max]) 绘制颜色。

    坐标说明：
    - theta（角度）：以弧度为单位，从正x轴（右侧）逆时针旋转。
        0 = 右侧, pi/2 = 上方, pi = 左侧, 3*pi/2 = 下方。
    - r（半径）：以像素为单位，从图像中心向外延伸。

    Parameters
    ----------
    image : np.ndarray
        BGRA 图像数组，形状为 (height, width, 4)。
    color : Tuple[int, int, int, int]
        BGRA 颜色值，范围 0-255，格式 (R, G, B, A)。
    theta_range : Tuple[float, float]
        角度范围 (theta_min, theta_max)，单位为弧度，范围 [0, 2*pi)。
    r_range : Tuple[float, float]
        半径范围 (r_min, r_max)，单位为像素（实际半径，由scale缩放）。
    scale : float
        缩放比例，实际半径 = r * scale。默认1.0。
    fill_value : Optional[int]
        若不为None，则在绘制区域使用此值替换原始颜色的alpha通道。
        可用于实现覆盖/混合效果。

    Returns
    -------
    np.ndarray
        绘制后的图像数组。
    """
    h, w = image.shape[:2]
    cx, cy = get_center(image)

    theta_min, theta_max = theta_range
    r_min, r_max = r_range

    # 生成网格坐标
    y_indices, x_indices = np.mgrid[0:h, 0:w]

    # 计算相对于中心的偏移量
    x_off = x_indices - cx
    y_off = y_indices - cy

    # 计算极坐标
    # theta: 从正x轴逆时针，atan2(y, x) 返回 [-pi, pi]，将其映射到 [0, 2*pi)
    theta = np.arctan2(y_off, x_off)
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)

    # 半径（考虑缩放因子）
    r = np.sqrt(x_off**2 + y_off**2) / scale

    # 处理角度跨越0的情况（例如 [3*pi/2, pi/2] 跨越了 0/2*pi 边界）
    if theta_min > theta_max:
        # 角度范围跨越了 0/2*pi 边界
        angle_mask = (theta >= theta_min) | (theta <= theta_max)
    else:
        angle_mask = (theta >= theta_min) & (theta <= theta_max)

    # 半径掩码
    r_mask = (r >= r_min) & (r <= r_max)

    # 组合掩码
    combined_mask = angle_mask & r_mask

    # 在指定区域绘制颜色
    if fill_value is not None:
        # 使用指定的fill_value作为alpha值
        draw_color = list(color)
        draw_color[3] = fill_value
        image[combined_mask] = draw_color
    else:
        image[combined_mask] = list(color)

    return image


def compute_keypoints(
    width: int, height: int, scale: float
) -> Tuple[
    List[Tuple[float, float]],
    List[Tuple[float, float]],
    List[Tuple[float, float]],
]:
    """
    计算所有49个关键点的像素坐标。

    Returns
    -------
    all_points : List[Tuple[float, float]]
        所有关键点列表，顺序：[center] + sector_corners + intersections
    sector_corners : List[Tuple[float, float]]
        16个扇形角点
    intersections : List[Tuple[float, float]]
        32个交点
    """
    cx = width / 2.0
    cy = height / 2.0

    # ========== 定义几何参数 ==========
    HALF_ANGLE_DEG = 7.0
    HALF_ANGLE = HALF_ANGLE_DEG * np.pi / 180.0

    # 外圈四条角向边线的半径值（排除内圈两条 r=15, r=35）
    ring_radii = [55.0, 75.0, 115.0, 135.0]

    # 扇形内径和外径（用于计算角点）
    sector_r_min = 50.0
    sector_r_max = 150.0

    # 四个扇形的八条径向边线的角度值
    sector_thetas = [
        (2 * np.pi - HALF_ANGLE, HALF_ANGLE),                # 扇形0: 右
        (np.pi / 2 - HALF_ANGLE, np.pi / 2 + HALF_ANGLE),    # 扇形1: 下
        (np.pi - HALF_ANGLE, np.pi + HALF_ANGLE),            # 扇形2: 左
        (3 * np.pi / 2 - HALF_ANGLE, 3 * np.pi / 2 + HALF_ANGLE),  # 扇形3: 上
    ]

    # 8条径向线的角度（每个扇形有theta_min和theta_max两条边界线）
    radial_angles = []
    for t_min, t_max in sector_thetas:
        radial_angles.append(t_min)
        radial_angles.append(t_max)

    # ========== 计算所有关键点 ==========
    all_points = []

    # 1. 圆心 (index 0)
    center = (float(cx), float(cy))
    all_points.append(center)

    # 2. 四个扇形各自的四个角点 (indices 1-16)
    sector_corners = []
    for t_min, t_max in sector_thetas:
        corners_rtheta = [
            (sector_r_min, t_min),
            (sector_r_max, t_min),
            (sector_r_max, t_max),
            (sector_r_min, t_max),
        ]
        for r, theta in corners_rtheta:
            px, py = polar_to_pixel(cx, cy, r, theta, scale)
            sector_corners.append((px, py))
            all_points.append((px, py))

    # 3. 八条径向边线与四条角向边线的32个交点 (indices 17-48)
    intersections = []
    for theta in radial_angles:
        for r in ring_radii:
            px, py = polar_to_pixel(cx, cy, r, theta, scale)
            intersections.append((px, py))
            all_points.append((px, py))

    return all_points, sector_corners, intersections


def draw_keypoints_on_image(
    image_bgra: np.ndarray,
    all_points: List[Tuple[float, float]],
) -> np.ndarray:
    """
    在图像上标出所有关键点。图像为BGRA格式，所有点标为绿色，标注序号和坐标。
    """
    # BGRA -> BGR (忽略alpha通道)
    img_bgr = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 3
    GREEN = (0, 255, 0)

    for idx, (x, y) in enumerate(all_points):
        pt = (int(round(x)), int(round(y)))

        # 绿色圆点（缩小到1/2）
        cv2.circle(img_bgr, pt, 12, GREEN, -1)
        cv2.circle(img_bgr, pt, 17, GREEN, 3)

        # 标注 "idx(x,y)"，无黑色背景
        label = f"{idx}({x:.1f},{y:.1f})"
        label_x = int(round(x)) + 15
        label_y = int(round(y)) - 15
        cv2.putText(
            img_bgr, label, (label_x, label_y),
            font, font_scale, GREEN, thickness, cv2.LINE_AA,
        )

    return img_bgr


def write_keypoints_file(
    filepath: str,
    all_points: List[Tuple[float, float]],
) -> None:
    """
    将关键点输出到文件，每行一个关键点：index x y
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for idx, (x, y) in enumerate(all_points):
            f.write(f"{idx} {x:.2f} {y:.2f}\n")


def main():
    """主函数：创建图像并在极坐标下绘制示例，最后保存。"""
    # ========== 配置参数 ==========
    WIDTH = 3000          # 图像宽度（像素）
    HEIGHT = 3000         # 图像高度（像素）
    SCALE = 10.0          # 缩放比例

    # ========== 创建画布 ==========
    canvas = create_canvas(WIDTH, HEIGHT)
    print(f"创建画布: {WIDTH}x{HEIGHT}, 缩放比例: {SCALE}")
    print(f"图像中心: {get_center(canvas)}")

    # ========== 绘制 ==========
    # 绘制三个圆环（蓝色）
    polar_draw(
        canvas,
        color=(0, 0, 255, 255),
        theta_range=(0, 2 * np.pi),
        r_range=(15, 35),
        scale=SCALE,
    )
    polar_draw(
        canvas,
        color=(0, 0, 255, 255),
        theta_range=(0, 2 * np.pi),
        r_range=(55, 75),
        scale=SCALE,
    )
    polar_draw(
        canvas,
        color=(0, 0, 255, 255),
        theta_range=(0, 2 * np.pi),
        r_range=(115, 135),
        scale=SCALE,
    )

    # 绘制四个扇形（蓝色）
    polar_draw(
        canvas,
        color=(0, 0, 255, 255),
        theta_range=(2 * np.pi - 7 / 180 * np.pi, 7 / 180 * np.pi),
        r_range=(50, 150),
        scale=SCALE,
    )
    polar_draw(
        canvas,
        color=(0, 0, 255, 255),
        theta_range=(np.pi / 2 - 7 / 180 * np.pi, np.pi / 2 + 7 / 180 * np.pi),
        r_range=(50, 150),
        scale=SCALE,
    )
    polar_draw(
        canvas,
        color=(0, 0, 255, 255),
        theta_range=(np.pi - 7 / 180 * np.pi, np.pi + 7 / 180 * np.pi),
        r_range=(50, 150),
        scale=SCALE,
    )
    polar_draw(
        canvas,
        color=(0, 0, 255, 255),
        theta_range=(np.pi * 3 / 2 - 7 / 180 * np.pi, np.pi * 3 / 2 + 7 / 180 * np.pi),
        r_range=(50, 150),
        scale=SCALE,
    )

    # ========== 计算关键点 ==========
    all_points, sector_corners, intersections = compute_keypoints(WIDTH, HEIGHT, SCALE)

    # 验证关键点数量
    assert len(all_points) == 49, f"关键点数量应为49，实际为{len(all_points)}"
    print(f"\n关键点总计: {len(all_points)}")
    print(f"  - 圆心: 1")
    print(f"  - 扇形角点: {len(sector_corners)}")
    print(f"  - 径向x角向交点: {len(intersections)}")

    # ========== 输出关键点到文件 ==========
    txt_path = "target.txt"
    write_keypoints_file(txt_path, all_points)
    print(f"关键点数据已保存至: {txt_path}")

    # ========== 在图像上标出关键点 ==========
    annotated = draw_keypoints_on_image(canvas, all_points)

    # ========== 保存图像 ==========
    output_path = "target.png"
    cv2.imwrite(output_path, canvas)
    print(f"原图已保存至: {output_path}")

    # 保存带标注的图像
    annotated_path = "target_annotated.png"
    cv2.imwrite(annotated_path, annotated)
    print(f"标注图已保存至: {annotated_path}")


if __name__ == "__main__":
    main()
