#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import numpy as np
import cv2
from typing import Tuple, List, Optional
from power_rune_client import PowerRuneRenderer, blend_with_background, draw_keypoints_opencv

# ------------------------------------------------------------
# 辅助函数：随机生成相机位姿（满足距离和角度约束）
# ------------------------------------------------------------
def random_camera_pose(center: Tuple[float, float, float] = (0.0, 0.0, 3.0),
                       distance_range: Tuple[float, float] = (2.0, 8.0),
                       max_angle_deg: float = 60.0,
                       random_rotation_range_deg: float = 10.0) -> Tuple[Tuple[float, float, float], float, float, float]:
    """
    随机生成相机位姿，相机位于机关前方圆锥与球壳交集内，并随机偏转欧拉角。
    返回: (pos_x, pos_y, pos_z), yaw, pitch, roll (弧度)
    """
    cx, cy, cz = center
    # 1. 随机方向（圆锥轴线为 (0,0,-1)，即机关正面朝向 -Z）
    max_angle_rad = math.radians(max_angle_deg)
    # 随机点积范围 [cos(max_angle), 1]
    dot = random.uniform(math.cos(max_angle_rad), 1.0)
    # 根据点积计算 z 分量（方向向量中的 z）
    # 方向向量 d 满足 d·(0,0,-1) = -dz = dot  => dz = -dot
    dz = -dot
    # 水平半径
    r = math.sqrt(1.0 - dz * dz)
    phi = random.uniform(0, 2 * math.pi)
    dx = r * math.cos(phi)
    dy = r * math.sin(phi)
    direction = np.array([dx, dy, dz])
    # 2. 随机距离
    distance = random.uniform(*distance_range)
    # 相机位置 = 机关中心 + 方向 * 距离
    pos = np.array([cx, cy, cz]) + direction * distance
    # 3. 计算基础欧拉角：使相机正对机关中心 (cx, cy, cz)
    target_vec = np.array([cx, cy, cz]) - pos
    target_vec = target_vec / np.linalg.norm(target_vec)
    # 默认相机前向为 +Z，需要将 +Z 旋转到 target_vec
    # yaw: 绕 Y 轴旋转，使 XZ 投影对准 target_vec
    yaw = math.atan2(target_vec[0], target_vec[2])
    # pitch: 绕 X 轴旋转，使 Y 分量对准
    # 旋转后的向量 (0, target_vec[1], len_xy) 旋转到 (0, 0, 1)
    len_xy = math.sqrt(target_vec[0]**2 + target_vec[2]**2)
    pitch = -math.atan2(target_vec[1], len_xy)
    roll = 0.0  # 基础 roll 为 0
    # 4. 添加随机偏转（± random_rotation_range_deg）
    rand_range_rad = math.radians(random_rotation_range_deg)
    yaw += random.uniform(-rand_range_rad, rand_range_rad)
    pitch += random.uniform(-rand_range_rad, rand_range_rad)
    roll += random.uniform(-rand_range_rad, rand_range_rad)
    return (pos[0], pos[1], pos[2]), yaw, pitch, roll


# ------------------------------------------------------------
# 随机生成扇叶状态及对应参数
# ------------------------------------------------------------
def random_fan_states_and_params(num_fans: int = 5) -> Tuple[List[int], List[Optional[float]], List[Optional[Tuple[float, float]]]]:
    """
    随机生成每个扇叶的状态（0~4）以及额外参数：
    - 状态1: 流动箭头偏移量 (offset, 0~1)
    - 状态4: 内外激活比例 (inner_ratio, outer_ratio)，满足 0.2~1.0 且相差 ≤0.2
    状态1以外的状态概率分布：
        状态0: 0.1, 状态2: 0.5, 状态3: 0.1, 状态4: 0.3
    返回值:
        states: list of int
        offsets: list of float or None (仅状态1有效)
        ratios: list of tuple or None (仅状态4有效)
    """
    # 随机选择 0~2 个扇叶作为状态1（流动箭头）
    num_state1 = random.randint(0, 2)
    state1_indices = random.sample(range(num_fans), num_state1) if num_state1 > 0 else []
    states = []
    offsets = [None] * num_fans
    ratios = [None] * num_fans

    # 其他状态及其权重（顺序：0,2,3,4）
    other_states = [0, 2, 3, 4]
    other_weights = [0.1, 0.6, 0.1, 0.2]

    for i in range(num_fans):
        if i in state1_indices:
            state = 1
            offsets[i] = random.uniform(0.0, 1.0)
        else:
            # 按权重随机选择状态
            state = random.choices(other_states, weights=other_weights, k=1)[0]
            if state == 4:
                # 生成 inner, outer 满足相差 ≤0.2
                inner = random.uniform(0.2, 1.0)
                lower = max(0.2, inner - 0.2)
                upper = min(1.0, inner + 0.2)
                outer = random.uniform(lower, upper)
                ratios[i] = (inner, outer)
        states.append(state)
    return states, offsets, ratios


# ------------------------------------------------------------
# 采样函数：生成一张图像和对应的关键点
# ------------------------------------------------------------
def generate_sample(renderer: PowerRuneRenderer,
                    center: Tuple[float, float, float] = (0.0, 0.0, 3.0)) -> Tuple[np.ndarray, List]:
    """
    随机生成一个样本。
    返回:
        image: RGBA 图像 (H, W, 4), 未合成背景，带透明度通道
        keypoint_groups: 列表，格式与 render() 返回的 groups 相同
    """
    # 1. 随机相机位姿
    pos, yaw, pitch, roll = random_camera_pose(center)
    renderer.set_camera_pose(pos[0], pos[1], pos[2], yaw, pitch, roll)

    # 2. 随机能量机关旋转角度
    rune_angle = random.uniform(0, 2 * math.pi)
    renderer.set_rune_rotation(rune_angle)

    # 3. 随机扇叶状态和参数
    states, offsets, ratios = random_fan_states_and_params(5)
    for idx in range(5):
        state = states[idx]
        renderer.set_fan_state(idx, state)
        if state == 1:
            renderer.set_flowing_arrow_offset(idx, offsets[idx])
        elif state == 4:
            inner, outer = ratios[idx]
            renderer.set_fan_big_activating_ratio(idx, inner, outer)
        # 其它状态无需额外参数，但为了防止残留，可重置比例（可选）
        else:
            # 确保大激活环不可见（设置比例为0）
            renderer.set_fan_big_activating_ratio(idx, 0.0, 0.0)
            # 流动箭头偏移不影响其他状态，但也可重置
            renderer.set_flowing_arrow_offset(idx, 0.0)

    # 4. 渲染
    rgba, groups = renderer.render()
    return rgba, groups


# ------------------------------------------------------------
# 可视化主函数
# ------------------------------------------------------------
def main():
    # 初始化渲染器（分辨率与相机内参匹配，使用演示中的参数）
    renderer = PowerRuneRenderer(logical_width=1280, logical_height=1024)
    renderer.create_power_rune(0.0, 0.0, 3.0)

    # 设置相机内参（与演示一致）
    renderer.set_camera(1.31280460e+03, 1.31309593e+03, 6.38736364e+02, 5.34133502e+02,
                        1280, 1024,
                        k1=-0.05392145, k2=-0.02516686, p1=-0.00222499, p2=-0.00149047, k3=0.43693918)

    # 交互设置
    cv2.namedWindow("Dataset Sample", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dataset Sample", 1280, 1024)

    sample_count = 0
    print("按 's' 保存当前样本（PNG + 关键点文件），按 'n' 生成下一个样本，按 ESC 退出")

    while True:
        # 生成随机样本
        rgba, groups = generate_sample(renderer)

        # 合成背景并绘制关键点用于显示（不影响原始数据）
        bg_color = (16, 16, 32)  # 深色背景
        display_img = blend_with_background(rgba, bg_color)
        draw_keypoints_opencv(display_img, groups, point_color=(0, 255, 0), radius=6)

        cv2.imshow("Dataset Sample", display_img)
        key = cv2.waitKey(0) & 0xFF  # 等待按键

        if key == 27:  # ESC
            break
        elif key == ord('s'):
            # 保存图像和关键点信息
            img_filename = f"sample_{sample_count:04d}.png"
            cv2.imwrite(img_filename, cv2.cvtColor(display_img, cv2.COLOR_BGR2BGRA))
            # 保存关键点（文本格式）
            kp_filename = f"sample_{sample_count:04d}_keypoints.txt"
            with open(kp_filename, 'w') as f:
                for obj_type, indices, xs, ys in groups:
                    f.write(f"object_type: {obj_type}\n")
                    for idx, x, y in zip(indices, xs, ys):
                        f.write(f"  {idx} {x} {y}\n")
            print(f"Saved: {img_filename} and {kp_filename}")
            sample_count += 1
        elif key == ord('n'):
            continue  # 生成下一张

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
