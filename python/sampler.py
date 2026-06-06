#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import numpy as np
import os
from typing import Tuple, List, Optional
from power_rune_client import PowerRuneRenderer
from image_process import sim_glow_and_color, sgac_params
import cv2

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

def random_camera_params(origin_image_size):
    # renderer.set_camera(1.31280460e+03, 1.31309593e+03, 6.38736364e+02, 5.34133502e+02,
    #                 origin_image_size[0], origin_image_size[1],
    #                 k1=-0.05392145, k2=-0.02516686, p1=-0.00222499, p2=-0.00149047, k3=0.43693918)
    fxy = max(random.normalvariate(1300, 300), 300)
    fx = max(fxy + random.normalvariate(0, 100), 300)
    fy = max(fxy + random.normalvariate(0, 100), 300)

    cx = random.normalvariate(origin_image_size[0] / 2, origin_image_size[0] * 0.05)
    cy = random.normalvariate(origin_image_size[1] / 2, origin_image_size[1] * 0.05)

    k1 = random.normalvariate(0, 0.06)
    k2 = random.normalvariate(0, 0.03)
    p1 = random.normalvariate(0, 0.002)
    p2 = random.normalvariate(0, 0.002)
    k3 = random.normalvariate(0, 0.5)

    return fx, fy, cx, cy, origin_image_size[0], origin_image_size[1], k1, k2, p1, p2, k3

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
def generate_power_rune_sample(renderer: PowerRuneRenderer, origin_image_size,
                    center: Tuple[float, float, float] = (0.0, 0.0, 3.0)) -> Tuple[np.ndarray, List]:
    """
    随机生成一个样本。
    返回:
        image: RGBA 图像 (H, W, 4), 未合成背景，带透明度通道
        keypoint_groups: 列表，格式与 render() 返回的 groups 相同
    """
    renderer.set_camera(*random_camera_params(origin_image_size))

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

def generate_noise_layer(H: int, W: int) -> np.ndarray:
    """
    生成与指定尺寸匹配的噪声 RGB 图层（float32, 范围 0~1）。
    生成规则：
        1. 随机灰度基底 (0~80)
        2. 每个通道独立随机偏移 (-20~20)
        3. 每个像素每个通道添加高斯噪声 N(0, 10)
        4. 裁剪到 [0, 255] 后归一化到 [0,1]
    """
    # 灰度基底
    base = random.randint(0, 80)
    # 每个通道的偏移
    offsets = [random.randint(-20, 20) for _ in range(3)]
    # 构建初始 RGB
    rgb = np.full((H, W, 3), base, dtype=np.float32)
    for c in range(3):
        rgb[..., c] += offsets[c]
    # 添加高斯噪声
    noise = np.random.normal(0, 10, size=(H, W, 3))
    rgb += noise
    # 裁剪到有效范围并转为 uint8
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    # 归一化到 [0,1] float32
    rgb_float = rgb.astype(np.float32) / 255.0
    return rgb_float

def sample_color_and_light(rgba, noise_rgb):
    rand_color = random.randint(0,1)
    randoms = np.random.random([11])
    return sim_glow_and_color(rgba, *sgac_params(rand_color, randoms), noise_layer_rgb=noise_rgb), rand_color

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


    
class BackgroundSampler:
    def __init__(self, target_size=(1280, 1024), backgrounds_path=None):
        if not backgrounds_path:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(script_dir)
            backgrounds_path = os.path.join(root_dir, "background_images")
        
        self.image_paths = []
        for file_name in os.listdir(backgrounds_path):
            if file_name.split(".")[-1] in ["jpg", "png"]:
                self.image_paths.append(os.path.join(backgrounds_path, file_name))

        self.image_num = len(self.image_paths)
        self.has_images = (self.image_num > 0)
        self.target_size = target_size

        # 可调节的变换参数范围
        self.crop_ratio_range = (0.6, 1.0)          # 随机裁切的比例范围
        self.angle_range = (-30, 30)                # 旋转角度范围（度）
        self.scale_range = (0.8, 1.2)               # 缩放范围
        self.translate_range = (-0.1, 0.1)          # 平移范围（相对图像尺寸的比例）
        self.shear_range = (-0.1, 0.1)              # 错切范围

    def sample_background(self):
        # ---------- 原有流程 ----------
        if self.has_images:
            image_index = random.randint(0, self.image_num - 1)
            image = cv2.imread(self.image_paths[image_index])
            image = cv2.resize(image, self.target_size).astype(np.float32)
            image *= random.random()                     # 随机亮度缩放
        else:
            image = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
        
        noise = np.transpose(np.stack([
            np.ones((self.target_size[1], self.target_size[0]), dtype=np.float32) * random.randint(0, 32),
            np.ones((self.target_size[1], self.target_size[0]), dtype=np.float32) * random.randint(0, 32),
            np.ones((self.target_size[1], self.target_size[0]), dtype=np.float32) * random.randint(0, 32)
        ]), (1, 2, 0))
        noise += np.random.normal(
            np.zeros_like(image),
            np.ones_like(image, dtype=np.float32) * random.random() * 32
        )
        noise = np.clip(noise, 0, 255).astype(np.uint8)
        result = cv2.add(image.astype(np.uint8), noise)

        # ---------- 新增：随机裁切 ----------
        H, W = self.target_size[1], self.target_size[0]
        crop_ratio_w = random.uniform(*self.crop_ratio_range)
        crop_ratio_h = random.uniform(*self.crop_ratio_range)
        crop_w = int(W * crop_ratio_w)
        crop_h = int(H * crop_ratio_h)
        x = random.randint(0, W - crop_w)
        y = random.randint(0, H - crop_h)
        cropped = result[y:y+crop_h, x:x+crop_w]
        result = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)

        # ---------- 新增：任意线性变换（仿射变换） ----------
        # 生成随机的仿射变换参数
        angle = random.uniform(*self.angle_range)
        scale = random.uniform(*self.scale_range)
        tx = random.uniform(*self.translate_range) * W
        ty = random.uniform(*self.translate_range) * H
        shear_x = random.uniform(*self.shear_range)
        shear_y = random.uniform(*self.shear_range)

        # 构建仿射变换矩阵 [a, b, c; d, e, f]
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        # 基础旋转+缩放矩阵
        a = scale * cos_a
        b = -scale * sin_a
        d = scale * sin_a
        e = scale * cos_a
        # 添加错切
        b += shear_x
        d += shear_y
        # 平移项
        c = tx
        f = ty
        M = np.array([[a, b, c], [d, e, f]], dtype=np.float32)

        # 应用仿射变换，输出尺寸固定为 target_size
        result = cv2.warpAffine(
            result, M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        return result
    


def blend_with_background(rgba: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """
    将 RGBA 图像与指定颜色的 BGR 背景进行 alpha 混合。
    返回 BGR 格式的图像（适合 OpenCV 显示）。
    """
    if rgba.shape[2] != 4:
        raise ValueError("Input image must be RGBA")
    # 分离通道
    r, g, b, a = cv2.split(rgba)
    alpha = a.astype(np.float32) / 255.0
    # 混合公式：result = foreground * alpha + background * (1 - alpha)
    for c in range(3):
        bg[:, :, c] = (bg[:, :, c] * (1 - alpha)).astype(np.uint8)
    fg = cv2.merge([b, g, r])  # OpenCV 是 BGR 顺序，注意这里直接使用 BGR
    fg = (fg * alpha[..., np.newaxis]).astype(np.uint8)
    result = cv2.add(fg, bg)
    return result 



# ------------------------ 新增：模糊辅助函数 ------------------------
def generate_motion_blur_kernel(length: int, angle: float) -> np.ndarray:
    """
    生成运动模糊核。
    参数:
        length: 运动模糊的像素长度（运动轨迹的长度）
        angle:  运动方向的角度（度），0° 表示水平向右，90° 表示垂直向下
    返回:
        归一化的二维核矩阵 (k, k)，k 为奇数
    """
    # 确保核尺寸足够大，至少 length+2 并取奇数
    k = max(3, int(length) + 2)
    if k % 2 == 0:
        k += 1
    center = k // 2
    kernel = np.zeros((k, k), dtype=np.float32)

    rad = np.deg2rad(angle)
    dx = np.cos(rad)
    dy = np.sin(rad)

    half_len = length / 2.0
    # 沿着运动方向，步长为 1.0 设置权重
    for t in np.arange(-half_len, half_len + 0.5, 1.0):
        x = center + t * dx
        y = center + t * dy
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < k and 0 <= iy < k:
            kernel[iy, ix] = 1.0

    # 如果没有点被覆盖（通常不会），退化为单点核
    if kernel.sum() == 0:
        kernel[center, center] = 1.0
    else:
        kernel /= kernel.sum()
    return kernel

def apply_random_blur(img: np.ndarray) -> np.ndarray:
    """
    对输入的 BGR 图像应用强度随机的高斯模糊和方向/强度随机的运动模糊。
    返回模糊后的图像（与原图尺寸相同）。
    """
    # 1. 随机高斯模糊
    sigma = random.uniform(-1.0, 2.0)          # 强度随机
    if sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # 2. 随机运动模糊
    motion_len = random.randint(-5, 15)         # 运动像素长度随机
    motion_angle = random.uniform(0, 360)      # 运动方向随机
    if motion_len > 0:
        kernel = generate_motion_blur_kernel(motion_len, motion_angle)
        # 使用 replicate 边界模式，避免边缘出现黑边
        img = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    return img
# -----------------------------------------------------------------

def sample(renderer, background_sampler: BackgroundSampler):
    rgba, groups = generate_power_rune_sample(renderer, background_sampler.target_size)
    # 生成噪声图层（尺寸与 rgba 相同）
    H, W = rgba.shape[0], rgba.shape[1]
    noise_rgb = generate_noise_layer(H, W)
    sim_rgba, light_color = sample_color_and_light(rgba, noise_rgb)
    background = background_sampler.sample_background()
    result_image = blend_with_background(sim_rgba, background)

    # ------------------------ 新增：添加随机模糊 ------------------------
    result_image = apply_random_blur(result_image)

    return result_image, light_color, groups