import numpy as np
import cv2
from typing import Tuple

def sim_glow_and_color(
    rgba: np.ndarray,                     # (H,W,4) uint8
    light_color: Tuple[int, int, int],    # (R,G,B) 0-255
    intensity: float,
    color_leak: float,
    sigma_blur: float,
    sigma_glow: float,
    sigma_leak: float
) -> np.ndarray:
    # 归一化到 [0,1]
    img = rgba.astype(np.float32) / 255.0
    rgb, alpha = img[..., :3], img[..., 3]

    # 预乘 RGB 并计算亮度（通道最大值）
    premul = rgb * alpha[..., np.newaxis]
    luminance = np.max(premul, axis=2)

    # 高斯模糊辅助函数
    def blur(x, sigma):
        return cv2.GaussianBlur(x, (0, 0), sigma, sigma) if sigma > 0 else x

    # 各种模糊结果
    lum_blur = blur(luminance, sigma_blur)
    lum_glow = blur(luminance, sigma_glow)
    lum_leak = blur(luminance, sigma_leak)
    alpha_blur = blur(alpha, sigma_blur)

    # 归一化光色和白色
    light = np.array(light_color, dtype=np.float32) / 255.0
    white = np.ones(3, dtype=np.float32)

    # 直接生成四个直通图层（RGB 常数，alpha 为对应的不透明度）
    def make_layer(rgb_const, alpha_map):
        a = np.clip(alpha_map, 0.0, 1.0)
        layer = np.zeros((*a.shape, 4), dtype=np.float32)
        layer[..., :3] = rgb_const
        layer[..., 3] = a
        return layer

    f = make_layer(light, lum_blur)
    g = make_layer(light, intensity * lum_glow)
    h = make_layer(white, color_leak * lum_leak)
    i = make_layer(np.zeros(3), alpha_blur)

    # 直通 Alpha 叠加（i 为背景，依次叠加上层）
    def over(under, over_layer):
        u_rgb, u_a = under[..., :3], under[..., 3]
        o_rgb, o_a = over_layer[..., :3], over_layer[..., 3]
        out_a = o_a + u_a * (1.0 - o_a)
        out_rgb = np.zeros_like(u_rgb)
        mask = out_a > 0
        if mask.any():
            out_rgb[mask] = (o_rgb[mask] * o_a[mask, np.newaxis] +
                             u_rgb[mask] * u_a[mask, np.newaxis] * (1.0 - o_a[mask, np.newaxis])) / out_a[mask, np.newaxis]
        return np.concatenate([out_rgb, out_a[..., np.newaxis]], axis=-1)

    result = over(i, f)
    result = over(result, g)
    result = over(result, h)

    return (result * 255.0).astype(np.uint8)

def sgac_params(color_type, ratio):
    return ([255, 30, 0] if color_type==0 else [0, 130, 255]), ratio*1.0, ratio, 1.0, 3.0+ratio*2.0, 1.0

