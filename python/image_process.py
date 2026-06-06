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
    sigma_leak: float,
    leak_k1: float,
    leak_k2: float,
    leak_k_white: float,
    noise_layer_rgb: np.ndarray = None    # (H,W,3) float32 0~1, 若不提供则使用纯黑
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
    leak_white = light.copy()
    leak_white = leak_white / np.max(leak_white)
    if leak_white[0] > leak_white[2]:
        leak_white[1] += leak_k1 * (leak_white[0] - leak_white[1])
        leak_white[2] += leak_k2 * (leak_white[1] - leak_white[2])
    else:
        leak_white[1] += leak_k1 * (leak_white[2] - leak_white[1])
        leak_white[0] += leak_k2 * (leak_white[1] - leak_white[0])
    leak_white = white * leak_k_white + leak_white * (1 - leak_k_white)

    # 直接生成四个直通图层（RGB 常数，alpha 为对应的不透明度）
    def make_layer(rgb_const, alpha_map):
        a = np.clip(alpha_map, 0.0, 1.0)
        layer = np.zeros((*a.shape, 4), dtype=np.float32)
        layer[..., :3] = rgb_const
        layer[..., 3] = a
        return layer

    f = make_layer(light, lum_blur)
    g = make_layer(light, intensity * lum_glow)
    h = make_layer(leak_white, color_leak * lum_leak)

    # 图层 i：RGB 使用噪声图（或纯黑），alpha 使用 alpha_blur
    if noise_layer_rgb is None:
        # 兼容旧行为
        i = make_layer(np.zeros(3), alpha_blur)
    else:
        # 确保噪声图尺寸和类型正确
        assert noise_layer_rgb.shape[:2] == alpha_blur.shape, "噪声图尺寸与 alpha_blur 不匹配"
        assert noise_layer_rgb.dtype == np.float32, "噪声图应为 float32 类型"
        i = np.zeros((*alpha_blur.shape, 4), dtype=np.float32)
        i[..., :3] = noise_layer_rgb
        i[..., 3] = np.clip(alpha_blur, 0.0, 1.0)

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



def sgac_params(color_type, params):
    return (([255-int(params[0]*30), 30+int((params[1]-0.5)*60), 0+int(params[2]*30)] 
             if color_type==0 else 
             [0+int(params[0]*30), 130+int((params[1]-0.5)*60), 255-int(params[2]*30)]), 
             params[3]*5.0, params[4], params[5]*1.0, 3.0+params[6]*8.0, params[7]*3.0, 
             params[8]*0.8, params[9]*0.4, params[10])

