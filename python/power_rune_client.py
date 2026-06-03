#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ctypes
import os
import sys
import time
import numpy as np
import cv2
from typing import List, Tuple, Optional

# ------------------------------------------------------------
# KeypointGroup 结构体（与 C 接口对应）
# ------------------------------------------------------------
class KeypointGroup(ctypes.Structure):
    _fields_ = [
        ("object_type", ctypes.c_int),
        ("num_keypoints", ctypes.c_int),
        ("indices", ctypes.POINTER(ctypes.c_int)),
        ("xs", ctypes.POINTER(ctypes.c_float)),
        ("ys", ctypes.POINTER(ctypes.c_float)),
    ]

# ------------------------------------------------------------
# PowerRuneRenderer 封装类（与之前相同，略作简化）
# ------------------------------------------------------------
class PowerRuneRenderer:
    """Python 封装类，负责加载动态库并管理渲染会话"""

    def __init__(self, logical_width: int = 3840, logical_height: int = 2160):
        lib_path = self._find_library("power_rune_c_api")
        if lib_path is None:
            raise RuntimeError("Could not find libpower_rune_c_api.so. Make sure build/ directory exists.")

        self._lib = ctypes.CDLL(lib_path)

        # 设置函数原型
        self._lib.create_render_session.argtypes = [ctypes.c_int, ctypes.c_int]
        self._lib.create_render_session.restype = ctypes.c_void_p

        self._lib.destroy_render_session.argtypes = [ctypes.c_void_p]
        self._lib.destroy_render_session.restype = None

        self._lib.set_camera_parameters.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double
        ]
        self._lib.set_camera_parameters.restype = None

        self._lib.set_camera_pose.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double
        ]
        self._lib.set_camera_pose.restype = None

        self._lib.create_power_rune.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        self._lib.create_power_rune.restype = ctypes.c_void_p

        self._lib.destroy_power_rune.argtypes = [ctypes.c_void_p]
        self._lib.destroy_power_rune.restype = None

        self._lib.set_rune_rotation.argtypes = [ctypes.c_void_p, ctypes.c_double]
        self._lib.set_rune_rotation.restype = None

        self._lib.set_fan_state.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self._lib.set_fan_state.restype = None

        self._lib.set_fan_big_activating_ratio.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_double]
        self._lib.set_fan_big_activating_ratio.restype = None

        self._lib.set_flowing_arrow_offset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
        self._lib.set_flowing_arrow_offset.restype = None

        self._lib.render_power_rune.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),  # out_image
            ctypes.POINTER(ctypes.c_int),     # out_width
            ctypes.POINTER(ctypes.c_int),     # out_height
            ctypes.POINTER(ctypes.c_void_p),  # out_groups
            ctypes.POINTER(ctypes.c_int)      # out_num_groups
        ]
        self._lib.render_power_rune.restype = ctypes.c_int

        self._lib.free_image.argtypes = [ctypes.c_void_p]
        self._lib.free_image.restype = None

        self._lib.free_keypoint_groups.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.free_keypoint_groups.restype = None

        # 创建会话
        self._session = self._lib.create_render_session(logical_width, logical_height)
        if not self._session:
            raise RuntimeError("Failed to create render session")

        self._rune = None
        self._logical_w = logical_width
        self._logical_h = logical_height

    @staticmethod
    def _find_library(name: str) -> Optional[str]:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        build_dir = os.path.join(root_dir, "build")

        if sys.platform == "win32":
            lib_name = f"{name}.dll"
        elif sys.platform == "darwin":
            lib_name = f"lib{name}.dylib"
        else:
            lib_name = f"lib{name}.so"

        candidate = os.path.join(build_dir, lib_name)
        if os.path.exists(candidate):
            return candidate
        if os.path.exists(lib_name):
            return lib_name
        return lib_name

    def __del__(self):
        if self._rune:
            self._lib.destroy_power_rune(self._rune)
        if self._session:
            self._lib.destroy_render_session(self._session)

    def create_power_rune(self, center_x: float, center_y: float, center_z: float):
        if self._rune:
            self._lib.destroy_power_rune(self._rune)
        self._rune = self._lib.create_power_rune(self._session, center_x, center_y, center_z)
        if not self._rune:
            raise RuntimeError("Failed to create PowerRune")

    def set_camera(self, fx: float, fy: float, cx: float, cy: float,
                   width: int, height: int,
                   k1: float = 0.0, k2: float = 0.0,
                   p1: float = 0.0, p2: float = 0.0, k3: float = 0.0):
        self._lib.set_camera_parameters(self._session, fx, fy, cx, cy,
                                        width, height,
                                        k1, k2, p1, p2, k3)

    def set_camera_pose(self, pos_x: float, pos_y: float, pos_z: float,
                        yaw: float, pitch: float, roll: float):
        self._lib.set_camera_pose(self._session, pos_x, pos_y, pos_z, yaw, pitch, roll)

    def set_rune_rotation(self, rad: float):
        self._lib.set_rune_rotation(self._rune, rad)

    def set_fan_state(self, index: int, state: int):
        self._lib.set_fan_state(self._rune, index, state)

    def set_fan_big_activating_ratio(self, index: int, inner_ratio: float, outer_ratio: float):
        self._lib.set_fan_big_activating_ratio(self._rune, index, inner_ratio, outer_ratio)

    def set_flowing_arrow_offset(self, index: int, offset: float):
        self._lib.set_flowing_arrow_offset(self._rune, index, offset)

    def render(self) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]]:
        out_image = ctypes.c_void_p()
        out_w = ctypes.c_int()
        out_h = ctypes.c_int()
        out_groups = ctypes.c_void_p()
        out_num_groups = ctypes.c_int()

        ret = self._lib.render_power_rune(
            self._session, self._rune,
            ctypes.byref(out_image), ctypes.byref(out_w), ctypes.byref(out_h),
            ctypes.byref(out_groups), ctypes.byref(out_num_groups)
        )
        if ret != 0:
            raise RuntimeError(f"Render failed with code {ret}")

        h, w = out_h.value, out_w.value
        data = ctypes.string_at(out_image, w * h * 4)
        image = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4).copy()
        self._lib.free_image(out_image)

        groups = []
        if out_num_groups.value > 0:
            group_ptr = ctypes.cast(out_groups, ctypes.POINTER(KeypointGroup))
            for i in range(out_num_groups.value):
                g = group_ptr[i]
                n = g.num_keypoints
                indices = np.array([g.indices[j] for j in range(n)], dtype=np.int32)
                xs = np.array([g.xs[j] for j in range(n)], dtype=np.float32)
                ys = np.array([g.ys[j] for j in range(n)], dtype=np.float32)
                groups.append((g.object_type, indices, xs, ys))
            self._lib.free_keypoint_groups(out_groups, out_num_groups)

        return image, groups

# ------------------------------------------------------------
# 辅助函数：背景合成 + 绘制关键点（仅圆点）
# ------------------------------------------------------------
def blend_with_background(rgba: np.ndarray, bg_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
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

# ------------------------------------------------------------
# 简单交互式显示（支持实时调整相机位姿等）
# ------------------------------------------------------------
def run_interactive_demo(renderer: PowerRuneRenderer,
                         bg_color: Tuple[int, int, int] = (16, 16, 32),
                         point_color: Tuple[int, int, int] = (0, 255, 0)):
    """
    交互式演示：循环渲染并显示，支持键盘控制。
    按键说明：
        ESC / q : 退出
        s       : 保存当前图像为 PNG
        w/s/a/d : 移动相机位置
        i/k/j/l : 调整相机视角（上下左右）
        r       : 重置相机位姿
        1-5     : 切换扇叶状态（索引 0）
        +/-     : 增加/减少旋转角度
    """
    cv2.namedWindow("PowerRune Render", cv2.WINDOW_NORMAL)
    # 可选设置窗口大小（根据屏幕调整）
    cv2.resizeWindow("PowerRune Render", 1280, 1024)

    # 相机控制变量
    camera_pos = [0.0, 0.0, 0.0]
    camera_yaw = 0.0
    camera_pitch = 0.0
    camera_roll = 0.0

    # 能量机关控制变量
    rune_angle = 0.0
    fan_state = 0
    fan_index = 0  # 只控制第一个扇叶示例

    # 从渲染器中获取初始相机参数（假设已经设置过）
    # 为了交互，我们手动设置一组初始相机内参（实际应从外部传入，这里示例使用默认）
    # 注意：这里假设外部已经调用了 renderer.set_camera(...)
    # 如果没有，请先在主程序中设置

    while True:
        # 更新渲染器中的相机位姿
        renderer.set_camera_pose(camera_pos[0], camera_pos[1], camera_pos[2],
                                 camera_yaw, camera_pitch, camera_roll)
        renderer.set_rune_rotation(rune_angle)
        renderer.set_fan_state(fan_index, fan_state)
        renderer.set_fan_big_activating_ratio(0, rune_angle%1.0, rune_angle%1.0)
        renderer.set_flowing_arrow_offset(0, rune_angle%1.0)

        # 渲染
        try:
            rgba, groups = renderer.render()
        except Exception as e:
            print(f"Render error: {e}")
            break

        # 合成背景并绘制关键点
        display_img = blend_with_background(rgba, bg_color)
        draw_keypoints_opencv(display_img, groups, point_color, radius=6, thickness=-1)

        cv2.imshow("PowerRune Render", display_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        # 相机移动（WASD）
        step = 0.1
        rot_step = 0.05
        if key == ord('w'):
            camera_pos[2] -= step
        elif key == ord('s'):
            camera_pos[2] += step
        elif key == ord('a'):
            camera_pos[0] -= step
        elif key == ord('d'):
            camera_pos[0] += step
        elif key == ord(' '):
            camera_pos[1] -= step  # 上升
        elif key == ord('c'):
            camera_pos[1] += step  # 下降
        # 视角调整（i/j/k/l）
        elif key == ord('i'):
            camera_pitch -= rot_step
        elif key == ord('k'):
            camera_pitch += rot_step
        elif key == ord('j'):
            camera_yaw -= rot_step
        elif key == ord('l'):
            camera_yaw += rot_step
        elif key == ord('r'):
            camera_pos = [0.0, 0.0, 0.0]
            camera_yaw = 0.0
            camera_pitch = 0.0
            camera_roll = 0.0
        # 扇叶状态切换（数字键 0-4）
        elif ord('0') <= key <= ord('4'):
            fan_state = key - ord('0')
            print(f"Set fan state to {fan_state}")
        # 旋转角度增加/减少
        elif key == ord('+') or key == ord('='):
            rune_angle += 0.1
        elif key == ord('-') or key == ord('_'):
            rune_angle -= 0.1

    cv2.destroyAllWindows()

# ------------------------------------------------------------
# 主程序示例
# ------------------------------------------------------------
if __name__ == "__main__":
    # 初始化渲染器（分辨率可根据需要调整）
    renderer = PowerRuneRenderer(logical_width=1280, logical_height=1024)

    # 创建能量机关模型（位置和实际场景中的原点对齐）
    renderer.create_power_rune(0.0, 0.0, 3.0)

    # 设置相机内参（与渲染分辨率匹配）
    renderer.set_camera(1.31280460e+03, 1.31309593e+03, 6.38736364e+02, 5.34133502e+02,
                        1280, 1024,
                        k1=-0.05392145, k2=-0.02516686, p1=-0.00222499, p2=-0.00149047, k3=0.43693918)

    # 可选的初始相机位姿（将在交互函数中被覆盖）
    renderer.set_camera_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # 启动交互式显示
    # 背景色： (B,G,R)
    run_interactive_demo(renderer, bg_color=(16, 16, 32), point_color=(0, 255, 0))
