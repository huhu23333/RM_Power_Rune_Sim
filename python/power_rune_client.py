#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ctypes
import os
import sys
import numpy as np
from typing import List, Tuple, Optional

# 可选导入 cv2，用于图像缩放；若没有则使用 numpy 简单缩放
try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

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
        ("world_xs", ctypes.POINTER(ctypes.c_float)),
        ("world_ys", ctypes.POINTER(ctypes.c_float)),
        ("world_zs", ctypes.POINTER(ctypes.c_float)),
        ("cam_xs", ctypes.POINTER(ctypes.c_float)),
        ("cam_ys", ctypes.POINTER(ctypes.c_float)),
        ("cam_zs", ctypes.POINTER(ctypes.c_float)),
        ("valids", ctypes.POINTER(ctypes.c_uint8)),
        ("occludeds", ctypes.POINTER(ctypes.c_uint8)),
    ]

# ------------------------------------------------------------
# PowerRuneRenderer 封装类（支持超采样抗锯齿）
# ------------------------------------------------------------
class PowerRuneRenderer:
    """Python 封装类，负责加载动态库并管理渲染会话，支持超采样抗锯齿"""

    def __init__(self, logical_width: int = 3840, logical_height: int = 2160,
                 super_sample_factor: float = 1.0):
        """
        参数:
            logical_width, logical_height: 最终输出图像的分辨率（宽、高）
            super_sample_factor: 超采样倍数（≥1），例如 2 表示内部以 2x 分辨率渲染，
                                 然后下采样到目标尺寸，实现抗锯齿。
        """
        self._logical_w = logical_width
        self._logical_h = logical_height
        self._super_sample_factor = max(1.0, float(super_sample_factor))
        self._internal_w = int(round(logical_width * self._super_sample_factor))
        self._internal_h = int(round(logical_height * self._super_sample_factor))

        # 保存用户原始相机内参（目标尺寸下的值）
        self._user_intrinsics = None  # 格式: (fx, fy, cx, cy, width, height, k1, k2, p1, p2, k3)

        lib_path = self._find_library("power_rune_c_api")
        if lib_path is None:
            raise RuntimeError("Could not find libpower_rune_c_api.so. Make sure build/ directory exists.")

        self._lib = ctypes.CDLL(lib_path)

        # 设置函数原型（与原来相同）
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

        # 创建会话（内部使用放大后的分辨率）
        self._session = self._lib.create_render_session(self._internal_w, self._internal_h)
        if not self._session:
            raise RuntimeError("Failed to create render session")

        self._rune = None

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
        """
        设置相机内参（目标分辨率下的值）。内部会按超采样倍数自动缩放。
        """
        # 保存用户原始参数（用于后续返回分辨率判断）
        self._user_intrinsics = (fx, fy, cx, cy, width, height, k1, k2, p1, p2, k3)

        # 按超采样倍数缩放内参
        sf = self._super_sample_factor
        internal_fx = fx * sf
        internal_fy = fy * sf
        internal_cx = cx * sf
        internal_cy = cy * sf
        internal_w = int(round(width * sf))
        internal_h = int(round(height * sf))

        self._lib.set_camera_parameters(
            self._session,
            internal_fx, internal_fy, internal_cx, internal_cy,
            internal_w, internal_h,
            k1, k2, p1, p2, k3
        )

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

    def _downsample_image(self, high_res_image: np.ndarray) -> np.ndarray:
        """
        将超采样后的 RGBA 图像降采样到目标分辨率。
        """
        h, w = high_res_image.shape[:2]
        target_w = self._logical_w
        target_h = self._logical_h

        if w == target_w and h == target_h:
            return high_res_image

        if HAVE_CV2:
            # 使用 OpenCV 的 resize，线性插值
            downsampled = cv2.resize(high_res_image, (target_w, target_h),
                                     interpolation=cv2.INTER_LINEAR)
        else:
            # 纯 numpy 实现（简单最近邻 + 平均，效果较差，但备用）
            scale_x = w / target_w
            scale_y = h / target_h
            downsampled = np.zeros((target_h, target_w, 4), dtype=np.uint8)
            for i in range(target_h):
                src_y = int(i * scale_y)
                for j in range(target_w):
                    src_x = int(j * scale_x)
                    downsampled[i, j] = high_res_image[src_y, src_x]
        return downsampled

    def render(self):
        """
        渲染并返回：
            - 图像：RGBA uint8 (H,W,4)
            - 关键点分组：每个分组的 (object_type, indices, xs, ys,
              world_xs, world_ys, world_zs, cam_xs, cam_ys, cam_zs,
              valids, occludeds)
              注意：2D 坐标已自动缩放到目标分辨率，3D 坐标保持原值。
        """
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
        high_res_image = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4).copy()
        self._lib.free_image(out_image)

        # 降采样到目标分辨率
        final_image = self._downsample_image(high_res_image)

        # 处理关键点：2D坐标从超采样分辨率缩放到目标分辨率，3D坐标保持原值
        groups = []
        sf = self._super_sample_factor
        if out_num_groups.value > 0:
            group_ptr = ctypes.cast(out_groups, ctypes.POINTER(KeypointGroup))
            for i in range(out_num_groups.value):
                g = group_ptr[i]
                n = g.num_keypoints
                indices = np.array([g.indices[j] for j in range(n)], dtype=np.int32)
                xs = np.array([g.xs[j] / sf for j in range(n)], dtype=np.float32)
                ys = np.array([g.ys[j] / sf for j in range(n)], dtype=np.float32)
                world_xs = np.array([g.world_xs[j] for j in range(n)], dtype=np.float32)
                world_ys = np.array([g.world_ys[j] for j in range(n)], dtype=np.float32)
                world_zs = np.array([g.world_zs[j] for j in range(n)], dtype=np.float32)
                cam_xs = np.array([g.cam_xs[j] for j in range(n)], dtype=np.float32)
                cam_ys = np.array([g.cam_ys[j] for j in range(n)], dtype=np.float32)
                cam_zs = np.array([g.cam_zs[j] for j in range(n)], dtype=np.float32)
                valids = np.array([g.valids[j] for j in range(n)], dtype=np.uint8)
                occludeds = np.array([g.occludeds[j] for j in range(n)], dtype=np.uint8)
                groups.append((g.object_type, indices, xs, ys,
                               world_xs, world_ys, world_zs,
                               cam_xs, cam_ys, cam_zs,
                               valids, occludeds))
            self._lib.free_keypoint_groups(out_groups, out_num_groups)

        return final_image, groups
    