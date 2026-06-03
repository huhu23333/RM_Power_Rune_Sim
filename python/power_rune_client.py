#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ctypes
import os
import sys
import numpy as np
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
