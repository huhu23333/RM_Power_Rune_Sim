#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 姿态数据集通用配置，供 generate_yolo_pose_dataset.py 与 generate_world_keypoints.py 共用。
"""

# 物体类型 → YOLO 类别索引
TYPE_TO_YOLO_CLASS = {
    0: 0,   # center_R  → class 0
    2: 1,   # target    → class 1
    3: 2,   # arrow     → class 2
    4: 3,   # small_activating → class 3
}

# 每个 YOLO 类别最多保留的关键点数量
FILTER_MAXNUMS = {
    0: 8,   # R (center_R)
    1: 9,   # target
    2: 4,   # arrow (flowing_arrow)
    3: 11,  # small_activating
}

# 灯效颜色列表
COLORS = ["red", "blue"]

# 基础类别名（不含颜色后缀），顺序与 TYPE_TO_YOLO_CLASS 的 value 对应
CLASS_NAMES_BASE = ["R", "target", "arrow", "small_activating"]

# 按规则生成带颜色后缀的完整类别名列表
def _generate_class_names():
    names = []
    for color in COLORS:
        for base in CLASS_NAMES_BASE:
            names.append(f"{base}_{color}")
    return names

CLASS_NAMES = _generate_class_names()
NUM_CLASSES = len(CLASS_NAMES)

# 各基础类别在全局关键点数组中的起始偏移
KPT_OFFSET = {}
_offset = 0
for _k in sorted(FILTER_MAXNUMS.keys()):
    KPT_OFFSET[_k] = _offset
    _offset += FILTER_MAXNUMS[_k]
TOTAL_KEYPOINTS = _offset  # 所有类别关键点总数
