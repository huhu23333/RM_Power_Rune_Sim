#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import cv2
import numpy as np
import yaml
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sampler
from sampler import BackgroundSampler, sample
from keypoint_utils import compute_bbox

# ------------------------------------------------------------
# 配置参数
# ------------------------------------------------------------
DATASET_VERSION = "dataset_v4"
OUTPUT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "generated_dataset", DATASET_VERSION
)

OUTPUT_SIZE = 640
RENDER_WIDTH = 1280
RENDER_HEIGHT = 1280
SCALE_X = OUTPUT_SIZE / RENDER_WIDTH
SCALE_Y = OUTPUT_SIZE / RENDER_HEIGHT
VISIBLE_VALID = 2
VISIBLE_OBSCURED = 1
VISIBLE_MISSING = 0
sampler.set_seed(42)

# 类别名称（与 class_id = obj_type + light_color * 7 对应）
# CLASS_NAMES = [
#     "R_red", "light_red", "target_red", "arrow_red", "small_red", "inner_red", "outer_red",
#     "R_blue", "light_blue", "target_blue", "arrow_blue", "small_blue", "inner_blue", "outer_blue"
# ]
CLASS_NAMES = [
    "R_red", "target_red", "arrow_red", "small_activating_red", 
    "R_blue", "target_blue", "arrow_blue", "small_activating_blue"
]
NUM_CLASSES = len(CLASS_NAMES)

TYPE_TO_YOLO_CLASS = {
    0 : 0,
    2 : 1,
    3 : 2,
    4 : 3
}

FILTER_MAXNUMS = {
    0 : 8,
    1 : 9,
    2 : 4,
    3 : 11
}
# 各基础类别在全局关键点数组中的起始偏移
_KPT_OFFSET = {}
_offset = 0
for _k in sorted(FILTER_MAXNUMS.keys()):
    _KPT_OFFSET[_k] = _offset
    _offset += FILTER_MAXNUMS[_k]
TOTAL_KEYPOINTS = _offset  # 所有类别关键点总数

# ------------------------------------------------------------
def filter_and_pad_keypoints(xs: np.ndarray, ys: np.ndarray, indices: np.ndarray, valids: np.ndarray, occludeds: np.ndarray, obj_type: int) -> np.ndarray:
    """将单个物体的关键点填入全局 TOTAL_KEYPOINTS 数组中,其它不相关索引 visibility=0"""
    c_ = TYPE_TO_YOLO_CLASS[obj_type]
    assert c_ in FILTER_MAXNUMS
    filter_maxnum = FILTER_MAXNUMS[c_]
    offset = _KPT_OFFSET[c_]
    padded = np.zeros((TOTAL_KEYPOINTS, 3), dtype=np.float32)
    for idx, x, y, valid, occluded in zip(indices, xs, ys, valids, occludeds):
        if idx >= filter_maxnum:
            continue
        invisible = occluded or (x<0 or x>1 or y<0 or y>1) or (not valid)
        x = max(min(x, 1.0), 0.0)
        y = max(min(y, 1.0), 0.0)
        global_idx = offset + idx
        padded[global_idx, 0] = x
        padded[global_idx, 1] = y
        padded[global_idx, 2] = VISIBLE_OBSCURED if invisible else VISIBLE_VALID
    return padded

def extract_group_2d(group_tuple):
    """
    从新格式的 group 元组中提取 2D 关键点数据。
    新格式: (object_type, indices, xs, ys, world_xs, world_ys, world_zs,
             cam_xs, cam_ys, cam_zs, valids, occludeds)
    旧格式: (object_type, indices, xs, ys, valids, occludeds)
    """
    obj_type, indices, xs, ys, \
        world_xs, world_ys, world_zs, \
        cam_xs, cam_ys, cam_zs, \
        valids, occludeds = group_tuple
    return obj_type, indices, xs, ys, valids, occludeds

# ------------------------------------------------------------
def groups_to_yolo_labels(groups: List,
                          light_color: int,
                          img_width: int, img_height: int) -> List[str]:
    lines = []
    for group in groups:
        obj_type, indices, xs, ys, valids, occludeds = group
        if len(xs) == 0:
            continue

        # ---- 过滤：超出画面比例 ≥30% 的物体丢弃 ----
        total_kps = len(xs)
        out_count = sum(1 for x, y in zip(xs, ys)
                        if x < 0 or x >= img_width or y < 0 or y >= img_height)
        if out_count >= math.ceil(total_kps * 0.3):
            continue  # 丢弃该物体

        if obj_type not in TYPE_TO_YOLO_CLASS:
            print(f"Warning: obj_type:{obj_type} not in TYPE_TO_YOLO_CLASS, skip")
            continue
        class_id = TYPE_TO_YOLO_CLASS[obj_type] + light_color * len(TYPE_TO_YOLO_CLASS)

        x1, y1, x2, y2 = compute_bbox(xs, ys, expand=0)
        if x1 >= x2 or y1 >= y2:
            continue

        box_cx = (x1 + x2) / 2.0
        box_cy = (y1 + y2) / 2.0
        box_w  = x2 - x1
        box_h  = y2 - y1
        box_cx_norm = box_cx / img_width
        box_cy_norm = box_cy / img_height
        box_w_norm  = box_w / img_width
        box_h_norm  = box_h / img_height

        xs_norm = xs / img_width
        ys_norm = ys / img_height
        padded_kps = filter_and_pad_keypoints(xs_norm, ys_norm, indices, valids, occludeds, obj_type)

        parts = [str(class_id), f"{box_cx_norm:.6f}", f"{box_cy_norm:.6f}",
                 f"{box_w_norm:.6f}", f"{box_h_norm:.6f}"]
        for i in range(TOTAL_KEYPOINTS):
            parts.append(f"{padded_kps[i,0]:.6f}")
            parts.append(f"{padded_kps[i,1]:.6f}")
            parts.append(str(int(padded_kps[i,2])))
        lines.append(" ".join(parts))
    return lines

# ------------------------------------------------------------
def generate_sample(renderer, bg_sampler, split: str, sample_idx: int) -> None:
    bgr_img, light_color, groups = sample(renderer, bg_sampler)

    resized_img = cv2.resize(bgr_img, (OUTPUT_SIZE, OUTPUT_SIZE),
                             interpolation=cv2.INTER_LINEAR)

    scaled_groups = []
    for group in groups:
        obj_type, indices, xs, ys, valids, occludeds = extract_group_2d(group)
        xs_scaled = xs * SCALE_X
        ys_scaled = ys * SCALE_Y
        scaled_groups.append((obj_type, indices, xs_scaled, ys_scaled, valids, occludeds))

    label_lines = groups_to_yolo_labels(scaled_groups, light_color, OUTPUT_SIZE, OUTPUT_SIZE)

    img_dir = os.path.join(OUTPUT_ROOT, "images", split)
    label_dir = os.path.join(OUTPUT_ROOT, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    img_filename = f"{sample_idx:06d}.jpg"
    img_path = os.path.join(img_dir, img_filename)
    cv2.imwrite(img_path, resized_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    label_filename = f"{sample_idx:06d}.txt"
    label_path = os.path.join(label_dir, label_filename)
    with open(label_path, "w") as f:
        f.write("\n".join(label_lines))

# ------------------------------------------------------------
def generate_dataset_yaml():
    yaml_content = {
        'path': os.path.abspath(OUTPUT_ROOT),
        'train': 'images/train',
        'val': 'images/val',
        'nc': NUM_CLASSES,
        'names': CLASS_NAMES,
        'kpt_shape': [TOTAL_KEYPOINTS, 3],
        'flip_idx': []   # 无镜像翻转，无需关键点对称映射
    }
    yaml_path = os.path.join(OUTPUT_ROOT, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    print(f"Generated dataset.yaml at {yaml_path}")

# ------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--val", type=int, default=200)
    parser.add_argument("--supersample", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sub_process_index", type=int, default=-1)
    parser.add_argument("--sub_process_start_index", type=int, default=0)
    parser.add_argument("--sub_process_train", type=int, default=0)
    parser.add_argument("--sub_process_val", type=int, default=0)
    parser.add_argument("--sub_process_yaml", type=int, default=0)
    args = parser.parse_args()

    sampler.set_seed(args.seed)

    if args.sub_process_index == -1:
        from power_rune_client import PowerRuneRenderer
        renderer = PowerRuneRenderer(logical_width=RENDER_WIDTH,
                                    logical_height=RENDER_HEIGHT,
                                    super_sample_factor=args.supersample)
        renderer.create_power_rune(0.0, 0.0, 3.0)

        renderer.set_camera(
            1.31280460e+03, 1.31309593e+03, 6.38736364e+02, 5.34133502e+02,
            RENDER_WIDTH, RENDER_HEIGHT,
            k1=-0.05392145, k2=-0.02516686, p1=-0.00222499, p2=-0.00149047, k3=0.43693918
        )

        bg_sampler = BackgroundSampler((RENDER_WIDTH, RENDER_HEIGHT))

        print(f"Generating validation set ({args.val} samples)...")
        for i in range(args.val):
            generate_sample(renderer, bg_sampler, "val", i)
            if (i+1) % 100 == 0:
                print(f"  Generated {i+1}")

        generate_dataset_yaml()

        print(f"Generating training set ({args.train} samples)...")
        for i in range(args.train):
            generate_sample(renderer, bg_sampler, "train", i)
            if (i+1) % 100 == 0:
                print(f"  Generated {i+1}")


        print(f"Dataset saved to {OUTPUT_ROOT}")

    else: 

        from power_rune_client import PowerRuneRenderer
        renderer = PowerRuneRenderer(logical_width=RENDER_WIDTH,
                                    logical_height=RENDER_HEIGHT,
                                    super_sample_factor=args.supersample)
        renderer.create_power_rune(0.0, 0.0, 3.0)

        renderer.set_camera(
            1.31280460e+03, 1.31309593e+03, 6.38736364e+02, 5.34133502e+02,
            RENDER_WIDTH, RENDER_HEIGHT,
            k1=-0.05392145, k2=-0.02516686, p1=-0.00222499, p2=-0.00149047, k3=0.43693918
        )

        bg_sampler = BackgroundSampler((RENDER_WIDTH, RENDER_HEIGHT))

        if args.sub_process_val > 0:
            print(f"Generating validation set ({args.sub_process_val} samples)...")
            for i in range(args.sub_process_start_index, args.sub_process_start_index + args.sub_process_val):
                generate_sample(renderer, bg_sampler, "val", i)
                if (i+1 - args.sub_process_start_index) % 100 == 0:
                    print(f"  Sub Process[{args.sub_process_index}] Generated {i+1 - args.sub_process_start_index}")

        if args.sub_process_train > 0:
            print(f"Generating training set ({args.sub_process_train} samples)...")
            for i in range(args.sub_process_start_index, args.sub_process_start_index + args.sub_process_train):
                generate_sample(renderer, bg_sampler, "train", i)
                if (i+1 - args.sub_process_start_index) % 100 == 0:
                    print(f"  Sub Process[{args.sub_process_index}] Generated {i+1 - args.sub_process_start_index}")

        if args.sub_process_yaml > 0:
            generate_dataset_yaml()

        print(f"Sub Process[{args.sub_process_index}] Finished")

if __name__ == "__main__":
    main()
