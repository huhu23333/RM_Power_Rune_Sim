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
DATASET_VERSION = "dataset_v1"
OUTPUT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "generated_dataset", DATASET_VERSION
)

OUTPUT_SIZE = 640
RENDER_WIDTH = 1280
RENDER_HEIGHT = 1024
SCALE_X = OUTPUT_SIZE / RENDER_WIDTH
SCALE_Y = OUTPUT_SIZE / RENDER_HEIGHT
MAX_KEYPOINTS = 13
VISIBLE_VALID = 2
VISIBLE_MISSING = 0
sampler.set_seed(42)

# 类别名称（与 class_id = obj_type + light_color * 7 对应）
CLASS_NAMES = [
    "R_red", "light_red", "target_red", "arrow_red", "small_red", "inner_red", "outer_red",
    "R_blue", "light_blue", "target_blue", "arrow_blue", "small_blue", "inner_blue", "outer_blue"
]
NUM_CLASSES = len(CLASS_NAMES)

# ------------------------------------------------------------
def pad_keypoints(xs: np.ndarray, ys: np.ndarray, indices: np.ndarray) -> np.ndarray:
    kp_dict = {idx: (x, y) for idx, x, y in zip(indices, xs, ys)}
    padded = np.zeros((MAX_KEYPOINTS, 3), dtype=np.float32)
    for i in range(MAX_KEYPOINTS):
        if i in kp_dict:
            x, y = kp_dict[i]
            padded[i, 0] = x
            padded[i, 1] = y
            padded[i, 2] = VISIBLE_VALID
        else:
            padded[i, 2] = VISIBLE_MISSING
    return padded

# ------------------------------------------------------------
def groups_to_yolo_labels(groups: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]],
                          light_color: int,
                          img_width: int, img_height: int) -> List[str]:
    lines = []
    for obj_type, indices, xs, ys in groups:
        if len(xs) == 0:
            continue

        # ---- 过滤：超出画面比例 ≥30% 的物体丢弃 ----
        total_kps = len(xs)
        out_count = sum(1 for x, y in zip(xs, ys)
                        if x < 0 or x >= img_width or y < 0 or y >= img_height)
        if out_count >= math.ceil(total_kps * 0.3):
            continue  # 丢弃该物体

        class_id = obj_type + light_color * 7

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
        padded_kps = pad_keypoints(xs_norm, ys_norm, indices)

        parts = [str(class_id), f"{box_cx_norm:.6f}", f"{box_cy_norm:.6f}",
                 f"{box_w_norm:.6f}", f"{box_h_norm:.6f}"]
        for i in range(MAX_KEYPOINTS):
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
    for obj_type, indices, xs, ys in groups:
        xs_scaled = xs * SCALE_X
        ys_scaled = ys * SCALE_Y
        scaled_groups.append((obj_type, indices, xs_scaled, ys_scaled))

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
        'kpt_shape': [MAX_KEYPOINTS, 3],
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
    args = parser.parse_args()

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

    bg_sampler = BackgroundSampler()
    
    generate_dataset_yaml()

    print(f"Generating validation set ({args.val} samples)...")
    for i in range(args.val):
        generate_sample(renderer, bg_sampler, "val", i)
        if (i+1) % 100 == 0:
            print(f"  Generated {i+1}")

    print(f"Generating training set ({args.train} samples)...")
    for i in range(args.train):
        generate_sample(renderer, bg_sampler, "train", i)
        if (i+1) % 100 == 0:
            print(f"  Generated {i+1}")


    print(f"Dataset saved to {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
