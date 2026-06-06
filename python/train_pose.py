#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from ultralytics import YOLO
import torch

def train():
    # 数据集配置文件路径（生成脚本输出的 dataset.yaml）
    dataset_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "generated_dataset", "dataset_v2", "dataset.yaml")
    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"Dataset yaml not found: {dataset_yaml}")

    model = YOLO("yolo26n-pose.yaml")

    # 训练参数（关键：禁用水平/垂直翻转）
    results = model.train(
        data=dataset_yaml,
        epochs=30,
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",                # GPU ID，若用 CPU 设为 'cpu'
        workers=8,
        patience=50,
        save=True,
        project="power_rune_train",
        name="power_rune_exp2",
        optimizer="AdamW",
        # --------------------------------------------------
        # 禁用镜像翻转增强
        fliplr=0.0,               # 水平翻转概率 0%
        flipud=0.0,               # 垂直翻转概率 0%
        # # 其他增强可根据需要调整，但不使用翻转
        # degrees=0.0,             # 旋转角度（若需要可保留，但注意关键点对称）
        # translate=0.1,           # 平移增强
        # scale=0.5,               # 缩放增强
        # shear=0.0,               # 剪切变换
        # perspective=0.0,         # 透视变换
        # # --------------------------------------------------
        # # 注意：YOLOv8 还支持 mosaic=1.0, mixup=0.0 等，这些不影响镜像翻转，可保留
        # mosaic=1.0,
        # mixup=0.0,
        # copy_paste=0.0,
    )

    print("Training finished.")

if __name__ == "__main__":
    train()
