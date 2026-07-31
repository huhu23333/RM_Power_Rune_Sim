#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from ultralytics import YOLO
import torch

def train():
    # 数据集配置文件路径（生成脚本输出的 dataset.yaml）
    dataset_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "generated_dataset", "dataset_v4", "dataset.yaml")
    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"Dataset yaml not found: {dataset_yaml}")

    model = YOLO("yolo11n-pose.yaml")

    # 训练参数（关键：禁用水平/垂直翻转）
    results = model.train(
        data=dataset_yaml,
        epochs=100,
        imgsz=640,
        batch=64,
        device=0 if torch.cuda.is_available() else "cpu",                # GPU ID，若用 CPU 设为 'cpu'
        workers=8,
        patience=50,
        save=True,
        project="power_rune_train",
        name="power_rune_exp4",
        # optimizer="AdamW",
        # --------------------------------------------------
        # 禁用镜像翻转增强
        hsv_h= 0.1,
        hsv_s= 0.8,
        hsv_v= 0.5,
        degrees= 10.0,
        translate= 0.1,
        scale= 0.3,
        shear= 3.0,
        perspective= 0.0003,
        flipud= 0.0,
        fliplr= 0.0,
        bgr= 0.0,
        mosaic= 0.5,
        mixup= 0.0,
        cutmix= 0.0,
        copy_paste= 0.0,
        erasing= 0.5,
        # --------------------------------------------------
        close_mosaic = 30,
    )

    print("Training finished.")

if __name__ == "__main__":
    train()
