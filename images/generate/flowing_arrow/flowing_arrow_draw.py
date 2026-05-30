import cv2
import numpy as np

width = 600
section_height = 300
num_sections = 11
total_height = section_height * num_sections

# 创建全透明图像 (BGRA, 4通道)
img = np.zeros((total_height, width, 4), dtype=np.uint8)

# 定义9个点相对于每个 section 左上角的坐标
points = {
    0: {0: (0, 0), 1: (width // 2, 0), 2: (width - 1, 0)},           # 上行
    1: {0: (0, section_height // 2), 1: (width // 2, section_height // 2), 2: (width - 1, section_height // 2)},  # 中行
    2: {0: (0, section_height - 1), 1: (width // 2, section_height - 1), 2: (width - 1, section_height - 1)}     # 下行
}

# 六边形顶点序列（行列编号）
sequence = ['10', '20', '11', '22', '12', '01']

# 将序列转为相对坐标列表
polygon_points = [points[int(s[0])][int(s[1])] for s in sequence]
polygon_points = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))  # shape = (6, 1, 2)

# 在每个 section 中绘制多边形
red_bgra = (0, 0, 255, 255)  # BGR: (0, 0, 255), 透明度255不透明
for i in range(num_sections):
    offset_y = i * section_height
    pts = polygon_points.copy()
    pts[:, 0, 1] += offset_y  # y坐标偏移，x不变
    cv2.fillPoly(img, [pts], red_bgra)

# 保存图片
cv2.imwrite('flowing_arrow.png', img)
print(f"图片已保存: flowing_arrow.png (宽{width}, 高{total_height})")