import cv2
from power_rune_client import PowerRuneRenderer
from image_process import blend_with_background
from visualize_utils import draw_keypoints_opencv
import sample_power_rune
from sample_power_rune import generate_sample, sample_color_and_light

# ------------------------------------------------------------
# 可视化主函数
# ------------------------------------------------------------
def main():
    sample_power_rune.set_seed(42)
    # 初始化渲染器（分辨率与相机内参匹配，使用演示中的参数）
    renderer = PowerRuneRenderer(logical_width=1280, logical_height=1024)
    renderer.create_power_rune(0.0, 0.0, 3.0)

    # 设置相机内参（与演示一致）
    renderer.set_camera(1.31280460e+03, 1.31309593e+03, 6.38736364e+02, 5.34133502e+02,
                        1280, 1024,
                        k1=-0.05392145, k2=-0.02516686, p1=-0.00222499, p2=-0.00149047, k3=0.43693918)

    # 交互设置
    cv2.namedWindow("Sample", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sample", 1280, 1024)

    sample_count = 0
    print("按 'n' 生成下一个样本，按 ESC 退出")

    while True:
        # 生成随机样本
        rgba, groups = generate_sample(renderer)
        sim_rgba, light_color = sample_color_and_light(rgba)

        # 合成背景并绘制关键点用于显示（不影响原始数据）
        bg_color = (16, 16, 32)  # 深色背景
        display_img = blend_with_background(sim_rgba, bg_color)
        draw_keypoints_opencv(display_img, groups, point_color=(0, 255, 0), radius=6)

        cv2.imshow("Sample", display_img)
        key = cv2.waitKey(0) & 0xFF  # 等待按键

        if key == 27:  # ESC
            break
        elif key == ord('n'):
            continue  # 生成下一张

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
