import cv2
from typing import Tuple
from power_rune_client import PowerRuneRenderer
from visualize_utils import draw_keypoints_opencv, blend_with_color_background

# ------------------------------------------------------------
# 简单交互式显示（支持实时调整相机位姿等）
# ------------------------------------------------------------
def run_interactive_demo(renderer: PowerRuneRenderer,
                         bg_color: Tuple[int, int, int] = (16, 16, 32)):
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
        display_img = blend_with_color_background(rgba, bg_color)
        draw_keypoints_opencv(display_img, groups, radius=6, thickness=-1)

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
    run_interactive_demo(renderer, bg_color=(16, 16, 32))
 
