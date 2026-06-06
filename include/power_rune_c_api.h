#ifndef POWER_RUNE_C_API_H
#define POWER_RUNE_C_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 不透明句柄类型 */
typedef void* RenderSession;
typedef void* PowerRuneHandle;

/* 关键点分组结构体（用于返回给 Python） */
typedef struct {
    int object_type;       // 物体类型：0=R,1=light,2=target,3=arrow,4=small,5=inner,6=outer
    int num_keypoints;
    int* indices;          // 长度 num_keypoints，由接口分配，free_keypoint_groups 负责释放
    float* xs;             // 像素 x 坐标
    float* ys;             // 像素 y 坐标
    uint8_t* valids;             // 是否在画面内
    uint8_t* occludeds;             // 是否被遮挡
} KeypointGroup;

/* ---------- 会话管理 ---------- */
RenderSession create_render_session(int logical_width, int logical_height);
void destroy_render_session(RenderSession session);

/* 设置相机内参和畸变系数（全局，用于本次渲染） */
void set_camera_parameters(RenderSession session,
                           double fx, double fy, double cx, double cy,
                           int width, int height,
                           double k1, double k2, double p1, double p2, double k3);

/* 设置相机位姿（全局，用于本次渲染） */
void set_camera_pose(RenderSession session,
                     double pos_x, double pos_y, double pos_z,
                     double yaw, double pitch, double roll);

/* ---------- PowerRune 实例管理 ---------- */
PowerRuneHandle create_power_rune(RenderSession session,
                                  double center_x, double center_y, double center_z);
void destroy_power_rune(PowerRuneHandle rune);

/* 设置整体旋转角度 */
void set_rune_rotation(PowerRuneHandle rune, double rad);

/* 设置单个扇叶的状态（0~4） */
void set_fan_state(PowerRuneHandle rune, int index, int state);

/* 设置扇叶的大激活环比例（仅当状态为4时生效） */
void set_fan_big_activating_ratio(PowerRuneHandle rune, int index,
                                  double inner_ratio, double outer_ratio);

/* 设置流动箭头的纹理偏移（offset 为 [0,1) 循环） */
void set_flowing_arrow_offset(PowerRuneHandle rune, int index, float offset);

/* ---------- 渲染并获取结果 ---------- */
/* 返回值：0 表示成功，非零表示错误 */
int render_power_rune(RenderSession session, PowerRuneHandle rune,
                      unsigned char** out_image, int* out_width, int* out_height,
                      KeypointGroup** out_groups, int* out_num_groups);

/* 释放由 render_power_rune 分配的内存 */
void free_image(unsigned char* image);
void free_keypoint_groups(KeypointGroup* groups, int num_groups);

#ifdef __cplusplus
}
#endif

#endif // POWER_RUNE_C_API_H
