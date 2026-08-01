#include "power_rune_c_api.h"

#include <SDL3/SDL.h>
#include <memory>
#include <vector>
#include <cstring>
#include <algorithm>

#include "scene_node.h"
#include "power_rune.hpp"

// 内部会话结构
struct RenderSessionImpl {
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* offscreen = nullptr;
    int logical_w = 0;
    int logical_h = 0;

    Scene scene;                             // 全局场景，所有 PowerRune 共享
    CameraIntrinsics intrinsics;
    DistortionCoefficients distortion;
    CameraPose camera_pose;

    ~RenderSessionImpl() {
        if (offscreen) SDL_DestroyTexture(offscreen);
        if (renderer) SDL_DestroyRenderer(renderer);
        if (window) SDL_DestroyWindow(window);
        SDL_Quit();
    }
};

// 内部 PowerRune 句柄
struct PowerRuneHandleImpl {
    std::unique_ptr<PowerRune> rune;
};

// 辅助函数：将 SDL_Surface 转换为 RGBA 字节数组（复制）
static unsigned char* surface_to_rgba(SDL_Surface* surf, int& out_w, int& out_h) {
    out_w = surf->w;
    out_h = surf->h;
    size_t size = static_cast<size_t>(out_w) * out_h * 4;
    unsigned char* data = static_cast<unsigned char*>(SDL_malloc(size));
    if (!data) return nullptr;
    // 预乘 alpha → 直通 alpha 转换（与 file_utils.cpp 中的截图保存一致）
    uint8_t* pixels = static_cast<uint8_t*>(surf->pixels);
    for (int i = 0; i < out_w * out_h; ++i) {
        uint8_t* p = pixels + i * 4;
        uint8_t a = p[3];
        if (a > 0 && a < 255) {
            p[0] = static_cast<uint8_t>((static_cast<uint16_t>(p[0]) * 255) / a);
            p[1] = static_cast<uint8_t>((static_cast<uint16_t>(p[1]) * 255) / a);
            p[2] = static_cast<uint8_t>((static_cast<uint16_t>(p[2]) * 255) / a);
        }
    }
    std::memcpy(data, surf->pixels, size);
    return data;
}

// 辅助函数：将 C++ 的关键点投影结果转换为 C 结构体数组
static KeypointGroup* convert_keypoint_groups(
    const std::vector<std::pair<int, std::vector<KeypointProjection>>>& groups,
    int& out_num_groups)
{
    out_num_groups = static_cast<int>(groups.size());
    if (out_num_groups == 0) return nullptr;

    KeypointGroup* c_groups = static_cast<KeypointGroup*>(
        SDL_calloc(out_num_groups, sizeof(KeypointGroup)));
    if (!c_groups) return nullptr;

    for (int gi = 0; gi < out_num_groups; ++gi) {
        const auto& group = groups[gi];
        c_groups[gi].object_type = group.first;
        int n = static_cast<int>(group.second.size());
        c_groups[gi].num_keypoints = n;
        if (n > 0) {
            c_groups[gi].indices = static_cast<int*>(SDL_malloc(n * sizeof(int)));
            c_groups[gi].xs = static_cast<float*>(SDL_malloc(n * sizeof(float)));
            c_groups[gi].ys = static_cast<float*>(SDL_malloc(n * sizeof(float)));
            c_groups[gi].world_xs = static_cast<float*>(SDL_malloc(n * sizeof(float)));
            c_groups[gi].world_ys = static_cast<float*>(SDL_malloc(n * sizeof(float)));
            c_groups[gi].world_zs = static_cast<float*>(SDL_malloc(n * sizeof(float)));
            c_groups[gi].cam_xs = static_cast<float*>(SDL_malloc(n * sizeof(float)));
            c_groups[gi].cam_ys = static_cast<float*>(SDL_malloc(n * sizeof(float)));
            c_groups[gi].cam_zs = static_cast<float*>(SDL_malloc(n * sizeof(float)));
            c_groups[gi].valids = static_cast<uint8_t*>(SDL_malloc(n * sizeof(uint8_t)));
            c_groups[gi].occludeds = static_cast<uint8_t*>(SDL_malloc(n * sizeof(uint8_t)));
            if (!c_groups[gi].indices || !c_groups[gi].xs || !c_groups[gi].ys ||
                !c_groups[gi].world_xs || !c_groups[gi].world_ys || !c_groups[gi].world_zs ||
                !c_groups[gi].cam_xs || !c_groups[gi].cam_ys || !c_groups[gi].cam_zs ||
                !c_groups[gi].valids || !c_groups[gi].occludeds) {
                // 清理已分配的内存并返回错误（这里简单置空，调用者需处理）
                for (int j = 0; j <= gi; ++j) {
                    SDL_free(c_groups[j].indices);
                    SDL_free(c_groups[j].xs);
                    SDL_free(c_groups[j].ys);
                    SDL_free(c_groups[j].world_xs);
                    SDL_free(c_groups[j].world_ys);
                    SDL_free(c_groups[j].world_zs);
                    SDL_free(c_groups[j].cam_xs);
                    SDL_free(c_groups[j].cam_ys);
                    SDL_free(c_groups[j].cam_zs);
                    SDL_free(c_groups[j].valids);
                    SDL_free(c_groups[j].occludeds);
                }
                SDL_free(c_groups);
                return nullptr;
            }
            for (int ki = 0; ki < n; ++ki) {
                const auto& proj = group.second[ki];
                c_groups[gi].indices[ki] = proj.index;
                c_groups[gi].xs[ki] = static_cast<float>(proj.screen_pt.x);
                c_groups[gi].ys[ki] = static_cast<float>(proj.screen_pt.y);
                c_groups[gi].world_xs[ki] = static_cast<float>(proj.world_pt.x);
                c_groups[gi].world_ys[ki] = static_cast<float>(proj.world_pt.y);
                c_groups[gi].world_zs[ki] = static_cast<float>(proj.world_pt.z);
                c_groups[gi].cam_xs[ki] = static_cast<float>(proj.cam_pt.x);
                c_groups[gi].cam_ys[ki] = static_cast<float>(proj.cam_pt.y);
                c_groups[gi].cam_zs[ki] = static_cast<float>(proj.cam_pt.z);
                c_groups[gi].valids[ki] = static_cast<uint8_t>(proj.valid);
                c_groups[gi].occludeds[ki] = static_cast<uint8_t>(proj.occluded);
            }
        } else {
            c_groups[gi].indices = nullptr;
            c_groups[gi].xs = nullptr;
            c_groups[gi].ys = nullptr;
            c_groups[gi].world_xs = nullptr;
            c_groups[gi].world_ys = nullptr;
            c_groups[gi].world_zs = nullptr;
            c_groups[gi].cam_xs = nullptr;
            c_groups[gi].cam_ys = nullptr;
            c_groups[gi].cam_zs = nullptr;
            c_groups[gi].valids = nullptr;
            c_groups[gi].occludeds = nullptr;
        }
    }
    return c_groups;
}

// ---------- 公共 API 实现 ----------

extern "C" {

RenderSession create_render_session(int logical_width, int logical_height)
{
    SDL_SetLogPriorities(SDL_LOG_PRIORITY_VERBOSE);
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_Log("SDL_Init failed: %s", SDL_GetError());
        return nullptr;
    }

    auto* session = new RenderSessionImpl();
    session->logical_w = logical_width;
    session->logical_h = logical_height;

    // 创建一个隐藏窗口（避免弹出，但必须创建窗口才能有渲染器）
    session->window = SDL_CreateWindow("PowerRune Offscreen",
                                       logical_width, logical_height,
                                       SDL_WINDOW_HIDDEN);
    if (!session->window) {
        SDL_Log("CreateWindow failed: %s", SDL_GetError());
        delete session;
        return nullptr;
    }

    session->renderer = SDL_CreateRenderer(session->window, "opengl");
    if (!session->renderer) {
        SDL_Log("CreateRenderer failed: %s", SDL_GetError());
        delete session;
        return nullptr;
    }
    SDL_SetRenderVSync(session->renderer, 0);

    // 创建离屏纹理
    session->offscreen = SDL_CreateTexture(session->renderer,
                                           SDL_PIXELFORMAT_RGBA32,
                                           SDL_TEXTUREACCESS_TARGET,
                                           logical_width, logical_height);
    if (!session->offscreen) {
        SDL_Log("CreateTexture failed: %s", SDL_GetError());
        delete session;
        return nullptr;
    }
    SDL_SetTextureBlendMode(session->offscreen, SDL_BLENDMODE_BLEND_PREMULTIPLIED);

    // 默认相机参数（稍后会由用户设置）
    session->intrinsics = { 960.0, 960.0,
                            logical_width / 2.0, logical_height / 2.0,
                            logical_width, logical_height };
    session->distortion = { 0.0, 0.0, 0.0, 0.0, 0.0 };
    session->camera_pose = { {0.0,0.0,0.0}, 0.0, 0.0, 0.0 };

    return session;
}

void destroy_render_session(RenderSession session)
{
    if (session) {
        delete static_cast<RenderSessionImpl*>(session);
    }
}

void set_camera_parameters(RenderSession session,
                           double fx, double fy, double cx, double cy,
                           int width, int height,
                           double k1, double k2, double p1, double p2, double k3)
{
    auto* s = static_cast<RenderSessionImpl*>(session);
    if (!s) return;
    s->intrinsics = { fx, fy, cx, cy, width, height };
    s->distortion = { k1, k2, p1, p2, k3 };
}

void set_camera_pose(RenderSession session,
                     double pos_x, double pos_y, double pos_z,
                     double yaw, double pitch, double roll)
{
    auto* s = static_cast<RenderSessionImpl*>(session);
    if (!s) return;
    s->camera_pose.position = { pos_x, pos_y, pos_z };
    s->camera_pose.yaw = yaw;
    s->camera_pose.pitch = pitch;
    s->camera_pose.roll = roll;
}

PowerRuneHandle create_power_rune(RenderSession session,
                                  double center_x, double center_y, double center_z)
{
    auto* s = static_cast<RenderSessionImpl*>(session);
    if (!s) return nullptr;
    auto* handle = new PowerRuneHandleImpl();
    // PowerRune 构造函数需要 SDL_Renderer*，Scene&，中心点
    handle->rune = std::make_unique<PowerRune>(s->renderer, s->scene,
                                               Point3D{center_x, center_y, center_z});
    return handle;
}

void destroy_power_rune(PowerRuneHandle rune)
{
    delete static_cast<PowerRuneHandleImpl*>(rune);
}

void set_rune_rotation(PowerRuneHandle rune, double rad)
{
    auto* h = static_cast<PowerRuneHandleImpl*>(rune);
    if (h && h->rune) h->rune->SetRotateAngle(rad);
}

void set_fan_state(PowerRuneHandle rune, int index, int state)
{
    auto* h = static_cast<PowerRuneHandleImpl*>(rune);
    if (h && h->rune) h->rune->SetFanState(static_cast<size_t>(index), state);
}

void set_fan_big_activating_ratio(PowerRuneHandle rune, int index,
                                  double inner_ratio, double outer_ratio)
{
    auto* h = static_cast<PowerRuneHandleImpl*>(rune);
    if (h && h->rune) h->rune->SetFanBigActivatingRatio(static_cast<size_t>(index),
                                                         inner_ratio, outer_ratio);
}

void set_flowing_arrow_offset(PowerRuneHandle rune, int index, float offset)
{
    auto* h = static_cast<PowerRuneHandleImpl*>(rune);
    if (h && h->rune) h->rune->SetFlowingArrowOffset(static_cast<size_t>(index), offset);
}

int render_power_rune(RenderSession session, PowerRuneHandle rune,
                      unsigned char** out_image, int* out_width, int* out_height,
                      KeypointGroup** out_groups, int* out_num_groups)
{
    if (!session || !rune || !out_image || !out_width || !out_height ||
        !out_groups || !out_num_groups) {
        return -1;
    }

    auto* s = static_cast<RenderSessionImpl*>(session);
    auto* h = static_cast<PowerRuneHandleImpl*>(rune);
    if (!s || !h || !h->rune) return -2;

    // 1. 开始离屏渲染（清除为透明）
    SDL_SetRenderTarget(s->renderer, s->offscreen);
    SDL_SetRenderDrawColor(s->renderer, 0, 0, 0, 0);
    SDL_RenderClear(s->renderer);

    // 2. 更新场景变换（所有节点，包括 PowerRune 内部节点）
    s->scene.UpdateAllTransforms();

    auto keypoint_groups = h->rune->getShownKeypoints(s->intrinsics, s->distortion, s->camera_pose);

    // 3. 渲染所有场景节点（不绘制十字丝，不绘制额外 UI）
    s->scene.RenderAll(s->renderer, s->intrinsics, s->distortion, s->camera_pose, keypoint_groups);
    SDL_FlushRenderer(s->renderer);

    // 4. 从离屏纹理读取像素数据
    SDL_Surface* surf = SDL_RenderReadPixels(s->renderer, nullptr);
    if (!surf) {
        SDL_Log("SDL_RenderReadPixels failed: %s", SDL_GetError());
        return -3;
    }
    *out_image = surface_to_rgba(surf, *out_width, *out_height);
    SDL_DestroySurface(surf);
    if (!*out_image) return -4;

    // 5. 获取关键点投影（按物体分组）
    std::vector<std::pair<int, std::vector<KeypointProjection>>> keypoint_groups_simple;
    for (auto& [extraInfos, projections] : keypoint_groups) {
        keypoint_groups_simple.push_back({extraInfos.type_index, projections});
    }
    *out_groups = convert_keypoint_groups(keypoint_groups_simple, *out_num_groups);
    if (*out_num_groups > 0 && !*out_groups) {
        // 转换失败，释放图像内存
        free_image(*out_image);
        *out_image = nullptr;
        return -5;
    }

    return 0;
}

void free_image(unsigned char* image)
{
    if (image) SDL_free(image);
}

void free_keypoint_groups(KeypointGroup* groups, int num_groups)
{
    if (!groups) return;
    for (int i = 0; i < num_groups; ++i) {
        SDL_free(groups[i].indices);
        SDL_free(groups[i].xs);
        SDL_free(groups[i].ys);
        SDL_free(groups[i].world_xs);
        SDL_free(groups[i].world_ys);
        SDL_free(groups[i].world_zs);
        SDL_free(groups[i].cam_xs);
        SDL_free(groups[i].cam_ys);
        SDL_free(groups[i].cam_zs);
        SDL_free(groups[i].valids);
        SDL_free(groups[i].occludeds);
    }
    SDL_free(groups);
}

} // extern "C"
