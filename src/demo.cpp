#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <camera_projection.h>
#include <scene_node.h>
#include <file_utils.h>
#include <render_utils.h>

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include <cstring>
#include <string>

struct fan_node_group_t {
    SceneNode* fan_node;
    ImageNode* fan_background_node;
    ImageNode* fan_light_node;
    ImageNode* target_node;
    ImageNode* flowing_arrow_node;
};

// -----------------------------------------------------------------------------
// 主函数
// -----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // ---------- 1. SDL 初始化 ----------
    SDL_SetLogPriorities(SDL_LOG_PRIORITY_VERBOSE);

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_Log("SDL_Init failed: %s", SDL_GetError());
        return -1;
    }

    const int LOGICAL_WIDTH = 3840;
    const int LOGICAL_HEIGHT = 2160;
    const int WINDOW_WIDTH = 1920;
    const int WINDOW_HEIGHT = 1080;

    SDL_Window* window = SDL_CreateWindow(
        "Demo - RM Power Rune",
        WINDOW_WIDTH, WINDOW_HEIGHT,
        SDL_WINDOW_RESIZABLE);
    if (!window) {
        SDL_Log("Could not create a window: %s", SDL_GetError());
        return -1;
    }
    SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);

    SDL_Renderer* renderer = SDL_CreateRenderer(window, "opengl");
    if (!renderer) {
        SDL_Log("Create renderer failed: %s", SDL_GetError());
        return -1;
    }
    SDL_SetRenderVSync(renderer, 0);

    SDL_Texture* offscreen = SDL_CreateTexture(
        renderer, SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_TARGET, LOGICAL_WIDTH, LOGICAL_HEIGHT);
    if (!offscreen) {
        SDL_Log("Create offscreen texture failed: %s", SDL_GetError());
    }
    SDL_SetTextureBlendMode(offscreen, SDL_BLENDMODE_BLEND_PREMULTIPLIED);

    // ---------- 2. 加载纹理 ----------
    TextureInfo center_R_tex_info = LoadTextureFromPNG(renderer, "images/results/center_R.png");
    SDL_Log("center_R texture size: %dx%d", center_R_tex_info.width, center_R_tex_info.height);

    TextureInfo target_tex_info = LoadTextureFromPNG(renderer, "images/results/target.png");
    SDL_Log("target texture size: %dx%d", target_tex_info.width, target_tex_info.height);

    TextureInfo flowing_arrow_tex_info = LoadTextureFromPNG(renderer, "images/results/flowing_arrow.png");
    SDL_Log("flowing_arrow texture size: %dx%d", flowing_arrow_tex_info.width, flowing_arrow_tex_info.height);

    TextureInfo fan_background_tex_info = LoadTextureFromPNG(renderer, "images/results/fan_background.png");
    SDL_Log("fan_background texture size: %dx%d", fan_background_tex_info.width, fan_background_tex_info.height);

    TextureInfo fan_light_tex_info = LoadTextureFromPNG(renderer, "images/results/fan_light.png");
    SDL_Log("fan_light texture size: %dx%d", fan_light_tex_info.width, fan_light_tex_info.height);

    // ---------- 2b. 读取关键点文件 ----------
    std::vector<Keypoint> target_keypoints = LoadKeypointsFromFile("images/results/target.txt");

    // ---------- 3. 设置相机参数 ----------
    CameraIntrinsics intrinsics{
        960, 960,
        LOGICAL_WIDTH / 2.0, LOGICAL_HEIGHT / 2.0,
        LOGICAL_WIDTH, LOGICAL_HEIGHT
    };
    DistortionCoefficients distortion{0.0, 0.0, 0.0, 0.0, 0.0};

    double max_half_fov = ComputeMaxHalfFovAngle(intrinsics, distortion).first;
    SDL_Log("Max half FOV angle (w/ distortion): %.2f deg",
            max_half_fov * 180.0 / M_PI);
    double diagonal_fov = 2.0 * std::atan(
        std::sqrt((double)(LOGICAL_WIDTH * LOGICAL_WIDTH +
                           LOGICAL_HEIGHT * LOGICAL_HEIGHT)) * 0.5 / 960.0);
    SDL_Log("Diagonal FOV: %.2f deg",
            diagonal_fov * 180.0 / M_PI);

    // ---------- 4. 构建场景节点系统 ----------
    // 单位：m
    Scene scene;

    SceneNode* rune_base_node = CreateSceneNode(scene, 0.0, 0.0, 3.0, nullptr);

    ImageNode* front_center_R_node = CreateImageNode(scene,
                                        center_R_tex_info.texture, center_R_tex_info.width, center_R_tex_info.height,
                                        0.106, 0.106,
                                        0.0, 0.0, -0.3328-0.1664,
                                        1.0f,
                                        {}, rune_base_node, 2);

    SceneNode* front_fan_rotation_center_node = CreateSceneNode(scene, 0.0, 0.0, -0.3328, rune_base_node);

    std::vector<fan_node_group_t> fan_node_groups(5);

    for (int i = 0; i < 5; i += 1) {
        double relative_rotate_rad = M_PI * 2.0 / 5.0 * i;
        auto& fan_node_group = fan_node_groups[i];

        fan_node_group.fan_node = CreateSceneNode(scene, 0.0, 0.0, 0.0, front_fan_rotation_center_node);
        fan_node_group.fan_node -> SetLocalRotation(0.0, 0.0, relative_rotate_rad);

        fan_node_group.fan_background_node = CreateImageNode(scene,
                                            fan_background_tex_info.texture, fan_background_tex_info.width, fan_background_tex_info.height,
                                            0.4151, 0.7419,
                                            0.0, -0.1543-0.7419/2.0, 0.0,
                                            1.0f,
                                            {}, fan_node_group.fan_node, 0);

        fan_node_group.fan_light_node = CreateImageNode(scene,
                                            fan_light_tex_info.texture, fan_light_tex_info.width, fan_light_tex_info.height,
                                            0.4151, 0.7419,
                                            0.0, -0.1543-0.7419/2.0, 0.0,
                                            1.0f,
                                            {}, fan_node_group.fan_node, 1);

        fan_node_group.target_node = CreateImageNode(scene,
                                            target_tex_info.texture, target_tex_info.width, target_tex_info.height,
                                            0.3, 0.3,
                                            0.0, -0.6996, 0.0,
                                            1.0f,
                                            target_keypoints, fan_node_group.fan_node, 1);
        
        fan_node_group.flowing_arrow_node = CreateImageNode(scene,
                                            flowing_arrow_tex_info.texture, flowing_arrow_tex_info.width, flowing_arrow_tex_info.height,
                                            0.06, 0.33,
                                            0.0, -0.1543-0.02-0.33/2.0, 0.0,
                                            1.0f,
                                            {}, fan_node_group.fan_node, 1);
    }


    bool show_keypoints = false;
    int show_light_type = 0;

    // ---------- 5. 控制状态 ----------
    bool mouse_grabbed = false;
    struct CameraPose {
        double yaw{0.0};
        double pitch{0.0};
        double roll{0.0};
    };
    CameraPose camera{};
    Point3D cam_pos{ 0.0, 0.0, 0.0 };
    float mouse_sensitivity = 0.002f;
    float move_speed = 2.0f;

    bool key_w = false, key_s = false, key_a = false, key_d = false;
    bool key_up = false, key_down = false;
    bool key_q = false, key_e = false;

    float roll_speed = 1.5f;
    bool screenshot_requested = false;

    // ---------- 6. 主循环 ----------
    SDL_Event event{};
    bool keep_going = true;
    int frame_count = 0;
    uint64_t fps_last_ticks = SDL_GetTicks();

    SDL_Log("Click inside the window to capture mouse. Move mouse to look around.");
    SDL_Log("WASD: move | SPACE: up | SHIFT: down | ESC: release/quit");
    SDL_Log("Q/E: roll camera | R: reset roll");
    SDL_Log("M: toggle keypoints | P: screenshot");
    SDL_Log("C: change light type");

    uint64_t prev_ticks = SDL_GetTicks();

    while (keep_going) {
        ++frame_count;

        uint64_t current_ticks = SDL_GetTicks();
        float dt = (current_ticks - prev_ticks) / 1000.0f;
        prev_ticks = current_ticks;
        if (dt > 0.05f) dt = 0.05f;

        if (current_ticks - fps_last_ticks >= 1000) {
            SDL_Log("FPS: %d", frame_count);
            frame_count = 0;
            fps_last_ticks = current_ticks;
        }

        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            case SDL_EVENT_QUIT:
                keep_going = false;
                break;

            case SDL_EVENT_KEY_DOWN:
                switch (event.key.key) {
                case SDLK_ESCAPE:
                    if (mouse_grabbed) {
                        SDL_SetWindowRelativeMouseMode(window, false);
                        mouse_grabbed = false;
                        SDL_Log("Mouse released. Click to capture again.");
                    } else {
                        keep_going = false;
                    }
                    break;
                case SDLK_W: key_w = true; break;
                case SDLK_S: key_s = true; break;
                case SDLK_A: key_a = true; break;
                case SDLK_D: key_d = true; break;
                case SDLK_SPACE: key_up = true; break;
                case SDLK_LSHIFT: case SDLK_RSHIFT: key_down = true; break;
                case SDLK_Q: key_q = true; break;
                case SDLK_E: key_e = true; break;
                case SDLK_R: camera.roll = 0.0; break;
                case SDLK_M:
                    if (event.key.repeat == 0) {
                        show_keypoints = !show_keypoints;
                        SDL_Log("Keypoints: %s", show_keypoints ? "ON" : "OFF");
                    }
                    break;
                case SDLK_P:
                    if (event.key.repeat == 0) {
                        screenshot_requested = true;
                    }
                    break;
                case SDLK_C:
                    if (event.key.repeat == 0) {
                        if (show_light_type == 2) {
                            show_light_type = 0;
                        } else {
                            show_light_type += 1;
                        }
                    }
                    break;
                default: break;
                }
                break;

            case SDL_EVENT_KEY_UP:
                switch (event.key.key) {
                case SDLK_W: key_w = false; break;
                case SDLK_S: key_s = false; break;
                case SDLK_A: key_a = false; break;
                case SDLK_D: key_d = false; break;
                case SDLK_SPACE: key_up = false; break;
                case SDLK_LSHIFT: case SDLK_RSHIFT: key_down = false; break;
                case SDLK_Q: key_q = false; break;
                case SDLK_E: key_e = false; break;
                default: break;
                }
                break;

            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                if (!mouse_grabbed) {
                    SDL_SetWindowRelativeMouseMode(window, true);
                    mouse_grabbed = true;
                    SDL_Log("Mouse captured.");
                }
                break;

            case SDL_EVENT_MOUSE_MOTION:
                if (mouse_grabbed) {
                    camera.yaw   += event.motion.xrel * mouse_sensitivity;
                    camera.pitch -= event.motion.yrel * mouse_sensitivity;
                    const double pitch_limit = 1.5;
                    if (camera.pitch >  pitch_limit) camera.pitch =  pitch_limit;
                    if (camera.pitch < -pitch_limit) camera.pitch = -pitch_limit;
                }
                break;

            case SDL_EVENT_WINDOW_RESIZED:
                break;
            }
        }

        // ---------- 摄像机移动 ----------
        double wf_x = std::sin(camera.yaw);
        double wf_z = std::cos(camera.yaw);
        double wr_x = std::cos(camera.yaw);
        double wr_z = -std::sin(camera.yaw);

        if (key_w) { cam_pos.x += wf_x * move_speed * dt; cam_pos.z += wf_z * move_speed * dt; }
        if (key_s) { cam_pos.x -= wf_x * move_speed * dt; cam_pos.z -= wf_z * move_speed * dt; }
        if (key_a) { cam_pos.x -= wr_x * move_speed * dt; cam_pos.z -= wr_z * move_speed * dt; }
        if (key_d) { cam_pos.x += wr_x * move_speed * dt; cam_pos.z += wr_z * move_speed * dt; }
        if (key_up)   cam_pos.y -= move_speed * dt;
        if (key_down) cam_pos.y += move_speed * dt;

        if (key_q) camera.roll += roll_speed * dt;
        if (key_e) camera.roll -= roll_speed * dt;

        // ---------- 图像更新 ----------
        switch (show_light_type)
        {
            case 0:
                for (auto& fan_node_group : fan_node_groups) {
                    fan_node_group.fan_light_node -> SetAlpha(0.0);
                    fan_node_group.flowing_arrow_node -> SetAlpha(0.0);
                }
                break;
            case 1:
                for (auto& fan_node_group : fan_node_groups) {
                    fan_node_group.fan_light_node -> SetAlpha(0.0);
                    fan_node_group.flowing_arrow_node -> SetAlpha(1.0);
                }
                break;
            case 2:
                for (auto& fan_node_group : fan_node_groups) {
                    fan_node_group.fan_light_node -> SetAlpha(1.0);
                    fan_node_group.flowing_arrow_node -> SetAlpha(0.0);
                }
                break;
            
            default:
                break;
        }
        
        for (auto& fan_node_group : fan_node_groups) {
            fan_node_group.flowing_arrow_node -> SetTextureOffset(
                0.0,
                fan_node_group.flowing_arrow_node -> GetTextureOffsetY() + dt * 1.0
        );

        // 更新所有场景节点的世界变换矩阵
        scene.UpdateAllTransforms();

        // =============================================================
        // 渲染步骤：离屏渲染 → 覆盖层 → 截图 → 显示
        // =============================================================

        // ---- Step 1: 开始离屏渲染 ----
        BeginOffscreenRender(renderer, offscreen);

        // ---- Step 2: 渲染场景节点 ----
        scene.RenderAll(renderer, intrinsics, distortion, cam_pos,
                        camera.yaw, camera.pitch, camera.roll);

        SDL_FlushRenderer(renderer);

        // ---- Step 3: 绘制十字丝 ----
        DrawCrosshair(renderer, (float)intrinsics.cx, (float)intrinsics.cy);

        // ---- Step 4: 关键点渲染 ----
        if (show_keypoints)
            for (auto& fan_node_group : fan_node_groups) {
                auto* target_node = fan_node_group.target_node;

                std::vector<KeypointProjection> projections;
                ComputeKeypointProjections(*target_node, intrinsics, distortion,
                                        cam_pos, camera.yaw, camera.pitch, camera.roll,
                                        projections);
                                        
                std::vector<std::vector<ExtraTextureInfo>> all_textures;
                target_node -> RenderKeypoints(renderer, intrinsics, distortion,
                                               cam_pos, camera.yaw, camera.pitch, camera.roll,
                                               all_textures, {0.0, 1.0, 1.0, 1.0});
            };
        }

        // ---- Step 5: 截图 ----
        if (screenshot_requested) {
            screenshot_requested = false;
            SaveScreenshot(renderer, "screenshot/demo_screenshot");
        }

        // ---- Step 6: 呈现到窗口 ----
        PresentOffscreenToWindow(renderer, offscreen,
                                 LOGICAL_WIDTH, LOGICAL_HEIGHT);
    }

    // ---------- 7. 清理 ----------
    if (mouse_grabbed) {
        SDL_SetWindowRelativeMouseMode(window, false);
    }
    if (target_tex_info.texture) SDL_DestroyTexture(target_tex_info.texture);
    SDL_DestroyTexture(offscreen);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}