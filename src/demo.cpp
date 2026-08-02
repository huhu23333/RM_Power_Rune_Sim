#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <camera_projection.h>
#include <scene_node.h>
#include <file_utils.h>
#include <render_utils.h>
#include <power_rune.hpp>
#include <MkvWriter.h>

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <ctime>

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

    // ---------- 2. 设置相机参数 ----------
    CameraIntrinsics intrinsics{
        1300, 1300,
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

    // 单位：m
    Scene scene;

    std::unique_ptr<PowerRune> power_rune = std::make_unique<PowerRune>(renderer, scene, Point3D({0.0, -3.0, 3.0}), true);

    // ---------- 3. 视频输出设置 ----------
    const int VID_WIDTH = 1920;
    const int VID_HEIGHT = 1080;
    const double VID_FPS = 30.0;

    std::unique_ptr<MkvAllIntraWriter> video_writer;
    uint64_t video_start_ticks = 0;
    int video_frame_index = 0;
    bool video_enabled = true;

    if (video_enabled) {
        video_writer = std::make_unique<MkvAllIntraWriter>(60);
        std::time_t now = std::time(nullptr);
        std::tm* local_tm = std::localtime(&now);
        char timestamp[32];
        std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", local_tm);
        char video_filename[256];
        std::snprintf(video_filename, sizeof(video_filename),
                      "screenshot/demo_video_%s.mkv", timestamp);
        if (!video_writer->open(video_filename, VID_WIDTH, VID_HEIGHT, VID_FPS, 8000000)) {
            SDL_Log("Failed to open video writer, video output disabled");
            video_writer.reset();
            video_enabled = false;
        } else {
            SDL_Log("Video output enabled: %s (%dx%d @ %.1f fps)",
                    video_filename, VID_WIDTH, VID_HEIGHT, VID_FPS);
            video_start_ticks = SDL_GetTicks();
        }
    }

    // ---------- 4. 控制状态 ----------
    bool mouse_grabbed = false;
    CameraPose camera_pose{
        Point3D({0.0, 0.0, 0.0}),
        0.0, 0.0, 0.0
    };
    float mouse_sensitivity = 0.002f;
    float move_speed = 2.0f;

    bool key_w = false, key_s = false, key_a = false, key_d = false;
    bool key_up = false, key_down = false;
    bool key_q = false, key_e = false;

    float cam_roll_speed = 1.5f;
    bool screenshot_requested = false;

    bool show_keypoints = false;
    int show_light_type = 0;

    double rune_roll_speed = 1.5;
    double rune_roll_rad = 0.0;

    double flowing_arrow_speed = 1.0;
    double flowing_arrow_offset = 0.0;

    // ---------- 5. 主循环 ----------
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
            size_t queue_size = 0;
            if (video_writer) {
                queue_size = video_writer->getQueueSize();
            }
            SDL_Log("FPS: %d | Video Queue: %zu", frame_count, queue_size);
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
                case SDLK_R: camera_pose.roll = 0.0; break;
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
                        if (show_light_type == 8) {
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
                    camera_pose.yaw   += event.motion.xrel * mouse_sensitivity;
                    camera_pose.pitch -= event.motion.yrel * mouse_sensitivity;
                    const double pitch_limit = 1.5;
                    if (camera_pose.pitch >  pitch_limit) camera_pose.pitch =  pitch_limit;
                    if (camera_pose.pitch < -pitch_limit) camera_pose.pitch = -pitch_limit;
                }
                break;

            case SDL_EVENT_WINDOW_RESIZED:
                break;
            }
        }

        // ---------- 节点位置变换 ----------
        rune_roll_rad += rune_roll_speed * dt;
        power_rune -> SetRotateAngle(rune_roll_rad);

        // ---------- 图像更新 ----------
        switch (show_light_type)
        {
            case 0:
            case 1:
            case 2:
            case 3:
                power_rune -> SetFanState(0, show_light_type);
                break;
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                {
                    double show_fan_light_ratio = (double)(show_light_type - 3) / 5.0;
                    power_rune -> SetFanState(0, 4);
                    power_rune -> SetFanBigActivatingRatio(0, show_fan_light_ratio, show_fan_light_ratio);
                }
                break;

            default:
                break;
        }

        flowing_arrow_offset += dt * flowing_arrow_speed;
        flowing_arrow_offset = flowing_arrow_offset - std::floor(flowing_arrow_offset);
        power_rune -> SetFlowingArrowOffset(0, flowing_arrow_offset);

        // ---------- 摄像机移动 ----------
        double wf_x = std::sin(camera_pose.yaw);
        double wf_z = std::cos(camera_pose.yaw);
        double wr_x = std::cos(camera_pose.yaw);
        double wr_z = -std::sin(camera_pose.yaw);

        if (key_w) { camera_pose.position.x += wf_x * move_speed * dt; camera_pose.position.z += wf_z * move_speed * dt; }
        if (key_s) { camera_pose.position.x -= wf_x * move_speed * dt; camera_pose.position.z -= wf_z * move_speed * dt; }
        if (key_a) { camera_pose.position.x -= wr_x * move_speed * dt; camera_pose.position.z -= wr_z * move_speed * dt; }
        if (key_d) { camera_pose.position.x += wr_x * move_speed * dt; camera_pose.position.z += wr_z * move_speed * dt; }
        if (key_up)   camera_pose.position.y -= move_speed * dt;
        if (key_down) camera_pose.position.y += move_speed * dt;

        if (key_q) camera_pose.roll += cam_roll_speed * dt;
        if (key_e) camera_pose.roll -= cam_roll_speed * dt;

        // 更新所有场景节点的世界变换矩阵
        scene.UpdateAllTransforms();

        // =============================================================
        // 渲染步骤：离屏渲染 → 覆盖层 → 截图 → 显示
        // =============================================================

        // ---- Step 1: 开始离屏渲染 ----
        BeginOffscreenRender(renderer, offscreen);

        // ---- Step 1.5: 关键点计算 ----
        std::vector<std::pair<KeypointExtraInfos, std::vector<KeypointProjection>>> keypoints
            = power_rune -> getShownKeypoints(intrinsics, distortion, camera_pose);

        // ---- Step 2: 渲染场景节点 ----
        scene.RenderAll(renderer, intrinsics, distortion, camera_pose, keypoints);

        // ---- Step 3: 绘制十字丝 ----
        DrawCrosshair(renderer, (float)intrinsics.cx, (float)intrinsics.cy);

        // ---- Step 4: 关键点渲染 ----
        if (show_keypoints) {
            for (auto& [type, projections] : keypoints) {
                std::vector<std::vector<ExtraTextureInfo>> all_textures;
                RenderKeypoints(renderer, intrinsics, distortion,
                                projections, all_textures, {0.0, 1.0, 1.0, 1.0}, {1.0, 0.0, 1.0, 1.0});
            };
        }

        // ---- Step 5: 视频帧写入（异步，与渲染并行） ----
        if (video_writer) {
            double video_elapsed = (current_ticks - video_start_ticks) / 1000.0;
            int expected_frame_index = (int)(video_elapsed * VID_FPS);
            if (expected_frame_index > video_frame_index) {
                // 捕获离屏渲染内容
                SDL_Surface* frame_surface = SDL_RenderReadPixels(renderer, nullptr);
                if (frame_surface) {
                    // 转换为 OpenCV Mat (RGBA → BGR，并缩放到视频分辨率)
                    cv::Mat rgba_mat(frame_surface->h, frame_surface->w, CV_8UC4,
                                     frame_surface->pixels, static_cast<size_t>(frame_surface->pitch));
                    cv::Mat bgr_mat;
                    cv::cvtColor(rgba_mat, bgr_mat, cv::COLOR_RGBA2BGR);
                    // 缩放至视频分辨率
                    if (frame_surface->w != VID_WIDTH || frame_surface->h != VID_HEIGHT) {
                        cv::resize(bgr_mat, bgr_mat, cv::Size(VID_WIDTH, VID_HEIGHT), 0, 0, cv::INTER_LINEAR);
                    }
                    // 异步写入（丢弃模式：队列满时丢弃该帧）
                    video_writer->writeFrame(bgr_mat, true);
                    video_frame_index = expected_frame_index;
                    SDL_DestroySurface(frame_surface);
                }
            }
        }

        // ---- Step 6: 截图 ----
        if (screenshot_requested) {
            screenshot_requested = false;
            SaveScreenshot(renderer, "screenshot/demo_screenshot");
        }

        // ---- Step 7: 呈现到窗口 ----
        PresentOffscreenToWindow(renderer, offscreen,
                                 LOGICAL_WIDTH, LOGICAL_HEIGHT);
    }

    // ---------- 6. 清理视频写入器 ----------
    if (video_writer) {
        SDL_Log("Closing video writer...");
        video_writer->close();
        video_writer.reset();
    }

    // ---------- 7. 计算视频对应的相机参数 ----------
    if (video_enabled) {
        double scale_x = (double)VID_WIDTH / (double)LOGICAL_WIDTH;
        double scale_y = (double)VID_HEIGHT / (double)LOGICAL_HEIGHT;

        CameraIntrinsics video_intrinsics{
            intrinsics.fx * scale_x,
            intrinsics.fy * scale_y,
            intrinsics.cx * scale_x,
            intrinsics.cy * scale_y,
            VID_WIDTH,
            VID_HEIGHT
        };

        // 畸变系数不随分辨率缩放而变化
        DistortionCoefficients video_distortion = distortion;

        SDL_Log("========== 视频相机参数 ==========");
        SDL_Log("渲染分辨率: %d x %d", LOGICAL_WIDTH, LOGICAL_HEIGHT);
        SDL_Log("视频分辨率: %d x %d", VID_WIDTH, VID_HEIGHT);
        SDL_Log("缩放因子: scale_x = %.6f, scale_y = %.6f", scale_x, scale_y);
        SDL_Log("--- 视频相机内参 ---");
        SDL_Log("  fx = %.6f", video_intrinsics.fx);
        SDL_Log("  fy = %.6f", video_intrinsics.fy);
        SDL_Log("  cx = %.6f", video_intrinsics.cx);
        SDL_Log("  cy = %.6f", video_intrinsics.cy);
        SDL_Log("  width = %d, height = %d", video_intrinsics.width, video_intrinsics.height);
        SDL_Log("--- 视频畸变系数 ---");
        SDL_Log("  k1 = %.6f", video_distortion.k1);
        SDL_Log("  k2 = %.6f", video_distortion.k2);
        SDL_Log("  p1 = %.6f", video_distortion.p1);
        SDL_Log("  p2 = %.6f", video_distortion.p2);
        SDL_Log("  k3 = %.6f", video_distortion.k3);
        SDL_Log("==================================");
    }

    // ---------- 8. 清理 ----------
    if (mouse_grabbed) {
        SDL_SetWindowRelativeMouseMode(window, false);
    }
    power_rune.reset();
    SDL_DestroyTexture(offscreen);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}