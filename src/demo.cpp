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
#include <random>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <chrono>

// -----------------------------------------------------------------------------
// 主函数
// -----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // ---------- 0. 参数解析 ----------
    bool video_enabled = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--video") == 0) {
            video_enabled = true;
        }
    }

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

    // 相机位姿记录文件
    std::ofstream pose_file;
    std::chrono::steady_clock::time_point last_video_write_time;
    bool first_video_frame = true;
    int pose_frame_index = 0;

    // 视频合成用离屏纹理（背景色 + 缩放的离屏内容 = 最终显示画面）
    SDL_Texture* video_compose_tex = nullptr;
    if (video_enabled) {
        video_compose_tex = SDL_CreateTexture(
            renderer, SDL_PIXELFORMAT_RGBA32,
            SDL_TEXTUREACCESS_TARGET, VID_WIDTH, VID_HEIGHT);
        if (!video_compose_tex) {
            SDL_Log("Create video compose texture failed: %s", SDL_GetError());
            video_enabled = false;
        }
    }

    if (video_enabled) {
        video_writer = std::make_unique<MkvAllIntraWriter>(60);
        std::time_t now = std::time(nullptr);
        std::tm* local_tm = std::localtime(&now);
        char timestamp[32];
        std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", local_tm);

        // 创建 screenshot 下的新文件夹
        char folder_path[256];
        std::snprintf(folder_path, sizeof(folder_path), "screenshot/demo_%s", timestamp);
        std::filesystem::create_directories(folder_path);

        // 视频文件路径
        char video_filename[256];
        std::snprintf(video_filename, sizeof(video_filename),
                      "%s/demo_video_%s.mkv", folder_path, timestamp);
        if (!video_writer->open(video_filename, VID_WIDTH, VID_HEIGHT, VID_FPS, 8000000)) {
            SDL_Log("Failed to open video writer, video output disabled");
            video_writer.reset();
            video_enabled = false;
        } else {
            SDL_Log("Video output enabled: %s (%dx%d @ %.1f fps)",
                    video_filename, VID_WIDTH, VID_HEIGHT, VID_FPS);
            video_start_ticks = SDL_GetTicks();

            // 打开相机位姿记录文件
            char pose_filename[256];
            std::snprintf(pose_filename, sizeof(pose_filename),
                          "%s/camera_pose.txt", folder_path);
            pose_file.open(pose_filename);
            if (!pose_file.is_open()) {
                SDL_Log("Warning: Failed to open pose file: %s", pose_filename);
            } else {
                pose_file << std::fixed << std::setprecision(12);
                SDL_Log("Pose file opened: %s", pose_filename);
            }
        }
    }

    // 视频合成背景色（与 PresentOffscreenToWindow 保持一致）
    const SDL_FColor video_bg_color = { 16.0f/255.0f, 16.0f/255.0f, 32.0f/255.0f, 1.0f };

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
    int current_mode = 0;          // 0, 1, 2

    double rune_roll_speed = M_PI / 3.0;
    double rune_roll_rad = 0.0;

    double flowing_arrow_speed = 1.0;
    double flowing_arrow_offset = 0.0;

    // ---------- 模式相关状态 ----------
    // Mode 1
    double mode1_timer = 0.0;
    enum class Mode1Phase { ACTIVATE, WAIT_ALL_ZERO, COOLDOWN };
    Mode1Phase mode1_phase = Mode1Phase::ACTIVATE;
    int mode1_current_fan = -1;
    std::vector<int> mode1_fan_states; // 0=idle, 1=active, 2=done
    double mode1_phase_timer = 0.0;

    // Mode 2
    double mode2_t = 0.0;           // 进入模式2后的累计时间
    double mode2_a = 0.913;         // 默认 a
    double mode2_omega = 1.942;     // 默认 ω
    double mode2_b = 2.090 - mode2_a;
    double mode2_param_timer = 0.0; // 参数刷新周期计时
    enum class Mode2Phase { WAIT_1, WAIT_2, WAIT_3, ALL_ZERO_COOLDOWN };
    Mode2Phase mode2_phase = Mode2Phase::WAIT_1;
    double mode2_phase_timer = 0.0;
    double mode2_ratio = 0.0;
    int mode2_fan1 = -1;
    int mode2_fan2 = -1;

    // 随机数生成器
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist_a(0.780, 1.045);
    std::uniform_real_distribution<double> dist_omega(1.884, 2.000);

    // Mode 2 文字显示
    SDL_Texture* mode2_text_tex = nullptr;
    int mode2_text_w = 0, mode2_text_h = 0;

    // 相机位姿文字显示（始终显示在屏幕最后一行）
    SDL_Texture* camera_text_tex = nullptr;
    int camera_text_w = 0, camera_text_h = 0;

    // ---------- 5. 初始化模式0默认状态 ----------
    for (int i = 0; i < 5; i++) {
        power_rune->SetFanState(i, 0);
    }

    // ---------- 6. 主循环 ----------
    SDL_Event event{};
    bool keep_going = true;
    int frame_count = 0;
    uint64_t fps_last_ticks = SDL_GetTicks();

    SDL_Log("Click inside the window to capture mouse. Move mouse to look around.");
    SDL_Log("WASD: move | SPACE: up | SHIFT: down | ESC: release/quit");
    SDL_Log("Q/E: roll camera | R: reset roll");
    SDL_Log("M: toggle keypoints | P: screenshot");
    SDL_Log("C: change mode (0/1/2)");
    SDL_Log("--video: enable video output");

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
            SDL_Log("FPS: %d | Mode: %d | Video Queue: %zu", frame_count, current_mode, queue_size);
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
                        current_mode = (current_mode + 1) % 3;
                        SDL_Log("Mode switched to: %d", current_mode);

                        // 初始化模式状态
                        if (current_mode == 1) {
                            rune_roll_speed = M_PI / 3.0;
                            mode1_timer = 0.0;
                            mode1_phase = Mode1Phase::ACTIVATE;
                            mode1_fan_states.assign(5, 0);
                            int r = std::uniform_int_distribution<int>(0, 4)(rng);
                            mode1_current_fan = r;
                            mode1_fan_states[r] = 1;
                            mode1_phase_timer = 0.0;
                            // 只显示激活的扇叶
                            for (int i = 0; i < 5; i++) {
                                power_rune->SetFanState(i, (i == r) ? 1 : 0);
                            }
                        } else if (current_mode == 2) {
                            mode2_t = 0.0;
                            mode2_a = dist_a(rng);
                            mode2_omega = dist_omega(rng);
                            mode2_b = 2.090 - mode2_a;
                            mode2_param_timer = 0.0;
                            mode2_phase = Mode2Phase::WAIT_1;
                            mode2_phase_timer = 0.0;
                            mode2_ratio = 0.0;

                            // 全部设为4
                            for (int i = 0; i < 5; i++) {
                                power_rune->SetFanState(i, 4);
                                power_rune->SetFanBigActivatingRatio(i, 0.0, 0.0);
                            }
                            // 随机两个扇形设为1
                            mode2_fan1 = std::uniform_int_distribution<int>(0, 4)(rng);
                            do {
                                mode2_fan2 = std::uniform_int_distribution<int>(0, 4)(rng);
                            } while (mode2_fan2 == mode2_fan1);
                            power_rune->SetFanState(mode2_fan1, 1);
                            power_rune->SetFanState(mode2_fan2, 1);
                        } else {
                            // Mode 0
                            rune_roll_speed = M_PI / 3.0;
                            for (int i = 0; i < 5; i++) {
                                power_rune->SetFanState(i, 0);
                            }
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
        if (current_mode == 2) {
            mode2_t += dt;
            rune_roll_speed = mode2_a * std::sin(mode2_omega * mode2_t) + mode2_b;

            // 每30秒随机更新参数
            mode2_param_timer += dt;
            if (mode2_param_timer >= 30.0) {
                mode2_param_timer -= 30.0;
                mode2_a = dist_a(rng);
                mode2_omega = dist_omega(rng);
                mode2_b = 2.090 - mode2_a;
            }
        }
        rune_roll_rad += rune_roll_speed * dt;
        power_rune -> SetRotateAngle(rune_roll_rad);

        // ---------- 模式逻辑更新 ----------
        if (current_mode == 0) {
            // 模式0: 全部扇叶状态为0
            for (int i = 0; i < 5; i++) {
                power_rune->SetFanState(i, 0);
            }
            rune_roll_speed = M_PI / 3.0;
        } else if (current_mode == 1) {
            // 模式1
            rune_roll_speed = M_PI / 3.0;
            mode1_timer += dt;

            if (mode1_phase == Mode1Phase::ACTIVATE) {
                mode1_phase_timer += dt;
                if (mode1_phase_timer >= 1.5) {
                    // 将当前扇叶状态变为2
                    power_rune->SetFanState(mode1_current_fan, 2);
                    mode1_fan_states[mode1_current_fan] = 2;
                    mode1_phase_timer = 0.0;

                    // 检查是否所有扇叶都非0
                    bool all_done = true;
                    for (int i = 0; i < 5; i++) {
                        if (mode1_fan_states[i] == 0) {
                            all_done = false;
                            break;
                        }
                    }
                    if (all_done) {
                        mode1_phase = Mode1Phase::WAIT_ALL_ZERO;
                        mode1_phase_timer = 0.0;
                    } else {
                        // 随机选择另一个状态为0的扇叶变为1
                        std::vector<int> candidates;
                        for (int i = 0; i < 5; i++) {
                            if (mode1_fan_states[i] == 0) candidates.push_back(i);
                        }
                        int next = candidates[std::uniform_int_distribution<int>(0, (int)candidates.size() - 1)(rng)];
                        mode1_current_fan = next;
                        mode1_fan_states[next] = 1;
                        power_rune->SetFanState(next, 1);
                    }
                }
            } else if (mode1_phase == Mode1Phase::WAIT_ALL_ZERO) {
                mode1_phase_timer += dt;
                if (mode1_phase_timer >= 1.5) {
                    // 全部扇叶状态变为0
                    for (int i = 0; i < 5; i++) {
                        power_rune->SetFanState(i, 0);
                        mode1_fan_states[i] = 0;
                    }
                    mode1_phase = Mode1Phase::COOLDOWN;
                    mode1_phase_timer = 0.0;
                }
            } else { // COOLDOWN
                mode1_phase_timer += dt;
                if (mode1_phase_timer >= 2.5) {
                    // 重新开始循环
                    mode1_phase = Mode1Phase::ACTIVATE;
                    mode1_phase_timer = 0.0;
                    mode1_fan_states.assign(5, 0);
                    int r = std::uniform_int_distribution<int>(0, 4)(rng);
                    mode1_current_fan = r;
                    mode1_fan_states[r] = 1;
                    power_rune->SetFanState(r, 1);
                    for (int i = 0; i < 5; i++) {
                        if (i != r) power_rune->SetFanState(i, 0);
                    }
                }
            }
        } else if (current_mode == 2) {
            // 模式2 扇叶激活循环
            mode2_phase_timer += dt;

            if (mode2_phase == Mode2Phase::WAIT_1) {
                if (mode2_phase_timer >= 1.5) {
                    mode2_phase_timer = 0.0;
                    // 将其中一个状态为1的扇叶变为4
                    power_rune->SetFanState(mode2_fan1, 4);
                    mode2_phase = Mode2Phase::WAIT_2;
                }
            } else if (mode2_phase == Mode2Phase::WAIT_2) {
                if (mode2_phase_timer >= 0.8) {
                    mode2_phase_timer = 0.0;
                    // 将另一个状态为1的扇叶也变为4
                    power_rune->SetFanState(mode2_fan2, 4);
                    mode2_phase = Mode2Phase::WAIT_3;
                }
            } else if (mode2_phase == Mode2Phase::WAIT_3) {
                if (mode2_phase_timer >= 0.2) {
                    mode2_phase_timer = 0.0;
                    // 更新比例
                    mode2_ratio += 0.2;
                    if (mode2_ratio > 1.0) {
                        mode2_ratio = 0.0;
                        // 比例回绕到0.0时，进入全0冷却阶段
                        for (int i = 0; i < 5; i++) {
                            power_rune->SetFanState(i, 0);
                        }
                        mode2_phase = Mode2Phase::ALL_ZERO_COOLDOWN;
                    } else {
                        // 全部设为4并设置比例
                        for (int i = 0; i < 5; i++) {
                            power_rune->SetFanState(i, 4);
                            power_rune->SetFanBigActivatingRatio(i, mode2_ratio, mode2_ratio);
                        }
                        // 随机两个扇叶变为1
                        mode2_fan1 = std::uniform_int_distribution<int>(0, 4)(rng);
                        do {
                            mode2_fan2 = std::uniform_int_distribution<int>(0, 4)(rng);
                        } while (mode2_fan2 == mode2_fan1);
                        power_rune->SetFanState(mode2_fan1, 1);
                        power_rune->SetFanState(mode2_fan2, 1);
                        mode2_phase = Mode2Phase::WAIT_1;
                    }
                }
            } else { // ALL_ZERO_COOLDOWN
                mode2_phase_timer += dt;
                if (mode2_phase_timer >= 2.5) {
                    mode2_phase_timer = 0.0;
                    // 全部设为4并设置比例（ratio已经是0.0）
                    for (int i = 0; i < 5; i++) {
                        power_rune->SetFanState(i, 4);
                        power_rune->SetFanBigActivatingRatio(i, mode2_ratio, mode2_ratio);
                    }
                    // 随机两个扇叶变为1
                    mode2_fan1 = std::uniform_int_distribution<int>(0, 4)(rng);
                    do {
                        mode2_fan2 = std::uniform_int_distribution<int>(0, 4)(rng);
                    } while (mode2_fan2 == mode2_fan1);
                    power_rune->SetFanState(mode2_fan1, 1);
                    power_rune->SetFanState(mode2_fan2, 1);
                    mode2_phase = Mode2Phase::WAIT_1;
                }
            }

            // 更新模式2文字纹理
            char mode2_text[256];
            std::snprintf(mode2_text, sizeof(mode2_text),
                "Mode 2 | a=%.4f  w=%.4f  b=%.4f  ratio=%.1f  speed=%.2f rad/s",
                mode2_a, mode2_omega, mode2_b, mode2_ratio, rune_roll_speed);
            if (mode2_text_tex) SDL_DestroyTexture(mode2_text_tex);
            mode2_text_tex = RenderTextToTexture(renderer, mode2_text,
                { 0.0f, 1.0f, 0.0f, 1.0f }, 1.5, 3);
            if (mode2_text_tex) {
                float tw, th;
                SDL_GetTextureSize(mode2_text_tex, &tw, &th);
                mode2_text_w = (int)tw;
                mode2_text_h = (int)th;
            }
        }

        flowing_arrow_offset += dt * flowing_arrow_speed;
        flowing_arrow_offset = flowing_arrow_offset - std::floor(flowing_arrow_offset);
        for (int i = 0; i < 5; i++) {
            power_rune -> SetFlowingArrowOffset(i, flowing_arrow_offset);
        }

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

        // 更新相机位姿文字纹理（始终显示，所有模式）
        {
            char camera_text[256];
            std::snprintf(camera_text, sizeof(camera_text),
                "Pos: (%.2f, %.2f, %.2f)  Yaw: %.2f  Pitch: %.2f  Roll: %.2f",
                camera_pose.position.x, camera_pose.position.z, -camera_pose.position.y,
                -camera_pose.yaw, camera_pose.pitch, -camera_pose.roll);
            if (camera_text_tex) SDL_DestroyTexture(camera_text_tex);
            camera_text_tex = RenderTextToTexture(renderer, camera_text,
                { 1.0f, 1.0f, 1.0f, 1.0f }, 1.5, 3);
            if (camera_text_tex) {
                float tw, th;
                SDL_GetTextureSize(camera_text_tex, &tw, &th);
                camera_text_w = (int)tw;
                camera_text_h = (int)th;
            }
        }

        // 更新所有场景节点的世界变换矩阵
        scene.UpdateAllTransforms();

        // =============================================================
        // 渲染步骤：离屏渲染 → 覆盖层 → 截图 → 显示
        // =============================================================

        // ---- Step 1: 开始离屏渲染 ----
        BeginOffscreenRender(renderer, offscreen);

        // ---- Step 1.5: 关键点计算 ----
        std::vector<std::pair<KeypointExtraInfos, std::vector<KeypointProjection>>> keypoints;
        if (show_keypoints) keypoints = power_rune -> getShownKeypoints(intrinsics, distortion, camera_pose);

        // ---- Step 2: 渲染场景节点 ----
        scene.RenderAll(renderer, intrinsics, distortion, camera_pose, keypoints);

        // ---- Step 3: 绘制十字丝 ----
        DrawCrosshair(renderer, (float)intrinsics.cx, (float)intrinsics.cy);

        // ---- Step 3.6: 绘制相机位姿文字（右下角，最后一行） ----
        if (camera_text_tex) {
            SDL_FRect text_rect = {
                (float)LOGICAL_WIDTH - (float)camera_text_w - 20.0f,
                (float)LOGICAL_HEIGHT - (float)camera_text_h - 20.0f,
                (float)camera_text_w,
                (float)camera_text_h
            };
            SDL_RenderTexture(renderer, camera_text_tex, nullptr, &text_rect);
        }

        // ---- Step 3.5: 绘制模式2文字（右下角，倒数第二行） ----
        if (current_mode == 2 && mode2_text_tex) {
            float extra_offset = camera_text_tex ? (float)camera_text_h + 10.0f : 0.0f;
            SDL_FRect text_rect = {
                (float)LOGICAL_WIDTH - (float)mode2_text_w - 20.0f,
                (float)LOGICAL_HEIGHT - (float)mode2_text_h - 20.0f - extra_offset,
                (float)mode2_text_w,
                (float)mode2_text_h
            };
            SDL_RenderTexture(renderer, mode2_text_tex, nullptr, &text_rect);
        }

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
                // 合成视频帧：背景色 + 缩放离屏内容 = 最终显示画面
                // 与 PresentOffscreenToWindow 的逻辑一致（但不加 letterbox，直接填充整个纹理）
                SDL_Texture* prev_target = SDL_GetRenderTarget(renderer);
                SDL_SetRenderTarget(renderer, video_compose_tex);

                // 填充背景色
                SDL_SetRenderDrawColor(renderer,
                    (uint8_t)(video_bg_color.r * 255),
                    (uint8_t)(video_bg_color.g * 255),
                    (uint8_t)(video_bg_color.b * 255),
                    (uint8_t)(video_bg_color.a * 255));
                SDL_RenderClear(renderer);

                // 将离屏内容缩放到整个视频纹理
                float scale = SDL_min(
                    (float)VID_WIDTH / (float)LOGICAL_WIDTH,
                    (float)VID_HEIGHT / (float)LOGICAL_HEIGHT);
                float dst_w = (float)LOGICAL_WIDTH * scale;
                float dst_h = (float)LOGICAL_HEIGHT * scale;
                SDL_FRect dst_rect = {
                    ((float)VID_WIDTH - dst_w) / 2.0f,
                    ((float)VID_HEIGHT - dst_h) / 2.0f,
                    dst_w, dst_h
                };
                SDL_RenderTexture(renderer, offscreen, nullptr, &dst_rect);

                // 读取合成后的像素
                SDL_Surface* frame_surface = SDL_RenderReadPixels(renderer, nullptr);
                SDL_SetRenderTarget(renderer, prev_target);

                if (frame_surface) {
                    // 转换为 OpenCV Mat (RGBA → BGR)
                    cv::Mat rgba_mat(frame_surface->h, frame_surface->w, CV_8UC4,
                                     frame_surface->pixels, static_cast<size_t>(frame_surface->pitch));
                    cv::Mat bgr_mat;
                    cv::cvtColor(rgba_mat, bgr_mat, cv::COLOR_RGBA2BGR);
                    // 异步写入（丢弃模式：队列满时丢弃该帧）
                    bool write_ok = video_writer->writeFrame(bgr_mat, true);
                    video_frame_index = expected_frame_index;

                    // 记录相机位姿到 txt 文件（仅在写入成功时记录）
                    if (write_ok) {
                        if (pose_file.is_open()) {
                            double dt_pose;
                            auto now_time = std::chrono::steady_clock::now();
                            if (first_video_frame) {
                                dt_pose = 0.0;
                                first_video_frame = false;
                            } else {
                                dt_pose = std::chrono::duration<double>(now_time - last_video_write_time).count();
                            }
                            last_video_write_time = now_time;
                            pose_file << pose_frame_index << " "
                                      << dt_pose << " "
                                      << camera_pose.position.x << " "
                                      << camera_pose.position.z << " "
                                      << -camera_pose.position.y << " "
                                      << -camera_pose.yaw << " "
                                      << camera_pose.pitch << " "
                                      << -camera_pose.roll << "\n";
                        }
                        pose_frame_index++;
                    }
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

    // ---------- 7. 清理视频写入器 ----------
    if (video_writer) {
        SDL_Log("Closing video writer...");
        video_writer->close();
        video_writer.reset();
    }
    // 关闭相机位姿记录文件
    if (pose_file.is_open()) {
        pose_file.close();
        SDL_Log("Pose file closed.");
    }

    // ---------- 8. 计算视频对应的相机参数 ----------
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

    // ---------- 9. 清理 ----------
    if (mode2_text_tex) {
        SDL_DestroyTexture(mode2_text_tex);
    }
    if (camera_text_tex) {
        SDL_DestroyTexture(camera_text_tex);
    }
    if (mouse_grabbed) {
        SDL_SetWindowRelativeMouseMode(window, false);
    }
    power_rune.reset();
    if (video_compose_tex) {
        SDL_DestroyTexture(video_compose_tex);
    }
    SDL_DestroyTexture(offscreen);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}