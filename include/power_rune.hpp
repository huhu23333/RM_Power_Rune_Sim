#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <camera_projection.h>
#include <scene_node.h>
#include <file_utils.h>
#include <render_utils.h>

#include <cmath>

struct fan_node_group_t {
    SceneNode* fan_node;
    ImageNode* fan_background_node;
    ImageNode* fan_light_node;
    ImageNode* target_node;
    ImageNode* flowing_arrow_node;
    ImageNode* fan_small_activating;
    ImageNode* fan_big_activating_inner;
    ImageNode* fan_big_activating_outer;

    SceneNode* sketchy_baffle_node;
    SceneNode* sketchy_baffle_oblique_node;
    SceneNode* sketchy_baffle_front_node;
    SceneNode* sketchy_baffle_left_side_node;
    SceneNode* sketchy_baffle_right_side_node;
    SceneNode* sketchy_baffle_behind_node;
};

class PowerRune {
public:
    TextureInfo center_R_tex_info;
    TextureInfo target_tex_info;
    TextureInfo flowing_arrow_tex_info;
    TextureInfo fan_background_tex_info;
    TextureInfo fan_light_tex_info;
    TextureInfo rectangle_tex_info;
    TextureInfo triangle_tex_info;
    TextureInfo fan_small_activating_tex_info;
    TextureInfo fan_big_activating_inner_tex_info;
    TextureInfo fan_big_activating_outer_tex_info;
    std::vector<Keypoint> target_keypoints;
    std::vector<Keypoint> flowing_arrow_keypoints;
    SceneNode* rune_base_node;
    ImageNode* front_center_R_node;
    SceneNode* front_fan_rotation_center_node;
    std::vector<fan_node_group_t> front_fan_node_groups;
    SceneNode* sketchy_support_node;
    ImageNode* sketchy_support_horizontal_node;
    ImageNode* sketchy_support_vertical_left_node;
    ImageNode* sketchy_support_vertical_right_node;
    SceneNode* behind_fan_rotation_center_node;
    std::vector<fan_node_group_t> behind_fan_node_groups;

    PowerRune(SDL_Renderer* renderer, Scene& scene) {
        // ---------- 加载纹理 ----------
        center_R_tex_info = LoadTextureFromPNG(renderer, "images/results/center_R.png");
        target_tex_info = LoadTextureFromPNG(renderer, "images/results/target.png");
        flowing_arrow_tex_info = LoadTextureFromPNG(renderer, "images/results/flowing_arrow.png");
        fan_background_tex_info = LoadTextureFromPNG(renderer, "images/results/fan_background.png");
        fan_light_tex_info = LoadTextureFromPNG(renderer, "images/results/fan_light.png");
        rectangle_tex_info = LoadTextureFromPNG(renderer, "images/results/rectangle.png");
        triangle_tex_info = LoadTextureFromPNG(renderer, "images/results/triangle.png");
        fan_small_activating_tex_info = LoadTextureFromPNG(renderer, "images/results/fan_small_activating.png");
        fan_big_activating_inner_tex_info = LoadTextureFromPNG(renderer, "images/results/fan_big_activating_inner.png");
        fan_big_activating_outer_tex_info = LoadTextureFromPNG(renderer, "images/results/fan_big_activating_outer.png");

        // ---------- 读取关键点文件 ----------
        target_keypoints = LoadKeypointsFromFile("images/results/target.txt");
        flowing_arrow_keypoints = LoadKeypointsFromFile("images/results/flowing_arrow.txt");

        // ---------- 构建场景节点系统 ----------
        // 中心节点
        rune_base_node = CreateSceneNode(scene, 0.0, 0.0, 3.0, nullptr);
        front_center_R_node = CreateImageNode(scene,
                                            center_R_tex_info.texture, center_R_tex_info.width, center_R_tex_info.height,
                                            0.106, 0.106,
                                            0.0, 0.0, -0.3328-0.1664,
                                            1.0f,
                                            {}, rune_base_node, 3);
        // 前方扇叶节点
        front_fan_rotation_center_node = CreateSceneNode(scene, 0.0, 0.0, -0.3328, rune_base_node);
        front_fan_node_groups.resize(5);
        for (int i = 0; i < 5; i += 1) {
            double relative_rotate_rad = M_PI * 2.0 / 5.0 * i;
            auto& fan_node_group = front_fan_node_groups[i];
            fan_node_group.fan_node = CreateSceneNode(scene, 0.0, 0.0, 0.0, front_fan_rotation_center_node);
            fan_node_group.fan_node -> SetLocalRotation(0.0, 0.0, relative_rotate_rad);
            fan_node_group.fan_background_node = CreateImageNode(scene,
                                                fan_background_tex_info.texture, fan_background_tex_info.width, fan_background_tex_info.height,
                                                0.4171, 0.7455,
                                                0.0, -0.1543-0.7455/2.0, 0.0,
                                                1.0f,
                                                {}, fan_node_group.fan_node, 0);
            fan_node_group.fan_light_node = CreateImageNode(scene,
                                                fan_light_tex_info.texture, fan_light_tex_info.width, fan_light_tex_info.height,
                                                0.4171, 0.7455,
                                                0.0, -0.1543-0.7455/2.0, 0.0,
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
                                                flowing_arrow_keypoints, fan_node_group.fan_node, 1);
            fan_node_group.fan_small_activating = CreateImageNode(scene,
                                                fan_small_activating_tex_info.texture, fan_small_activating_tex_info.width, fan_small_activating_tex_info.height,
                                                0.4171, 0.7455,
                                                0.0, -0.1543-0.7455/2.0, 0.0,
                                                1.0f,
                                                {}, fan_node_group.fan_node, 1);
            fan_node_group.fan_big_activating_inner = CreateImageNode(scene,
                                                fan_big_activating_inner_tex_info.texture, fan_big_activating_inner_tex_info.width, fan_big_activating_inner_tex_info.height,
                                                0.4171, 0.7455,
                                                0.0, -0.1543-0.7455/2.0, 0.0,
                                                1.0f,
                                                {}, fan_node_group.fan_node, 1);
            fan_node_group.fan_big_activating_outer = CreateImageNode(scene,
                                                fan_big_activating_outer_tex_info.texture, fan_big_activating_outer_tex_info.width, fan_big_activating_outer_tex_info.height,
                                                0.4171, 0.7455,
                                                0.0, -0.1543-0.7455/2.0, 0.0,
                                                1.0f,
                                                {}, fan_node_group.fan_node, 1);
            // 挡板节点
            fan_node_group.sketchy_baffle_node = CreateSceneNode(scene, 0.0, 0.0, 0.0, fan_node_group.fan_node);
            fan_node_group.sketchy_baffle_oblique_node = CreateSceneNode(scene, 0.0, 0.0, 0.0, fan_node_group.sketchy_baffle_node);
            fan_node_group.sketchy_baffle_oblique_node -> SetLocalRotation(0.0, 0.0,  M_PI * 2.0 / 5.0 / 2.0);
            fan_node_group.sketchy_baffle_front_node = CreateImageNode(scene,
                                                rectangle_tex_info.texture, rectangle_tex_info.width, rectangle_tex_info.height,
                                                0.0675, 0.3512,
                                                0.0, -0.3093/2.0, -0.1664/2.0,
                                                1.0f,
                                                {}, fan_node_group.sketchy_baffle_oblique_node, 2);
            fan_node_group.sketchy_baffle_front_node -> SetLocalRotation(0.0, 0.49357497674, 0.0);
            fan_node_group.sketchy_baffle_left_side_node = CreateImageNode(scene,
                                                triangle_tex_info.texture, triangle_tex_info.width, triangle_tex_info.height,
                                                0.3093, 0.1664,
                                                -0.0675/2.0, -0.3093/2.0, -0.1664/2.0,
                                                1.0f,
                                                {}, fan_node_group.sketchy_baffle_oblique_node, 2);
            fan_node_group.sketchy_baffle_left_side_node -> SetLocalRotation(M_PI/2.0, 0.0, M_PI/2.0);
            fan_node_group.sketchy_baffle_right_side_node = CreateImageNode(scene,
                                                triangle_tex_info.texture, triangle_tex_info.width, triangle_tex_info.height,
                                                0.3093, 0.1664,
                                                0.0675/2.0, -0.3093/2.0, -0.1664/2.0,
                                                1.0f,
                                                {}, fan_node_group.sketchy_baffle_oblique_node, 2);
            fan_node_group.sketchy_baffle_right_side_node -> SetLocalRotation(M_PI/2.0, 0.0, M_PI/2.0);
            fan_node_group.sketchy_baffle_behind_node = CreateImageNode(scene,
                                                rectangle_tex_info.texture, rectangle_tex_info.width, rectangle_tex_info.height,
                                                0.145, 0.1543,
                                                0.0, -0.1543/2.0, 0.0,
                                                1.0f,
                                                {}, fan_node_group.sketchy_baffle_node, 2);
        }
        // 支架节点
        sketchy_support_node = CreateSceneNode(scene, 0.0, 0.0, 0.0, rune_base_node);
        sketchy_support_horizontal_node = CreateImageNode(scene,
                                            rectangle_tex_info.texture, rectangle_tex_info.width, rectangle_tex_info.height,
                                            2.06, 0.175,
                                            0.0, 0.0, 0.0,
                                            1.0f,
                                            {}, sketchy_support_node, 0);
        sketchy_support_vertical_left_node = CreateImageNode(scene,
                                            rectangle_tex_info.texture, rectangle_tex_info.width, rectangle_tex_info.height,
                                            0.175, 2.3,
                                            -2.06/2.0-0.175/2.0, 2.3/2.0-0.175/2.0, 0.0,
                                            1.0f,
                                            {}, sketchy_support_node, 0);
        sketchy_support_vertical_right_node = CreateImageNode(scene,
                                            rectangle_tex_info.texture, rectangle_tex_info.width, rectangle_tex_info.height,
                                            0.175, 2.3,
                                            2.06/2.0+0.175/2.0, 2.3/2.0-0.175/2.0, 0.0,
                                            1.0f,
                                            {}, sketchy_support_node, 0);
        // 后方扇叶节点
        behind_fan_rotation_center_node = CreateSceneNode(scene, 0.0, 0.0, 0.3328, rune_base_node);
        behind_fan_rotation_center_node -> SetLocalRotation(M_PI, 0.0, 0.0);
        behind_fan_node_groups.resize(5);
        for (int i = 0; i < 5; i += 1) {
            double relative_rotate_rad = M_PI * 2.0 / 5.0 * i;
            auto& fan_node_group = behind_fan_node_groups[i];
            fan_node_group.fan_node = CreateSceneNode(scene, 0.0, 0.0, 0.0, behind_fan_rotation_center_node);
            fan_node_group.fan_node -> SetLocalRotation(0.0, 0.0, relative_rotate_rad);
            fan_node_group.fan_background_node = CreateImageNode(scene,
                                                fan_background_tex_info.texture, fan_background_tex_info.width, fan_background_tex_info.height,
                                                0.4171, 0.7455,
                                                0.0, -0.1543-0.7455/2.0, 0.0,
                                                1.0f,
                                                {}, fan_node_group.fan_node, 0);
            fan_node_group.fan_light_node = nullptr;
            fan_node_group.target_node = nullptr;
            fan_node_group.flowing_arrow_node = nullptr;
            fan_node_group.fan_small_activating = nullptr;
            fan_node_group.fan_big_activating_inner = nullptr;
            fan_node_group.fan_big_activating_outer = nullptr;
            // 挡板节点
            fan_node_group.sketchy_baffle_node = CreateSceneNode(scene, 0.0, 0.0, 0.0, fan_node_group.fan_node);
            fan_node_group.sketchy_baffle_oblique_node = CreateSceneNode(scene, 0.0, 0.0, 0.0, fan_node_group.sketchy_baffle_node);
            fan_node_group.sketchy_baffle_oblique_node -> SetLocalRotation(0.0, 0.0,  M_PI * 2.0 / 5.0 / 2.0);
            fan_node_group.sketchy_baffle_front_node = CreateImageNode(scene,
                                                rectangle_tex_info.texture, rectangle_tex_info.width, rectangle_tex_info.height,
                                                0.0675, 0.3512,
                                                0.0, -0.3093/2.0+(0.3512-0.3093)/2.0, 0.0,
                                                1.0f,
                                                {}, fan_node_group.sketchy_baffle_oblique_node, 0);
            fan_node_group.sketchy_baffle_left_side_node = nullptr;
            fan_node_group.sketchy_baffle_right_side_node = nullptr;
            fan_node_group.sketchy_baffle_behind_node = CreateImageNode(scene,
                                                rectangle_tex_info.texture, rectangle_tex_info.width, rectangle_tex_info.height,
                                                0.145, 0.1543,
                                                0.0, -0.1543/2.0, 0.0,
                                                1.0f,
                                                {}, fan_node_group.sketchy_baffle_node, 0);
        }
    }

    ~PowerRune() {
        if (center_R_tex_info.texture) SDL_DestroyTexture(center_R_tex_info.texture);
        if (target_tex_info.texture) SDL_DestroyTexture(target_tex_info.texture);
        if (flowing_arrow_tex_info.texture) SDL_DestroyTexture(flowing_arrow_tex_info.texture);
        if (fan_background_tex_info.texture) SDL_DestroyTexture(fan_background_tex_info.texture);
        if (fan_light_tex_info.texture) SDL_DestroyTexture(fan_light_tex_info.texture);
        if (rectangle_tex_info.texture) SDL_DestroyTexture(rectangle_tex_info.texture);
        if (triangle_tex_info.texture) SDL_DestroyTexture(triangle_tex_info.texture);
        if (fan_small_activating_tex_info.texture) SDL_DestroyTexture(fan_small_activating_tex_info.texture);
        if (fan_big_activating_inner_tex_info.texture) SDL_DestroyTexture(fan_big_activating_inner_tex_info.texture);
        if (fan_big_activating_outer_tex_info.texture) SDL_DestroyTexture(fan_big_activating_outer_tex_info.texture);
    }
};