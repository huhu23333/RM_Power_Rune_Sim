#ifndef SCENE_NODE_H
#define SCENE_NODE_H

#include <camera_projection.h>
#include <SDL3/SDL.h>

#include <vector>
#include <memory>
#include <string>

// #define SORT_BY_TRIANGLES

#ifdef SORT_BY_TRIANGLES
const double SUBDIV_SCALE = 0.005;
const int SUBDIV_MAXNUM = 16;
#else
const double SUBDIV_SCALE = 0.005;
const int SUBDIV_MAXNUM = 16;
#endif

void EulerToMatrix(double yaw, double pitch, double roll, double rot[3][3]);
void MultiplyMatrix(const double a[3][3], const double b[3][3], double out[3][3]);
void InverseRotationMatrix(const double r[3][3], double inv[3][3]);

// 相机位姿（位置 + 欧拉角 Yaw/Pitch/Roll，顺序：Yaw → Pitch → Roll）
struct CameraPose
{
    Point3D position{0.0, 0.0, 0.0};
    double yaw{0.0};    // 绕 Y 轴旋转（弧度）
    double pitch{0.0};  // 绕 X 轴旋转（弧度）
    double roll{0.0};   // 绕 Z 轴旋转（弧度）

    // 计算世界 → 相机的旋转矩阵（3x3）
    void GetWorldToCameraMatrix(double rot[3][3]) const;

    // 便捷函数：将世界点变换到相机坐标系
    Point3D WorldToCamera(const Point3D& world_pt) const;
};

// -----------------------------------------------------------------------------
// 附加纹理信息（用于关键点渲染时在外部管理的附加纹理）
// -----------------------------------------------------------------------------
struct ExtraTextureInfo
{
    SDL_Texture* texture = nullptr;
    float offset_x = 0.0f;
    float offset_y = 0.0f;
};

// -----------------------------------------------------------------------------
// 三维变换：位移 + 旋转（内禀 ZYX 顺序）
// -----------------------------------------------------------------------------
struct Transform3D
{
    double tx = 0.0, ty = 0.0, tz = 0.0;
    double yaw   = 0.0;
    double pitch = 0.0;
    double roll  = 0.0;

    Point3D ToParent(const Point3D& local) const;
    Point3D FromParent(const Point3D& parent) const;
};

// -----------------------------------------------------------------------------
// 场景节点基类
// -----------------------------------------------------------------------------
class SceneNode;
using SceneNodePtr = std::shared_ptr<SceneNode>;

class SceneNode
{
public:
    SceneNode();
    virtual ~SceneNode();

    void SetParent(SceneNode* parent);
    SceneNode* GetParent() const;
    // void AddChild(SceneNode* child);
    const std::vector<SceneNode*>& GetChildren() const;

    void SetLocalTransform(const Transform3D& t);
    const Transform3D& GetLocalTransform() const;
    Transform3D& GetLocalTransform();

    void SetLocalPosition(double x, double y, double z);
    void SetLocalRotation(double yaw, double pitch, double roll);

    // 世界变换相关（直接通过矩阵乘法计算）
    Point3D LocalToWorld(const Point3D& local_pt) const;
    Point3D WorldToLocal(const Point3D& world_pt) const;
    Point3D ChildLocalToWorld(const Point3D& child_local, const SceneNode* child) const;

    // 根据 m_local 计算局部矩阵
    bool UpdateLocalMatrix();
    // 更新世界变换（递归）
    void UpdateWorldTransform(const SceneNode* parent = nullptr, bool parent_updated = false);

    // 标记当前节点的局部变换已改变
    void MarkTransformDirty();

    virtual void Render(SDL_Renderer* renderer,
                        const CameraIntrinsics& intrinsics,
                        const DistortionCoefficients& distortion,
                        const CameraPose& camera_pose) {}

protected:
    SceneNode* m_parent = nullptr;
    std::vector<SceneNode*> m_children;
    Transform3D m_local;                 // 原始欧拉角/平移（用户接口）

    // 矩阵缓存（从 m_local 计算得到）
    double m_local_rot[3][3];    // 局部旋转矩阵
    double m_local_trans[3];     // 局部平移向量

    // 世界变换矩阵（局部 → 世界）
    double m_world_rot[3][3];    // 世界旋转矩阵
    double m_world_trans[3];     // 世界平移向量

    bool m_transform_dirty;      // 局部矩阵是否需要重新计算
};

// -----------------------------------------------------------------------------
// 结构体声明
// -----------------------------------------------------------------------------
struct WorldVertex {
    Point3D pos;
    float u, v;
};

struct TriIndices {
    size_t i, j, k;
};

struct RenderFace {
    std::vector<WorldVertex> world_verts;
    std::vector<TriIndices> world_verts_indices;
    SDL_Texture* texture = nullptr;
    SDL_FColor color = { 1.0f, 1.0f, 1.0f, 1.0f };
};

// 工具函数
void BuildFaceTriangles(RenderFace& face,
                        double cx, double cy, double cz,
                        double width, double height,
                        float uv_l, float uv_t,
                        float uv_r, float uv_b);

Point3D WorldToCameraTransform(const Point3D& world_pt, const Point3D& cam_pos,
                               const double cam_rot[3][3]);

void DrawFilledCircle(SDL_Renderer* renderer, float cx, float cy, float radius, SDL_FColor color);

// -----------------------------------------------------------------------------
// 关键点
// -----------------------------------------------------------------------------
struct Keypoint
{
    int index;
    double pixel_x;
    double pixel_y;
};

// 关键点投影结果（世界坐标、相机坐标、屏幕像素坐标及有效性）
struct KeypointProjection {
    Point3D world_pt;   // 世界坐标
    Point3D cam_pt;     // 相机坐标系坐标
    Point2D screen_pt;  // 投影后的屏幕坐标（像素）
    bool   valid;       // 是否在相机前方且投影有效
};

// -----------------------------------------------------------------------------
// 图像节点类
// -----------------------------------------------------------------------------
class ImageNode : public SceneNode
{
public:
    ImageNode();
    ~ImageNode() override;

    void SetTexture(SDL_Texture* tex, int w, int h);
    void SetTextureFromMemory(SDL_Renderer* renderer,
                               const unsigned char* pixel_data,
                               int width, int height,
                               SDL_PixelFormat format = SDL_PIXELFORMAT_RGBA32);

    void SetDisplaySize(double width, double height);

    void SetAlpha(float alpha);
    float GetAlpha() const;

    void SetTextureOffset(float offset_x, float offset_y);
    float GetTextureOffsetX() const;
    float GetTextureOffsetY() const;

    const RenderFace& GetFace() const;
    RenderFace& GetFace();

    SDL_Texture* GetTexture() const;
    int GetTexWidth() const;
    int GetTexHeight() const;

    void SetKeypoints(const std::vector<Keypoint>& kps);
    const std::vector<Keypoint>& GetKeypoints() const;
    Point3D GetKeypointWorldPos(size_t index) const;

    void RenderKeypoints(
        SDL_Renderer* renderer,
        const CameraIntrinsics& intrinsics,
        const DistortionCoefficients& distortion,
        const std::vector<KeypointProjection>& projections,
        const std::vector<std::vector<ExtraTextureInfo>>& all_extra_textures,
        SDL_FColor kp_color_dot = { 0.0f, 1.0f, 0.0f, 1.0f }) const;

    void Render(SDL_Renderer* renderer,
                const CameraIntrinsics& intrinsics,
                const DistortionCoefficients& distortion,
                const CameraPose& camera_pose) override;
    
    void SetRenderPriority(int render_priority);
    int getRenderPriority() const;


    /**
     * @brief 计算一个 ImageNode 中所有关键点的投影数据
     *
     * @param node         目标 ImageNode
     * @param intrinsics   相机内参
     * @param distortion   畸变系数
     * @param camera_pose  相机位姿
     * @param out_projections  输出的投影结果，顺序与 node.GetKeypoints() 一致
     */
    void ComputeKeypointProjections(
        const CameraIntrinsics& intrinsics,
        const DistortionCoefficients& distortion,
        const CameraPose& camera_pose,
        std::vector<KeypointProjection>& out_projections);

    // 设置显示裁剪矩形（比例值，相对于节点的局部坐标范围）
    // 参数范围 [0, 1]，且 min < max。例如 (0.2f, 0.8f, 0.2f, 0.8f) 表示只显示中央 60% 区域。
    // 传入 (0,1,0,1) 可恢复全范围显示。
    void SetDisplayClip(double min_x, double max_x, double min_y, double max_y);
    // 获取当前裁剪矩形
    void GetDisplayClip(double& min_x, double& max_x, double& min_y, double& max_y) const;

protected:
    void UpdateFaces();

private:
    SDL_Texture* m_texture = nullptr;
    bool m_owns_texture = false;
    int m_tex_width = 0;
    int m_tex_height = 0;
    double m_display_width = 2.0;
    double m_display_height = 2.0;
    float m_alpha = 1.0f;
    float m_offset_x = 0.0f;
    float m_offset_y = 0.0f;
    RenderFace m_face;
    std::vector<Keypoint> m_keypoints;
    int m_render_priority = 0;
    
    double m_clip_min_x = 0.0;
    double m_clip_max_x = 1.0;
    double m_clip_min_y = 0.0;
    double m_clip_max_y = 1.0;
};

// -----------------------------------------------------------------------------
// 场景管理类
// -----------------------------------------------------------------------------
class Scene
{
public:
    Scene();
    ~Scene();

    SceneNode* AddNode(SceneNodePtr node);
    std::vector<SceneNode*> GetRootNodes() const;

    // 更新所有节点的世界变换（基于脏标记）
    void UpdateAllTransforms();

    void RenderAll(SDL_Renderer* renderer,
                   const CameraIntrinsics& intrinsics,
                   const DistortionCoefficients& distortion,
                   const CameraPose& camera_pose) const;

    const std::vector<SceneNodePtr>& GetAllNodes() const;

private:
    std::vector<SceneNodePtr> m_nodes;
};

#endif // SCENE_NODE_H
