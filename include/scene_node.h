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
const double SUBDIV_SCALE = 0.0005;
const int SUBDIV_MAXNUM = 64;
#endif

void ComputeWorldToCameraMatrix(double yaw, double pitch, double roll, double rot[3][3]);
void EulerToMatrix(double yaw, double pitch, double roll, double rot[3][3]);
void MultiplyMatrix(const double a[3][3], const double b[3][3], double out[3][3]);
void InverseRotationMatrix(const double r[3][3], double inv[3][3]);

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
                        const Point3D& cam_pos,
                        double cam_yaw, double cam_pitch, double cam_roll) {}

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

struct TargetFaceInfo {
    double center_x, center_y, center_z;
    double half_width, half_height;
    double width, height;
};

Point3D KeypointPixelToWorld(const Keypoint& kp, const TargetFaceInfo& face,
                              int tex_w, int tex_h);

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

    void RenderKeypoints(SDL_Renderer* renderer,
                          const CameraIntrinsics& intrinsics,
                          const DistortionCoefficients& distortion,
                          const Point3D& cam_pos,
                          double cam_yaw, double cam_pitch, double cam_roll,
                          const std::vector<std::vector<ExtraTextureInfo>>& all_extra_textures,
                          SDL_FColor kp_color_dot = { 0.0f, 1.0f, 0.0f, 1.0f }) const;

    void Render(SDL_Renderer* renderer,
                const CameraIntrinsics& intrinsics,
                const DistortionCoefficients& distortion,
                const Point3D& cam_pos,
                double cam_yaw, double cam_pitch, double cam_roll) override;
    
    void SetRenderPriority(int render_priority);
    int getRenderPriority() const;

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
                   const Point3D& cam_pos,
                   double cam_yaw, double cam_pitch, double cam_roll) const;

    const std::vector<SceneNodePtr>& GetAllNodes() const;

private:
    std::vector<SceneNodePtr> m_nodes;
};

#endif // SCENE_NODE_H
