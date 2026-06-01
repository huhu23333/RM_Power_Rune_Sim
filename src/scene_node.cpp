#include <scene_node.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

void CameraPose::GetWorldToCameraMatrix(double rot[3][3]) const
{
    double cy = std::cos(yaw);
    double sy = std::sin(yaw);
    double cp = std::cos(pitch);
    double sp = std::sin(pitch);
    double cr = std::cos(roll);
    double sr = std::sin(roll);

    // Ry(yaw)  矩阵（绕 Y 轴）
    double Ry[3][3] = {
        { cy, 0.0, -sy },
        { 0.0, 1.0, 0.0 },
        { sy, 0.0,  cy }
    };
    // Rx(pitch) 矩阵（绕 X 轴）
    double Rx[3][3] = {
        { 1.0, 0.0, 0.0 },
        { 0.0,  cp,  sp },
        { 0.0, -sp,  cp }
    };
    // Rz(roll)  矩阵（绕 Z 轴）
    double Rz[3][3] = {
        { cr, -sr, 0.0 },
        { sr,  cr, 0.0 },
        { 0.0, 0.0, 1.0 }
    };

    double temp[3][3];
    MultiplyMatrix(Rx, Ry, temp);    // Rx * Ry
    MultiplyMatrix(Rz, temp, rot);   // Rz * (Rx * Ry)
}

Point3D CameraPose::WorldToCamera(const Point3D& world_pt) const
{
    double rot[3][3];
    GetWorldToCameraMatrix(rot);
    double dx = world_pt.x - position.x;
    double dy = world_pt.y - position.y;
    double dz = world_pt.z - position.z;
    return Point3D{
        rot[0][0] * dx + rot[0][1] * dy + rot[0][2] * dz,
        rot[1][0] * dx + rot[1][1] * dy + rot[1][2] * dz,
        rot[2][0] * dx + rot[2][1] * dy + rot[2][2] * dz
    };
}

// =============================================================================
// 辅助函数：根据欧拉角构建世界→相机旋转矩阵（顺序：Yaw -> Pitch -> Roll）
// =============================================================================
void ComputeWorldToCameraMatrix(double yaw, double pitch, double roll, double rot[3][3])
{
    double cy = std::cos(yaw);
    double sy = std::sin(yaw);
    double cp = std::cos(pitch);
    double sp = std::sin(pitch);
    double cr = std::cos(roll);
    double sr = std::sin(roll);

    // Ry(yaw)  矩阵（绕 Y 轴）
    double Ry[3][3] = {
        { cy, 0.0, -sy },
        { 0.0, 1.0, 0.0 },
        { sy, 0.0,  cy }
    };
    // Rx(pitch) 矩阵（绕 X 轴）
    double Rx[3][3] = {
        { 1.0, 0.0, 0.0 },
        { 0.0,  cp,  sp },
        { 0.0, -sp,  cp }
    };
    // Rz(roll)  矩阵（绕 Z 轴）
    double Rz[3][3] = {
        { cr, -sr, 0.0 },
        { sr,  cr, 0.0 },
        { 0.0, 0.0, 1.0 }
    };

    double temp[3][3];
    // 计算 Rx * Ry
    MultiplyMatrix(Rx, Ry, temp);
    // 计算 Rz * (Rx * Ry) = Rz * Rx * Ry
    MultiplyMatrix(Rz, temp, rot);
}

// =============================================================================
// 辅助函数：欧拉角 → 旋转矩阵（ZYX 顺序，与 ToParent 一致）
// =============================================================================
void EulerToMatrix(double yaw, double pitch, double roll, double rot[3][3])
{
    double cy = std::cos(yaw);
    double sy = std::sin(yaw);
    double cp = std::cos(pitch);
    double sp = std::sin(pitch);
    double cr = std::cos(roll);
    double sr = std::sin(roll);

    // 旋转顺序：R = Ry(yaw) * Rx(pitch') * Rz(roll)，其中 pitch' = -pitch
    // 矩阵元素由原始 ToParent 推导得出
    rot[0][0] =  cr * cy - sr * sp * sy;
    rot[0][1] = -sr * cy - cr * sp * sy;
    rot[0][2] =  cp * sy;

    rot[1][0] =  sr * cp;
    rot[1][1] =  cr * cp;
    rot[1][2] =  sp;

    rot[2][0] = -cr * sy - sr * sp * cy;
    rot[2][1] =  sr * sy - cr * sp * cy;
    rot[2][2] =  cp * cy;
}

// =============================================================================
// 辅助函数：矩阵乘法（3x3 乘以 3x3）
// =============================================================================
void MultiplyMatrix(const double a[3][3], const double b[3][3], double out[3][3])
{
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i][j] = a[i][0] * b[0][j] +
                        a[i][1] * b[1][j] +
                        a[i][2] * b[2][j];
        }
    }
}

// =============================================================================
// 辅助函数：3x3 矩阵求逆（仅适用于旋转矩阵，转置即逆）
// =============================================================================
void InverseRotationMatrix(const double r[3][3], double inv[3][3])
{
    // 旋转矩阵是正交矩阵，逆 = 转置
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            inv[i][j] = r[j][i];
}

// =============================================================================
// Transform3D（保留原有实现，但不再被递归使用，仅作存储）
// =============================================================================
Point3D Transform3D::ToParent(const Point3D& local) const
{
    double dx = local.x;
    double dy = local.y;
    double dz = local.z;

    double cy = std::cos(yaw);
    double sy = std::sin(yaw);
    double cp = std::cos(pitch);
    double sp = std::sin(pitch);
    double cr = std::cos(roll);
    double sr = std::sin(roll);

    double x1 = dx * cr - dy * sr;
    double y1 = dx * sr + dy * cr;
    double z1 = dz;

    double x2 = x1;
    double y2 = y1 * cp + z1 * sp;
    double z2 = -y1 * sp + z1 * cp;

    double x3 = x2 * cy + z2 * sy;
    double y3 = y2;
    double z3 = -x2 * sy + z2 * cy;

    return { x3 + tx, y3 + ty, z3 + tz };
}

Point3D Transform3D::FromParent(const Point3D& parent) const
{
    double dx = parent.x - tx;
    double dy = parent.y - ty;
    double dz = parent.z - tz;

    double cy = std::cos(yaw);
    double sy = std::sin(yaw);
    double cp = std::cos(pitch);
    double sp = std::sin(pitch);
    double cr = std::cos(roll);
    double sr = std::sin(roll);

    double x1 = dx * cy - dz * sy;
    double y1 = dy;
    double z1 = dx * sy + dz * cy;

    double x2 = x1;
    double y2 = y1 * cp - z1 * sp;
    double z2 = y1 * sp + z1 * cp;

    double x3 = x2 * cr + y2 * sr;
    double y3 = -x2 * sr + y2 * cr;
    double z3 = z2;

    return { x3, y3, z3 };
}

// =============================================================================
// SceneNode
// =============================================================================

SceneNode::SceneNode()
{
    // 初始化局部矩阵为单位矩阵
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            m_local_rot[i][j] = (i == j) ? 1.0 : 0.0;
            m_world_rot[i][j] = (i == j) ? 1.0 : 0.0;
        }
        m_local_trans[i] = 0.0;
        m_world_trans[i] = 0.0;
    }
    m_transform_dirty = true;
}

SceneNode::~SceneNode() = default;

void SceneNode::SetParent(SceneNode* parent)
{
    if (m_parent) {
        auto& siblings = m_parent->m_children;
        siblings.erase(std::remove(siblings.begin(), siblings.end(), this), siblings.end());
    }
    m_parent = parent;
    if (parent) {
        parent->m_children.push_back(this);
    }
    // 父节点改变，整个子树的世界变换需要重新计算，标记当前节点及其子节点为脏
    MarkTransformDirty();
}

SceneNode* SceneNode::GetParent() const { return m_parent; }

// void SceneNode::AddChild(SceneNode* child)
// {
//     if (child) {
//         child->SetParent(this);
//     }
// }

const std::vector<SceneNode*>& SceneNode::GetChildren() const { return m_children; }

void SceneNode::SetLocalTransform(const Transform3D& t)
{
    m_local = t;
    MarkTransformDirty();
}

const Transform3D& SceneNode::GetLocalTransform() const { return m_local; }
Transform3D& SceneNode::GetLocalTransform() { return m_local; }

void SceneNode::SetLocalPosition(double x, double y, double z)
{
    m_local.tx = x;
    m_local.ty = y;
    m_local.tz = z;
    MarkTransformDirty();
}

void SceneNode::SetLocalRotation(double yaw, double pitch, double roll)
{
    m_local.yaw = yaw;
    m_local.pitch = pitch;
    m_local.roll = roll;
    MarkTransformDirty();
}

void SceneNode::MarkTransformDirty()
{
    m_transform_dirty = true;
    // 子节点的世界变换依赖于父节点，因此也需要标记（但不必立即递归，更新时会处理）
}

// 根据 m_local 计算局部矩阵
bool SceneNode::UpdateLocalMatrix()
{
    if (!m_transform_dirty) return false;

    // 根据欧拉角计算旋转矩阵
    EulerToMatrix(m_local.yaw, m_local.pitch, m_local.roll, m_local_rot);
    m_local_trans[0] = m_local.tx;
    m_local_trans[1] = m_local.ty;
    m_local_trans[2] = m_local.tz;

    m_transform_dirty = false;

    return true;
}

// 递归更新世界矩阵
void SceneNode::UpdateWorldTransform(const SceneNode* parent, bool parent_updated)
{
    // 先确保当前节点的局部矩阵是最新的
    bool local_matrix_updated = UpdateLocalMatrix();

    bool world_transform_need_to_update = local_matrix_updated || parent_updated;

    if (world_transform_need_to_update) {
        // 计算世界矩阵
        if (parent) {
            // 世界旋转 = 父世界旋转 * 局部旋转
            MultiplyMatrix(parent->m_world_rot, m_local_rot, m_world_rot);
            // 世界平移 = 父世界旋转 * 局部平移 + 父世界平移
            for (int i = 0; i < 3; ++i) {
                double sum = parent->m_world_trans[i];
                for (int j = 0; j < 3; ++j) {
                    sum += parent->m_world_rot[i][j] * m_local_trans[j];
                }
                m_world_trans[i] = sum;
            }
        } else {
            // 根节点：世界矩阵 = 局部矩阵
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    m_world_rot[i][j] = m_local_rot[i][j];
                }
                m_world_trans[i] = m_local_trans[i];
            }
        }
    }

    // 递归更新子节点（子节点总是要重新计算，因为父世界矩阵可能变了）
    for (auto* child : m_children) {
        child->UpdateWorldTransform(this, world_transform_need_to_update);
    }
}

Point3D SceneNode::LocalToWorld(const Point3D& local_pt) const
{
    // 应用世界矩阵
    double x = m_world_rot[0][0] * local_pt.x +
               m_world_rot[0][1] * local_pt.y +
               m_world_rot[0][2] * local_pt.z +
               m_world_trans[0];
    double y = m_world_rot[1][0] * local_pt.x +
               m_world_rot[1][1] * local_pt.y +
               m_world_rot[1][2] * local_pt.z +
               m_world_trans[1];
    double z = m_world_rot[2][0] * local_pt.x +
               m_world_rot[2][1] * local_pt.y +
               m_world_rot[2][2] * local_pt.z +
               m_world_trans[2];
    return { x, y, z };
}

Point3D SceneNode::WorldToLocal(const Point3D& world_pt) const
{
    // 计算世界旋转矩阵的逆（转置）和平移的逆变换
    double inv_rot[3][3];
    InverseRotationMatrix(m_world_rot, inv_rot);

    // 先将点平移到世界坐标系原点
    double dx = world_pt.x - m_world_trans[0];
    double dy = world_pt.y - m_world_trans[1];
    double dz = world_pt.z - m_world_trans[2];

    // 应用逆旋转
    double x = inv_rot[0][0] * dx + inv_rot[0][1] * dy + inv_rot[0][2] * dz;
    double y = inv_rot[1][0] * dx + inv_rot[1][1] * dy + inv_rot[1][2] * dz;
    double z = inv_rot[2][0] * dx + inv_rot[2][1] * dy + inv_rot[2][2] * dz;
    return { x, y, z };
}

Point3D SceneNode::ChildLocalToWorld(const Point3D& child_local, const SceneNode* child) const
{
    // 等价于从 child 的局部坐标变换到世界坐标
    // 先计算 child 的世界矩阵（需要确保 child 的 WorldTransform 已更新）
    // 这里简单调用 child->LocalToWorld
    return child->LocalToWorld(child_local);
}

// =============================================================================
// 工具函数（保持不变）
// =============================================================================

void BuildFaceTriangles(RenderFace& face,
                        double cx, double cy, double cz,
                        double width, double height,
                        float uv_l, float uv_t,
                        float uv_r, float uv_b)
{
    // 与原来相同，省略...
    int segs_w = std::max(1, std::min((int)std::ceil(width / SUBDIV_SCALE), SUBDIV_MAXNUM));
    int segs_h = std::max(1, std::min((int)std::ceil(height / SUBDIV_SCALE), SUBDIV_MAXNUM));
    double hw = width / 2.0, hh = height / 2.0;

    face.world_verts.clear();
    face.world_verts_indices.clear();
    face.world_verts.reserve((segs_w + 1) * (segs_h + 1));
    face.world_verts_indices.reserve(segs_w * segs_h * 2);

    for (int gy = 0; gy < segs_h+1; ++gy) {
        for (int gx = 0; gx < segs_w+1; ++gx) {
            double txl = (double)gx / segs_w;
            double tyb = (double)gy / segs_h;

            double xl = cx - hw + txl * width;
            double yb = cy - hh + tyb * height;

            float ul = uv_l + (uv_r - uv_l) * (float)txl;
            float vb = uv_t + (uv_b - uv_t) * (float)tyb;

            face.world_verts.push_back({ { xl, yb, cz }, ul, vb });
        }
    }

    for (int gy = 0; gy < segs_h; ++gy) {
        for (int gx = 0; gx < segs_w; ++gx) {
            size_t index_0 = gx + (segs_w+1) * gy;
            size_t index_1 = gx + 1 + (segs_w+1) * gy;
            size_t index_2 = gx + (segs_w+1) * (gy+1);
            size_t index_3 = gx + 1 + (segs_w+1) * (gy+1);

            face.world_verts_indices.push_back({index_0, index_1, index_3});
            face.world_verts_indices.push_back({index_0, index_3, index_2});
        }
    }
}

Point3D WorldToCameraTransform(const Point3D& world_pt, const Point3D& cam_pos,
                               const double cam_rot[3][3])
{
    double dx = world_pt.x - cam_pos.x;
    double dy = world_pt.y - cam_pos.y;
    double dz = world_pt.z - cam_pos.z;
    double cx = cam_rot[0][0] * dx + cam_rot[0][1] * dy + cam_rot[0][2] * dz;
    double cy = cam_rot[1][0] * dx + cam_rot[1][1] * dy + cam_rot[1][2] * dz;
    double cz = cam_rot[2][0] * dx + cam_rot[2][1] * dy + cam_rot[2][2] * dz;
    return { cx, cy, cz };
}

void DrawFilledCircle(SDL_Renderer* renderer, float cx, float cy, float radius, SDL_FColor color)
{
    // 与原来相同，省略...
    const int SEGMENTS = 24;
    std::vector<SDL_Vertex> verts;
    verts.reserve((SEGMENTS + 1) * 3);
    SDL_Vertex center = { { cx, cy }, color, { 0.0f, 0.0f } };
    for (int i = 0; i < SEGMENTS; ++i) {
        float a1 = (float)(2.0 * M_PI * i / SEGMENTS);
        float a2 = (float)(2.0 * M_PI * (i + 1) / SEGMENTS);
        float x1 = cx + radius * std::cos(a1);
        float y1 = cy + radius * std::sin(a1);
        float x2 = cx + radius * std::cos(a2);
        float y2 = cy + radius * std::sin(a2);
        verts.push_back(center);
        verts.push_back({ { x1, y1 }, color, { 0.0f, 0.0f } });
        verts.push_back({ { x2, y2 }, color, { 0.0f, 0.0f } });
    }
    SDL_RenderGeometry(renderer, nullptr, verts.data(), (int)verts.size(), nullptr, 0);
}

Point3D KeypointPixelToWorld(const Keypoint& kp, const TargetFaceInfo& face,
                              int tex_w, int tex_h)
{
    double u = kp.pixel_x / tex_w;
    double v = kp.pixel_y / tex_h;
    double wx = face.center_x - face.half_width  + u * face.width;
    double wy = face.center_y - face.half_height + v * face.height;
    double wz = face.center_z;
    return { wx, wy, wz };
}

// =============================================================================
// ImageNode
// =============================================================================

ImageNode::ImageNode() : SceneNode() {}
ImageNode::~ImageNode()
{
    if (m_texture && m_owns_texture) {
        SDL_DestroyTexture(m_texture);
    }
}

void ImageNode::SetTexture(SDL_Texture* tex, int w, int h)
{
    m_texture = tex;
    m_tex_width = w;
    m_tex_height = h;
    if (m_texture) {
        SDL_SetTextureScaleMode(m_texture, SDL_SCALEMODE_LINEAR);
    }
    UpdateFaces();
}

void ImageNode::SetTextureFromMemory(SDL_Renderer* renderer,
                                      const unsigned char* pixel_data,
                                      int width, int height,
                                      SDL_PixelFormat format)
{
    if (m_texture && m_owns_texture) {
        SDL_DestroyTexture(m_texture);
    }
    m_texture = SDL_CreateTexture(renderer, format,
                                   SDL_TEXTUREACCESS_STATIC, width, height);
    if (m_texture) {
        SDL_UpdateTexture(m_texture, nullptr, pixel_data, width * 4);
        m_tex_width = width;
        m_tex_height = height;
        m_owns_texture = true;
        UpdateFaces();
    }
}

void ImageNode::SetDisplaySize(double width, double height)
{
    m_display_width = width;
    m_display_height = height;
    UpdateFaces();
}

void ImageNode::SetAlpha(float alpha) { m_alpha = alpha; }
float ImageNode::GetAlpha() const { return m_alpha; }

void ImageNode::SetTextureOffset(float offset_x, float offset_y)
{
    offset_x = offset_x - std::floor(offset_x);
    offset_y = offset_y - std::floor(offset_y);
    m_offset_x = offset_x;
    m_offset_y = offset_y;
    UpdateFaces();
}

float ImageNode::GetTextureOffsetX() const { return m_offset_x; }
float ImageNode::GetTextureOffsetY() const { return m_offset_y; }

const RenderFace& ImageNode::GetFace() const { return m_face; }
RenderFace& ImageNode::GetFace() { return m_face; }

SDL_Texture* ImageNode::GetTexture() const { return m_texture; }
int ImageNode::GetTexWidth() const { return m_tex_width; }
int ImageNode::GetTexHeight() const { return m_tex_height; }

void ImageNode::SetKeypoints(const std::vector<Keypoint>& kps) { m_keypoints = kps; }
const std::vector<Keypoint>& ImageNode::GetKeypoints() const { return m_keypoints; }

Point3D ImageNode::GetKeypointWorldPos(size_t index) const
{
    if (index >= m_keypoints.size()) return {0,0,0};

    const Keypoint& kp = m_keypoints[index];
    double hw = m_display_width / 2.0;
    double hh = m_display_height / 2.0;

    double u = kp.pixel_x / m_tex_width;
    double v = kp.pixel_y / m_tex_height;
    double wx = -hw + u * m_display_width;
    double wy = -hh + v * m_display_height;
    double wz = 0.0;

    Point3D local_pt = { wx, wy, wz };
    return LocalToWorld(local_pt);
}

void ImageNode::RenderKeypoints(SDL_Renderer* renderer,
                                 const CameraIntrinsics& intrinsics,
                                 const DistortionCoefficients& distortion,
                                 const CameraPose& camera_pose,
                                 const std::vector<std::vector<ExtraTextureInfo>>& all_extra_textures,
                                 SDL_FColor kp_color_dot) const
{
    if (m_keypoints.empty()) return;

    const float MAX_COORD = 1e6f;

    // 预先计算世界 → 相机的旋转矩阵
    double cam_rot[3][3];
    ComputeWorldToCameraMatrix(camera_pose.yaw, camera_pose.pitch, camera_pose.roll, cam_rot);

    for (size_t ki = 0; ki < m_keypoints.size(); ++ki) {
        Point3D world_pt = GetKeypointWorldPos(ki);
        Point3D cam_pt = WorldToCameraTransform(world_pt, camera_pose.position, cam_rot);
        if (cam_pt.z <= 0.001) continue;

        Point2D screen_pt = ProjectPoint(cam_pt, intrinsics, distortion);
        if (std::isnan(screen_pt.x) || std::isnan(screen_pt.y) || 
            std::abs(screen_pt.x) > MAX_COORD || std::abs(screen_pt.y) > MAX_COORD || 
            !screen_pt.valid)
            continue;

        float sx = (float)screen_pt.x;
        float sy = (float)screen_pt.y;

        DrawFilledCircle(renderer, sx, sy, 10.0f, kp_color_dot);

        if (ki < all_extra_textures.size()) {
            for (const auto& extra : all_extra_textures[ki]) {
                if (!extra.texture) continue;
                float tw, th;
                SDL_GetTextureSize(extra.texture, &tw, &th);
                SDL_FRect dst = { extra.offset_x, extra.offset_y, tw, th };
                SDL_RenderTexture(renderer, extra.texture, nullptr, &dst);
            }
        }
    }
}

void ImageNode::UpdateFaces()
{
    if (!m_texture) return;

    double hw = m_display_width / 2.0;
    double hh = m_display_height / 2.0;

    RenderFace face;
    BuildFaceTriangles(face,
                       0, 0, 0,
                       m_display_width, m_display_height,
                       m_offset_x, m_offset_y,
                       m_offset_x + 1.0f, m_offset_y + 1.0f);
    face.texture = m_texture;
    face.color = { 1.0f, 1.0f, 1.0f, m_alpha };
    m_face = std::move(face);
}

void ImageNode::Render(SDL_Renderer* renderer,
                        const CameraIntrinsics& intrinsics,
                        const DistortionCoefficients& distortion,
                        const CameraPose& camera_pose)
{
    std::vector<SDL_Vertex> sf_sdl_verts;
    const float MAX_COORD = 1e6f;

    // 预先计算世界 → 相机的旋转矩阵
    double cam_rot[3][3];
    ComputeWorldToCameraMatrix(camera_pose.yaw, camera_pose.pitch, camera_pose.roll, cam_rot);

    SDL_TextureAddressMode prev_u, prev_v;
    SDL_GetRenderTextureAddressMode(renderer, &prev_u, &prev_v);
    SDL_SetRenderTextureAddressMode(renderer, SDL_TEXTURE_ADDRESS_WRAP, SDL_TEXTURE_ADDRESS_WRAP);

    sf_sdl_verts.reserve(m_face.world_verts_indices.size());
    
    std::vector<Point2DUVD> p2duv_verts;
    for (size_t i = 0; i < m_face.world_verts.size(); i += 1) {
        const WorldVertex& wv0 = m_face.world_verts[i];
        Point3D r0 = LocalToWorld(wv0.pos);
        Point3D c0 = WorldToCameraTransform(r0, camera_pose.position, cam_rot);
        Point2D pp0 = ProjectPoint(c0, intrinsics, distortion);

        p2duv_verts.push_back({pp0, wv0.u, wv0.v});
    }

    for (size_t i = 0; i < m_face.world_verts_indices.size(); i += 1) {
        TriIndices indices = m_face.world_verts_indices[i];

        Point2DUVD& pp0uv = p2duv_verts[indices.i];
        Point2DUVD& pp1uv = p2duv_verts[indices.j];
        Point2DUVD& pp2uv = p2duv_verts[indices.k];

        Point2D& pp0 = pp0uv.point2d;
        Point2D& pp1 = pp1uv.point2d;
        Point2D& pp2 = pp2uv.point2d;

        if (!pp0.valid && !pp1.valid && !pp2.valid)
            continue;
        if (std::isnan(pp0.x) || std::isnan(pp0.y) ||
            std::isnan(pp1.x) || std::isnan(pp1.y) ||
            std::isnan(pp2.x) || std::isnan(pp2.y))
            continue;
        if (std::abs(pp0.x) > MAX_COORD || std::abs(pp0.y) > MAX_COORD ||
            std::abs(pp1.x) > MAX_COORD || std::abs(pp1.y) > MAX_COORD ||
            std::abs(pp2.x) > MAX_COORD || std::abs(pp2.y) > MAX_COORD)
            continue;

        SDL_FColor face_color = m_face.color;
        face_color.a = m_alpha;

        sf_sdl_verts.push_back({ { (float)pp0.x, (float)pp0.y }, face_color, { pp0uv.u, pp0uv.v } });
        sf_sdl_verts.push_back({ { (float)pp1.x, (float)pp1.y }, face_color, { pp1uv.u, pp1uv.v } });
        sf_sdl_verts.push_back({ { (float)pp2.x, (float)pp2.y }, face_color, { pp2uv.u, pp2uv.v } });
    }

    SDL_RenderGeometry(renderer, m_face.texture,
                        sf_sdl_verts.data(), (int)sf_sdl_verts.size(),
                        nullptr, 0);

    SDL_SetRenderTextureAddressMode(renderer, prev_u, prev_v);
}

void ImageNode::SetRenderPriority(int render_priority) {
    m_render_priority = render_priority;
}

int ImageNode::getRenderPriority() const {
    return m_render_priority;
}

// =============================================================================
// Scene
// =============================================================================

Scene::Scene() = default;
Scene::~Scene() = default;

SceneNode* Scene::AddNode(SceneNodePtr node)
{
    SceneNode* ptr = node.get();
    m_nodes.push_back(std::move(node));
    return ptr;
}

std::vector<SceneNode*> Scene::GetRootNodes() const
{
    std::vector<SceneNode*> roots;
    for (auto& node : m_nodes) {
        if (!node->GetParent()) {
            roots.push_back(node.get());
        }
    }
    return roots;
}

void Scene::UpdateAllTransforms()
{
    auto roots = GetRootNodes();
    for (auto* root : roots) {
        root->UpdateWorldTransform(nullptr);
    }
}

#ifdef SORT_BY_TRIANGLES
void Scene::RenderAll(SDL_Renderer* renderer,
                       const CameraIntrinsics& intrinsics,
                       const DistortionCoefficients& distortion,
                       const CameraPose& camera_pose) const
{
    struct SortedTriangle {
        SDL_Texture* texture;
        SDL_Vertex verts[3];
        double distance;        // 三角形中心到相机的距离
        int render_priority;
    };
    std::vector<SortedTriangle> sorted_triangles;

    const float MAX_COORD = 1e6f;

    // 预先计算世界 → 相机的旋转矩阵
    double cam_rot[3][3];
    ComputeWorldToCameraMatrix(camera_pose.yaw, camera_pose.pitch, camera_pose.roll, cam_rot);

    // 设置纹理地址模式（所有三角形共用，只需设置一次）
    SDL_TextureAddressMode prev_u, prev_v;
    SDL_GetRenderTextureAddressMode(renderer, &prev_u, &prev_v);
    SDL_SetRenderTextureAddressMode(renderer, SDL_TEXTURE_ADDRESS_WRAP, SDL_TEXTURE_ADDRESS_WRAP);

    for (const auto& node : m_nodes) {
        const ImageNode* img_node = dynamic_cast<const ImageNode*>(node.get());
        if (!img_node) continue;

        const RenderFace& face = img_node->GetFace();
        const int priority = img_node->getRenderPriority();
        const float alpha = img_node->GetAlpha();

        // 计算所有顶点的屏幕坐标和相机距离
        std::vector<Point2DUVD> p2duv_verts;
        p2duv_verts.reserve(face.world_verts.size());

        for (size_t i = 0; i < face.world_verts.size(); ++i) {
            const WorldVertex& wv = face.world_verts[i];
            Point3D world_pt = img_node->LocalToWorld(wv.pos);
            Point3D cam_pt = WorldToCameraTransform(world_pt, camera_pose.position, cam_rot);
            Point2D screen_pt = ProjectPoint(cam_pt, intrinsics, distortion);
            double distance = std::sqrt(cam_pt.x * cam_pt.x + cam_pt.y * cam_pt.y + cam_pt.z * cam_pt.z);
            p2duv_verts.push_back({ screen_pt, wv.u, wv.v, distance });
        }

        // 遍历所有三角形，构造 SortedTriangle
        for (size_t i = 0; i < face.world_verts_indices.size(); ++i) {
            const TriIndices& idx = face.world_verts_indices[i];
            const Point2DUVD& p0 = p2duv_verts[idx.i];
            const Point2DUVD& p1 = p2duv_verts[idx.j];
            const Point2DUVD& p2 = p2duv_verts[idx.k];

            // 跳过完全无效的三角形
            if (!p0.point2d.valid && !p1.point2d.valid && !p2.point2d.valid)
                continue;
            // 跳过包含 NaN 或超大坐标的三角形
            if (std::isnan(p0.point2d.x) || std::isnan(p0.point2d.y) ||
                std::isnan(p1.point2d.x) || std::isnan(p1.point2d.y) ||
                std::isnan(p2.point2d.x) || std::isnan(p2.point2d.y))
                continue;
            if (std::abs(p0.point2d.x) > MAX_COORD || std::abs(p0.point2d.y) > MAX_COORD ||
                std::abs(p1.point2d.x) > MAX_COORD || std::abs(p1.point2d.y) > MAX_COORD ||
                std::abs(p2.point2d.x) > MAX_COORD || std::abs(p2.point2d.y) > MAX_COORD)
                continue;

            // 三角形中心距离：三个顶点距离的平均值
            double tri_distance = (p0.distance + p1.distance + p2.distance) / 3.0;

            SortedTriangle tri;
            tri.texture = face.texture;
            tri.distance = tri_distance;
            tri.render_priority = priority;

            // 顶点颜色（使用面颜色乘以 alpha）
            SDL_FColor vert_color = face.color;
            vert_color.a = alpha;

            tri.verts[0] = { { (float)p0.point2d.x, (float)p0.point2d.y }, vert_color, { p0.u, p0.v } };
            tri.verts[1] = { { (float)p1.point2d.x, (float)p1.point2d.y }, vert_color, { p1.u, p1.v } };
            tri.verts[2] = { { (float)p2.point2d.x, (float)p2.point2d.y }, vert_color, { p2.u, p2.v } };

            sorted_triangles.push_back(tri);
        }
    }

    // 排序：先按 render_priority 升序（小优先级先渲染），再按距离降序（远的先渲染）
    std::sort(sorted_triangles.begin(), sorted_triangles.end(),
        [](const SortedTriangle& a, const SortedTriangle& b) {
            if (a.render_priority != b.render_priority)
                return a.render_priority < b.render_priority;
            return a.distance > b.distance;   // 从远到近
        });

    // 逐个渲染三角形
    for (const SortedTriangle& tri : sorted_triangles) {
        SDL_RenderGeometry(renderer, tri.texture,
                           tri.verts, 3,
                           nullptr, 0);
    }

    // 恢复纹理地址模式
    SDL_SetRenderTextureAddressMode(renderer, prev_u, prev_v);
}
#else
void Scene::RenderAll(SDL_Renderer* renderer,
                       const CameraIntrinsics& intrinsics,
                       const DistortionCoefficients& distortion,
                       const CameraPose& camera_pose) const
{
    struct SortedFace {
        const RenderFace* face;
        const ImageNode* node;
        std::vector<SDL_Vertex> sdl_verts;
        double cam_distance;
        int render_priority;
    };
    std::vector<SortedFace> sorted_faces;
    const float MAX_COORD = 1e6f;

    // 预先计算世界 → 相机的旋转矩阵
    double cam_rot[3][3];
    ComputeWorldToCameraMatrix(camera_pose.yaw, camera_pose.pitch, camera_pose.roll, cam_rot);

    SDL_TextureAddressMode prev_u, prev_v;
    SDL_GetRenderTextureAddressMode(renderer, &prev_u, &prev_v);
    SDL_SetRenderTextureAddressMode(renderer, SDL_TEXTURE_ADDRESS_WRAP, SDL_TEXTURE_ADDRESS_WRAP);

    for (const auto& node : m_nodes) {
        const ImageNode* img_node = dynamic_cast<const ImageNode*>(node.get());
        if (!img_node) continue;

        auto& face = img_node->GetFace();
        SortedFace sf;
        sf.face = &face;
        sf.node = img_node;
        sf.sdl_verts.reserve(face.world_verts.size());
        double distance_sum = 0.0;
        int visible_tris = 0;
        sf.render_priority = img_node->getRenderPriority();

        std::vector<Point2DUVD> p2duv_verts;
        for (size_t i = 0; i < face.world_verts.size(); i += 1) {
            const WorldVertex& wv0 = face.world_verts[i];
            Point3D r0 = img_node->LocalToWorld(wv0.pos);
            Point3D c0 = WorldToCameraTransform(r0, camera_pose.position, cam_rot);
            Point2D pp0 = ProjectPoint(c0, intrinsics, distortion);

            double distance = std::sqrt(c0.x * c0.x + c0.y * c0.y + c0.z * c0.z);
            p2duv_verts.push_back({pp0, wv0.u, wv0.v, distance});
        }

        for (size_t i = 0; i < face.world_verts_indices.size(); i += 1) {
            TriIndices indices = face.world_verts_indices[i];

            Point2DUVD& pp0uv = p2duv_verts[indices.i];
            Point2DUVD& pp1uv = p2duv_verts[indices.j];
            Point2DUVD& pp2uv = p2duv_verts[indices.k];

            Point2D& pp0 = pp0uv.point2d;
            Point2D& pp1 = pp1uv.point2d;
            Point2D& pp2 = pp2uv.point2d;

            if (!pp0.valid && !pp1.valid && !pp2.valid)
                continue;
            if (std::isnan(pp0.x) || std::isnan(pp0.y) ||
                std::isnan(pp1.x) || std::isnan(pp1.y) ||
                std::isnan(pp2.x) || std::isnan(pp2.y))
                continue;
            if (std::abs(pp0.x) > MAX_COORD || std::abs(pp0.y) > MAX_COORD ||
                std::abs(pp1.x) > MAX_COORD || std::abs(pp1.y) > MAX_COORD ||
                std::abs(pp2.x) > MAX_COORD || std::abs(pp2.y) > MAX_COORD)
                continue;

            SDL_FColor face_color = face.color;
            face_color.a = img_node->GetAlpha();

            sf.sdl_verts.push_back({ { (float)pp0.x, (float)pp0.y }, face_color, { pp0uv.u, pp0uv.v } });
            sf.sdl_verts.push_back({ { (float)pp1.x, (float)pp1.y }, face_color, { pp1uv.u, pp1uv.v } });
            sf.sdl_verts.push_back({ { (float)pp2.x, (float)pp2.y }, face_color, { pp2uv.u, pp2uv.v } });
            
            distance_sum += pp0uv.distance + pp1uv.distance + pp2uv.distance;
            visible_tris += 1;
        }

        if (visible_tris > 0) {
            sf.cam_distance = distance_sum / visible_tris;
            sorted_faces.push_back(std::move(sf));
        }
    }

    std::sort(sorted_faces.begin(), sorted_faces.end(),
        [](const SortedFace& a, const SortedFace& b) { 
            if (a.render_priority < b.render_priority) return true;
            if (a.render_priority > b.render_priority) return false;
            return a.cam_distance > b.cam_distance; 
        });

    for (const auto& sf : sorted_faces) {
        SDL_RenderGeometry(renderer, sf.face->texture,
                           sf.sdl_verts.data(), (int)sf.sdl_verts.size(),
                           nullptr, 0);
    }

    SDL_SetRenderTextureAddressMode(renderer, prev_u, prev_v);
}
#endif

const std::vector<SceneNodePtr>& Scene::GetAllNodes() const { return m_nodes; }
