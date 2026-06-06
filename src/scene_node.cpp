#include <scene_node.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <set>

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

// =============================================================================
// ImageNode
// =============================================================================

ImageNode::ImageNode() : SceneNode() {
    static int image_node_counter = 0;
    m_image_node_index = image_node_counter;
    image_node_counter += 1;
}
ImageNode::~ImageNode()
{
    if (m_texture && m_owns_texture) {
        SDL_DestroyTexture(m_texture);
    }
}

int ImageNode::GetImageNodeIndex() const {
    return m_image_node_index;
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

Point3D ImageNode::GetKeypointWorldPos(const Keypoint& kp) const
{
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

void ImageNode::ComputeKeypointProjections(
    const CameraIntrinsics& intrinsics,
    const DistortionCoefficients& distortion,
    const CameraPose& camera_pose,
    std::vector<KeypointProjection>& out_projections)
{
    const auto& keypoints = GetKeypoints();
    out_projections.clear();
    out_projections.reserve(keypoints.size());

    // 预先计算世界 → 相机的旋转矩阵
    double cam_rot[3][3];
    camera_pose.GetWorldToCameraMatrix(cam_rot);

    bool has_frame_size = (intrinsics.width > 0 && intrinsics.height > 0);

    for (size_t ki = 0; ki < keypoints.size(); ++ki) {
        KeypointProjection proj;
        const Keypoint& kp = m_keypoints[ki];
        proj.world_pt = GetKeypointWorldPos(kp);
        proj.cam_pt = WorldToCameraTransform(proj.world_pt, camera_pose.position, cam_rot);

        if (proj.cam_pt.z <= 0.001) {
            proj.valid = false;
            proj.occluded = true;
            out_projections.push_back(proj);
            continue;
        }

        proj.screen_pt = ProjectPoint(proj.cam_pt, intrinsics, distortion);
        if (!proj.screen_pt.valid) {
            proj.valid = false;
            proj.occluded = true;
        } else {
            if (has_frame_size) {
                if (
                    proj.screen_pt.x < 0 ||
                    proj.screen_pt.x >= intrinsics.width ||
                    proj.screen_pt.y < 0 ||
                    proj.screen_pt.y >= intrinsics.height
                ) {
                    proj.valid = false;
                    proj.occluded = true;
                } else {
                    proj.valid = true;
                    proj.occluded = false;
                }
            }
            else {
                proj.valid = true;
                proj.occluded = false;
            }
        }
        proj.index = kp.index;
        out_projections.push_back(proj);
    }
}

void RenderKeypoints(
    SDL_Renderer* renderer,
    const CameraIntrinsics& intrinsics,
    const DistortionCoefficients& distortion,
    const std::vector<KeypointProjection>& projections,
    const std::vector<std::vector<ExtraTextureInfo>>& all_extra_textures,
    SDL_FColor kp_color_dot, SDL_FColor kp_color_dot_occluded)
{
    for (size_t ki = 0; ki < projections.size(); ++ki) {
        const auto& proj = projections[ki];
        if (!proj.valid) continue;

        float sx = (float)proj.screen_pt.x;
        float sy = (float)proj.screen_pt.y;
        DrawFilledCircle(renderer, sx, sy, 10.0f, proj.occluded ? kp_color_dot_occluded : kp_color_dot);

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

    // 完整显示矩形的半宽半高（局部坐标）
    double full_hw = m_display_width / 2.0;
    double full_hh = m_display_height / 2.0;

    // 是否启用裁剪（范围不是完整 [0,1]）
    bool use_clip = (m_clip_min_x != 0.0 || m_clip_max_x != 1.0 ||
                     m_clip_min_y != 0.0 || m_clip_max_y != 1.0);

    double center_x = 0.0, center_y = 0.0;
    double width = m_display_width;
    double height = m_display_height;
    float uv_left   = m_offset_x;
    float uv_top    = m_offset_y;
    float uv_right  = m_offset_x + 1.0f;
    float uv_bottom = m_offset_y + 1.0f;

    if (use_clip) {
        // 计算裁剪后矩形的局部坐标边界
        double left   = -full_hw + m_clip_min_x * m_display_width;
        double right  = -full_hw + m_clip_max_x * m_display_width;
        double bottom = -full_hh + m_clip_min_y * m_display_height;
        double top    = -full_hh + m_clip_max_y * m_display_height;

        width  = right - left;
        height = top - bottom;
        // 新矩形的中心（局部坐标）
        center_x = (left + right) / 2.0;
        center_y = (bottom + top) / 2.0;

        // 对应的纹理坐标范围
        uv_left   = m_offset_x + (float)m_clip_min_x;
        uv_right  = m_offset_x + (float)m_clip_max_x;
        uv_top    = m_offset_y + (float)m_clip_min_y;
        uv_bottom = m_offset_y + (float)m_clip_max_y;
    }

    // 若宽度或高度为 0，不生成任何三角形（完全透明）
    if (width <= 0.0 || height <= 0.0) {
        m_face.world_verts.clear();
        m_face.world_verts_indices.clear();
        m_face.texture = m_texture;
        m_face.color = { 1.0f, 1.0f, 1.0f, m_alpha };
        return;
    }

    RenderFace face;
    BuildFaceTriangles(face,
                       center_x, center_y, 0.0,   // 使用计算出的中心位置
                       width, height,
                       uv_left, uv_top,
                       uv_right, uv_bottom);
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

    // 预先计算世界 → 相机的旋转矩阵
    double cam_rot[3][3];
    camera_pose.GetWorldToCameraMatrix(cam_rot);

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

void ImageNode::SetDisplayClip(double min_x, double max_x, double min_y, double max_y)
{
    // 钳位到 [0,1]
    min_x = std::clamp(min_x, 0.0, 1.0);
    max_x = std::clamp(max_x, 0.0, 1.0);
    min_y = std::clamp(min_y, 0.0, 1.0);
    max_y = std::clamp(max_y, 0.0, 1.0);

    // 无效范围时不显示
    if (min_x >= max_x || min_y >= max_y) {
        m_clip_min_x = 0.0;
        m_clip_max_x = 0.0;
        m_clip_min_y = 0.0;
        m_clip_max_y = 0.0;
    } else {
        m_clip_min_x = min_x;
        m_clip_max_x = max_x;
        m_clip_min_y = min_y;
        m_clip_max_y = max_y;
    }
    UpdateFaces();
}

void ImageNode::GetDisplayClip(double& min_x, double& max_x, double& min_y, double& max_y) const
{
    min_x = m_clip_min_x;
    max_x = m_clip_max_x;
    min_y = m_clip_min_y;
    max_y = m_clip_max_y;
}

// =============================================================================
// Scene
// =============================================================================

Scene::Scene() = default;
Scene::~Scene() {
    if (m_offscreen_od_1) {
        SDL_DestroyTexture(m_offscreen_od_1);
        SDL_DestroyTexture(m_offscreen_od_2);
    }
}

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

struct KeypointPixelInfos {
    int x;
    int y;
    bool operator<(const KeypointPixelInfos& other) const {
        if (x != other.x) return x < other.x;
        return y < other.y;
    }
    int index_outer;
    int index_inner;
    int image_node_index;
    uint8_t rgba_1[4];
    uint8_t rgba_2[4];
};

void preProcessKeypoints(
    std::vector<std::pair<KeypointExtraInfos, std::vector<KeypointProjection>>>& keypoints,
    const SDL_Texture* o_offscreen,  // 假设 o_offscreen 有 int w, h 成员
    std::set<KeypointPixelInfos>& keypoint_pixel_set)
{
    if (!o_offscreen || o_offscreen->w <= 0 || o_offscreen->h <= 0)
        return;  // 无效参数

    for (size_t i_outer = 0; i_outer < keypoints.size(); ++i_outer) {
        const auto& extra = keypoints[i_outer].first;
        auto& projections = keypoints[i_outer].second;  // 非常量引用，以便修改 occluded

        for (size_t i_inner = 0; i_inner < projections.size(); ++i_inner) {
            auto& proj = projections[i_inner];
            // 只处理有效投影（可根据需要调整）
            if (!proj.valid)
            {
                proj.occluded = true;
                continue;
            }

            // 1. 将 screen_pt 转为整数并截断到 [0, w-1] × [0, h-1]
            int px = static_cast<int>(proj.screen_pt.x);
            int py = static_cast<int>(proj.screen_pt.y);

            if (px < 0 || py < 0 || px >= o_offscreen->w || py >= o_offscreen->h)
            {
                proj.occluded = true;
                continue;
            }

            px = std::clamp(px, 0, o_offscreen->w - 1);
            py = std::clamp(py, 0, o_offscreen->h - 1);

            // 2. 构建候选 KeypointPixelInfos
            KeypointPixelInfos candidate{
                px, py,
                static_cast<int>(i_outer),
                static_cast<int>(i_inner),
                extra.image_node_index
            };

            // 3. 查找是否已有相同坐标的像素点
            auto it = keypoint_pixel_set.find(candidate);
            if (it == keypoint_pixel_set.end()) {
                // 无冲突，直接插入
                keypoint_pixel_set.insert(candidate);
            } else {
                // 有冲突：比较 cam_pt 到原点的距离（平方距离，避免开方）
                const auto& existing_info = *it;
                auto& existing_proj = keypoints[existing_info.index_outer].second[existing_info.index_inner];

                auto squared_distance = [](const Point3D& pt) {
                    return pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
                };
                double dist_existing = squared_distance(existing_proj.cam_pt);
                double dist_new = squared_distance(proj.cam_pt);

                if (dist_new < dist_existing) {
                    // 新点更近 -> 旧点被遮挡，移除旧点，插入新点
                    existing_proj.occluded = true;
                    keypoint_pixel_set.erase(it);
                    keypoint_pixel_set.insert(candidate);
                } else {
                    // 旧点更近或距离相等 → 新点被遮挡
                    proj.occluded = true;
                    // 不插入新点，set 保持不变
                }
            }
        }
    }
}

std::vector<KeypointPixelInfos> getKeypointsByImageNodeIndex(
    const std::set<KeypointPixelInfos>& keypoint_pixel_set,
    int target_image_node_index)
{
    std::vector<KeypointPixelInfos> result;
    for (const auto& info : keypoint_pixel_set) {
        if (info.image_node_index == target_image_node_index) {
            result.push_back(info);
        }
    }
    return result;
}

// #include <iostream>

void Scene::RenderAll(SDL_Renderer* renderer,
                       const CameraIntrinsics& intrinsics,
                       const DistortionCoefficients& distortion,
                       const CameraPose& camera_pose,
                       std::vector<std::pair<KeypointExtraInfos, std::vector<KeypointProjection>>>& keypoints)
{
    bool has_keypoints = false;
    SDL_Texture* o_offscreen = SDL_GetRenderTarget(renderer);; // original_offscreen
    std::set<KeypointPixelInfos> keypoint_pixel_set;
    if (!keypoints.empty()) {
        has_keypoints = true;
        if (!m_offscreen_od_1) {
            m_offscreen_od_1 = SDL_CreateTexture(
                renderer, SDL_PIXELFORMAT_RGBA32,
                SDL_TEXTUREACCESS_TARGET, o_offscreen->w, o_offscreen->h);
            if (!m_offscreen_od_1) {
                SDL_Log("Create m_offscreen_od_1 texture failed: %s", SDL_GetError());
            }
            SDL_SetTextureBlendMode(m_offscreen_od_1, SDL_BLENDMODE_BLEND_PREMULTIPLIED);
            m_offscreen_od_2 = SDL_CreateTexture(
                renderer, SDL_PIXELFORMAT_RGBA32,
                SDL_TEXTUREACCESS_TARGET, o_offscreen->w, o_offscreen->h);
            if (!m_offscreen_od_2) {
                SDL_Log("Create m_offscreen_od_2 texture failed: %s", SDL_GetError());
            }
            SDL_SetTextureBlendMode(m_offscreen_od_2, SDL_BLENDMODE_BLEND_PREMULTIPLIED);
        }

        SDL_SetRenderTarget(renderer, m_offscreen_od_1);
        SDL_RenderClear(renderer);

        SDL_SetRenderTarget(renderer, m_offscreen_od_2);
        SDL_RenderClear(renderer);
        
        SDL_SetRenderTarget(renderer, o_offscreen);

        preProcessKeypoints(keypoints, o_offscreen, keypoint_pixel_set);
    }

    struct SortedFace {
        const RenderFace* face;
        const ImageNode* node;
        std::vector<SDL_Vertex> sdl_verts;
        double cam_distance;
        int render_priority;
        int image_node_index;
    };
    std::vector<SortedFace> sorted_faces;

    // 预先计算世界 → 相机的旋转矩阵
    double cam_rot[3][3];
    camera_pose.GetWorldToCameraMatrix(cam_rot);

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
            sf.image_node_index = img_node -> GetImageNodeIndex();
            sorted_faces.push_back(std::move(sf));
        }
    }

    std::sort(sorted_faces.begin(), sorted_faces.end(),
        [](const SortedFace& a, const SortedFace& b) { 
            if (a.render_priority < b.render_priority) return true;
            if (a.render_priority > b.render_priority) return false;
            return a.cam_distance > b.cam_distance; 
        });

    std::vector<std::pair<SortedFace, std::vector<KeypointPixelInfos>>> sorted_faces_after_has_keypoints;

    bool start_has_keypoints = false;
    for (const auto& sf : sorted_faces) {
        SDL_RenderGeometry(renderer, sf.face->texture,
                           sf.sdl_verts.data(), (int)sf.sdl_verts.size(),
                           nullptr, 0);
        int image_node_index = sf.image_node_index;
        if (has_keypoints) {
            std::vector<KeypointPixelInfos> its_keypoints = getKeypointsByImageNodeIndex(keypoint_pixel_set, image_node_index);
            if (start_has_keypoints || !its_keypoints.empty()) {
                start_has_keypoints = true;
                sorted_faces_after_has_keypoints.push_back({sf, its_keypoints});
            }
        }
    }

    if (has_keypoints) {
        SDL_FColor white_color = {1.0, 1.0, 1.0, 1.0};
        SDL_FColor black_color = {0.0, 0.0, 0.0, 1.0};
        float r_square = 1.0f;
        SDL_SetRenderTarget(renderer, m_offscreen_od_1);
        for (const auto& [sf, its_keypoints] : sorted_faces_after_has_keypoints) {
            SDL_RenderGeometry(renderer, sf.face->texture,
                            sf.sdl_verts.data(), (int)sf.sdl_verts.size(),
                            nullptr, 0);
            for (auto& kpi : its_keypoints) {
                SDL_Vertex verts[6] = {
                    { { (float)kpi.x - r_square, (float)kpi.y - r_square }, black_color, { 0, 0 } },
                    { { (float)kpi.x - r_square, (float)kpi.y + r_square }, black_color, { 0, 0 } },
                    { { (float)kpi.x + r_square, (float)kpi.y + r_square }, black_color, { 0, 0 } },
                    { { (float)kpi.x - r_square, (float)kpi.y - r_square }, black_color, { 0, 0 } },
                    { { (float)kpi.x + r_square, (float)kpi.y - r_square }, black_color, { 0, 0 } },
                    { { (float)kpi.x + r_square, (float)kpi.y + r_square }, black_color, { 0, 0 } },
                };
                SDL_RenderGeometry(renderer, nullptr, verts, 6, nullptr, 0);
            }
        }
        SDL_SetRenderTarget(renderer, m_offscreen_od_2);
        for (const auto& [sf, its_keypoints] : sorted_faces_after_has_keypoints) {
            SDL_RenderGeometry(renderer, sf.face->texture,
                            sf.sdl_verts.data(), (int)sf.sdl_verts.size(),
                            nullptr, 0);
            for (auto& kpi : its_keypoints) {
                SDL_Vertex verts[6] = {
                    { { (float)kpi.x - r_square, (float)kpi.y - r_square }, white_color, { 0, 0 } },
                    { { (float)kpi.x - r_square, (float)kpi.y + r_square }, white_color, { 0, 0 } },
                    { { (float)kpi.x + r_square, (float)kpi.y + r_square }, white_color, { 0, 0 } },
                    { { (float)kpi.x - r_square, (float)kpi.y - r_square }, white_color, { 0, 0 } },
                    { { (float)kpi.x + r_square, (float)kpi.y - r_square }, white_color, { 0, 0 } },
                    { { (float)kpi.x + r_square, (float)kpi.y + r_square }, white_color, { 0, 0 } },
                };
                SDL_RenderGeometry(renderer, nullptr, verts, 6, nullptr, 0);
            }

        }
    }

    if (has_keypoints) {
        std::vector<KeypointPixelInfos> keypoint_pixel_vector;

        for (auto& kpi : keypoint_pixel_set) {
            keypoint_pixel_vector.push_back(kpi);
        }

        SDL_SetRenderTarget(renderer, m_offscreen_od_1);

        for (KeypointPixelInfos& kpi : keypoint_pixel_vector) {
            SDL_Rect pixel_rect;
            pixel_rect.h = 1;
            pixel_rect.w = 1;
            pixel_rect.x = kpi.x;
            pixel_rect.y = kpi.y;
            SDL_Surface* od_surface = SDL_RenderReadPixels(renderer, &pixel_rect);
            memcpy(kpi.rgba_1, od_surface->pixels, 4);
            SDL_DestroySurface(od_surface);
        }

        SDL_SetRenderTarget(renderer, m_offscreen_od_2);

        for (KeypointPixelInfos& kpi : keypoint_pixel_vector) {
            SDL_Rect pixel_rect;
            pixel_rect.h = 1;
            pixel_rect.w = 1;
            pixel_rect.x = kpi.x;
            pixel_rect.y = kpi.y;
            SDL_Surface* od_surface = SDL_RenderReadPixels(renderer, &pixel_rect);
            memcpy(kpi.rgba_2, od_surface->pixels, 4);
            SDL_DestroySurface(od_surface);

            if (memcmp(kpi.rgba_1, kpi.rgba_2, 4) == 0) {
                keypoints[kpi.index_outer].second[kpi.index_inner].occluded = true;
            } else {
                keypoints[kpi.index_outer].second[kpi.index_inner].occluded = false;
            }
        }

        SDL_SetRenderTarget(renderer, o_offscreen);
    }

    SDL_SetRenderTextureAddressMode(renderer, prev_u, prev_v);
}

const std::vector<SceneNodePtr>& Scene::GetAllNodes() const { return m_nodes; }
