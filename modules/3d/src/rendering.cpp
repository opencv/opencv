#include "precomp.hpp"

namespace cv {

enum class ShadingType
{
    White = 0,
    Flat = 1,
    Shaded = 2
};

enum class CullingMode
{
    None,
    CW,
    CCW
};

static Vec3f normalize_vector(Vec3f a)
{
    float length = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    return Vec3f(a[0] / length, a[1] / length, a[2] / length);
}

static Matx44f lookAtMatrixCal(const Vec3f& position, const Vec3f& lookat, const Vec3f& upVector)
{
    Vec3f w = normalize_vector(position - lookat);
    Vec3f u = normalize_vector(upVector.cross(w));

    Vec3f v = w.cross(u);

    Matx44f res(u[0], u[1], u[2],   0,
                v[0], v[1], v[2],   0,
                w[0], w[1], w[2],   0,
                   0,    0,    0,   1.f);

    Matx44f translate(1.f,   0,   0, -position[0],
                        0, 1.f,   0, -position[1],
                        0,   0, 1.f, -position[2],
                        0,   0,   0,          1.0f);
    res = res * translate;

    return res;
}


static Matx44f perspectMatrixCal(float aspect, float fovy, float zNear, float zFar)
{
    float d = 1.0f / std::tan(fovy / 2);
    float a = (zNear + zFar) / (zNear - zFar);
    float b = 2 * zFar * zNear / (zNear - zFar);
    float c = d / aspect;
    Matx44f res(c,  0,  0,  0,
                0,  d,  0,  0,
                0,  0,  a,  b,
                0,  0, -1,  0);
    return res;
}


static void drawTriangle(Vec4f verts[3], Vec3f colors[3], Mat& depthBuf, Mat& colorBuf,
                         bool invDepthMode, ShadingType shadingType, CullingMode cullingMode)
{
    // any of buffers can be empty
    int width  = std::max(colorBuf.cols, depthBuf.cols);
    int height = std::max(colorBuf.rows, depthBuf.rows);

    Point minPt(width, height), maxPt(0, 0);
    for (int i = 0; i < 3; i++)
    {
        // round down to cover the whole pixel
        int x = (int)(verts[i][0]), y = (int)(verts[i][1]);
        minPt.x = std::min(    x, minPt.x);
        minPt.y = std::min(    y, minPt.y);
        maxPt.x = std::max(x + 1, maxPt.x);
        maxPt.y = std::max(y + 1, maxPt.y);
    }

    minPt.x = std::max(minPt.x, 0); maxPt.x = std::min(maxPt.x, width);
    minPt.y = std::max(minPt.y, 0); maxPt.y = std::min(maxPt.y, height);

    Point2f a(verts[0][0], verts[0][1]), b(verts[1][0], verts[1][1]), c(verts[2][0], verts[2][1]);
    Point2f bc = b - c, ac = a - c;
    float d = ac.x*bc.y - ac.y*bc.x;

    // culling and degenerated triangle removal
    if ((cullingMode == CullingMode::CW  && d <= 0) ||
        (cullingMode == CullingMode::CCW && d >= 0) ||
        (abs(d) < 1e-6))
    {
        return;
    }

    float invd = 1.f / d;
    float invz[3] = { 1.f / verts[0][3], 1.f / verts[1][3], 1.f / verts[2][3] };

    for (int y = minPt.y; y < maxPt.y; y++)
    {
        for (int x = minPt.x; x < maxPt.x; x++)
        {
            Point2f p(x + 0.5f, y + 0.5f), pc = p - c;
            // barycentric coordinates
            Vec3f f;
            f[0] = ( pc.x * bc.y - pc.y * bc.x) * invd;
            f[1] = ( pc.y * ac.x - pc.x * ac.y) * invd;
            f[2] = 1.f - f[0] - f[1];
            // if inside the triangle
            if ((f[0] >= 0) && (f[1] >= 0) && (f[2] >= 0))
            {
                bool update = false;
                float zInter;
                if (!depthBuf.empty())
                {
                    zInter = 1.0f / (f[0] * invz[0] + f[1] * invz[1] + f[2] * invz[2]);
                    float zCurrent = depthBuf.at<float>(y, x);
                    if (zInter < zCurrent)
                    {
                        update = true;
                        depthBuf.at<float>(y, x) = zInter;
                    }
                }
                else
                {
                    update = true;
                }

                if (!colorBuf.empty() && update)
                {
                    Vec3f color {0, 0, 0};
                    if (shadingType == ShadingType::White)
                    {
                        color = { 1.f, 1.f, 1.f };
                    }
                    if (shadingType == ShadingType::Flat)
                    {
                        color = colors[0];
                    }
                    else // ShadingType::Shaded
                    {
                        Vec3f interp;
                        for (int i = 0; i < 3; i++)
                        {
                            interp[i] = f[i] * zInter * invz[i];
                        }

                        for (int i = 0; i < 3; i++)
                        {
                            for (int j = 0; j < 3; j++)
                            {
                                color[i] += interp[j] * colors[j][i];
                            }
                        }
                    }
                    colorBuf.at<Vec3f>(y, x) = color;
                }
            }
        }
    }
}


void triangleRasterize(InputArray _vertices, InputArray _indices, InputArray _colors,
                       InputArray cameraMatrix, int width, int height, bool shadingMode,
                       OutputArray _depthBuffer, OutputArray _colorBuffer)
{
    //TODO: fix this
    Mat camera = cameraMatrix.getMat();
    Vec3f position = camera.row(0);
    Vec3f lookat = camera.row(1);
    Vec3f upVector = camera.row(2);
    float fovy = camera.at<float>(3, 0), zNear = camera.at<float>(3, 1), zFar = camera.at<float>(3, 2);

    //TODO: add this to args
    bool invDepthMode = false;
    //TODO: add this to args
    // default mode is CW
    CullingMode cullingMode = CullingMode::None;

    bool needDepth = _depthBuffer.needed();
    bool needColor = _colorBuffer.needed();

    if (!needDepth && !needColor)
    {
        CV_Error(Error::StsBadArg, "No depth nor color output image provided");
    }

    bool hasVerts  = !_vertices.empty();
    bool hasIdx    = !_indices.empty();
    bool hasColors = !_colors.empty();

    Mat vertices, colors, triangles;
    int nVerts = 0, nColors = 0, nTriangles = 0;

    ShadingType shadingType = ShadingType::White;

    if (hasIdx)
    {
        if (!hasVerts)
        {
            CV_Error(Error::StsBadArg, "No vertices provided");
        }
        else
        {
            //TODO: check rows/cols/channels
            CV_Assert(_vertices.depth() == CV_32F);
            bool vert3f = (_vertices.channels() * _vertices.total() % 3 == 0);
            // not supported yet
            //bool vert4f = (_vertices.channels() * _vertices.total() % 4 == 0);
            //CV_Assert(vert3f || vert4f);
            CV_Assert(vert3f);

            vertices = _vertices.getMat().reshape(3, 1).t();
            nVerts = vertices.total();

            //TODO: check rows/cols/channels
            // the rest int types are not supported yet
            CV_Assert(_indices.depth() == CV_32S);
            CV_Assert(_indices.channels() * _indices.total() % 3 == 0);

            triangles = _indices.getMat().reshape(3, 1).t();
            nTriangles = triangles.total();

            if (hasColors)
            {
                //TODO: check rows/cols/channels
                CV_Assert(_colors.depth() == CV_32F);
                bool col3f = (_colors.channels() * _colors.total() % 3 == 0);
                // 4f is not supported yet
                CV_Assert(col3f);

                colors = _colors.getMat().reshape(3, 1).t();
                nColors = colors.total();

                CV_Assert(nColors == nVerts);

                shadingType = shadingMode ? ShadingType::Shaded : ShadingType::Flat;
            }
            else
            {
                shadingType = ShadingType::White;
            }
        }
    }

    Mat depthBuf;
    if (needDepth)
    {
        if (_depthBuffer.empty())
        {
            // 64f is not supported yet
            _depthBuffer.create(cv::Size(width, height), CV_32FC1);
            //TODO: wrong value, should be 1
            float maxv = invDepthMode ? 1.f : zFar;
            _depthBuffer.setTo(maxv);
        }
        else
        {
            CV_Assert(_depthBuffer.size() == cv::Size(width, height));
            CV_Assert(_depthBuffer.type() == CV_32FC1);
        }

        if (hasIdx)
        {
            depthBuf = _depthBuffer.getMat();
        }
    }
    else if (hasIdx && hasColors)
    {
        invDepthMode = true;
        depthBuf.create(cv::Size(width, height), CV_32FC1);
        //TODO: wrong value, should be 1
        float maxv = invDepthMode ? 1.f : zFar;
        depthBuf.setTo(maxv);
    }

    Mat colorBuf;
    if (needColor)
    {
        if (_colorBuffer.empty())
        {
            // other types are not supported yet
            _colorBuffer.create(cv::Size(width, height), CV_32FC3);
            _colorBuffer.setTo(cv::Scalar(0, 0, 0));
        }
        else
        {
            CV_Assert(_colorBuffer.size() == cv::Size(width, height));
            CV_Assert(_colorBuffer.type() == CV_32FC3);
        }

        if (hasIdx)
        {
            // other types are not supported yet
            if (_colorBuffer.empty())
            {
                _colorBuffer.create(cv::Size(width, height), CV_32FC3);
                _colorBuffer.setTo(cv::Scalar(0, 0, 0));
            }

            colorBuf = _colorBuffer.getMat();
        }
    }

    // world-to-camera coord system
    Matx44f lookAtMatrix = lookAtMatrixCal(position, lookat, upVector);
    // camera to NDC: [-1, 1]^3
    //TODO: argument angle in radians
    float fovyDegrees = fovy;
    float fovyRadians = fovyDegrees * CV_PI/180.0;
    Matx44f perspectMatrix = perspectMatrixCal((float)width / (float)height, fovyRadians, zNear, zFar);

    //TODO: find out what the heck is this and fix it
    Matx44f modelMatrix;
    {
        Matx44f modelMatrix_scale(15,  0,  0,  0,
                                   0, 15,  0,  0,
                                   0,  0, 15,  0,
                                   0,  0,  0,  1);
        Matx44f modelMatrix_translate(1,  0,  0,   0,
                                      0,  1,  0,  20,
                                      0,  0,  1, -15,
                                      0,  0,  0,   1);
        
        Matx44f modelMatrix_rotate_y(-1, 0,  0, 0,
                                      0, 1,  0, 0,
                                      0, 0, -1, 0,
                                      0, 0,  0, 1);

        modelMatrix = modelMatrix_translate * modelMatrix_scale * modelMatrix_rotate_y;
    }
    //Matx44f mvpMatrix = perspectMatrix * lookAtMatrix * modelMatrix;

    Matx44f mvpMatrix = perspectMatrix * lookAtMatrix;

    for (int t = 0; t < nTriangles; t++)
    {
        Vec3i tri = triangles.at<Vec3i>(t);

        Vec3f col[3];
        Vec4f ver[3];
        for (int i = 0; i < 3; i++)
        {
            int idx = tri[i];

            CV_DbgAssert(idx >= 0 && idx < nVerts);

            Vec3f vglobal = vertices.at<Vec3f>(idx);

            Vec3f c(0, 0, 0);
            if (!colors.empty())
            {
                c = colors.at<Vec3f>(idx);
            }
            col[i] = c;

            Vec4f vndc = mvpMatrix * Vec4f(vglobal[0], vglobal[1], vglobal[2], 1.f);

            float invw = 1.f / vndc[3];
            Vec4f vdiv = { vndc[0] * invw, vndc[1] * invw, vndc[2] * invw, vndc[3] };

            // [-1, 1]^3 => [0, width] x [0, height] x [0, 1]
            Vec4f vscreen = {
                (vdiv[0] + 1.f) * 0.5f * width,
                (vdiv[1] + 1.f) * 0.5f * height,
                (vdiv[2] + 1.f) * 0.5f,
                vdiv[3]
            };

            ver[i] = vscreen;
        }

        drawTriangle(ver, col, depthBuf, colorBuf, invDepthMode, shadingType, cullingMode);
    }
}

} // namespace cv
