#include "precomp.hpp"

namespace cv {

RasterizeSettings::RasterizeSettings()
{
    shadingType = ShadingType::Shaded;
    cullingMode = CullingMode::CW;
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
                         bool invDepthMode, RasterizeSettings settings)
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
    if ((settings.cullingMode == CullingMode::CW  && d <= 0) ||
        (settings.cullingMode == CullingMode::CCW && d >= 0) ||
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
                float zInter = 0.f;
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
                else // Shading::White
                {
                    update = true;
                }

                if (!colorBuf.empty() && update)
                {
                    Vec3f color {0, 0, 0};
                    if (settings.shadingType == ShadingType::White)
                    {
                        color = { 1.f, 1.f, 1.f };
                    }
                    if (settings.shadingType == ShadingType::Flat)
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


CV_EXPORTS  void triangleRasterize(InputArray _vertices, InputArray _indices, InputArray _colors,
                                   InputArray cameraPose, float fovyRadians, float zNear, float zFar,
                                   int width, int height, RasterizeSettings settings,
                                   OutputArray _depthBuffer, OutputArray _colorBuffer)
{
    bool needDepth = _depthBuffer.needed();
    bool needColor = _colorBuffer.needed();

    if (!needDepth && !needColor)
    {
        CV_Error(Error::StsBadArg, "No depth nor color output image provided");
    }

    CV_Assert(cameraPose.type() == CV_32F);
    CV_Assert((cameraPose.size() == Size {4, 3}) || (cameraPose.size() == Size {4, 4}));

    CV_Assert((fovyRadians > 0) && (fovyRadians < CV_PI));
    CV_Assert(zNear > 0);
    CV_Assert(zFar > zNear);
    CV_Assert((width > 0) && (height > 0));

    Mat cpMat = cameraPose.getMat();
    Matx44f camPoseMat = Matx44f::eye();
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            camPoseMat(i, j) = cpMat.at<float>(i, j);
        }
    }

    // world-to-camera coord system
    Matx44f lookAtMatrix = camPoseMat;

    // camera to NDC: [-1, 1]^3
    Matx44f perspectMatrix = perspectMatrixCal((float)width / (float)height, fovyRadians, zNear, zFar);

    Matx44f mvpMatrix = perspectMatrix * lookAtMatrix;

    bool hasIdx    = !_indices.empty();
    bool hasColors = !_colors.empty();

    Mat vertices, colors, triangles;
    int nVerts = 0, nColors = 0, nTriangles = 0;

    if (hasIdx)
    {
        CV_CheckFalse(_vertices.empty(), "No vertices provided along with indices array");

        int vertexType = _vertices.type();
        CV_Assert(vertexType == CV_32FC1 || vertexType == CV_32FC3);
        vertices = _vertices.getMat();
        if ((_vertices.channels() == 1) && (_vertices.rows() == 3))
        {
            vertices = vertices.t();
        }
        vertices = vertices.reshape(3, 1).t();
        nVerts = vertices.total();

        int indexType = _indices.type();
        CV_Assert(indexType == CV_32SC1 || indexType == CV_32SC3);
        triangles = _indices.getMat();
        if ((_indices.channels() == 1) && (_indices.rows() == 3))
        {
            triangles = triangles.t();
        }
        triangles = triangles.reshape(3, 1).t();
        nTriangles = triangles.total();

        if (hasColors)
        {
            int colorType = _colors.type();
            CV_Assert(colorType == CV_32FC1 || colorType == CV_32FC3);
            colors = _colors.getMat();
            if ((_colors.channels() == 1) && (_colors.rows() == 3))
            {
                colors = colors.t();
            }
            colors = colors.reshape(3, 1).t();
            nColors = colors.total();

            CV_Assert(nColors == nVerts);
            CV_Assert(settings.shadingType == ShadingType::Flat ||
                      settings.shadingType == ShadingType::Shaded);
        }
        else
        {
            CV_Assert(settings.shadingType == ShadingType::White);
        }
    }

    //TODO: add this to settings
    bool invDepthMode = false;

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
            _colorBuffer.setTo(cv::Scalar::all(0));
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
                _colorBuffer.setTo(cv::Scalar::all(0));
            }

            colorBuf = _colorBuffer.getMat();
        }
    }

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

        drawTriangle(ver, col, depthBuf, colorBuf, invDepthMode, settings);
    }
}

} // namespace cv
