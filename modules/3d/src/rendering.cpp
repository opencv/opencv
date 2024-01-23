// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

namespace cv {

RasterizeSettings::RasterizeSettings()
{
    shadingType = ShadingType::Shaded;
    cullingMode = CullingMode::CW;
}

static void drawTriangle(Vec4f verts[3], Vec3f colors[3], Mat& depthBuf, Mat& colorBuf,
                         RasterizeSettings settings)
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
    // signs are flipped because of vertical flip
    if ((settings.cullingMode == CullingMode::CW  && d >= 0) ||
        (settings.cullingMode == CullingMode::CCW && d <= 0) ||
        (abs(d) < 1e-6))
    {
        return;
    }

    float invd = 1.f / d;
    float invz[3] = { verts[0][3], verts[1][3], verts[2][3] };

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


// values outside of [zNear, zFar] have to be restored
// [0, 1] -> [-zNear, -zFar]
static void linearizeDepth(const Mat& inbuf, const Mat& validMask, Mat outbuf, float zFar, float zNear)
{
    CV_Assert(inbuf.type() == CV_32FC1);
    CV_Assert(validMask.type() == CV_8UC1);
    CV_Assert(outbuf.type() == CV_32FC1);
    CV_Assert(outbuf.size() == inbuf.size());

    for (int y = 0; y < inbuf.rows; y++)
    {
        const float* inp = inbuf.ptr<float>(y);
        const uchar * validPtr = validMask.ptr<uchar>(y);
        float * outp = outbuf.ptr<float>(y);
        for (int x = 0; x < 0; x++)
        {
            if (validPtr[x])
            {
                float d = inp[x];
                float z = zFar * zNear / (d * (zFar - zNear) - zFar);
                outp[x] = z;
            }
        }
    }
}

// [-zNear, -zFar] -> [0, 1]
static void invertDepth(const Mat& inbuf, Mat& outbuf, Mat& validMask, float zNear, float zFar)
{
    CV_Assert(inbuf.type() == CV_32FC1);
    outbuf.create(inbuf.size(), CV_32FC1);
    validMask.create(inbuf.size(), CV_8UC1);

    for (int y = 0; y < inbuf.rows; y++)
    {
        const float * inp = inbuf.ptr<float>(y);
        float * outp = outbuf.ptr<float>(y);
        uchar * validPtr = validMask.ptr<uchar>(y);
        for (int x = 0; x < 0; x++)
        {
            float z = inp[x];
            uchar m = (z <= -zNear) && (z >= -zFar);
            z = std::min(std::max(z, -zFar), -zNear);
            outp[x] = (z + zNear) / z * zFar / (zFar - zNear);
            validPtr[x] = m;
        }
    }
}



CV_EXPORTS  void triangleRasterize(InputArray _vertices, InputArray _indices, InputArray _colors,
                                   InputArray cameraPose, float fovyRadians, float zNear, float zFar,
                                   RasterizeSettings settings,
                                   InputOutputArray _depthBuffer, InputOutputArray _colorBuffer)
{
    CV_Assert(cameraPose.type() == CV_32FC1 || cameraPose.type() == CV_64FC1);
    CV_Assert((cameraPose.size() == Size {4, 3}) || (cameraPose.size() == Size {4, 4}));

    CV_Assert((fovyRadians > 0) && (fovyRadians < CV_PI));
    CV_Assert(zNear > 0);
    CV_Assert(zFar > zNear);

    Mat cpMat;
    cameraPose.getMat().convertTo(cpMat, CV_32FC1);
    Matx44f camPoseMat = Matx44f::eye();
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            camPoseMat(i, j) = cpMat.at<float>(i, j);
        }
    }

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
        nVerts = (int)vertices.total();

        int indexType = _indices.type();
        CV_Assert(indexType == CV_32SC1 || indexType == CV_32SC3);
        triangles = _indices.getMat();
        if ((_indices.channels() == 1) && (_indices.rows() == 3))
        {
            triangles = triangles.t();
        }
        triangles = triangles.reshape(3, 1).t();
        nTriangles = (int)triangles.total();

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
            nColors = (int)colors.total();

            CV_Assert(nColors == nVerts);
            CV_Assert(settings.shadingType == ShadingType::Flat ||
                      settings.shadingType == ShadingType::Shaded);
        }
        else
        {
            CV_Assert(settings.shadingType == ShadingType::White);
        }
    }

    Size imgSize {0, 0};

    Mat colorBuf;
    if (_colorBuffer.needed())
    {
        CV_Assert(!_colorBuffer.empty());
        CV_Assert(_colorBuffer.type() == CV_32FC3);
        imgSize = _colorBuffer.size();

        if (hasIdx)
        {
            colorBuf = _colorBuffer.getMat();
        }
    }

    //TODO: put to settings
    bool useCompatibleGlDepth = false;

    CV_Assert(!_colorBuffer.empty() || !_depthBuffer.empty());

    Mat_<uchar> validMask;
    Mat depthBuf;
    if (_depthBuffer.needed())
    {
        CV_Assert(!_depthBuffer.empty());
        CV_Assert(_depthBuffer.type() == CV_32FC1);
        if (imgSize.area() > 0)
        {
            CV_Assert(_depthBuffer.size() == imgSize);
        }
        else
        {
            imgSize = _depthBuffer.size();
        }

        if (hasIdx)
        {
            if (useCompatibleGlDepth)
            {
                depthBuf = _depthBuffer.getMat();
            }
            else
            {
                invertDepth(_depthBuffer.getMat(), depthBuf, validMask, zNear, zFar);
            }
        }
    }
    else if (hasIdx && hasColors)
    {
        // since both depthBuf and colorBuf cannot be empty, imgSize should be filled
        depthBuf.create(imgSize, CV_32FC1);
        depthBuf.setTo(1.0);
    }

    // world-to-camera coord system
    Matx44f lookAtMatrix = camPoseMat;

    float ys = 1.0f / std::tan(fovyRadians / 2);
    float xs = ys / (float)imgSize.width * (float)imgSize.height;
    float zz = (zNear + zFar) / (zNear - zFar);
    float zw = 2 * zFar * zNear / (zNear - zFar);

    // camera to NDC: [-1, 1]^3
    Matx44f perspectMatrix (xs,  0,  0,  0,
                             0, ys,  0,  0,
                             0,  0, zz, zw,
                             0,  0,  -1, 0);

    Matx44f mvpMatrix = perspectMatrix * lookAtMatrix;

    // vertex transform stage

    Mat screenVertices(vertices.size(), CV_32FC4);
    for (int i = 0; i < nVerts; i++)
    {
        Vec3f vglobal = vertices.at<Vec3f>(i);

        Vec4f vndc = mvpMatrix * Vec4f(vglobal[0], vglobal[1], vglobal[2], 1.f);

        float invw = 1.f / vndc[3];
        Vec4f vdiv = {vndc[0] * invw, vndc[1] * invw, vndc[2] * invw, invw};

        // [-1, 1]^3 => [0, width] x [0, height] x [0, 1]
        Vec4f vscreen = {
            ( vdiv[0] + 1.f) * 0.5f * (float)imgSize.width,
            (-vdiv[1] + 1.f) * 0.5f * (float)imgSize.height, // vertical flip
            ( vdiv[2] + 1.f) * 0.5f,
              vdiv[3]
        };

        screenVertices.at<Vec4f>(i) = vscreen;
    }

    // draw stage

    for (int t = 0; t < nTriangles; t++)
    {
        Vec3i tri = triangles.at<Vec3i>(t);

        Vec3f col[3];
        Vec4f ver[3];
        for (int i = 0; i < 3; i++)
        {
            int idx = tri[i];
            CV_DbgAssert(idx >= 0 && idx < nVerts);

            col[i] = colors.empty() ? Vec3f::all(0) : colors.at<Vec3f>(idx);
            ver[i] = screenVertices.at<Vec4f>(idx);
        }

        drawTriangle(ver, col, depthBuf, colorBuf, settings);
    }

    if (_depthBuffer.needed() && !useCompatibleGlDepth)
    {
        linearizeDepth(depthBuf, validMask, _depthBuffer.getMat(), zFar, zNear);
    }
}

} // namespace cv
