// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

namespace cv {

TriangleRasterizeSettings::TriangleRasterizeSettings()
{
    shadingType = RASTERIZE_SHADING_SHADED;
    cullingMode = RASTERIZE_CULLING_CW;
    glCompatibleMode = RASTERIZE_COMPAT_DISABLED;
}

static void drawTriangle(Vec4f verts[3], Vec3f colors[3], Mat& depthBuf, Mat& colorBuf,
                         TriangleRasterizeSettings settings)
{
    // this will be useful during refactoring
    // if there's gonna be more supported data types
    CV_DbgAssert(depthBuf.empty() || depthBuf.type() == CV_32FC1);
    CV_DbgAssert(colorBuf.empty() || colorBuf.type() == CV_32FC3);

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
    if ((settings.cullingMode == RASTERIZE_CULLING_CW  && d <= 0) ||
        (settings.cullingMode == RASTERIZE_CULLING_CCW && d >= 0) ||
        (abs(d) < 1e-6))
    {
        return;
    }

    float invd = 1.f / d;
    Vec3f zinv { verts[0][2], verts[1][2], verts[2][2] };
    Vec3f w { verts[0][3], verts[1][3], verts[2][3] };

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
                if (!depthBuf.empty())
                {
                    float zCurrent = depthBuf.at<float>(height - 1 - y, x);
                    float zNew = f[0] * zinv[0] + f[1] * zinv[1] + f[2] * zinv[2];
                    if (zNew < zCurrent)
                    {
                        update = true;
                        depthBuf.at<float>(height - 1 - y, x) = zNew;
                    }
                }
                else // RASTERIZE_SHADING_WHITE
                {
                    update = true;
                }

                if (!colorBuf.empty() && update)
                {
                    Vec3f color;
                    if (settings.shadingType == RASTERIZE_SHADING_WHITE)
                    {
                        color = { 1.f, 1.f, 1.f };
                    }
                    else if (settings.shadingType == RASTERIZE_SHADING_FLAT)
                    {
                        color = colors[0];
                    }
                    else // TriangleShadingType::Shaded
                    {
                        float zInter = 1.0f / (f[0] * w[0] + f[1] * w[1] + f[2] * w[2]);
                        color = { 0, 0, 0 };
                        for (int j = 0; j < 3; j++)
                        {
                            color += (f[j] * w[j]) * colors[j];
                        }
                        color *= zInter;
                    }
                    colorBuf.at<Vec3f>(height - 1 - y, x) = color;
                }
            }
        }
    }
}


// values outside of [zNear, zFar] have to be restored
// [0, 1] -> [zNear, zFar]
static void linearizeDepth(const Mat& inbuf, const Mat& validMask, Mat outbuf, double zFar, double zNear)
{
    CV_Assert(inbuf.type() == CV_32FC1);
    CV_Assert(validMask.type() == CV_8UC1 || validMask.type() == CV_8SC1 || validMask.type() == CV_BoolC1);
    CV_Assert(outbuf.type() == CV_32FC1);
    CV_Assert(outbuf.size() == inbuf.size());

    float scaleNear = (float)(1.0 / zNear);
    float scaleFar  = (float)(1.0 / zFar);
    for (int y = 0; y < inbuf.rows; y++)
    {
        const float* inp = inbuf.ptr<float>(y);
        const uchar * validPtr = validMask.ptr<uchar>(y);
        float * outp = outbuf.ptr<float>(y);
        for (int x = 0; x < inbuf.cols; x++)
        {
            if (validPtr[x])
            {
                float d = inp[x];
                // precision-optimized version of this:
                //float z = - zFar * zNear / (d * (zFar - zNear) - zFar);
                float z =  1.f / ((1.f - d) * scaleNear + d * scaleFar );
                outp[x] = z;
            }
        }
    }
}

// [zNear, zFar] -> [0, 1]
static void invertDepth(const Mat& inbuf, Mat& outbuf, Mat& validMask, double zNear, double zFar)
{
    CV_Assert(inbuf.type() == CV_32FC1);
    outbuf.create(inbuf.size(), CV_32FC1);
    validMask.create(inbuf.size(), CV_8UC1);

    float fNear = (float)zNear, fFar = (float)zFar;
    float zadd = (float)(zFar / (zFar - zNear));
    float zmul = (float)(-zNear * zFar / (zFar - zNear));
    for (int y = 0; y < inbuf.rows; y++)
    {
        const float * inp = inbuf.ptr<float>(y);
        float * outp = outbuf.ptr<float>(y);
        uchar * validPtr = validMask.ptr<uchar>(y);
        for (int x = 0; x < inbuf.cols; x++)
        {
            float z = inp[x];
            uchar m = (z >= fNear) && (z <= fFar);
            z = std::max(std::min(z, fFar), fNear);
            // precision-optimized version of this:
            // outp[x] = (z - zNear) / z * zFar / (zFar - zNear);
            outp[x] = zadd + zmul / z;
            validPtr[x] = m;
        }
    }
}


static void triangleRasterizeInternal(InputArray _vertices, InputArray _indices, InputArray _colors,
                                      Mat& colorBuf, Mat& depthBuf,
                                      InputArray world2cam, double fovyRadians, double zNear, double zFar,
                                      const TriangleRasterizeSettings& settings)
{
    CV_Assert(world2cam.type() == CV_32FC1 || world2cam.type() == CV_64FC1);
    CV_Assert((world2cam.size() == Size {4, 3}) || (world2cam.size() == Size {4, 4}));

    CV_Assert((fovyRadians > 0) && (fovyRadians < CV_PI));
    CV_Assert(zNear > 0);
    CV_Assert(zFar > zNear);

    Mat cpMat;
    world2cam.getMat().convertTo(cpMat, CV_64FC1);
    Matx44d camPoseMat = Matx44d::eye();
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            camPoseMat(i, j) = cpMat.at<double>(i, j);
        }
    }

    if(_indices.empty())
    {
        return;
    }

    CV_CheckFalse(_vertices.empty(), "No vertices provided along with indices array");

    Mat vertices, colors, triangles;
    int nVerts = 0, nColors = 0, nTriangles = 0;

    int vertexType = _vertices.type();
    CV_Assert(vertexType == CV_32FC1 || vertexType == CV_32FC3);
    vertices = _vertices.getMat();
    // transform 3xN matrix to Nx3, except 3x3
    if ((_vertices.channels() == 1) && (_vertices.rows() == 3) && (_vertices.cols() != 3))
    {
        vertices = vertices.t();
    }
    // This transposition is performed on 1xN matrix so it's almost free in terms of performance
    vertices = vertices.reshape(3, 1).t();
    nVerts = (int)vertices.total();

    int indexType = _indices.type();
    CV_Assert(indexType == CV_32SC1 || indexType == CV_32SC3);
    triangles = _indices.getMat();
    // transform 3xN matrix to Nx3, except 3x3
    if ((_indices.channels() == 1) && (_indices.rows() == 3) && (_indices.cols() != 3))
    {
        triangles = triangles.t();
    }
    // This transposition is performed on 1xN matrix so it's almost free in terms of performance
    triangles = triangles.reshape(3, 1).t();
    nTriangles = (int)triangles.total();

    if (!_colors.empty())
    {
        int colorType = _colors.type();
        CV_Assert(colorType == CV_32FC1 || colorType == CV_32FC3);
        colors = _colors.getMat();
        // transform 3xN matrix to Nx3, except 3x3
        if ((_colors.channels() == 1) && (_colors.rows() == 3) && (_colors.cols() != 3))
        {
            colors = colors.t();
        }
        colors = colors.reshape(3, 1).t();
        nColors = (int)colors.total();

        CV_Assert(nColors == nVerts);
    }

    // any of buffers can be empty
    Size imgSize {std::max(colorBuf.cols, depthBuf.cols), std::max(colorBuf.rows, depthBuf.rows)};

    // world-to-camera coord system
    Matx44d lookAtMatrix = camPoseMat;

    double ys = 1.0 / std::tan(fovyRadians / 2);
    double xs = ys / (double)imgSize.width * (double)imgSize.height;
    double zz = (zNear + zFar) / (zNear - zFar);
    double zw = 2.0 * zFar * zNear / (zNear - zFar);

    // camera to NDC: [-1, 1]^3
    Matx44d perspectMatrix (xs,  0,  0,  0,
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
            (vdiv[0] + 1.f) * 0.5f * (float)imgSize.width,
            (vdiv[1] + 1.f) * 0.5f * (float)imgSize.height,
            (vdiv[2] + 1.f) * 0.5f,
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
}


void triangleRasterizeDepth(InputArray _vertices, InputArray _indices, InputOutputArray _depthBuf,
                            InputArray world2cam, double fovY, double zNear, double zFar,
                            const TriangleRasterizeSettings& settings)
{
    CV_Assert(!_depthBuf.empty());
    CV_Assert(_depthBuf.type() == CV_32FC1);

    Mat emptyColorBuf;
    // out-of-range values from user-provided depthBuf should not be altered, let's mark them
    Mat_<uchar> validMask;
    Mat depthBuf;
    if (settings.glCompatibleMode == RASTERIZE_COMPAT_INVDEPTH)
    {
        depthBuf = _depthBuf.getMat();
    }
    else // RASTERIZE_COMPAT_DISABLED
    {
        invertDepth(_depthBuf.getMat(), depthBuf, validMask, zNear, zFar);
    }

    triangleRasterizeInternal(_vertices, _indices, noArray(), emptyColorBuf, depthBuf, world2cam, fovY, zNear, zFar, settings);

    if (settings.glCompatibleMode == RASTERIZE_COMPAT_DISABLED)
    {
        linearizeDepth(depthBuf, validMask, _depthBuf.getMat(), zFar, zNear);
    }
}

void triangleRasterizeColor(InputArray _vertices, InputArray _indices, InputArray _colors, InputOutputArray _colorBuf,
                            InputArray world2cam, double fovY, double zNear, double zFar,
                            const TriangleRasterizeSettings& settings)
{
    CV_Assert(!_colorBuf.empty());
    CV_Assert(_colorBuf.type() == CV_32FC3);
    Mat colorBuf = _colorBuf.getMat();

    Mat depthBuf;
    if (_colors.empty())
    {
        // full white shading does not require depth test
        CV_Assert(settings.shadingType == RASTERIZE_SHADING_WHITE);
    }
    else
    {
        // internal depth buffer is not exposed outside
        depthBuf.create(_colorBuf.size(), CV_32FC1);
        depthBuf.setTo(1.0);
    }

    triangleRasterizeInternal(_vertices, _indices, _colors, colorBuf, depthBuf, world2cam, fovY, zNear, zFar, settings);
}

void triangleRasterize(InputArray _vertices, InputArray _indices, InputArray _colors,
                       InputOutputArray _colorBuffer, InputOutputArray _depthBuffer,
                       InputArray world2cam, double fovyRadians, double zNear, double zFar,
                       const TriangleRasterizeSettings& settings)
{
    if (_colors.empty())
    {
        CV_Assert(settings.shadingType == RASTERIZE_SHADING_WHITE);
    }

    CV_Assert(!_colorBuffer.empty());
    CV_Assert(_colorBuffer.type() == CV_32FC3);
    CV_Assert(!_depthBuffer.empty());
    CV_Assert(_depthBuffer.type() == CV_32FC1);

    CV_Assert(_depthBuffer.size() == _colorBuffer.size());

    Mat colorBuf = _colorBuffer.getMat();

    // out-of-range values from user-provided depthBuf should not be altered, let's mark them
    Mat_<uchar> validMask;
    Mat depthBuf;
    if (settings.glCompatibleMode == RASTERIZE_COMPAT_INVDEPTH)
    {
        depthBuf = _depthBuffer.getMat();
    }
    else // RASTERIZE_COMPAT_DISABLED
    {
        invertDepth(_depthBuffer.getMat(), depthBuf, validMask, zNear, zFar);
    }

    triangleRasterizeInternal(_vertices, _indices, _colors, colorBuf, depthBuf, world2cam, fovyRadians, zNear, zFar, settings);

    if (settings.glCompatibleMode == RASTERIZE_COMPAT_DISABLED)
    {
        linearizeDepth(depthBuf, validMask, _depthBuffer.getMat(), zFar, zNear);
    }
}
} // namespace cv
