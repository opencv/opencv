// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "viz3d_private.hpp"
#include "opencv2/core/utils/logger.hpp"
#include "opencv2/imgproc.hpp"

namespace cv { namespace viz3d {

#ifdef HAVE_OPENGL

static void openGlDrawCallback(void* data)
{
    Window* win = static_cast<Window*>(data);
    if (win)
        win->draw();
}

static void openGlFreeCallback(void* data)
{
    Window* win = static_cast<Window*>(data);
    if (win)
        delete win;
}

static void mouseCallback(int event, int x, int y, int flags, void* data)
{
    Window* win = static_cast<Window*>(data);
    if (win)
        win->onMouse(event, x, y, flags);
}

static Window* getWindow(const String& win_name)
{
    namedWindow(win_name, WINDOW_OPENGL);
    const double useGl = getWindowProperty(win_name, WND_PROP_OPENGL);
    if (useGl <= 0)
        CV_Error(cv::Error::StsBadArg, "OpenCV/UI: viz3d can't be used because the window was created without an OpenGL context");

    Window* win = static_cast<Window*>(getOpenGlUserData(win_name));

    if (!win)
    {
		setOpenGlContext(win_name);
		win = new Window(win_name);
		setOpenGlDrawCallback(win_name, &openGlDrawCallback, win);
        setOpenGlFreeCallback(win_name, &openGlFreeCallback);
        setMouseCallback(win_name, &mouseCallback, win);
    }
    else
    {
        auto callback = getOpenGlDrawCallback(win_name);
        if (callback && callback != openGlDrawCallback)
            CV_Error(cv::Error::StsBadArg, "OpenCV/UI: viz3d can't be used because the OpenGL callback is already being used on this window");
    }

    return win;
}

#endif // HAVE_OPENGL

void setPerspective(const String& win_name, float fov, float z_near, float z_far)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENGL
    CV_UNUSED(win_name);
    CV_UNUSED(fov);
    CV_UNUSED(z_near);
    CV_UNUSED(z_far);
    CV_Error(cv::Error::OpenGlNotSupported, "The library is compiled without OpenGL support");
#else
    Window* win = getWindow(win_name);
    win->getView().setPerspective(fov, z_near, z_far);
    updateWindow(win_name);
#endif
}

void setGridVisible(const String& win_name, bool visible)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENGL
    CV_UNUSED(win_name);
    CV_UNUSED(visible);
    CV_Error(cv::Error::OpenGlNotSupported, "The library is compiled without OpenGL support");
#else
    Window* win = getWindow(win_name);
    win->setGridVisible(visible);
    updateWindow(win_name);
#endif
}

void setSun(const String& win_name, const Vec3f& direction, const Vec3f& ambient, const Vec3f& diffuse)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENGL
    CV_UNUSED(win_name);
    CV_UNUSED(direction);
    CV_UNUSED(ambient);
    CV_UNUSED(diffuse);
    CV_Error(cv::Error::OpenGlNotSupported, "The library is compiled without OpenGL support");
#else
    Window* win = getWindow(win_name);
    win->setSun(direction, ambient, diffuse);
    updateWindow(win_name);
#endif
}

void setSky(const String& win_name, const Vec3f& color)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENGL
    CV_UNUSED(win_name);
    CV_UNUSED(color);
    CV_Error(cv::Error::OpenGlNotSupported, "The library is compiled without OpenGL support");
#else
    Window* win = getWindow(win_name);
    win->setSky(color);
    updateWindow(win_name);
#endif
}

void showBox(const String& win_name, const String& obj_name, const Vec3f& size, const Vec3f& color, RenderMode mode)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENGL
    CV_UNUSED(win_name);
    CV_UNUSED(obj_name);
    CV_UNUSED(size);
    CV_UNUSED(color);
    CV_UNUSED(mode);
    CV_Error(cv::Error::OpenGlNotSupported, "The library is compiled without OpenGL support");
#else
    if (mode == RENDER_WIREFRAME)
    {
        float points_data[] = {
            -size(0), -size(1), -size(2), color(0), color(1), color(2),
            +size(0), -size(1), -size(2), color(0), color(1), color(2),
            -size(0), +size(1), -size(2), color(0), color(1), color(2),
            +size(0), +size(1), -size(2), color(0), color(1), color(2),
            -size(0), -size(1), +size(2), color(0), color(1), color(2),
            +size(0), -size(1), +size(2), color(0), color(1), color(2),
            -size(0), +size(1), +size(2), color(0), color(1), color(2),
            +size(0), +size(1), +size(2), color(0), color(1), color(2),

            -size(0), -size(1), -size(2), color(0), color(1), color(2),
            -size(0), +size(1), -size(2), color(0), color(1), color(2),
            +size(0), -size(1), -size(2), color(0), color(1), color(2),
            +size(0), +size(1), -size(2), color(0), color(1), color(2),
            -size(0), -size(1), +size(2), color(0), color(1), color(2),
            -size(0), +size(1), +size(2), color(0), color(1), color(2),
            +size(0), -size(1), +size(2), color(0), color(1), color(2),
            +size(0), +size(1), +size(2), color(0), color(1), color(2),

            -size(0), -size(1), -size(2), color(0), color(1), color(2),
            -size(0), -size(1), +size(2), color(0), color(1), color(2),
            +size(0), -size(1), -size(2), color(0), color(1), color(2),
            +size(0), -size(1), +size(2), color(0), color(1), color(2),
            -size(0), +size(1), -size(2), color(0), color(1), color(2),
            -size(0), +size(1), +size(2), color(0), color(1), color(2),
            +size(0), +size(1), -size(2), color(0), color(1), color(2),
            +size(0), +size(1), +size(2), color(0), color(1), color(2),
        };

        const Mat points_mat = Mat(Size(6, 24), CV_32F, &points_data);

        showLines(win_name, obj_name, points_mat);
    }
    else if (mode == RENDER_SIMPLE)
    {
        float verts_data[] = {
            -size(0), -size(1), -size(2), color(0), color(1), color(2),
            +size(0), -size(1), -size(2), color(0), color(1), color(2),
            +size(0), +size(1), -size(2), color(0), color(1), color(2),
            -size(0), +size(1), -size(2), color(0), color(1), color(2),
            -size(0), -size(1), +size(2), color(0), color(1), color(2),
            +size(0), -size(1), +size(2), color(0), color(1), color(2),
            +size(0), +size(1), +size(2), color(0), color(1), color(2),
            -size(0), +size(1), +size(2), color(0), color(1), color(2),
        };

        static unsigned char indices_data[] = {
            0, 1, 2,
            2, 3, 0,
            4, 5, 6,
            6, 7, 4,
            0, 1, 5,
            5, 4, 0,
            3, 2, 6,
            6, 7, 3,
            0, 3, 7,
            7, 4, 0,
            1, 2, 6,
            6, 5, 1,
        };

        const Mat verts_mat = Mat(Size(6, 8), CV_32F, &verts_data);
        const Mat indices_mat = Mat(Size(3, 12), CV_8U, &indices_data);

        showMesh(win_name, obj_name, verts_mat, indices_mat);
    }
    else if (mode == RENDER_SHADING)
    {
        float verts_data[] = {
            -size(0), -size(1), -size(2), color(0), color(1), color(2), 0.0f, 0.0f, -1.0f, // 0
            +size(0), -size(1), -size(2), color(0), color(1), color(2), 0.0f, 0.0f, -1.0f, // 1
            +size(0), +size(1), -size(2), color(0), color(1), color(2), 0.0f, 0.0f, -1.0f, // 2
            +size(0), +size(1), -size(2), color(0), color(1), color(2), 0.0f, 0.0f, -1.0f, // 2
            -size(0), +size(1), -size(2), color(0), color(1), color(2), 0.0f, 0.0f, -1.0f, // 3
            -size(0), -size(1), -size(2), color(0), color(1), color(2), 0.0f, 0.0f, -1.0f, // 0

            -size(0), -size(1), +size(2), color(0), color(1), color(2), 0.0f, 0.0f, +1.0f, // 4
            +size(0), -size(1), +size(2), color(0), color(1), color(2), 0.0f, 0.0f, +1.0f, // 5
            +size(0), +size(1), +size(2), color(0), color(1), color(2), 0.0f, 0.0f, +1.0f, // 6
            +size(0), +size(1), +size(2), color(0), color(1), color(2), 0.0f, 0.0f, +1.0f, // 6
            -size(0), +size(1), +size(2), color(0), color(1), color(2), 0.0f, 0.0f, +1.0f, // 7
            -size(0), -size(1), +size(2), color(0), color(1), color(2), 0.0f, 0.0f, +1.0f, // 4


            -size(0), -size(1), -size(2), color(0), color(1), color(2), 0.0f, -1.0f, 0.0f, // 0
            +size(0), -size(1), -size(2), color(0), color(1), color(2), 0.0f, -1.0f, 0.0f, // 1
            +size(0), -size(1), +size(2), color(0), color(1), color(2), 0.0f, -1.0f, 0.0f, // 5
            +size(0), -size(1), +size(2), color(0), color(1), color(2), 0.0f, -1.0f, 0.0f, // 5
            -size(0), -size(1), +size(2), color(0), color(1), color(2), 0.0f, -1.0f, 0.0f, // 4
            -size(0), -size(1), -size(2), color(0), color(1), color(2), 0.0f, -1.0f, 0.0f, // 0

            -size(0), +size(1), -size(2), color(0), color(1), color(2), 0.0f, +1.0f, 0.0f, // 3
            +size(0), +size(1), -size(2), color(0), color(1), color(2), 0.0f, +1.0f, 0.0f, // 2
            +size(0), +size(1), +size(2), color(0), color(1), color(2), 0.0f, +1.0f, 0.0f, // 6
            +size(0), +size(1), +size(2), color(0), color(1), color(2), 0.0f, +1.0f, 0.0f, // 6
            -size(0), +size(1), +size(2), color(0), color(1), color(2), 0.0f, +1.0f, 0.0f, // 7
            -size(0), +size(1), -size(2), color(0), color(1), color(2), 0.0f, +1.0f, 0.0f, // 3


            -size(0), -size(1), -size(2), color(0), color(1), color(2), -1.0f, 0.0f, 0.0f, // 0
            -size(0), +size(1), -size(2), color(0), color(1), color(2), -1.0f, 0.0f, 0.0f, // 3
            -size(0), +size(1), +size(2), color(0), color(1), color(2), -1.0f, 0.0f, 0.0f, // 7
            -size(0), +size(1), +size(2), color(0), color(1), color(2), -1.0f, 0.0f, 0.0f, // 7
            -size(0), -size(1), +size(2), color(0), color(1), color(2), -1.0f, 0.0f, 0.0f, // 4
            -size(0), -size(1), -size(2), color(0), color(1), color(2), -1.0f, 0.0f, 0.0f, // 0

            +size(0), -size(1), -size(2), color(0), color(1), color(2), +1.0f, 0.0f, 0.0f, // 1
            +size(0), +size(1), -size(2), color(0), color(1), color(2), +1.0f, 0.0f, 0.0f, // 2
            +size(0), +size(1), +size(2), color(0), color(1), color(2), +1.0f, 0.0f, 0.0f, // 6
            +size(0), +size(1), +size(2), color(0), color(1), color(2), +1.0f, 0.0f, 0.0f, // 6
            +size(0), -size(1), +size(2), color(0), color(1), color(2), +1.0f, 0.0f, 0.0f, // 5
            +size(0), -size(1), -size(2), color(0), color(1), color(2), +1.0f, 0.0f, 0.0f, // 1
        };

        const Mat verts_mat = Mat(Size(9, 36), CV_32F, &verts_data);

        showMesh(win_name, obj_name, verts_mat);
    }
#endif
}

void showPlane(const String& win_name, const String& obj_name, const Vec2f& size, const Vec3f& color, RenderMode mode)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENGL
    CV_UNUSED(win_name);
    CV_UNUSED(obj_name);
    CV_UNUSED(size);
    CV_UNUSED(color);
    CV_UNUSED(mode);
    CV_Error(cv::Error::OpenGlNotSupported, "The library is compiled without OpenGL support");
#else
    if (mode == RENDER_WIREFRAME)
    {
        float points_data[] = {
            -size(0), 0.0f, -size(1), color(0), color(1), color(2),
            +size(0), 0.0f, -size(1), color(0), color(1), color(2),
            +size(0), 0.0f, -size(1), color(0), color(1), color(2),
            +size(0), 0.0f, +size(1), color(0), color(1), color(2),
            +size(0), 0.0f, +size(1), color(0), color(1), color(2),
            -size(0), 0.0f, +size(1), color(0), color(1), color(2),
            -size(0), 0.0f, +size(1), color(0), color(1), color(2),
            -size(0), 0.0f, -size(1), color(0), color(1), color(2),
        };

        const Mat points_mat = Mat(Size(6, 8), CV_32F, points_data);

        showLines(win_name, obj_name, points_mat);
    }
    else if (mode == RENDER_SIMPLE)
    {
        float verts_data[] = {
            -size(0), 0.0f, -size(1), color(0), color(1), color(2),
            +size(0), 0.0f, -size(1), color(0), color(1), color(2),
            +size(0), 0.0f, +size(1), color(0), color(1), color(2),
            -size(0), 0.0f, -size(1), color(0), color(1), color(2),
            -size(0), 0.0f, +size(1), color(0), color(1), color(2),
            +size(0), 0.0f, +size(1), color(0), color(1), color(2),
        };

        const Mat verts_mat = Mat(Size(6, 6), CV_32F, verts_data);

        showMesh(win_name, obj_name, verts_mat);
    }
    else if (mode == RENDER_SHADING)
    {
        float verts_data[] = {
            -size(0), 0.0f, -size(1), color(0), color(1), color(2), 0.0f, 1.0f, 0.0f,
            +size(0), 0.0f, -size(1), color(0), color(1), color(2), 0.0f, 1.0f, 0.0f,
            +size(0), 0.0f, +size(1), color(0), color(1), color(2), 0.0f, 1.0f, 0.0f,
            -size(0), 0.0f, -size(1), color(0), color(1), color(2), 0.0f, 1.0f, 0.0f,
            -size(0), 0.0f, +size(1), color(0), color(1), color(2), 0.0f, 1.0f, 0.0f,
            +size(0), 0.0f, +size(1), color(0), color(1), color(2), 0.0f, 1.0f, 0.0f,
        };

        const Mat verts_mat = Mat(Size(9, 6), CV_32F, verts_data);

        showMesh(win_name, obj_name, verts_mat);
    }
#endif
}

void showSphere(const String& win_name, const String& obj_name, float radius, const Vec3f& color, RenderMode mode, int divs)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENGL
    CV_UNUSED(win_name);
    CV_UNUSED(obj_name);
    CV_UNUSED(radius);
    CV_UNUSED(color);
    CV_UNUSED(mode);
    CV_UNUSED(divs);
    CV_Error(cv::Error::OpenGlNotSupported, "The library is compiled without OpenGL support");
#else
    CV_Assert(divs >= 1);

    if (mode == RENDER_WIREFRAME)
    {
        const float limit = 2.0f * 3.1415f;
        std::vector<float> points_data;

        for (int e = 0; e < 3; ++e)
        {
            auto ex = Vec3f::zeros();
            auto ey = Vec3f::zeros();
            ex((e + 0) % 3) = radius;
            ey((e + 1) % 3) = radius;

            for (float t = 0.0f, n; t < limit;)
            {
                n = t + limit / (divs * 4);
                points_data.insert(points_data.end(), {
                    ex(0) * cos(t) + ey(0) * sin(t), ex(1) * cos(t) + ey(1) * sin(t), ex(2) * cos(t) + ey(2) * sin(t), color(0), color(1), color(2),
                    ex(0) * cos(n) + ey(0) * sin(n), ex(1) * cos(n) + ey(1) * sin(n), ex(2) * cos(n) + ey(2) * sin(n), color(0), color(1), color(2),
                    });
                t = n;
            }
        }

        const Mat points_mat = Mat(Size(6, points_data.size() / 6), CV_32F, points_data.data());

        showLines(win_name, obj_name, points_mat);
    }
    else
    {
        std::vector<float> verts_data;

        for (int s = -1; s <= 1; s += 2)
            for (int e = 0; e < 3; ++e)
            {
                auto ex = Vec3f::zeros();
                auto ey = Vec3f::zeros();
                auto pz = Vec3f::zeros();
                ex((e + 1) % 3) = 1.0f / divs;
                ey((e + 2) % 3) = 1.0f / divs;
                pz(e) = s;

                for (int x = -divs; x < divs; ++x)
                    for (int y = -divs; y < divs; ++y)
                        if (mode == RENDER_SIMPLE)
                            verts_data.insert(verts_data.end(), {
                                ex(0) * (x + 0) + ey(0) * (y + 0) + pz(0), ex(1) * (x + 0) + ey(1) * (y + 0) + pz(1), ex(2) * (x + 0) + ey(2) * (y + 0) + pz(2), color(0), color(1), color(2),
                                ex(0) * (x + 1) + ey(0) * (y + 0) + pz(0), ex(1) * (x + 1) + ey(1) * (y + 0) + pz(1), ex(2) * (x + 1) + ey(2) * (y + 0) + pz(2), color(0), color(1), color(2),
                                ex(0) * (x + 1) + ey(0) * (y + 1) + pz(0), ex(1) * (x + 1) + ey(1) * (y + 1) + pz(1), ex(2) * (x + 1) + ey(2) * (y + 1) + pz(2), color(0), color(1), color(2),
                                ex(0) * (x + 1) + ey(0) * (y + 1) + pz(0), ex(1) * (x + 1) + ey(1) * (y + 1) + pz(1), ex(2) * (x + 1) + ey(2) * (y + 1) + pz(2), color(0), color(1), color(2),
                                ex(0) * (x + 0) + ey(0) * (y + 1) + pz(0), ex(1) * (x + 0) + ey(1) * (y + 1) + pz(1), ex(2) * (x + 0) + ey(2) * (y + 1) + pz(2), color(0), color(1), color(2),
                                ex(0) * (x + 0) + ey(0) * (y + 0) + pz(0), ex(1) * (x + 0) + ey(1) * (y + 0) + pz(1), ex(2) * (x + 0) + ey(2) * (y + 0) + pz(2), color(0), color(1), color(2),
                                });
                        else
                            verts_data.insert(verts_data.end(), {
                                ex(0) * (x + 0) + ey(0) * (y + 0) + pz(0), ex(1) * (x + 0) + ey(1) * (y + 0) + pz(1), ex(2) * (x + 0) + ey(2) * (y + 0) + pz(2), color(0), color(1), color(2), 0.0f, 0.0f, 0.0f,
                                ex(0) * (x + 1) + ey(0) * (y + 0) + pz(0), ex(1) * (x + 1) + ey(1) * (y + 0) + pz(1), ex(2) * (x + 1) + ey(2) * (y + 0) + pz(2), color(0), color(1), color(2), 0.0f, 0.0f, 0.0f,
                                ex(0) * (x + 1) + ey(0) * (y + 1) + pz(0), ex(1) * (x + 1) + ey(1) * (y + 1) + pz(1), ex(2) * (x + 1) + ey(2) * (y + 1) + pz(2), color(0), color(1), color(2), 0.0f, 0.0f, 0.0f,
                                ex(0) * (x + 1) + ey(0) * (y + 1) + pz(0), ex(1) * (x + 1) + ey(1) * (y + 1) + pz(1), ex(2) * (x + 1) + ey(2) * (y + 1) + pz(2), color(0), color(1), color(2), 0.0f, 0.0f, 0.0f,
                                ex(0) * (x + 0) + ey(0) * (y + 1) + pz(0), ex(1) * (x + 0) + ey(1) * (y + 1) + pz(1), ex(2) * (x + 0) + ey(2) * (y + 1) + pz(2), color(0), color(1), color(2), 0.0f, 0.0f, 0.0f,
                                ex(0) * (x + 0) + ey(0) * (y + 0) + pz(0), ex(1) * (x + 0) + ey(1) * (y + 0) + pz(1), ex(2) * (x + 0) + ey(2) * (y + 0) + pz(2), color(0), color(1), color(2), 0.0f, 0.0f, 0.0f,
                                });
            }

        if (mode == RENDER_SIMPLE)
        {
            for (int i = 0; i < verts_data.size() / 6; ++i)
            {
                float l = sqrtf(verts_data[6 * i + 0] * verts_data[6 * i + 0] + verts_data[6 * i + 1] * verts_data[6 * i + 1] + verts_data[6 * i + 2] * verts_data[6 * i + 2]);
                verts_data[6 * i + 0] *= radius / l;
                verts_data[6 * i + 1] *= radius / l;
                verts_data[6 * i + 2] *= radius / l;
            }

            const Mat verts_mat = Mat(Size(6, verts_data.size() / 6), CV_32F, verts_data.data());
            showMesh(win_name, obj_name, verts_mat);
        }
        else
        {
            for (int i = 0; i < verts_data.size() / 9; ++i)
            {
                float l = sqrtf(verts_data[9 * i + 0] * verts_data[9 * i + 0] + verts_data[9 * i + 1] * verts_data[9 * i + 1] + verts_data[9 * i + 2] * verts_data[9 * i + 2]);
                verts_data[9 * i + 6] = verts_data[9 * i + 0] / l;
                verts_data[9 * i + 7] = verts_data[9 * i + 1] / l;
                verts_data[9 * i + 8] = verts_data[9 * i + 2] / l;
                verts_data[9 * i + 0] *= radius / l;
                verts_data[9 * i + 1] *= radius / l;
                verts_data[9 * i + 2] *= radius / l;
            }

            const Mat verts_mat = Mat(Size(9, verts_data.size() / 9), CV_32F, verts_data.data());
            showMesh(win_name, obj_name, verts_mat);
        }
    }
#endif
}

void showCameraTrajectory(
    const String& win_name, const String& obj_name, InputArray trajectory,
    float aspect, float scale, Vec3f frustum_color, Vec3f line_color)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENGL
    CV_UNUSED(win_name);
    CV_UNUSED(obj_name);
    CV_UNUSED(trajectory);
    CV_UNUSED(aspect);
    CV_UNUSED(scale);
    CV_UNUSED(frustum_color);
    CV_UNUSED(line_color);
    CV_Error(cv::Error::OpenGlNotSupported, "The library is compiled without OpenGL support");
#else
    CV_Assert(trajectory.dims() == 2 && trajectory.cols() == 6 && trajectory.depth() == CV_32F);

    auto data = trajectory.getMat();
    std::vector<float> points_data;

    // Add frustums
    for (int i = 0; i < data.rows; ++i)
    {
        Vec3f position = { data.at<float>(i, 0), data.at<float>(i, 1), data.at<float>(i, 2) };
        Vec3f forward = normalize(Vec3f { data.at<float>(i, 3), data.at<float>(i, 4), data.at<float>(i, 5) });
        Vec3f world_up = { 0.0f, 1.0f, 0.0f };
        Vec3f right = forward.cross(world_up);
        Vec3f up = forward.cross(right);

        Vec3f back_f[4] = {
            position + (-right * aspect - up) * scale,
            position + ( right * aspect - up) * scale,
            position + ( right * aspect + up) * scale,
            position + (-right * aspect + up) * scale,
        };

        Vec3f front_f[4] = {
            position + (-right * aspect - up) * scale * 1.5f + forward * scale * 2.0f,
            position + (right * aspect - up) * scale * 1.5f + forward * scale * 2.0f,
            position + (right * aspect + up) * scale * 1.5f + forward * scale * 2.0f,
            position + (-right * aspect + up) * scale * 1.5f + forward * scale * 2.0f,
        };

        points_data.insert(points_data.end(), {
            back_f[0](0), back_f[0](1), back_f[0](2), frustum_color(0), frustum_color(1), frustum_color(2),
            back_f[1](0), back_f[1](1), back_f[1](2), frustum_color(0), frustum_color(1), frustum_color(2),
            back_f[1](0), back_f[1](1), back_f[1](2), frustum_color(0), frustum_color(1), frustum_color(2),
            back_f[2](0), back_f[2](1), back_f[2](2), frustum_color(0), frustum_color(1), frustum_color(2),
            back_f[2](0), back_f[2](1), back_f[2](2), frustum_color(0), frustum_color(1), frustum_color(2),
            back_f[3](0), back_f[3](1), back_f[3](2), frustum_color(0), frustum_color(1), frustum_color(2),
            back_f[3](0), back_f[3](1), back_f[3](2), frustum_color(0), frustum_color(1), frustum_color(2),
            back_f[0](0), back_f[0](1), back_f[0](2), frustum_color(0), frustum_color(1), frustum_color(2),

            front_f[0](0), front_f[0](1), front_f[0](2), frustum_color(0), frustum_color(1), frustum_color(2),
            front_f[1](0), front_f[1](1), front_f[1](2), frustum_color(0), frustum_color(1), frustum_color(2),
            front_f[1](0), front_f[1](1), front_f[1](2), frustum_color(0), frustum_color(1), frustum_color(2),
            front_f[2](0), front_f[2](1), front_f[2](2), frustum_color(0), frustum_color(1), frustum_color(2),
            front_f[2](0), front_f[2](1), front_f[2](2), frustum_color(0), frustum_color(1), frustum_color(2),
            front_f[3](0), front_f[3](1), front_f[3](2), frustum_color(0), frustum_color(1), frustum_color(2),
            front_f[3](0), front_f[3](1), front_f[3](2), frustum_color(0), frustum_color(1), frustum_color(2),
            front_f[0](0), front_f[0](1), front_f[0](2), frustum_color(0), frustum_color(1), frustum_color(2),

            back_f[0](0), back_f[0](1), back_f[0](2), frustum_color(0), frustum_color(1), frustum_color(2),
            front_f[0](0), front_f[0](1), front_f[0](2), frustum_color(0), frustum_color(1), frustum_color(2),
            back_f[1](0), back_f[1](1), back_f[1](2), frustum_color(0), frustum_color(1), frustum_color(2),
            front_f[1](0), front_f[1](1), front_f[1](2), frustum_color(0), frustum_color(1), frustum_color(2),
            back_f[2](0), back_f[2](1), back_f[2](2), frustum_color(0), frustum_color(1), frustum_color(2),
            front_f[2](0), front_f[2](1), front_f[2](2), frustum_color(0), frustum_color(1), frustum_color(2),
            back_f[3](0), back_f[3](1), back_f[3](2), frustum_color(0), frustum_color(1), frustum_color(2),
            front_f[3](0), front_f[3](1), front_f[3](2), frustum_color(0), frustum_color(1), frustum_color(2),
        });

    }

    // Add trajectory line
    for (int i = 1; i < data.rows; ++i)
    {
        points_data.insert(points_data.end(), {
            data.at<float>(i - 1, 0), data.at<float>(i - 1, 1), data.at<float>(i - 1, 2), line_color(0), line_color(1), line_color(2),
            data.at<float>(i, 0), data.at<float>(i, 1), data.at<float>(i, 2), line_color(0), line_color(1), line_color(2),
        });
    }

    const Mat points_mat = Mat(Size(6, points_data.size() / 6), CV_32F, points_data.data());
    showLines(win_name, obj_name, points_mat);

#endif
}

void showMesh(const String& win_name, const String& obj_name, InputArray verts, InputArray indices)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENGL
    CV_UNUSED(win_name);
    CV_UNUSED(obj_name);
    CV_UNUSED(verts);
    CV_UNUSED(indices);
    CV_Error(cv::Error::OpenGlNotSupported, "The library is compiled without OpenGL support");
#else
    Window* win = getWindow(win_name);
    Object* obj = win->get(obj_name);
    if (obj)
        delete obj;
    setOpenGlContext(win_name);
    obj = new Mesh(verts, indices);
    win->set(obj_name, obj);
    updateWindow(win_name);
#endif
}

void showMesh(const String& win_name, const String& obj_name, InputArray verts)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENGL
    CV_UNUSED(win_name);
    CV_UNUSED(obj_name);
    CV_UNUSED(verts);
    CV_Error(cv::Error::OpenGlNotSupported, "The library is compiled without OpenGL support");
#else
    Window* win = getWindow(win_name);
    Object* obj = win->get(obj_name);
    if (obj)
        delete obj;
    setOpenGlContext(win_name);
    obj = new Mesh(verts);
    win->set(obj_name, obj);
    updateWindow(win_name);
#endif
}

void showPoints(const String& win_name, const String& obj_name, InputArray points)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENGL
    CV_UNUSED(win_name);
    CV_UNUSED(obj_name);
    CV_UNUSED(points);
    CV_Error(cv::Error::OpenGlNotSupported, "The library is compiled without OpenGL support");
#else
    Window* win = getWindow(win_name);
    Object* obj = win->get(obj_name);
    if (obj)
        delete obj;
    setOpenGlContext(win_name);
    obj = new PointCloud(points);
    win->set(obj_name, obj);
    updateWindow(win_name);
#endif
}

void showRGBD(const String& win_name, const String& obj_name, InputArray img, const Matx33f& intrinsics, float scale)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENGL
    CV_UNUSED(win_name);
    CV_UNUSED(obj_name);
    CV_UNUSED(img);
    CV_UNUSED(intrinsics);
    CV_UNUSED(scale);
    CV_Error(cv::Error::OpenGlNotSupported, "The library is compiled without OpenGL support");
#else
    CV_Assert(img.dims() == 2 && img.channels() == 4 && img.type() == CV_32FC4);

    Mat mat = img.getMat();
    Mat points;

    // This section (RGBD to point cloud) should be changed to use the 3d module when
    // #20013 is merged.

    float fx = intrinsics(0, 0);
    float fy = intrinsics(1, 1);
    float cx = intrinsics(0, 2);
    float cy = intrinsics(1, 2);

    for (int u = 0; u < mat.cols; ++u)
        for (int v = 0; v < mat.rows; ++v)
        {
            Vec4f c = mat.at<Vec4f>(v, u);
            float d = c(3) * 0.001f; // mm to m

            float x_over_z = (cx - (float)u) / fx;
            float y_over_z = (cy - (float)v) / fy;
            float z = d/* / sqrt(1.0f + x_over_z * x_over_z + y_over_z * y_over_z)*/;
            float x = x_over_z * z;
            float y = y_over_z * z;

            float point[] = {
                x * scale, y * scale, z * scale,
                c(0) / 255.0f, c(1) / 255.0f, c(2) / 255.0f,
            };
            points.push_back(Mat(1, 6, CV_32F, point));
        }

    showPoints(win_name, obj_name, points);
#endif
}

void showLines(const String& win_name, const String& obj_name, InputArray points)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENGL
    CV_UNUSED(win_name);
    CV_UNUSED(obj_name);
    CV_UNUSED(points);
    CV_Error(cv::Error::OpenGlNotSupported, "The library is compiled without OpenGL support");
#else
    Window* win = getWindow(win_name);
    Object* obj = win->get(obj_name);
    if (obj)
        delete obj;
    setOpenGlContext(win_name);
    obj = new Lines(points);
    win->set(obj_name, obj);
    updateWindow(win_name);
#endif
}

void setObjectPosition(const String& win_name, const String& obj_name, const Vec3f& position)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENGL
    CV_UNUSED(win_name);
    CV_UNUSED(obj_name);
    CV_UNUSED(position);
    CV_Error(cv::Error::OpenGlNotSupported, "The library is compiled without OpenGL support");
#else
    Window* win = getWindow(win_name);
    Object* obj = win->get(obj_name);
    if (!obj)
        CV_Error(cv::Error::StsObjectNotFound, "Object not found");
    obj->setPosition(position);
    updateWindow(win_name);
#endif
}

void setObjectRotation(const String& win_name, const String& obj_name, const Vec3f& rotation)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENGL
    CV_UNUSED(win_name);
    CV_UNUSED(obj_name);
    CV_UNUSED(rotation);
    CV_Error(cv::Error::OpenGlNotSupported, "The library is compiled without OpenGL support");
#else
    Window* win = getWindow(win_name);
    Object* obj = win->get(obj_name);
    if (!obj)
        CV_Error(cv::Error::StsObjectNotFound, "Object not found");
    obj->setRotation(rotation);
    updateWindow(win_name);
#endif
}

void destroyObject(const String& win_name, const String& obj_name)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENGL
    CV_UNUSED(win_name);
    CV_UNUSED(obj_name);
    CV_Error(cv::Error::OpenGlNotSupported, "The library is compiled without OpenGL support");
#else
    Window* win = getWindow(win_name);
    win->set(obj_name, nullptr);
    updateWindow(win_name);
#endif
}

#ifdef HAVE_OPENGL

View::View()
{
    this->origin = { 0.0f, 0.0f, 0.0f };
    this->distance = 10.0f;
    this->position = { 0.0f, 0.0f, this->distance };
    this->up = { 0.0f, 1.0f, 0.0f };

    this->aspect = 1.0f;
    this->setPerspective(1.3f, 0.1f, 2000.0f);
    this->lookAt(this->origin, { 0.0f, 1.0f, 0.0f });
}

void View::setAspect(float aspect)
{
    if (this->aspect != aspect)
    {
        this->aspect = aspect;
        this->setPerspective(this->fov, this->z_near, this->z_far);
    }
}

void View::setPerspective(float fov, float z_near, float z_far)
{
    this->fov = fov;
    this->z_near = z_near;
    this->z_far = z_far;

    float tan_half_fovy = ::tan(this->fov / 2.0f);
    this->proj = Matx44f::zeros();
    this->proj(0, 0) = 1.0f / (this->aspect * tan_half_fovy);
    this->proj(1, 1) = 1.0f / tan_half_fovy;
    this->proj(2, 2) = (this->z_far + this->z_near) / (this->z_far - this->z_near);
    this->proj(2, 3) = 1.0f;
    this->proj(3, 2) = -(2.0f * this->z_far * this->z_near) / (this->z_far - this->z_near);
}

void View::rotate(float dx, float dy)
{
    this->position = normalize(this->position - this->origin);
    float theta = atan2(this->position(2), this->position(0));
    float phi = ::asin(this->position(1));
    theta -= dx * 0.05f;
    phi += dy * 0.05f;
    phi = max(-1.5f, min(1.5f, phi));

    this->position(0) = ::cos(theta) * ::cos(phi) * this->distance;
    this->position(1) = ::sin(phi) * this->distance;
    this->position(2) = ::sin(theta) * ::cos(phi) * this->distance;
    this->position += this->origin;

    this->lookAt(this->origin, this->up);
}

void View::move(float dx, float dy)
{
    Vec3f forward = normalize(this->position - this->origin);
    Vec3f right = normalize(this->up.cross(forward));
    Vec3f up = right.cross(forward);
    Vec3f delta = normalize(right * dx - up * dy) * this->distance * 0.01f;

    this->origin += delta;
    this->position += delta;
    this->lookAt(this->origin, this->up);
}

void View::scaleDistance(float amount)
{
    this->distance *= amount;
    this->distance = max(0.1f, this->distance);
    this->position = normalize(this->position - this->origin) * this->distance + this->origin;

    this->lookAt(this->origin, this->up);
}

void View::lookAt(const Vec3f& point, const Vec3f& up)
{
    Vec3f f = normalize(point - this->position);
    Vec3f s = normalize(up.cross(f));
    Vec3f u = f.cross(s);

    this->view = Matx44f::zeros();
    this->view(0, 0) = s[0];
    this->view(1, 0) = s[1];
    this->view(2, 0) = s[2];
    this->view(0, 1) = u[0];
    this->view(1, 1) = u[1];
    this->view(2, 1) = u[2];
    this->view(0, 2) = f[0];
    this->view(1, 2) = f[1];
    this->view(2, 2) = f[2];
    this->view(3, 0) = -s.dot(this->position);
    this->view(3, 1) = -u.dot(this->position);
    this->view(3, 2) = -f.dot(this->position);
    this->view(3, 3) = 1.0f;
}

Window::Window(const String& name)
{
    this->name = name;
    this->sun.direction = normalize(Vec3f(0.3f, 1.0f, 0.5f));
    this->sun.ambient = { 0.1f, 0.1f, 0.1f };
    this->sun.diffuse = { 1.0f, 1.0f, 1.0f };
    this->sky_color = { 0.0f, 0.0f, 0.0f };

	float points[] = {
		0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
		0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 1.0f,
	};

	Mat points_mat = Mat(Size(6, 6), CV_32F, points);

	this->crosshair = new Lines(points_mat);
	this->shaders[this->crosshair->getShaderName()] = this->crosshair->buildShader();
	this->crosshair->setShader(this->shaders[this->crosshair->getShaderName()]);

    this->grid = nullptr;
}

Window::~Window()
{
	delete this->crosshair;
    if (this->grid)
        delete this->grid;

    for (auto obj : this->objects)
		delete obj.second;
}

Object* Window::get(const String& obj_name)
{
	auto it = this->objects.find(obj_name);
	if (it == this->objects.end())
		return nullptr;
	return it->second;
}

void Window::set(const String& obj_name, Object* obj)
{
	auto it = this->objects.find(obj_name);
	if (it != this->objects.end() && it->second != obj)
	{
		delete it->second;

		if (obj == nullptr)
			this->objects.erase(it);
		else
			it->second = obj;
	}
	else if (obj)
		this->objects[obj_name] = obj;

    if (obj)
    {
        String name = obj->getShaderName();
        auto it = this->shaders.find(name);
        if (it == this->shaders.end())
            this->shaders[name] = obj->buildShader();
        obj->setShader(this->shaders[name]);
    }
}

void Window::setSun(const Vec3f& direction, const Vec3f& ambient, const Vec3f& diffuse)
{
    this->sun.direction = normalize(direction);
    this->sun.ambient = ambient;
    this->sun.diffuse = diffuse;
}

void Window::setSky(const Vec3f& color)
{
    this->sky_color = color;
}

static Mat getGridVertices(const View& view)
{
    const Vec3f grid_color = { 0.5f, 0.5f, 0.5f };
    const Vec3f axis_color = { 0.8f, 0.8f, 0.8f };
    const Vec3f center = view.getOrigin();
    const Vec3f camera_dir = view.getOrigin() - view.getPosition();
    const float scale = 0.3f;

    float tick_step = 1.0f;
    while (view.getDistance() * scale / tick_step > 4.0f)
        tick_step *= 2.0f;
    while (view.getDistance() * scale / tick_step < 2.0f)
        tick_step *= 0.5f;

    Mat points;
    float face_sign[3];

    const Vec3f min_p = center - Vec3f(1.0f, 1.0f, 1.0f) * view.getDistance() * scale;
    const Vec3f max_p = center + Vec3f(1.0f, 1.0f, 1.0f) * view.getDistance() * scale;

    // For each axis add a grid
    for (int ai = 0; ai < 3; ++ai)
    {
        Vec3f az = { 0.0f, 0.0f, 0.0f };
        az((ai + 2) % 3) = 1.0f;

        // Check if face is positive or negative along the az axis
        face_sign[ai] = camera_dir.dot(az) > 0.0f ? 1.0f : -1.0f;

        float x = (floor(min_p(ai) / tick_step) + 1.0f) * tick_step;
        for (; x < max_p(ai); x += tick_step)
        {
            Vec3f a = min_p;
            Vec3f b = max_p;
            a((ai + 0) % 3) = x;
            b((ai + 0) % 3) = x;
            if (face_sign[ai] > 0.0f)
                a((ai + 2) % 3) = b((ai + 2) % 3);
            else
                b((ai + 2) % 3) = a((ai + 2) % 3);

            float data[] = {
                a(0), a(1), a(2), grid_color(0), grid_color(1), grid_color(2),
                b(0), b(1), b(2), grid_color(0), grid_color(1), grid_color(2),
            };

            points.push_back(Mat(2, 6, CV_32F, data));
        }

        float y = (floor(min_p((ai + 1) % 3) / tick_step) + 1.0f) * tick_step;
        for (; y < max_p((ai + 1) % 3); y += tick_step)
        {
            Vec3f a = min_p;
            Vec3f b = max_p;
            a((ai + 1) % 3) = y;
            b((ai + 1) % 3) = y;
            if (face_sign[ai] > 0.0f)
                a((ai + 2) % 3) = b((ai + 2) % 3);
            else
                b((ai + 2) % 3) = a((ai + 2) % 3);

            float data[] = {
                a(0), a(1), a(2), grid_color(0), grid_color(1), grid_color(2),
                b(0), b(1), b(2), grid_color(0), grid_color(1), grid_color(2),
            };

            points.push_back(Mat(2, 6, CV_32F, data));
        }
    }

    // Draw Ox, Oy and Oz axes and ticks
    {
        Vec3f a = { face_sign[1] > 0.0f ? min_p(0) : max_p(0), face_sign[2] > 0.0f ? max_p(1) : min_p(1), face_sign[0] > 0.0f ? max_p(2) : min_p(2) };
        Vec3f b = { face_sign[1] > 0.0f ? min_p(0) : max_p(0), face_sign[2] > 0.0f ? max_p(1) : min_p(1), face_sign[0] > 0.0f ? min_p(2) : max_p(2) };
        Vec3f c = { face_sign[1] > 0.0f ? max_p(0) : min_p(0), face_sign[2] > 0.0f ? max_p(1) : min_p(1), face_sign[0] > 0.0f ? min_p(2) : max_p(2) };
        Vec3f d = { face_sign[1] > 0.0f ? max_p(0) : min_p(0), face_sign[2] > 0.0f ? min_p(1) : max_p(1), face_sign[0] > 0.0f ? min_p(2) : max_p(2) };

        float data[] = {
            a(0), a(1), a(2), 0.0f, 0.0f, 0.8f,
            b(0), b(1), b(2), 0.0f, 0.0f, 0.8f,
            b(0), b(1), b(2), 0.8f, 0.0f, 0.0f,
            c(0), c(1), c(2), 0.8f, 0.0f, 0.0f,
            c(0), c(1), c(2), 0.0f, 0.8f, 0.0f,
            d(0), d(1), d(2), 0.0f, 0.8f, 0.0f,
        };

        points.push_back(Mat(6, 6, CV_32F, data));

        float x = (floor(min_p(0) / tick_step) + 1.0f) * tick_step;
        for (; x < max_p(0); x += tick_step)
        {
            Vec3f a, b, c;
            a(0) = b(0) = x;
            a(1) = b(1) = face_sign[2] > 0.0f ? max_p(1) : min_p(1);
            a(2) = face_sign[0] > 0.0f ? min_p(2) : max_p(2);
            b(2) = a(2) - face_sign[0] * 0.03f * scale * view.getDistance();

            float line[] = {
                a(0), a(1), a(2), 0.8f, 0.0f, 0.0f,
                b(0), b(1), b(2), 0.8f, 0.0f, 0.0f,
            };

            points.push_back(Mat(2, 6, CV_32F, line));
        }

        float y = (floor(min_p(1) / tick_step) + 1.0f) * tick_step;
        for (; y < max_p(1); y += tick_step)
        {
            Vec3f a, b;
            a(0) = b(0) = face_sign[1] > 0.0f ? max_p(0) : min_p(0);
            a(1) = b(1) = y;
            a(2) = face_sign[0] > 0.0f ? min_p(2) : max_p(2);
            b(2) = a(2) - face_sign[0] * 0.03f * scale * view.getDistance();

            float line[] = {
                a(0), a(1), a(2), 0.0f, 0.8f, 0.0f,
                b(0), b(1), b(2), 0.0f, 0.8f, 0.0f,
            };

            points.push_back(Mat(2, 6, CV_32F, line));
        }

        float z = (floor(min_p(2) / tick_step) + 1.0f) * tick_step;
        for (; z < max_p(2); z += tick_step)
        {
            Vec3f a, b;
            a(0) = face_sign[1] > 0.0f ? min_p(0) : max_p(0);
            b(0) = a(0) - face_sign[1] * 0.03f * scale * view.getDistance();
            a(1) = b(1) = face_sign[2] > 0.0f ? max_p(1) : min_p(1);
            a(2) = b(2) = z;

            float line[] = {
                a(0), a(1), a(2), 0.0f, 0.0f, 0.8f,
                b(0), b(1), b(2), 0.0f, 0.0f, 0.8f,
            };

            points.push_back(Mat(2, 6, CV_32F, line));
        }
    }

    return points;
}

void Window::setGridVisible(bool visible)
{
    if (visible)
    {
        this->grid = new Lines(Mat(4096, 6, CV_32F), 0);
        this->grid->setShader(this->shaders[this->grid->getShaderName()]);
    }
    else if (this->grid)
    {
        delete this->grid;
        this->grid = nullptr;
    }
}

void Window::draw()
{
    Rect rect = getWindowImageRect(this->name);
    float aspect = (float)rect.width / (float)rect.height;
    this->view.setAspect(aspect);

    ogl::enable(ogl::DEPTH_TEST);
    ogl::clearColor({ double(this->sky_color(0) * 255.0f), double(this->sky_color(1) * 255.0f), double(this->sky_color(2) * 255.0f), 255.0 });

    if (this->grid)
    {
        Mat labels;
        static_cast<Lines*>(this->grid)->update(getGridVertices(this->view));
        this->grid->draw(this->view, this->sun);
    }
    else
    {
        this->crosshair->setPosition(this->view.getOrigin());
        this->crosshair->draw(this->view, this->sun);
    }

    for (auto& obj : this->objects)
        obj.second->draw(this->view, this->sun);
}

void Window::onMouse(int event, int x, int y, int flags)
{
    if (event == EVENT_LBUTTONDOWN || event == EVENT_RBUTTONDOWN)
    {
        this->l_mouse_x = x;
        this->l_mouse_y = y;
    }
    else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
    {
        this->view.rotate(x - this->l_mouse_x, y - this->l_mouse_y);
        updateWindow(this->name);
        this->l_mouse_x = x;
        this->l_mouse_y = y;
    }
    else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_RBUTTON))
    {
        this->view.move(x - this->l_mouse_x, y - this->l_mouse_y);
        updateWindow(this->name);
        this->l_mouse_x = x;
        this->l_mouse_y = y;
    }
    else if (event == EVENT_MOUSEWHEEL)
    {
        this->view.scaleDistance(min(1.2f, max(0.8f, 1.0f - getMouseWheelDelta(flags) / 12.0f)));
        updateWindow(this->name);
    }
}

Object::Object()
{
	this->position = { 0.0f, 0.0f, 0.0f };
	this->rotation = { 0.0f, 0.0f, 0.0f };
	this->model = Matx44f::eye();
}

void Object::setPosition(const Vec3f& position)
{
    this->position = position;
    this->updateModel();
}

void Object::setRotation(const Vec3f& rotation)
{
    this->rotation = rotation;
    this->updateModel();
}

void Object::updateModel()
{
    // Calculate rotation matrices
    Matx44f rot_a = Matx44f::eye();
    rot_a(0, 0) = ::cos(this->rotation(0));
    rot_a(1, 0) = ::sin(this->rotation(0));
    rot_a(0, 1) = -::sin(this->rotation(0));
    rot_a(1, 1) = ::cos(this->rotation(0));

    Matx44f rot_b = Matx44f::eye();
    rot_b(1, 1) = ::cos(this->rotation(1));
    rot_b(2, 1) = ::sin(this->rotation(1));
    rot_b(1, 2) = -::sin(this->rotation(1));
    rot_b(2, 2) = ::cos(this->rotation(1));

    Matx44f rot_c = Matx44f::eye();
    rot_c(0, 0) = ::cos(this->rotation(2));
    rot_c(2, 0) = ::sin(this->rotation(2));
    rot_c(0, 2) = -::sin(this->rotation(2));
    rot_c(2, 2) = ::cos(this->rotation(2));

    // Calculate translation matrix
    Matx44f trans = Matx44f::eye();
    trans(3, 0) = this->position(0);
    trans(3, 1) = this->position(1);
    trans(3, 2) = this->position(2);

    // Multiply matrices
    this->model = rot_c * rot_b * rot_a * trans;
}

Mesh::Mesh(InputArray verts, InputArray indices)
{
    // Check parameter validity
    CV_Assert(verts.channels() == 1 && verts.dims() == 2 && (verts.size().width == 3 || verts.size().width == 6 || verts.size().width == 9));
    CV_Assert(verts.depth() == CV_32F);
    CV_Assert(indices.channels() == 1 && indices.dims() == 2 && indices.size().width == 3);
    CV_Assert(indices.depth() == CV_8U || indices.depth() == CV_16U || indices.depth() == CV_32S);

    // Prepare buffers
    if (verts.kind() == _InputArray::OPENGL_BUFFER)
        this->verts = verts.getOGlBuffer();
    else
        this->verts.copyFrom(verts, ogl::Buffer::ARRAY_BUFFER);

    if (indices.kind() == _InputArray::OPENGL_BUFFER)
        this->indices = indices.getOGlBuffer();
    else
        this->indices.copyFrom(indices, ogl::Buffer::ELEMENT_ARRAY_BUFFER);

    switch (indices.depth())
    {
    case CV_8U:
        this->index_type = ogl::UNSIGNED_BYTE;
        break;
    case CV_16U:
        this->index_type = ogl::UNSIGNED_SHORT;
        break;
    case CV_32S:
        this->index_type = ogl::UNSIGNED_INT;
        break;
    }

    // Prepare vertex array
    if (verts.size().width == 3)
    {
        this->va.create({
            {
                this->verts,
                3 * sizeof(float), 0,
                3, ogl::Attribute::FLOAT,
                false, false,
                0
            }
        });
    }
    else if (verts.size().width == 6)
    {
        this->va.create({
		    {
			    this->verts,
			    6 * sizeof(float), 0,
			    3, ogl::Attribute::FLOAT,
			    false, false,
			    0
            },
		    {
			    this->verts,
			    6 * sizeof(float), 3 * sizeof(float),
			    3, ogl::Attribute::FLOAT,
			    false, false,
			    1
            }
	    });
    }
    else if (verts.size().width == 9)
    {
        this->va.create({
		    {
			    this->verts,
			    9 * sizeof(float), 0,
			    3, ogl::Attribute::FLOAT,
			    false, false,
			    0
            },
		    {
			    this->verts,
			    9 * sizeof(float), 3 * sizeof(float),
			    3, ogl::Attribute::FLOAT,
			    false, false,
			    1
            },
            {
                this->verts,
                9 * sizeof(float), 6 * sizeof(float),
                3, ogl::Attribute::FLOAT,
                false, false,
                2
            }
	    });
    }
}

Mesh::Mesh(InputArray verts)
{
    // Check parameter validity
    CV_Assert(verts.channels() == 1 && verts.dims() == 2 && (verts.size().width == 3 || verts.size().width == 6 || verts.size().width == 9));
    CV_Assert(verts.depth() == CV_32F);

    // Prepare buffers
    if (verts.kind() == _InputArray::OPENGL_BUFFER)
        this->verts = verts.getOGlBuffer();
    else
        this->verts.copyFrom(verts, ogl::Buffer::ARRAY_BUFFER);

    this->index_type = 0;

    // Prepare vertex array
    if (verts.size().width == 3)
    {
        this->va.create({
            {
                this->verts,
                3 * sizeof(float), 0,
                3, ogl::Attribute::FLOAT,
                false, false,
                0
            }
            });
    }
    else if (verts.size().width == 6)
    {
        this->va.create({
            {
                this->verts,
                6 * sizeof(float), 0,
                3, ogl::Attribute::FLOAT,
                false, false,
                0
            },
            {
                this->verts,
                6 * sizeof(float), 3 * sizeof(float),
                3, ogl::Attribute::FLOAT,
                false, false,
                1
            }
        });
    }
    else if (verts.size().width == 9)
    {
        this->va.create({
            {
                this->verts,
                9 * sizeof(float), 0,
                3, ogl::Attribute::FLOAT,
                false, false,
                0
            },
            {
                this->verts,
                9 * sizeof(float), 3 * sizeof(float),
                3, ogl::Attribute::FLOAT,
                false, false,
                1
            },
            {
                this->verts,
                9 * sizeof(float), 6 * sizeof(float),
                3, ogl::Attribute::FLOAT,
                false, false,
                2
            }
        });
    }
}

void Mesh::draw(const View& view, const Light& light)
{
    this->program.bind();
    this->va.bind();
    ogl::Program::setUniformMat4x4(this->model_loc, this->getModel());
    ogl::Program::setUniformMat4x4(this->view_loc, view.getView());
    ogl::Program::setUniformMat4x4(this->proj_loc, view.getProj());

    if (this->sun_direction_loc != -1)
    {
        ogl::Program::setUniformVec3(this->sun_direction_loc, light.direction);
        ogl::Program::setUniformVec3(this->sun_ambient_loc, light.ambient);
        ogl::Program::setUniformVec3(this->sun_diffuse_loc, light.diffuse);
    }

    if (this->index_type == 0)
        ogl::drawArrays(0, this->verts.size().height, ogl::TRIANGLES);
    else
    {
        this->indices.bind(ogl::Buffer::ELEMENT_ARRAY_BUFFER);
        ogl::drawElements(0, this->indices.size().area(), this->index_type, ogl::TRIANGLES);
    }
}

String Mesh::getShaderName()
{
    if (this->verts.size().width == 3)
        return "mesh-xyz";
    else if (this->verts.size().width == 6)
        return "mesh-xyz-rgb";
    else
        return "mesh-xyz-rgb-uvw";
}

ogl::Program Mesh::buildShader()
{
    ogl::Shader vs, fs;

	// Setup shader pipeline
    if (this->verts.size().width == 3)
    {
        vs = ogl::Shader(R"(
            #version 330 core

            layout (location = 0) in vec3 vert_pos;

            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 proj;

            void main() {
                gl_Position = vec4(vert_pos, 1.0) * model * view * proj;
            }
        )", ogl::Shader::VERTEX);

        fs = ogl::Shader(R"(
            #version 330 core

            out vec4 frag_color;

            void main() {
                frag_color = vec4(1.0, 1.0, 1.0, 1.0);
            }
        )", ogl::Shader::FRAGMENT);
    }
    else if (this->verts.size().width == 6)
    {
        vs = ogl::Shader(R"(
            #version 330 core

            layout (location = 0) in vec3 vert_pos;
            layout (location = 1) in vec3 vert_color;

            out vec3 frag_color;

            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 proj;

            void main() {
                frag_color = vert_color;
                gl_Position = vec4(vert_pos, 1.0) * model * view * proj;
            }
        )", ogl::Shader::VERTEX);

	    fs = ogl::Shader(R"(
            #version 330 core

            in vec3 frag_color;

            out vec4 color;

            void main() {
                color = vec4(frag_color, 1.0);
            }
        )", ogl::Shader::FRAGMENT);
    }
    else if (this->verts.size().width == 9)
    {
        vs = ogl::Shader(R"(
            #version 330 core

            layout (location = 0) in vec3 vert_pos;
            layout (location = 1) in vec3 vert_color;
            layout (location = 2) in vec3 vert_normal;

            out vec3 frag_color;
            out vec3 frag_normal;

            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 proj;

            void main() {
                frag_color = vert_color;
                frag_normal = vert_normal * mat3(transpose(inverse(model)));
                gl_Position = vec4(vert_pos, 1.0) * model * view * proj;
            }
        )", ogl::Shader::VERTEX);

        fs = ogl::Shader(R"(
            #version 330 core

            in vec3 frag_color;
            in vec3 frag_normal;

            out vec4 color;

            uniform vec3 sun_direction;
            uniform vec3 sun_ambient;
            uniform vec3 sun_diffuse;

            void main() {
                float diff = max(dot(frag_normal, sun_direction), 0.0);
                vec3 ambient = frag_color * sun_ambient;
                vec3 diffuse = frag_color * diff * sun_diffuse;
                color = vec4(ambient + diffuse, 1.0);
            }
        )", ogl::Shader::FRAGMENT);
    }

    return ogl::Program(vs, fs);
}

void Mesh::setShader(ogl::Program program)
{
	this->program = program;
	this->model_loc = this->program.getUniformLocation("model");
	this->view_loc = this->program.getUniformLocation("view");
	this->proj_loc = this->program.getUniformLocation("proj");
    this->sun_direction_loc = -1;
    if (this->verts.size().width == 9)
    {
        this->sun_direction_loc = this->program.getUniformLocation("sun_direction");
        this->sun_ambient_loc = this->program.getUniformLocation("sun_ambient");
        this->sun_diffuse_loc = this->program.getUniformLocation("sun_diffuse");
    }
}

Lines::Lines(InputArray points, int count)
{
	// Check parameter validity
	CV_Assert(points.channels() == 1 && points.dims() == 2 && points.size().width == 6);
	CV_Assert(points.depth() == CV_32F);

	// Prepare buffers
	if (points.kind() == _InputArray::OPENGL_BUFFER)
		this->points = points.getOGlBuffer();
    else
    {
        this->points.create(points.size(), points.type(), ogl::Buffer::ARRAY_BUFFER);
        if (count == -1 || count > 0)
            this->points.copyFrom(points, ogl::Buffer::ARRAY_BUFFER);
    }

	// Prepare vertex array
	this->va.create({
		{
			this->points,
			6 * sizeof(float), 0,
			3, ogl::Attribute::FLOAT,
			false, false,
			0
        },
		{
			this->points,
			6 * sizeof(float), 3 * sizeof(float),
			3, ogl::Attribute::FLOAT,
			false, false,
			1
        }
	});

    if (count == -1)
        this->count = this->points.size().height;
    else
        this->count = count;
}

void Lines::draw(const View& view, const Light& light)
{
    CV_UNUSED(light);

    if (this->count > 0)
    {
        this->program.bind();
        this->va.bind();

        ogl::Program::setUniformMat4x4(this->model_loc, this->getModel());
        ogl::Program::setUniformMat4x4(this->view_loc, view.getView());
        ogl::Program::setUniformMat4x4(this->proj_loc, view.getProj());

        ogl::drawArrays(0, this->count, ogl::LINES);
    }
}

void Lines::update(InputArray points)
{
    // Check parameter validity
    CV_Assert(points.channels() == 1 && points.dims() == 2 && points.size().width == 6);
    CV_Assert(points.depth() == CV_32F);

    this->points.copyFrom(points, ogl::Buffer::ARRAY_BUFFER);
    this->count = points.size().height;
}

String Lines::getShaderName()
{
	return "lines";
}

ogl::Program Lines::buildShader()
{
	// Setup shader pipeline
	auto vs = ogl::Shader(R"(
        #version 330 core

        layout (location = 0) in vec3 vert_pos;
        layout (location = 1) in vec3 vert_color;

        out vec3 frag_color;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 proj;

        void main() {
            frag_color = vert_color;
            gl_Position = vec4(vert_pos, 1.0) * model * view * proj;
        }
    )", ogl::Shader::VERTEX);

	auto fs = ogl::Shader(R"(
        #version 330 core

        in vec3 frag_color;

        out vec4 color;

        void main() {
            color = vec4(frag_color, 1.0);
        }
    )", ogl::Shader::FRAGMENT);

	return ogl::Program(vs, fs);
}

void Lines::setShader(ogl::Program program)
{
	this->program = program;
	this->model_loc = this->program.getUniformLocation("model");
	this->view_loc = this->program.getUniformLocation("view");
	this->proj_loc = this->program.getUniformLocation("proj");
}

PointCloud::PointCloud(InputArray points)
{
    // Check parameter validity
    CV_Assert(points.channels() == 1 && points.dims() == 2 && points.size().width == 6);
    CV_Assert(points.depth() == CV_32F);

    // Prepare buffers
    if (points.kind() == _InputArray::OPENGL_BUFFER)
        this->points = points.getOGlBuffer();
    else
        this->points.copyFrom(points, ogl::Buffer::ARRAY_BUFFER);

    // Prepare vertex array
    this->va.create({
        {
            this->points,
            6 * sizeof(float), 0,
            3, ogl::Attribute::FLOAT,
            false, false,
            0
        },
        {
            this->points,
            6 * sizeof(float), 3 * sizeof(float),
            3, ogl::Attribute::FLOAT,
            false, false,
            1
        }
    });
}

void PointCloud::draw(const View& view, const Light& light)
{
    CV_UNUSED(light);

    this->program.bind();
    this->va.bind();

    ogl::Program::setUniformMat4x4(this->model_loc, this->getModel());
    ogl::Program::setUniformMat4x4(this->view_loc, view.getView());
    ogl::Program::setUniformMat4x4(this->proj_loc, view.getProj());

    ogl::drawArrays(0, this->points.size().height, ogl::POINTS);
}

String PointCloud::getShaderName()
{
	return "points";
}

ogl::Program PointCloud::buildShader()
{
	// Setup shader pipeline
	auto vs = ogl::Shader(R"(
        #version 330 core

        layout (location = 0) in vec3 vert_pos;
        layout (location = 1) in vec3 vert_color;

        out vec3 frag_color;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 proj;

        void main() {
            frag_color = vert_color;
            gl_Position = vec4(vert_pos, 1.0) * model * view * proj;
        }
    )", ogl::Shader::VERTEX);

	auto fs = ogl::Shader(R"(
        #version 330 core

        in vec3 frag_color;

        out vec4 color;

        void main() {
            color = vec4(frag_color, 1.0);
        }
    )", ogl::Shader::FRAGMENT);

	return ogl::Program(vs, fs);
}

void PointCloud::setShader(ogl::Program program)
{
	this->program = program;
	this->model_loc = this->program.getUniformLocation("model");
	this->view_loc = this->program.getUniformLocation("view");
	this->proj_loc = this->program.getUniformLocation("proj");
}

} // namespace viz3d
} // namespace cv

#endif
