/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

inline void verify_size(const std::string &nm, const cv::Mat &img)
{
    EXPECT_NO_THROW(imshow(nm, img));
    EXPECT_EQ(-1, waitKey(200));

    // see https://github.com/opencv/opencv/issues/25550
    // Wayland backend is not supported getWindowImageRect().
    string framework;
    EXPECT_NO_THROW(framework = currentUIFramework());
    if(framework == "WAYLAND")
    {
       return;
    }

    Rect rc;
    EXPECT_NO_THROW(rc = getWindowImageRect(nm));
    EXPECT_EQ(rc.size(), img.size());
}

#if (!defined(ENABLE_PLUGINS) \
        && !defined HAVE_GTK \
        && !defined HAVE_QT \
        && !defined HAVE_WIN32UI \
        && !defined HAVE_COCOA \
        && !defined HAVE_WAYLAND \
    )
TEST(Highgui_GUI, DISABLED_regression)
#else
TEST(Highgui_GUI, regression)
#endif
{
    const std::string window_name("opencv_highgui_test_window");
    const cv::Size image_size(800, 600);

    EXPECT_NO_THROW(destroyAllWindows());
    ASSERT_NO_THROW(namedWindow(window_name));
    const vector<int> channels = {1, 3, 4};
    const vector<int> depths = {CV_8U, CV_8S, CV_16U, CV_16S, CV_32F, CV_64F};
    for(int cn : channels)
    {
        SCOPED_TRACE(cn);
        for(int depth : depths)
        {
            SCOPED_TRACE(depth);
            double min_val = 0.;
            double max_val = 256.;
            switch(depth)
            {
            case CV_8S:
                min_val = static_cast<double>(-0x7F);
                max_val = static_cast<double>(0x7F + 1);
                break;
            case CV_16S:
                min_val = static_cast<double>(-0x7FFF);
                max_val = static_cast<double>(0x7FFF + 1);
                break;
            case CV_16U:
                max_val = static_cast<double>(0xFFFF + 1);
                break;
            case CV_32F:
            case CV_64F:
                max_val = 1.0;
                break;
            }
            Mat m = cvtest::randomMat(TS::ptr()->get_rng(), image_size, CV_MAKE_TYPE(depth, cn), min_val, max_val, false);
            verify_size(window_name, m);

            Mat bgr(image_size, CV_MAKE_TYPE(depth, cn));
            int b_g = image_size.width / 3, g_r = b_g * 2;
            if (cn > 1)
            {
                bgr.colRange(0, b_g).setTo(cv::Scalar(max_val, min_val, min_val));
                bgr.colRange(b_g, g_r).setTo(cv::Scalar(min_val, max_val, min_val));
                bgr.colRange(g_r, image_size.width).setTo(cv::Scalar(min_val, min_val, max_val));
            }
            else
            {
                bgr.colRange(0, b_g).setTo(cv::Scalar::all(min_val));
                bgr.colRange(b_g, g_r).setTo(cv::Scalar::all((min_val + max_val) / 2));
                bgr.colRange(g_r, image_size.width).setTo(cv::Scalar::all(max_val));
            }
            verify_size(window_name, bgr);
        }
    }
    EXPECT_NO_THROW(destroyAllWindows());
}

//==================================================================================================

static void Foo(int, void* counter)
{
    if (counter)
    {
        int *counter_int = static_cast<int*>(counter);
        (*counter_int)++;
    }
}

#if (!defined(ENABLE_PLUGINS) \
        && !defined HAVE_GTK \
        && !defined HAVE_QT \
        && !defined HAVE_WIN32UI \
        && !defined HAVE_WAYLAND \
    ) \
    || defined(__APPLE__)  /* test fails on Mac (cocoa) */ \
    || defined HAVE_FRAMEBUFFER /* trackbar is not supported */
TEST(Highgui_GUI, DISABLED_trackbar_unsafe)
#else
TEST(Highgui_GUI, trackbar_unsafe)
#endif
{
    int value = 50;
    int callback_count = 0;
    const std::string window_name("trackbar_test_window");
    const std::string trackbar_name("trackbar");

    EXPECT_NO_THROW(destroyAllWindows());
    ASSERT_NO_THROW(namedWindow(window_name));
    EXPECT_EQ((int)1, createTrackbar(trackbar_name, window_name, &value, 100, Foo, &callback_count));
    EXPECT_EQ(value, getTrackbarPos(trackbar_name, window_name));
    EXPECT_GE(callback_count, 0);
    EXPECT_LE(callback_count, 1);
    int callback_count_base = callback_count;
    EXPECT_NO_THROW(setTrackbarPos(trackbar_name, window_name, 90));
    EXPECT_EQ(callback_count_base + 1, callback_count);
    EXPECT_EQ(90, value);
    EXPECT_EQ(90, getTrackbarPos(trackbar_name, window_name));
    EXPECT_NO_THROW(destroyAllWindows());
}

static
void testTrackbarCallback(int pos, void* param)
{
    CV_Assert(param);
    int* status = (int*)param;
    status[0] = pos;
    status[1]++;
}

#if (!defined(ENABLE_PLUGINS) \
        && !defined HAVE_GTK \
        && !defined HAVE_QT \
        && !defined HAVE_WIN32UI \
        && !defined HAVE_WAYLAND \
    ) \
    || defined(__APPLE__) /* test fails on Mac (cocoa) */ \
    || defined HAVE_FRAMEBUFFER /* trackbar is not supported */
TEST(Highgui_GUI, DISABLED_trackbar)
#else
TEST(Highgui_GUI, trackbar)
#endif
{
    int status[2] = {-1, 0};  // pos, counter
    const std::string window_name("trackbar_test_window");
    const std::string trackbar_name("trackbar");

    EXPECT_NO_THROW(destroyAllWindows());
    ASSERT_NO_THROW(namedWindow(window_name));
    EXPECT_EQ((int)1, createTrackbar(trackbar_name, window_name, NULL, 100, testTrackbarCallback, status));
    EXPECT_EQ(0, getTrackbarPos(trackbar_name, window_name));
    int callback_count = status[1];
    EXPECT_GE(callback_count, 0);
    EXPECT_LE(callback_count, 1);
    int callback_count_base = callback_count;
    EXPECT_NO_THROW(setTrackbarPos(trackbar_name, window_name, 90));
    callback_count = status[1];
    EXPECT_EQ(callback_count_base + 1, callback_count);
    int value = status[0];
    EXPECT_EQ(90, value);
    EXPECT_EQ(90, getTrackbarPos(trackbar_name, window_name));
    EXPECT_NO_THROW(destroyAllWindows());
}

// See https://github.com/opencv/opencv/issues/25560
#if (!defined(ENABLE_PLUGINS) \
        && !defined HAVE_GTK \
        && !defined HAVE_QT \
        && !defined HAVE_WIN32UI \
        && !defined HAVE_WAYLAND)
TEST(Highgui_GUI, DISABLED_small_width_image)
#else
TEST(Highgui_GUI, small_width_image)
#endif
{
    const std::string window_name("trackbar_test_window");
    cv::Mat src(1,1,CV_8UC3,cv::Scalar(0));
    EXPECT_NO_THROW(destroyAllWindows());
    ASSERT_NO_THROW(namedWindow(window_name));
    ASSERT_NO_THROW(imshow(window_name, src));
    EXPECT_NO_THROW(waitKey(10));
    EXPECT_NO_THROW(destroyAllWindows());
}

TEST(Highgui_GUI, currentUIFramework)
{
    auto framework = currentUIFramework();
    std::cout << "UI framework: \"" << framework << "\"" << std::endl;
#if (!defined(ENABLE_PLUGINS) \
        && !defined HAVE_GTK \
        && !defined HAVE_QT \
        && !defined HAVE_WIN32UI \
        && !defined HAVE_COCOA \
        && !defined HAVE_WAYLAND \
    )
    EXPECT_TRUE(framework.empty());
#elif !defined(ENABLE_PLUGINS)
    EXPECT_GT(framework.size(), 0);  // builtin backends
#endif
}
#if (!defined(ENABLE_PLUGINS) \
        && !defined HAVE_GTK \
        && !defined HAVE_QT \
        && !defined HAVE_WIN32UI \
        && !defined HAVE_COCOA \
        && !defined HAVE_WAYLAND \
    ) || !defined(HAVE_OPENGL)
TEST(Highgui_GUI, DISABLED_gl)
#else
#include<epoxy/gl.h>
#include<GL/gl.h>
static GLuint create_shader(const char* source, GLenum type) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    return shader;
}
struct DrawData
{
    GLuint vao, vbo, program;
};
void draw(void* userdata){
    DrawData* data = static_cast<DrawData*>(userdata);
    glBindVertexArray(data->vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
}
TEST(Highgui_GUI, gl)
#endif
{
    const std::string window_name("gl_test_window");
    const Size image_size(800, 600);
    EXPECT_NO_THROW(destroyAllWindows());
    ASSERT_NO_THROW(namedWindow(window_name, WINDOW_OPENGL));
    ASSERT_NO_THROW(resizeWindow(window_name, image_size));
    DrawData data;
    const char *vertex_shader_source =
            "#version 330 core\n"
            "layout (location = 0) in vec3 position;\n"
            "void main() {\n"
            "   gl_Position = vec4(position, 1.0);\n"
            "}\n";
    const char *fragment_shader_source =
            "#version 330 core\n"
            "out vec4 color;\n"
            "void main() {\n"
            "   color = vec4(1.0, 1.0, 1.0, 1.0); // white\n"
            "}\n";
    GLuint vertex_shader = create_shader(vertex_shader_source, GL_VERTEX_SHADER);
    GLuint fragment_shader = create_shader(fragment_shader_source, GL_FRAGMENT_SHADER);
    data.program = glCreateProgram();
    glAttachShader(data.program, vertex_shader);
    glAttachShader(data.program, fragment_shader);
    glLinkProgram(data.program);
    glUseProgram(data.program);
    GLfloat vertices[] = {
            0.0f,  0.5f, 0.0f,
            -0.5f, -0.5f, 0.0f,
            0.5f, -0.5f, 0.0f
    };
    glGenVertexArrays(1, &data.vao);
    glBindVertexArray(data.vao);
    glGenBuffers(1, &data.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, data.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
    ASSERT_NO_THROW(setOpenGlDrawCallback(window_name, draw, &data));
    EXPECT_NO_THROW(waitKey(10000));
    EXPECT_NO_THROW(setOpenGlDrawCallback(window_name, 0));
    EXPECT_NO_THROW(destroyAllWindows());
}
}} // namespace
