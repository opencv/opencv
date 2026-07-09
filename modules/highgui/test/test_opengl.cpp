// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/core/opengl.hpp>

namespace opencv_test { namespace {

using namespace cv;

// The ogl wrappers need a current GL context. Create an OpenGL window to obtain
// one, or skip where none is available (headless CI without a display/GL).
static void ensureGlContext(const String& w)
{
    try
    {
        namedWindow(w, WINDOW_OPENGL);
        if (getWindowProperty(w, WND_PROP_OPENGL) <= 0)
            throw cvtest::SkipTestException("window created without an OpenGL context");
        setOpenGlContext(w);
    }
    catch (const cv::Exception&)
    {
        throw cvtest::SkipTestException("OpenGL/display not available");
    }
}

// Regression for the in-place Buffer::resize double-free fix: growing a
// sole-owner buffer must keep the same GL id (bound VAOs stay valid).
TEST(OpenGL_Buffer, resize_keeps_id)
{
    const String w = "ogl_buf_test";
    ensureGlContext(w);

    ogl::Buffer buf;
    buf.create(4, 3, CV_32F, ogl::Buffer::ARRAY_BUFFER, false);
    unsigned int id1 = buf.bufId();
    EXPECT_NE(id1, 0u);

    buf.create(16, 3, CV_32F, ogl::Buffer::ARRAY_BUFFER, false);   // grow -> resize in place
    EXPECT_EQ(buf.bufId(), id1);

    destroyWindow(w);
}

// Exercises Shader compile + Shader::type() (the previously-undefined symbol).
TEST(OpenGL_Shader, compile_and_type)
{
    const String w = "ogl_shader_test";
    ensureGlContext(w);

    const char* vs = "#version 330 core\nvoid main(){ gl_Position = vec4(0.0); }";
    ogl::Shader sh(vs, ogl::Shader::VERTEX, false);
    EXPECT_NE(sh.shaderId(), 0u);
    EXPECT_EQ(sh.type(), ogl::Shader::VERTEX);

    destroyWindow(w);
}

}} // namespace
