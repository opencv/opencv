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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//     Wenju He, wenju@multicorewareinc.com
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

#include "precomp.hpp"
#include "opencv2/core/opengl.hpp"
#include "opencv2/highgui.hpp"

#ifdef HAVE_OPENCL
#  ifdef __APPLE__
#    include <OpenCL/cl_gl_ext.h>
#  else
#    include <CL/cl_gl_ext.h>
#  endif
#endif

#if defined WIN32 || defined _WIN32 || defined WINCE
#  include <windows.h>
#  undef small
#  undef min
#  undef max
#  undef abs
#endif

using namespace cv;
using namespace cv::ocl;

#if defined WIN32 || defined _WIN32
//! parent window, HDC and associated OpenGL context,
// this parent context is shared to ocl::imshow display windows
static HWND g_hWnd = NULL;
static HWND g_mainhWnd = NULL;
static HDC g_hDC = NULL;
static HGLRC g_hGLRC = NULL;
#endif

#if defined HAVE_OPENGL && defined HAVE_OPENCL

///////////////////////////////////////////////////////////////////////////////
namespace
{
    //! Pointer for OpenGL buffer memory.
    class ClGlBuffer: public ogl::Buffer
    {
    public:
        ClGlBuffer();

        //! copy from device memory
        void copyFrom(const oclMat& mat);

        void release();
    private:
        cl_mem clbuffer_;
        bool initialized_;
    };

    //! pointer for OpenGL 2d texture memory with reference counting.
    class ClGlTexture: public ogl::Texture2D
    {
    public:
        ClGlTexture();

        //! copy from buffer object
        void copyFrom(const ClGlBuffer& buf);
    };

    ClGlBuffer::ClGlBuffer() : clbuffer_((cl_mem)NULL), initialized_(false)
    {
    }

    void ClGlBuffer::copyFrom(const oclMat& mat)
    {
        const int atype = mat.ocltype();
        const int depth = CV_MAT_DEPTH(atype);
        const int cn = CV_MAT_CN(atype);

        CV_Assert( depth <= CV_32F );
        CV_Assert( cn == 1 || cn == 3 || cn == 4 );
        CV_Assert( mat.isContinuous() );

        if ((rows() != mat.rows) && (cols() != mat.cols) && (type() != atype))
            initialized_ = false;

        create(mat.rows, mat.cols, atype);

        cl_context ctx = *((cl_context*) getoclContext());
        cl_command_queue cq = *((cl_command_queue*) getoclCommandQueue());

        if (!initialized_)
        {
            if (clbuffer_ != (cl_mem) NULL)
                openCLSafeCall(clReleaseMemObject(clbuffer_));

            cl_int status;

            // Prior to calling clEnqueueAcquireGLObjects, 
            // ensure that any pending GL operations have completed

            // create OpenCL buffer from GL buffer
            clbuffer_ = clCreateFromGLBuffer(ctx, CL_MEM_READ_WRITE, bufId(), &status);
            openCLVerifyCall(status);

            // Acquire GL buffer
            openCLSafeCall(clEnqueueAcquireGLObjects(cq, 1, &clbuffer_, 0, NULL, NULL));

            initialized_ = true;
        }

        openCLCopyBuffer2D(mat.clCxt, clbuffer_, mat.cols * elemSize(), 0,
                           mat.data, mat.step, mat.cols * mat.elemSize(), mat.rows, 0);

        openCLSafeCall(clEnqueueReleaseGLObjects(cq, 1, &clbuffer_, 0, NULL, NULL));

        // after calling clEnqueueReleaseGLObjects, 
        // ensure that any pending OpenCL operations have completed
        openCLSafeCall(clFinish(cq));
    }

    void ClGlBuffer::release()
    {
        if (clbuffer_ != (cl_mem) NULL)
            openCLSafeCall(clReleaseMemObject(clbuffer_));

        clbuffer_ = (cl_mem) NULL;
        initialized_ = false;
    }

    ClGlTexture::ClGlTexture()
    {
    }

    void ClGlTexture::copyFrom(const ClGlBuffer& buf)
    {
        ogl::Texture2D::copyFrom(buf);
    }

    std::map<cv::String, ClGlTexture> ownWndTexs;
    std::map<cv::String, ClGlBuffer> ownWndBufs;

    void glDrawTextureCallback(void* userdata)
    {
        ClGlTexture *tex = (ClGlTexture *)userdata;
        ogl::render(*tex);
    }

#if defined WIN32 || defined _WIN32

    //! create an invisible window,
    //  use its associate OpenGL context as parent rendering context
    void createGlContext(HWND hWnd, HDC& hGLDC, HGLRC& hGLRC, bool& useGl)
    {
        useGl = false;

        int PixelFormat;

        static PIXELFORMATDESCRIPTOR pfd =
        {
            sizeof(PIXELFORMATDESCRIPTOR), // Size Of This Pixel Format Descriptor
            1,                             // Version Number
            PFD_DRAW_TO_WINDOW |           // Format Must Support Window
            PFD_SUPPORT_OPENGL |           // Format Must Support OpenGL
            PFD_DOUBLEBUFFER,              // Must Support Double Buffering
            PFD_TYPE_RGBA,                 // Request An RGBA Format
            32,                            // Select Our Color Depth
            0, 0, 0, 0, 0, 0,              // Color Bits Ignored
            0,                             // No Alpha Buffer
            0,                             // Shift Bit Ignored
            0,                             // No Accumulation Buffer
            0, 0, 0, 0,                    // Accumulation Bits Ignored
            32,                            // 32 Bit Z-Buffer (Depth Buffer)  
            0,                             // No Stencil Buffer
            0,                             // No Auxiliary Buffer
            PFD_MAIN_PLANE,                // Main Drawing Layer
            0,                             // Reserved
            0, 0, 0	                       // Layer Masks Ignored
        };

        hGLDC = GetDC(hWnd);
        if (!hGLDC)
            CV_Error( Error::OpenGlApiCallError, "Can't Create A GL Device Context" );

        PixelFormat = ChoosePixelFormat(hGLDC, &pfd);
        if (!PixelFormat)
            CV_Error( Error::OpenGlApiCallError, "Can't Find A Suitable PixelFormat" );

        if (!SetPixelFormat(hGLDC, PixelFormat, &pfd))
            CV_Error( Error::OpenGlApiCallError, "Can't Set The PixelFormat" );

        hGLRC = wglCreateContext(hGLDC);
        if (!hGLRC)
            CV_Error( Error::OpenGlApiCallError, "Can't Create A GL Rendering Context" );

        if (!wglMakeCurrent(hGLDC, hGLRC))
            CV_Error( Error::OpenGlApiCallError, "Can't Activate The GL Rendering Context" );

        useGl = true;
    }

#endif

} // namespace

#endif // HAVE_OPENGL && HAVE_OPENCL

bool ocl::initOpenGLContext(cl_context_properties *cps)
{
    bool useGl = false;

#if defined WIN32 || defined _WIN32
    const char* name = "initOpenGLContext hidden window";
    HDC hGLDC = NULL;
    HGLRC hGLRC = NULL;

    DWORD defStyle = WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU | WS_SIZEBOX;
    Rect rect(10, 10, 100, 100);
    HINSTANCE hg_hinstance = 0;

    g_mainhWnd = CreateWindow( "Main HighGUI class", name, defStyle | WS_OVERLAPPED,
                             rect.x, rect.y, rect.width, rect.height, 0, 0, hg_hinstance, 0 );
    if( !g_mainhWnd )
        CV_Error( Error::StsError, "Frame window can not be created" );

    g_hWnd = CreateWindow("HighGUI class", "", (defStyle & ~WS_SIZEBOX) | WS_CHILD, CW_USEDEFAULT, 
                        0, rect.width, rect.height, g_mainhWnd, 0, hg_hinstance, 0);
    if( !g_hWnd )
        CV_Error( Error::StsError, "Frame window can not be created" );

    createGlContext(g_hWnd, hGLDC, hGLRC, useGl);

    if (useGl)
    {
        g_hDC = hGLDC;
        g_hGLRC = hGLRC;
#ifdef HAVE_OPENCL
        cps[2] = CL_GL_CONTEXT_KHR;
        cps[3] = (cl_context_properties) hGLRC;
        cps[4] = CL_WGL_HDC_KHR;
        cps[5] = (cl_context_properties) hGLDC;
#endif
    }
#else
    (void)cps;
#endif

    return useGl;
}

void ocl::releaseOpenGLContext()
{
#if defined WIN32 || defined _WIN32
    if (g_hGLRC != NULL)
    {
        //wglDeleteContext(g_hGLRC);
        g_hGLRC = NULL;
    }
    if (g_hDC != NULL)
    {
        ReleaseDC(g_hWnd, g_hDC);
        g_hDC = NULL;
    }
    if ( g_hWnd != NULL)
    {
        SendMessage(g_hWnd, WM_CLOSE, 0, 0);
        g_hWnd = NULL;
    }
    if ( g_mainhWnd != NULL)
    {
        SendMessage(g_mainhWnd, WM_CLOSE, 0, 0);
        g_mainhWnd = NULL;
    }
#endif
}

// Display the image in the window using OpenGL-OpenCL interoperation, 
// if cl_khr_gl_sharing extension is supported by OpenCL platform. 
// This avoids data transfers between device and host.
void ocl::imshow( const String& winname, const oclMat& _img )
{
#ifndef HAVE_OPENCL
    (void)winname;
    (void)_img;
    throw_nogpu();
#else
#if !defined HAVE_OPENGL || (!defined WIN32 && !defined _WIN32)
    if (_img.empty())
    {
        CV_Error( Error::StsError, "ocl::imshow input oclMatrix data is NULL" );
    }
    else
    {
        Mat img;
        _img.download(img);
        cv::imshow(winname, img);
    }
#else
#if defined WIN32 || defined _WIN32
    if (_img.empty())
    {
        CV_Error( Error::StsError, "ocl::imshow input oclMatrix is empty" );
        return;
    }

    double useGl = getWindowProperty(winname, WND_PROP_OPENGL);
    if (useGl == 0)
    {
        destroyWindow(winname);
        namedWindow(winname, WINDOW_AUTOSIZE | WINDOW_OPENGL);
    }

    // window not found
    if (useGl == -1)
        namedWindow(winname, WINDOW_AUTOSIZE | WINDOW_OPENGL);

    double autoSize = getWindowProperty(winname, WND_PROP_AUTOSIZE);
    if (autoSize > 0)
    {
        Size size = _img.size();
        resizeWindow(winname, size.width, size.height);
    }

    // ensure rendering context of the window is current
    setOpenGlContext(winname.c_str());

    // shares display lists
    HGLRC hGLRC = wglGetCurrentContext();
    wglShareLists(g_hGLRC, hGLRC);
#endif

    ClGlTexture& tex = ownWndTexs[winname];
    ClGlBuffer& buf = ownWndBufs[winname];

    buf.copyFrom(_img);
    tex.copyFrom(buf);

    setOpenGlDrawCallback(winname, glDrawTextureCallback, &tex);
    updateWindow(winname);

#endif
#endif // HAVE_OPENCL
}
