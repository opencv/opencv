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
#include "opencv2/core/opengl_interop.hpp"
#include "opencv2/highgui/highgui.hpp"

#if defined WIN32 || defined _WIN32 || defined WINCE
#include <windows.h>
#undef small
#undef min
#undef max
#undef abs
#endif

#if defined HAVE_OPENGL && defined HAVE_OPENCL
    #ifdef __APPLE__
        #include <OpenGL/gl.h>
        #include <OpenGL/glu.h>
    #else
        #include <GL/gl.h>
        #include <GL/glu.h>
    #endif
    #include <CL/cl_gl_ext.h>
#endif

using namespace std;
using namespace cv;
using namespace cv::ocl;

#ifndef HAVE_OPENGL
    #define throw_nogl CV_Error(CV_OpenGlNotSupported, \
                                "The library is compiled without OpenGL support")
#else
    #define throw_nogl CV_Error(CV_OpenGlNotSupported, \
                                "OpenGL context doesn't exist")
#endif

#if defined HAVE_OPENGL && defined HAVE_OPENCL

    #ifndef GL_DYNAMIC_DRAW
        #define GL_DYNAMIC_DRAW 0x88E8
    #endif

    #ifndef GL_BGR
        #define GL_BGR 0x80E0
    #endif

    #ifndef GL_BGRA
        #define GL_BGRA 0x80E1
    #endif

    #ifndef APIENTRY
        #define APIENTRY
    #endif

    #ifndef APIENTRYP
        #define APIENTRYP APIENTRY *
    #endif

    #ifndef GL_VERSION_1_5
        //! GL types for handling large vertex buffer objects
        typedef ptrdiff_t GLintptr;
        typedef ptrdiff_t GLsizeiptr;
    #endif

namespace
{
#if defined WIN32 || defined _WIN32
    //! parent window, HDC and associated OpenGL context,
    // this parent context is shared to ocl::imshow display windows
    HWND g_hWnd = NULL;
    HWND g_mainhWnd = NULL;
    HDC g_hDC = NULL;
    HGLRC g_hGLRC = NULL;
#endif

    const GLenum gl_types[] = {GL_UNSIGNED_BYTE, GL_BYTE, GL_UNSIGNED_SHORT, 
                               GL_SHORT, GL_INT, GL_FLOAT, GL_DOUBLE};

#if defined WIN32 || defined _WIN32
    typedef void (APIENTRYP PFNGLGENBUFFERSPROC   ) (GLsizei n, GLuint *buffers);
    typedef void (APIENTRYP PFNGLDELETEBUFFERSPROC) (GLsizei n, const GLuint *buffers);

    typedef void (APIENTRYP PFNGLBUFFERDATAPROC   ) 
        (GLenum target, GLsizeiptr size, const GLvoid *data, GLenum usage);
    typedef void (APIENTRYP PFNGLBUFFERSUBDATAPROC) 
        (GLenum target, GLintptr offset, GLsizeiptr size, const GLvoid* data);

    typedef void (APIENTRYP PFNGLBINDBUFFERPROC   ) (GLenum target, GLuint buffer);

    typedef GLvoid* (APIENTRYP PFNGLMAPBUFFERPROC) (GLenum target, GLenum access);
    typedef GLboolean (APIENTRYP PFNGLUNMAPBUFFERPROC) (GLenum target);
#endif

    class GlFuncTab
    {
    public:
        GlFuncTab();
        ~GlFuncTab();

        void genBuffers(int n, unsigned int* buffers) const;
        void deleteBuffers(int n, const unsigned int* buffers) const;

        void bufferData(unsigned int target, ptrdiff_t size, const void* data, 
                        unsigned int usage) const;
        void bufferSubData(unsigned int target, ptrdiff_t offset, ptrdiff_t size, 
                        const void* data) const;

        void bindBuffer(unsigned int target, unsigned int buffer) const;

        void* mapBuffer(unsigned int target, unsigned int access) const;
        void unmapBuffer(unsigned int target) const;

        void generateBitmapFont(const std::string& family, int height, int weight, 
            bool italic, bool underline, int start, int count, int base) const;

        bool isGlContextInitialized() const;
            
        PFNGLGENBUFFERSPROC    glGenBuffersExt;
        PFNGLDELETEBUFFERSPROC glDeleteBuffersExt;

        PFNGLBUFFERDATAPROC    glBufferDataExt;
        PFNGLBUFFERSUBDATAPROC glBufferSubDataExt;

        PFNGLBINDBUFFERPROC    glBindBufferExt;

        PFNGLMAPBUFFERPROC     glMapBufferExt;
        PFNGLUNMAPBUFFERPROC   glUnmapBufferExt;

        bool initialized_;
    };

    GlFuncTab* g_glFuncTab = NULL;

    GlFuncTab::GlFuncTab()
    {
        glGenBuffersExt    = 0;
        glDeleteBuffersExt = 0;

        glBufferDataExt    = 0;
        glBufferSubDataExt = 0;

        glBindBufferExt    = 0;

        glMapBufferExt     = 0;
        glUnmapBufferExt   = 0;

        initialized_ = false;
    }

    GlFuncTab::~GlFuncTab()
    {
        if (g_glFuncTab == this)
            g_glFuncTab = 0;

    }
    void GlFuncTab::genBuffers(int n, unsigned int* buffers) const
    {
        CV_FUNCNAME( "GlFuncTab::genBuffers" );

        __BEGIN__;

        if (!glGenBuffersExt)
            CV_ERROR(CV_OpenGlApiCallError, 
                     "OpenGL implementation doesn't support glGenBuffers extension");

        glGenBuffersExt(n, buffers);
        CV_CheckGlError();

        __END__;
    }

    void GlFuncTab::deleteBuffers(int n, const unsigned int* buffers) const
    {
        CV_FUNCNAME( "GlFuncTab::deleteBuffers" );

        __BEGIN__;

        if (!glDeleteBuffersExt)
            CV_ERROR(CV_OpenGlApiCallError, 
                     "OpenGL implementation doesn't support glDeleteBuffers extension");

        glDeleteBuffersExt(n, buffers);
        CV_CheckGlError();

        __END__;
    }

    void GlFuncTab::bufferData(unsigned int target, ptrdiff_t size, 
                               const void* data, unsigned int usage) const
    {
        CV_FUNCNAME( "GlFuncTab::bufferData" );

        __BEGIN__;

        if (!glBufferDataExt)
            CV_ERROR(CV_OpenGlApiCallError, 
                     "OpenGL implementation doesn't support glBufferData extension");

        glBufferDataExt(target, size, data, usage);
        CV_CheckGlError();

        __END__;
    }

    void GlFuncTab::bufferSubData(unsigned int target, ptrdiff_t offset, 
                                  ptrdiff_t size, const void* data) const
    {
        CV_FUNCNAME( "GlFuncTab::bufferSubData" );

        __BEGIN__;

        if (!glBufferSubDataExt)
            CV_ERROR(CV_OpenGlApiCallError, 
                     "OpenGL implementation doesn't support glBufferSubData extension");

        glBufferSubDataExt(target, offset, size, data);
        CV_CheckGlError();

        __END__;
    }

    void* GlFuncTab::mapBuffer(unsigned int target, unsigned int access) const
    {
        CV_FUNCNAME( "GlFuncTab::mapBuffer" );

        void* res = 0;

        __BEGIN__;

        if (!glMapBufferExt)
            CV_ERROR(CV_OpenGlApiCallError, 
                     "OpenGL implementation doesn't support glMapBuffer extension");

        res = glMapBufferExt(target, access);
        CV_CheckGlError();

        __END__;

        return res;
    }

    void GlFuncTab::unmapBuffer(unsigned int target) const
    {
        CV_FUNCNAME( "GlFuncTab::unmapBuffer" );

        __BEGIN__;

        if (!glUnmapBufferExt)
            CV_ERROR(CV_OpenGlApiCallError, 
                     "OpenGL implementation doesn't support glUnmapBuffer extension");

        glUnmapBufferExt(target);
        CV_CheckGlError();

        __END__;
    }

    void GlFuncTab::bindBuffer(unsigned int target, unsigned int buffer) const
    {
        CV_FUNCNAME( "GlFuncTab::bindBuffer" );

        __BEGIN__;

        if (!glBindBufferExt)
            CV_ERROR(CV_OpenGlApiCallError, 
                     "OpenGL implementation doesn't support glBindBuffer extension");

        glBindBufferExt(target, buffer);
        CV_CheckGlError();

        __END__;
    }

    void GlFuncTab::generateBitmapFont(const std::string& family, int height, 
                                       int weight, bool italic, bool underline, 
                                       int start, int count, int base) const
    {
#if defined WIN32 || defined _WIN32
        CV_FUNCNAME( "GlFuncTab::generateBitmapFont" );

        __BEGIN__;
        HFONT font = CreateFont
        (
            -height,                     // height
            0,                           // cell width
            0,                           // Angle of Escapement
            0,                           // Orientation Angle
            weight,                      // font weight
            italic ? TRUE : FALSE,       // Italic
            underline ? TRUE : FALSE,    // Underline
            FALSE,                       // StrikeOut  
            ANSI_CHARSET,                // CharSet  
            OUT_TT_PRECIS,               // OutPrecision
            CLIP_DEFAULT_PRECIS,         // ClipPrecision
            ANTIALIASED_QUALITY,         // Quality
            FF_DONTCARE | DEFAULT_PITCH, // PitchAndFamily
            family.c_str()               // FaceName
        );

        SelectObject(g_hDC, font);

        if (!wglUseFontBitmaps(g_hDC, start, count, base))
            CV_ERROR(CV_OpenGlApiCallError, "Can't create font");
        
        __END__;
#else
        (void)family;
        (void)height;
        (void)weight;
        (void)italic;
        (void)underline;
        (void)start;
        (void)count;
        (void)base;
#endif
    }

    bool GlFuncTab::isGlContextInitialized() const
    {
        return initialized_;
    }

    void initGl()
    {
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

        if (g_glFuncTab != NULL)
            return;

        g_glFuncTab = new GlFuncTab();

        // Load extensions

        #if defined WIN32 || defined _WIN32
            PROC func;
            func = wglGetProcAddress("glGenBuffers");
            g_glFuncTab->glGenBuffersExt = (PFNGLGENBUFFERSPROC)func;

            func = wglGetProcAddress("glDeleteBuffers");
            g_glFuncTab->glDeleteBuffersExt = (PFNGLDELETEBUFFERSPROC)func;

            func = wglGetProcAddress("glBufferData");
            g_glFuncTab->glBufferDataExt = (PFNGLBUFFERDATAPROC)func;

            func = wglGetProcAddress("glBufferSubData");
            g_glFuncTab->glBufferSubDataExt = (PFNGLBUFFERSUBDATAPROC)func;

            func = wglGetProcAddress("glBindBuffer");
            g_glFuncTab->glBindBufferExt = (PFNGLBINDBUFFERPROC)func;

            func = wglGetProcAddress("glMapBuffer");
            g_glFuncTab->glMapBufferExt = (PFNGLMAPBUFFERPROC)func;

            func = wglGetProcAddress("glUnmapBuffer");
            g_glFuncTab->glUnmapBufferExt = (PFNGLUNMAPBUFFERPROC)func;
        #endif

        g_glFuncTab->initialized_ = true;
    }
}

///////////////////////////////////////////////////////////////////////////////
namespace
{
    //! Pointer for OpenGL buffer memory.
    class ClGlBuffer
    {
    public:
        enum Usage
        {
            //! buffer will use for OpenGL arrays (vertices, colors, normals, etc)
            ARRAY_BUFFER = 0x8892,
            //! buffer will use for OpenGL textures
            TEXTURE_BUFFER = 0x88EC
        };

        //! create empty buffer
        ClGlBuffer();

        //! create buffer
        ClGlBuffer(int rows, int cols, int type, Usage usage = TEXTURE_BUFFER);

        //! copy from device memory
        void copyFrom(const oclMat& mat);

        void createGlBuffer();

        void createClBuffer(cl_context cxt, cl_command_queue cq);

        void release();

        inline int rows() const { return rows_; }
        inline int cols() const { return cols_; }

        inline int type() const { return type_; }
        inline int depth() const { return CV_MAT_DEPTH(type_); }
        inline int channels() const { return CV_MAT_CN(type_); }
        inline int elemSize() const { return CV_ELEM_SIZE(type_); }
        inline int elemSize1() const { return CV_ELEM_SIZE1(type_); }

        inline Usage usage() const { return usage_; }

        void bind() const;
        void unbind() const;

    private:
        int rows_;
        int cols_;
        int type_;
        Usage usage_;
        GLuint buffer_;
        cl_mem clbuffer_;
        bool initialized_;
    };
    ClGlBuffer g_glBuf;

    //! pointer for OpenGL 2d texture memory with reference counting.
    class ClGlTexture
    {
    public:
        //! create empty texture
        ClGlTexture();

        //! create texture
        ClGlTexture(int rows, int cols, int type);

        void create(const ClGlBuffer& buf, bool bgra);
        void release();

        //! copy from buffer object
        void copyFrom(const ClGlBuffer& buf, bool bgra = true);

        void bind() const;
        void unbind() const;
        
        inline bool empty() const { return rows_ == 0 || cols_ == 0; }
    private:
        int rows_;
        int cols_;
        int type_;
        GLuint tex_;
        bool initialized_;
    };

    ClGlBuffer::ClGlBuffer() : rows_(0), cols_(0), type_(0), usage_(TEXTURE_BUFFER),
        buffer_(0), clbuffer_((cl_mem)NULL), initialized_(false)
    {
    }

    ClGlBuffer::ClGlBuffer(int rows, int cols, int type, Usage usage):
        buffer_(0), clbuffer_((cl_mem)NULL), initialized_(false)
    {
        rows_ = rows;
        cols_ = cols;
        type_ = type;
        usage_ = usage;
    }

    void ClGlBuffer::copyFrom(const oclMat& mat)
    {
        if ((rows_ != mat.rows) && (cols_ != mat.cols) && (type_ != mat.ocltype()))
            initialized_ = false;

        rows_ = mat.rows;
        cols_ = mat.cols;
        type_ = mat.ocltype();

        createGlBuffer();

        cl_context ctx = mat.clCxt->impl->clContext;
        cl_command_queue cq = mat.clCxt->impl->clCmdQueue;

        createClBuffer(ctx, cq);
        initialized_ = true;

        openCLCopyBuffer2D(mat.clCxt, clbuffer_, cols_ * elemSize(), 0,
                           mat.data, mat.step, mat.cols * mat.elemSize(), mat.rows, 0);

        openCLSafeCall(clEnqueueReleaseGLObjects(cq, 1, &clbuffer_, 0, NULL, NULL));

        // after calling clEnqueueReleaseGLObjects, 
        // ensure that any pending OpenCL operations have completed
        openCLSafeCall(clFinish(cq));
    }

    void ClGlBuffer::createGlBuffer()
    {
        if (initialized_)
            return;

        if (buffer_)
            g_glFuncTab->deleteBuffers(1, &buffer_);

        g_glFuncTab->genBuffers(1, &buffer_);

        bind();

        unsigned int sz = rows_ * cols_ * elemSize();
        g_glFuncTab->bufferData(usage_, sz, NULL, GL_DYNAMIC_DRAW);

        unbind();

    }

    void ClGlBuffer::createClBuffer(cl_context cxt, cl_command_queue cq)
    {
        // Prior to calling clEnqueueAcquireGLObjects, 
        // ensure that any pending GL operations have completed
        glFinish();

        if (initialized_)
            return;

        if (clbuffer_ != (cl_mem) NULL)
            openCLSafeCall(clReleaseMemObject(clbuffer_));

        cl_int status;

        // create OpenCL buffer from GL buffer
        clbuffer_ = clCreateFromGLBuffer(cxt, CL_MEM_READ_WRITE, buffer_, &status);
        if (status != CL_SUCCESS)
            CV_Error_(CV_GpuApiCallError, 
                ("clCreateFromGLBuffer failed: %s\n", getOpenCLErrorString(status)));

        // Acquire GL buffer
        openCLSafeCall(clEnqueueAcquireGLObjects(cq, 1, &clbuffer_, 0, NULL, NULL));
    }

    void ClGlBuffer::release()
    {
        if (clbuffer_ != (cl_mem) NULL)
            openCLSafeCall(clReleaseMemObject(clbuffer_));

        if (buffer_)
            g_glFuncTab->deleteBuffers(1, &buffer_);

        buffer_ = 0;
        clbuffer_ = (cl_mem) NULL;
        initialized_ = false;
    }

    void ClGlBuffer::bind() const
    {
        CV_Assert(buffer_ != 0);
        
        g_glFuncTab->bindBuffer(usage_, buffer_);
        CV_CheckGlError();
    }

    void ClGlBuffer::unbind() const
    {
        g_glFuncTab->bindBuffer(usage_, 0);
    }

    ClGlTexture::ClGlTexture() : rows_(0), cols_(0), type_(0)
    {
    }

    ClGlTexture::ClGlTexture(int rows, int cols, int type)
    {
        rows_ = rows;
        cols_ = cols;
        type_ = type;
    }

    void ClGlTexture::create(const ClGlBuffer& buf, bool bgra)
    {
        int depth = buf.depth();
        int cn = buf.channels();
        rows_ = buf.rows();
        cols_ = buf.cols();
        type_ = buf.type();

        CV_DbgAssert(rows_ > 0 && cols_ > 0);
        CV_Assert(cn == 1 || cn == 3 || cn == 4);
        CV_Assert(depth >= 0 && depth <= CV_32F);
        CV_Assert(buf.usage() == ClGlBuffer::TEXTURE_BUFFER);

        glGenTextures(1, &tex_);
        CV_CheckGlError();
        CV_Assert(tex_ != 0);

        glBindTexture(GL_TEXTURE_2D, tex_);
        CV_CheckGlError();

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        CV_CheckGlError();

        GLenum format = cn == 1 ? GL_LUMINANCE : 
            (cn == 3 ? (bgra ? GL_BGR : GL_RGB) : (bgra ? GL_BGRA : GL_RGBA));

        buf.bind();

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        CV_CheckGlError();

        glTexImage2D(GL_TEXTURE_2D, 0, cn, cols_, rows_, 0, format, gl_types[depth], 0);
        CV_CheckGlError();

        buf.unbind();
    }
    void ClGlTexture::copyFrom(const ClGlBuffer& buf, bool bgra)
    {
        create(buf, bgra);

        bind();

        buf.bind();

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        CV_CheckGlError();

        int cn = buf.channels();
        GLenum format = cn == 1 ? GL_LUMINANCE : 
            (cn == 3 ? (bgra ? GL_BGR : GL_RGB) : (bgra ? GL_BGRA : GL_RGBA));

        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cols_, rows_, format, gl_types[buf.depth()], 0);
        CV_CheckGlError();

        buf.unbind();

        unbind();
    }

    inline void ClGlTexture::bind() const
    {
        CV_Assert(tex_ != 0);

        glEnable(GL_TEXTURE_2D);
        CV_CheckGlError();

        glBindTexture(GL_TEXTURE_2D, tex_);
        CV_CheckGlError();
    }

    inline void ClGlTexture::unbind() const
    {
        glBindTexture(GL_TEXTURE_2D, 0);

        glDisable(GL_TEXTURE_2D);
    }

    void ClGlTexture::release()
    {
        if (tex_)
            glDeleteTextures(1, &tex_);
    }
}

///////////////////////////////////////////////////////////////////////////////
namespace
{
    //! create an invisible window,
    //  use its associate OpenGL context as parent rendering context
#if defined WIN32 || defined _WIN32
    void createGlContext(HWND hWnd, HDC& hGLDC, HGLRC& hGLRC, bool& useGl)
    {
        CV_FUNCNAME( "createGlContext" );

        __BEGIN__;

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
            CV_ERROR( CV_OpenGlApiCallError, "Can't Create A GL Device Context" );

        PixelFormat = ChoosePixelFormat(hGLDC, &pfd);
        if (!PixelFormat)
            CV_ERROR( CV_OpenGlApiCallError, "Can't Find A Suitable PixelFormat" );

        if (!SetPixelFormat(hGLDC, PixelFormat, &pfd))
            CV_ERROR( CV_OpenGlApiCallError, "Can't Set The PixelFormat" );

        hGLRC = wglCreateContext(hGLDC);
        if (!hGLRC)
            CV_ERROR( CV_OpenGlApiCallError, "Can't Create A GL Rendering Context" );

        if (!wglMakeCurrent(hGLDC, hGLRC))
            CV_ERROR( CV_OpenGlApiCallError, "Can't Activate The GL Rendering Context" );

        useGl = true;

        __END__;
    }
#endif

    //! render OpenGL texture object
    void render(ClGlTexture& tex, 
                Rect_<double> wndRect = Rect_<double>(0.0, 0.0, 1.0, 1.0), 
                Rect_<double> texRect = Rect_<double>(0.0, 0.0, 1.0, 1.0))
    {
        if (!tex.empty())
        {
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0.0, 1.0, 1.0, 0.0, -1.0, 1.0);

            tex.bind();

            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

            glBegin(GL_QUADS);
                glTexCoord2d(texRect.x, texRect.y);
                glVertex2d(wndRect.x, wndRect.y);

                glTexCoord2d(texRect.x, texRect.y + texRect.height);
                glVertex2d(wndRect.x, (wndRect.y + wndRect.height));

                glTexCoord2d(texRect.x + texRect.width, texRect.y + texRect.height);
                glVertex2d(wndRect.x + wndRect.width, (wndRect.y + wndRect.height));

                glTexCoord2d(texRect.x + texRect.width, texRect.y);
                glVertex2d(wndRect.x + wndRect.width, wndRect.y);
            glEnd();

            CV_CheckGlError();

            tex.unbind();

            tex.release();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    //! linked list of ClGlTexture objects

    const int CV_TEXTURE_MAGIC_VAL = 0x00287653;

    struct GlObjBase
    {
        int flag;
        GlObjBase* next;
        GlObjBase* prev;
        std::string winname;

        virtual ~GlObjBase() {}
    };

    GlObjBase* g_glObjs = 0;

    GlObjBase* findGlObjByName(const std::string& winname)
    {
        GlObjBase* obj = g_glObjs;

        while(obj && obj->winname != winname)
            obj = obj->next;

        return obj;
    }

    void addGlObj(GlObjBase* glObj)
    {
        glObj->next = g_glObjs;
        glObj->prev = 0;
        if (g_glObjs)
            g_glObjs->prev = glObj;
        g_glObjs = glObj;
    }

    void removeGlObj(GlObjBase* glObj)
    {
        if (glObj->prev)
            glObj->prev->next = glObj->next;
        else
            g_glObjs = glObj->next;

        if (glObj->next)
            glObj->next->prev = glObj->prev;

        delete glObj;
    }

    void releaseGlObj()
    {
        GlObjBase* obj = g_glObjs;

        while (obj)
        {
            GlObjBase* obj_cur = obj->next;
            delete obj;
            obj = obj_cur;
        }
    }

    struct GlObjTex : GlObjBase
    {
        ClGlTexture tex;
    };

    void CV_CDECL glDrawTextureCallback(void* userdata)
    {
        GlObjTex* texObj = static_cast<GlObjTex*>(userdata);

        CV_DbgAssert(texObj->flag == CV_TEXTURE_MAGIC_VAL);

        render(texObj->tex);
    }

    void CV_CDECL glCleanCallback(void* userdata)
    {
        GlObjBase* glObj = static_cast<GlObjBase*>(userdata);

        removeGlObj(glObj);
    }
}

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

    cvInitSystem(0,0);

    g_mainhWnd = CreateWindow( "Main HighGUI class", name, defStyle | WS_OVERLAPPED,
                             rect.x, rect.y, rect.width, rect.height, 0, 0, hg_hinstance, 0 );
    if( !g_mainhWnd )
        CV_Error( CV_StsError, "Frame window can not be created" );

    g_hWnd = CreateWindow("HighGUI class", "", (defStyle & ~WS_SIZEBOX) | WS_CHILD, CW_USEDEFAULT, 
                        0, rect.width, rect.height, g_mainhWnd, 0, hg_hinstance, 0);
    if( !g_hWnd )
        CV_Error( CV_StsError, "Frame window can not be created" );

    createGlContext(g_hWnd, hGLDC, hGLRC, useGl);

    if (useGl)
    {
        initGl();
        g_hDC = hGLDC;
        g_hGLRC = hGLRC;
        
        cps[2] = CL_GL_CONTEXT_KHR;
        cps[3] = (cl_context_properties) hGLRC;
        cps[4] = CL_WGL_HDC_KHR;
        cps[5] = (cl_context_properties) hGLDC;
    }
#endif

    return useGl;
}

void ocl::releaseOpenGLContext()
{
    g_glBuf.release();

    if (g_glFuncTab != NULL)
    {
        delete g_glFuncTab;
        g_glFuncTab = NULL;
    }

#if defined WIN32 || defined _WIN32
    if (g_hGLRC != NULL)
    {
        wglDeleteContext(g_hGLRC);
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

    releaseGlObj();
}

#endif // HAVE_OPENGL && HAVE_OPENCL

// Display the image in the window using OpenGL-OpenCL interoperation, 
// if cl_khr_gl_sharing extension is supported by OpenCL platform. 
// This avoids data transfers between device and host.
void ocl::imshow( const string& winname, const oclMat& _img )
{
#ifndef HAVE_OPENCL
    (void)winname;
    (void)_img;
    throw_nogpu();
#else
#if !defined HAVE_OPENGL || (!defined WIN32 && !defined _WIN32)
    if (_img.empty())
    {
        CV_Error( CV_StsError, "ocl::imshow input oclMatrix data is NULL" );
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
        CV_Error( CV_StsError, "ocl::imshow input oclMatrix data is NULL" );
        return;
    }

    double useGl = getWindowProperty(winname, WND_PROP_OPENGL);
    if (useGl == 0)
    {
        destroyWindow(winname);
        namedWindow(winname, CV_WINDOW_AUTOSIZE | CV_WINDOW_OPENGL);
    }
    // window not found
    if (useGl == -1)
        namedWindow(winname, CV_WINDOW_AUTOSIZE | CV_WINDOW_OPENGL);

    double autoSize = getWindowProperty(winname, WND_PROP_AUTOSIZE);
    if (autoSize > 0)
    {
        Size size = _img.size();
        resizeWindow(winname, size.width, size.height);
    }

    // ensure rendering context of the window is current
    cvSetOpenGlContext(winname.c_str());

    // shares display lists
    HGLRC hGLRC = wglGetCurrentContext();
    wglShareLists(g_hGLRC, hGLRC);
#endif
    GlObjBase* glObj = findGlObjByName(winname);

    if (glObj && glObj->flag != CV_TEXTURE_MAGIC_VAL)
    {
        glObj = 0;
    }

    if (glObj)
    {
        GlObjTex* texObj = static_cast<GlObjTex*>(glObj);
        ClGlBuffer buf;
        g_glBuf.copyFrom(_img);
        texObj->tex.copyFrom(g_glBuf);
    }
    else
    {
        GlObjTex* texObj = new GlObjTex;
        g_glBuf.copyFrom(_img);
        texObj->tex.copyFrom(g_glBuf);

        glObj = texObj;
        glObj->flag = CV_TEXTURE_MAGIC_VAL;
        glObj->winname = winname;

        addGlObj(glObj);
    }
    setOpenGlDrawCallback(winname, glDrawTextureCallback, glObj);
    updateWindow(winname);

#endif
#endif // HAVE_OPENCL
}
