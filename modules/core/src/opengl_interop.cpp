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

#include "precomp.hpp"
#include <iostream>
#include "opencv2/core/opengl_interop.hpp"
#include "opencv2/core/gpumat.hpp"

#ifdef HAVE_OPENGL
    #ifdef __APPLE__
        #include <OpenGL/gl.h>
        #include <OpenGL/glu.h>
    #else
        #include <GL/gl.h>
        #include <GL/glu.h>
    #endif

    #ifdef HAVE_CUDA
        #include <cuda_runtime.h>
        #include <cuda_gl_interop.h>
    #endif
#endif

using namespace std;
using namespace cv;
using namespace cv::gpu;

#ifndef HAVE_OPENGL
    #define throw_nogl CV_Error(CV_OpenGlNotSupported, "The library is compiled without OpenGL support")
    #define throw_nocuda CV_Error(CV_GpuNotSupported, "The library is compiled without CUDA support")
#else
    #define throw_nogl CV_Error(CV_OpenGlNotSupported, "OpenGL context doesn't exist")

    #ifndef HAVE_CUDA
        #define throw_nocuda CV_Error(CV_GpuNotSupported, "The library is compiled without CUDA support")
    #else
        #if defined(__GNUC__)
            #define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILE__, __LINE__, __func__)
        #else /* defined(__CUDACC__) || defined(__MSVC__) */
            #define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILE__, __LINE__)
        #endif

        namespace
        {
            inline void ___cudaSafeCall(cudaError_t err, const char *file, const int line, const char *func = "")
            {
                if (cudaSuccess != err)
                    cv::gpu::error(cudaGetErrorString(err), file, line, func);
            }
        }
    #endif // HAVE_CUDA
#endif

namespace
{
    class EmptyGlFuncTab : public CvOpenGlFuncTab
    {
    public:
        void genBuffers(int, unsigned int*) const { throw_nogl; }
        void deleteBuffers(int, const unsigned int*) const { throw_nogl; }

        void bufferData(unsigned int, ptrdiff_t, const void*, unsigned int) const { throw_nogl; }
        void bufferSubData(unsigned int, ptrdiff_t, ptrdiff_t, const void*) const { throw_nogl; }

        void bindBuffer(unsigned int, unsigned int) const { throw_nogl; }

        void* mapBuffer(unsigned int, unsigned int) const { throw_nogl; return 0; }
        void unmapBuffer(unsigned int) const { throw_nogl; }

        void generateBitmapFont(const std::string&, int, int, bool, bool, int, int, int) const { throw_nogl; }

        bool isGlContextInitialized() const { return false; }
    };

    const CvOpenGlFuncTab* g_glFuncTab = 0;

    const CvOpenGlFuncTab* glFuncTab()
    {
        static EmptyGlFuncTab empty;
        return g_glFuncTab ? g_glFuncTab : &empty;
    }
}

CvOpenGlFuncTab::~CvOpenGlFuncTab()
{
    if (g_glFuncTab == this)
        g_glFuncTab = 0;
}

void icvSetOpenGlFuncTab(const CvOpenGlFuncTab* tab)
{
    g_glFuncTab = tab;
}

#ifdef HAVE_OPENGL
    #ifndef GL_DYNAMIC_DRAW
        #define GL_DYNAMIC_DRAW 0x88E8
    #endif

    #ifndef GL_READ_WRITE
        #define GL_READ_WRITE 0x88BA
    #endif

    #ifndef GL_BGR
        #define GL_BGR 0x80E0
    #endif

    #ifndef GL_BGRA
        #define GL_BGRA 0x80E1
    #endif

    namespace
    {
        const GLenum gl_types[] = {GL_UNSIGNED_BYTE, GL_BYTE, GL_UNSIGNED_SHORT, GL_SHORT, GL_INT, GL_FLOAT, GL_DOUBLE};

    #ifdef HAVE_CUDA
        bool g_isCudaGlDeviceInitialized = false;
    #endif
    }
#endif // HAVE_OPENGL

void cv::gpu::setGlDevice(int device)
{
#ifndef HAVE_CUDA
    throw_nocuda;
#else
    #ifndef HAVE_OPENGL
        throw_nogl;
    #else
        if (!glFuncTab()->isGlContextInitialized())
            throw_nogl;

        cudaSafeCall( cudaGLSetGLDevice(device) );

        g_isCudaGlDeviceInitialized = true;
    #endif
#endif
}

////////////////////////////////////////////////////////////////////////
// CudaGlInterop

#if defined HAVE_CUDA && defined HAVE_OPENGL
namespace
{
    class CudaGlInterop
    {
    public:
        CudaGlInterop();
        ~CudaGlInterop();

        void registerBuffer(unsigned int buffer);

        void copyFrom(const GpuMat& mat, cudaStream_t stream = 0);

        GpuMat map(int rows, int cols, int type, cudaStream_t stream = 0);
        void unmap(cudaStream_t stream = 0);

    private:
        cudaGraphicsResource_t resource_;
    };

    inline CudaGlInterop::CudaGlInterop() : resource_(0)
    {
    }

    CudaGlInterop::~CudaGlInterop()
    {
        if (resource_)
        {
            cudaGraphicsUnregisterResource(resource_);
            resource_ = 0;
        }
    }

    void CudaGlInterop::registerBuffer(unsigned int buffer)
    {
        if (!g_isCudaGlDeviceInitialized)
            cvError(CV_GpuApiCallError, "registerBuffer", "cuda GL device wasn't initialized, call setGlDevice", __FILE__, __LINE__);

        cudaGraphicsResource_t resource;
        cudaSafeCall( cudaGraphicsGLRegisterBuffer(&resource, buffer, cudaGraphicsMapFlagsNone) );

        resource_ = resource;
    }

    void CudaGlInterop::copyFrom(const GpuMat& mat, cudaStream_t stream)
    {
        CV_Assert(resource_ != 0);

        cudaSafeCall( cudaGraphicsMapResources(1, &resource_, stream) );

        void* dst_ptr;
        size_t num_bytes;
        cudaSafeCall( cudaGraphicsResourceGetMappedPointer(&dst_ptr, &num_bytes, resource_) );

        const void* src_ptr = mat.ptr();
        size_t widthBytes = mat.cols * mat.elemSize();

        CV_Assert(widthBytes * mat.rows <= num_bytes);

        if (stream == 0)
            cudaSafeCall( cudaMemcpy2D(dst_ptr, widthBytes, src_ptr, mat.step, widthBytes, mat.rows, cudaMemcpyDeviceToDevice) );
        else
            cudaSafeCall( cudaMemcpy2DAsync(dst_ptr, widthBytes, src_ptr, mat.step, widthBytes, mat.rows, cudaMemcpyDeviceToDevice, stream) );

        cudaGraphicsUnmapResources(1, &resource_, stream);
    }

    GpuMat CudaGlInterop::map(int rows, int cols, int type, cudaStream_t stream)
    {
        CV_Assert(resource_ != 0);

        cudaSafeCall( cudaGraphicsMapResources(1, &resource_, stream) );

        void* ptr;
        size_t num_bytes;
        cudaSafeCall( cudaGraphicsResourceGetMappedPointer(&ptr, &num_bytes, resource_) );

        CV_Assert( static_cast<size_t>(cols) * CV_ELEM_SIZE(type) * rows <= num_bytes );

        return GpuMat(rows, cols, type, ptr);
    }

    inline void CudaGlInterop::unmap(cudaStream_t stream)
    {
        cudaGraphicsUnmapResources(1, &resource_, stream);
    }
}
#endif // HAVE_CUDA && HAVE_OPENGL

////////////////////////////////////////////////////////////////////////
// GlBuffer

#ifndef HAVE_OPENGL

class cv::GlBuffer::Impl
{
};

#else

class cv::GlBuffer::Impl
{
public:
    static const Ptr<Impl>& empty();
    
    Impl(int rows, int cols, int type, unsigned int target);
    Impl(const Mat& m, unsigned int target);
    ~Impl();

    void copyFrom(const Mat& m, unsigned int target);

#ifdef HAVE_CUDA
    void copyFrom(const GpuMat& mat, cudaStream_t stream = 0);
#endif

    void bind(unsigned int target) const;
    void unbind(unsigned int target) const;

    Mat mapHost(int rows, int cols, int type, unsigned int target);
    void unmapHost(unsigned int target);

#ifdef HAVE_CUDA
    GpuMat mapDevice(int rows, int cols, int type, cudaStream_t stream = 0);
    void unmapDevice(cudaStream_t stream = 0);
#endif

private:
    Impl();
    
    unsigned int buffer_;

#ifdef HAVE_CUDA
    CudaGlInterop cudaGlInterop_;
#endif
};

inline const Ptr<cv::GlBuffer::Impl>& cv::GlBuffer::Impl::empty()
{
    static Ptr<Impl> p(new Impl);
    return p;
}

inline cv::GlBuffer::Impl::Impl() : buffer_(0)
{
}

cv::GlBuffer::Impl::Impl(int rows, int cols, int type, unsigned int target) : buffer_(0)
{
    if (!glFuncTab()->isGlContextInitialized())
        throw_nogl;

    CV_DbgAssert(rows > 0 && cols > 0);
    CV_DbgAssert(CV_MAT_DEPTH(type) >= 0 && CV_MAT_DEPTH(type) <= CV_64F);

    glFuncTab()->genBuffers(1, &buffer_);
    CV_CheckGlError();
    CV_Assert(buffer_ != 0);

    size_t size = rows * cols * CV_ELEM_SIZE(type);

    glFuncTab()->bindBuffer(target, buffer_);
    CV_CheckGlError();

    glFuncTab()->bufferData(target, size, 0, GL_DYNAMIC_DRAW);
    CV_CheckGlError();

    glFuncTab()->bindBuffer(target, 0);

#ifdef HAVE_CUDA
    if (g_isCudaGlDeviceInitialized)
        cudaGlInterop_.registerBuffer(buffer_);
#endif
}

cv::GlBuffer::Impl::Impl(const Mat& m, unsigned int target) : buffer_(0)
{
    if (!glFuncTab()->isGlContextInitialized())
        throw_nogl;

    CV_DbgAssert(m.rows > 0 && m.cols > 0);
    CV_DbgAssert(m.depth() >= 0 && m.depth() <= CV_64F);
    CV_Assert(m.isContinuous());

    glFuncTab()->genBuffers(1, &buffer_);
    CV_CheckGlError();
    CV_Assert(buffer_ != 0);

    size_t size = m.rows * m.cols * m.elemSize();

    glFuncTab()->bindBuffer(target, buffer_);
    CV_CheckGlError();

    glFuncTab()->bufferData(target, size, m.data, GL_DYNAMIC_DRAW);
    CV_CheckGlError();

    glFuncTab()->bindBuffer(target, 0);

#ifdef HAVE_CUDA
    if (g_isCudaGlDeviceInitialized)
        cudaGlInterop_.registerBuffer(buffer_);
#endif
}

cv::GlBuffer::Impl::~Impl()
{
    try
    {
        if (buffer_)
            glFuncTab()->deleteBuffers(1, &buffer_);
    }
#ifdef _DEBUG
    catch(const exception& e)
    {
        cerr << e.what() << endl;
    }
#endif
    catch(...)
    {
    }
}

void cv::GlBuffer::Impl::copyFrom(const Mat& m, unsigned int target)
{
    CV_Assert(buffer_ != 0);

    CV_Assert(m.isContinuous());

    bind(target);

    size_t size = m.rows * m.cols * m.elemSize();

    glFuncTab()->bufferSubData(target, 0, size, m.data);
    CV_CheckGlError();

    unbind(target);
}

#ifdef HAVE_CUDA

void cv::GlBuffer::Impl::copyFrom(const GpuMat& mat, cudaStream_t stream)
{
    if (!g_isCudaGlDeviceInitialized)
        cvError(CV_GpuApiCallError, "copyFrom", "cuda GL device wasn't initialized, call setGlDevice", __FILE__, __LINE__);

    CV_Assert(buffer_ != 0);

    cudaGlInterop_.copyFrom(mat, stream);
}

#endif // HAVE_CUDA

inline void cv::GlBuffer::Impl::bind(unsigned int target) const
{
    CV_Assert(buffer_ != 0);

    glFuncTab()->bindBuffer(target, buffer_);
    CV_CheckGlError();
}

inline void cv::GlBuffer::Impl::unbind(unsigned int target) const
{
    glFuncTab()->bindBuffer(target, 0);
}

inline Mat cv::GlBuffer::Impl::mapHost(int rows, int cols, int type, unsigned int target)
{
    void* ptr = glFuncTab()->mapBuffer(target, GL_READ_WRITE);
    CV_CheckGlError();

    return Mat(rows, cols, type, ptr);
}

inline void cv::GlBuffer::Impl::unmapHost(unsigned int target)
{
    glFuncTab()->unmapBuffer(target);
}

#ifdef HAVE_CUDA

inline GpuMat cv::GlBuffer::Impl::mapDevice(int rows, int cols, int type, cudaStream_t stream)
{
    if (!g_isCudaGlDeviceInitialized)
        cvError(CV_GpuApiCallError, "copyFrom", "cuda GL device wasn't initialized, call setGlDevice", __FILE__, __LINE__);

    CV_Assert(buffer_ != 0);

    return cudaGlInterop_.map(rows, cols, type, stream);
}

inline void cv::GlBuffer::Impl::unmapDevice(cudaStream_t stream)
{
    if (!g_isCudaGlDeviceInitialized)
        cvError(CV_GpuApiCallError, "copyFrom", "cuda GL device wasn't initialized, call setGlDevice", __FILE__, __LINE__);

    cudaGlInterop_.unmap(stream);
}

#endif // HAVE_CUDA

#endif // HAVE_OPENGL

cv::GlBuffer::GlBuffer(Usage usage) : rows_(0), cols_(0), type_(0), usage_(usage)
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    impl_ = Impl::empty();
#endif
}

cv::GlBuffer::GlBuffer(int rows, int cols, int type, Usage usage) : rows_(0), cols_(0), type_(0), usage_(usage)
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    impl_ = new Impl(rows, cols, type, usage);
    rows_ = rows;
    cols_ = cols;
    type_ = type;
#endif
}

cv::GlBuffer::GlBuffer(Size size, int type, Usage usage) : rows_(0), cols_(0), type_(0), usage_(usage)
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    impl_ = new Impl(size.height, size.width, type, usage);
    rows_ = size.height;
    cols_ = size.width;
    type_ = type;
#endif
}

cv::GlBuffer::GlBuffer(InputArray mat_, Usage usage) : rows_(0), cols_(0), type_(0), usage_(usage)
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    int kind = mat_.kind();
    Size size = mat_.size();
    int type = mat_.type();

    if (kind == _InputArray::GPU_MAT)
    {
        #ifndef HAVE_CUDA
            throw_nocuda;
        #else
            GpuMat d_mat = mat_.getGpuMat();
            impl_ = new Impl(d_mat.rows, d_mat.cols, d_mat.type(), usage);
            impl_->copyFrom(d_mat);
        #endif
    }
    else
    {
        Mat mat = mat_.getMat();
        impl_ = new Impl(mat, usage);
    }

    rows_ = size.height;
    cols_ = size.width;
    type_ = type;
#endif
}

void cv::GlBuffer::create(int rows, int cols, int type, Usage usage)
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    if (rows_ != rows || cols_ != cols || type_ != type || usage_ != usage)
    {
        impl_ = new Impl(rows, cols, type, usage);
        rows_ = rows;
        cols_ = cols;
        type_ = type;
        usage_ = usage;
    }
#endif
}

void cv::GlBuffer::release()
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    impl_ = Impl::empty();
#endif
}

void cv::GlBuffer::copyFrom(InputArray mat_)
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    int kind = mat_.kind();
    Size size = mat_.size();
    int type = mat_.type();

    create(size, type);

    switch (kind)
    {
    case _InputArray::OPENGL_BUFFER:
        {
            GlBuffer buf = mat_.getGlBuffer();
            *this = buf;
            break;
        }
    case _InputArray::GPU_MAT:
        {
            #ifndef HAVE_CUDA
                throw_nocuda;
            #else
                GpuMat d_mat = mat_.getGpuMat();
                impl_->copyFrom(d_mat);
            #endif

            break;
        }
    default:
        {
            Mat mat = mat_.getMat();
            impl_->copyFrom(mat, usage_);
        }
    }
#endif
}

void cv::GlBuffer::bind() const
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    impl_->bind(usage_);
#endif
}

void cv::GlBuffer::unbind() const
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    impl_->unbind(usage_);
#endif
}

Mat cv::GlBuffer::mapHost()
{
#ifndef HAVE_OPENGL
    throw_nogl;
    return Mat();
#else
    return impl_->mapHost(rows_, cols_, type_, usage_);
#endif
}

void cv::GlBuffer::unmapHost()
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    impl_->unmapHost(usage_);
#endif
}

GpuMat cv::GlBuffer::mapDevice()
{
#ifndef HAVE_OPENGL
    throw_nogl;
    return GpuMat();
#else
    #ifndef HAVE_CUDA
        throw_nocuda;
        return GpuMat();
    #else
        return impl_->mapDevice(rows_, cols_, type_);
    #endif
#endif
}

void cv::GlBuffer::unmapDevice()
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    #ifndef HAVE_CUDA
        throw_nocuda;
    #else
        impl_->unmapDevice();
    #endif
#endif
}

template <> void cv::Ptr<cv::GlBuffer::Impl>::delete_obj()
{
    if (obj) delete obj;
}

//////////////////////////////////////////////////////////////////////////////////////////
// GlTexture

#ifndef HAVE_OPENGL

class cv::GlTexture::Impl
{
};

#else

class cv::GlTexture::Impl
{
public:
    static const Ptr<Impl> empty();

    Impl(int rows, int cols, int type);

    Impl(const Mat& mat, bool bgra);
    Impl(const GlBuffer& buf, bool bgra);

    ~Impl();

    void copyFrom(const Mat& mat, bool bgra);
    void copyFrom(const GlBuffer& buf, bool bgra);

    void bind() const;
    void unbind() const;

private:
    Impl();
    
    GLuint tex_;
};

inline const Ptr<cv::GlTexture::Impl> cv::GlTexture::Impl::empty()
{
    static Ptr<Impl> p(new Impl);
    return p;
}

inline cv::GlTexture::Impl::Impl() : tex_(0)
{
}

cv::GlTexture::Impl::Impl(int rows, int cols, int type) : tex_(0)
{
    if (!glFuncTab()->isGlContextInitialized())
        throw_nogl;

    int depth = CV_MAT_DEPTH(type);
    int cn = CV_MAT_CN(type);

    CV_DbgAssert(rows > 0 && cols > 0);
    CV_Assert(cn == 1 || cn == 3 || cn == 4);
    CV_Assert(depth >= 0 && depth <= CV_32F);

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

    GLenum format = cn == 1 ? GL_LUMINANCE : cn == 3 ? GL_BGR : GL_BGRA;

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    CV_CheckGlError();

    glTexImage2D(GL_TEXTURE_2D, 0, cn, cols, rows, 0, format, gl_types[depth], 0);
    CV_CheckGlError();
}

cv::GlTexture::Impl::Impl(const Mat& mat, bool bgra) : tex_(0)
{
    if (!glFuncTab()->isGlContextInitialized())
        throw_nogl;

    int depth = mat.depth();
    int cn = mat.channels();

    CV_DbgAssert(mat.rows > 0 && mat.cols > 0);
    CV_Assert(cn == 1 || cn == 3 || cn == 4);
    CV_Assert(depth >= 0 && depth <= CV_32F);
    CV_Assert(mat.isContinuous());

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

    GLenum format = cn == 1 ? GL_LUMINANCE : (cn == 3 ? (bgra ? GL_BGR : GL_RGB) : (bgra ? GL_BGRA : GL_RGBA));

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    CV_CheckGlError();

    glTexImage2D(GL_TEXTURE_2D, 0, cn, mat.cols, mat.rows, 0, format, gl_types[depth], mat.data);
    CV_CheckGlError();
}

cv::GlTexture::Impl::Impl(const GlBuffer& buf, bool bgra) : tex_(0)
{
    if (!glFuncTab()->isGlContextInitialized())
        throw_nogl;

    int depth = buf.depth();
    int cn = buf.channels();

    CV_DbgAssert(buf.rows > 0 && buf.cols > 0);
    CV_Assert(cn == 1 || cn == 3 || cn == 4);
    CV_Assert(depth >= 0 && depth <= CV_32F);
    CV_Assert(buf.usage() == GlBuffer::TEXTURE_BUFFER);

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

    GLenum format = cn == 1 ? GL_LUMINANCE : (cn == 3 ? (bgra ? GL_BGR : GL_RGB) : (bgra ? GL_BGRA : GL_RGBA));

    buf.bind();

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    CV_CheckGlError();

    glTexImage2D(GL_TEXTURE_2D, 0, cn, buf.cols(), buf.rows(), 0, format, gl_types[depth], 0);
    CV_CheckGlError();

    buf.unbind();
}

inline cv::GlTexture::Impl::~Impl()
{
    if (tex_)
        glDeleteTextures(1, &tex_);
}

void cv::GlTexture::Impl::copyFrom(const Mat& mat, bool bgra)
{
    CV_Assert(tex_ != 0);

    bind();

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    CV_CheckGlError();

    int cn = mat.channels();
    GLenum format = cn == 1 ? GL_LUMINANCE : (cn == 3 ? (bgra ? GL_BGR : GL_RGB) : (bgra ? GL_BGRA : GL_RGBA));

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mat.cols, mat.rows, format, gl_types[mat.depth()], mat.data);
    CV_CheckGlError();

    unbind();
}

void cv::GlTexture::Impl::copyFrom(const GlBuffer& buf, bool bgra)
{
    CV_Assert(tex_ != 0);
    CV_Assert(buf.usage() == GlBuffer::TEXTURE_BUFFER);

    bind();

    buf.bind();

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    CV_CheckGlError();

    int cn = buf.channels();
    GLenum format = cn == 1 ? GL_LUMINANCE : (cn == 3 ? (bgra ? GL_BGR : GL_RGB) : (bgra ? GL_BGRA : GL_RGBA));

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, buf.cols(), buf.rows(), format, gl_types[buf.depth()], 0);
    CV_CheckGlError();

    buf.unbind();

    unbind();
}

inline void cv::GlTexture::Impl::bind() const
{
    CV_Assert(tex_ != 0);

    glEnable(GL_TEXTURE_2D);
    CV_CheckGlError();

    glBindTexture(GL_TEXTURE_2D, tex_);
    CV_CheckGlError();
}

inline void cv::GlTexture::Impl::unbind() const
{
    glBindTexture(GL_TEXTURE_2D, 0);

    glDisable(GL_TEXTURE_2D);
}

#endif // HAVE_OPENGL

cv::GlTexture::GlTexture() : rows_(0), cols_(0), type_(0)
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    impl_ = Impl::empty();
#endif
}

cv::GlTexture::GlTexture(int rows, int cols, int type) : rows_(0), cols_(0), type_(0)
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    impl_ = new Impl(rows, cols, type);
    rows_ = rows;
    cols_ = cols;
    type_ = type;
#endif
}

cv::GlTexture::GlTexture(Size size, int type) : rows_(0), cols_(0), type_(0)
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    impl_ = new Impl(size.height, size.width, type);
    rows_ = size.height;
    cols_ = size.width;
    type_ = type;
#endif
}

cv::GlTexture::GlTexture(InputArray mat_, bool bgra) : rows_(0), cols_(0), type_(0)
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else    
    int kind = mat_.kind();
    Size size = mat_.size();
    int type = mat_.type();

    switch (kind)
    {
    case _InputArray::OPENGL_BUFFER:
        {
            GlBuffer buf = mat_.getGlBuffer();
            impl_ = new Impl(buf, bgra);
            break;
        }
    case _InputArray::GPU_MAT:
        {
            #ifndef HAVE_CUDA
                throw_nocuda;
            #else
                GpuMat d_mat = mat_.getGpuMat();
                GlBuffer buf(d_mat, GlBuffer::TEXTURE_BUFFER);
                impl_ = new Impl(buf, bgra);
            #endif

            break;
        }
    default:
        {
            Mat mat = mat_.getMat();
            impl_ = new Impl(mat, bgra);
            break;
        }
    }

    rows_ = size.height;
    cols_ = size.width;
    type_ = type;
#endif
}

void cv::GlTexture::create(int rows, int cols, int type)
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    if (rows_ != rows || cols_ != cols || type_ != type)
    {
        impl_ = new Impl(rows, cols, type);
        rows_ = rows;
        cols_ = cols;
        type_ = type;
    }
#endif
}

void cv::GlTexture::release()
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    impl_ = Impl::empty();
#endif
}

void cv::GlTexture::copyFrom(InputArray mat_, bool bgra)
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    int kind = mat_.kind();
    Size size = mat_.size();
    int type = mat_.type();

    create(size, type);

    switch(kind)
    {
    case _InputArray::OPENGL_TEXTURE:
        {
            GlTexture tex = mat_.getGlTexture();
            *this = tex;
            break;
        }
    case _InputArray::OPENGL_BUFFER:
        {
            GlBuffer buf = mat_.getGlBuffer();
            impl_->copyFrom(buf, bgra);
            break;
        }
    case _InputArray::GPU_MAT:
        {
            #ifndef HAVE_CUDA
                throw_nocuda;
            #else
                GpuMat d_mat = mat_.getGpuMat();
                GlBuffer buf(d_mat, GlBuffer::TEXTURE_BUFFER);
                impl_->copyFrom(buf, bgra);
            #endif

            break;
        }
    default:
        {
            Mat mat = mat_.getMat();
            impl_->copyFrom(mat, bgra);
        }
    }
#endif
}

void cv::GlTexture::bind() const
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    impl_->bind();
#endif
}

void cv::GlTexture::unbind() const
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    impl_->unbind();
#endif
}

template <> void cv::Ptr<cv::GlTexture::Impl>::delete_obj()
{
    if (obj) delete obj;
}

////////////////////////////////////////////////////////////////////////
// GlArrays

void cv::GlArrays::setVertexArray(InputArray vertex)
{
    int cn = vertex.channels();
    int depth = vertex.depth();

    CV_Assert(cn == 2 || cn == 3 || cn == 4);
    CV_Assert(depth == CV_16S || depth == CV_32S || depth == CV_32F || depth == CV_64F);

    vertex_.copyFrom(vertex);
}

void cv::GlArrays::setColorArray(InputArray color, bool bgra)
{
    int cn = color.channels();

    CV_Assert((cn == 3 && !bgra) || cn == 4);

    color_.copyFrom(color);
    bgra_ = bgra;
}

void cv::GlArrays::setNormalArray(InputArray normal)
{
    int cn = normal.channels();
    int depth = normal.depth();

    CV_Assert(cn == 3);
    CV_Assert(depth == CV_8S || depth == CV_16S || depth == CV_32S || depth == CV_32F || depth == CV_64F);

    normal_.copyFrom(normal);
}

void cv::GlArrays::setTexCoordArray(InputArray texCoord)
{
    int cn = texCoord.channels();
    int depth = texCoord.depth();

    CV_Assert(cn >= 1 && cn <= 4);
    CV_Assert(depth == CV_16S || depth == CV_32S || depth == CV_32F || depth == CV_64F);

    texCoord_.copyFrom(texCoord);
}

void cv::GlArrays::bind() const
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    CV_DbgAssert(texCoord_.empty() || texCoord_.size().area() == vertex_.size().area());
    CV_DbgAssert(normal_.empty() || normal_.size().area() == vertex_.size().area());
    CV_DbgAssert(color_.empty() || color_.size().area() == vertex_.size().area());

    if (!texCoord_.empty())
    {
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
        CV_CheckGlError();

        texCoord_.bind();

        glTexCoordPointer(texCoord_.channels(), gl_types[texCoord_.depth()], 0, 0);
        CV_CheckGlError();

        texCoord_.unbind();
    }

    if (!normal_.empty())
    {
        glEnableClientState(GL_NORMAL_ARRAY);
        CV_CheckGlError();

        normal_.bind();

        glNormalPointer(gl_types[normal_.depth()], 0, 0);
        CV_CheckGlError();

        normal_.unbind();
    }

    if (!color_.empty())
    {
        glEnableClientState(GL_COLOR_ARRAY);
        CV_CheckGlError();

        color_.bind();

        int cn = color_.channels();
        int format = cn == 3 ? cn : (bgra_ ? GL_BGRA : 4);

        glColorPointer(format, gl_types[color_.depth()], 0, 0);
        CV_CheckGlError();

        color_.unbind();
    }

    if (!vertex_.empty())
    {
        glEnableClientState(GL_VERTEX_ARRAY);
        CV_CheckGlError();

        vertex_.bind();

        glVertexPointer(vertex_.channels(), gl_types[vertex_.depth()], 0, 0);
        CV_CheckGlError();

        vertex_.unbind();
    }
#endif
}

void cv::GlArrays::unbind() const
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    if (!texCoord_.empty())
    {
        glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        CV_CheckGlError();
    }

    if (!normal_.empty())
    {
        glDisableClientState(GL_NORMAL_ARRAY);
        CV_CheckGlError();
    }

    if (!color_.empty())
    {
        glDisableClientState(GL_COLOR_ARRAY);
        CV_CheckGlError();
    }

    if (!vertex_.empty())
    {
        glDisableClientState(GL_VERTEX_ARRAY);
        CV_CheckGlError();
    }
#endif
}

////////////////////////////////////////////////////////////////////////
// GlFont

cv::GlFont::GlFont(const string& family, int height, Weight weight, Style style)
    : family_(family), height_(height), weight_(weight), style_(style), base_(0)
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    base_ = glGenLists(256);
    CV_CheckGlError();

    glFuncTab()->generateBitmapFont(family, height, weight, style & STYLE_ITALIC, style & STYLE_UNDERLINE, 0, 256, base_);
#endif
}

void cv::GlFont::draw(const char* str, int len) const
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    if (base_ && len > 0)
    {
        glPushAttrib(GL_LIST_BIT);
        glListBase(base_);

        glCallLists(len, GL_UNSIGNED_BYTE, str);

        glPopAttrib();

        CV_CheckGlError();
    }
#endif
}

namespace
{
    class FontCompare : public unary_function<Ptr<GlFont>, bool>
    {
    public:
        inline FontCompare(const string& family, int height, GlFont::Weight weight, GlFont::Style style) 
            : family_(family), height_(height), weight_(weight), style_(style)
        {
        }

        bool operator ()(const cv::Ptr<GlFont>& font)
        {
            return font->family() == family_ && font->height() == height_ && font->weight() == weight_ && font->style() == style_;
        }

    private:
        string family_;
        int height_;
        GlFont::Weight weight_;
        GlFont::Style style_;
    };
}

Ptr<GlFont> cv::GlFont::get(const std::string& family, int height, Weight weight, Style style)
{
#ifndef HAVE_OPENGL
    throw_nogl;
    return Ptr<GlFont>();
#else
    static vector< Ptr<GlFont> > fonts;
    fonts.reserve(10);

    vector< Ptr<GlFont> >::iterator fontIt = find_if(fonts.begin(), fonts.end(), FontCompare(family, height, weight, style));

    if (fontIt == fonts.end())
    {
        fonts.push_back(new GlFont(family, height, weight, style));

        fontIt = fonts.end() - 1;
    }

    return *fontIt;
#endif
}

////////////////////////////////////////////////////////////////////////
// Rendering

void cv::render(const GlTexture& tex, Rect_<double> wndRect, Rect_<double> texRect)
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    if (!tex.empty())
    {
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
    }
#endif
}

void cv::render(const GlArrays& arr, int mode)
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    arr.bind();

    glDrawArrays(mode, 0, arr.size().area());

    arr.unbind();
#endif
}

void cv::render(const string& str, const Ptr<GlFont>& font, Scalar color, Point2d pos)
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    glPushAttrib(GL_DEPTH_BUFFER_BIT);

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    glDisable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glRasterPos2d(2.0 * (viewport[0] + pos.x) / viewport[2] - 1.0, 1.0 - 2.0 * (viewport[1] + pos.y + font->height()) / viewport[3]);

    glColor4dv(color.val);
    font->draw(str.c_str(), str.length());

    glPopAttrib();
#endif
}

////////////////////////////////////////////////////////////////////////
// GlCamera

cv::GlCamera::GlCamera() :
    eye_(0.0, 0.0, -5.0), center_(0.0, 0.0, 0.0), up_(0.0, 1.0, 0.0),
    pos_(0.0, 0.0, -5.0), yaw_(0.0), pitch_(0.0), roll_(0.0),
    useLookAtParams_(false),

    scale_(1.0, 1.0, 1.0),

    projectionMatrix_(),
    fov_(45.0), aspect_(0.0),
    left_(0.0), right_(1.0), bottom_(1.0), top_(0.0),
    zNear_(-1.0), zFar_(1.0),
    perspectiveProjection_(false)
{
}

void cv::GlCamera::lookAt(Point3d eye, Point3d center, Point3d up)
{
    eye_ = eye;
    center_ = center;
    up_ = up;
    useLookAtParams_ = true;
}

void cv::GlCamera::setCameraPos(Point3d pos, double yaw, double pitch, double roll)
{
    pos_ = pos;
    yaw_ = yaw;
    pitch_ = pitch;
    roll_ = roll;
    useLookAtParams_ = false;
}

void cv::GlCamera::setScale(Point3d scale)
{
    scale_ = scale;
}

void cv::GlCamera::setProjectionMatrix(const Mat& projectionMatrix, bool transpose)
{
    CV_Assert(projectionMatrix.type() == CV_32F || projectionMatrix.type() == CV_64F);
    CV_Assert(projectionMatrix.cols == 4 && projectionMatrix.rows == 4);

    projectionMatrix_ = transpose ? projectionMatrix.t() : projectionMatrix;
}

void cv::GlCamera::setPerspectiveProjection(double fov, double aspect, double zNear, double zFar)
{
    fov_ = fov;
    aspect_ = aspect;
    zNear_ = zNear;
    zFar_ = zFar;

    projectionMatrix_.release();
    perspectiveProjection_ = true;
}

void cv::GlCamera::setOrthoProjection(double left, double right, double bottom, double top, double zNear, double zFar)
{
    left_ = left;
    right_ = right;
    bottom_ = bottom;
    top_ = top;
    zNear_ = zNear;
    zFar_ = zFar;

    projectionMatrix_.release();
    perspectiveProjection_ = false;
}

void cv::GlCamera::setupProjectionMatrix() const
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    if (projectionMatrix_.empty())
    {
        if (perspectiveProjection_)
            gluPerspective(fov_, aspect_, zNear_, zFar_);
        else
            glOrtho(left_, right_, bottom_, top_, zNear_, zFar_);
    }
    else
    {
        if (projectionMatrix_.type() == CV_32F)
            glLoadMatrixf(projectionMatrix_.ptr<float>());
        else
            glLoadMatrixd(projectionMatrix_.ptr<double>());
    }

    CV_CheckGlError();
#endif
}

void cv::GlCamera::setupModelViewMatrix() const
{
#ifndef HAVE_OPENGL
    throw_nogl;
#else
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    if (useLookAtParams_)
        gluLookAt(eye_.x, eye_.y, eye_.z, center_.x, center_.y, center_.z, up_.x, up_.y, up_.z);
    else
    {
        glRotated(-yaw_, 0.0, 1.0, 0.0);
        glRotated(-pitch_, 1.0, 0.0, 0.0);
        glRotated(-roll_, 0.0, 0.0, 1.0);
        glTranslated(-pos_.x, -pos_.y, -pos_.z);
    }

    glScaled(scale_.x, scale_.y, scale_.z);

    CV_CheckGlError();
#endif
}

////////////////////////////////////////////////////////////////////////
// Error handling

bool icvCheckGlError(const char* file, const int line, const char* func)
{
#ifndef HAVE_OPENGL
    return true;
#else
    GLenum err = glGetError();

    if (err != GL_NO_ERROR)
    {
        const char* msg;

        switch (err)
        {
        case GL_INVALID_ENUM:
            msg = "An unacceptable value is specified for an enumerated argument";
            break;
        case GL_INVALID_VALUE:
            msg = "A numeric argument is out of range";
            break;
        case GL_INVALID_OPERATION:
            msg = "The specified operation is not allowed in the current state";
            break;
        case GL_STACK_OVERFLOW:
            msg = "This command would cause a stack overflow";
            break;
        case GL_STACK_UNDERFLOW:
            msg = "This command would cause a stack underflow";
            break;
        case GL_OUT_OF_MEMORY:
            msg = "There is not enough memory left to execute the command";
            break;
        default:
            msg = "Unknown error";
        };

        cvError(CV_OpenGlApiCallError, func, msg, file, line);

        return false;
    }

    return true;
#endif
}
