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

#ifdef HAVE_OPENGL
#  include "gl_core_3_1.hpp"
#  ifdef HAVE_CUDA
#    include <cuda_gl_interop.h>
#  endif
#endif

using namespace cv;
using namespace cv::cuda;

namespace
{
    #ifndef HAVE_OPENGL
        inline void throw_no_ogl() { CV_Error(cv::Error::OpenGlNotSupported, "The library is compiled without OpenGL support"); }
    #else
        inline void throw_no_ogl() { CV_Error(cv::Error::OpenGlApiCallError, "OpenGL context doesn't exist"); }
    #endif

    bool checkError(const char* file, const int line, const char* func = 0)
    {
    #ifndef HAVE_OPENGL
        (void) file;
        (void) line;
        (void) func;
        return true;
    #else
        GLenum err = gl::GetError();

        if (err != gl::NO_ERROR_)
        {
            const char* msg;

            switch (err)
            {
            case gl::INVALID_ENUM:
                msg = "An unacceptable value is specified for an enumerated argument";
                break;

            case gl::INVALID_VALUE:
                msg = "A numeric argument is out of range";
                break;

            case gl::INVALID_OPERATION:
                msg = "The specified operation is not allowed in the current state";
                break;

            case gl::OUT_OF_MEMORY:
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

    #define CV_CheckGlError() CV_DbgAssert( (checkError(__FILE__, __LINE__, CV_Func)) )
} // namespace

#ifdef HAVE_OPENGL
namespace
{
    const GLenum gl_types[] = { gl::UNSIGNED_BYTE, gl::BYTE, gl::UNSIGNED_SHORT, gl::SHORT, gl::INT, gl::FLOAT, gl::DOUBLE };
}
#endif

////////////////////////////////////////////////////////////////////////
// setGlDevice

void cv::cuda::setGlDevice(int device)
{
#ifndef HAVE_OPENGL
    (void) device;
    throw_no_ogl();
#else
    #ifndef HAVE_CUDA
        (void) device;
        throw_no_cuda();
    #else
        cudaSafeCall( cudaGLSetGLDevice(device) );
    #endif
#endif
}

////////////////////////////////////////////////////////////////////////
// CudaResource

#if defined(HAVE_OPENGL) && defined(HAVE_CUDA)

namespace
{
    class CudaResource
    {
    public:
        CudaResource();
        ~CudaResource();

        void registerBuffer(GLuint buffer);
        void release();

        void copyFrom(const void* src, size_t spitch, size_t width, size_t height, cudaStream_t stream = 0);
        void copyTo(void* dst, size_t dpitch, size_t width, size_t height, cudaStream_t stream = 0);

        void* map(cudaStream_t stream = 0);
        void unmap(cudaStream_t stream = 0);

    private:
        cudaGraphicsResource_t resource_;
        GLuint buffer_;

        class GraphicsMapHolder;
    };

    CudaResource::CudaResource() : resource_(0), buffer_(0)
    {
    }

    CudaResource::~CudaResource()
    {
        release();
    }

    void CudaResource::registerBuffer(GLuint buffer)
    {
        CV_DbgAssert( buffer != 0 );

        if (buffer_ == buffer)
            return;

        cudaGraphicsResource_t resource;
        cudaSafeCall( cudaGraphicsGLRegisterBuffer(&resource, buffer, cudaGraphicsMapFlagsNone) );

        release();

        resource_ = resource;
        buffer_ = buffer;
    }

    void CudaResource::release()
    {
        if (resource_)
            cudaGraphicsUnregisterResource(resource_);

        resource_ = 0;
        buffer_ = 0;
    }

    class CudaResource::GraphicsMapHolder
    {
    public:
        GraphicsMapHolder(cudaGraphicsResource_t* resource, cudaStream_t stream);
        ~GraphicsMapHolder();

        void reset();

    private:
        cudaGraphicsResource_t* resource_;
        cudaStream_t stream_;
    };

    CudaResource::GraphicsMapHolder::GraphicsMapHolder(cudaGraphicsResource_t* resource, cudaStream_t stream) : resource_(resource), stream_(stream)
    {
        if (resource_)
            cudaSafeCall( cudaGraphicsMapResources(1, resource_, stream_) );
    }

    CudaResource::GraphicsMapHolder::~GraphicsMapHolder()
    {
        if (resource_)
            cudaGraphicsUnmapResources(1, resource_, stream_);
    }

    void CudaResource::GraphicsMapHolder::reset()
    {
        resource_ = 0;
    }

    void CudaResource::copyFrom(const void* src, size_t spitch, size_t width, size_t height, cudaStream_t stream)
    {
        CV_DbgAssert( resource_ != 0 );

        GraphicsMapHolder h(&resource_, stream);
        (void) h;

        void* dst;
        size_t size;
        cudaSafeCall( cudaGraphicsResourceGetMappedPointer(&dst, &size, resource_) );

        CV_DbgAssert( width * height == size );

        if (stream == 0)
            cudaSafeCall( cudaMemcpy2D(dst, width, src, spitch, width, height, cudaMemcpyDeviceToDevice) );
        else
            cudaSafeCall( cudaMemcpy2DAsync(dst, width, src, spitch, width, height, cudaMemcpyDeviceToDevice, stream) );
    }

    void CudaResource::copyTo(void* dst, size_t dpitch, size_t width, size_t height, cudaStream_t stream)
    {
        CV_DbgAssert( resource_ != 0 );

        GraphicsMapHolder h(&resource_, stream);
        (void) h;

        void* src;
        size_t size;
        cudaSafeCall( cudaGraphicsResourceGetMappedPointer(&src, &size, resource_) );

        CV_DbgAssert( width * height == size );

        if (stream == 0)
            cudaSafeCall( cudaMemcpy2D(dst, dpitch, src, width, width, height, cudaMemcpyDeviceToDevice) );
        else
            cudaSafeCall( cudaMemcpy2DAsync(dst, dpitch, src, width, width, height, cudaMemcpyDeviceToDevice, stream) );
    }

    void* CudaResource::map(cudaStream_t stream)
    {
        CV_DbgAssert( resource_ != 0 );

        GraphicsMapHolder h(&resource_, stream);

        void* ptr;
        size_t size;
        cudaSafeCall( cudaGraphicsResourceGetMappedPointer(&ptr, &size, resource_) );

        h.reset();

        return ptr;
    }

    void CudaResource::unmap(cudaStream_t stream)
    {
        CV_Assert( resource_ != 0 );

        cudaGraphicsUnmapResources(1, &resource_, stream);
    }
}

#endif

////////////////////////////////////////////////////////////////////////
// ogl::Buffer

#ifndef HAVE_OPENGL

class cv::ogl::Buffer::Impl
{
};

#else

class cv::ogl::Buffer::Impl
{
public:
    static const Ptr<Impl>& empty();

    Impl(GLuint bufId, bool autoRelease);
    Impl(GLsizeiptr size, const GLvoid* data, GLenum target, bool autoRelease);
    ~Impl();

    void bind(GLenum target) const;

    void copyFrom(GLuint srcBuf, GLsizeiptr size);

    void copyFrom(GLsizeiptr size, const GLvoid* data);
    void copyTo(GLsizeiptr size, GLvoid* data) const;

    void* mapHost(GLenum access);
    void unmapHost();

#ifdef HAVE_CUDA
    void copyFrom(const void* src, size_t spitch, size_t width, size_t height, cudaStream_t stream = 0);
    void copyTo(void* dst, size_t dpitch, size_t width, size_t height, cudaStream_t stream = 0) const;

    void* mapDevice(cudaStream_t stream = 0);
    void unmapDevice(cudaStream_t stream = 0);
#endif

    void setAutoRelease(bool flag) { autoRelease_ = flag; }

    GLuint bufId() const { return bufId_; }

private:
    Impl();

    GLuint bufId_;
    bool autoRelease_;

#ifdef HAVE_CUDA
    mutable CudaResource cudaResource_;
#endif
};

const Ptr<cv::ogl::Buffer::Impl>& cv::ogl::Buffer::Impl::empty()
{
    static Ptr<Impl> p(new Impl);
    return p;
}

cv::ogl::Buffer::Impl::Impl() : bufId_(0), autoRelease_(false)
{
}

cv::ogl::Buffer::Impl::Impl(GLuint abufId, bool autoRelease) : bufId_(abufId), autoRelease_(autoRelease)
{
    CV_Assert( gl::IsBuffer(abufId) == gl::TRUE_ );
}

cv::ogl::Buffer::Impl::Impl(GLsizeiptr size, const GLvoid* data, GLenum target, bool autoRelease) : bufId_(0), autoRelease_(autoRelease)
{
    gl::GenBuffers(1, &bufId_);
    CV_CheckGlError();

    CV_Assert( bufId_ != 0 );

    gl::BindBuffer(target, bufId_);
    CV_CheckGlError();

    gl::BufferData(target, size, data, gl::DYNAMIC_DRAW);
    CV_CheckGlError();

    gl::BindBuffer(target, 0);
    CV_CheckGlError();
}

cv::ogl::Buffer::Impl::~Impl()
{
    if (autoRelease_ && bufId_)
        gl::DeleteBuffers(1, &bufId_);
}

void cv::ogl::Buffer::Impl::bind(GLenum target) const
{
    gl::BindBuffer(target, bufId_);
    CV_CheckGlError();
}

void cv::ogl::Buffer::Impl::copyFrom(GLuint srcBuf, GLsizeiptr size)
{
    gl::BindBuffer(gl::COPY_WRITE_BUFFER, bufId_);
    CV_CheckGlError();

    gl::BindBuffer(gl::COPY_READ_BUFFER, srcBuf);
    CV_CheckGlError();

    gl::CopyBufferSubData(gl::COPY_READ_BUFFER, gl::COPY_WRITE_BUFFER, 0, 0, size);
    CV_CheckGlError();
}

void cv::ogl::Buffer::Impl::copyFrom(GLsizeiptr size, const GLvoid* data)
{
    gl::BindBuffer(gl::COPY_WRITE_BUFFER, bufId_);
    CV_CheckGlError();

    gl::BufferSubData(gl::COPY_WRITE_BUFFER, 0, size, data);
    CV_CheckGlError();
}

void cv::ogl::Buffer::Impl::copyTo(GLsizeiptr size, GLvoid* data) const
{
    gl::BindBuffer(gl::COPY_READ_BUFFER, bufId_);
    CV_CheckGlError();

    gl::GetBufferSubData(gl::COPY_READ_BUFFER, 0, size, data);
    CV_CheckGlError();
}

void* cv::ogl::Buffer::Impl::mapHost(GLenum access)
{
    gl::BindBuffer(gl::COPY_READ_BUFFER, bufId_);
    CV_CheckGlError();

    GLvoid* data = gl::MapBuffer(gl::COPY_READ_BUFFER, access);
    CV_CheckGlError();

    return data;
}

void cv::ogl::Buffer::Impl::unmapHost()
{
    gl::UnmapBuffer(gl::COPY_READ_BUFFER);
}

#ifdef HAVE_CUDA

void cv::ogl::Buffer::Impl::copyFrom(const void* src, size_t spitch, size_t width, size_t height, cudaStream_t stream)
{
    cudaResource_.registerBuffer(bufId_);
    cudaResource_.copyFrom(src, spitch, width, height, stream);
}

void cv::ogl::Buffer::Impl::copyTo(void* dst, size_t dpitch, size_t width, size_t height, cudaStream_t stream) const
{
    cudaResource_.registerBuffer(bufId_);
    cudaResource_.copyTo(dst, dpitch, width, height, stream);
}

void* cv::ogl::Buffer::Impl::mapDevice(cudaStream_t stream)
{
    cudaResource_.registerBuffer(bufId_);
    return cudaResource_.map(stream);
}

void cv::ogl::Buffer::Impl::unmapDevice(cudaStream_t stream)
{
    cudaResource_.unmap(stream);
}

#endif // HAVE_CUDA

#endif // HAVE_OPENGL

cv::ogl::Buffer::Buffer() : rows_(0), cols_(0), type_(0)
{
#ifndef HAVE_OPENGL
    throw_no_ogl();
#else
    impl_ = Impl::empty();
#endif
}

cv::ogl::Buffer::Buffer(int arows, int acols, int atype, unsigned int abufId, bool autoRelease) : rows_(0), cols_(0), type_(0)
{
#ifndef HAVE_OPENGL
    (void) arows;
    (void) acols;
    (void) atype;
    (void) abufId;
    (void) autoRelease;
    throw_no_ogl();
#else
    impl_.reset(new Impl(abufId, autoRelease));
    rows_ = arows;
    cols_ = acols;
    type_ = atype;
#endif
}

cv::ogl::Buffer::Buffer(Size asize, int atype, unsigned int abufId, bool autoRelease) : rows_(0), cols_(0), type_(0)
{
#ifndef HAVE_OPENGL
    (void) asize;
    (void) atype;
    (void) abufId;
    (void) autoRelease;
    throw_no_ogl();
#else
    impl_.reset(new Impl(abufId, autoRelease));
    rows_ = asize.height;
    cols_ = asize.width;
    type_ = atype;
#endif
}

cv::ogl::Buffer::Buffer(InputArray arr, Target target, bool autoRelease) : rows_(0), cols_(0), type_(0)
{
#ifndef HAVE_OPENGL
    (void) arr;
    (void) target;
    (void) autoRelease;
    throw_no_ogl();
#else
    const int kind = arr.kind();

    switch (kind)
    {
    case _InputArray::OPENGL_BUFFER:
    case _InputArray::GPU_MAT:
        copyFrom(arr, target, autoRelease);
        break;

    default:
        {
            Mat mat = arr.getMat();
            CV_Assert( mat.isContinuous() );
            const GLsizeiptr asize = mat.rows * mat.cols * mat.elemSize();
            impl_.reset(new Impl(asize, mat.data, target, autoRelease));
            rows_ = mat.rows;
            cols_ = mat.cols;
            type_ = mat.type();
            break;
        }
    }
#endif
}

void cv::ogl::Buffer::create(int arows, int acols, int atype, Target target, bool autoRelease)
{
#ifndef HAVE_OPENGL
    (void) arows;
    (void) acols;
    (void) atype;
    (void) target;
    (void) autoRelease;
    throw_no_ogl();
#else
    if (rows_ != arows || cols_ != acols || type_ != atype)
    {
        const GLsizeiptr asize = arows * acols * CV_ELEM_SIZE(atype);
        impl_.reset(new Impl(asize, 0, target, autoRelease));
        rows_ = arows;
        cols_ = acols;
        type_ = atype;
    }
#endif
}

void cv::ogl::Buffer::release()
{
#ifdef HAVE_OPENGL
    if (impl_)
        impl_->setAutoRelease(true);
    impl_ = Impl::empty();
    rows_ = 0;
    cols_ = 0;
    type_ = 0;
#endif
}

void cv::ogl::Buffer::setAutoRelease(bool flag)
{
#ifndef HAVE_OPENGL
    (void) flag;
    throw_no_ogl();
#else
    impl_->setAutoRelease(flag);
#endif
}

void cv::ogl::Buffer::copyFrom(InputArray arr, Target target, bool autoRelease)
{
#ifndef HAVE_OPENGL
    (void) arr;
    (void) target;
    (void) autoRelease;
    throw_no_ogl();
#else
    const int kind = arr.kind();

    const Size asize = arr.size();
    const int atype = arr.type();
    create(asize, atype, target, autoRelease);

    switch (kind)
    {
    case _InputArray::OPENGL_BUFFER:
        {
            ogl::Buffer buf = arr.getOGlBuffer();
            impl_->copyFrom(buf.bufId(), asize.area() * CV_ELEM_SIZE(atype));
            break;
        }

    case _InputArray::GPU_MAT:
        {
            #ifndef HAVE_CUDA
                throw_no_cuda();
            #else
                GpuMat dmat = arr.getGpuMat();
                impl_->copyFrom(dmat.data, dmat.step, dmat.cols * dmat.elemSize(), dmat.rows);
            #endif

            break;
        }

    default:
        {
            Mat mat = arr.getMat();
            CV_Assert( mat.isContinuous() );
            impl_->copyFrom(asize.area() * CV_ELEM_SIZE(atype), mat.data);
        }
    }
#endif
}

void cv::ogl::Buffer::copyFrom(InputArray arr, cuda::Stream& stream, Target target, bool autoRelease)
{
#ifndef HAVE_OPENGL
    (void) arr;
    (void) stream;
    (void) target;
    (void) autoRelease;
    throw_no_ogl();
#else
    #ifndef HAVE_CUDA
        (void) arr;
        (void) stream;
        (void) target;
        (void) autoRelease;
        throw_no_cuda();
    #else
        GpuMat dmat = arr.getGpuMat();

        create(dmat.size(), dmat.type(), target, autoRelease);

        impl_->copyFrom(dmat.data, dmat.step, dmat.cols * dmat.elemSize(), dmat.rows, cuda::StreamAccessor::getStream(stream));
    #endif
#endif
}

void cv::ogl::Buffer::copyTo(OutputArray arr) const
{
#ifndef HAVE_OPENGL
    (void) arr;
    throw_no_ogl();
#else
    const int kind = arr.kind();

    switch (kind)
    {
    case _InputArray::OPENGL_BUFFER:
        {
            arr.getOGlBufferRef().copyFrom(*this);
            break;
        }

    case _InputArray::GPU_MAT:
        {
            #ifndef HAVE_CUDA
                throw_no_cuda();
            #else
                GpuMat& dmat = arr.getGpuMatRef();
                dmat.create(rows_, cols_, type_);
                impl_->copyTo(dmat.data, dmat.step, dmat.cols * dmat.elemSize(), dmat.rows);
            #endif

            break;
        }

    default:
        {
            arr.create(rows_, cols_, type_);
            Mat mat = arr.getMat();
            CV_Assert( mat.isContinuous() );
            impl_->copyTo(mat.rows * mat.cols * mat.elemSize(), mat.data);
        }
    }
#endif
}

void cv::ogl::Buffer::copyTo(OutputArray arr, cuda::Stream& stream) const
{
#ifndef HAVE_OPENGL
    (void) arr;
    (void) stream;
    throw_no_ogl();
#else
    #ifndef HAVE_CUDA
        (void) arr;
        (void) stream;
        throw_no_cuda();
    #else
        arr.create(rows_, cols_, type_);
        GpuMat dmat = arr.getGpuMat();
        impl_->copyTo(dmat.data, dmat.step, dmat.cols * dmat.elemSize(), dmat.rows, cuda::StreamAccessor::getStream(stream));
    #endif
#endif
}

cv::ogl::Buffer cv::ogl::Buffer::clone(Target target, bool autoRelease) const
{
#ifndef HAVE_OPENGL
    (void) target;
    (void) autoRelease;
    throw_no_ogl();
    return cv::ogl::Buffer();
#else
    ogl::Buffer buf;
    buf.copyFrom(*this, target, autoRelease);
    return buf;
#endif
}

void cv::ogl::Buffer::bind(Target target) const
{
#ifndef HAVE_OPENGL
    (void) target;
    throw_no_ogl();
#else
    impl_->bind(target);
#endif
}

void cv::ogl::Buffer::unbind(Target target)
{
#ifndef HAVE_OPENGL
    (void) target;
    throw_no_ogl();
#else
    gl::BindBuffer(target, 0);
    CV_CheckGlError();
#endif
}

Mat cv::ogl::Buffer::mapHost(Access access)
{
#ifndef HAVE_OPENGL
    (void) access;
    throw_no_ogl();
    return Mat();
#else
    return Mat(rows_, cols_, type_, impl_->mapHost(access));
#endif
}

void cv::ogl::Buffer::unmapHost()
{
#ifndef HAVE_OPENGL
    throw_no_ogl();
#else
    return impl_->unmapHost();
#endif
}

GpuMat cv::ogl::Buffer::mapDevice()
{
#ifndef HAVE_OPENGL
    throw_no_ogl();
    return GpuMat();
#else
    #ifndef HAVE_CUDA
        throw_no_cuda();
        return GpuMat();
    #else
        return GpuMat(rows_, cols_, type_, impl_->mapDevice());
    #endif
#endif
}

void cv::ogl::Buffer::unmapDevice()
{
#ifndef HAVE_OPENGL
    throw_no_ogl();
#else
    #ifndef HAVE_CUDA
        throw_no_cuda();
    #else
        impl_->unmapDevice();
    #endif
#endif
}

cuda::GpuMat cv::ogl::Buffer::mapDevice(cuda::Stream& stream)
{
#ifndef HAVE_OPENGL
    (void) stream;
    throw_no_ogl();
    return GpuMat();
#else
    #ifndef HAVE_CUDA
        (void) stream;
        throw_no_cuda();
        return GpuMat();
    #else
        return GpuMat(rows_, cols_, type_, impl_->mapDevice(cuda::StreamAccessor::getStream(stream)));
    #endif
#endif
}

void cv::ogl::Buffer::unmapDevice(cuda::Stream& stream)
{
#ifndef HAVE_OPENGL
    (void) stream;
    throw_no_ogl();
#else
    #ifndef HAVE_CUDA
        (void) stream;
        throw_no_cuda();
    #else
        impl_->unmapDevice(cuda::StreamAccessor::getStream(stream));
    #endif
#endif
}

unsigned int cv::ogl::Buffer::bufId() const
{
#ifndef HAVE_OPENGL
    throw_no_ogl();
    return 0;
#else
    return impl_->bufId();
#endif
}


//////////////////////////////////////////////////////////////////////////////////////////
// ogl::Texture

#ifndef HAVE_OPENGL

class cv::ogl::Texture2D::Impl
{
};

#else

class cv::ogl::Texture2D::Impl
{
public:
    static const Ptr<Impl> empty();

    Impl(GLuint texId, bool autoRelease);
    Impl(GLint internalFormat, GLsizei width, GLsizei height, GLenum format, GLenum type, const GLvoid* pixels, bool autoRelease);
    ~Impl();

    void copyFrom(GLsizei width, GLsizei height, GLenum format, GLenum type, const GLvoid *pixels);
    void copyTo(GLenum format, GLenum type, GLvoid* pixels) const;

    void bind() const;

    void setAutoRelease(bool flag) { autoRelease_ = flag; }

    GLuint texId() const { return texId_; }

private:
    Impl();

    GLuint texId_;
    bool autoRelease_;
};

const Ptr<cv::ogl::Texture2D::Impl> cv::ogl::Texture2D::Impl::empty()
{
    static Ptr<Impl> p(new Impl);
    return p;
}

cv::ogl::Texture2D::Impl::Impl() : texId_(0), autoRelease_(false)
{
}

cv::ogl::Texture2D::Impl::Impl(GLuint atexId, bool autoRelease) : texId_(atexId), autoRelease_(autoRelease)
{
    CV_Assert( gl::IsTexture(atexId) == gl::TRUE_ );
}

cv::ogl::Texture2D::Impl::Impl(GLint internalFormat, GLsizei width, GLsizei height, GLenum format, GLenum type, const GLvoid* pixels, bool autoRelease) : texId_(0), autoRelease_(autoRelease)
{
    gl::GenTextures(1, &texId_);
    CV_CheckGlError();

    CV_Assert(texId_ != 0);

    gl::BindTexture(gl::TEXTURE_2D, texId_);
    CV_CheckGlError();

    gl::PixelStorei(gl::UNPACK_ALIGNMENT, 1);
    CV_CheckGlError();

    gl::TexImage2D(gl::TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, pixels);
    CV_CheckGlError();

    gl::GenerateMipmap(gl::TEXTURE_2D);
    CV_CheckGlError();
}

cv::ogl::Texture2D::Impl::~Impl()
{
    if (autoRelease_ && texId_)
        gl::DeleteTextures(1, &texId_);
}

void cv::ogl::Texture2D::Impl::copyFrom(GLsizei width, GLsizei height, GLenum format, GLenum type, const GLvoid *pixels)
{
    gl::BindTexture(gl::TEXTURE_2D, texId_);
    CV_CheckGlError();

    gl::PixelStorei(gl::UNPACK_ALIGNMENT, 1);
    CV_CheckGlError();

    gl::TexSubImage2D(gl::TEXTURE_2D, 0, 0, 0, width, height, format, type, pixels);
    CV_CheckGlError();

    gl::GenerateMipmap(gl::TEXTURE_2D);
    CV_CheckGlError();
}

void cv::ogl::Texture2D::Impl::copyTo(GLenum format, GLenum type, GLvoid* pixels) const
{
    gl::BindTexture(gl::TEXTURE_2D, texId_);
    CV_CheckGlError();

    gl::PixelStorei(gl::PACK_ALIGNMENT, 1);
    CV_CheckGlError();

    gl::GetTexImage(gl::TEXTURE_2D, 0, format, type, pixels);
    CV_CheckGlError();
}

void cv::ogl::Texture2D::Impl::bind() const
{
    gl::BindTexture(gl::TEXTURE_2D, texId_);
    CV_CheckGlError();
}

#endif // HAVE_OPENGL

cv::ogl::Texture2D::Texture2D() : rows_(0), cols_(0), format_(NONE)
{
#ifndef HAVE_OPENGL
    throw_no_ogl();
#else
    impl_ = Impl::empty();
#endif
}

cv::ogl::Texture2D::Texture2D(int arows, int acols, Format aformat, unsigned int atexId, bool autoRelease) : rows_(0), cols_(0), format_(NONE)
{
#ifndef HAVE_OPENGL
    (void) arows;
    (void) acols;
    (void) aformat;
    (void) atexId;
    (void) autoRelease;
    throw_no_ogl();
#else
    impl_.reset(new Impl(atexId, autoRelease));
    rows_ = arows;
    cols_ = acols;
    format_ = aformat;
#endif
}

cv::ogl::Texture2D::Texture2D(Size asize, Format aformat, unsigned int atexId, bool autoRelease) : rows_(0), cols_(0), format_(NONE)
{
#ifndef HAVE_OPENGL
    (void) asize;
    (void) aformat;
    (void) atexId;
    (void) autoRelease;
    throw_no_ogl();
#else
    impl_.reset(new Impl(atexId, autoRelease));
    rows_ = asize.height;
    cols_ = asize.width;
    format_ = aformat;
#endif
}

cv::ogl::Texture2D::Texture2D(InputArray arr, bool autoRelease) : rows_(0), cols_(0), format_(NONE)
{
#ifndef HAVE_OPENGL
    (void) arr;
    (void) autoRelease;
    throw_no_ogl();
#else
    const int kind = arr.kind();

    const Size asize = arr.size();
    const int atype = arr.type();

    const int depth = CV_MAT_DEPTH(atype);
    const int cn = CV_MAT_CN(atype);

    CV_Assert( depth <= CV_32F );
    CV_Assert( cn == 1 || cn == 3 || cn == 4 );

    const Format internalFormats[] =
    {
        NONE, DEPTH_COMPONENT, NONE, RGB, RGBA
    };
    const GLenum srcFormats[] =
    {
        0, gl::DEPTH_COMPONENT, 0, gl::BGR, gl::BGRA
    };

    switch (kind)
    {
    case _InputArray::OPENGL_BUFFER:
        {
            ogl::Buffer buf = arr.getOGlBuffer();
            buf.bind(ogl::Buffer::PIXEL_UNPACK_BUFFER);
            impl_.reset(new Impl(internalFormats[cn], asize.width, asize.height, srcFormats[cn], gl_types[depth], 0, autoRelease));
            ogl::Buffer::unbind(ogl::Buffer::PIXEL_UNPACK_BUFFER);
            break;
        }

    case _InputArray::GPU_MAT:
        {
            #ifndef HAVE_CUDA
                throw_no_cuda();
            #else
                GpuMat dmat = arr.getGpuMat();
                ogl::Buffer buf(dmat, ogl::Buffer::PIXEL_UNPACK_BUFFER);
                buf.setAutoRelease(true);
                buf.bind(ogl::Buffer::PIXEL_UNPACK_BUFFER);
                impl_.reset(new Impl(internalFormats[cn], asize.width, asize.height, srcFormats[cn], gl_types[depth], 0, autoRelease));
                ogl::Buffer::unbind(ogl::Buffer::PIXEL_UNPACK_BUFFER);
            #endif

            break;
        }

    default:
        {
            Mat mat = arr.getMat();
            CV_Assert( mat.isContinuous() );
            ogl::Buffer::unbind(ogl::Buffer::PIXEL_UNPACK_BUFFER);
            impl_.reset(new Impl(internalFormats[cn], asize.width, asize.height, srcFormats[cn], gl_types[depth], mat.data, autoRelease));
            break;
        }
    }

    rows_ = asize.height;
    cols_ = asize.width;
    format_ = internalFormats[cn];
#endif
}

void cv::ogl::Texture2D::create(int arows, int acols, Format aformat, bool autoRelease)
{
#ifndef HAVE_OPENGL
    (void) arows;
    (void) acols;
    (void) aformat;
    (void) autoRelease;
    throw_no_ogl();
#else
    if (rows_ != arows || cols_ != acols || format_ != aformat)
    {
        ogl::Buffer::unbind(ogl::Buffer::PIXEL_UNPACK_BUFFER);
        impl_.reset(new Impl(aformat, acols, arows, aformat, gl::FLOAT, 0, autoRelease));
        rows_ = arows;
        cols_ = acols;
        format_ = aformat;
    }
#endif
}

void cv::ogl::Texture2D::release()
{
#ifdef HAVE_OPENGL
    if (impl_)
        impl_->setAutoRelease(true);
    impl_ = Impl::empty();
    rows_ = 0;
    cols_ = 0;
    format_ = NONE;
#endif
}

void cv::ogl::Texture2D::setAutoRelease(bool flag)
{
#ifndef HAVE_OPENGL
    (void) flag;
    throw_no_ogl();
#else
    impl_->setAutoRelease(flag);
#endif
}

void cv::ogl::Texture2D::copyFrom(InputArray arr, bool autoRelease)
{
#ifndef HAVE_OPENGL
    (void) arr;
    (void) autoRelease;
    throw_no_ogl();
#else
    const int kind = arr.kind();

    const Size asize = arr.size();
    const int atype = arr.type();

    const int depth = CV_MAT_DEPTH(atype);
    const int cn = CV_MAT_CN(atype);

    CV_Assert( depth <= CV_32F );
    CV_Assert( cn == 1 || cn == 3 || cn == 4 );

    const Format internalFormats[] =
    {
        NONE, DEPTH_COMPONENT, NONE, RGB, RGBA
    };
    const GLenum srcFormats[] =
    {
        0, gl::DEPTH_COMPONENT, 0, gl::BGR, gl::BGRA
    };

    create(asize, internalFormats[cn], autoRelease);

    switch(kind)
    {
    case _InputArray::OPENGL_BUFFER:
        {
            ogl::Buffer buf = arr.getOGlBuffer();
            buf.bind(ogl::Buffer::PIXEL_UNPACK_BUFFER);
            impl_->copyFrom(asize.width, asize.height, srcFormats[cn], gl_types[depth], 0);
            ogl::Buffer::unbind(ogl::Buffer::PIXEL_UNPACK_BUFFER);
            break;
        }

    case _InputArray::GPU_MAT:
        {
            #ifndef HAVE_CUDA
                throw_no_cuda();
            #else
                GpuMat dmat = arr.getGpuMat();
                ogl::Buffer buf(dmat, ogl::Buffer::PIXEL_UNPACK_BUFFER);
                buf.setAutoRelease(true);
                buf.bind(ogl::Buffer::PIXEL_UNPACK_BUFFER);
                impl_->copyFrom(asize.width, asize.height, srcFormats[cn], gl_types[depth], 0);
                ogl::Buffer::unbind(ogl::Buffer::PIXEL_UNPACK_BUFFER);
            #endif

            break;
        }

    default:
        {
            Mat mat = arr.getMat();
            CV_Assert( mat.isContinuous() );
            ogl::Buffer::unbind(ogl::Buffer::PIXEL_UNPACK_BUFFER);
            impl_->copyFrom(asize.width, asize.height, srcFormats[cn], gl_types[depth], mat.data);
        }
    }
#endif
}

void cv::ogl::Texture2D::copyTo(OutputArray arr, int ddepth, bool autoRelease) const
{
#ifndef HAVE_OPENGL
    (void) arr;
    (void) ddepth;
    (void) autoRelease;
    throw_no_ogl();
#else
    const int kind = arr.kind();

    const int cn = format_ == DEPTH_COMPONENT ? 1: format_ == RGB ? 3 : 4;
    const GLenum dstFormat = format_ == DEPTH_COMPONENT ? gl::DEPTH_COMPONENT : format_ == RGB ? gl::BGR : gl::BGRA;

    switch(kind)
    {
    case _InputArray::OPENGL_BUFFER:
        {
            ogl::Buffer& buf = arr.getOGlBufferRef();
            buf.create(rows_, cols_, CV_MAKE_TYPE(ddepth, cn), ogl::Buffer::PIXEL_PACK_BUFFER, autoRelease);
            buf.bind(ogl::Buffer::PIXEL_PACK_BUFFER);
            impl_->copyTo(dstFormat, gl_types[ddepth], 0);
            ogl::Buffer::unbind(ogl::Buffer::PIXEL_PACK_BUFFER);
            break;
        }

    case _InputArray::GPU_MAT:
        {
            #ifndef HAVE_CUDA
                throw_no_cuda();
            #else
                ogl::Buffer buf(rows_, cols_, CV_MAKE_TYPE(ddepth, cn), ogl::Buffer::PIXEL_PACK_BUFFER);
                buf.setAutoRelease(true);
                buf.bind(ogl::Buffer::PIXEL_PACK_BUFFER);
                impl_->copyTo(dstFormat, gl_types[ddepth], 0);
                ogl::Buffer::unbind(ogl::Buffer::PIXEL_PACK_BUFFER);
                buf.copyTo(arr);
            #endif

            break;
        }

    default:
        {
            arr.create(rows_, cols_, CV_MAKE_TYPE(ddepth, cn));
            Mat mat = arr.getMat();
            CV_Assert( mat.isContinuous() );
            ogl::Buffer::unbind(ogl::Buffer::PIXEL_PACK_BUFFER);
            impl_->copyTo(dstFormat, gl_types[ddepth], mat.data);
        }
    }
#endif
}

void cv::ogl::Texture2D::bind() const
{
#ifndef HAVE_OPENGL
    throw_no_ogl();
#else
    impl_->bind();
#endif
}

unsigned int cv::ogl::Texture2D::texId() const
{
#ifndef HAVE_OPENGL
    throw_no_ogl();
    return 0;
#else
    return impl_->texId();
#endif
}


////////////////////////////////////////////////////////////////////////
// ogl::Arrays

void cv::ogl::Arrays::setVertexArray(InputArray vertex)
{
    const int cn = vertex.channels();
    const int depth = vertex.depth();

    CV_Assert( cn == 2 || cn == 3 || cn == 4 );
    CV_Assert( depth == CV_16S || depth == CV_32S || depth == CV_32F || depth == CV_64F );

    if (vertex.kind() == _InputArray::OPENGL_BUFFER)
        vertex_ = vertex.getOGlBuffer();
    else
        vertex_.copyFrom(vertex);

    size_ = vertex_.size().area();
}

void cv::ogl::Arrays::resetVertexArray()
{
    vertex_.release();
    size_ = 0;
}

void cv::ogl::Arrays::setColorArray(InputArray color)
{
    const int cn = color.channels();

    CV_Assert( cn == 3 || cn == 4 );

    if (color.kind() == _InputArray::OPENGL_BUFFER)
        color_ = color.getOGlBuffer();
    else
        color_.copyFrom(color);
}

void cv::ogl::Arrays::resetColorArray()
{
    color_.release();
}

void cv::ogl::Arrays::setNormalArray(InputArray normal)
{
    const int cn = normal.channels();
    const int depth = normal.depth();

    CV_Assert( cn == 3 );
    CV_Assert( depth == CV_8S || depth == CV_16S || depth == CV_32S || depth == CV_32F || depth == CV_64F );

    if (normal.kind() == _InputArray::OPENGL_BUFFER)
        normal_ = normal.getOGlBuffer();
    else
        normal_.copyFrom(normal);
}

void cv::ogl::Arrays::resetNormalArray()
{
    normal_.release();
}

void cv::ogl::Arrays::setTexCoordArray(InputArray texCoord)
{
    const int cn = texCoord.channels();
    const int depth = texCoord.depth();

    CV_Assert( cn >= 1 && cn <= 4 );
    CV_Assert( depth == CV_16S || depth == CV_32S || depth == CV_32F || depth == CV_64F );

    if (texCoord.kind() == _InputArray::OPENGL_BUFFER)
        texCoord_ = texCoord.getOGlBuffer();
    else
        texCoord_.copyFrom(texCoord);
}

void cv::ogl::Arrays::resetTexCoordArray()
{
    texCoord_.release();
}

void cv::ogl::Arrays::release()
{
    resetVertexArray();
    resetColorArray();
    resetNormalArray();
    resetTexCoordArray();
}

void cv::ogl::Arrays::setAutoRelease(bool flag)
{
    vertex_.setAutoRelease(flag);
    color_.setAutoRelease(flag);
    normal_.setAutoRelease(flag);
    texCoord_.setAutoRelease(flag);
}

void cv::ogl::Arrays::bind() const
{
#ifndef HAVE_OPENGL
    throw_no_ogl();
#else
    CV_Assert( texCoord_.empty() || texCoord_.size().area() == size_ );
    CV_Assert( normal_.empty() || normal_.size().area() == size_ );
    CV_Assert( color_.empty() || color_.size().area() == size_ );

    if (texCoord_.empty())
    {
        gl::DisableClientState(gl::TEXTURE_COORD_ARRAY);
        CV_CheckGlError();
    }
    else
    {
        gl::EnableClientState(gl::TEXTURE_COORD_ARRAY);
        CV_CheckGlError();

        texCoord_.bind(ogl::Buffer::ARRAY_BUFFER);

        gl::TexCoordPointer(texCoord_.channels(), gl_types[texCoord_.depth()], 0, 0);
        CV_CheckGlError();
    }

    if (normal_.empty())
    {
        gl::DisableClientState(gl::NORMAL_ARRAY);
        CV_CheckGlError();
    }
    else
    {
        gl::EnableClientState(gl::NORMAL_ARRAY);
        CV_CheckGlError();

        normal_.bind(ogl::Buffer::ARRAY_BUFFER);

        gl::NormalPointer(gl_types[normal_.depth()], 0, 0);
        CV_CheckGlError();
    }

    if (color_.empty())
    {
        gl::DisableClientState(gl::COLOR_ARRAY);
        CV_CheckGlError();
    }
    else
    {
        gl::EnableClientState(gl::COLOR_ARRAY);
        CV_CheckGlError();

        color_.bind(ogl::Buffer::ARRAY_BUFFER);

        const int cn = color_.channels();

        gl::ColorPointer(cn, gl_types[color_.depth()], 0, 0);
        CV_CheckGlError();
    }

    if (vertex_.empty())
    {
        gl::DisableClientState(gl::VERTEX_ARRAY);
        CV_CheckGlError();
    }
    else
    {
        gl::EnableClientState(gl::VERTEX_ARRAY);
        CV_CheckGlError();

        vertex_.bind(ogl::Buffer::ARRAY_BUFFER);

        gl::VertexPointer(vertex_.channels(), gl_types[vertex_.depth()], 0, 0);
        CV_CheckGlError();
    }

    ogl::Buffer::unbind(ogl::Buffer::ARRAY_BUFFER);
#endif
}

////////////////////////////////////////////////////////////////////////
// Rendering

void cv::ogl::render(const ogl::Texture2D& tex, Rect_<double> wndRect, Rect_<double> texRect)
{
#ifndef HAVE_OPENGL
    (void) tex;
    (void) wndRect;
    (void) texRect;
    throw_no_ogl();
#else
    if (!tex.empty())
    {
        gl::MatrixMode(gl::PROJECTION);
        gl::LoadIdentity();
        gl::Ortho(0.0, 1.0, 1.0, 0.0, -1.0, 1.0);
        CV_CheckGlError();

        gl::MatrixMode(gl::MODELVIEW);
        gl::LoadIdentity();
        CV_CheckGlError();

        gl::Disable(gl::LIGHTING);
        CV_CheckGlError();

        tex.bind();

        gl::Enable(gl::TEXTURE_2D);
        CV_CheckGlError();

        gl::TexEnvi(gl::TEXTURE_ENV, gl::TEXTURE_ENV_MODE, gl::REPLACE);
        CV_CheckGlError();

        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR);
        CV_CheckGlError();

        const float vertex[] =
        {
            wndRect.x, wndRect.y, 0.0f,
            wndRect.x, (wndRect.y + wndRect.height), 0.0f,
            wndRect.x + wndRect.width, (wndRect.y + wndRect.height), 0.0f,
            wndRect.x + wndRect.width, wndRect.y, 0.0f
        };
        const float texCoords[] =
        {
            texRect.x, texRect.y,
            texRect.x, texRect.y + texRect.height,
            texRect.x + texRect.width, texRect.y + texRect.height,
            texRect.x + texRect.width, texRect.y
        };

        ogl::Buffer::unbind(ogl::Buffer::ARRAY_BUFFER);

        gl::EnableClientState(gl::TEXTURE_COORD_ARRAY);
        CV_CheckGlError();

        gl::TexCoordPointer(2, gl::FLOAT, 0, texCoords);
        CV_CheckGlError();

        gl::DisableClientState(gl::NORMAL_ARRAY);
        gl::DisableClientState(gl::COLOR_ARRAY);
        CV_CheckGlError();

        gl::EnableClientState(gl::VERTEX_ARRAY);
        CV_CheckGlError();

        gl::VertexPointer(3, gl::FLOAT, 0, vertex);
        CV_CheckGlError();

        gl::DrawArrays(gl::QUADS, 0, 4);
        CV_CheckGlError();
    }
#endif
}

void cv::ogl::render(const ogl::Arrays& arr, int mode, Scalar color)
{
#ifndef HAVE_OPENGL
    (void) arr;
    (void) mode;
    (void) color;
    throw_no_ogl();
#else
    if (!arr.empty())
    {
        gl::Color3d(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0);

        arr.bind();

        gl::DrawArrays(mode, 0, arr.size());
    }
#endif
}

void cv::ogl::render(const ogl::Arrays& arr, InputArray indices, int mode, Scalar color)
{
#ifndef HAVE_OPENGL
    (void) arr;
    (void) indices;
    (void) mode;
    (void) color;
    throw_no_ogl();
#else
    if (!arr.empty() && !indices.empty())
    {
        gl::Color3d(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0);

        arr.bind();

        const int kind = indices.kind();

        switch (kind)
        {
        case _InputArray::OPENGL_BUFFER :
            {
                ogl::Buffer buf = indices.getOGlBuffer();

                const int depth = buf.depth();

                CV_Assert( buf.channels() == 1 );
                CV_Assert( depth <= CV_32S );

                GLenum type;
                if (depth < CV_16U)
                    type = gl::UNSIGNED_BYTE;
                else if (depth < CV_32S)
                    type = gl::UNSIGNED_SHORT;
                else
                    type = gl::UNSIGNED_INT;

                buf.bind(ogl::Buffer::ELEMENT_ARRAY_BUFFER);

                gl::DrawElements(mode, buf.size().area(), type, 0);

                ogl::Buffer::unbind(ogl::Buffer::ELEMENT_ARRAY_BUFFER);

                break;
            }

        default:
            {
                Mat mat = indices.getMat();

                const int depth = mat.depth();

                CV_Assert( mat.channels() == 1 );
                CV_Assert( depth <= CV_32S );
                CV_Assert( mat.isContinuous() );

                GLenum type;
                if (depth < CV_16U)
                    type = gl::UNSIGNED_BYTE;
                else if (depth < CV_32S)
                    type = gl::UNSIGNED_SHORT;
                else
                    type = gl::UNSIGNED_INT;

                ogl::Buffer::unbind(ogl::Buffer::ELEMENT_ARRAY_BUFFER);

                gl::DrawElements(mode, mat.size().area(), type, mat.data);
            }
        }
    }
#endif
}
