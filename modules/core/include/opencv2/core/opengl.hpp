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

#ifndef __OPENCV_CORE_OPENGL_HPP__
#define __OPENCV_CORE_OPENGL_HPP__

#ifndef __cplusplus
#  error opengl.hpp header must be compiled as C++
#endif

#include "opencv2/core.hpp"

namespace cv { namespace ogl {

/////////////////// OpenGL Objects ///////////////////

//! Smart pointer for OpenGL buffer memory with reference counting.
class CV_EXPORTS Buffer
{
public:
    enum Target
    {
        ARRAY_BUFFER         = 0x8892, //!< The buffer will be used as a source for vertex data
        ELEMENT_ARRAY_BUFFER = 0x8893, //!< The buffer will be used for indices (in glDrawElements, for example)
        PIXEL_PACK_BUFFER    = 0x88EB, //!< The buffer will be used for reading from OpenGL textures
        PIXEL_UNPACK_BUFFER  = 0x88EC  //!< The buffer will be used for writing to OpenGL textures
    };

    enum Access
    {
        READ_ONLY  = 0x88B8,
        WRITE_ONLY = 0x88B9,
        READ_WRITE = 0x88BA
    };

    //! create empty buffer
    Buffer();

    //! create buffer from existed buffer id
    Buffer(int arows, int acols, int atype, unsigned int abufId, bool autoRelease = false);
    Buffer(Size asize, int atype, unsigned int abufId, bool autoRelease = false);

    //! create buffer
    Buffer(int arows, int acols, int atype, Target target = ARRAY_BUFFER, bool autoRelease = false);
    Buffer(Size asize, int atype, Target target = ARRAY_BUFFER, bool autoRelease = false);

    //! copy from host/device memory
    explicit Buffer(InputArray arr, Target target = ARRAY_BUFFER, bool autoRelease = false);

    //! create buffer
    void create(int arows, int acols, int atype, Target target = ARRAY_BUFFER, bool autoRelease = false);
    void create(Size asize, int atype, Target target = ARRAY_BUFFER, bool autoRelease = false);

    //! release memory and delete buffer object
    void release();

    //! set auto release mode (if true, release will be called in object's destructor)
    void setAutoRelease(bool flag);

    //! copy from host/device memory (blocking)
    void copyFrom(InputArray arr, Target target = ARRAY_BUFFER, bool autoRelease = false);
    //! copy from device memory (non blocking)
    void copyFrom(InputArray arr, cuda::Stream& stream, Target target = ARRAY_BUFFER, bool autoRelease = false);

    //! copy to host/device memory (blocking)
    void copyTo(OutputArray arr) const;
    //! copy to device memory (non blocking)
    void copyTo(OutputArray arr, cuda::Stream& stream) const;

    //! create copy of current buffer
    Buffer clone(Target target = ARRAY_BUFFER, bool autoRelease = false) const;

    //! bind buffer for specified target
    void bind(Target target) const;

    //! unbind any buffers from specified target
    static void unbind(Target target);

    //! map to host memory
    Mat mapHost(Access access);
    void unmapHost();

    //! map to device memory (blocking)
    cuda::GpuMat mapDevice();
    void unmapDevice();

    //! map to device memory (non blocking)
    cuda::GpuMat mapDevice(cuda::Stream& stream);
    void unmapDevice(cuda::Stream& stream);

    int rows() const;
    int cols() const;
    Size size() const;
    bool empty() const;

    int type() const;
    int depth() const;
    int channels() const;
    int elemSize() const;
    int elemSize1() const;

    //! get OpenGL opject id
    unsigned int bufId() const;

    class Impl;

private:
    Ptr<Impl> impl_;
    int rows_;
    int cols_;
    int type_;
};

//! Smart pointer for OpenGL 2D texture memory with reference counting.
class CV_EXPORTS Texture2D
{
public:
    enum Format
    {
        NONE            = 0,
        DEPTH_COMPONENT = 0x1902, //!< Depth
        RGB             = 0x1907, //!< Red, Green, Blue
        RGBA            = 0x1908  //!< Red, Green, Blue, Alpha
    };

    //! create empty texture
    Texture2D();

    //! create texture from existed texture id
    Texture2D(int arows, int acols, Format aformat, unsigned int atexId, bool autoRelease = false);
    Texture2D(Size asize, Format aformat, unsigned int atexId, bool autoRelease = false);

    //! create texture
    Texture2D(int arows, int acols, Format aformat, bool autoRelease = false);
    Texture2D(Size asize, Format aformat, bool autoRelease = false);

    //! copy from host/device memory
    explicit Texture2D(InputArray arr, bool autoRelease = false);

    //! create texture
    void create(int arows, int acols, Format aformat, bool autoRelease = false);
    void create(Size asize, Format aformat, bool autoRelease = false);

    //! release memory and delete texture object
    void release();

    //! set auto release mode (if true, release will be called in object's destructor)
    void setAutoRelease(bool flag);

    //! copy from host/device memory
    void copyFrom(InputArray arr, bool autoRelease = false);

    //! copy to host/device memory
    void copyTo(OutputArray arr, int ddepth = CV_32F, bool autoRelease = false) const;

    //! bind texture to current active texture unit for GL_TEXTURE_2D target
    void bind() const;

    int rows() const;
    int cols() const;
    Size size() const;
    bool empty() const;

    Format format() const;

    //! get OpenGL opject id
    unsigned int texId() const;

    class Impl;

private:
    Ptr<Impl> impl_;
    int rows_;
    int cols_;
    Format format_;
};

//! OpenGL Arrays
class CV_EXPORTS Arrays
{
public:
    Arrays();

    void setVertexArray(InputArray vertex);
    void resetVertexArray();

    void setColorArray(InputArray color);
    void resetColorArray();

    void setNormalArray(InputArray normal);
    void resetNormalArray();

    void setTexCoordArray(InputArray texCoord);
    void resetTexCoordArray();

    void release();

    void setAutoRelease(bool flag);

    void bind() const;

    int size() const;
    bool empty() const;

private:
    int size_;
    Buffer vertex_;
    Buffer color_;
    Buffer normal_;
    Buffer texCoord_;
};

/////////////////// Render Functions ///////////////////

//! render texture rectangle in window
CV_EXPORTS void render(const Texture2D& tex,
    Rect_<double> wndRect = Rect_<double>(0.0, 0.0, 1.0, 1.0),
    Rect_<double> texRect = Rect_<double>(0.0, 0.0, 1.0, 1.0));

//! render mode
enum {
    POINTS         = 0x0000,
    LINES          = 0x0001,
    LINE_LOOP      = 0x0002,
    LINE_STRIP     = 0x0003,
    TRIANGLES      = 0x0004,
    TRIANGLE_STRIP = 0x0005,
    TRIANGLE_FAN   = 0x0006,
    QUADS          = 0x0007,
    QUAD_STRIP     = 0x0008,
    POLYGON        = 0x0009
};

//! render OpenGL arrays
CV_EXPORTS void render(const Arrays& arr, int mode = POINTS, Scalar color = Scalar::all(255));
CV_EXPORTS void render(const Arrays& arr, InputArray indices, int mode = POINTS, Scalar color = Scalar::all(255));

}} // namespace cv::ogl

namespace cv { namespace cuda {

//! set a CUDA device to use OpenGL interoperability
CV_EXPORTS void setGlDevice(int device = 0);

}}


////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

inline
cv::ogl::Buffer::Buffer(int arows, int acols, int atype, Target target, bool autoRelease) : rows_(0), cols_(0), type_(0)
{
    create(arows, acols, atype, target, autoRelease);
}

inline
cv::ogl::Buffer::Buffer(Size asize, int atype, Target target, bool autoRelease) : rows_(0), cols_(0), type_(0)
{
    create(asize, atype, target, autoRelease);
}

inline
void cv::ogl::Buffer::create(Size asize, int atype, Target target, bool autoRelease)
{
    create(asize.height, asize.width, atype, target, autoRelease);
}

inline
int cv::ogl::Buffer::rows() const
{
    return rows_;
}

inline
int cv::ogl::Buffer::cols() const
{
    return cols_;
}

inline
cv::Size cv::ogl::Buffer::size() const
{
    return Size(cols_, rows_);
}

inline
bool cv::ogl::Buffer::empty() const
{
    return rows_ == 0 || cols_ == 0;
}

inline
int cv::ogl::Buffer::type() const
{
    return type_;
}

inline
int cv::ogl::Buffer::depth() const
{
    return CV_MAT_DEPTH(type_);
}

inline
int cv::ogl::Buffer::channels() const
{
    return CV_MAT_CN(type_);
}

inline
int cv::ogl::Buffer::elemSize() const
{
    return CV_ELEM_SIZE(type_);
}

inline
int cv::ogl::Buffer::elemSize1() const
{
    return CV_ELEM_SIZE1(type_);
}

///////

inline
cv::ogl::Texture2D::Texture2D(int arows, int acols, Format aformat, bool autoRelease) : rows_(0), cols_(0), format_(NONE)
{
    create(arows, acols, aformat, autoRelease);
}

inline
cv::ogl::Texture2D::Texture2D(Size asize, Format aformat, bool autoRelease) : rows_(0), cols_(0), format_(NONE)
{
    create(asize, aformat, autoRelease);
}

inline
void cv::ogl::Texture2D::create(Size asize, Format aformat, bool autoRelease)
{
    create(asize.height, asize.width, aformat, autoRelease);
}

inline
int cv::ogl::Texture2D::rows() const
{
    return rows_;
}

inline
int cv::ogl::Texture2D::cols() const
{
    return cols_;
}

inline
cv::Size cv::ogl::Texture2D::size() const
{
    return Size(cols_, rows_);
}

inline
bool cv::ogl::Texture2D::empty() const
{
    return rows_ == 0 || cols_ == 0;
}

inline
cv::ogl::Texture2D::Format cv::ogl::Texture2D::format() const
{
    return format_;
}

///////

inline
cv::ogl::Arrays::Arrays() : size_(0)
{
}

inline
int cv::ogl::Arrays::size() const
{
    return size_;
}

inline
bool cv::ogl::Arrays::empty() const
{
    return size_ == 0;
}

#endif /* __OPENCV_CORE_OPENGL_HPP__ */
