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
//     and/or other GpuMaterials provided with the distribution.
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

#ifndef __OPENCV_OPENGL_INTEROP_HPP__
#define __OPENCV_OPENGL_INTEROP_HPP__

#ifdef __cplusplus

#include "opencv2/core/core.hpp"
#include "opencv2/core/opengl_interop_deprecated.hpp"

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
    void create(Size asize, int atype, Target target = ARRAY_BUFFER, bool autoRelease = false) { create(asize.height, asize.width, atype, target, autoRelease); }

    //! release memory and delete buffer object
    void release();

    //! set auto release mode (if true, release will be called in object's destructor)
    void setAutoRelease(bool flag);

    //! copy from host/device memory
    void copyFrom(InputArray arr, Target target = ARRAY_BUFFER, bool autoRelease = false);

    //! copy to host/device memory
    void copyTo(OutputArray arr, Target target = ARRAY_BUFFER, bool autoRelease = false) const;

    //! create copy of current buffer
    Buffer clone(Target target = ARRAY_BUFFER, bool autoRelease = false) const;

    //! bind buffer for specified target
    void bind(Target target) const;

    //! unbind any buffers from specified target
    static void unbind(Target target);

    //! map to host memory
    Mat mapHost(Access access);
    void unmapHost();

    //! map to device memory
    gpu::GpuMat mapDevice();
    void unmapDevice();

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    Size size() const { return Size(cols_, rows_); }
    bool empty() const { return rows_ == 0 || cols_ == 0; }

    int type() const { return type_; }
    int depth() const { return CV_MAT_DEPTH(type_); }
    int channels() const { return CV_MAT_CN(type_); }
    int elemSize() const { return CV_ELEM_SIZE(type_); }
    int elemSize1() const { return CV_ELEM_SIZE1(type_); }

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
    void create(Size asize, Format aformat, bool autoRelease = false) { create(asize.height, asize.width, aformat, autoRelease); }

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

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    Size size() const { return Size(cols_, rows_); }
    bool empty() const { return rows_ == 0 || cols_ == 0; }

    Format format() const { return format_; }

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

    int size() const { return size_; }
    bool empty() const { return size_ == 0; }

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

}} // namespace cv::gl

namespace cv { namespace gpu {

//! set a CUDA device to use OpenGL interoperability
CV_EXPORTS void setGlDevice(int device = 0);

}}

namespace cv {

template <> CV_EXPORTS void Ptr<cv::ogl::Buffer::Impl>::delete_obj();
template <> CV_EXPORTS void Ptr<cv::ogl::Texture2D::Impl>::delete_obj();

}

#endif // __cplusplus

#endif // __OPENCV_OPENGL_INTEROP_HPP__
