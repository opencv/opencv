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

namespace cv 
{
    //! Smart pointer for OpenGL buffer memory with reference counting.
    class CV_EXPORTS GlBuffer
    {
    public:
        enum Usage
        {
            ARRAY_BUFFER = 0x8892,  // buffer will use for OpenGL arrays (vertices, colors, normals, etc)
            TEXTURE_BUFFER = 0x88EC // buffer will ise for OpenGL textures
        };

        //! create empty buffer
        explicit GlBuffer(Usage usage);

        //! create buffer
        GlBuffer(int rows, int cols, int type, Usage usage);
        GlBuffer(Size size, int type, Usage usage);

        //! copy from host/device memory
        GlBuffer(InputArray mat, Usage usage);

        void create(int rows, int cols, int type, Usage usage);
        inline void create(Size size, int type, Usage usage) { create(size.height, size.width, type, usage); }
        inline void create(int rows, int cols, int type) { create(rows, cols, type, usage()); }
        inline void create(Size size, int type) { create(size.height, size.width, type, usage()); }

        void release();

        //! copy from host/device memory
        void copyFrom(InputArray mat);

        void bind() const;
        void unbind() const;

        //! map to host memory
        Mat mapHost();
        void unmapHost();

        //! map to device memory
        gpu::GpuMat mapDevice();
        void unmapDevice();
        
        inline int rows() const { return rows_; }
        inline int cols() const { return cols_; }
        inline Size size() const { return Size(cols_, rows_); }
        inline bool empty() const { return rows_ == 0 || cols_ == 0; }

        inline int type() const { return type_; }
        inline int depth() const { return CV_MAT_DEPTH(type_); }
        inline int channels() const { return CV_MAT_CN(type_); }
        inline int elemSize() const { return CV_ELEM_SIZE(type_); }
        inline int elemSize1() const { return CV_ELEM_SIZE1(type_); }

        inline Usage usage() const { return usage_; }

    private:
        int rows_;
        int cols_;
        int type_;
        Usage usage_;

        class Impl;
        Ptr<Impl> impl_;
    };

    template <> CV_EXPORTS void Ptr<GlBuffer::Impl>::delete_obj();

    //! Smart pointer for OpenGL 2d texture memory with reference counting.
    class CV_EXPORTS GlTexture
    {
    public:
        //! create empty texture
        GlTexture();

        //! create texture
        GlTexture(int rows, int cols, int type);
        GlTexture(Size size, int type);

        //! copy from host/device memory
        explicit GlTexture(InputArray mat, bool bgra = true);

        void create(int rows, int cols, int type);
        inline void create(Size size, int type) { create(size.height, size.width, type); }
        void release();

        //! copy from host/device memory
        void copyFrom(InputArray mat, bool bgra = true);

        void bind() const;
        void unbind() const;

        inline int rows() const { return rows_; }
        inline int cols() const { return cols_; }
        inline Size size() const { return Size(cols_, rows_); }
        inline bool empty() const { return rows_ == 0 || cols_ == 0; }

        inline int type() const { return type_; }
        inline int depth() const { return CV_MAT_DEPTH(type_); }
        inline int channels() const { return CV_MAT_CN(type_); }
        inline int elemSize() const { return CV_ELEM_SIZE(type_); }
        inline int elemSize1() const { return CV_ELEM_SIZE1(type_); }

    private:
        int rows_;
        int cols_;
        int type_;

        class Impl;
        Ptr<Impl> impl_;
    };

    template <> CV_EXPORTS void Ptr<GlTexture::Impl>::delete_obj();

    //! OpenGL Arrays
    class CV_EXPORTS GlArrays
    {
    public:
        inline GlArrays() 
            : vertex_(GlBuffer::ARRAY_BUFFER), color_(GlBuffer::ARRAY_BUFFER), bgra_(true), normal_(GlBuffer::ARRAY_BUFFER), texCoord_(GlBuffer::ARRAY_BUFFER)
        {
        }

        void setVertexArray(InputArray vertex);
        inline void resetVertexArray() { vertex_.release(); }

        void setColorArray(InputArray color, bool bgra = true);
        inline void resetColorArray() { color_.release(); }
        
        void setNormalArray(InputArray normal);
        inline void resetNormalArray() { normal_.release(); }
        
        void setTexCoordArray(InputArray texCoord);
        inline void resetTexCoordArray() { texCoord_.release(); }

        void bind() const;
        void unbind() const;

        inline int rows() const { return vertex_.rows(); }
        inline int cols() const { return vertex_.cols(); }
        inline Size size() const { return vertex_.size(); }
        inline bool empty() const { return vertex_.empty(); }

    private:
        GlBuffer vertex_;
        GlBuffer color_;
        bool bgra_;
        GlBuffer normal_;
        GlBuffer texCoord_;
    };

    //! OpenGL Font
    class CV_EXPORTS GlFont
    {
    public:
        enum Weight 
        {
            WEIGHT_LIGHT    = 300,
            WEIGHT_NORMAL   = 400,
            WEIGHT_SEMIBOLD = 600,
            WEIGHT_BOLD     = 700,
            WEIGHT_BLACK    = 900
        };

        enum Style 
        {  
            STYLE_NORMAL    = 0,
            STYLE_ITALIC    = 1,
            STYLE_UNDERLINE = 2
        };

        static Ptr<GlFont> get(const std::string& family, int height = 12, Weight weight = WEIGHT_NORMAL, Style style = STYLE_NORMAL);

        void draw(const char* str, int len) const;

        inline const std::string& family() const { return family_; }
        inline int height() const { return height_; }
        inline Weight weight() const { return weight_; }
        inline Style style() const { return style_; }

    private:
        GlFont(const std::string& family, int height, Weight weight, Style style);

        std::string family_;
        int height_;
        Weight weight_;
        Style style_;

        unsigned int base_;

        GlFont(const GlFont&);
        GlFont& operator =(const GlFont&);
    };

    //! render functions

    //! render texture rectangle in window
    CV_EXPORTS void render(const GlTexture& tex, 
        Rect_<double> wndRect = Rect_<double>(0.0, 0.0, 1.0, 1.0), 
        Rect_<double> texRect = Rect_<double>(0.0, 0.0, 1.0, 1.0));

    //! render mode
    namespace RenderMode {
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
    }

    //! render OpenGL arrays
    CV_EXPORTS void render(const GlArrays& arr, int mode = RenderMode::POINTS, Scalar color = Scalar::all(255));

    CV_EXPORTS void render(const std::string& str, const Ptr<GlFont>& font, Scalar color, Point2d pos);

    //! OpenGL camera
    class CV_EXPORTS GlCamera
    {
    public:
        GlCamera();

        void lookAt(Point3d eye, Point3d center, Point3d up);
        void setCameraPos(Point3d pos, double yaw, double pitch, double roll);

        void setScale(Point3d scale);

        void setProjectionMatrix(const Mat& projectionMatrix, bool transpose = true);
        void setPerspectiveProjection(double fov, double aspect, double zNear, double zFar);
        void setOrthoProjection(double left, double right, double bottom, double top, double zNear, double zFar);

        void setupProjectionMatrix() const;
        void setupModelViewMatrix() const;

    private:
        Point3d eye_;
        Point3d center_;
        Point3d up_;

        Point3d pos_;
        double yaw_;
        double pitch_;
        double roll_;

        bool useLookAtParams_;

        Point3d scale_;

        Mat projectionMatrix_;

        double fov_;
        double aspect_;

        double left_;
        double right_;
        double bottom_;
        double top_;

        double zNear_;
        double zFar_;

        bool perspectiveProjection_;
    };

    namespace gpu 
    {
        //! set a CUDA device to use OpenGL interoperability
        CV_EXPORTS void setGlDevice(int device = 0);
    }
} // namespace cv

#endif // __cplusplus

#endif // __OPENCV_OPENGL_INTEROP_HPP__
