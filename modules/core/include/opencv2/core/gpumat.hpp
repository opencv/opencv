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

#ifndef __OPENCV_GPUMAT_HPP__
#define __OPENCV_GPUMAT_HPP__

#ifdef __cplusplus

#include "opencv2/core/core.hpp"
#include "opencv2/core/devmem2d.hpp"

namespace cv { namespace gpu
{
    ////////////////////////////////////////////////////////////////////////
    // GpuMat

    //! Smart pointer for GPU memory with reference counting. Its interface is mostly similar with cv::Mat.
    class CV_EXPORTS GpuMat
    {
    public:
        //! default constructor
        GpuMat();

        //! constructs GpuMatrix of the specified size and type (_type is CV_8UC1, CV_64FC3, CV_32SC(12) etc.)
        GpuMat(int rows, int cols, int type);
        GpuMat(Size size, int type);

        //! constucts GpuMatrix and fills it with the specified value _s.
        GpuMat(int rows, int cols, int type, Scalar s);
        GpuMat(Size size, int type, Scalar s);

        //! copy constructor
        GpuMat(const GpuMat& m);

        //! constructor for GpuMatrix headers pointing to user-allocated data
        GpuMat(int rows, int cols, int type, void* data, size_t step = Mat::AUTO_STEP);
        GpuMat(Size size, int type, void* data, size_t step = Mat::AUTO_STEP);

        //! creates a matrix header for a part of the bigger matrix
        GpuMat(const GpuMat& m, Range rowRange, Range colRange);
        GpuMat(const GpuMat& m, Rect roi);
        
        //! builds GpuMat from Mat. Perfom blocking upload to device.
        explicit GpuMat(const Mat& m);

        //! destructor - calls release()
        ~GpuMat();

        //! assignment operators
        GpuMat& operator = (const GpuMat& m);
        
        //! pefroms blocking upload data to GpuMat.
        void upload(const Mat& m);

        //! downloads data from device to host memory. Blocking calls.
        void download(Mat& m) const;

        //! returns a new GpuMatrix header for the specified row
        GpuMat row(int y) const;
        //! returns a new GpuMatrix header for the specified column
        GpuMat col(int x) const;
        //! ... for the specified row span
        GpuMat rowRange(int startrow, int endrow) const;
        GpuMat rowRange(Range r) const;
        //! ... for the specified column span
        GpuMat colRange(int startcol, int endcol) const;
        GpuMat colRange(Range r) const;

        //! returns deep copy of the GpuMatrix, i.e. the data is copied
        GpuMat clone() const;
        //! copies the GpuMatrix content to "m".
        // It calls m.create(this->size(), this->type()).
        void copyTo(GpuMat& m) const;
        //! copies those GpuMatrix elements to "m" that are marked with non-zero mask elements.
        void copyTo(GpuMat& m, const GpuMat& mask) const;
        //! converts GpuMatrix to another datatype with optional scalng. See cvConvertScale.
        void convertTo(GpuMat& m, int rtype, double alpha = 1, double beta = 0) const;

        void assignTo(GpuMat& m, int type=-1) const;

        //! sets every GpuMatrix element to s
        GpuMat& operator = (Scalar s);
        //! sets some of the GpuMatrix elements to s, according to the mask
        GpuMat& setTo(Scalar s, const GpuMat& mask = GpuMat());
        //! creates alternative GpuMatrix header for the same data, with different
        // number of channels and/or different number of rows. see cvReshape.
        GpuMat reshape(int cn, int rows = 0) const;

        //! allocates new GpuMatrix data unless the GpuMatrix already has specified size and type.
        // previous data is unreferenced if needed.
        void create(int rows, int cols, int type);
        void create(Size size, int type);
        //! decreases reference counter;
        // deallocate the data when reference counter reaches 0.
        void release();

        //! swaps with other smart pointer
        void swap(GpuMat& mat);

        //! locates GpuMatrix header within a parent GpuMatrix. See below
        void locateROI(Size& wholeSize, Point& ofs) const;
        //! moves/resizes the current GpuMatrix ROI inside the parent GpuMatrix.
        GpuMat& adjustROI(int dtop, int dbottom, int dleft, int dright);
        //! extracts a rectangular sub-GpuMatrix
        // (this is a generalized form of row, rowRange etc.)
        GpuMat operator()(Range rowRange, Range colRange) const;
        GpuMat operator()(Rect roi) const;

        //! returns true iff the GpuMatrix data is continuous
        // (i.e. when there are no gaps between successive rows).
        // similar to CV_IS_GpuMat_CONT(cvGpuMat->type)
        bool isContinuous() const;
        //! returns element size in bytes,
        // similar to CV_ELEM_SIZE(cvMat->type)
        size_t elemSize() const;
        //! returns the size of element channel in bytes.
        size_t elemSize1() const;
        //! returns element type, similar to CV_MAT_TYPE(cvMat->type)
        int type() const;
        //! returns element type, similar to CV_MAT_DEPTH(cvMat->type)
        int depth() const;
        //! returns element type, similar to CV_MAT_CN(cvMat->type)
        int channels() const;
        //! returns step/elemSize1()
        size_t step1() const;
        //! returns GpuMatrix size:
        // width == number of columns, height == number of rows
        Size size() const;
        //! returns true if GpuMatrix data is NULL
        bool empty() const;

        //! returns pointer to y-th row
        uchar* ptr(int y = 0);
        const uchar* ptr(int y = 0) const;

        //! template version of the above method
        template<typename _Tp> _Tp* ptr(int y = 0);
        template<typename _Tp> const _Tp* ptr(int y = 0) const;

        template <typename _Tp> operator DevMem2D_<_Tp>() const;
        template <typename _Tp> operator PtrStep_<_Tp>() const;
        template <typename _Tp> operator PtrStep<_Tp>() const;

        /*! includes several bit-fields:
        - the magic signature
        - continuity flag
        - depth
        - number of channels
        */
        int flags;

        //! the number of rows and columns
        int rows, cols;

        //! a distance between successive rows in bytes; includes the gap if any
        size_t step;

        //! pointer to the data
        uchar* data;

        //! pointer to the reference counter;
        // when GpuMatrix points to user-allocated data, the pointer is NULL
        int* refcount;

        //! helper fields used in locateROI and adjustROI
        uchar* datastart;
        uchar* dataend;
    };

    //! Creates continuous GPU matrix
    CV_EXPORTS void createContinuous(int rows, int cols, int type, GpuMat& m);
    CV_EXPORTS GpuMat createContinuous(int rows, int cols, int type);
    CV_EXPORTS void createContinuous(Size size, int type, GpuMat& m);
    CV_EXPORTS GpuMat createContinuous(Size size, int type);

    //! Ensures that size of the given matrix is not less than (rows, cols) size
    //! and matrix type is match specified one too
    CV_EXPORTS void ensureSizeIsEnough(int rows, int cols, int type, GpuMat& m);
    CV_EXPORTS void ensureSizeIsEnough(Size size, int type, GpuMat& m);

    ////////////////////////////////////////////////////////////////////////
    // OpenGL

    //! set a CUDA device to use OpenGL interoperability
    CV_EXPORTS void setGlDevice(int device = 0);

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
        GlBuffer(const GpuMat& d_mat, Usage usage);

        GlBuffer(const GlBuffer& other);

        ~GlBuffer();

        GlBuffer& operator =(const GlBuffer& other);

        void create(int rows, int cols, int type, Usage usage);
        inline void create(Size size, int type, Usage usage) { create(size.height, size.width, type, usage); }
        inline void create(int rows, int cols, int type) { create(rows, cols, type, usage()); }
        inline void create(Size size, int type) { create(size.height, size.width, type, usage()); }

        void release();

        //! copy from host/device memory
        void copyFrom(InputArray mat);
        void copyFrom(const GpuMat& d_mat);

        void bind() const;
        void unbind() const;

        //! map to host memory
        Mat mapHost();
        void unmapHost();

        //! map to device memory
        GpuMat mapDevice();
        void unmapDevice();
        
        int rows;
        int cols;

        inline Size size() const { return Size(cols, rows); }
        inline bool empty() const { return rows == 0 || cols == 0; }

        inline int type() const { return type_; }
        inline int depth() const { return CV_MAT_DEPTH(type_); }
        inline int channels() const { return CV_MAT_CN(type_); }
        inline int elemSize() const { return CV_ELEM_SIZE(type_); }
        inline int elemSize1() const { return CV_ELEM_SIZE1(type_); }

        inline Usage usage() const { return usage_; }

    private:
        int type_;
        Usage usage_;

        class Impl;
        Ptr<Impl> impl_;
    };

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
        explicit GlTexture(const GlBuffer& buf, bool bgra = true);

        GlTexture(const GlTexture& other);

        ~GlTexture();

        GlTexture& operator =(const GlTexture& other);

        void create(int rows, int cols, int type);
        inline void create(Size size, int type) { create(size.height, size.width, type); }
        void release();

        //! copy from host/device memory
        void copyFrom(InputArray mat, bool bgra = true);
        void copyFrom(const GlBuffer& buf, bool bgra = true);

        void bind() const;
        void unbind() const;

        int rows;
        int cols;

        inline Size size() const { return Size(cols, rows); }
        inline bool empty() const { return rows == 0 || cols == 0; }

        inline int type() const { return type_; }
        inline int depth() const { return CV_MAT_DEPTH(type_); }
        inline int channels() const { return CV_MAT_CN(type_); }
        inline int elemSize() const { return CV_ELEM_SIZE(type_); }
        inline int elemSize1() const { return CV_ELEM_SIZE1(type_); }

    private:
        int type_;

        class Impl;
        Ptr<Impl> impl_;
    };

    //! OpenGL Arrays
    class CV_EXPORTS GlArrays
    {
    public:
        inline GlArrays() 
            : vertex_(GlBuffer::ARRAY_BUFFER), color_(GlBuffer::ARRAY_BUFFER), bgra_(true), normal_(GlBuffer::ARRAY_BUFFER), texCoord_(GlBuffer::ARRAY_BUFFER)
        {
        }

        void setVertexArray(const GlBuffer& vertex);
        void setVertexArray(const GpuMat& vertex);
        void setVertexArray(InputArray vertex);
        inline void resetVertexArray() { vertex_.release(); }

        void setColorArray(const GlBuffer& color, bool bgra = true);
        void setColorArray(const GpuMat& color, bool bgra = true);
        void setColorArray(InputArray color, bool bgra = true);
        inline void resetColorArray() { color_.release(); }
        
        void setNormalArray(const GlBuffer& normal);
        void setNormalArray(const GpuMat& normal);
        void setNormalArray(InputArray normal);
        inline void resetNormalArray() { normal_.release(); }
        
        void setTexCoordArray(const GlBuffer& texCoord);
        void setTexCoordArray(const GpuMat& texCoord);
        void setTexCoordArray(InputArray texCoord);
        inline void resetTexCoordArray() { texCoord_.release(); }

        void bind() const;
        void unbind() const;

        inline int rows() const { return vertex_.rows; }
        inline int cols() const { return vertex_.cols; }
        inline Size size() const { return vertex_.size(); }
        inline bool empty() const { return vertex_.empty(); }

    private:
        GlBuffer vertex_;
        GlBuffer color_;
        bool bgra_;
        GlBuffer normal_;
        GlBuffer texCoord_;
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
    CV_EXPORTS void render(const GlArrays& arr, int mode = RenderMode::POINTS);

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

    //! OpenGL extension table
    class CV_EXPORTS GlFuncTab
    {
    public:
        virtual ~GlFuncTab();

        virtual void genBuffers(int n, unsigned int* buffers) const = 0;        
        virtual void deleteBuffers(int n, const unsigned int* buffers) const = 0;

        virtual void bufferData(unsigned int target, ptrdiff_t size, const void* data, unsigned int usage) const = 0;
        virtual void bufferSubData(unsigned int target, ptrdiff_t offset, ptrdiff_t size, const void* data) const = 0;

        virtual void bindBuffer(unsigned int target, unsigned int buffer) const = 0;

        virtual void* mapBuffer(unsigned int target, unsigned int access) const = 0;
        virtual void unmapBuffer(unsigned int target) const = 0;

        virtual bool isGlContextInitialized() const = 0;
    };

    CV_EXPORTS void setGlFuncTab(const GlFuncTab* tab);

    ////////////////////////////////////////////////////////////////////////
    // Error handling

    CV_EXPORTS void error(const char* error_string, const char* file, const int line, const char* func = "");
    CV_EXPORTS bool checkGlError(const char* file, const int line, const char* func = "");

    #if defined(__GNUC__)
        #define CV_CheckGlError() CV_DbgAssert( (cv::gpu::checkGlError(__FILE__, __LINE__, __func__)) )
    #else
        #define CV_CheckGlError() CV_DbgAssert( (cv::gpu::checkGlError(__FILE__, __LINE__)) )
    #endif

    ////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////

    inline GpuMat::GpuMat() 
        : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0) 
    {
    }

    inline GpuMat::GpuMat(int rows_, int cols_, int type_) 
        : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0)
    {
        if (rows_ > 0 && cols_ > 0)
            create(rows_, cols_, type_);
    }

    inline GpuMat::GpuMat(Size size_, int type_) 
        : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0)
    {
        if (size_.height > 0 && size_.width > 0)
            create(size_.height, size_.width, type_);
    }

    inline GpuMat::GpuMat(int rows_, int cols_, int type_, Scalar s_) 
        : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0)
    {
        if (rows_ > 0 && cols_ > 0)
        {
            create(rows_, cols_, type_);
            setTo(s_);
        }
    }

    inline GpuMat::GpuMat(Size size_, int type_, Scalar s_) 
        : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0)
    {
        if (size_.height > 0 && size_.width > 0)
        {
            create(size_.height, size_.width, type_);
            setTo(s_);
        }
    }    

    inline GpuMat::~GpuMat() 
    { 
        release(); 
    }

    inline GpuMat GpuMat::clone() const
    {
        GpuMat m;
        copyTo(m);
        return m;
    }

    inline void GpuMat::assignTo(GpuMat& m, int type) const
    {
        if (type < 0)
            m = *this;
        else
            convertTo(m, type);
    }

    inline size_t GpuMat::step1() const 
    { 
        return step / elemSize1(); 
    }

    inline bool GpuMat::empty() const 
    { 
        return data == 0; 
    }

    template<typename _Tp> inline _Tp* GpuMat::ptr(int y)
    {
        return (_Tp*)ptr(y);
    }

    template<typename _Tp> inline const _Tp* GpuMat::ptr(int y) const
    {
        return (const _Tp*)ptr(y);
    }

    inline void swap(GpuMat& a, GpuMat& b) 
    { 
        a.swap(b); 
    }

    inline GpuMat GpuMat::row(int y) const 
    { 
        return GpuMat(*this, Range(y, y+1), Range::all()); 
    }

    inline GpuMat GpuMat::col(int x) const 
    { 
        return GpuMat(*this, Range::all(), Range(x, x+1)); 
    }

    inline GpuMat GpuMat::rowRange(int startrow, int endrow) const 
    { 
        return GpuMat(*this, Range(startrow, endrow), Range::all()); 
    }

    inline GpuMat GpuMat::rowRange(Range r) const 
    { 
        return GpuMat(*this, r, Range::all()); 
    }

    inline GpuMat GpuMat::colRange(int startcol, int endcol) const 
    { 
        return GpuMat(*this, Range::all(), Range(startcol, endcol)); 
    }

    inline GpuMat GpuMat::colRange(Range r) const 
    { 
        return GpuMat(*this, Range::all(), r); 
    }

    inline void GpuMat::create(Size size_, int type_) 
    { 
        create(size_.height, size_.width, type_); 
    }

    inline GpuMat GpuMat::operator()(Range rowRange, Range colRange) const 
    { 
        return GpuMat(*this, rowRange, colRange); 
    }

    inline GpuMat GpuMat::operator()(Rect roi) const 
    { 
        return GpuMat(*this, roi); 
    }

    inline bool GpuMat::isContinuous() const 
    { 
        return (flags & Mat::CONTINUOUS_FLAG) != 0; 
    }

    inline size_t GpuMat::elemSize() const 
    { 
        return CV_ELEM_SIZE(flags); 
    }

    inline size_t GpuMat::elemSize1() const 
    { 
        return CV_ELEM_SIZE1(flags); 
    }

    inline int GpuMat::type() const 
    { 
        return CV_MAT_TYPE(flags); 
    }

    inline int GpuMat::depth() const 
    { 
        return CV_MAT_DEPTH(flags); 
    }

    inline int GpuMat::channels() const 
    { 
        return CV_MAT_CN(flags); 
    }

    inline Size GpuMat::size() const 
    { 
        return Size(cols, rows); 
    }

    inline uchar* GpuMat::ptr(int y)
    {
        CV_DbgAssert((unsigned)y < (unsigned)rows);
        return data + step * y;
    }

    inline const uchar* GpuMat::ptr(int y) const
    {
        CV_DbgAssert((unsigned)y < (unsigned)rows);
        return data + step * y;
    }

    inline GpuMat& GpuMat::operator = (Scalar s)
    {
        setTo(s);
        return *this;
    }

    template <class T> inline GpuMat::operator DevMem2D_<T>() const 
    { 
        return DevMem2D_<T>(rows, cols, (T*)data, step); 
    }

    template <class T> inline GpuMat::operator PtrStep_<T>() const 
    { 
        return PtrStep_<T>(static_cast< DevMem2D_<T> >(*this)); 
    }

    template <class T> inline GpuMat::operator PtrStep<T>() const 
    { 
        return PtrStep<T>((T*)data, step); 
    }

    inline GpuMat createContinuous(int rows, int cols, int type)
    {
        GpuMat m;
        createContinuous(rows, cols, type, m);
        return m;
    }

    inline void createContinuous(Size size, int type, GpuMat& m)
    {
        createContinuous(size.height, size.width, type, m);
    }

    inline GpuMat createContinuous(Size size, int type)
    {
        GpuMat m;
        createContinuous(size, type, m);
        return m;
    }

    inline void ensureSizeIsEnough(Size size, int type, GpuMat& m)
    {
        ensureSizeIsEnough(size.height, size.width, type, m);
    }

    inline void createContinuous(int rows, int cols, int type, GpuMat& m)
    {
        int area = rows * cols;
        if (!m.isContinuous() || m.type() != type || m.size().area() != area)
            m.create(1, area, type);
        m = m.reshape(0, rows);
    }

    inline void ensureSizeIsEnough(int rows, int cols, int type, GpuMat& m)
    {
        if (m.type() == type && m.rows >= rows && m.cols >= cols)
            m = m(Rect(0, 0, cols, rows));
        else
            m.create(rows, cols, type);
    }
}}

#endif // __cplusplus

#endif // __OPENCV_GPUMAT_HPP__
