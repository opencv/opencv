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

    CV_EXPORTS GpuMat allocMatFromBuf(int rows, int cols, int type, GpuMat &mat);

    ////////////////////////////////////////////////////////////////////////
    // Error handling

    CV_EXPORTS void error(const char* error_string, const char* file, const int line, const char* func = "");

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
            ensureSizeIsEnough(1, area, type, m);
        m = m.reshape(0, rows);
    }

    inline void ensureSizeIsEnough(int rows, int cols, int type, GpuMat& m)
    {
        if (m.type() == type && m.rows >= rows && m.cols >= cols)
            m = m(Rect(0, 0, cols, rows));
        else
            m.create(rows, cols, type);
    }

    inline GpuMat allocMatFromBuf(int rows, int cols, int type, GpuMat &mat)
    {
        if (!mat.empty() && mat.type() == type && mat.rows >= rows && mat.cols >= cols)
            return mat(Rect(0, 0, cols, rows));
        return mat = GpuMat(rows, cols, type);
    }
}}

#endif // __cplusplus

#endif // __OPENCV_GPUMAT_HPP__
