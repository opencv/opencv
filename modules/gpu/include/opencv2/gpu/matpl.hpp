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

#ifndef __OPENCV_GPU_MATPL_HPP__
#define __OPENCV_GPU_MATPL_HPP__

#include "opencv2/core/core.hpp"

namespace cv
{
    namespace gpu
    {   

        //////////////////////////////// MatPL ////////////////////////////////

        //class CV_EXPORTS MatPL : private Mat
        //{
        //public:            
        //    MatPL() {}            
        //    MatPL(int _rows, int _cols, int _type) : Mat(_rows, _cols, _type) {}
        //    MatPL(Size _size, int _type) : Mat(_size, _type) {}
        //    
        //    Mat(int _rows, int _cols, int _type, const Scalar& _s) : Mat
        //    MatPL(Size _size, int _type, const Scalar& _s);
        //    //! copy constructor
        //    MatPL(const Mat& m);
        //    //! constructor for matrix headers pointing to user-allocated data
        //    MatPL(int _rows, int _cols, int _type, void* _data, size_t _step=AUTO_STEP);
        //    MatPL(Size _size, int _type, void* _data, size_t _step=AUTO_STEP);
        //    //! creates a matrix header for a part of the bigger matrix
        //    MatPL(const Mat& m, const Range& rowRange, const Range& colRange);
        //    MatPL(const Mat& m, const Rect& roi);
        //    //! converts old-style CvMat to the new matrix; the data is not copied by default
        //    Mat(const CvMat* m, bool copyData=false);
        //    MatPL converts old-style IplImage to the new matrix; the data is not copied by default
        //    MatPL(const IplImage* img, bool copyData=false);
        //    //! builds matrix from std::vector with or without copying the data
        //    template<typename _Tp> explicit Mat(const vector<_Tp>& vec, bool copyData=false);
        //    //! builds matrix from cv::Vec; the data is copied by default
        //    template<typename _Tp, int n> explicit Mat(const Vec<_Tp, n>& vec,
        //        bool copyData=true);
        //    //! builds matrix from cv::Matx; the data is copied by default
        //    template<typename _Tp, int m, int n> explicit Mat(const Matx<_Tp, m, n>& mtx,
        //        bool copyData=true);
        //    //! builds matrix from a 2D point
        //    template<typename _Tp> explicit Mat(const Point_<_Tp>& pt);
        //    //! builds matrix from a 3D point
        //    template<typename _Tp> explicit Mat(const Point3_<_Tp>& pt);
        //    //! builds matrix from comma initializer
        //    template<typename _Tp> explicit Mat(const MatCommaInitializer_<_Tp>& commaInitializer);
        //    //! helper constructor to compile matrix expressions
        //    Mat(const MatExpr_Base& expr);
        //    //! destructor - calls release()
        //    ~Mat();
        //    //! assignment operators
        //    Mat& operator = (const Mat& m);
        //    Mat& operator = (const MatExpr_Base& expr);

        //    operator MatExpr_<Mat, Mat>() const;

        //    //! returns a new matrix header for the specified row
        //    Mat row(int y) const;
        //    //! returns a new matrix header for the specified column
        //    Mat col(int x) const;
        //    //! ... for the specified row span
        //    Mat rowRange(int startrow, int endrow) const;
        //    Mat rowRange(const Range& r) const;
        //    //! ... for the specified column span
        //    Mat colRange(int startcol, int endcol) const;
        //    Mat colRange(const Range& r) const;
        //    //! ... for the specified diagonal
        //    // (d=0 - the main diagonal,
        //    //  >0 - a diagonal from the lower half,
        //    //  <0 - a diagonal from the upper half)
        //    Mat diag(int d=0) const;
        //    //! constructs a square diagonal matrix which main diagonal is vector "d"
        //    static Mat diag(const Mat& d);

        //    //! returns deep copy of the matrix, i.e. the data is copied
        //    Mat clone() const;
        //    //! copies the matrix content to "m".
        //    // It calls m.create(this->size(), this->type()).
        //    void copyTo( Mat& m ) const;
        //    //! copies those matrix elements to "m" that are marked with non-zero mask elements.
        //    void copyTo( Mat& m, const Mat& mask ) const;
        //    //! converts matrix to another datatype with optional scalng. See cvConvertScale.
        //    void convertTo( Mat& m, int rtype, double alpha=1, double beta=0 ) const;

        //    void assignTo( Mat& m, int type=-1 ) const;

        //    //! sets every matrix element to s
        //    Mat& operator = (const Scalar& s);
        //    //! sets some of the matrix elements to s, according to the mask
        //    Mat& setTo(const Scalar& s, const Mat& mask=Mat());
        //    //! creates alternative matrix header for the same data, with different
        //    // number of channels and/or different number of rows. see cvReshape.
        //    Mat reshape(int _cn, int _rows=0) const;

        //    //! matrix transposition by means of matrix expressions
        //    MatExpr_<MatExpr_Op2_<Mat, double, Mat, MatOp_T_<Mat> >, Mat>
        //        t() const;
        //    //! matrix inversion by means of matrix expressions
        //    MatExpr_<MatExpr_Op2_<Mat, int, Mat, MatOp_Inv_<Mat> >, Mat>
        //        inv(int method=DECOMP_LU) const;
        //    MatExpr_<MatExpr_Op4_<Mat, Mat, double, char, Mat, MatOp_MulDiv_<Mat> >, Mat>
        //        //! per-element matrix multiplication by means of matrix expressions
        //        mul(const Mat& m, double scale=1) const;
        //    MatExpr_<MatExpr_Op4_<Mat, Mat, double, char, Mat, MatOp_MulDiv_<Mat> >, Mat>
        //        mul(const MatExpr_<MatExpr_Op2_<Mat, double, Mat, MatOp_Scale_<Mat> >, Mat>& m, double scale=1) const;
        //    MatExpr_<MatExpr_Op4_<Mat, Mat, double, char, Mat, MatOp_MulDiv_<Mat> >, Mat>    
        //        mul(const MatExpr_<MatExpr_Op2_<Mat, double, Mat, MatOp_DivRS_<Mat> >, Mat>& m, double scale=1) const;

        //    //! computes cross-product of 2 3D vectors
        //    Mat cross(const Mat& m) const;
        //    //! computes dot-product
        //    double dot(const Mat& m) const;

        //    //! Matlab-style matrix initialization
        //    static MatExpr_Initializer zeros(int rows, int cols, int type);
        //    static MatExpr_Initializer zeros(Size size, int type);
        //    static MatExpr_Initializer ones(int rows, int cols, int type);
        //    static MatExpr_Initializer ones(Size size, int type);
        //    static MatExpr_Initializer eye(int rows, int cols, int type);
        //    static MatExpr_Initializer eye(Size size, int type);

        //    //! allocates new matrix data unless the matrix already has specified size and type.
        //    // previous data is unreferenced if needed.
        //    void create(int _rows, int _cols, int _type);
        //    void create(Size _size, int _type);
        //    //! increases the reference counter; use with care to avoid memleaks
        //    void addref();
        //    //! decreases reference counter;
        //    // deallocate the data when reference counter reaches 0.
        //    void release();

        //    //! locates matrix header within a parent matrix. See below
        //    void locateROI( Size& wholeSize, Point& ofs ) const;
        //    //! moves/resizes the current matrix ROI inside the parent matrix.
        //    Mat& adjustROI( int dtop, int dbottom, int dleft, int dright );
        //    //! extracts a rectangular sub-matrix
        //    // (this is a generalized form of row, rowRange etc.)
        //    Mat operator()( Range rowRange, Range colRange ) const;
        //    Mat operator()( const Rect& roi ) const;

        //    //! converts header to CvMat; no data is copied
        //    operator CvMat() const;
        //    //! converts header to IplImage; no data is copied
        //    operator IplImage() const;

        //    //! returns true iff the matrix data is continuous
        //    // (i.e. when there are no gaps between successive rows).
        //    // similar to CV_IS_MAT_CONT(cvmat->type)
        //    bool isContinuous() const;
        //    //! returns element size in bytes,
        //    // similar to CV_ELEM_SIZE(cvmat->type)
        //    size_t elemSize() const;
        //    //! returns the size of element channel in bytes.
        //    size_t elemSize1() const;
        //    //! returns element type, similar to CV_MAT_TYPE(cvmat->type)
        //    int type() const;
        //    //! returns element type, similar to CV_MAT_DEPTH(cvmat->type)
        //    int depth() const;
        //    //! returns element type, similar to CV_MAT_CN(cvmat->type)
        //    int channels() const;
        //    //! returns step/elemSize1()
        //    size_t step1() const;
        //    //! returns matrix size:
        //    // width == number of columns, height == number of rows
        //    Size size() const;
        //    //! returns true if matrix data is NULL
        //    bool empty() const;

        //    //! returns pointer to y-th row
        //    uchar* ptr(int y=0);
        //    const uchar* ptr(int y=0) const;

        //    //! template version of the above method
        //    template<typename _Tp> _Tp* ptr(int y=0);
        //    template<typename _Tp> const _Tp* ptr(int y=0) const;

        //    //! template methods for read-write or read-only element access.
        //    // note that _Tp must match the actual matrix type -
        //    // the functions do not do any on-fly type conversion
        //    template<typename _Tp> _Tp& at(int y, int x);
        //    template<typename _Tp> _Tp& at(Point pt);
        //    template<typename _Tp> const _Tp& at(int y, int x) const;
        //    template<typename _Tp> const _Tp& at(Point pt) const;
        //    template<typename _Tp> _Tp& at(int i);
        //    template<typename _Tp> const _Tp& at(int i) const;

        //    //! template methods for iteration over matrix elements.
        //    // the iterators take care of skipping gaps in the end of rows (if any)
        //    template<typename _Tp> MatIterator_<_Tp> begin();
        //    template<typename _Tp> MatIterator_<_Tp> end();
        //    template<typename _Tp> MatConstIterator_<_Tp> begin() const;
        //    template<typename _Tp> MatConstIterator_<_Tp> end() const;

        //    enum { MAGIC_VAL=0x42FF0000, AUTO_STEP=0, CONTINUOUS_FLAG=CV_MAT_CONT_FLAG };

        //    /*! includes several bit-fields:
        //    - the magic signature
        //    - continuity flag
        //    - depth
        //    - number of channels
        //    */
        //    int flags;
        //    //! the number of rows and columns
        //    int rows, cols;
        //    //! a distance between successive rows in bytes; includes the gap if any
        //    size_t step;
        //    //! pointer to the data
        //    uchar* data;

        //    //! pointer to the reference counter;
        //    // when matrix points to user-allocated data, the pointer is NULL
        //    int* refcount;

        //    //! helper fields used in locateROI and adjustROI
        //    uchar* datastart;
        //    uchar* dataend;
        //};
    }
}


#endif /* __OPENCV_GPU_MATPL_HPP__ */