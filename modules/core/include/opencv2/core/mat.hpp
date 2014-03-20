/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#ifndef __OPENCV_CORE_MAT_HPP__
#define __OPENCV_CORE_MAT_HPP__

#ifndef __cplusplus
#  error mat.hpp header must be compiled as C++
#endif

#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"

#include "opencv2/core/bufferpool.hpp"

namespace cv
{

enum { ACCESS_READ=1<<24, ACCESS_WRITE=1<<25,
    ACCESS_RW=3<<24, ACCESS_MASK=ACCESS_RW, ACCESS_FAST=1<<26 };

class CV_EXPORTS _OutputArray;

//////////////////////// Input/Output Array Arguments /////////////////////////////////

/*!
 Proxy datatype for passing Mat's and vector<>'s as input parameters
 */
class CV_EXPORTS _InputArray
{
public:
    enum {
        KIND_SHIFT = 16,
        FIXED_TYPE = 0x8000 << KIND_SHIFT,
        FIXED_SIZE = 0x4000 << KIND_SHIFT,
        KIND_MASK = 31 << KIND_SHIFT,

        NONE              = 0 << KIND_SHIFT,
        MAT               = 1 << KIND_SHIFT,
        MATX              = 2 << KIND_SHIFT,
        STD_VECTOR        = 3 << KIND_SHIFT,
        STD_VECTOR_VECTOR = 4 << KIND_SHIFT,
        STD_VECTOR_MAT    = 5 << KIND_SHIFT,
        EXPR              = 6 << KIND_SHIFT,
        OPENGL_BUFFER     = 7 << KIND_SHIFT,
        CUDA_MEM          = 8 << KIND_SHIFT,
        GPU_MAT           = 9 << KIND_SHIFT,
        UMAT              =10 << KIND_SHIFT,
        STD_VECTOR_UMAT   =11 << KIND_SHIFT
    };

    _InputArray();
    _InputArray(int _flags, void* _obj);
    _InputArray(const Mat& m);
    _InputArray(const MatExpr& expr);
    _InputArray(const std::vector<Mat>& vec);
    template<typename _Tp> _InputArray(const Mat_<_Tp>& m);
    template<typename _Tp> _InputArray(const std::vector<_Tp>& vec);
    template<typename _Tp> _InputArray(const std::vector<std::vector<_Tp> >& vec);
    template<typename _Tp> _InputArray(const std::vector<Mat_<_Tp> >& vec);
    template<typename _Tp> _InputArray(const _Tp* vec, int n);
    template<typename _Tp, int m, int n> _InputArray(const Matx<_Tp, m, n>& matx);
    _InputArray(const double& val);
    _InputArray(const cuda::GpuMat& d_mat);
    _InputArray(const ogl::Buffer& buf);
    _InputArray(const cuda::CudaMem& cuda_mem);
    template<typename _Tp> _InputArray(const cudev::GpuMat_<_Tp>& m);
    _InputArray(const UMat& um);
    _InputArray(const std::vector<UMat>& umv);

    virtual Mat getMat(int idx=-1) const;
    virtual UMat getUMat(int idx=-1) const;
    virtual void getMatVector(std::vector<Mat>& mv) const;
    virtual void getUMatVector(std::vector<UMat>& umv) const;
    virtual cuda::GpuMat getGpuMat() const;
    virtual ogl::Buffer getOGlBuffer() const;
    void* getObj() const;

    virtual int kind() const;
    virtual int dims(int i=-1) const;
    virtual int cols(int i=-1) const;
    virtual int rows(int i=-1) const;
    virtual Size size(int i=-1) const;
    virtual int sizend(int* sz, int i=-1) const;
    virtual bool sameSize(const _InputArray& arr) const;
    virtual size_t total(int i=-1) const;
    virtual int type(int i=-1) const;
    virtual int depth(int i=-1) const;
    virtual int channels(int i=-1) const;
    virtual bool isContinuous(int i=-1) const;
    virtual bool isSubmatrix(int i=-1) const;
    virtual bool empty() const;
    virtual void copyTo(const _OutputArray& arr) const;
    virtual size_t offset(int i=-1) const;
    virtual size_t step(int i=-1) const;
    bool isMat() const;
    bool isUMat() const;
    bool isMatVector() const;
    bool isUMatVector() const;
    bool isMatx() const;

    virtual ~_InputArray();

protected:
    int flags;
    void* obj;
    Size sz;

    void init(int _flags, const void* _obj);
    void init(int _flags, const void* _obj, Size _sz);
};


/*!
 Proxy datatype for passing Mat's and vector<>'s as input parameters
 */
class CV_EXPORTS _OutputArray : public _InputArray
{
public:
    enum
    {
        DEPTH_MASK_8U = 1 << CV_8U,
        DEPTH_MASK_8S = 1 << CV_8S,
        DEPTH_MASK_16U = 1 << CV_16U,
        DEPTH_MASK_16S = 1 << CV_16S,
        DEPTH_MASK_32S = 1 << CV_32S,
        DEPTH_MASK_32F = 1 << CV_32F,
        DEPTH_MASK_64F = 1 << CV_64F,
        DEPTH_MASK_ALL = (DEPTH_MASK_64F<<1)-1,
        DEPTH_MASK_ALL_BUT_8S = DEPTH_MASK_ALL & ~DEPTH_MASK_8S,
        DEPTH_MASK_FLT = DEPTH_MASK_32F + DEPTH_MASK_64F
    };

    _OutputArray();
    _OutputArray(int _flags, void* _obj);
    _OutputArray(Mat& m);
    _OutputArray(std::vector<Mat>& vec);
    _OutputArray(cuda::GpuMat& d_mat);
    _OutputArray(ogl::Buffer& buf);
    _OutputArray(cuda::CudaMem& cuda_mem);
    template<typename _Tp> _OutputArray(cudev::GpuMat_<_Tp>& m);
    template<typename _Tp> _OutputArray(std::vector<_Tp>& vec);
    template<typename _Tp> _OutputArray(std::vector<std::vector<_Tp> >& vec);
    template<typename _Tp> _OutputArray(std::vector<Mat_<_Tp> >& vec);
    template<typename _Tp> _OutputArray(Mat_<_Tp>& m);
    template<typename _Tp> _OutputArray(_Tp* vec, int n);
    template<typename _Tp, int m, int n> _OutputArray(Matx<_Tp, m, n>& matx);
    _OutputArray(UMat& m);
    _OutputArray(std::vector<UMat>& vec);

    _OutputArray(const Mat& m);
    _OutputArray(const std::vector<Mat>& vec);
    _OutputArray(const cuda::GpuMat& d_mat);
    _OutputArray(const ogl::Buffer& buf);
    _OutputArray(const cuda::CudaMem& cuda_mem);
    template<typename _Tp> _OutputArray(const cudev::GpuMat_<_Tp>& m);
    template<typename _Tp> _OutputArray(const std::vector<_Tp>& vec);
    template<typename _Tp> _OutputArray(const std::vector<std::vector<_Tp> >& vec);
    template<typename _Tp> _OutputArray(const std::vector<Mat_<_Tp> >& vec);
    template<typename _Tp> _OutputArray(const Mat_<_Tp>& m);
    template<typename _Tp> _OutputArray(const _Tp* vec, int n);
    template<typename _Tp, int m, int n> _OutputArray(const Matx<_Tp, m, n>& matx);
    _OutputArray(const UMat& m);
    _OutputArray(const std::vector<UMat>& vec);

    virtual bool fixedSize() const;
    virtual bool fixedType() const;
    virtual bool needed() const;
    virtual Mat& getMatRef(int i=-1) const;
    virtual UMat& getUMatRef(int i=-1) const;
    virtual cuda::GpuMat& getGpuMatRef() const;
    virtual ogl::Buffer& getOGlBufferRef() const;
    virtual cuda::CudaMem& getCudaMemRef() const;
    virtual void create(Size sz, int type, int i=-1, bool allowTransposed=false, int fixedDepthMask=0) const;
    virtual void create(int rows, int cols, int type, int i=-1, bool allowTransposed=false, int fixedDepthMask=0) const;
    virtual void create(int dims, const int* size, int type, int i=-1, bool allowTransposed=false, int fixedDepthMask=0) const;
    virtual void createSameSize(const _InputArray& arr, int mtype) const;
    virtual void release() const;
    virtual void clear() const;
    virtual void setTo(const _InputArray& value, const _InputArray & mask = _InputArray()) const;
};


class CV_EXPORTS _InputOutputArray : public _OutputArray
{
public:
    _InputOutputArray();
    _InputOutputArray(int _flags, void* _obj);
    _InputOutputArray(Mat& m);
    _InputOutputArray(std::vector<Mat>& vec);
    _InputOutputArray(cuda::GpuMat& d_mat);
    _InputOutputArray(ogl::Buffer& buf);
    _InputOutputArray(cuda::CudaMem& cuda_mem);
    template<typename _Tp> _InputOutputArray(cudev::GpuMat_<_Tp>& m);
    template<typename _Tp> _InputOutputArray(std::vector<_Tp>& vec);
    template<typename _Tp> _InputOutputArray(std::vector<std::vector<_Tp> >& vec);
    template<typename _Tp> _InputOutputArray(std::vector<Mat_<_Tp> >& vec);
    template<typename _Tp> _InputOutputArray(Mat_<_Tp>& m);
    template<typename _Tp> _InputOutputArray(_Tp* vec, int n);
    template<typename _Tp, int m, int n> _InputOutputArray(Matx<_Tp, m, n>& matx);
    _InputOutputArray(UMat& m);
    _InputOutputArray(std::vector<UMat>& vec);

    _InputOutputArray(const Mat& m);
    _InputOutputArray(const std::vector<Mat>& vec);
    _InputOutputArray(const cuda::GpuMat& d_mat);
    _InputOutputArray(const ogl::Buffer& buf);
    _InputOutputArray(const cuda::CudaMem& cuda_mem);
    template<typename _Tp> _InputOutputArray(const cudev::GpuMat_<_Tp>& m);
    template<typename _Tp> _InputOutputArray(const std::vector<_Tp>& vec);
    template<typename _Tp> _InputOutputArray(const std::vector<std::vector<_Tp> >& vec);
    template<typename _Tp> _InputOutputArray(const std::vector<Mat_<_Tp> >& vec);
    template<typename _Tp> _InputOutputArray(const Mat_<_Tp>& m);
    template<typename _Tp> _InputOutputArray(const _Tp* vec, int n);
    template<typename _Tp, int m, int n> _InputOutputArray(const Matx<_Tp, m, n>& matx);
    _InputOutputArray(const UMat& m);
    _InputOutputArray(const std::vector<UMat>& vec);
};

typedef const _InputArray& InputArray;
typedef InputArray InputArrayOfArrays;
typedef const _OutputArray& OutputArray;
typedef OutputArray OutputArrayOfArrays;
typedef const _InputOutputArray& InputOutputArray;
typedef InputOutputArray InputOutputArrayOfArrays;

CV_EXPORTS InputOutputArray noArray();

/////////////////////////////////// MatAllocator //////////////////////////////////////

//! Usage flags for allocator
enum UMatUsageFlags
{
    USAGE_DEFAULT = 0,

    // default allocation policy is platform and usage specific
    USAGE_ALLOCATE_HOST_MEMORY = 1 << 0,
    USAGE_ALLOCATE_DEVICE_MEMORY = 1 << 1,

    __UMAT_USAGE_FLAGS_32BIT = 0x7fffffff // Binary compatibility hint
};

struct CV_EXPORTS UMatData;

/*!
   Custom array allocator

*/
class CV_EXPORTS MatAllocator
{
public:
    MatAllocator() {}
    virtual ~MatAllocator() {}

    // let's comment it off for now to detect and fix all the uses of allocator
    //virtual void allocate(int dims, const int* sizes, int type, int*& refcount,
    //                      uchar*& datastart, uchar*& data, size_t* step) = 0;
    //virtual void deallocate(int* refcount, uchar* datastart, uchar* data) = 0;
    virtual UMatData* allocate(int dims, const int* sizes, int type,
                               void* data, size_t* step, int flags, UMatUsageFlags usageFlags) const = 0;
    virtual bool allocate(UMatData* data, int accessflags, UMatUsageFlags usageFlags) const = 0;
    virtual void deallocate(UMatData* data) const = 0;
    virtual void map(UMatData* data, int accessflags) const;
    virtual void unmap(UMatData* data) const;
    virtual void download(UMatData* data, void* dst, int dims, const size_t sz[],
                          const size_t srcofs[], const size_t srcstep[],
                          const size_t dststep[]) const;
    virtual void upload(UMatData* data, const void* src, int dims, const size_t sz[],
                        const size_t dstofs[], const size_t dststep[],
                        const size_t srcstep[]) const;
    virtual void copy(UMatData* srcdata, UMatData* dstdata, int dims, const size_t sz[],
                      const size_t srcofs[], const size_t srcstep[],
                      const size_t dstofs[], const size_t dststep[], bool sync) const;

    // default implementation returns DummyBufferPoolController
    virtual BufferPoolController* getBufferPoolController() const;
};


//////////////////////////////// MatCommaInitializer //////////////////////////////////

/*!
 Comma-separated Matrix Initializer

 The class instances are usually not created explicitly.
 Instead, they are created on "matrix << firstValue" operator.

 The sample below initializes 2x2 rotation matrix:

 \code
 double angle = 30, a = cos(angle*CV_PI/180), b = sin(angle*CV_PI/180);
 Mat R = (Mat_<double>(2,2) << a, -b, b, a);
 \endcode
*/
template<typename _Tp> class MatCommaInitializer_
{
public:
    //! the constructor, created by "matrix << firstValue" operator, where matrix is cv::Mat
    MatCommaInitializer_(Mat_<_Tp>* _m);
    //! the operator that takes the next value and put it to the matrix
    template<typename T2> MatCommaInitializer_<_Tp>& operator , (T2 v);
    //! another form of conversion operator
    operator Mat_<_Tp>() const;
protected:
    MatIterator_<_Tp> it;
};


/////////////////////////////////////// Mat ///////////////////////////////////////////

// note that umatdata might be allocated together
// with the matrix data, not as a separate object.
// therefore, it does not have constructor or destructor;
// it should be explicitly initialized using init().
struct CV_EXPORTS UMatData
{
    enum { COPY_ON_MAP=1, HOST_COPY_OBSOLETE=2,
        DEVICE_COPY_OBSOLETE=4, TEMP_UMAT=8, TEMP_COPIED_UMAT=24,
        USER_ALLOCATED=32 };
    UMatData(const MatAllocator* allocator);
    ~UMatData();

    // provide atomic access to the structure
    void lock();
    void unlock();

    bool hostCopyObsolete() const;
    bool deviceCopyObsolete() const;
    bool copyOnMap() const;
    bool tempUMat() const;
    bool tempCopiedUMat() const;
    void markHostCopyObsolete(bool flag);
    void markDeviceCopyObsolete(bool flag);

    const MatAllocator* prevAllocator;
    const MatAllocator* currAllocator;
    int urefcount;
    int refcount;
    uchar* data;
    uchar* origdata;
    size_t size, capacity;

    int flags;
    void* handle;
    void* userdata;
    int allocatorFlags_;
};


struct CV_EXPORTS UMatDataAutoLock
{
    UMatDataAutoLock(UMatData* u);
    ~UMatDataAutoLock();
    UMatData* u;
};


struct CV_EXPORTS MatSize
{
    MatSize(int* _p);
    Size operator()() const;
    const int& operator[](int i) const;
    int& operator[](int i);
    operator const int*() const;
    bool operator == (const MatSize& sz) const;
    bool operator != (const MatSize& sz) const;

    int* p;
};

struct CV_EXPORTS MatStep
{
    MatStep();
    MatStep(size_t s);
    const size_t& operator[](int i) const;
    size_t& operator[](int i);
    operator size_t() const;
    MatStep& operator = (size_t s);

    size_t* p;
    size_t buf[2];
protected:
    MatStep& operator = (const MatStep&);
};

 /*!
   The n-dimensional matrix class.

   The class represents an n-dimensional dense numerical array that can act as
   a matrix, image, optical flow map, 3-focal tensor etc.
   It is very similar to CvMat and CvMatND types from earlier versions of OpenCV,
   and similarly to those types, the matrix can be multi-channel. It also fully supports ROI mechanism.

   There are many different ways to create cv::Mat object. Here are the some popular ones:
   <ul>
   <li> using cv::Mat::create(nrows, ncols, type) method or
     the similar constructor cv::Mat::Mat(nrows, ncols, type[, fill_value]) constructor.
     A new matrix of the specified size and specifed type will be allocated.
     "type" has the same meaning as in cvCreateMat function,
     e.g. CV_8UC1 means 8-bit single-channel matrix, CV_32FC2 means 2-channel (i.e. complex)
     floating-point matrix etc:

     \code
     // make 7x7 complex matrix filled with 1+3j.
     cv::Mat M(7,7,CV_32FC2,Scalar(1,3));
     // and now turn M to 100x60 15-channel 8-bit matrix.
     // The old content will be deallocated
     M.create(100,60,CV_8UC(15));
     \endcode

     As noted in the introduction of this chapter, Mat::create()
     will only allocate a new matrix when the current matrix dimensionality
     or type are different from the specified.

   <li> by using a copy constructor or assignment operator, where on the right side it can
     be a matrix or expression, see below. Again, as noted in the introduction,
     matrix assignment is O(1) operation because it only copies the header
     and increases the reference counter. cv::Mat::clone() method can be used to get a full
     (a.k.a. deep) copy of the matrix when you need it.

   <li> by constructing a header for a part of another matrix. It can be a single row, single column,
     several rows, several columns, rectangular region in the matrix (called a minor in algebra) or
     a diagonal. Such operations are also O(1), because the new header will reference the same data.
     You can actually modify a part of the matrix using this feature, e.g.

     \code
     // add 5-th row, multiplied by 3 to the 3rd row
     M.row(3) = M.row(3) + M.row(5)*3;

     // now copy 7-th column to the 1-st column
     // M.col(1) = M.col(7); // this will not work
     Mat M1 = M.col(1);
     M.col(7).copyTo(M1);

     // create new 320x240 image
     cv::Mat img(Size(320,240),CV_8UC3);
     // select a roi
     cv::Mat roi(img, Rect(10,10,100,100));
     // fill the ROI with (0,255,0) (which is green in RGB space);
     // the original 320x240 image will be modified
     roi = Scalar(0,255,0);
     \endcode

     Thanks to the additional cv::Mat::datastart and cv::Mat::dataend members, it is possible to
     compute the relative sub-matrix position in the main "container" matrix using cv::Mat::locateROI():

     \code
     Mat A = Mat::eye(10, 10, CV_32S);
     // extracts A columns, 1 (inclusive) to 3 (exclusive).
     Mat B = A(Range::all(), Range(1, 3));
     // extracts B rows, 5 (inclusive) to 9 (exclusive).
     // that is, C ~ A(Range(5, 9), Range(1, 3))
     Mat C = B(Range(5, 9), Range::all());
     Size size; Point ofs;
     C.locateROI(size, ofs);
     // size will be (width=10,height=10) and the ofs will be (x=1, y=5)
     \endcode

     As in the case of whole matrices, if you need a deep copy, use cv::Mat::clone() method
     of the extracted sub-matrices.

   <li> by making a header for user-allocated-data. It can be useful for
      <ol>
      <li> processing "foreign" data using OpenCV (e.g. when you implement
         a DirectShow filter or a processing module for gstreamer etc.), e.g.

         \code
         void process_video_frame(const unsigned char* pixels,
                                  int width, int height, int step)
         {
            cv::Mat img(height, width, CV_8UC3, pixels, step);
            cv::GaussianBlur(img, img, cv::Size(7,7), 1.5, 1.5);
         }
         \endcode

      <li> for quick initialization of small matrices and/or super-fast element access

         \code
         double m[3][3] = {{a, b, c}, {d, e, f}, {g, h, i}};
         cv::Mat M = cv::Mat(3, 3, CV_64F, m).inv();
         \endcode
      </ol>

       partial yet very common cases of this "user-allocated data" case are conversions
       from CvMat and IplImage to cv::Mat. For this purpose there are special constructors
       taking pointers to CvMat or IplImage and the optional
       flag indicating whether to copy the data or not.

       Backward conversion from cv::Mat to CvMat or IplImage is provided via cast operators
       cv::Mat::operator CvMat() an cv::Mat::operator IplImage().
       The operators do not copy the data.


       \code
       IplImage* img = cvLoadImage("greatwave.jpg", 1);
       Mat mtx(img); // convert IplImage* -> cv::Mat
       CvMat oldmat = mtx; // convert cv::Mat -> CvMat
       CV_Assert(oldmat.cols == img->width && oldmat.rows == img->height &&
           oldmat.data.ptr == (uchar*)img->imageData && oldmat.step == img->widthStep);
       \endcode

   <li> by using MATLAB-style matrix initializers, cv::Mat::zeros(), cv::Mat::ones(), cv::Mat::eye(), e.g.:

   \code
   // create a double-precision identity martix and add it to M.
   M += Mat::eye(M.rows, M.cols, CV_64F);
   \endcode

   <li> by using comma-separated initializer:

   \code
   // create 3x3 double-precision identity matrix
   Mat M = (Mat_<double>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
   \endcode

   here we first call constructor of cv::Mat_ class (that we describe further) with the proper matrix,
   and then we just put "<<" operator followed by comma-separated values that can be constants,
   variables, expressions etc. Also, note the extra parentheses that are needed to avoid compiler errors.

   </ul>

   Once matrix is created, it will be automatically managed by using reference-counting mechanism
   (unless the matrix header is built on top of user-allocated data,
   in which case you should handle the data by yourself).
   The matrix data will be deallocated when no one points to it;
   if you want to release the data pointed by a matrix header before the matrix destructor is called,
   use cv::Mat::release().

   The next important thing to learn about the matrix class is element access. Here is how the matrix is stored.
   The elements are stored in row-major order (row by row). The cv::Mat::data member points to the first element of the first row,
   cv::Mat::rows contains the number of matrix rows and cv::Mat::cols - the number of matrix columns. There is yet another member,
   cv::Mat::step that is used to actually compute address of a matrix element. cv::Mat::step is needed because the matrix can be
   a part of another matrix or because there can some padding space in the end of each row for a proper alignment.

   \image html roi.png

   Given these parameters, address of the matrix element M_{ij} is computed as following:

   addr(M_{ij})=M.data + M.step*i + j*M.elemSize()

   if you know the matrix element type, e.g. it is float, then you can use cv::Mat::at() method:

   addr(M_{ij})=&M.at<float>(i,j)

   (where & is used to convert the reference returned by cv::Mat::at() to a pointer).
   if you need to process a whole row of matrix, the most efficient way is to get
   the pointer to the row first, and then just use plain C operator []:

   \code
   // compute sum of positive matrix elements
   // (assuming that M is double-precision matrix)
   double sum=0;
   for(int i = 0; i < M.rows; i++)
   {
       const double* Mi = M.ptr<double>(i);
       for(int j = 0; j < M.cols; j++)
           sum += std::max(Mi[j], 0.);
   }
   \endcode

   Some operations, like the above one, do not actually depend on the matrix shape,
   they just process elements of a matrix one by one (or elements from multiple matrices
   that are sitting in the same place, e.g. matrix addition). Such operations are called
   element-wise and it makes sense to check whether all the input/output matrices are continuous,
   i.e. have no gaps in the end of each row, and if yes, process them as a single long row:

   \code
   // compute sum of positive matrix elements, optimized variant
   double sum=0;
   int cols = M.cols, rows = M.rows;
   if(M.isContinuous())
   {
       cols *= rows;
       rows = 1;
   }
   for(int i = 0; i < rows; i++)
   {
       const double* Mi = M.ptr<double>(i);
       for(int j = 0; j < cols; j++)
           sum += std::max(Mi[j], 0.);
   }
   \endcode
   in the case of continuous matrix the outer loop body will be executed just once,
   so the overhead will be smaller, which will be especially noticeable in the case of small matrices.

   Finally, there are STL-style iterators that are smart enough to skip gaps between successive rows:
   \code
   // compute sum of positive matrix elements, iterator-based variant
   double sum=0;
   MatConstIterator_<double> it = M.begin<double>(), it_end = M.end<double>();
   for(; it != it_end; ++it)
       sum += std::max(*it, 0.);
   \endcode

   The matrix iterators are random-access iterators, so they can be passed
   to any STL algorithm, including std::sort().
*/
class CV_EXPORTS Mat
{
public:
    //! default constructor
    Mat();
    //! constructs 2D matrix of the specified size and type
    // (_type is CV_8UC1, CV_64FC3, CV_32SC(12) etc.)
    Mat(int rows, int cols, int type);
    Mat(Size size, int type);
    //! constucts 2D matrix and fills it with the specified value _s.
    Mat(int rows, int cols, int type, const Scalar& s);
    Mat(Size size, int type, const Scalar& s);

    //! constructs n-dimensional matrix
    Mat(int ndims, const int* sizes, int type);
    Mat(int ndims, const int* sizes, int type, const Scalar& s);

    //! copy constructor
    Mat(const Mat& m);
    //! constructor for matrix headers pointing to user-allocated data
    Mat(int rows, int cols, int type, void* data, size_t step=AUTO_STEP);
    Mat(Size size, int type, void* data, size_t step=AUTO_STEP);
    Mat(int ndims, const int* sizes, int type, void* data, const size_t* steps=0);

    //! creates a matrix header for a part of the bigger matrix
    Mat(const Mat& m, const Range& rowRange, const Range& colRange=Range::all());
    Mat(const Mat& m, const Rect& roi);
    Mat(const Mat& m, const Range* ranges);
    //! builds matrix from std::vector with or without copying the data
    template<typename _Tp> explicit Mat(const std::vector<_Tp>& vec, bool copyData=false);
    //! builds matrix from cv::Vec; the data is copied by default
    template<typename _Tp, int n> explicit Mat(const Vec<_Tp, n>& vec, bool copyData=true);
    //! builds matrix from cv::Matx; the data is copied by default
    template<typename _Tp, int m, int n> explicit Mat(const Matx<_Tp, m, n>& mtx, bool copyData=true);
    //! builds matrix from a 2D point
    template<typename _Tp> explicit Mat(const Point_<_Tp>& pt, bool copyData=true);
    //! builds matrix from a 3D point
    template<typename _Tp> explicit Mat(const Point3_<_Tp>& pt, bool copyData=true);
    //! builds matrix from comma initializer
    template<typename _Tp> explicit Mat(const MatCommaInitializer_<_Tp>& commaInitializer);

    //! download data from GpuMat
    explicit Mat(const cuda::GpuMat& m);

    //! destructor - calls release()
    ~Mat();
    //! assignment operators
    Mat& operator = (const Mat& m);
    Mat& operator = (const MatExpr& expr);

    //! retrieve UMat from Mat
    UMat getUMat(int accessFlags, UMatUsageFlags usageFlags = USAGE_DEFAULT) const;

    //! returns a new matrix header for the specified row
    Mat row(int y) const;
    //! returns a new matrix header for the specified column
    Mat col(int x) const;
    //! ... for the specified row span
    Mat rowRange(int startrow, int endrow) const;
    Mat rowRange(const Range& r) const;
    //! ... for the specified column span
    Mat colRange(int startcol, int endcol) const;
    Mat colRange(const Range& r) const;
    //! ... for the specified diagonal
    // (d=0 - the main diagonal,
    //  >0 - a diagonal from the lower half,
    //  <0 - a diagonal from the upper half)
    Mat diag(int d=0) const;
    //! constructs a square diagonal matrix which main diagonal is vector "d"
    static Mat diag(const Mat& d);

    //! returns deep copy of the matrix, i.e. the data is copied
    Mat clone() const;
    //! copies the matrix content to "m".
    // It calls m.create(this->size(), this->type()).
    void copyTo( OutputArray m ) const;
    //! copies those matrix elements to "m" that are marked with non-zero mask elements.
    void copyTo( OutputArray m, InputArray mask ) const;
    //! converts matrix to another datatype with optional scalng. See cvConvertScale.
    void convertTo( OutputArray m, int rtype, double alpha=1, double beta=0 ) const;

    void assignTo( Mat& m, int type=-1 ) const;

    //! sets every matrix element to s
    Mat& operator = (const Scalar& s);
    //! sets some of the matrix elements to s, according to the mask
    Mat& setTo(InputArray value, InputArray mask=noArray());
    //! creates alternative matrix header for the same data, with different
    // number of channels and/or different number of rows. see cvReshape.
    Mat reshape(int cn, int rows=0) const;
    Mat reshape(int cn, int newndims, const int* newsz) const;

    //! matrix transposition by means of matrix expressions
    MatExpr t() const;
    //! matrix inversion by means of matrix expressions
    MatExpr inv(int method=DECOMP_LU) const;
    //! per-element matrix multiplication by means of matrix expressions
    MatExpr mul(InputArray m, double scale=1) const;

    //! computes cross-product of 2 3D vectors
    Mat cross(InputArray m) const;
    //! computes dot-product
    double dot(InputArray m) const;

    //! Matlab-style matrix initialization
    static MatExpr zeros(int rows, int cols, int type);
    static MatExpr zeros(Size size, int type);
    static MatExpr zeros(int ndims, const int* sz, int type);
    static MatExpr ones(int rows, int cols, int type);
    static MatExpr ones(Size size, int type);
    static MatExpr ones(int ndims, const int* sz, int type);
    static MatExpr eye(int rows, int cols, int type);
    static MatExpr eye(Size size, int type);

    //! allocates new matrix data unless the matrix already has specified size and type.
    // previous data is unreferenced if needed.
    void create(int rows, int cols, int type);
    void create(Size size, int type);
    void create(int ndims, const int* sizes, int type);

    //! increases the reference counter; use with care to avoid memleaks
    void addref();
    //! decreases reference counter;
    // deallocates the data when reference counter reaches 0.
    void release();

    //! deallocates the matrix data
    void deallocate();
    //! internal use function; properly re-allocates _size, _step arrays
    void copySize(const Mat& m);

    //! reserves enough space to fit sz hyper-planes
    void reserve(size_t sz);
    //! resizes matrix to the specified number of hyper-planes
    void resize(size_t sz);
    //! resizes matrix to the specified number of hyper-planes; initializes the newly added elements
    void resize(size_t sz, const Scalar& s);
    //! internal function
    void push_back_(const void* elem);
    //! adds element to the end of 1d matrix (or possibly multiple elements when _Tp=Mat)
    template<typename _Tp> void push_back(const _Tp& elem);
    template<typename _Tp> void push_back(const Mat_<_Tp>& elem);
    void push_back(const Mat& m);
    //! removes several hyper-planes from bottom of the matrix
    void pop_back(size_t nelems=1);

    //! locates matrix header within a parent matrix. See below
    void locateROI( Size& wholeSize, Point& ofs ) const;
    //! moves/resizes the current matrix ROI inside the parent matrix.
    Mat& adjustROI( int dtop, int dbottom, int dleft, int dright );
    //! extracts a rectangular sub-matrix
    // (this is a generalized form of row, rowRange etc.)
    Mat operator()( Range rowRange, Range colRange ) const;
    Mat operator()( const Rect& roi ) const;
    Mat operator()( const Range* ranges ) const;

    // //! converts header to CvMat; no data is copied
    // operator CvMat() const;
    // //! converts header to CvMatND; no data is copied
    // operator CvMatND() const;
    // //! converts header to IplImage; no data is copied
    // operator IplImage() const;

    template<typename _Tp> operator std::vector<_Tp>() const;
    template<typename _Tp, int n> operator Vec<_Tp, n>() const;
    template<typename _Tp, int m, int n> operator Matx<_Tp, m, n>() const;

    //! returns true iff the matrix data is continuous
    // (i.e. when there are no gaps between successive rows).
    // similar to CV_IS_MAT_CONT(cvmat->type)
    bool isContinuous() const;

    //! returns true if the matrix is a submatrix of another matrix
    bool isSubmatrix() const;

    //! returns element size in bytes,
    // similar to CV_ELEM_SIZE(cvmat->type)
    size_t elemSize() const;
    //! returns the size of element channel in bytes.
    size_t elemSize1() const;
    //! returns element type, similar to CV_MAT_TYPE(cvmat->type)
    int type() const;
    //! returns element type, similar to CV_MAT_DEPTH(cvmat->type)
    int depth() const;
    //! returns element type, similar to CV_MAT_CN(cvmat->type)
    int channels() const;
    //! returns step/elemSize1()
    size_t step1(int i=0) const;
    //! returns true if matrix data is NULL
    bool empty() const;
    //! returns the total number of matrix elements
    size_t total() const;

    //! returns N if the matrix is 1-channel (N x ptdim) or ptdim-channel (1 x N) or (N x 1); negative number otherwise
    int checkVector(int elemChannels, int depth=-1, bool requireContinuous=true) const;

    //! returns pointer to i0-th submatrix along the dimension #0
    uchar* ptr(int i0=0);
    const uchar* ptr(int i0=0) const;

    //! returns pointer to (i0,i1) submatrix along the dimensions #0 and #1
    uchar* ptr(int i0, int i1);
    const uchar* ptr(int i0, int i1) const;

    //! returns pointer to (i0,i1,i3) submatrix along the dimensions #0, #1, #2
    uchar* ptr(int i0, int i1, int i2);
    const uchar* ptr(int i0, int i1, int i2) const;

    //! returns pointer to the matrix element
    uchar* ptr(const int* idx);
    //! returns read-only pointer to the matrix element
    const uchar* ptr(const int* idx) const;

    template<int n> uchar* ptr(const Vec<int, n>& idx);
    template<int n> const uchar* ptr(const Vec<int, n>& idx) const;

    //! template version of the above method
    template<typename _Tp> _Tp* ptr(int i0=0);
    template<typename _Tp> const _Tp* ptr(int i0=0) const;

    template<typename _Tp> _Tp* ptr(int i0, int i1);
    template<typename _Tp> const _Tp* ptr(int i0, int i1) const;

    template<typename _Tp> _Tp* ptr(int i0, int i1, int i2);
    template<typename _Tp> const _Tp* ptr(int i0, int i1, int i2) const;

    template<typename _Tp> _Tp* ptr(const int* idx);
    template<typename _Tp> const _Tp* ptr(const int* idx) const;

    template<typename _Tp, int n> _Tp* ptr(const Vec<int, n>& idx);
    template<typename _Tp, int n> const _Tp* ptr(const Vec<int, n>& idx) const;

    //! the same as above, with the pointer dereferencing
    template<typename _Tp> _Tp& at(int i0=0);
    template<typename _Tp> const _Tp& at(int i0=0) const;

    template<typename _Tp> _Tp& at(int i0, int i1);
    template<typename _Tp> const _Tp& at(int i0, int i1) const;

    template<typename _Tp> _Tp& at(int i0, int i1, int i2);
    template<typename _Tp> const _Tp& at(int i0, int i1, int i2) const;

    template<typename _Tp> _Tp& at(const int* idx);
    template<typename _Tp> const _Tp& at(const int* idx) const;

    template<typename _Tp, int n> _Tp& at(const Vec<int, n>& idx);
    template<typename _Tp, int n> const _Tp& at(const Vec<int, n>& idx) const;

    //! special versions for 2D arrays (especially convenient for referencing image pixels)
    template<typename _Tp> _Tp& at(Point pt);
    template<typename _Tp> const _Tp& at(Point pt) const;

    //! template methods for iteration over matrix elements.
    // the iterators take care of skipping gaps in the end of rows (if any)
    template<typename _Tp> MatIterator_<_Tp> begin();
    template<typename _Tp> MatIterator_<_Tp> end();
    template<typename _Tp> MatConstIterator_<_Tp> begin() const;
    template<typename _Tp> MatConstIterator_<_Tp> end() const;

    enum { MAGIC_VAL  = 0x42FF0000, AUTO_STEP = 0, CONTINUOUS_FLAG = CV_MAT_CONT_FLAG, SUBMATRIX_FLAG = CV_SUBMAT_FLAG };
    enum { MAGIC_MASK = 0xFFFF0000, TYPE_MASK = 0x00000FFF, DEPTH_MASK = 7 };

    /*! includes several bit-fields:
         - the magic signature
         - continuity flag
         - depth
         - number of channels
     */
    int flags;
    //! the matrix dimensionality, >= 2
    int dims;
    //! the number of rows and columns or (-1, -1) when the matrix has more than 2 dimensions
    int rows, cols;
    //! pointer to the data
    uchar* data;

    //! helper fields used in locateROI and adjustROI
    uchar* datastart;
    uchar* dataend;
    uchar* datalimit;

    //! custom allocator
    MatAllocator* allocator;
    //! and the standard allocator
    static MatAllocator* getStdAllocator();

    //! interaction with UMat
    UMatData* u;

    MatSize size;
    MatStep step;

protected:
};


///////////////////////////////// Mat_<_Tp> ////////////////////////////////////

/*!
 Template matrix class derived from Mat

 The class Mat_ is a "thin" template wrapper on top of cv::Mat. It does not have any extra data fields,
 nor it or cv::Mat have any virtual methods and thus references or pointers to these two classes
 can be safely converted one to another. But do it with care, for example:

 \code
 // create 100x100 8-bit matrix
 Mat M(100,100,CV_8U);
 // this will compile fine. no any data conversion will be done.
 Mat_<float>& M1 = (Mat_<float>&)M;
 // the program will likely crash at the statement below
 M1(99,99) = 1.f;
 \endcode

 While cv::Mat is sufficient in most cases, cv::Mat_ can be more convenient if you use a lot of element
 access operations and if you know matrix type at compile time.
 Note that cv::Mat::at<_Tp>(int y, int x) and cv::Mat_<_Tp>::operator ()(int y, int x) do absolutely the
 same thing and run at the same speed, but the latter is certainly shorter:

 \code
 Mat_<double> M(20,20);
 for(int i = 0; i < M.rows; i++)
    for(int j = 0; j < M.cols; j++)
       M(i,j) = 1./(i+j+1);
 Mat E, V;
 eigen(M,E,V);
 cout << E.at<double>(0,0)/E.at<double>(M.rows-1,0);
 \endcode

 It is easy to use Mat_ for multi-channel images/matrices - just pass cv::Vec as cv::Mat_ template parameter:

 \code
 // allocate 320x240 color image and fill it with green (in RGB space)
 Mat_<Vec3b> img(240, 320, Vec3b(0,255,0));
 // now draw a diagonal white line
 for(int i = 0; i < 100; i++)
     img(i,i)=Vec3b(255,255,255);
 // and now modify the 2nd (red) channel of each pixel
 for(int i = 0; i < img.rows; i++)
    for(int j = 0; j < img.cols; j++)
       img(i,j)[2] ^= (uchar)(i ^ j); // img(y,x)[c] accesses c-th channel of the pixel (x,y)
 \endcode
*/
template<typename _Tp> class Mat_ : public Mat
{
public:
    typedef _Tp value_type;
    typedef typename DataType<_Tp>::channel_type channel_type;
    typedef MatIterator_<_Tp> iterator;
    typedef MatConstIterator_<_Tp> const_iterator;

    //! default constructor
    Mat_();
    //! equivalent to Mat(_rows, _cols, DataType<_Tp>::type)
    Mat_(int _rows, int _cols);
    //! constructor that sets each matrix element to specified value
    Mat_(int _rows, int _cols, const _Tp& value);
    //! equivalent to Mat(_size, DataType<_Tp>::type)
    explicit Mat_(Size _size);
    //! constructor that sets each matrix element to specified value
    Mat_(Size _size, const _Tp& value);
    //! n-dim array constructor
    Mat_(int _ndims, const int* _sizes);
    //! n-dim array constructor that sets each matrix element to specified value
    Mat_(int _ndims, const int* _sizes, const _Tp& value);
    //! copy/conversion contructor. If m is of different type, it's converted
    Mat_(const Mat& m);
    //! copy constructor
    Mat_(const Mat_& m);
    //! constructs a matrix on top of user-allocated data. step is in bytes(!!!), regardless of the type
    Mat_(int _rows, int _cols, _Tp* _data, size_t _step=AUTO_STEP);
    //! constructs n-dim matrix on top of user-allocated data. steps are in bytes(!!!), regardless of the type
    Mat_(int _ndims, const int* _sizes, _Tp* _data, const size_t* _steps=0);
    //! selects a submatrix
    Mat_(const Mat_& m, const Range& rowRange, const Range& colRange=Range::all());
    //! selects a submatrix
    Mat_(const Mat_& m, const Rect& roi);
    //! selects a submatrix, n-dim version
    Mat_(const Mat_& m, const Range* ranges);
    //! from a matrix expression
    explicit Mat_(const MatExpr& e);
    //! makes a matrix out of Vec, std::vector, Point_ or Point3_. The matrix will have a single column
    explicit Mat_(const std::vector<_Tp>& vec, bool copyData=false);
    template<int n> explicit Mat_(const Vec<typename DataType<_Tp>::channel_type, n>& vec, bool copyData=true);
    template<int m, int n> explicit Mat_(const Matx<typename DataType<_Tp>::channel_type, m, n>& mtx, bool copyData=true);
    explicit Mat_(const Point_<typename DataType<_Tp>::channel_type>& pt, bool copyData=true);
    explicit Mat_(const Point3_<typename DataType<_Tp>::channel_type>& pt, bool copyData=true);
    explicit Mat_(const MatCommaInitializer_<_Tp>& commaInitializer);

    Mat_& operator = (const Mat& m);
    Mat_& operator = (const Mat_& m);
    //! set all the elements to s.
    Mat_& operator = (const _Tp& s);
    //! assign a matrix expression
    Mat_& operator = (const MatExpr& e);

    //! iterators; they are smart enough to skip gaps in the end of rows
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;

    //! equivalent to Mat::create(_rows, _cols, DataType<_Tp>::type)
    void create(int _rows, int _cols);
    //! equivalent to Mat::create(_size, DataType<_Tp>::type)
    void create(Size _size);
    //! equivalent to Mat::create(_ndims, _sizes, DatType<_Tp>::type)
    void create(int _ndims, const int* _sizes);
    //! cross-product
    Mat_ cross(const Mat_& m) const;
    //! data type conversion
    template<typename T2> operator Mat_<T2>() const;
    //! overridden forms of Mat::row() etc.
    Mat_ row(int y) const;
    Mat_ col(int x) const;
    Mat_ diag(int d=0) const;
    Mat_ clone() const;

    //! overridden forms of Mat::elemSize() etc.
    size_t elemSize() const;
    size_t elemSize1() const;
    int type() const;
    int depth() const;
    int channels() const;
    size_t step1(int i=0) const;
    //! returns step()/sizeof(_Tp)
    size_t stepT(int i=0) const;

    //! overridden forms of Mat::zeros() etc. Data type is omitted, of course
    static MatExpr zeros(int rows, int cols);
    static MatExpr zeros(Size size);
    static MatExpr zeros(int _ndims, const int* _sizes);
    static MatExpr ones(int rows, int cols);
    static MatExpr ones(Size size);
    static MatExpr ones(int _ndims, const int* _sizes);
    static MatExpr eye(int rows, int cols);
    static MatExpr eye(Size size);

    //! some more overriden methods
    Mat_& adjustROI( int dtop, int dbottom, int dleft, int dright );
    Mat_ operator()( const Range& rowRange, const Range& colRange ) const;
    Mat_ operator()( const Rect& roi ) const;
    Mat_ operator()( const Range* ranges ) const;

    //! more convenient forms of row and element access operators
    _Tp* operator [](int y);
    const _Tp* operator [](int y) const;

    //! returns reference to the specified element
    _Tp& operator ()(const int* idx);
    //! returns read-only reference to the specified element
    const _Tp& operator ()(const int* idx) const;

    //! returns reference to the specified element
    template<int n> _Tp& operator ()(const Vec<int, n>& idx);
    //! returns read-only reference to the specified element
    template<int n> const _Tp& operator ()(const Vec<int, n>& idx) const;

    //! returns reference to the specified element (1D case)
    _Tp& operator ()(int idx0);
    //! returns read-only reference to the specified element (1D case)
    const _Tp& operator ()(int idx0) const;
    //! returns reference to the specified element (2D case)
    _Tp& operator ()(int idx0, int idx1);
    //! returns read-only reference to the specified element (2D case)
    const _Tp& operator ()(int idx0, int idx1) const;
    //! returns reference to the specified element (3D case)
    _Tp& operator ()(int idx0, int idx1, int idx2);
    //! returns read-only reference to the specified element (3D case)
    const _Tp& operator ()(int idx0, int idx1, int idx2) const;

    _Tp& operator ()(Point pt);
    const _Tp& operator ()(Point pt) const;

    //! conversion to vector.
    operator std::vector<_Tp>() const;
    //! conversion to Vec
    template<int n> operator Vec<typename DataType<_Tp>::channel_type, n>() const;
    //! conversion to Matx
    template<int m, int n> operator Matx<typename DataType<_Tp>::channel_type, m, n>() const;
};

typedef Mat_<uchar> Mat1b;
typedef Mat_<Vec2b> Mat2b;
typedef Mat_<Vec3b> Mat3b;
typedef Mat_<Vec4b> Mat4b;

typedef Mat_<short> Mat1s;
typedef Mat_<Vec2s> Mat2s;
typedef Mat_<Vec3s> Mat3s;
typedef Mat_<Vec4s> Mat4s;

typedef Mat_<ushort> Mat1w;
typedef Mat_<Vec2w> Mat2w;
typedef Mat_<Vec3w> Mat3w;
typedef Mat_<Vec4w> Mat4w;

typedef Mat_<int>   Mat1i;
typedef Mat_<Vec2i> Mat2i;
typedef Mat_<Vec3i> Mat3i;
typedef Mat_<Vec4i> Mat4i;

typedef Mat_<float> Mat1f;
typedef Mat_<Vec2f> Mat2f;
typedef Mat_<Vec3f> Mat3f;
typedef Mat_<Vec4f> Mat4f;

typedef Mat_<double> Mat1d;
typedef Mat_<Vec2d> Mat2d;
typedef Mat_<Vec3d> Mat3d;
typedef Mat_<Vec4d> Mat4d;

class CV_EXPORTS UMat
{
public:
    //! default constructor
    UMat(UMatUsageFlags usageFlags = USAGE_DEFAULT);
    //! constructs 2D matrix of the specified size and type
    // (_type is CV_8UC1, CV_64FC3, CV_32SC(12) etc.)
    UMat(int rows, int cols, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    UMat(Size size, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    //! constucts 2D matrix and fills it with the specified value _s.
    UMat(int rows, int cols, int type, const Scalar& s, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    UMat(Size size, int type, const Scalar& s, UMatUsageFlags usageFlags = USAGE_DEFAULT);

    //! constructs n-dimensional matrix
    UMat(int ndims, const int* sizes, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    UMat(int ndims, const int* sizes, int type, const Scalar& s, UMatUsageFlags usageFlags = USAGE_DEFAULT);

    //! copy constructor
    UMat(const UMat& m);

    //! creates a matrix header for a part of the bigger matrix
    UMat(const UMat& m, const Range& rowRange, const Range& colRange=Range::all());
    UMat(const UMat& m, const Rect& roi);
    UMat(const UMat& m, const Range* ranges);
    //! builds matrix from std::vector with or without copying the data
    template<typename _Tp> explicit UMat(const std::vector<_Tp>& vec, bool copyData=false);
    //! builds matrix from cv::Vec; the data is copied by default
    template<typename _Tp, int n> explicit UMat(const Vec<_Tp, n>& vec, bool copyData=true);
    //! builds matrix from cv::Matx; the data is copied by default
    template<typename _Tp, int m, int n> explicit UMat(const Matx<_Tp, m, n>& mtx, bool copyData=true);
    //! builds matrix from a 2D point
    template<typename _Tp> explicit UMat(const Point_<_Tp>& pt, bool copyData=true);
    //! builds matrix from a 3D point
    template<typename _Tp> explicit UMat(const Point3_<_Tp>& pt, bool copyData=true);
    //! builds matrix from comma initializer
    template<typename _Tp> explicit UMat(const MatCommaInitializer_<_Tp>& commaInitializer);

    //! destructor - calls release()
    ~UMat();
    //! assignment operators
    UMat& operator = (const UMat& m);

    Mat getMat(int flags) const;

    //! returns a new matrix header for the specified row
    UMat row(int y) const;
    //! returns a new matrix header for the specified column
    UMat col(int x) const;
    //! ... for the specified row span
    UMat rowRange(int startrow, int endrow) const;
    UMat rowRange(const Range& r) const;
    //! ... for the specified column span
    UMat colRange(int startcol, int endcol) const;
    UMat colRange(const Range& r) const;
    //! ... for the specified diagonal
    // (d=0 - the main diagonal,
    //  >0 - a diagonal from the lower half,
    //  <0 - a diagonal from the upper half)
    UMat diag(int d=0) const;
    //! constructs a square diagonal matrix which main diagonal is vector "d"
    static UMat diag(const UMat& d);

    //! returns deep copy of the matrix, i.e. the data is copied
    UMat clone() const;
    //! copies the matrix content to "m".
    // It calls m.create(this->size(), this->type()).
    void copyTo( OutputArray m ) const;
    //! copies those matrix elements to "m" that are marked with non-zero mask elements.
    void copyTo( OutputArray m, InputArray mask ) const;
    //! converts matrix to another datatype with optional scalng. See cvConvertScale.
    void convertTo( OutputArray m, int rtype, double alpha=1, double beta=0 ) const;

    void assignTo( UMat& m, int type=-1 ) const;

    //! sets every matrix element to s
    UMat& operator = (const Scalar& s);
    //! sets some of the matrix elements to s, according to the mask
    UMat& setTo(InputArray value, InputArray mask=noArray());
    //! creates alternative matrix header for the same data, with different
    // number of channels and/or different number of rows. see cvReshape.
    UMat reshape(int cn, int rows=0) const;
    UMat reshape(int cn, int newndims, const int* newsz) const;

    //! matrix transposition by means of matrix expressions
    UMat t() const;
    //! matrix inversion by means of matrix expressions
    UMat inv(int method=DECOMP_LU) const;
    //! per-element matrix multiplication by means of matrix expressions
    UMat mul(InputArray m, double scale=1) const;

    //! computes dot-product
    double dot(InputArray m) const;

    //! Matlab-style matrix initialization
    static UMat zeros(int rows, int cols, int type);
    static UMat zeros(Size size, int type);
    static UMat zeros(int ndims, const int* sz, int type);
    static UMat ones(int rows, int cols, int type);
    static UMat ones(Size size, int type);
    static UMat ones(int ndims, const int* sz, int type);
    static UMat eye(int rows, int cols, int type);
    static UMat eye(Size size, int type);

    //! allocates new matrix data unless the matrix already has specified size and type.
    // previous data is unreferenced if needed.
    void create(int rows, int cols, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    void create(Size size, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    void create(int ndims, const int* sizes, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);

    //! increases the reference counter; use with care to avoid memleaks
    void addref();
    //! decreases reference counter;
    // deallocates the data when reference counter reaches 0.
    void release();

    //! deallocates the matrix data
    void deallocate();
    //! internal use function; properly re-allocates _size, _step arrays
    void copySize(const UMat& m);

    //! locates matrix header within a parent matrix. See below
    void locateROI( Size& wholeSize, Point& ofs ) const;
    //! moves/resizes the current matrix ROI inside the parent matrix.
    UMat& adjustROI( int dtop, int dbottom, int dleft, int dright );
    //! extracts a rectangular sub-matrix
    // (this is a generalized form of row, rowRange etc.)
    UMat operator()( Range rowRange, Range colRange ) const;
    UMat operator()( const Rect& roi ) const;
    UMat operator()( const Range* ranges ) const;

    //! returns true iff the matrix data is continuous
    // (i.e. when there are no gaps between successive rows).
    // similar to CV_IS_MAT_CONT(cvmat->type)
    bool isContinuous() const;

    //! returns true if the matrix is a submatrix of another matrix
    bool isSubmatrix() const;

    //! returns element size in bytes,
    // similar to CV_ELEM_SIZE(cvmat->type)
    size_t elemSize() const;
    //! returns the size of element channel in bytes.
    size_t elemSize1() const;
    //! returns element type, similar to CV_MAT_TYPE(cvmat->type)
    int type() const;
    //! returns element type, similar to CV_MAT_DEPTH(cvmat->type)
    int depth() const;
    //! returns element type, similar to CV_MAT_CN(cvmat->type)
    int channels() const;
    //! returns step/elemSize1()
    size_t step1(int i=0) const;
    //! returns true if matrix data is NULL
    bool empty() const;
    //! returns the total number of matrix elements
    size_t total() const;

    //! returns N if the matrix is 1-channel (N x ptdim) or ptdim-channel (1 x N) or (N x 1); negative number otherwise
    int checkVector(int elemChannels, int depth=-1, bool requireContinuous=true) const;

    void* handle(int accessFlags) const;
    void ndoffset(size_t* ofs) const;

    enum { MAGIC_VAL  = 0x42FF0000, AUTO_STEP = 0, CONTINUOUS_FLAG = CV_MAT_CONT_FLAG, SUBMATRIX_FLAG = CV_SUBMAT_FLAG };
    enum { MAGIC_MASK = 0xFFFF0000, TYPE_MASK = 0x00000FFF, DEPTH_MASK = 7 };

    /*! includes several bit-fields:
         - the magic signature
         - continuity flag
         - depth
         - number of channels
     */
    int flags;
    //! the matrix dimensionality, >= 2
    int dims;
    //! the number of rows and columns or (-1, -1) when the matrix has more than 2 dimensions
    int rows, cols;

    //! custom allocator
    MatAllocator* allocator;
    UMatUsageFlags usageFlags; // usage flags for allocator
    //! and the standard allocator
    static MatAllocator* getStdAllocator();

    // black-box container of UMat data
    UMatData* u;

    // offset of the submatrix (or 0)
    size_t offset;

    MatSize size;
    MatStep step;

protected:
};


/////////////////////////// multi-dimensional sparse matrix //////////////////////////

/*!
 Sparse matrix class.

 The class represents multi-dimensional sparse numerical arrays. Such a sparse array can store elements
 of any type that cv::Mat is able to store. "Sparse" means that only non-zero elements
 are stored (though, as a result of some operations on a sparse matrix, some of its stored elements
 can actually become 0. It's user responsibility to detect such elements and delete them using cv::SparseMat::erase().
 The non-zero elements are stored in a hash table that grows when it's filled enough,
 so that the search time remains O(1) in average. Elements can be accessed using the following methods:

 <ol>
 <li>Query operations: cv::SparseMat::ptr() and the higher-level cv::SparseMat::ref(),
      cv::SparseMat::value() and cv::SparseMat::find, for example:
 \code
 const int dims = 5;
 int size[] = {10, 10, 10, 10, 10};
 SparseMat sparse_mat(dims, size, CV_32F);
 for(int i = 0; i < 1000; i++)
 {
     int idx[dims];
     for(int k = 0; k < dims; k++)
        idx[k] = rand()%sparse_mat.size(k);
     sparse_mat.ref<float>(idx) += 1.f;
 }
 \endcode

 <li>Sparse matrix iterators. Like cv::Mat iterators and unlike cv::Mat iterators, the sparse matrix iterators are STL-style,
 that is, the iteration is done as following:
 \code
 // prints elements of a sparse floating-point matrix and the sum of elements.
 SparseMatConstIterator_<float>
        it = sparse_mat.begin<float>(),
        it_end = sparse_mat.end<float>();
 double s = 0;
 int dims = sparse_mat.dims();
 for(; it != it_end; ++it)
 {
     // print element indices and the element value
     const Node* n = it.node();
     printf("(")
     for(int i = 0; i < dims; i++)
        printf("%3d%c", n->idx[i], i < dims-1 ? ',' : ')');
     printf(": %f\n", *it);
     s += *it;
 }
 printf("Element sum is %g\n", s);
 \endcode
 If you run this loop, you will notice that elements are enumerated
 in no any logical order (lexicographical etc.),
 they come in the same order as they stored in the hash table, i.e. semi-randomly.

 You may collect pointers to the nodes and sort them to get the proper ordering.
 Note, however, that pointers to the nodes may become invalid when you add more
 elements to the matrix; this is because of possible buffer reallocation.

 <li>A combination of the above 2 methods when you need to process 2 or more sparse
 matrices simultaneously, e.g. this is how you can compute unnormalized
 cross-correlation of the 2 floating-point sparse matrices:
 \code
 double crossCorr(const SparseMat& a, const SparseMat& b)
 {
     const SparseMat *_a = &a, *_b = &b;
     // if b contains less elements than a,
     // it's faster to iterate through b
     if(_a->nzcount() > _b->nzcount())
        std::swap(_a, _b);
     SparseMatConstIterator_<float> it = _a->begin<float>(),
                                    it_end = _a->end<float>();
     double ccorr = 0;
     for(; it != it_end; ++it)
     {
         // take the next element from the first matrix
         float avalue = *it;
         const Node* anode = it.node();
         // and try to find element with the same index in the second matrix.
         // since the hash value depends only on the element index,
         // we reuse hashvalue stored in the node
         float bvalue = _b->value<float>(anode->idx,&anode->hashval);
         ccorr += avalue*bvalue;
     }
     return ccorr;
 }
 \endcode
 </ol>
*/
class CV_EXPORTS SparseMat
{
public:
    typedef SparseMatIterator iterator;
    typedef SparseMatConstIterator const_iterator;

    enum { MAGIC_VAL=0x42FD0000, MAX_DIM=32, HASH_SCALE=0x5bd1e995, HASH_BIT=0x80000000 };

    //! the sparse matrix header
    struct CV_EXPORTS Hdr
    {
        Hdr(int _dims, const int* _sizes, int _type);
        void clear();
        int refcount;
        int dims;
        int valueOffset;
        size_t nodeSize;
        size_t nodeCount;
        size_t freeList;
        std::vector<uchar> pool;
        std::vector<size_t> hashtab;
        int size[MAX_DIM];
    };

    //! sparse matrix node - element of a hash table
    struct CV_EXPORTS Node
    {
        //! hash value
        size_t hashval;
        //! index of the next node in the same hash table entry
        size_t next;
        //! index of the matrix element
        int idx[MAX_DIM];
    };

    //! default constructor
    SparseMat();
    //! creates matrix of the specified size and type
    SparseMat(int dims, const int* _sizes, int _type);
    //! copy constructor
    SparseMat(const SparseMat& m);
    //! converts dense 2d matrix to the sparse form
    /*!
     \param m the input matrix
    */
    explicit SparseMat(const Mat& m);
    //! converts old-style sparse matrix to the new-style. All the data is copied
    //SparseMat(const CvSparseMat* m);
    //! the destructor
    ~SparseMat();

    //! assignment operator. This is O(1) operation, i.e. no data is copied
    SparseMat& operator = (const SparseMat& m);
    //! equivalent to the corresponding constructor
    SparseMat& operator = (const Mat& m);

    //! creates full copy of the matrix
    SparseMat clone() const;

    //! copies all the data to the destination matrix. All the previous content of m is erased
    void copyTo( SparseMat& m ) const;
    //! converts sparse matrix to dense matrix.
    void copyTo( Mat& m ) const;
    //! multiplies all the matrix elements by the specified scale factor alpha and converts the results to the specified data type
    void convertTo( SparseMat& m, int rtype, double alpha=1 ) const;
    //! converts sparse matrix to dense n-dim matrix with optional type conversion and scaling.
    /*!
      \param rtype The output matrix data type. When it is =-1, the output array will have the same data type as (*this)
      \param alpha The scale factor
      \param beta The optional delta added to the scaled values before the conversion
    */
    void convertTo( Mat& m, int rtype, double alpha=1, double beta=0 ) const;

    // not used now
    void assignTo( SparseMat& m, int type=-1 ) const;

    //! reallocates sparse matrix.
    /*!
        If the matrix already had the proper size and type,
        it is simply cleared with clear(), otherwise,
        the old matrix is released (using release()) and the new one is allocated.
    */
    void create(int dims, const int* _sizes, int _type);
    //! sets all the sparse matrix elements to 0, which means clearing the hash table.
    void clear();
    //! manually increments the reference counter to the header.
    void addref();
    // decrements the header reference counter. When the counter reaches 0, the header and all the underlying data are deallocated.
    void release();

    //! converts sparse matrix to the old-style representation; all the elements are copied.
    //operator CvSparseMat*() const;
    //! returns the size of each element in bytes (not including the overhead - the space occupied by SparseMat::Node elements)
    size_t elemSize() const;
    //! returns elemSize()/channels()
    size_t elemSize1() const;

    //! returns type of sparse matrix elements
    int type() const;
    //! returns the depth of sparse matrix elements
    int depth() const;
    //! returns the number of channels
    int channels() const;

    //! returns the array of sizes, or NULL if the matrix is not allocated
    const int* size() const;
    //! returns the size of i-th matrix dimension (or 0)
    int size(int i) const;
    //! returns the matrix dimensionality
    int dims() const;
    //! returns the number of non-zero elements (=the number of hash table nodes)
    size_t nzcount() const;

    //! computes the element hash value (1D case)
    size_t hash(int i0) const;
    //! computes the element hash value (2D case)
    size_t hash(int i0, int i1) const;
    //! computes the element hash value (3D case)
    size_t hash(int i0, int i1, int i2) const;
    //! computes the element hash value (nD case)
    size_t hash(const int* idx) const;

    //@{
    /*!
     specialized variants for 1D, 2D, 3D cases and the generic_type one for n-D case.

     return pointer to the matrix element.
     <ul>
      <li>if the element is there (it's non-zero), the pointer to it is returned
      <li>if it's not there and createMissing=false, NULL pointer is returned
      <li>if it's not there and createMissing=true, then the new element
        is created and initialized with 0. Pointer to it is returned
      <li>if the optional hashval pointer is not NULL, the element hash value is
      not computed, but *hashval is taken instead.
     </ul>
    */
    //! returns pointer to the specified element (1D case)
    uchar* ptr(int i0, bool createMissing, size_t* hashval=0);
    //! returns pointer to the specified element (2D case)
    uchar* ptr(int i0, int i1, bool createMissing, size_t* hashval=0);
    //! returns pointer to the specified element (3D case)
    uchar* ptr(int i0, int i1, int i2, bool createMissing, size_t* hashval=0);
    //! returns pointer to the specified element (nD case)
    uchar* ptr(const int* idx, bool createMissing, size_t* hashval=0);
    //@}

    //@{
    /*!
     return read-write reference to the specified sparse matrix element.

     ref<_Tp>(i0,...[,hashval]) is equivalent to *(_Tp*)ptr(i0,...,true[,hashval]).
     The methods always return a valid reference.
     If the element did not exist, it is created and initialiazed with 0.
    */
    //! returns reference to the specified element (1D case)
    template<typename _Tp> _Tp& ref(int i0, size_t* hashval=0);
    //! returns reference to the specified element (2D case)
    template<typename _Tp> _Tp& ref(int i0, int i1, size_t* hashval=0);
    //! returns reference to the specified element (3D case)
    template<typename _Tp> _Tp& ref(int i0, int i1, int i2, size_t* hashval=0);
    //! returns reference to the specified element (nD case)
    template<typename _Tp> _Tp& ref(const int* idx, size_t* hashval=0);
    //@}

    //@{
    /*!
     return value of the specified sparse matrix element.

     value<_Tp>(i0,...[,hashval]) is equivalent

     \code
     { const _Tp* p = find<_Tp>(i0,...[,hashval]); return p ? *p : _Tp(); }
     \endcode

     That is, if the element did not exist, the methods return 0.
     */
    //! returns value of the specified element (1D case)
    template<typename _Tp> _Tp value(int i0, size_t* hashval=0) const;
    //! returns value of the specified element (2D case)
    template<typename _Tp> _Tp value(int i0, int i1, size_t* hashval=0) const;
    //! returns value of the specified element (3D case)
    template<typename _Tp> _Tp value(int i0, int i1, int i2, size_t* hashval=0) const;
    //! returns value of the specified element (nD case)
    template<typename _Tp> _Tp value(const int* idx, size_t* hashval=0) const;
    //@}

    //@{
    /*!
     Return pointer to the specified sparse matrix element if it exists

     find<_Tp>(i0,...[,hashval]) is equivalent to (_const Tp*)ptr(i0,...false[,hashval]).

     If the specified element does not exist, the methods return NULL.
    */
    //! returns pointer to the specified element (1D case)
    template<typename _Tp> const _Tp* find(int i0, size_t* hashval=0) const;
    //! returns pointer to the specified element (2D case)
    template<typename _Tp> const _Tp* find(int i0, int i1, size_t* hashval=0) const;
    //! returns pointer to the specified element (3D case)
    template<typename _Tp> const _Tp* find(int i0, int i1, int i2, size_t* hashval=0) const;
    //! returns pointer to the specified element (nD case)
    template<typename _Tp> const _Tp* find(const int* idx, size_t* hashval=0) const;

    //! erases the specified element (2D case)
    void erase(int i0, int i1, size_t* hashval=0);
    //! erases the specified element (3D case)
    void erase(int i0, int i1, int i2, size_t* hashval=0);
    //! erases the specified element (nD case)
    void erase(const int* idx, size_t* hashval=0);

    //@{
    /*!
       return the sparse matrix iterator pointing to the first sparse matrix element
    */
    //! returns the sparse matrix iterator at the matrix beginning
    SparseMatIterator begin();
    //! returns the sparse matrix iterator at the matrix beginning
    template<typename _Tp> SparseMatIterator_<_Tp> begin();
    //! returns the read-only sparse matrix iterator at the matrix beginning
    SparseMatConstIterator begin() const;
    //! returns the read-only sparse matrix iterator at the matrix beginning
    template<typename _Tp> SparseMatConstIterator_<_Tp> begin() const;
    //@}
    /*!
       return the sparse matrix iterator pointing to the element following the last sparse matrix element
    */
    //! returns the sparse matrix iterator at the matrix end
    SparseMatIterator end();
    //! returns the read-only sparse matrix iterator at the matrix end
    SparseMatConstIterator end() const;
    //! returns the typed sparse matrix iterator at the matrix end
    template<typename _Tp> SparseMatIterator_<_Tp> end();
    //! returns the typed read-only sparse matrix iterator at the matrix end
    template<typename _Tp> SparseMatConstIterator_<_Tp> end() const;

    //! returns the value stored in the sparse martix node
    template<typename _Tp> _Tp& value(Node* n);
    //! returns the value stored in the sparse martix node
    template<typename _Tp> const _Tp& value(const Node* n) const;

    ////////////// some internal-use methods ///////////////
    Node* node(size_t nidx);
    const Node* node(size_t nidx) const;

    uchar* newNode(const int* idx, size_t hashval);
    void removeNode(size_t hidx, size_t nidx, size_t previdx);
    void resizeHashTab(size_t newsize);

    int flags;
    Hdr* hdr;
};



///////////////////////////////// SparseMat_<_Tp> ////////////////////////////////////

/*!
 The Template Sparse Matrix class derived from cv::SparseMat

 The class provides slightly more convenient operations for accessing elements.

 \code
 SparseMat m;
 ...
 SparseMat_<int> m_ = (SparseMat_<int>&)m;
 m_.ref(1)++; // equivalent to m.ref<int>(1)++;
 m_.ref(2) += m_(3); // equivalent to m.ref<int>(2) += m.value<int>(3);
 \endcode
*/
template<typename _Tp> class SparseMat_ : public SparseMat
{
public:
    typedef SparseMatIterator_<_Tp> iterator;
    typedef SparseMatConstIterator_<_Tp> const_iterator;

    //! the default constructor
    SparseMat_();
    //! the full constructor equivelent to SparseMat(dims, _sizes, DataType<_Tp>::type)
    SparseMat_(int dims, const int* _sizes);
    //! the copy constructor. If DataType<_Tp>.type != m.type(), the m elements are converted
    SparseMat_(const SparseMat& m);
    //! the copy constructor. This is O(1) operation - no data is copied
    SparseMat_(const SparseMat_& m);
    //! converts dense matrix to the sparse form
    SparseMat_(const Mat& m);
    //! converts the old-style sparse matrix to the C++ class. All the elements are copied
    //SparseMat_(const CvSparseMat* m);
    //! the assignment operator. If DataType<_Tp>.type != m.type(), the m elements are converted
    SparseMat_& operator = (const SparseMat& m);
    //! the assignment operator. This is O(1) operation - no data is copied
    SparseMat_& operator = (const SparseMat_& m);
    //! converts dense matrix to the sparse form
    SparseMat_& operator = (const Mat& m);

    //! makes full copy of the matrix. All the elements are duplicated
    SparseMat_ clone() const;
    //! equivalent to cv::SparseMat::create(dims, _sizes, DataType<_Tp>::type)
    void create(int dims, const int* _sizes);
    //! converts sparse matrix to the old-style CvSparseMat. All the elements are copied
    //operator CvSparseMat*() const;

    //! returns type of the matrix elements
    int type() const;
    //! returns depth of the matrix elements
    int depth() const;
    //! returns the number of channels in each matrix element
    int channels() const;

    //! equivalent to SparseMat::ref<_Tp>(i0, hashval)
    _Tp& ref(int i0, size_t* hashval=0);
    //! equivalent to SparseMat::ref<_Tp>(i0, i1, hashval)
    _Tp& ref(int i0, int i1, size_t* hashval=0);
    //! equivalent to SparseMat::ref<_Tp>(i0, i1, i2, hashval)
    _Tp& ref(int i0, int i1, int i2, size_t* hashval=0);
    //! equivalent to SparseMat::ref<_Tp>(idx, hashval)
    _Tp& ref(const int* idx, size_t* hashval=0);

    //! equivalent to SparseMat::value<_Tp>(i0, hashval)
    _Tp operator()(int i0, size_t* hashval=0) const;
    //! equivalent to SparseMat::value<_Tp>(i0, i1, hashval)
    _Tp operator()(int i0, int i1, size_t* hashval=0) const;
    //! equivalent to SparseMat::value<_Tp>(i0, i1, i2, hashval)
    _Tp operator()(int i0, int i1, int i2, size_t* hashval=0) const;
    //! equivalent to SparseMat::value<_Tp>(idx, hashval)
    _Tp operator()(const int* idx, size_t* hashval=0) const;

    //! returns sparse matrix iterator pointing to the first sparse matrix element
    SparseMatIterator_<_Tp> begin();
    //! returns read-only sparse matrix iterator pointing to the first sparse matrix element
    SparseMatConstIterator_<_Tp> begin() const;
    //! returns sparse matrix iterator pointing to the element following the last sparse matrix element
    SparseMatIterator_<_Tp> end();
    //! returns read-only sparse matrix iterator pointing to the element following the last sparse matrix element
    SparseMatConstIterator_<_Tp> end() const;
};



////////////////////////////////// MatConstIterator //////////////////////////////////

class CV_EXPORTS MatConstIterator
{
public:
    typedef uchar* value_type;
    typedef ptrdiff_t difference_type;
    typedef const uchar** pointer;
    typedef uchar* reference;

#ifndef OPENCV_NOSTL
    typedef std::random_access_iterator_tag iterator_category;
#endif

    //! default constructor
    MatConstIterator();
    //! constructor that sets the iterator to the beginning of the matrix
    MatConstIterator(const Mat* _m);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator(const Mat* _m, int _row, int _col=0);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator(const Mat* _m, Point _pt);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator(const Mat* _m, const int* _idx);
    //! copy constructor
    MatConstIterator(const MatConstIterator& it);

    //! copy operator
    MatConstIterator& operator = (const MatConstIterator& it);
    //! returns the current matrix element
    uchar* operator *() const;
    //! returns the i-th matrix element, relative to the current
    uchar* operator [](ptrdiff_t i) const;

    //! shifts the iterator forward by the specified number of elements
    MatConstIterator& operator += (ptrdiff_t ofs);
    //! shifts the iterator backward by the specified number of elements
    MatConstIterator& operator -= (ptrdiff_t ofs);
    //! decrements the iterator
    MatConstIterator& operator --();
    //! decrements the iterator
    MatConstIterator operator --(int);
    //! increments the iterator
    MatConstIterator& operator ++();
    //! increments the iterator
    MatConstIterator operator ++(int);
    //! returns the current iterator position
    Point pos() const;
    //! returns the current iterator position
    void pos(int* _idx) const;

    ptrdiff_t lpos() const;
    void seek(ptrdiff_t ofs, bool relative = false);
    void seek(const int* _idx, bool relative = false);

    const Mat* m;
    size_t elemSize;
    uchar* ptr;
    uchar* sliceStart;
    uchar* sliceEnd;
};



////////////////////////////////// MatConstIterator_ /////////////////////////////////

/*!
 Matrix read-only iterator
 */
template<typename _Tp>
class MatConstIterator_ : public MatConstIterator
{
public:
    typedef _Tp value_type;
    typedef ptrdiff_t difference_type;
    typedef const _Tp* pointer;
    typedef const _Tp& reference;

#ifndef OPENCV_NOSTL
    typedef std::random_access_iterator_tag iterator_category;
#endif

    //! default constructor
    MatConstIterator_();
    //! constructor that sets the iterator to the beginning of the matrix
    MatConstIterator_(const Mat_<_Tp>* _m);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator_(const Mat_<_Tp>* _m, int _row, int _col=0);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator_(const Mat_<_Tp>* _m, Point _pt);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator_(const Mat_<_Tp>* _m, const int* _idx);
    //! copy constructor
    MatConstIterator_(const MatConstIterator_& it);

    //! copy operator
    MatConstIterator_& operator = (const MatConstIterator_& it);
    //! returns the current matrix element
    _Tp operator *() const;
    //! returns the i-th matrix element, relative to the current
    _Tp operator [](ptrdiff_t i) const;

    //! shifts the iterator forward by the specified number of elements
    MatConstIterator_& operator += (ptrdiff_t ofs);
    //! shifts the iterator backward by the specified number of elements
    MatConstIterator_& operator -= (ptrdiff_t ofs);
    //! decrements the iterator
    MatConstIterator_& operator --();
    //! decrements the iterator
    MatConstIterator_ operator --(int);
    //! increments the iterator
    MatConstIterator_& operator ++();
    //! increments the iterator
    MatConstIterator_ operator ++(int);
    //! returns the current iterator position
    Point pos() const;
};



//////////////////////////////////// MatIterator_ ////////////////////////////////////

/*!
 Matrix read-write iterator
*/
template<typename _Tp>
class MatIterator_ : public MatConstIterator_<_Tp>
{
public:
    typedef _Tp* pointer;
    typedef _Tp& reference;

#ifndef OPENCV_NOSTL
    typedef std::random_access_iterator_tag iterator_category;
#endif

    //! the default constructor
    MatIterator_();
    //! constructor that sets the iterator to the beginning of the matrix
    MatIterator_(Mat_<_Tp>* _m);
    //! constructor that sets the iterator to the specified element of the matrix
    MatIterator_(Mat_<_Tp>* _m, int _row, int _col=0);
    //! constructor that sets the iterator to the specified element of the matrix
    MatIterator_(const Mat_<_Tp>* _m, Point _pt);
    //! constructor that sets the iterator to the specified element of the matrix
    MatIterator_(const Mat_<_Tp>* _m, const int* _idx);
    //! copy constructor
    MatIterator_(const MatIterator_& it);
    //! copy operator
    MatIterator_& operator = (const MatIterator_<_Tp>& it );

    //! returns the current matrix element
    _Tp& operator *() const;
    //! returns the i-th matrix element, relative to the current
    _Tp& operator [](ptrdiff_t i) const;

    //! shifts the iterator forward by the specified number of elements
    MatIterator_& operator += (ptrdiff_t ofs);
    //! shifts the iterator backward by the specified number of elements
    MatIterator_& operator -= (ptrdiff_t ofs);
    //! decrements the iterator
    MatIterator_& operator --();
    //! decrements the iterator
    MatIterator_ operator --(int);
    //! increments the iterator
    MatIterator_& operator ++();
    //! increments the iterator
    MatIterator_ operator ++(int);
};



/////////////////////////////// SparseMatConstIterator ///////////////////////////////

/*!
 Read-Only Sparse Matrix Iterator.
 Here is how to use the iterator to compute the sum of floating-point sparse matrix elements:

 \code
 SparseMatConstIterator it = m.begin(), it_end = m.end();
 double s = 0;
 CV_Assert( m.type() == CV_32F );
 for( ; it != it_end; ++it )
    s += it.value<float>();
 \endcode
*/
class CV_EXPORTS SparseMatConstIterator
{
public:
    //! the default constructor
    SparseMatConstIterator();
    //! the full constructor setting the iterator to the first sparse matrix element
    SparseMatConstIterator(const SparseMat* _m);
    //! the copy constructor
    SparseMatConstIterator(const SparseMatConstIterator& it);

    //! the assignment operator
    SparseMatConstIterator& operator = (const SparseMatConstIterator& it);

    //! template method returning the current matrix element
    template<typename _Tp> const _Tp& value() const;
    //! returns the current node of the sparse matrix. it.node->idx is the current element index
    const SparseMat::Node* node() const;

    //! moves iterator to the previous element
    SparseMatConstIterator& operator --();
    //! moves iterator to the previous element
    SparseMatConstIterator operator --(int);
    //! moves iterator to the next element
    SparseMatConstIterator& operator ++();
    //! moves iterator to the next element
    SparseMatConstIterator operator ++(int);

    //! moves iterator to the element after the last element
    void seekEnd();

    const SparseMat* m;
    size_t hashidx;
    uchar* ptr;
};



////////////////////////////////// SparseMatIterator /////////////////////////////////

/*!
 Read-write Sparse Matrix Iterator

 The class is similar to cv::SparseMatConstIterator,
 but can be used for in-place modification of the matrix elements.
*/
class CV_EXPORTS SparseMatIterator : public SparseMatConstIterator
{
public:
    //! the default constructor
    SparseMatIterator();
    //! the full constructor setting the iterator to the first sparse matrix element
    SparseMatIterator(SparseMat* _m);
    //! the full constructor setting the iterator to the specified sparse matrix element
    SparseMatIterator(SparseMat* _m, const int* idx);
    //! the copy constructor
    SparseMatIterator(const SparseMatIterator& it);

    //! the assignment operator
    SparseMatIterator& operator = (const SparseMatIterator& it);
    //! returns read-write reference to the current sparse matrix element
    template<typename _Tp> _Tp& value() const;
    //! returns pointer to the current sparse matrix node. it.node->idx is the index of the current element (do not modify it!)
    SparseMat::Node* node() const;

    //! moves iterator to the next element
    SparseMatIterator& operator ++();
    //! moves iterator to the next element
    SparseMatIterator operator ++(int);
};



/////////////////////////////// SparseMatConstIterator_ //////////////////////////////

/*!
 Template Read-Only Sparse Matrix Iterator Class.

 This is the derived from SparseMatConstIterator class that
 introduces more convenient operator *() for accessing the current element.
*/
template<typename _Tp> class SparseMatConstIterator_ : public SparseMatConstIterator
{
public:

#ifndef OPENCV_NOSTL
    typedef std::forward_iterator_tag iterator_category;
#endif

    //! the default constructor
    SparseMatConstIterator_();
    //! the full constructor setting the iterator to the first sparse matrix element
    SparseMatConstIterator_(const SparseMat_<_Tp>* _m);
    SparseMatConstIterator_(const SparseMat* _m);
    //! the copy constructor
    SparseMatConstIterator_(const SparseMatConstIterator_& it);

    //! the assignment operator
    SparseMatConstIterator_& operator = (const SparseMatConstIterator_& it);
    //! the element access operator
    const _Tp& operator *() const;

    //! moves iterator to the next element
    SparseMatConstIterator_& operator ++();
    //! moves iterator to the next element
    SparseMatConstIterator_ operator ++(int);
};



///////////////////////////////// SparseMatIterator_ /////////////////////////////////

/*!
 Template Read-Write Sparse Matrix Iterator Class.

 This is the derived from cv::SparseMatConstIterator_ class that
 introduces more convenient operator *() for accessing the current element.
*/
template<typename _Tp> class SparseMatIterator_ : public SparseMatConstIterator_<_Tp>
{
public:

#ifndef OPENCV_NOSTL
    typedef std::forward_iterator_tag iterator_category;
#endif

    //! the default constructor
    SparseMatIterator_();
    //! the full constructor setting the iterator to the first sparse matrix element
    SparseMatIterator_(SparseMat_<_Tp>* _m);
    SparseMatIterator_(SparseMat* _m);
    //! the copy constructor
    SparseMatIterator_(const SparseMatIterator_& it);

    //! the assignment operator
    SparseMatIterator_& operator = (const SparseMatIterator_& it);
    //! returns the reference to the current element
    _Tp& operator *() const;

    //! moves the iterator to the next element
    SparseMatIterator_& operator ++();
    //! moves the iterator to the next element
    SparseMatIterator_ operator ++(int);
};



/////////////////////////////////// NAryMatIterator //////////////////////////////////

/*!
 n-Dimensional Dense Matrix Iterator Class.

 The class cv::NAryMatIterator is used for iterating over one or more n-dimensional dense arrays (cv::Mat's).

 The iterator is completely different from cv::Mat_ and cv::SparseMat_ iterators.
 It iterates through the slices (or planes), not the elements, where "slice" is a continuous part of the arrays.

 Here is the example on how the iterator can be used to normalize 3D histogram:

 \code
 void normalizeColorHist(Mat& hist)
 {
 #if 1
     // intialize iterator (the style is different from STL).
     // after initialization the iterator will contain
     // the number of slices or planes
     // the iterator will go through
     Mat* arrays[] = { &hist, 0 };
     Mat planes[1];
     NAryMatIterator it(arrays, planes);
     double s = 0;
     // iterate through the matrix. on each iteration
     // it.planes[i] (of type Mat) will be set to the current plane of
     // i-th n-dim matrix passed to the iterator constructor.
     for(int p = 0; p < it.nplanes; p++, ++it)
        s += sum(it.planes[0])[0];
     it = NAryMatIterator(hist);
     s = 1./s;
     for(int p = 0; p < it.nplanes; p++, ++it)
        it.planes[0] *= s;
 #elif 1
     // this is a shorter implementation of the above
     // using built-in operations on Mat
     double s = sum(hist)[0];
     hist.convertTo(hist, hist.type(), 1./s, 0);
 #else
     // and this is even shorter one
     // (assuming that the histogram elements are non-negative)
     normalize(hist, hist, 1, 0, NORM_L1);
 #endif
 }
 \endcode

 You can iterate through several matrices simultaneously as long as they have the same geometry
 (dimensionality and all the dimension sizes are the same), which is useful for binary
 and n-ary operations on such matrices. Just pass those matrices to cv::MatNDIterator.
 Then, during the iteration it.planes[0], it.planes[1], ... will
 be the slices of the corresponding matrices
*/
class CV_EXPORTS NAryMatIterator
{
public:
    //! the default constructor
    NAryMatIterator();
    //! the full constructor taking arbitrary number of n-dim matrices
    NAryMatIterator(const Mat** arrays, uchar** ptrs, int narrays=-1);
    //! the full constructor taking arbitrary number of n-dim matrices
    NAryMatIterator(const Mat** arrays, Mat* planes, int narrays=-1);
    //! the separate iterator initialization method
    void init(const Mat** arrays, Mat* planes, uchar** ptrs, int narrays=-1);

    //! proceeds to the next plane of every iterated matrix
    NAryMatIterator& operator ++();
    //! proceeds to the next plane of every iterated matrix (postfix increment operator)
    NAryMatIterator operator ++(int);

    //! the iterated arrays
    const Mat** arrays;
    //! the current planes
    Mat* planes;
    //! data pointers
    uchar** ptrs;
    //! the number of arrays
    int narrays;
    //! the number of hyper-planes that the iterator steps through
    size_t nplanes;
    //! the size of each segment (in elements)
    size_t size;
protected:
    int iterdepth;
    size_t idx;
};



///////////////////////////////// Matrix Expressions /////////////////////////////////

class CV_EXPORTS MatOp
{
public:
    MatOp();
    virtual ~MatOp();

    virtual bool elementWise(const MatExpr& expr) const;
    virtual void assign(const MatExpr& expr, Mat& m, int type=-1) const = 0;
    virtual void roi(const MatExpr& expr, const Range& rowRange,
                     const Range& colRange, MatExpr& res) const;
    virtual void diag(const MatExpr& expr, int d, MatExpr& res) const;
    virtual void augAssignAdd(const MatExpr& expr, Mat& m) const;
    virtual void augAssignSubtract(const MatExpr& expr, Mat& m) const;
    virtual void augAssignMultiply(const MatExpr& expr, Mat& m) const;
    virtual void augAssignDivide(const MatExpr& expr, Mat& m) const;
    virtual void augAssignAnd(const MatExpr& expr, Mat& m) const;
    virtual void augAssignOr(const MatExpr& expr, Mat& m) const;
    virtual void augAssignXor(const MatExpr& expr, Mat& m) const;

    virtual void add(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res) const;
    virtual void add(const MatExpr& expr1, const Scalar& s, MatExpr& res) const;

    virtual void subtract(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res) const;
    virtual void subtract(const Scalar& s, const MatExpr& expr, MatExpr& res) const;

    virtual void multiply(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res, double scale=1) const;
    virtual void multiply(const MatExpr& expr1, double s, MatExpr& res) const;

    virtual void divide(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res, double scale=1) const;
    virtual void divide(double s, const MatExpr& expr, MatExpr& res) const;

    virtual void abs(const MatExpr& expr, MatExpr& res) const;

    virtual void transpose(const MatExpr& expr, MatExpr& res) const;
    virtual void matmul(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res) const;
    virtual void invert(const MatExpr& expr, int method, MatExpr& res) const;

    virtual Size size(const MatExpr& expr) const;
    virtual int type(const MatExpr& expr) const;
};


class CV_EXPORTS MatExpr
{
public:
    MatExpr();
    explicit MatExpr(const Mat& m);

    MatExpr(const MatOp* _op, int _flags, const Mat& _a = Mat(), const Mat& _b = Mat(),
            const Mat& _c = Mat(), double _alpha = 1, double _beta = 1, const Scalar& _s = Scalar());

    operator Mat() const;
    template<typename _Tp> operator Mat_<_Tp>() const;

    Size size() const;
    int type() const;

    MatExpr row(int y) const;
    MatExpr col(int x) const;
    MatExpr diag(int d = 0) const;
    MatExpr operator()( const Range& rowRange, const Range& colRange ) const;
    MatExpr operator()( const Rect& roi ) const;

    MatExpr t() const;
    MatExpr inv(int method = DECOMP_LU) const;
    MatExpr mul(const MatExpr& e, double scale=1) const;
    MatExpr mul(const Mat& m, double scale=1) const;

    Mat cross(const Mat& m) const;
    double dot(const Mat& m) const;

    const MatOp* op;
    int flags;

    Mat a, b, c;
    double alpha, beta;
    Scalar s;
};


CV_EXPORTS MatExpr operator + (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator + (const Mat& a, const Scalar& s);
CV_EXPORTS MatExpr operator + (const Scalar& s, const Mat& a);
CV_EXPORTS MatExpr operator + (const MatExpr& e, const Mat& m);
CV_EXPORTS MatExpr operator + (const Mat& m, const MatExpr& e);
CV_EXPORTS MatExpr operator + (const MatExpr& e, const Scalar& s);
CV_EXPORTS MatExpr operator + (const Scalar& s, const MatExpr& e);
CV_EXPORTS MatExpr operator + (const MatExpr& e1, const MatExpr& e2);

CV_EXPORTS MatExpr operator - (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator - (const Mat& a, const Scalar& s);
CV_EXPORTS MatExpr operator - (const Scalar& s, const Mat& a);
CV_EXPORTS MatExpr operator - (const MatExpr& e, const Mat& m);
CV_EXPORTS MatExpr operator - (const Mat& m, const MatExpr& e);
CV_EXPORTS MatExpr operator - (const MatExpr& e, const Scalar& s);
CV_EXPORTS MatExpr operator - (const Scalar& s, const MatExpr& e);
CV_EXPORTS MatExpr operator - (const MatExpr& e1, const MatExpr& e2);

CV_EXPORTS MatExpr operator - (const Mat& m);
CV_EXPORTS MatExpr operator - (const MatExpr& e);

CV_EXPORTS MatExpr operator * (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator * (const Mat& a, double s);
CV_EXPORTS MatExpr operator * (double s, const Mat& a);
CV_EXPORTS MatExpr operator * (const MatExpr& e, const Mat& m);
CV_EXPORTS MatExpr operator * (const Mat& m, const MatExpr& e);
CV_EXPORTS MatExpr operator * (const MatExpr& e, double s);
CV_EXPORTS MatExpr operator * (double s, const MatExpr& e);
CV_EXPORTS MatExpr operator * (const MatExpr& e1, const MatExpr& e2);

CV_EXPORTS MatExpr operator / (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator / (const Mat& a, double s);
CV_EXPORTS MatExpr operator / (double s, const Mat& a);
CV_EXPORTS MatExpr operator / (const MatExpr& e, const Mat& m);
CV_EXPORTS MatExpr operator / (const Mat& m, const MatExpr& e);
CV_EXPORTS MatExpr operator / (const MatExpr& e, double s);
CV_EXPORTS MatExpr operator / (double s, const MatExpr& e);
CV_EXPORTS MatExpr operator / (const MatExpr& e1, const MatExpr& e2);

CV_EXPORTS MatExpr operator < (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator < (const Mat& a, double s);
CV_EXPORTS MatExpr operator < (double s, const Mat& a);

CV_EXPORTS MatExpr operator <= (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator <= (const Mat& a, double s);
CV_EXPORTS MatExpr operator <= (double s, const Mat& a);

CV_EXPORTS MatExpr operator == (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator == (const Mat& a, double s);
CV_EXPORTS MatExpr operator == (double s, const Mat& a);

CV_EXPORTS MatExpr operator != (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator != (const Mat& a, double s);
CV_EXPORTS MatExpr operator != (double s, const Mat& a);

CV_EXPORTS MatExpr operator >= (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator >= (const Mat& a, double s);
CV_EXPORTS MatExpr operator >= (double s, const Mat& a);

CV_EXPORTS MatExpr operator > (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator > (const Mat& a, double s);
CV_EXPORTS MatExpr operator > (double s, const Mat& a);

CV_EXPORTS MatExpr operator & (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator & (const Mat& a, const Scalar& s);
CV_EXPORTS MatExpr operator & (const Scalar& s, const Mat& a);

CV_EXPORTS MatExpr operator | (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator | (const Mat& a, const Scalar& s);
CV_EXPORTS MatExpr operator | (const Scalar& s, const Mat& a);

CV_EXPORTS MatExpr operator ^ (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator ^ (const Mat& a, const Scalar& s);
CV_EXPORTS MatExpr operator ^ (const Scalar& s, const Mat& a);

CV_EXPORTS MatExpr operator ~(const Mat& m);

CV_EXPORTS MatExpr min(const Mat& a, const Mat& b);
CV_EXPORTS MatExpr min(const Mat& a, double s);
CV_EXPORTS MatExpr min(double s, const Mat& a);

CV_EXPORTS MatExpr max(const Mat& a, const Mat& b);
CV_EXPORTS MatExpr max(const Mat& a, double s);
CV_EXPORTS MatExpr max(double s, const Mat& a);

CV_EXPORTS MatExpr abs(const Mat& m);
CV_EXPORTS MatExpr abs(const MatExpr& e);

} // cv

#include "opencv2/core/mat.inl.hpp"

#endif // __OPENCV_CORE_MAT_HPP__
