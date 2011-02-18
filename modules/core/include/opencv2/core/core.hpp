/*! \file core.hpp
    \brief The Core Functionality
 */
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

#ifndef __OPENCV_CORE_HPP__
#define __OPENCV_CORE_HPP__

#include "opencv2/core/types_c.h"
#include "opencv2/core/version.hpp"

#ifdef __cplusplus

#ifndef SKIP_INCLUDES
#include <limits.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <new>
#include <string>
#include <vector>
#endif // SKIP_INCLUDES

/*! \namespace cv
    Namespace where all the C++ OpenCV functionality resides
*/ 
namespace cv {

#undef abs
#undef min
#undef max
#undef Complex

using std::vector;
using std::string;
    
template<typename _Tp> class CV_EXPORTS Size_;
template<typename _Tp> class CV_EXPORTS Point_;
template<typename _Tp> class CV_EXPORTS Rect_;
template<typename _Tp, int cn> class CV_EXPORTS Vec;
template<typename _Tp, int m, int n> class CV_EXPORTS Matx;
    
typedef std::string String;
typedef std::basic_string<wchar_t> WString;

class Mat;
class SparseMat;
typedef Mat MatND;

class CV_EXPORTS MatExpr;
class CV_EXPORTS MatOp_Base;
class CV_EXPORTS MatArg;
class CV_EXPORTS MatConstIterator;

template<typename _Tp> class CV_EXPORTS Mat_;
template<typename _Tp> class CV_EXPORTS MatIterator_;
template<typename _Tp> class CV_EXPORTS MatConstIterator_;
template<typename _Tp> class CV_EXPORTS MatCommaInitializer_;
    
CV_EXPORTS string fromUtf16(const WString& str);
CV_EXPORTS WString toUtf16(const string& str);

CV_EXPORTS string format( const char* fmt, ... );

    
// matrix decomposition types
enum { DECOMP_LU=0, DECOMP_SVD=1, DECOMP_EIG=2, DECOMP_CHOLESKY=3, DECOMP_QR=4, DECOMP_NORMAL=16 };
enum { NORM_INF=1, NORM_L1=2, NORM_L2=4, NORM_TYPE_MASK=7, NORM_RELATIVE=8, NORM_MINMAX=32};
enum { CMP_EQ=0, CMP_GT=1, CMP_GE=2, CMP_LT=3, CMP_LE=4, CMP_NE=5 };
enum { GEMM_1_T=1, GEMM_2_T=2, GEMM_3_T=4 };
enum { DFT_INVERSE=1, DFT_SCALE=2, DFT_ROWS=4, DFT_COMPLEX_OUTPUT=16, DFT_REAL_OUTPUT=32,
    DCT_INVERSE = DFT_INVERSE, DCT_ROWS=DFT_ROWS };

    
/*!
 The standard OpenCV exception class.
 Instances of the class are thrown by various functions and methods in the case of critical errors.
 */
class CV_EXPORTS Exception : public std::exception
{
public:
	/*!
     Default constructor
     */
    Exception() { code = 0; line = 0; }
    /*!
     Full constructor. Normally the constuctor is not called explicitly.
     Instead, the macros CV_Error(), CV_Error_() and CV_Assert() are used.
    */
	Exception(int _code, const string& _err, const string& _func, const string& _file, int _line)
		: code(_code), err(_err), func(_func), file(_file), line(_line)
    { formatMessage(); }
    
	virtual ~Exception() throw() {}

    /*!
     \return the error description and the context as a text string.
    */ 
	virtual const char *what() const throw() { return msg.c_str(); }

    void formatMessage()
    {
        if( func.size() > 0 )
            msg = format("%s:%d: error: (%d) %s in function %s\n", file.c_str(), line, code, err.c_str(), func.c_str());
        else
            msg = format("%s:%d: error: (%d) %s\n", file.c_str(), line, code, err.c_str());
    }
    
	string msg; ///< the formatted error message

	int code; ///< error code @see CVStatus
	string err; ///< error description
	string func; ///< function name. Available only when the compiler supports __func__ macro
	string file; ///< source file name where the error has occured
	int line; ///< line number in the source file where the error has occured 
};


//! Signals an error and raises the exception.
 
/*!
  By default the function prints information about the error to stderr,
  then it either stops if setBreakOnError() had been called before or raises the exception.
  It is possible to alternate error processing by using redirectError().
 
  \param exc the exception raisen.
 */
CV_EXPORTS void error( const Exception& exc );

//! Sets/resets the break-on-error mode.

/*!
  When the break-on-error mode is set, the default error handler
  issues a hardware exception, which can make debugging more convenient.
 
  \return the previous state
 */
CV_EXPORTS bool setBreakOnError(bool flag);
    
typedef int (CV_CDECL *ErrorCallback)( int status, const char* func_name,
                                       const char* err_msg, const char* file_name,
                                       int line, void* userdata );

//! Sets the new error handler and the optional user data.

/*!
  The function sets the new error handler, called from cv::error().
  
  \param errCallback the new error handler. If NULL, the default error handler is used.
  \param userdata the optional user data pointer, passed to the callback.
  \param prevUserdata the optional output parameter where the previous user data pointer is stored
  
  \return the previous error handler
*/  
CV_EXPORTS ErrorCallback redirectError( ErrorCallback errCallback,
                                        void* userdata=0, void** prevUserdata=0);
    
#ifdef __GNUC__
#define CV_Error( code, msg ) cv::error( cv::Exception(code, msg, __func__, __FILE__, __LINE__) )
#define CV_Error_( code, args ) cv::error( cv::Exception(code, cv::format args, __func__, __FILE__, __LINE__) )
#define CV_Assert( expr ) if((expr)) ; else cv::error( cv::Exception(CV_StsAssert, #expr, __func__, __FILE__, __LINE__) )
#else
#define CV_Error( code, msg ) cv::error( cv::Exception(code, msg, "", __FILE__, __LINE__) )
#define CV_Error_( code, args ) cv::error( cv::Exception(code, cv::format args, "", __FILE__, __LINE__) )
#define CV_Assert( expr ) if((expr)) ; else cv::error( cv::Exception(CV_StsAssert, #expr, "", __FILE__, __LINE__) )
#endif
    
#ifdef _DEBUG
#define CV_DbgAssert(expr) CV_Assert(expr)
#else
#define CV_DbgAssert(expr)
#endif

CV_EXPORTS void setNumThreads(int nthreads);
CV_EXPORTS int getNumThreads();
CV_EXPORTS int getThreadNum();

//! Returns the number of ticks.

/*!
  The function returns the number of ticks since the certain event (e.g. when the machine was turned on).
  It can be used to initialize cv::RNG or to measure a function execution time by reading the tick count
  before and after the function call. The granularity of ticks depends on the hardware and OS used. Use
  cv::getTickFrequency() to convert ticks to seconds.
*/
CV_EXPORTS int64 getTickCount();

/*!
  Returns the number of ticks per seconds.

  The function returns the number of ticks (as returned by cv::getTickCount()) per second.
  The following code computes the execution time in milliseconds:
  
  \code
  double exec_time = (double)getTickCount();
  // do something ...
  exec_time = ((double)getTickCount() - exec_time)*1000./getTickFrequency();
  \endcode
*/
CV_EXPORTS_W double getTickFrequency();

/*!
  Returns the number of CPU ticks.

  On platforms where the feature is available, the function returns the number of CPU ticks
  since the certain event (normally, the system power-on moment). Using this function
  one can accurately measure the execution time of very small code fragments,
  for which cv::getTickCount() granularity is not enough.
*/
CV_EXPORTS int64 getCPUTickCount();

/*!
  Returns SSE etc. support status
  
  The function returns true if certain hardware features are available.
  Currently, the following features are recognized:
  - CV_CPU_MMX - MMX
  - CV_CPU_SSE - SSE
  - CV_CPU_SSE2 - SSE 2
  - CV_CPU_SSE3 - SSE 3
  - CV_CPU_SSSE3 - SSSE 3
  - CV_CPU_SSE4_1 - SSE 4.1
  - CV_CPU_SSE4_2 - SSE 4.2
  - CV_CPU_AVX - AVX
  
  \note {Note that the function output is not static. Once you called cv::useOptimized(false),
  most of the hardware acceleration is disabled and thus the function will returns false,
  until you call cv::useOptimized(true)}
*/
CV_EXPORTS_W bool checkHardwareSupport(int feature);

/*!
  Allocates memory buffer
  
  This is specialized OpenCV memory allocation function that returns properly aligned memory buffers.
  The usage is identical to malloc(). The allocated buffers must be freed with cv::fastFree().
  If there is not enough memory, the function calls cv::error(), which raises an exception.
  
  \param bufSize buffer size in bytes
  \return the allocated memory buffer.
*/ 
CV_EXPORTS void* fastMalloc(size_t bufSize);

/*!
  Frees the memory allocated with cv::fastMalloc
  
  This is the corresponding deallocation function for cv::fastMalloc().
  When ptr==NULL, the function has no effect.
*/
CV_EXPORTS void fastFree(void* ptr);

template<typename _Tp> static inline _Tp* allocate(size_t n)
{
    return new _Tp[n];
}

template<typename _Tp> static inline void deallocate(_Tp* ptr, size_t)
{
    delete[] ptr;
}

/*!
  Aligns pointer by the certain number of bytes
  
  This small inline function aligns the pointer by the certian number of bytes by shifting
  it forward by 0 or a positive offset.
*/  
template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

/*!
  Aligns buffer size by the certain number of bytes
  
  This small inline function aligns a buffer size by the certian number of bytes by enlarging it.
*/
static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n-1) & -n;
}

/*!
  Turns on/off available optimization
  
  The function turns on or off the optimized code in OpenCV. Some optimization can not be enabled
  or disabled, but, for example, most of SSE code in OpenCV can be temporarily turned on or off this way.
  
  \note{Since optimization may imply using special data structures, it may be unsafe
  to call this function anywhere in the code. Instead, call it somewhere at the top level.}
*/  
CV_EXPORTS_W void setUseOptimized(bool onoff);

/*!
  Returns the current optimization status
  
  The function returns the current optimization status, which is controlled by cv::setUseOptimized().
*/  
CV_EXPORTS_W bool useOptimized();

/*!
  The STL-compilant memory Allocator based on cv::fastMalloc() and cv::fastFree()
*/
template<typename _Tp> class CV_EXPORTS Allocator
{
public: 
    typedef _Tp value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    template<typename U> class rebind { typedef Allocator<U> other; };

    explicit Allocator() {}
    ~Allocator() {}
    explicit Allocator(Allocator const&) {}
    template<typename U>
    explicit Allocator(Allocator<U> const&) {}

    // address
    pointer address(reference r) { return &r; }
    const_pointer address(const_reference r) { return &r; }

    pointer allocate(size_type count, const void* =0)
    { return reinterpret_cast<pointer>(fastMalloc(count * sizeof (_Tp))); }

    void deallocate(pointer p, size_type) {fastFree(p); }

    size_type max_size() const
    { return max(static_cast<_Tp>(-1)/sizeof(_Tp), 1); }

    void construct(pointer p, const _Tp& v) { new(static_cast<void*>(p)) _Tp(v); }
    void destroy(pointer p) { p->~_Tp(); }
};

/////////////////////// Vec (used as element of multi-channel images ///////////////////// 

/*!
  A helper class for cv::DataType
  
  The class is specialized for each fundamental numerical data type supported by OpenCV.
  It provides DataDepth<T>::value constant.
*/  
template<typename _Tp> class CV_EXPORTS DataDepth {};

template<> class DataDepth<bool> { public: enum { value = CV_8U, fmt=(int)'u' }; };
template<> class DataDepth<uchar> { public: enum { value = CV_8U, fmt=(int)'u' }; };
template<> class DataDepth<schar> { public: enum { value = CV_8S, fmt=(int)'c' }; };
template<> class DataDepth<char> { public: enum { value = CV_8S, fmt=(int)'c' }; };
template<> class DataDepth<ushort> { public: enum { value = CV_16U, fmt=(int)'w' }; };
template<> class DataDepth<short> { public: enum { value = CV_16S, fmt=(int)'s' }; };
template<> class DataDepth<int> { public: enum { value = CV_32S, fmt=(int)'i' }; };
// this is temporary solution to support 32-bit unsigned integers
template<> class DataDepth<unsigned> { public: enum { value = CV_32S, fmt=(int)'i' }; };
template<> class DataDepth<float> { public: enum { value = CV_32F, fmt=(int)'f' }; };
template<> class DataDepth<double> { public: enum { value = CV_64F, fmt=(int)'d' }; };
template<typename _Tp> class DataDepth<_Tp*> { public: enum { value = CV_USRTYPE1, fmt=(int)'r' }; };


////////////////////////////// Small Matrix ///////////////////////////
    
/*!
 A short numerical vector.
 
 This template class represents short numerical vectors (of 1, 2, 3, 4 ... elements)
 on which you can perform basic arithmetical operations, access individual elements using [] operator etc.
 The vectors are allocated on stack, as opposite to std::valarray, std::vector, cv::Mat etc.,
 which elements are dynamically allocated in the heap.
 
 The template takes 2 parameters:
 -# _Tp element type
 -# cn the number of elements
 
 In addition to the universal notation like Vec<float, 3>, you can use shorter aliases
 for the most popular specialized variants of Vec, e.g. Vec3f ~ Vec<float, 3>. 
 */
    
struct CV_EXPORTS Matx_AddOp {};
struct CV_EXPORTS Matx_SubOp {};
struct CV_EXPORTS Matx_ScaleOp {};
struct CV_EXPORTS Matx_MulOp {};
struct CV_EXPORTS Matx_MatMulOp {};
struct CV_EXPORTS Matx_TOp {};
    
template<typename _Tp, int m, int n> class CV_EXPORTS Matx
{
public:
    typedef _Tp value_type;
    typedef Matx<_Tp, MIN(m, n), 1> diag_type;
    typedef Matx<_Tp, m, n> mat_type;
    enum { depth = DataDepth<_Tp>::value, rows = m, cols = n, channels = rows*cols,
           type = CV_MAKETYPE(depth, channels) };
    
    //! default constructor
    Matx();
    
    Matx(_Tp v0); //!< 1x1 matrix
    Matx(_Tp v0, _Tp v1); //!< 1x2 or 2x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2); //!< 1x3 or 3x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3); //!< 1x4, 2x2 or 4x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4); //!< 1x5 or 5x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5); //!< 1x6, 2x3, 3x2 or 6x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6); //!< 1x7 or 7x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7); //!< 1x8, 2x4, 4x2 or 8x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8); //!< 1x9, 3x3 or 9x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9); //!< 1x10, 2x5 or 5x2 or 10x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3,
         _Tp v4, _Tp v5, _Tp v6, _Tp v7,
         _Tp v8, _Tp v9, _Tp v10, _Tp v11); //!< 1x12, 2x6, 3x4, 4x3, 6x2 or 12x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3,
         _Tp v4, _Tp v5, _Tp v6, _Tp v7,
         _Tp v8, _Tp v9, _Tp v10, _Tp v11,
         _Tp v12, _Tp v13, _Tp v14, _Tp v15); //!< 1x16, 4x4 or 16x1 matrix
    explicit Matx(const _Tp* vals); //!< initialize from a plain array
    
    static Matx all(_Tp alpha);
    static Matx zeros();
    static Matx ones();
    static Matx eye();
    static Matx diag(const diag_type& d);
    static Matx randu(_Tp a, _Tp b);
    static Matx randn(_Tp a, _Tp b);
    
    //! dot product computed with the default precision
    _Tp dot(const Matx<_Tp, m, n>& v) const;
    
    //! dot product computed in double-precision arithmetics
    double ddot(const Matx<_Tp, m, n>& v) const;

    //! convertion to another data type
    template<typename T2> operator Matx<T2, m, n>() const;
    
    //! change the matrix shape
    template<int m1, int n1> Matx<_Tp, m1, n1> reshape() const;
    
    //! extract part of the matrix
    template<int m1, int n1> Matx<_Tp, m1, n1> get_minor(int i, int j) const;
    
    //! extract the matrix row
    Matx<_Tp, 1, n> row(int i) const;
    
    //! extract the matrix column
    Matx<_Tp, m, 1> col(int i) const;
    
    //! extract the matrix diagonal
    Matx<_Tp, MIN(m,n), 1> diag() const;
    
    //! transpose the matrix
    Matx<_Tp, n, m> t() const;
    
    //! invert matrix the matrix
    Matx<_Tp, n, m> inv(int method=DECOMP_LU) const;
    
    //! solve linear system
    template<int l> Matx<_Tp, n, l> solve(const Matx<_Tp, m, l>& rhs, int flags=DECOMP_LU) const;
    Matx<_Tp, n, 1> solve(const Matx<_Tp, m, 1>& rhs, int method) const;
    
    //! multiply two matrices element-wise
    Matx<_Tp, m, n> mul(const Matx<_Tp, m, n>& a) const;
    
    //! element access
    const _Tp& operator ()(int i, int j) const;
    _Tp& operator ()(int i, int j);
    
    //! 1D element access
    const _Tp& operator ()(int i) const;
    _Tp& operator ()(int i);
    
    Matx(const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b, Matx_AddOp);
    Matx(const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b, Matx_SubOp);
    template<typename _T2> Matx(const Matx<_Tp, m, n>& a, _T2 alpha, Matx_ScaleOp);
    Matx(const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b, Matx_MulOp);
    template<int l> Matx(const Matx<_Tp, m, l>& a, const Matx<_Tp, l, n>& b, Matx_MatMulOp);
    Matx(const Matx<_Tp, n, m>& a, Matx_TOp);

    _Tp val[m*n]; //< matrix elements
};

    
typedef Matx<float, 1, 2> Matx12f;
typedef Matx<double, 1, 2> Matx12d;
typedef Matx<float, 1, 3> Matx13f;
typedef Matx<double, 1, 3> Matx13d;
typedef Matx<float, 1, 4> Matx14f;
typedef Matx<double, 1, 4> Matx14d;
typedef Matx<float, 1, 6> Matx16f;
typedef Matx<double, 1, 6> Matx16d;
    
typedef Matx<float, 2, 1> Matx21f;
typedef Matx<double, 2, 1> Matx21d;
typedef Matx<float, 3, 1> Matx31f;
typedef Matx<double, 3, 1> Matx31d;
typedef Matx<float, 4, 1> Matx41f;
typedef Matx<double, 4, 1> Matx41d;
typedef Matx<float, 6, 1> Matx61f;
typedef Matx<double, 6, 1> Matx61d;

typedef Matx<float, 2, 2> Matx22f;
typedef Matx<double, 2, 2> Matx22d;
typedef Matx<float, 2, 3> Matx23f;
typedef Matx<double, 2, 3> Matx23d;
typedef Matx<float, 3, 2> Matx32f;
typedef Matx<double, 3, 2> Matx32d;
    
typedef Matx<float, 3, 3> Matx33f;
typedef Matx<double, 3, 3> Matx33d;
    
typedef Matx<float, 3, 4> Matx34f;
typedef Matx<double, 3, 4> Matx34d;
typedef Matx<float, 4, 3> Matx43f;
typedef Matx<double, 4, 3> Matx43d;
    
typedef Matx<float, 4, 4> Matx44f;
typedef Matx<double, 4, 4> Matx44d;
typedef Matx<float, 6, 6> Matx66f;
typedef Matx<double, 6, 6> Matx66d;    


/*!
  A short numerical vector.
  
  This template class represents short numerical vectors (of 1, 2, 3, 4 ... elements)
  on which you can perform basic arithmetical operations, access individual elements using [] operator etc.
  The vectors are allocated on stack, as opposite to std::valarray, std::vector, cv::Mat etc.,
  which elements are dynamically allocated in the heap.
  
  The template takes 2 parameters:
  -# _Tp element type
  -# cn the number of elements
  
  In addition to the universal notation like Vec<float, 3>, you can use shorter aliases
  for the most popular specialized variants of Vec, e.g. Vec3f ~ Vec<float, 3>. 
*/ 
template<typename _Tp, int cn> class CV_EXPORTS Vec : public Matx<_Tp, cn, 1>
{
public:
    typedef _Tp value_type;
    enum { depth = DataDepth<_Tp>::value, channels = cn, type = CV_MAKETYPE(depth, channels) };
    
    //! default constructor
    Vec();

    Vec(_Tp v0); //!< 1-element vector constructor
    Vec(_Tp v0, _Tp v1); //!< 2-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2); //!< 3-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3); //!< 4-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4); //!< 5-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5); //!< 6-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6); //!< 7-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7); //!< 8-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8); //!< 9-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9); //!< 10-element vector constructor
    explicit Vec(const _Tp* values);

    Vec(const Vec<_Tp, cn>& v);
    static Vec all(_Tp alpha);

    //! per-element multiplication
    Vec mul(const Vec<_Tp, cn>& v) const;
    
    /*!
      cross product of the two 3D vectors.
    
      For other dimensionalities the exception is raised
    */
    Vec cross(const Vec& v) const;
    //! convertion to another data type
    template<typename T2> operator Vec<T2, cn>() const;
    //! conversion to 4-element CvScalar.
    operator CvScalar() const;
    
    /*! element access */
    const _Tp& operator [](int i) const;
    _Tp& operator[](int i);
    const _Tp& operator ()(int i) const;
    _Tp& operator ()(int i);
};


/* \typedef

   Shorter aliases for the most popular specializations of Vec<T,n>
*/
typedef Vec<uchar, 2> Vec2b;
typedef Vec<uchar, 3> Vec3b;
typedef Vec<uchar, 4> Vec4b;

typedef Vec<short, 2> Vec2s;
typedef Vec<short, 3> Vec3s;
typedef Vec<short, 4> Vec4s;

typedef Vec<ushort, 2> Vec2w;
typedef Vec<ushort, 3> Vec3w;
typedef Vec<ushort, 4> Vec4w;    
    
typedef Vec<int, 2> Vec2i;
typedef Vec<int, 3> Vec3i;
typedef Vec<int, 4> Vec4i;

typedef Vec<float, 2> Vec2f;
typedef Vec<float, 3> Vec3f;
typedef Vec<float, 4> Vec4f;
typedef Vec<float, 6> Vec6f;

typedef Vec<double, 2> Vec2d;
typedef Vec<double, 3> Vec3d;
typedef Vec<double, 4> Vec4d;
typedef Vec<double, 6> Vec6d;


//////////////////////////////// Complex //////////////////////////////

/*!
  A complex number class.
  
  The template class is similar and compatible with std::complex, however it provides slightly
  more convenient access to the real and imaginary parts using through the simple field access, as opposite
  to std::complex::real() and std::complex::imag().
*/
template<typename _Tp> class CV_EXPORTS Complex
{
public:

    //! constructors
    Complex();
    Complex( _Tp _re, _Tp _im=0 );
    Complex( const std::complex<_Tp>& c );

    //! conversion to another data type
    template<typename T2> operator Complex<T2>() const;
    //! conjugation
    Complex conj() const;
    //! conversion to std::complex
    operator std::complex<_Tp>() const;

    _Tp re, im; //< the real and the imaginary parts
};


/*!
  \typedef
*/
typedef Complex<float> Complexf;
typedef Complex<double> Complexd;


//////////////////////////////// Point_ ////////////////////////////////

/*!
  template 2D point class.
  
  The class defines a point in 2D space. Data type of the point coordinates is specified
  as a template parameter. There are a few shorter aliases available for user convenience. 
  See cv::Point, cv::Point2i, cv::Point2f and cv::Point2d.
*/  
template<typename _Tp> class CV_EXPORTS Point_
{
public:
    typedef _Tp value_type;
    
    // various constructors
    Point_();
    Point_(_Tp _x, _Tp _y);
    Point_(const Point_& pt);
    Point_(const CvPoint& pt);
    Point_(const CvPoint2D32f& pt);
    Point_(const Size_<_Tp>& sz);
    Point_(const Vec<_Tp, 2>& v);

    Point_& operator = (const Point_& pt);
    //! conversion to another data type
    template<typename _Tp2> operator Point_<_Tp2>() const;

    //! conversion to the old-style C structures
    operator CvPoint() const;
    operator CvPoint2D32f() const;
    operator Vec<_Tp, 2>() const;

    //! dot product
    _Tp dot(const Point_& pt) const;
    //! dot product computed in double-precision arithmetics
    double ddot(const Point_& pt) const;
    //! checks whether the point is inside the specified rectangle
    bool inside(const Rect_<_Tp>& r) const;
    
    _Tp x, y; //< the point coordinates
};

/*!
  template 3D point class.
  
  The class defines a point in 3D space. Data type of the point coordinates is specified
  as a template parameter.
  
  \see cv::Point3i, cv::Point3f and cv::Point3d
*/
template<typename _Tp> class CV_EXPORTS Point3_
{
public:
    typedef _Tp value_type;
    
    // various constructors
    Point3_();
    Point3_(_Tp _x, _Tp _y, _Tp _z);
    Point3_(const Point3_& pt);
	explicit Point3_(const Point_<_Tp>& pt);
    Point3_(const CvPoint3D32f& pt);
    Point3_(const Vec<_Tp, 3>& v);

    Point3_& operator = (const Point3_& pt);
    //! conversion to another data type
    template<typename _Tp2> operator Point3_<_Tp2>() const;
    //! conversion to the old-style CvPoint...
    operator CvPoint3D32f() const;
    //! conversion to cv::Vec<>
    operator Vec<_Tp, 3>() const;

    //! dot product
    _Tp dot(const Point3_& pt) const;
    //! dot product computed in double-precision arithmetics
    double ddot(const Point3_& pt) const;
    //! cross product of the 2 3D points
    Point3_ cross(const Point3_& pt) const;
    
    _Tp x, y, z; //< the point coordinates
};

//////////////////////////////// Size_ ////////////////////////////////

/*!
  The 2D size class
  
  The class represents the size of a 2D rectangle, image size, matrix size etc.
  Normally, cv::Size ~ cv::Size_<int> is used.
*/
template<typename _Tp> class CV_EXPORTS Size_
{
public:
    typedef _Tp value_type;
    
    //! various constructors
    Size_();
    Size_(_Tp _width, _Tp _height);
    Size_(const Size_& sz);
    Size_(const CvSize& sz);
    Size_(const CvSize2D32f& sz);
    Size_(const Point_<_Tp>& pt);

    Size_& operator = (const Size_& sz);
    //! the area (width*height)
    _Tp area() const;

    //! conversion of another data type.
    template<typename _Tp2> operator Size_<_Tp2>() const;
    
    //! conversion to the old-style OpenCV types
    operator CvSize() const;
    operator CvSize2D32f() const;

    _Tp width, height; // the width and the height
};

//////////////////////////////// Rect_ ////////////////////////////////

/*!
  The 2D up-right rectangle class
  
  The class represents a 2D rectangle with coordinates of the specified data type.
  Normally, cv::Rect ~ cv::Rect_<int> is used.
*/
template<typename _Tp> class CV_EXPORTS Rect_
{
public:
    typedef _Tp value_type;
    
    //! various constructors
    Rect_();
    Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
    Rect_(const Rect_& r);
    Rect_(const CvRect& r);
    Rect_(const Point_<_Tp>& org, const Size_<_Tp>& sz);
    Rect_(const Point_<_Tp>& pt1, const Point_<_Tp>& pt2);

    Rect_& operator = ( const Rect_& r );
    //! the top-left corner
    Point_<_Tp> tl() const;
    //! the bottom-right corner
    Point_<_Tp> br() const;
    
    //! size (width, height) of the rectangle
    Size_<_Tp> size() const;
    //! area (width*height) of the rectangle
    _Tp area() const;

    //! conversion to another data type
    template<typename _Tp2> operator Rect_<_Tp2>() const;
    //! conversion to the old-style CvRect
    operator CvRect() const;

    //! checks whether the rectangle contains the point
    bool contains(const Point_<_Tp>& pt) const;

    _Tp x, y, width, height; //< the top-left corner, as well as width and height of the rectangle
};


/*!
  \typedef
  
  shorter aliases for the most popular cv::Point_<>, cv::Size_<> and cv::Rect_<> specializations
*/
typedef Point_<int> Point2i;
typedef Point2i Point;
typedef Size_<int> Size2i;
typedef Size2i Size;
typedef Rect_<int> Rect;
typedef Point_<float> Point2f;
typedef Point_<double> Point2d;
typedef Size_<float> Size2f;
typedef Point3_<int> Point3i;
typedef Point3_<float> Point3f;
typedef Point3_<double> Point3d;


/*!
  The rotated 2D rectangle.
  
  The class represents rotated (i.e. not up-right) rectangles on a plane.
  Each rectangle is described by the center point (mass center), length of each side
  (represented by cv::Size2f structure) and the rotation angle in degrees.
*/  
class CV_EXPORTS RotatedRect
{
public:
    //! various constructors
    RotatedRect();
    RotatedRect(const Point2f& _center, const Size2f& _size, float _angle);
    RotatedRect(const CvBox2D& box);

    //! returns 4 vertices of the rectangle
    void points(Point2f pts[]) const;
    //! returns the minimal up-right rectangle containing the rotated rectangle
    Rect boundingRect() const;
    //! conversion to the old-style CvBox2D structure
    operator CvBox2D() const;
    
    Point2f center; //< the rectangle mass center
    Size2f size;    //< width and height of the rectangle
    float angle;    //< the rotation angle. When the angle is 0, 90, 180, 270 etc., the rectangle becomes an up-right rectangle. 
};

//////////////////////////////// Scalar_ ///////////////////////////////

/*!
   The template scalar class.
   
   This is partially specialized cv::Vec class with the number of elements = 4, i.e. a short vector of four elements.
   Normally, cv::Scalar ~ cv::Scalar_<double> is used. 
*/
template<typename _Tp> class CV_EXPORTS Scalar_ : public Vec<_Tp, 4>
{
public:
    //! various constructors
    Scalar_();
    Scalar_(_Tp v0, _Tp v1, _Tp v2=0, _Tp v3=0);
    Scalar_(const CvScalar& s);
    Scalar_(_Tp v0);

    //! returns a scalar with all elements set to v0
    static Scalar_<_Tp> all(_Tp v0);
    //! conversion to the old-style CvScalar
    operator CvScalar() const;

    //! conversion to another data type
    template<typename T2> operator Scalar_<T2>() const;

    //! per-element product
    Scalar_<_Tp> mul(const Scalar_<_Tp>& t, double scale=1 ) const;
    
    // returns (v0, -v1, -v2, -v3)
    Scalar_<_Tp> conj() const;
    
    // returns true iff v1 == v2 == v3 == 0
    bool isReal() const;
};

typedef Scalar_<double> Scalar;

CV_EXPORTS void scalarToRawData(const Scalar& s, void* buf, int type, int unroll_to=0);

//////////////////////////////// Range /////////////////////////////////

/*!
   The 2D range class
   
   This is the class used to specify a continuous subsequence, i.e. part of a contour, or a column span in a matrix.
*/
class CV_EXPORTS Range
{
public:
    Range();
    Range(int _start, int _end);
    Range(const CvSlice& slice);
    int size() const;
    bool empty() const;
    static Range all();
    operator CvSlice() const;

    int start, end;
};

/////////////////////////////// DataType ////////////////////////////////

/*!
   Informative template class for OpenCV "scalars".
   
   The class is specialized for each primitive numerical type supported by OpenCV (such as unsigned char or float),
   as well as for more complex types, like cv::Complex<>, std::complex<>, cv::Vec<> etc.
   The common property of all such types (called "scalars", do not confuse it with cv::Scalar_)
   is that each of them is basically a tuple of numbers of the same type. Each "scalar" can be represented
   by the depth id (CV_8U ... CV_64F) and the number of channels.
   OpenCV matrices, 2D or nD, dense or sparse, can store "scalars",
   as long as the number of channels does not exceed CV_CN_MAX.
*/
template<typename _Tp> class DataType
{
public:
    typedef _Tp value_type;
    typedef value_type work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    
    enum { generic_type = 1, depth = DataDepth<channel_type>::value, channels = 1,
        fmt=DataDepth<channel_type>::fmt,
        type = CV_MAKETYPE(depth, channels) };
};

template<> class DataType<bool>
{
public:
    typedef bool value_type;
    typedef int work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = 1,
           fmt=DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
};

template<> class DataType<uchar>
{
public:
    typedef uchar value_type;
    typedef int work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = 1,
           fmt=DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
};

template<> class DataType<schar>
{
public:
    typedef schar value_type;
    typedef int work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = 1,
           fmt=DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
};

template<> class DataType<char>
{
public:
    typedef schar value_type;
    typedef int work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = 1,
           fmt=DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
};

template<> class DataType<ushort>
{
public:
    typedef ushort value_type;
    typedef int work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = 1,
           fmt=DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
};

template<> class DataType<short>
{
public:
    typedef short value_type;
    typedef int work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = 1,
           fmt=DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
};

template<> class DataType<int>
{
public:
    typedef int value_type;
    typedef value_type work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = 1,
           fmt=DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
};

template<> class DataType<float>
{
public:
    typedef float value_type;
    typedef value_type work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = 1,
           fmt=DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
};

template<> class DataType<double>
{
public:
    typedef double value_type;
    typedef value_type work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = 1,
           fmt=DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
};

template<typename _Tp, int cn> class DataType<Vec<_Tp, cn> >
{
public:
    typedef Vec<_Tp, cn> value_type;
    typedef Vec<typename DataType<_Tp>::work_type, cn> work_type;
    typedef _Tp channel_type;
    typedef value_type vec_type;
    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = cn,
           fmt = ((channels-1)<<8) + DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
};

template<typename _Tp> class DataType<std::complex<_Tp> >
{
public:
    typedef std::complex<_Tp> value_type;
    typedef value_type work_type;
    typedef _Tp channel_type;
    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = 2,
           fmt = ((channels-1)<<8) + DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
    typedef Vec<channel_type, channels> vec_type;
};

template<typename _Tp> class DataType<Complex<_Tp> >
{
public:
    typedef Complex<_Tp> value_type;
    typedef value_type work_type;
    typedef _Tp channel_type;
    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = 2,
           fmt = ((channels-1)<<8) + DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
    typedef Vec<channel_type, channels> vec_type;
};

template<typename _Tp> class DataType<Point_<_Tp> >
{
public:
    typedef Point_<_Tp> value_type;
    typedef Point_<typename DataType<_Tp>::work_type> work_type;
    typedef _Tp channel_type;
    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = 2,
           fmt = ((channels-1)<<8) + DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
    typedef Vec<channel_type, channels> vec_type;
};

template<typename _Tp> class DataType<Point3_<_Tp> >
{
public:
    typedef Point3_<_Tp> value_type;
    typedef Point3_<typename DataType<_Tp>::work_type> work_type;
    typedef _Tp channel_type;
    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = 3,
           fmt = ((channels-1)<<8) + DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
    typedef Vec<channel_type, channels> vec_type;
};

template<typename _Tp> class DataType<Size_<_Tp> >
{
public:
    typedef Size_<_Tp> value_type;
    typedef Size_<typename DataType<_Tp>::work_type> work_type;
    typedef _Tp channel_type;
    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = 2,
           fmt = ((channels-1)<<8) + DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
    typedef Vec<channel_type, channels> vec_type;
};

template<typename _Tp> class DataType<Rect_<_Tp> >
{
public:
    typedef Rect_<_Tp> value_type;
    typedef Rect_<typename DataType<_Tp>::work_type> work_type;
    typedef _Tp channel_type;
    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = 4,
           fmt = ((channels-1)<<8) + DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
    typedef Vec<channel_type, channels> vec_type;
};

template<typename _Tp> class DataType<Scalar_<_Tp> >
{
public:
    typedef Scalar_<_Tp> value_type;
    typedef Scalar_<typename DataType<_Tp>::work_type> work_type;
    typedef _Tp channel_type;
    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = 4,
           fmt = ((channels-1)<<8) + DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
    typedef Vec<channel_type, channels> vec_type;
};

template<> class DataType<Range>
{
public:
    typedef Range value_type;
    typedef value_type work_type;
    typedef int channel_type;
    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = 2,
           fmt = ((channels-1)<<8) + DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
    typedef Vec<channel_type, channels> vec_type;
};

    
//////////////////// generic_type ref-counting pointer class for C/C++ objects ////////////////////////

/*!
  Smart pointer to dynamically allocated objects.
  
  This is template pointer-wrapping class that stores the associated reference counter along with the
  object pointer. The class is similar to std::smart_ptr<> from the recent addons to the C++ standard,
  but is shorter to write :) and self-contained (i.e. does add any dependency on the compiler or an external library).
  
  Basically, you can use "Ptr<MyObjectType> ptr" (or faster "const Ptr<MyObjectType>& ptr" for read-only access)
  everywhere instead of "MyObjectType* ptr", where MyObjectType is some C structure or a C++ class.
  To make it all work, you need to specialize Ptr<>::delete_obj(), like:
  
  \code
  template<> void Ptr<MyObjectType>::delete_obj() { call_destructor_func(obj); }
  \endcode
  
  \note{if MyObjectType is a C++ class with a destructor, you do not need to specialize delete_obj(),
  since the default implementation calls "delete obj;"}
  
  \note{Another good property of the class is that the operations on the reference counter are atomic,
  i.e. it is safe to use the class in multi-threaded applications}
*/
template<typename _Tp> class CV_EXPORTS Ptr
{
public:
    //! empty constructor
    Ptr();
    //! take ownership of the pointer. The associated reference counter is allocated and set to 1
    Ptr(_Tp* _obj);
    //! calls release()
    ~Ptr();
    //! copy constructor. Copies the members and calls addref()
    Ptr(const Ptr& ptr);
    //! copy operator. Calls ptr.addref() and release() before copying the members
    Ptr& operator = (const Ptr& ptr);
    //! increments the reference counter
    void addref();
    //! decrements the reference counter. If it reaches 0, delete_obj() is called
    void release();
    //! deletes the object. Override if needed
    void delete_obj();
    //! returns true iff obj==NULL
    bool empty() const;

    
    //! helper operators making "Ptr<T> ptr" use very similar to "T* ptr".
    _Tp* operator -> ();
    const _Tp* operator -> () const;

    operator _Tp* ();
    operator const _Tp*() const;
    
protected:
    _Tp* obj; //< the object pointer.
    int* refcount; //< the associated                         bbbbbbbbbbbbbbbbbb reference counter
};

//////////////////////////////// Mat ////////////////////////////////

enum { MAGIC_MASK=0xFFFF0000, TYPE_MASK=0x00000FFF, DEPTH_MASK=7 };

static inline size_t getElemSize(int type) { return CV_ELEM_SIZE(type); }

/*!
   Custom array allocator
 
*/
class CV_EXPORTS MatAllocator
{
public:
    MatAllocator() {}
    virtual ~MatAllocator() {}
    virtual void allocate(int dims, const int* sizes, int type, int*& refcount,
                          uchar*& datastart, uchar*& data, size_t* step) = 0;
    virtual void deallocate(int* refcount, uchar* datastart, uchar* data) = 0;
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
    Mat(int _rows, int _cols, int _type);
    Mat(Size _size, int _type);
    //! constucts 2D matrix and fills it with the specified value _s.
    Mat(int _rows, int _cols, int _type, const Scalar& _s);
    Mat(Size _size, int _type, const Scalar& _s);
    
    //! constructs n-dimensional matrix
    Mat(int _ndims, const int* _sizes, int _type);
    Mat(int _ndims, const int* _sizes, int _type, const Scalar& _s);
    
    //! copy constructor
    Mat(const Mat& m);
    //! constructor for matrix headers pointing to user-allocated data
    Mat(int _rows, int _cols, int _type, void* _data, size_t _step=AUTO_STEP);
    Mat(Size _size, int _type, void* _data, size_t _step=AUTO_STEP);
    Mat(int _ndims, const int* _sizes, int _type, void* _data, const size_t* _steps=0);
    
    //! creates a matrix header for a part of the bigger matrix
    Mat(const Mat& m, const Range& rowRange, const Range& colRange=Range::all());
    Mat(const Mat& m, const Rect& roi);
    Mat(const Mat& m, const Range* ranges);
    //! converts old-style CvMat to the new matrix; the data is not copied by default
    Mat(const CvMat* m, bool copyData=false);
    //! converts old-style CvMatND to the new matrix; the data is not copied by default
    Mat(const CvMatND* m, bool copyData=false);
    //! converts old-style IplImage to the new matrix; the data is not copied by default
    Mat(const IplImage* img, bool copyData=false);
    //! builds matrix from std::vector with or without copying the data
    template<typename _Tp> explicit Mat(const vector<_Tp>& vec, bool copyData=false);
    //! builds matrix from cv::Vec; the data is copied by default
    template<typename _Tp, int n> explicit Mat(const Vec<_Tp, n>& vec,
                                               bool copyData=true);
    //! builds matrix from cv::Matx; the data is copied by default
    template<typename _Tp, int m, int n> explicit Mat(const Matx<_Tp, m, n>& mtx,
                                                      bool copyData=true);
    //! builds matrix from a 2D point
    template<typename _Tp> explicit Mat(const Point_<_Tp>& pt, bool copyData=true);
    //! builds matrix from a 3D point
    template<typename _Tp> explicit Mat(const Point3_<_Tp>& pt, bool copyData=true);
    //! builds matrix from comma initializer
    template<typename _Tp> explicit Mat(const MatCommaInitializer_<_Tp>& commaInitializer);
    //! destructor - calls release()
    ~Mat();
    //! assignment operators
    Mat& operator = (const Mat& m);
    Mat& operator = (const MatExpr& expr);

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
    void copyTo( Mat& m ) const;
    template<typename _Tp> void copyTo( vector<_Tp>& v ) const;
    //! copies those matrix elements to "m" that are marked with non-zero mask elements.
    void copyTo( Mat& m, const Mat& mask ) const;
    //! converts matrix to another datatype with optional scalng. See cvConvertScale.
    void convertTo( Mat& m, int rtype, double alpha=1, double beta=0 ) const;

    void assignTo( Mat& m, int type=-1 ) const;

    //! sets every matrix element to s
    Mat& operator = (const Scalar& s);
    //! sets some of the matrix elements to s, according to the mask
    Mat& setTo(const Scalar& s, const Mat& mask=Mat());
    //! creates alternative matrix header for the same data, with different
    // number of channels and/or different number of rows. see cvReshape.
    Mat reshape(int _cn, int _rows=0) const;
    Mat reshape(int _cn, int _newndims, const int* _newsz) const;

    //! matrix transposition by means of matrix expressions
    MatExpr t() const;
    //! matrix inversion by means of matrix expressions
    MatExpr inv(int method=DECOMP_LU) const;
    //! per-element matrix multiplication by means of matrix expressions
    MatExpr mul(const Mat& m, double scale=1) const;
    MatExpr mul(const MatExpr& m, double scale=1) const;
    
    //! computes cross-product of 2 3D vectors
    Mat cross(const Mat& m) const;
    //! computes dot-product
    double dot(const Mat& m) const;

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
    void create(int _rows, int _cols, int _type);
    void create(Size _size, int _type);
    void create(int _ndims, const int* _sizes, int _type);
    
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

    //! converts header to CvMat; no data is copied
    operator CvMat() const;
    //! converts header to CvMatND; no data is copied
    operator CvMatND() const;
    //! converts header to IplImage; no data is copied
    operator IplImage() const;
    
    template<typename _Tp> operator vector<_Tp>() const;
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

    enum { MAGIC_VAL=0x42FF0000, AUTO_STEP=0, CONTINUOUS_FLAG=CV_MAT_CONT_FLAG, SUBMATRIX_FLAG=CV_SUBMAT_FLAG };

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

    //! pointer to the reference counter;
    // when matrix points to user-allocated data, the pointer is NULL
    int* refcount;
    
    //! helper fields used in locateROI and adjustROI
    uchar* datastart;
    uchar* dataend;
    uchar* datalimit;
    
    //! custom allocator
    MatAllocator* allocator;
    
    struct CV_EXPORTS MSize
    {
        MSize(int* _p);
        Size operator()() const;
        const int& operator[](int i) const;
        int& operator[](int i);
        operator const int*() const;
        bool operator == (const MSize& sz) const;
        bool operator != (const MSize& sz) const;
        
        int* p;
    };
    
    struct CV_EXPORTS MStep
    {
        MStep();
        MStep(size_t s);
        const size_t& operator[](int i) const;
        size_t& operator[](int i);
        operator size_t() const;
        MStep& operator = (size_t s);
        
        size_t* p;
        size_t buf[2];
    protected:
        MStep& operator = (const MStep&);
    };
    
    MSize size;
    MStep step;
};

 
/*!
   Random Number Generator
 
   The class implements RNG using Multiply-with-Carry algorithm
*/
class CV_EXPORTS RNG
{
public:
    enum { UNIFORM=0, NORMAL=1 };

    RNG();
    RNG(uint64 _state);
    //! updates the state and returns the next 32-bit unsigned integer random number
    unsigned next();

    operator uchar();
    operator schar();
    operator ushort();
    operator short();
    operator unsigned();
	//! returns a random integer sampled uniformly from [0, N).
	unsigned operator()(unsigned N);
	unsigned operator ()();
    operator int();
    operator float();
    operator double();
    //! returns uniformly distributed integer random number from [a,b) range
    int uniform(int a, int b);
    //! returns uniformly distributed floating-point random number from [a,b) range
    float uniform(float a, float b);
    //! returns uniformly distributed double-precision floating-point random number from [a,b) range
    double uniform(double a, double b);
    void fill( Mat& mat, int distType, const Scalar& a, const Scalar& b );
	//! returns Gaussian random variate with mean zero.
	double gaussian(double sigma);

    uint64 state;
};
    
    


/*!
 Termination criteria in iterative algorithms
 */
class CV_EXPORTS TermCriteria
{
public:
    enum
    {
        COUNT=1, //!< the maximum number of iterations or elements to compute
        MAX_ITER=COUNT, //!< ditto
        EPS=2 //!< the desired accuracy or change in parameters at which the iterative algorithm stops
    };

    //! default constructor
    TermCriteria();
    //! full constructor
    TermCriteria(int _type, int _maxCount, double _epsilon);
    //! conversion from CvTermCriteria
    TermCriteria(const CvTermCriteria& criteria);
    //! conversion from CvTermCriteria
    operator CvTermCriteria() const;
    
    int type; //!< the type of termination criteria: COUNT, EPS or COUNT + EPS
    int maxCount; // the maximum number of iterations/elements
    double epsilon; // the desired accuracy
};

//! swaps two matrices
CV_EXPORTS void swap(Mat& a, Mat& b);
    
//! converts array (CvMat or IplImage) to cv::Mat
CV_EXPORTS Mat cvarrToMat(const CvArr* arr, bool copyData=false,
                          bool allowND=true, int coiMode=0);
//! extracts Channel of Interest from CvMat or IplImage and makes cv::Mat out of it.
CV_EXPORTS void extractImageCOI(const CvArr* arr, CV_OUT Mat& coiimg, int coi=-1);
//! inserts single-channel cv::Mat into a multi-channel CvMat or IplImage
CV_EXPORTS void insertImageCOI(const Mat& coiimg, CvArr* arr, int coi=-1);

//! adds one matrix to another (dst = src1 + src2)
CV_EXPORTS_W void add(const Mat& src1, const Mat& src2, CV_OUT Mat& dst, const Mat& mask CV_WRAP_DEFAULT(Mat()));
//! subtracts one matrix from another (dst = src1 - src2) 
CV_EXPORTS_W void subtract(const Mat& src1, const Mat& src2, CV_OUT Mat& dst, const Mat& mask CV_WRAP_DEFAULT(Mat()));
//! adds one matrix to another (dst = src1 + src2)    
CV_EXPORTS void add(const Mat& src1, const Mat& src2, CV_OUT Mat& dst);
//! subtracts one matrix from another (dst = src1 - src2) 
CV_EXPORTS void subtract(const Mat& src1, const Mat& src2, CV_OUT Mat& dst);
//! adds scalar to a matrix (dst = src1 + src2)
CV_EXPORTS_W void add(const Mat& src1, const Scalar& src2, CV_OUT Mat& dst, const Mat& mask=Mat());
//! subtracts scalar from a matrix (dst = src1 - src2)    
CV_EXPORTS_W void subtract(const Mat& src1, const Scalar& src2, CV_OUT Mat& dst, const Mat& mask=Mat());
//! subtracts matrix from scalar (dst = src1 - src2)    
CV_EXPORTS_W void subtract(const Scalar& src1, const Mat& src2, CV_OUT Mat& dst, const Mat& mask=Mat());

//! computes element-wise weighted product of the two arrays (dst = scale*src1*src2)
CV_EXPORTS_W void multiply(const Mat& src1, const Mat& src2, CV_OUT Mat& dst, double scale=1);
//! computes element-wise weighted quotient of the two arrays (dst = scale*src1/src2)
CV_EXPORTS_W void divide(const Mat& src1, const Mat& src2, CV_OUT Mat& dst, double scale=1);
//! computes element-wise weighted reciprocal of an array (dst = scale/src2)
CV_EXPORTS_W void divide(double scale, const Mat& src2, CV_OUT Mat& dst);

//! adds scaled array to another one (dst = alpha*src1 + src2)
CV_EXPORTS_W void scaleAdd(const Mat& src1, double alpha, const Mat& src2, CV_OUT Mat& dst);
//! computes weighted sum of two arrays (dst = alpha*src1 + beta*src2 + gamma)
CV_EXPORTS_W void addWeighted(const Mat& src1, double alpha, const Mat& src2,
                            double beta, double gamma, CV_OUT Mat& dst);
//! scales array elements, computes absolute values and converts the results to 8-bit unsigned integers: dst(i)=saturate_cast<uchar>abs(src(i)*alpha+beta)
CV_EXPORTS_W void convertScaleAbs(const Mat& src, CV_OUT Mat& dst, double alpha=1, double beta=0);
//! transforms 8-bit unsigned integers using lookup table: dst(i)=lut(src(i))
CV_EXPORTS_W void LUT(const Mat& src, const Mat& lut, CV_OUT Mat& dst);

//! computes sum of array elements
CV_EXPORTS_W Scalar sum(const Mat& src);
//! computes the number of nonzero array elements
CV_EXPORTS_W int countNonZero( const Mat& src );

//! computes mean value of array elements
CV_EXPORTS Scalar mean(const Mat& src);
//! computes mean value of selected array elements
CV_EXPORTS_W Scalar mean(const Mat& src, const Mat& mask CV_WRAP_DEFAULT(Mat()));
//! computes mean value and standard deviation of all or selected array elements
CV_EXPORTS_W void meanStdDev(const Mat& src, CV_OUT Scalar& mean, CV_OUT Scalar& stddev, const Mat& mask=Mat());
//! computes norm of array
CV_EXPORTS double norm(const Mat& src1, int normType=NORM_L2);
//! computes norm of the difference between two arrays
CV_EXPORTS double norm(const Mat& src1, const Mat& src2, int normType=NORM_L2);
//! computes norm of the selected array part
CV_EXPORTS_W double norm(const Mat& src1, int normType, const Mat& mask CV_WRAP_DEFAULT(Mat()));
//! computes norm of selected part of the difference between two arrays
CV_EXPORTS_W double norm(const Mat& src1, const Mat& src2,
                       int normType, const Mat& mask CV_WRAP_DEFAULT(Mat()));
//! scales and shifts array elements so that either the specified norm (alpha) or the minimum (alpha) and maximum (beta) array values get the specified values 
CV_EXPORTS_W void normalize( const Mat& src, CV_OUT Mat& dst, double alpha=1, double beta=0,
                           int norm_type=NORM_L2, int rtype=-1, const Mat& mask=Mat());

//! finds global minimum and maximum array elements and returns their values and their locations
CV_EXPORTS_W void minMaxLoc(const Mat& src, CV_OUT double* minVal,
                          CV_OUT double* maxVal=0, CV_OUT Point* minLoc=0,
                          CV_OUT Point* maxLoc=0, const Mat& mask=Mat());
CV_EXPORTS void minMaxIdx(const Mat& src, double* minVal, double* maxVal,
                          int* minIdx=0, int* maxIdx=0, const Mat& mask=Mat());
    
//! transforms 2D matrix to 1D row or column vector by taking sum, minimum, maximum or mean value over all the rows
CV_EXPORTS_W void reduce(const Mat& src, CV_OUT Mat& dst, int dim, int rtype, int dtype=-1);
//! makes multi-channel array out of several single-channel arrays
CV_EXPORTS void merge(const Mat* mv, size_t count, CV_OUT Mat& dst);
//! makes multi-channel array out of several single-channel arrays
CV_EXPORTS_W void merge(const vector<Mat>& mv, Mat& dst);
    
//! copies each plane of a multi-channel array to a dedicated array
CV_EXPORTS void split(const Mat& src, Mat* mvbegin);
//! copies each plane of a multi-channel array to a dedicated array
CV_EXPORTS_W void split(const Mat& m, vector<Mat>& mv);
    
//! copies selected channels from the input arrays to the selected channels of the output arrays
CV_EXPORTS void mixChannels(const Mat* src, size_t nsrcs, Mat* dst, size_t ndsts,
                            const int* fromTo, size_t npairs);
CV_EXPORTS void mixChannels(const vector<Mat>& src, vector<Mat>& dst,
                            const int* fromTo, int npairs);

//! reverses the order of the rows, columns or both in a matrix
CV_EXPORTS_W void flip(const Mat& src, CV_OUT Mat& dst, int flipCode);

//! replicates the input matrix the specified number of times in the horizontal and/or vertical direction
CV_EXPORTS_W void repeat(const Mat& src, int ny, int nx, CV_OUT Mat& dst);
CV_EXPORTS Mat repeat(const Mat& src, int ny, int nx);
        
CV_EXPORTS void hconcat(const Mat* src, size_t nsrc, Mat& dst);
CV_EXPORTS void hconcat(const Mat& src1, const Mat& src2, Mat& dst);
CV_EXPORTS_W void hconcat(const vector<Mat>& src, CV_OUT Mat& dst);

CV_EXPORTS void vconcat(const Mat* src, size_t nsrc, Mat& dst);
CV_EXPORTS void vconcat(const Mat& src1, const Mat& src2, Mat& dst);
CV_EXPORTS_W void vconcat(const vector<Mat>& src, CV_OUT Mat& dst);
    
//! computes bitwise conjunction of the two arrays (dst = src1 & src2)
CV_EXPORTS_W void bitwise_and(const Mat& src1, const Mat& src2, CV_OUT Mat& dst, const Mat& mask=Mat());
//! computes bitwise disjunction of the two arrays (dst = src1 | src2)
CV_EXPORTS_W void bitwise_or(const Mat& src1, const Mat& src2, CV_OUT Mat& dst, const Mat& mask=Mat());
//! computes bitwise exclusive-or of the two arrays (dst = src1 ^ src2)
CV_EXPORTS_W void bitwise_xor(const Mat& src1, const Mat& src2, CV_OUT Mat& dst, const Mat& mask=Mat());
//! computes bitwise conjunction of an array and scalar (dst = src1 & src2)
CV_EXPORTS_W void bitwise_and(const Mat& src1, const Scalar& src2, CV_OUT Mat& dst, const Mat& mask=Mat());
//! computes bitwise disjunction of an array and scalar (dst = src1 | src2)
CV_EXPORTS_W void bitwise_or(const Mat& src1, const Scalar& src2, CV_OUT Mat& dst, const Mat& mask=Mat());
//! computes bitwise exclusive-or of an array and scalar (dst = src1 ^ src2)
CV_EXPORTS_W void bitwise_xor(const Mat& src1, const Scalar& src2, CV_OUT Mat& dst, const Mat& mask=Mat());
//! inverts each bit of array (dst = ~src)
CV_EXPORTS_W void bitwise_not(const Mat& src, CV_OUT Mat& dst);
//! computes element-wise absolute difference of two arrays (dst = abs(src1 - src2))
CV_EXPORTS_W void absdiff(const Mat& src1, const Mat& src2, CV_OUT Mat& dst);
//! computes element-wise absolute difference of array and scalar (dst = abs(src1 - src2))
CV_EXPORTS_W void absdiff(const Mat& src1, const Scalar& src2, CV_OUT Mat& dst);
//! set mask elements for those array elements which are within the element-specific bounding box (dst = lowerb <= src && src < upperb)    
CV_EXPORTS_W void inRange(const Mat& src, const Mat& lowerb,
                        const Mat& upperb, CV_OUT Mat& dst);
//! set mask elements for those array elements which are within the fixed bounding box (dst = lowerb <= src && src < upperb)    
CV_EXPORTS_W void inRange(const Mat& src, const Scalar& lowerb,
                        const Scalar& upperb, CV_OUT Mat& dst);
//! compares elements of two arrays (dst = src1 <cmpop> src2)
CV_EXPORTS_W void compare(const Mat& src1, const Mat& src2, CV_OUT Mat& dst, int cmpop);
//! compares elements of array with scalar (dst = src1 <cmpop> src2)
CV_EXPORTS_W void compare(const Mat& src1, double s, CV_OUT Mat& dst, int cmpop);
//! computes per-element minimum of two arrays (dst = min(src1, src2))
CV_EXPORTS_W void min(const Mat& src1, const Mat& src2, CV_OUT Mat& dst);
//! computes per-element minimum of array and scalar (dst = min(src1, src2))
CV_EXPORTS_W void min(const Mat& src1, double src2, CV_OUT Mat& dst);
//! computes per-element maximum of two arrays (dst = max(src1, src2))
CV_EXPORTS_W void max(const Mat& src1, const Mat& src2, CV_OUT Mat& dst);
//! computes per-element maximum of array and scalar (dst = max(src1, src2))
CV_EXPORTS_W void max(const Mat& src1, double src2, CV_OUT Mat& dst);

//! computes square root of each matrix element (dst = src**0.5)
CV_EXPORTS_W void sqrt(const Mat& src, CV_OUT Mat& dst);
//! raises the input matrix elements to the specified power (b = a**power) 
CV_EXPORTS_W void pow(const Mat& src, double power, CV_OUT Mat& dst);
//! computes exponent of each matrix element (dst = e**src)
CV_EXPORTS_W void exp(const Mat& src, CV_OUT Mat& dst);
//! computes natural logarithm of absolute value of each matrix element: dst = log(abs(src))
CV_EXPORTS_W void log(const Mat& src, CV_OUT Mat& dst);
//! computes cube root of the argument
CV_EXPORTS_W float cubeRoot(float val);
//! computes the angle in degrees (0..360) of the vector (x,y)
CV_EXPORTS_W float fastAtan2(float y, float x);
//! converts polar coordinates to Cartesian
CV_EXPORTS_W void polarToCart(const Mat& magnitude, const Mat& angle,
                            CV_OUT Mat& x, CV_OUT Mat& y, bool angleInDegrees=false);
//! converts Cartesian coordinates to polar
CV_EXPORTS_W void cartToPolar(const Mat& x, const Mat& y,
                            CV_OUT Mat& magnitude, CV_OUT Mat& angle,
                            bool angleInDegrees=false);
//! computes angle (angle(i)) of each (x(i), y(i)) vector
CV_EXPORTS_W void phase(const Mat& x, const Mat& y, CV_OUT Mat& angle,
                        bool angleInDegrees=false);
//! computes magnitude (magnitude(i)) of each (x(i), y(i)) vector
CV_EXPORTS_W void magnitude(const Mat& x, const Mat& y, CV_OUT Mat& magnitude);
//! checks that each matrix element is within the specified range.
CV_EXPORTS_W bool checkRange(const Mat& a, bool quiet=true, CV_OUT Point* pt=0,
                            double minVal=-DBL_MAX, double maxVal=DBL_MAX);
//! implements generalized matrix product algorithm GEMM from BLAS
CV_EXPORTS_W void gemm(const Mat& src1, const Mat& src2, double alpha,
                       const Mat& src3, double gamma, CV_OUT Mat& dst, int flags=0);
//! multiplies matrix by its transposition from the left or from the right
CV_EXPORTS_W void mulTransposed( const Mat& src, CV_OUT Mat& dst, bool aTa,
                                 const Mat& delta=Mat(),
                                 double scale=1, int rtype=-1 );
//! transposes the matrix
CV_EXPORTS_W void transpose(const Mat& src, CV_OUT Mat& dst);
//! performs affine transformation of each element of multi-channel input matrix
CV_EXPORTS_W void transform(const Mat& src, CV_OUT Mat& dst, const Mat& m );
//! performs perspective transformation of each element of multi-channel input matrix
CV_EXPORTS_W void perspectiveTransform(const Mat& src, CV_OUT Mat& dst, const Mat& m );

//! extends the symmetrical matrix from the lower half or from the upper half 
CV_EXPORTS_W void completeSymm(Mat& mtx, bool lowerToUpper=false);
//! initializes scaled identity matrix
CV_EXPORTS_W void setIdentity(Mat& mtx, const Scalar& s=Scalar(1));
//! computes determinant of a square matrix
CV_EXPORTS_W double determinant(const Mat& mtx);
//! computes trace of a matrix
CV_EXPORTS_W Scalar trace(const Mat& mtx);
//! computes inverse or pseudo-inverse matrix
CV_EXPORTS_W double invert(const Mat& src, CV_OUT Mat& dst, int flags=DECOMP_LU);
//! solves linear system or a least-square problem
CV_EXPORTS_W bool solve(const Mat& src1, const Mat& src2, CV_OUT Mat& dst, int flags=DECOMP_LU);
//! sorts independently each matrix row or each matrix column
CV_EXPORTS_W void sort(const Mat& src, CV_OUT Mat& dst, int flags);
//! sorts independently each matrix row or each matrix column
CV_EXPORTS_W void sortIdx(const Mat& src, CV_OUT Mat& dst, int flags);
//! finds real roots of a cubic polynomial
CV_EXPORTS_W int solveCubic(const Mat& coeffs, CV_OUT Mat& roots);
//! finds real and complex roots of a polynomial
CV_EXPORTS_W double solvePoly(const Mat& coeffs, CV_OUT Mat& roots, int maxIters=300);
//! finds eigenvalues of a symmetric matrix
CV_EXPORTS bool eigen(const Mat& src, CV_OUT Mat& eigenvalues, int lowindex=-1,
                      int highindex=-1);
//! finds eigenvalues and eigenvectors of a symmetric matrix
CV_EXPORTS bool eigen(const Mat& src, CV_OUT Mat& eigenvalues, CV_OUT Mat& eigenvectors,
                      int lowindex=-1, int highindex=-1);
//! computes covariation matrix of a set of samples
CV_EXPORTS void calcCovarMatrix( const Mat* samples, int nsamples, Mat& covar, Mat& mean,
                                 int flags, int ctype=CV_64F);
//! computes covariation matrix of a set of samples
CV_EXPORTS_W void calcCovarMatrix( const Mat& samples, CV_OUT Mat& covar, CV_OUT Mat& mean,
                                 int flags, int ctype=CV_64F);

/*!
    Principal Component Analysis
 
    The class PCA is used to compute the special basis for a set of vectors.
    The basis will consist of eigenvectors of the covariance matrix computed
    from the input set of vectors. After PCA is performed, vectors can be transformed from
    the original high-dimensional space to the subspace formed by a few most
    prominent eigenvectors (called the principal components),
    corresponding to the largest eigenvalues of the covariation matrix.
    Thus the dimensionality of the vector and the correlation between the coordinates is reduced.
 
    The following sample is the function that takes two matrices. The first one stores the set
    of vectors (a row per vector) that is used to compute PCA, the second one stores another
    "test" set of vectors (a row per vector) that are first compressed with PCA,
    then reconstructed back and then the reconstruction error norm is computed and printed for each vector.
 
    \code
    using namespace cv;

    PCA compressPCA(const Mat& pcaset, int maxComponents,
                    const Mat& testset, Mat& compressed)
    {
        PCA pca(pcaset, // pass the data
                Mat(), // we do not have a pre-computed mean vector,
                       // so let the PCA engine to compute it
                CV_PCA_DATA_AS_ROW, // indicate that the vectors
                                    // are stored as matrix rows
                                    // (use CV_PCA_DATA_AS_COL if the vectors are
                                    // the matrix columns)
                maxComponents // specify, how many principal components to retain
                );
        // if there is no test data, just return the computed basis, ready-to-use
        if( !testset.data )
            return pca;
        CV_Assert( testset.cols == pcaset.cols );
 
        compressed.create(testset.rows, maxComponents, testset.type());
     
        Mat reconstructed;
        for( int i = 0; i < testset.rows; i++ )
        {
            Mat vec = testset.row(i), coeffs = compressed.row(i), reconstructed;
            // compress the vector, the result will be stored
            // in the i-th row of the output matrix
            pca.project(vec, coeffs);
            // and then reconstruct it
            pca.backProject(coeffs, reconstructed);
            // and measure the error
            printf("%d. diff = %g\n", i, norm(vec, reconstructed, NORM_L2));
        }
        return pca;
    }
    \endcode
*/
class CV_EXPORTS PCA
{
public:
    //! default constructor
    PCA();
    //! the constructor that performs PCA
    PCA(const Mat& data, const Mat& mean, int flags, int maxComponents=0);
    //! operator that performs PCA. The previously stored data, if any, is released
    PCA& operator()(const Mat& data, const Mat& mean, int flags, int maxComponents=0);
    //! projects vector from the original space to the principal components subspace
    Mat project(const Mat& vec) const;
    //! projects vector from the original space to the principal components subspace
    void project(const Mat& vec, CV_OUT Mat& result) const;
    //! reconstructs the original vector from the projection
    Mat backProject(const Mat& vec) const;
    //! reconstructs the original vector from the projection
    void backProject(const Mat& vec, CV_OUT Mat& result) const;

    Mat eigenvectors; //!< eigenvectors of the covariation matrix
    Mat eigenvalues; //!< eigenvalues of the covariation matrix
    Mat mean; //!< mean value subtracted before the projection and added after the back projection
};

/*!
    Singular Value Decomposition class
 
    The class is used to compute Singular Value Decomposition of a floating-point matrix and then
    use it to solve least-square problems, under-determined linear systems, invert matrices,
    compute condition numbers etc.
    
    For a bit faster operation you can pass flags=SVD::MODIFY_A|... to modify the decomposed matrix
    when it is not necessarily to preserve it. If you want to compute condition number of a matrix
    or absolute value of its determinant - you do not need SVD::u or SVD::vt,
    so you can pass flags=SVD::NO_UV|... . Another flag SVD::FULL_UV indicates that the full-size SVD::u and SVD::vt
    must be computed, which is not necessary most of the time.
*/
class CV_EXPORTS SVD
{
public:
    enum { MODIFY_A=1, NO_UV=2, FULL_UV=4 };
    //! the default constructor
    SVD();
    //! the constructor that performs SVD
    SVD( const Mat& src, int flags=0 );
    //! the operator that performs SVD. The previously allocated SVD::u, SVD::w are SVD::vt are released.
    SVD& operator ()( const Mat& src, int flags=0 );

    //! decomposes matrix and stores the results to user-provided matrices
    static void compute( const Mat& src, CV_OUT Mat& w, CV_OUT Mat& u, CV_OUT Mat& vt, int flags=0 );
    //! computes singular values of a matrix
    static void compute( const Mat& src, CV_OUT Mat& w, int flags=0 );
    //! performs back substitution
    static void backSubst( const Mat& w, const Mat& u, const Mat& vt,
                           const Mat& rhs, CV_OUT Mat& dst );
    
    template<typename _Tp, int m, int n, int nm> static void compute( const Matx<_Tp, m, n>& a,
        Matx<_Tp, nm, 1>& w, Matx<_Tp, m, nm>& u, Matx<_Tp, n, nm>& vt );
    template<typename _Tp, int m, int n, int nm> static void compute( const Matx<_Tp, m, n>& a,
        Matx<_Tp, nm, 1>& w );
    template<typename _Tp, int m, int n, int nm, int nb> static void backSubst( const Matx<_Tp, nm, 1>& w,
        const Matx<_Tp, m, nm>& u, const Matx<_Tp, n, nm>& vt, const Matx<_Tp, m, nb>& rhs, Matx<_Tp, n, nb>& dst );
    
    //! finds dst = arg min_{|dst|=1} |m*dst|
    static void solveZ( const Mat& src, CV_OUT Mat& dst );
    //! performs back substitution, so that dst is the solution or pseudo-solution of m*dst = rhs, where m is the decomposed matrix 
    void backSubst( const Mat& rhs, CV_OUT Mat& dst ) const;

    Mat u, w, vt;
};

//! computes Mahalanobis distance between two vectors: sqrt((v1-v2)'*icovar*(v1-v2)), where icovar is the inverse covariation matrix
CV_EXPORTS_W double Mahalanobis(const Mat& v1, const Mat& v2, const Mat& icovar);
//! a synonym for Mahalanobis
CV_EXPORTS double Mahalonobis(const Mat& v1, const Mat& v2, const Mat& icovar);

//! performs forward or inverse 1D or 2D Discrete Fourier Transformation
CV_EXPORTS_W void dft(const Mat& src, CV_OUT Mat& dst, int flags=0, int nonzeroRows=0);
//! performs inverse 1D or 2D Discrete Fourier Transformation
CV_EXPORTS_W void idft(const Mat& src, CV_OUT Mat& dst, int flags=0, int nonzeroRows=0);
//! performs forward or inverse 1D or 2D Discrete Cosine Transformation
CV_EXPORTS_W void dct(const Mat& src, CV_OUT Mat& dst, int flags=0);
//! performs inverse 1D or 2D Discrete Cosine Transformation
CV_EXPORTS_W void idct(const Mat& src, CV_OUT Mat& dst, int flags=0);
//! computes element-wise product of the two Fourier spectrums. The second spectrum can optionally be conjugated before the multiplication
CV_EXPORTS_W void mulSpectrums(const Mat& a, const Mat& b, CV_OUT Mat& c,
                             int flags, bool conjB=false);
//! computes the minimal vector size vecsize1 >= vecsize so that the dft() of the vector of length vecsize1 can be computed efficiently
CV_EXPORTS_W int getOptimalDFTSize(int vecsize);

/*!
 Various k-Means flags
*/
enum
{
    KMEANS_RANDOM_CENTERS=0, // Chooses random centers for k-Means initialization
    KMEANS_PP_CENTERS=2,     // Uses k-Means++ algorithm for initialization
    KMEANS_USE_INITIAL_LABELS=1 // Uses the user-provided labels for K-Means initialization
};
//! clusters the input data using k-Means algorithm
CV_EXPORTS_W double kmeans( const Mat& data, int K, CV_OUT Mat& bestLabels,
                          TermCriteria criteria, int attempts,
                          int flags, CV_OUT Mat* centers=0 );

//! returns the thread-local Random number generator
CV_EXPORTS RNG& theRNG();

//! returns the next unifomly-distributed random number of the specified type
template<typename _Tp> static inline _Tp randu() { return (_Tp)theRNG(); }

//! fills array with uniformly-distributed random numbers from the range [low, high)
CV_EXPORTS_W void randu(CV_OUT Mat& dst, const Scalar& low, const Scalar& high);
    
//! fills array with normally-distributed random numbers with the specified mean and the standard deviation
CV_EXPORTS_W void randn(CV_OUT Mat& dst, const Scalar& mean, const Scalar& stddev);

//! shuffles the input array elements
CV_EXPORTS void randShuffle(Mat& dst, double iterFactor=1., RNG* rng=0);

//! draws the line segment (pt1, pt2) in the image
CV_EXPORTS_W void line(Mat& img, Point pt1, Point pt2, const Scalar& color,
                     int thickness=1, int lineType=8, int shift=0);

//! draws the rectangle outline or a solid rectangle with the opposite corners pt1 and pt2 in the image
CV_EXPORTS_W void rectangle(Mat& img, Point pt1, Point pt2,
                          const Scalar& color, int thickness=1,
                          int lineType=8, int shift=0);
    
//! draws the rectangle outline or a solid rectangle covering rec in the image 
CV_EXPORTS void rectangle(Mat& img, Rect rec,
                          const Scalar& color, int thickness=1,
                          int lineType=8, int shift=0);

//! draws the circle outline or a solid circle in the image
CV_EXPORTS_W void circle(Mat& img, Point center, int radius,
                       const Scalar& color, int thickness=1,
                       int lineType=8, int shift=0);

//! draws an elliptic arc, ellipse sector or a rotated ellipse in the image
CV_EXPORTS_W void ellipse(Mat& img, Point center, Size axes,
                        double angle, double startAngle, double endAngle,
                        const Scalar& color, int thickness=1,
                        int lineType=8, int shift=0);

//! draws a rotated ellipse in the image
CV_EXPORTS_W void ellipse(Mat& img, const RotatedRect& box, const Scalar& color,
                        int thickness=1, int lineType=8);

//! draws a filled convex polygon in the image
CV_EXPORTS void fillConvexPoly(Mat& img, const Point* pts, int npts,
                               const Scalar& color, int lineType=8,
                               int shift=0);

//! fills an area bounded by one or more polygons
CV_EXPORTS void fillPoly(Mat& img, const Point** pts,
                         const int* npts, int ncontours,
                         const Scalar& color, int lineType=8, int shift=0,
                         Point offset=Point() );

//! draws one or more polygonal curves
CV_EXPORTS void polylines(Mat& img, const Point** pts, const int* npts,
                          int ncontours, bool isClosed, const Scalar& color,
                          int thickness=1, int lineType=8, int shift=0 );

//! clips the line segment by the rectangle Rect(0, 0, imgSize.width, imgSize.height)
CV_EXPORTS bool clipLine(Size imgSize, CV_IN_OUT Point& pt1, CV_IN_OUT Point& pt2);

//! clips the line segment by the rectangle imgRect
CV_EXPORTS_W bool clipLine(Rect imgRect, CV_IN_OUT Point& pt1, CV_IN_OUT Point& pt2);

/*!
   Line iterator class
 
   The class is used to iterate over all the pixels on the raster line
   segment connecting two specified points.
*/
class CV_EXPORTS LineIterator
{
public:
    //! intializes the iterator
    LineIterator( const Mat& img, Point pt1, Point pt2,
                  int connectivity=8, bool leftToRight=false );
    //! returns pointer to the current pixel
    uchar* operator *();
    //! prefix increment operator (++it). shifts iterator to the next pixel
    LineIterator& operator ++();
    //! postfix increment operator (it++). shifts iterator to the next pixel
    LineIterator operator ++(int);
    //! returns coordinates of the current pixel
    Point pos() const;

    uchar* ptr;
    const uchar* ptr0;
    int step, elemSize;
    int err, count;
    int minusDelta, plusDelta;
    int minusStep, plusStep;
};

//! converts elliptic arc to a polygonal curve
CV_EXPORTS_W void ellipse2Poly( Point center, Size axes, int angle,
                              int arcStart, int arcEnd, int delta,
                              CV_OUT vector<Point>& pts );

enum
{
    FONT_HERSHEY_SIMPLEX = 0,
    FONT_HERSHEY_PLAIN = 1,
    FONT_HERSHEY_DUPLEX = 2,
    FONT_HERSHEY_COMPLEX = 3,
    FONT_HERSHEY_TRIPLEX = 4,
    FONT_HERSHEY_COMPLEX_SMALL = 5,
    FONT_HERSHEY_SCRIPT_SIMPLEX = 6,
    FONT_HERSHEY_SCRIPT_COMPLEX = 7,
    FONT_ITALIC = 16
};

//! renders text string in the image
CV_EXPORTS_W void putText( Mat& img, const string& text, Point org,
                         int fontFace, double fontScale, Scalar color,
                         int thickness=1, int linetype=8,
                         bool bottomLeftOrigin=false );

//! returns bounding box of the text string
CV_EXPORTS_W Size getTextSize(const string& text, int fontFace,
                            double fontScale, int thickness,
                            CV_OUT int* baseLine);

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
template<typename _Tp> class CV_EXPORTS Mat_ : public Mat
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
    //! makes a matrix out of Vec, std::vector, Point_ or Point3_. The matrix will have a single column
    explicit Mat_(const vector<_Tp>& vec, bool copyData=false);
    template<int n> explicit Mat_(const Vec<typename DataType<_Tp>::channel_type, n>& vec, bool copyData=true);
    template<int m, int n> explicit Mat_(const Matx<typename DataType<_Tp>::channel_type, m, n>& mtx, bool copyData=true);
    explicit Mat_(const Point_<typename DataType<_Tp>::channel_type>& pt, bool copyData=true);
    explicit Mat_(const Point3_<typename DataType<_Tp>::channel_type>& pt, bool copyData=true);
    explicit Mat_(const MatCommaInitializer_<_Tp>& commaInitializer);

    Mat_& operator = (const Mat& m);
    Mat_& operator = (const Mat_& m);
    //! set all the elements to s.
    Mat_& operator = (const _Tp& s);

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
    //! to support complex matrix expressions
    Mat_& operator = (const MatExpr& expr);
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
    Mat_ reshape(int _rows) const;
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
    operator vector<_Tp>() const;
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

//////////// Iterators & Comma initializers //////////////////

class CV_EXPORTS MatConstIterator
{
public:
    typedef uchar* value_type;
    typedef ptrdiff_t difference_type;
    typedef const uchar** pointer;
    typedef uchar* reference;
    typedef std::random_access_iterator_tag iterator_category;
    
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
    void seek(ptrdiff_t ofs, bool relative=false);
    void seek(const int* _idx, bool relative=false);
    
    const Mat* m;
    size_t elemSize;
    uchar* ptr;
    uchar* sliceStart;
    uchar* sliceEnd;
};
    
/*!
 Matrix read-only iterator
 
 */    
template<typename _Tp>
class CV_EXPORTS MatConstIterator_ : public MatConstIterator
{
public:
    typedef _Tp value_type;
    typedef ptrdiff_t difference_type;
    typedef const _Tp* pointer;
    typedef const _Tp& reference;
    typedef std::random_access_iterator_tag iterator_category;

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


/*!
 Matrix read-write iterator
 
*/
template<typename _Tp>
class CV_EXPORTS MatIterator_ : public MatConstIterator_<_Tp>
{
public:
    typedef _Tp* pointer;
    typedef _Tp& reference;
    typedef std::random_access_iterator_tag iterator_category;

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

template<typename _Tp> class CV_EXPORTS MatOp_Iter_;

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
template<typename _Tp> class CV_EXPORTS MatCommaInitializer_
{
public:
    //! the constructor, created by "matrix << firstValue" operator, where matrix is cv::Mat
    MatCommaInitializer_(Mat_<_Tp>* _m);
    //! the operator that takes the next value and put it to the matrix
    template<typename T2> MatCommaInitializer_<_Tp>& operator , (T2 v);
    //! another form of conversion operator
    Mat_<_Tp> operator *() const;
    operator Mat_<_Tp>() const;
protected:
    MatIterator_<_Tp> it;
};


template<typename _Tp, int m, int n> class CV_EXPORTS MatxCommaInitializer
{
public:
    MatxCommaInitializer(Matx<_Tp, m, n>* _mtx);
    template<typename T2> MatxCommaInitializer<_Tp, m, n>& operator , (T2 val);
    Matx<_Tp, m, n> operator *() const;

    Matx<_Tp, m, n>* dst;
    int idx;
};

template<typename _Tp, int m> class CV_EXPORTS VecCommaInitializer : public MatxCommaInitializer<_Tp, m, 1>
{
public:
    VecCommaInitializer(Vec<_Tp, m>* _vec);
    template<typename T2> VecCommaInitializer<_Tp, m>& operator , (T2 val);
    Vec<_Tp, m> operator *() const;
};
        
/*!
 Automatically Allocated Buffer Class
 
 The class is used for temporary buffers in functions and methods.
 If a temporary buffer is usually small (a few K's of memory),
 but its size depends on the parameters, it makes sense to create a small
 fixed-size array on stack and use it if it's large enough. If the required buffer size
 is larger than the fixed size, another buffer of sufficient size is allocated dynamically
 and released after the processing. Therefore, in typical cases, when the buffer size is small,
 there is no overhead associated with malloc()/free().
 At the same time, there is no limit on the size of processed data.
 
 This is what AutoBuffer does. The template takes 2 parameters - type of the buffer elements and
 the number of stack-allocated elements. Here is how the class is used:
 
 \code
 void my_func(const cv::Mat& m)
 {
    cv::AutoBuffer<float, 1000> buf; // create automatic buffer containing 1000 floats
 
    buf.allocate(m.rows); // if m.rows <= 1000, the pre-allocated buffer is used,
                          // otherwise the buffer of "m.rows" floats will be allocated
                          // dynamically and deallocated in cv::AutoBuffer destructor
    ...
 }
 \endcode
*/
template<typename _Tp, size_t fixed_size=4096/sizeof(_Tp)+8> class CV_EXPORTS AutoBuffer
{
public:
    typedef _Tp value_type;

    //! the default contructor
    AutoBuffer();
    //! constructor taking the real buffer size
    AutoBuffer(size_t _size);
    //! destructor. calls deallocate() 
    ~AutoBuffer();

    //! allocates the new buffer of size _size. if the _size is small enough, stack-allocated buffer is used 
    void allocate(size_t _size);
    //! deallocates the buffer if it was dynamically allocated
    void deallocate();
    //! returns pointer to the real buffer, stack-allocated or head-allocated
    operator _Tp* ();
    //! returns read-only pointer to the real buffer, stack-allocated or head-allocated
    operator const _Tp* () const;

protected:
    //! pointer to the real buffer, can point to buf if the buffer is small enough
    _Tp* ptr;
    //! size of the real buffer
    size_t size;
    //! pre-allocated buffer
    _Tp buf[fixed_size];
};

/////////////////////////// multi-dimensional dense matrix //////////////////////////

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
    NAryMatIterator(const Mat** arrays, Mat* planes, int narrays=-1);
    //! the separate iterator initialization method
    void init(const Mat** arrays, Mat* planes, int narrays=-1);

    //! proceeds to the next plane of every iterated matrix 
    NAryMatIterator& operator ++();
    //! proceeds to the next plane of every iterated matrix (postfix increment operator)
    NAryMatIterator operator ++(int);

    //! the iterated arrays
    const Mat** arrays;
    //! the current planes
    Mat* planes;
    //! the number of arrays
    int narrays;
    //! the number of planes in each array
    int nplanes;
protected:
    int iterdepth, idx;
};
    
//typedef NAryMatIterator NAryMatNDIterator;

typedef void (*ConvertData)(const void* from, void* to, int cn);
typedef void (*ConvertScaleData)(const void* from, void* to, int cn, double alpha, double beta);

//! returns the function for converting pixels from one data type to another
CV_EXPORTS ConvertData getConvertElem(int fromType, int toType);
//! returns the function for converting pixels from one data type to another with the optional scaling
CV_EXPORTS ConvertScaleData getConvertScaleElem(int fromType, int toType);

    
/////////////////////////// multi-dimensional sparse matrix //////////////////////////

class SparseMatIterator;
class SparseMatConstIterator;
template<typename _Tp> class SparseMatIterator_;
template<typename _Tp> class SparseMatConstIterator_;

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
        vector<uchar> pool;
        vector<size_t> hashtab;
        int size[CV_MAX_DIM];
    };

    //! sparse matrix node - element of a hash table
    struct CV_EXPORTS Node
    {
        //! hash value
        size_t hashval;
        //! index of the next node in the same hash table entry
        size_t next;
        //! index of the matrix element
        int idx[CV_MAX_DIM];
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
     \param try1d if true and m is a single-column matrix (Nx1),
            then the sparse matrix will be 1-dimensional.
    */ 
    SparseMat(const Mat& m);
    //! converts old-style sparse matrix to the new-style. All the data is copied
    SparseMat(const CvSparseMat* m);
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
    operator CvSparseMat*() const;
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

    enum { MAGIC_VAL=0x42FD0000, MAX_DIM=CV_MAX_DIM, HASH_SCALE=0x5bd1e995, HASH_BIT=0x80000000 };

    int flags;
    Hdr* hdr;
};

//! finds global minimum and maximum sparse array elements and returns their values and their locations
CV_EXPORTS void minMaxLoc(const SparseMat& a, double* minVal,
                          double* maxVal, int* minIdx=0, int* maxIdx=0);
//! computes norm of a sparse matrix
CV_EXPORTS double norm( const SparseMat& src, int normType );
//! scales and shifts array elements so that either the specified norm (alpha) or the minimum (alpha) and maximum (beta) array values get the specified values 
CV_EXPORTS void normalize( const SparseMat& src, SparseMat& dst, double alpha, int normType );

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
template<typename _Tp> class CV_EXPORTS SparseMat_ : public SparseMat
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
    SparseMat_(const CvSparseMat* m);
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
    operator CvSparseMat*() const;

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


/*!
 Template Read-Only Sparse Matrix Iterator Class.
 
 This is the derived from SparseMatConstIterator class that
 introduces more convenient operator *() for accessing the current element.
*/
template<typename _Tp> class CV_EXPORTS SparseMatConstIterator_ : public SparseMatConstIterator
{
public:
    typedef std::forward_iterator_tag iterator_category;
    
    //! the default constructor
    SparseMatConstIterator_();
    //! the full constructor setting the iterator to the first sparse matrix element
    SparseMatConstIterator_(const SparseMat_<_Tp>* _m);
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

/*!
 Template Read-Write Sparse Matrix Iterator Class.
 
 This is the derived from cv::SparseMatConstIterator_ class that
 introduces more convenient operator *() for accessing the current element.
*/
template<typename _Tp> class CV_EXPORTS SparseMatIterator_ : public SparseMatConstIterator_<_Tp>
{
public:
    typedef std::forward_iterator_tag iterator_category;
    
    //! the default constructor
    SparseMatIterator_();
    //! the full constructor setting the iterator to the first sparse matrix element
    SparseMatIterator_(SparseMat_<_Tp>* _m);
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

//////////////////// Fast Nearest-Neighbor Search Structure ////////////////////

/*!
 Fast Nearest Neighbor Search Class.
 
 The class implements D. Lowe BBF (Best-Bin-First) algorithm for the last
 approximate (or accurate) nearest neighbor search in multi-dimensional spaces.
 
 First, a set of vectors is passed to KDTree::KDTree() constructor
 or KDTree::build() method, where it is reordered.
 
 Then arbitrary vectors can be passed to KDTree::findNearest() methods, which
 find the K nearest neighbors among the vectors from the initial set.
 The user can balance between the speed and accuracy of the search by varying Emax
 parameter, which is the number of leaves that the algorithm checks.
 The larger parameter values yield more accurate results at the expense of lower processing speed.
 
 \code
 KDTree T(points, false);
 const int K = 3, Emax = INT_MAX;
 int idx[K];
 float dist[K];
 T.findNearest(query_vec, K, Emax, idx, 0, dist);
 CV_Assert(dist[0] <= dist[1] && dist[1] <= dist[2]);
 \endcode
*/ 
class CV_EXPORTS_W KDTree
{
public:
    /*!
        The node of the search tree.
    */  
    struct Node
    {
        Node() : idx(-1), left(-1), right(-1), boundary(0.f) {}
        Node(int _idx, int _left, int _right, float _boundary)
            : idx(_idx), left(_left), right(_right), boundary(_boundary) {}
        //! split dimension; >=0 for nodes (dim), < 0 for leaves (index of the point)
        int idx;
        //! node indices of the left and the right branches
        int left, right;
        //! go to the left if query_vec[node.idx]<=node.boundary, otherwise go to the right
        float boundary;
    };

    //! the default constructor
    CV_WRAP KDTree();
    //! the full constructor that builds the search tree
    CV_WRAP KDTree(const Mat& _points, bool copyAndReorderPoints=false);
    //! the full constructor that builds the search tree
    CV_WRAP KDTree(const Mat& _points, const Mat& _labels, bool copyAndReorderPoints=false);
    //! builds the search tree
    CV_WRAP void build(const Mat& _points, bool copyAndReorderPoints=false);
    //! builds the search tree
    CV_WRAP void build(const Mat& _points, const Mat& _labels, bool copyAndReorderPoints=false);
    //! finds the K nearest neighbors of "vec" while looking at Emax (at most) leaves
    int findNearest(const float* vec,
                    int K, int Emax, int* neighborsIdx,
                    Mat* neighbors=0, float* dist=0, int* labels=0) const;
    //! finds the K nearest neighbors while looking at Emax (at most) leaves
    int findNearest(const float* vec, int K, int Emax,
                    vector<int>* neighborsIdx,
                    Mat* neighbors=0,
                    vector<float>* dist=0,
                    vector<int>* labels=0) const;
    CV_WRAP int findNearest(const vector<float>& vec, int K, int Emax,
                    CV_OUT vector<int>* neighborsIdx,
                    CV_OUT Mat* neighbors=0,
                    CV_OUT vector<float>* dist=0,
                    CV_OUT vector<int>* labels=0) const;
    //! finds all the points from the initial set that belong to the specified box 
    void findOrthoRange(const float* minBounds, const float* maxBounds,
                        vector<int>* neighborsIdx, Mat* neighbors=0,
                        vector<int>* labels=0) const;
    CV_WRAP void findOrthoRange(const vector<float>& minBounds, const vector<float>& maxBounds,
                                CV_OUT vector<int>* neighborsIdx, CV_OUT Mat* neighbors=0,
                                CV_OUT vector<int>* labels=0) const;
    //! returns vectors with the specified indices
    void getPoints(const int* idx, size_t nidx, Mat& pts, vector<int>* labels=0) const;
    //! returns vectors with the specified indices
    CV_WRAP void getPoints(const vector<int>& idxs, Mat& pts, CV_OUT vector<int>* labels=0) const;
    //! return a vector with the specified index
    const float* getPoint(int ptidx, int* label=0) const;
    //! returns the search space dimensionality
    CV_WRAP int dims() const;

    vector<Node> nodes; //!< all the tree nodes
    CV_PROP Mat points; //!< all the points. It can be a reordered copy of the input vector set or the original vector set.
    CV_PROP vector<int> labels; //!< the parallel array of labels.
    CV_PROP int maxDepth; //!< maximum depth of the search tree. Do not modify it
    CV_PROP_RW int normType; //!< type of the distance (cv::NORM_L1 or cv::NORM_L2) used for search. Initially set to cv::NORM_L2, but you can modify it
};

//////////////////////////////////////// XML & YAML I/O ////////////////////////////////////

class CV_EXPORTS FileNode;

/*!
 XML/YAML File Storage Class.
 
 The class describes an object associated with XML or YAML file.
 It can be used to store data to such a file or read and decode the data.
 
 The storage is organized as a tree of nested sequences (or lists) and mappings.
 Sequence is a heterogenious array, which elements are accessed by indices or sequentially using an iterator.
 Mapping is analogue of std::map or C structure, which elements are accessed by names.
 The most top level structure is a mapping.
 Leaves of the file storage tree are integers, floating-point numbers and text strings. 
 
 For example, the following code:
 
 \code
 // open file storage for writing. Type of the file is determined from the extension
 FileStorage fs("test.yml", FileStorage::WRITE);
 fs << "test_int" << 5 << "test_real" << 3.1 << "test_string" << "ABCDEFGH";
 fs << "test_mat" << Mat::eye(3,3,CV_32F);
 
 fs << "test_list" << "[" << 0.0000000000001 << 2 << CV_PI << -3435345 << "2-502 2-029 3egegeg" <<
 "{:" << "month" << 12 << "day" << 31 << "year" << 1969 << "}" << "]";
 fs << "test_map" << "{" << "x" << 1 << "y" << 2 << "width" << 100 << "height" << 200 << "lbp" << "[:";
 
 const uchar arr[] = {0, 1, 1, 0, 1, 1, 0, 1};
 fs.writeRaw("u", arr, (int)(sizeof(arr)/sizeof(arr[0])));
 
 fs << "]" << "}";
 \endcode
 
 will produce the following file:

 \verbatim
 %YAML:1.0
 test_int: 5
 test_real: 3.1000000000000001e+00
 test_string: ABCDEFGH
 test_mat: !!opencv-matrix
     rows: 3
     cols: 3
     dt: f
     data: [ 1., 0., 0., 0., 1., 0., 0., 0., 1. ]
 test_list:
     - 1.0000000000000000e-13
     - 2
     - 3.1415926535897931e+00
     - -3435345
     - "2-502 2-029 3egegeg"
     - { month:12, day:31, year:1969 }
 test_map:
     x: 1
     y: 2
     width: 100
     height: 200
     lbp: [ 0, 1, 1, 0, 1, 1, 0, 1 ]
 \endverbatim
 
 and to read the file above, the following code can be used:
 
 \code
 // open file storage for reading.
 // Type of the file is determined from the content, not the extension
 FileStorage fs("test.yml", FileStorage::READ);
 int test_int = (int)fs["test_int"];
 double test_real = (double)fs["test_real"];
 string test_string = (string)fs["test_string"];
 
 Mat M;
 fs["test_mat"] >> M;
 
 FileNode tl = fs["test_list"];
 CV_Assert(tl.type() == FileNode::SEQ && tl.size() == 6);
 double tl0 = (double)tl[0];
 int tl1 = (int)tl[1];
 double tl2 = (double)tl[2];
 int tl3 = (int)tl[3];
 string tl4 = (string)tl[4];
 CV_Assert(tl[5].type() == FileNode::MAP && tl[5].size() == 3);
 
 int month = (int)tl[5]["month"];
 int day = (int)tl[5]["day"];
 int year = (int)tl[5]["year"];
 
 FileNode tm = fs["test_map"];
 
 int x = (int)tm["x"];
 int y = (int)tm["y"];
 int width = (int)tm["width"];
 int height = (int)tm["height"];
  
 int lbp_val = 0;
 FileNodeIterator it = tm["lbp"].begin();

 for(int k = 0; k < 8; k++, ++it)
    lbp_val |= ((int)*it) << k;
 \endcode
*/
class CV_EXPORTS_W FileStorage
{
public:
    //! file storage mode
    enum
    {
        READ=0, //! read mode
        WRITE=1, //! write mode
        APPEND=2 //! append mode
    };
    enum
    {
        UNDEFINED=0, 
        VALUE_EXPECTED=1,
        NAME_EXPECTED=2,
        INSIDE_MAP=4
    };
    //! the default constructor
    CV_WRAP FileStorage();
    //! the full constructor that opens file storage for reading or writing
    CV_WRAP FileStorage(const string& filename, int flags);
    //! the constructor that takes pointer to the C FileStorage structure 
    FileStorage(CvFileStorage* fs);
    //! the destructor. calls release()
    virtual ~FileStorage();

    //! opens file storage for reading or writing. The previous storage is closed with release()
    CV_WRAP virtual bool open(const string& filename, int flags);
    //! returns true if the object is associated with currently opened file.
    CV_WRAP virtual bool isOpened() const;
    //! closes the file and releases all the memory buffers
    CV_WRAP virtual void release();

    //! returns the first element of the top-level mapping
    CV_WRAP FileNode getFirstTopLevelNode() const;
    //! returns the top-level mapping. YAML supports multiple streams
    CV_WRAP FileNode root(int streamidx=0) const;
    //! returns the specified element of the top-level mapping
    FileNode operator[](const string& nodename) const;
    //! returns the specified element of the top-level mapping
    CV_WRAP FileNode operator[](const char* nodename) const;

    //! returns pointer to the underlying C FileStorage structure
    CvFileStorage* operator *() { return fs; }
    //! returns pointer to the underlying C FileStorage structure
    const CvFileStorage* operator *() const { return fs; }
    //! writes one or more numbers of the specified format to the currently written structure
    void writeRaw( const string& fmt, const uchar* vec, size_t len );
    //! writes the registered C structure (CvMat, CvMatND, CvSeq). See cvWrite()
    void writeObj( const string& name, const void* obj );

    //! returns the normalized object name for the specified file name
    static string getDefaultObjectName(const string& filename);

    Ptr<CvFileStorage> fs; //!< the underlying C FileStorage structure
    string elname; //!< the currently written element
    vector<char> structs; //!< the stack of written structures
    int state; //!< the writer state
};

class CV_EXPORTS FileNodeIterator;

/*!
 File Storage Node class
 
 The node is used to store each and every element of the file storage opened for reading -
 from the primitive objects, such as numbers and text strings, to the complex nodes:
 sequences, mappings and the registered objects.
 
 Note that file nodes are only used for navigating file storages opened for reading.
 When a file storage is opened for writing, no data is stored in memory after it is written.
*/
class CV_EXPORTS_W_SIMPLE FileNode
{
public:
    //! type of the file storage node
    enum
    {
        NONE=0, //!< empty node
        INT=1, //!< an integer
        REAL=2, //!< floating-point number
        FLOAT=REAL, //!< synonym or REAL
        STR=3, //!< text string in UTF-8 encoding
        STRING=STR, //!< synonym for STR
        REF=4, //!< integer of size size_t. Typically used for storing complex dynamic structures where some elements reference the others 
        SEQ=5, //!< sequence
        MAP=6, //!< mapping
        TYPE_MASK=7,
        FLOW=8, //!< compact representation of a sequence or mapping. Used only by YAML writer
        USER=16, //!< a registered object (e.g. a matrix)
        EMPTY=32, //!< empty structure (sequence or mapping)
        NAMED=64 //!< the node has a name (i.e. it is element of a mapping)
    };
    //! the default constructor
    CV_WRAP FileNode();
    //! the full constructor wrapping CvFileNode structure.
    FileNode(const CvFileStorage* fs, const CvFileNode* node);
    //! the copy constructor
    FileNode(const FileNode& node);
    //! returns element of a mapping node
    FileNode operator[](const string& nodename) const;
    //! returns element of a mapping node
    CV_WRAP FileNode operator[](const char* nodename) const;
    //! returns element of a sequence node
    CV_WRAP FileNode operator[](int i) const;
    //! returns type of the node
    CV_WRAP int type() const;

    //! returns true if the node is empty 
    CV_WRAP bool empty() const;
    //! returns true if the node is a "none" object 
    CV_WRAP bool isNone() const;
    //! returns true if the node is a sequence
    CV_WRAP bool isSeq() const;
    //! returns true if the node is a mapping
    CV_WRAP bool isMap() const;
    //! returns true if the node is an integer
    CV_WRAP bool isInt() const;
    //! returns true if the node is a floating-point number
    CV_WRAP bool isReal() const;
    //! returns true if the node is a text string
    CV_WRAP bool isString() const;
    //! returns true if the node has a name
    CV_WRAP bool isNamed() const;
    //! returns the node name or an empty string if the node is nameless
    CV_WRAP string name() const;
    //! returns the number of elements in the node, if it is a sequence or mapping, or 1 otherwise.
    CV_WRAP size_t size() const;
    //! returns the node content as an integer. If the node stores floating-point number, it is rounded.
    operator int() const;
    //! returns the node content as float
    operator float() const;
    //! returns the node content as double
    operator double() const;
    //! returns the node content as text string
    operator string() const;
    
    //! returns pointer to the underlying file node
    CvFileNode* operator *();
    //! returns pointer to the underlying file node
    const CvFileNode* operator* () const;

    //! returns iterator pointing to the first node element
    FileNodeIterator begin() const;
    //! returns iterator pointing to the element following the last node element
    FileNodeIterator end() const;

    //! reads node elements to the buffer with the specified format
    void readRaw( const string& fmt, uchar* vec, size_t len ) const;
    //! reads the registered object and returns pointer to it
    void* readObj() const;

    // do not use wrapper pointer classes for better efficiency
    const CvFileStorage* fs;
    const CvFileNode* node;
};


/*!
 File Node Iterator
 
 The class is used for iterating sequences (usually) and mappings.
 */
class CV_EXPORTS FileNodeIterator
{
public:
    //! the default constructor
    FileNodeIterator();
    //! the full constructor set to the ofs-th element of the node
    FileNodeIterator(const CvFileStorage* fs, const CvFileNode* node, size_t ofs=0);
    //! the copy constructor
    FileNodeIterator(const FileNodeIterator& it);
    //! returns the currently observed element
    FileNode operator *() const;
    //! accesses the currently observed element methods
    FileNode operator ->() const;

    //! moves iterator to the next node
    FileNodeIterator& operator ++ ();
    //! moves iterator to the next node
    FileNodeIterator operator ++ (int);
    //! moves iterator to the previous node
    FileNodeIterator& operator -- ();
    //! moves iterator to the previous node
    FileNodeIterator operator -- (int);
    //! moves iterator forward by the specified offset (possibly negative)
    FileNodeIterator& operator += (int);
    //! moves iterator backward by the specified offset (possibly negative)
    FileNodeIterator& operator -= (int);

    //! reads the next maxCount elements (or less, if the sequence/mapping last element occurs earlier) to the buffer with the specified format
    FileNodeIterator& readRaw( const string& fmt, uchar* vec,
                               size_t maxCount=(size_t)INT_MAX );
    
    const CvFileStorage* fs;
    const CvFileNode* container;
    CvSeqReader reader;
    size_t remaining;
};

////////////// convenient wrappers for operating old-style dynamic structures //////////////

template<typename _Tp> class SeqIterator;

typedef Ptr<CvMemStorage> MemStorage;

/*!
 Template Sequence Class derived from CvSeq

 The class provides more convenient access to sequence elements,
 STL-style operations and iterators.
 
 \note The class is targeted for simple data types,
    i.e. no constructors or destructors
    are called for the sequence elements.
*/
template<typename _Tp> class CV_EXPORTS Seq
{
public:
    typedef SeqIterator<_Tp> iterator;
    typedef SeqIterator<_Tp> const_iterator;
    
    //! the default constructor
    Seq();
    //! the constructor for wrapping CvSeq structure. The real element type in CvSeq should match _Tp.
    Seq(const CvSeq* seq);
    //! creates the empty sequence that resides in the specified storage
    Seq(MemStorage& storage, int headerSize = sizeof(CvSeq));
    //! returns read-write reference to the specified element
    _Tp& operator [](int idx);
    //! returns read-only reference to the specified element
    const _Tp& operator[](int idx) const;
    //! returns iterator pointing to the beginning of the sequence
    SeqIterator<_Tp> begin() const;
    //! returns iterator pointing to the element following the last sequence element
    SeqIterator<_Tp> end() const;
    //! returns the number of elements in the sequence
    size_t size() const;
    //! returns the type of sequence elements (CV_8UC1 ... CV_64FC(CV_CN_MAX) ...)
    int type() const;
    //! returns the depth of sequence elements (CV_8U ... CV_64F)
    int depth() const;
    //! returns the number of channels in each sequence element
    int channels() const;
    //! returns the size of each sequence element
    size_t elemSize() const;
    //! returns index of the specified sequence element
    size_t index(const _Tp& elem) const;
    //! appends the specified element to the end of the sequence
    void push_back(const _Tp& elem);
    //! appends the specified element to the front of the sequence
    void push_front(const _Tp& elem);
    //! appends zero or more elements to the end of the sequence
    void push_back(const _Tp* elems, size_t count);
    //! appends zero or more elements to the front of the sequence
    void push_front(const _Tp* elems, size_t count);
    //! inserts the specified element to the specified position
    void insert(int idx, const _Tp& elem);
    //! inserts zero or more elements to the specified position
    void insert(int idx, const _Tp* elems, size_t count);
    //! removes element at the specified position
    void remove(int idx);
    //! removes the specified subsequence
    void remove(const Range& r);
    
    //! returns reference to the first sequence element
    _Tp& front();
    //! returns read-only reference to the first sequence element
    const _Tp& front() const;
    //! returns reference to the last sequence element
    _Tp& back();
    //! returns read-only reference to the last sequence element
    const _Tp& back() const;
    //! returns true iff the sequence contains no elements
    bool empty() const;

    //! removes all the elements from the sequence
    void clear();
    //! removes the first element from the sequence
    void pop_front();
    //! removes the last element from the sequence
    void pop_back();
    //! removes zero or more elements from the beginning of the sequence
    void pop_front(_Tp* elems, size_t count);
    //! removes zero or more elements from the end of the sequence
    void pop_back(_Tp* elems, size_t count);

    //! copies the whole sequence or the sequence slice to the specified vector
    void copyTo(vector<_Tp>& vec, const Range& range=Range::all()) const;
    //! returns the vector containing all the sequence elements
    operator vector<_Tp>() const;
    
    CvSeq* seq;
};

    
/*!
 STL-style Sequence Iterator inherited from the CvSeqReader structure
*/
template<typename _Tp> class CV_EXPORTS SeqIterator : public CvSeqReader
{
public:
    //! the default constructor
    SeqIterator();
    //! the constructor setting the iterator to the beginning or to the end of the sequence
    SeqIterator(const Seq<_Tp>& seq, bool seekEnd=false);
    //! positions the iterator within the sequence
    void seek(size_t pos);
    //! reports the current iterator position
    size_t tell() const;
    //! returns reference to the current sequence element
    _Tp& operator *();
    //! returns read-only reference to the current sequence element
    const _Tp& operator *() const;
    //! moves iterator to the next sequence element
    SeqIterator& operator ++();
    //! moves iterator to the next sequence element
    SeqIterator operator ++(int) const;
    //! moves iterator to the previous sequence element
    SeqIterator& operator --();
    //! moves iterator to the previous sequence element
    SeqIterator operator --(int) const;

    //! moves iterator forward by the specified offset (possibly negative)
    SeqIterator& operator +=(int);
    //! moves iterator backward by the specified offset (possibly negative)
    SeqIterator& operator -=(int);

    // this is index of the current element module seq->total*2
    // (to distinguish between 0 and seq->total)
    int index;
};

    
class CV_EXPORTS AlgorithmImpl;

/*!
  Base class for high-level OpenCV algorithms
*/    
class CV_EXPORTS Algorithm
{
public:
    virtual ~Algorithm();
    virtual string name() const;
    
    template<typename _Tp> _Tp get(int paramId) const;
    template<typename _Tp> bool set(int paramId, const _Tp& value);
    string paramName(int paramId) const;
    string paramHelp(int paramId) const;
    int paramType(int paramId) const;
    int findParam(const string& name) const;
    template<typename _Tp> _Tp paramDefaultValue(int paramId) const;
    template<typename _Tp> bool paramRange(int paramId, _Tp& minVal, _Tp& maxVal) const;
    
    virtual void getParams(vector<int>& ids) const;
    virtual void write(vector<uchar>& buf) const;
    virtual bool read(const vector<uchar>& buf);
    
    typedef Algorithm* (*Constructor)(void);
    static void add(const string& name, Constructor create);
    static void getList(vector<string>& algorithms);
    static Ptr<Algorithm> create(const string& name);
    
protected:
    template<typename _Tp> void addParam(int propId, _Tp& value, bool readOnly, const string& name,
                                         const string& help=string(), const _Tp& defaultValue=_Tp(),
                                         _Tp (Algorithm::*getter)()=0, bool (Algorithm::*setter)(const _Tp&)=0);
    template<typename _Tp> void setParamRange(int propId, const _Tp& minVal, const _Tp& maxVal);
    
    bool set_(int paramId, int argType, const void* value);
    void get_(int paramId, int argType, void* value);
    void paramDefaultValue_(int paramId, int argType, void* value);
    void paramRange_(int paramId, int argType, void* minval, void* maxval);
    void addParam_(int propId, int argType, void* value, bool readOnly, const string& name,
                  const string& help, const void* defaultValue, void* getter, void* setter);
    void setParamRange_(int propId, int argType, const void* minVal, const void* maxVal);
    
    Ptr<AlgorithmImpl> impl;
};
    
}

#endif // __cplusplus

#include "opencv2/core/operations.hpp"
#include "opencv2/core/mat.hpp"

#endif /*__OPENCV_CORE_HPP__*/
