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

typedef std::string String;
typedef std::basic_string<wchar_t> WString;

CV_EXPORTS string fromUtf16(const WString& str);
CV_EXPORTS WString toUtf16(const string& str);

CV_EXPORTS string format( const char* fmt, ... );

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


/*!
  \fn void error(const Exception& exc);
  \brief Signals an error and raises the exception.
 
  By default the function prints information about the error to stderr,
  then it either stops if setBreakOnError() had been called before or raises the exception.
  It is possible to alternate error processing by using redirectError().
 
  \param exc the exception raisen.
 */
CV_EXPORTS void error( const Exception& exc );

/*!
  \fn bool setBreakOnError(bool flag);
  \brief Sets/resets the break-on-error mode.
 
  When the break-on-error mode is set, the default error handler
  issues a hardware exception, which can make debugging more convenient.
 
  \return the previous state
 */
CV_EXPORTS bool setBreakOnError(bool flag);
    
typedef int (CV_CDECL *ErrorCallback)( int status, const char* func_name,
                                       const char* err_msg, const char* file_name,
                                       int line, void* userdata );

/*!
  \fn ErrorCallback redirectError( ErrorCallback errCallback,
                                   void* userdata=0, void** prevUserdata=0);
  \brief Sets the new error handler and the optional user data.

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
#define CV_Assert( expr ) { if(!(expr)) cv::error( cv::Exception(CV_StsAssert, #expr, __func__, __FILE__, __LINE__) ); }
#else
#define CV_Error( code, msg ) cv::error( cv::Exception(code, msg, "", __FILE__, __LINE__) )
#define CV_Error_( code, args ) cv::error( cv::Exception(code, cv::format args, "", __FILE__, __LINE__) )
#define CV_Assert( expr ) { if(!(expr)) cv::error( cv::Exception(CV_StsAssert, #expr, "", __FILE__, __LINE__) ); }
#endif
    
#ifdef _DEBUG
#define CV_DbgAssert(expr) CV_Assert(expr)
#else
#define CV_DbgAssert(expr)
#endif

CV_EXPORTS void setNumThreads(int);
CV_EXPORTS int getNumThreads();
CV_EXPORTS int getThreadNum();

/*!
  \fn int64 getTickCount();
  \brief Returns the number of ticks.

  The function returns the number of ticks since the certain event (e.g. when the machine was turned on).
  It can be used to initialize cv::RNG or to measure a function execution time by reading the tick count
  before and after the function call. The granularity of ticks depends on the hardware and OS used. Use
  cv::getTickFrequency() to convert ticks to seconds.
*/
CV_EXPORTS int64 getTickCount();

/*!
  \fn int64 getTickCount();
  \brief Returns the number of ticks per seconds.

  The function returns the number of ticks (as returned by cv::getTickCount()) per second.
  The following code computes the execution time in milliseconds:
  
  \code
  double exec_time = (double)getTickCount();
  // do something ...
  exec_time = ((double)getTickCount() - exec_time)*1000./getTickFrequency();
  \endcode
*/
CV_EXPORTS double getTickFrequency();

/*!
  \fn int64 getCPUTickCount();
  \brief Returns the number of CPU ticks.

  On platforms where the feature is available, the function returns the number of CPU ticks
  since the certain event (normally, the system power-on moment). Using this function
  one can accurately measure the execution time of very small code fragments,
  for which cv::getTickCount() granularity is not enough.
*/
CV_EXPORTS int64 getCPUTickCount();

/*!
  \fn bool checkHardwareSupport(int feature);
  \brief Returns SSE etc. support status
  
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
CV_EXPORTS bool checkHardwareSupport(int feature);

/*!
  \fn void* fastMalloc(size_t bufSize);
  \brief allocates memory buffer
  
  This is specialized OpenCV memory allocation function that returns properly aligned memory buffers.
  The usage is identical to malloc(). The allocated buffers must be freed with cv::fastFree().
  If there is not enough memory, the function calls cv::error(), which raises an exception.
  
  \param bufSize buffer size in bytes
  \return the allocated memory buffer.
*/ 
CV_EXPORTS void* fastMalloc(size_t bufSize);

/*!
  \fn void fastFree(void* ptr);
  \brief frees the memory allocated with cv::fastMalloc
  
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
  \fn template<typename _Tp> inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp));
  \brief aligns pointer by the certain number of bytes
  
  This small inline function aligns the pointer by the certian number of bytes by shifting
  it forward by 0 or a positive offset.
*/  
template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

/*!
  \fn inline size_t alignSize(size_t sz, int n);
  \brief aligns buffer size by the certain number of bytes
  
  This small inline function aligns a buffer size by the certian number of bytes by enlarging it.
*/
static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n-1) & -n;
}

/*!
  \fn void setUseOptimized(bool flag);
  \brief Turns on/off available optimization
  
  The function turns on or off the optimized code in OpenCV. Some optimization can not be enabled
  or disabled, but, for example, most of SSE code in OpenCV can be temporarily turned on or off this way.
  
  \note{Since optimization may imply using special data structures, it may be unsafe
  to call this function anywhere in the code. Instead, call it somewhere at the top level.}
*/  
CV_EXPORTS void setUseOptimized(bool);

/*!
  \fn bool useOptimized();
  \brief Returns the current optimization status
  
  The function returns the current optimization status, which is controlled by cv::setUseOptimized().
*/  
CV_EXPORTS bool useOptimized();

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
template<typename _Tp> class CV_EXPORTS DataDepth { public: enum { value = -1, fmt=(int)'\0' }; };

template<> class DataDepth<bool> { public: enum { value = CV_8U, fmt=(int)'u' }; };
template<> class DataDepth<uchar> { public: enum { value = CV_8U, fmt=(int)'u' }; };
template<> class DataDepth<schar> { public: enum { value = CV_8S, fmt=(int)'c' }; };
template<> class DataDepth<ushort> { public: enum { value = CV_16U, fmt=(int)'w' }; };
template<> class DataDepth<short> { public: enum { value = CV_16S, fmt=(int)'s' }; };
template<> class DataDepth<int> { public: enum { value = CV_32S, fmt=(int)'i' }; };
template<> class DataDepth<float> { public: enum { value = CV_32F, fmt=(int)'f' }; };
template<> class DataDepth<double> { public: enum { value = CV_64F, fmt=(int)'d' }; };
template<typename _Tp> class DataDepth<_Tp*> { public: enum { value = CV_USRTYPE1, fmt=(int)'r' }; };

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
template<typename _Tp, int cn> class CV_EXPORTS Vec
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

    Vec(const Vec<_Tp, cn>& v);
    static Vec all(_Tp alpha);
    //! dot product
    _Tp dot(const Vec& v) const;
    //! dot product computed in double-precision arithmetics
    double ddot(const Vec& v) const;
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

    _Tp val[cn]; //< vector elements
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
    //! another helper conversion method. \see cvScalarToRawData
    template<typename T2> void convertTo(T2* buf, int channels, int unroll_to=0) const;
};

typedef Scalar_<double> Scalar;

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
   as long as the number of channels does not exceed CV_MAX_CN (currently set to 32).
*/
template<typename _Tp> class DataType
{
public:
    typedef _Tp value_type;
    typedef value_type work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum { depth = DataDepth<channel_type>::value, channels = 1,
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
    enum { depth = DataDepth<channel_type>::value, channels = 1,
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
    enum { depth = DataDepth<channel_type>::value, channels = 1,
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
    enum { depth = DataDepth<channel_type>::value, channels = 1,
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
    enum { depth = DataDepth<channel_type>::value, channels = 1,
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
    enum { depth = DataDepth<channel_type>::value, channels = 1,
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
    enum { depth = DataDepth<channel_type>::value, channels = 1,
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
    enum { depth = DataDepth<channel_type>::value, channels = 1,
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
    enum { depth = DataDepth<channel_type>::value, channels = 1,
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
    enum { depth = DataDepth<channel_type>::value, channels = cn,
           fmt = ((channels-1)<<8) + DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
};

template<typename _Tp> class DataType<std::complex<_Tp> >
{
public:
    typedef std::complex<_Tp> value_type;
    typedef value_type work_type;
    typedef _Tp channel_type;
    enum { depth = DataDepth<channel_type>::value, channels = 2,
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
    enum { depth = DataDepth<channel_type>::value, channels = 2,
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
    enum { depth = DataDepth<channel_type>::value, channels = 2,
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
    enum { depth = DataDepth<channel_type>::value, channels = 3,
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
    enum { depth = DataDepth<channel_type>::value, channels = 2,
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
    enum { depth = DataDepth<channel_type>::value, channels = 4,
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
    enum { depth = DataDepth<channel_type>::value, channels = 4,
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
    enum { depth = DataDepth<channel_type>::value, channels = 2,
           fmt = ((channels-1)<<8) + DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
    typedef Vec<channel_type, channels> vec_type;
};

    
//////////////////// Generic ref-counting pointer class for C/C++ objects ////////////////////////

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
    int* refcount; //< the associated reference counter
};

//////////////////////////////// Mat ////////////////////////////////

class Mat;
class MatND;
template<typename M> class CV_EXPORTS MatExpr_Base_;
typedef MatExpr_Base_<Mat> MatExpr_Base;
template<typename E, typename M> class MatExpr_;
template<typename A1, typename M, typename Op> class MatExpr_Op1_;
template<typename A1, typename A2, typename M, typename Op> class MatExpr_Op2_;
template<typename A1, typename A2, typename A3, typename M, typename Op> class MatExpr_Op3_;
template<typename A1, typename A2, typename A3, typename A4,
        typename M, typename Op> class MatExpr_Op4_;
template<typename A1, typename A2, typename A3, typename A4,
        typename A5, typename M, typename Op> class MatExpr_Op5_;
template<typename M> class CV_EXPORTS MatOp_DivRS_;
template<typename M> class CV_EXPORTS MatOp_Inv_;
template<typename M> class CV_EXPORTS MatOp_MulDiv_;
template<typename M> class CV_EXPORTS MatOp_Repeat_;
template<typename M> class CV_EXPORTS MatOp_Set_;
template<typename M> class CV_EXPORTS MatOp_Scale_;
template<typename M> class CV_EXPORTS MatOp_T_;

typedef MatExpr_<MatExpr_Op4_<Size, int, Scalar,
    int, Mat, MatOp_Set_<Mat> >, Mat> MatExpr_Initializer;

template<typename _Tp> class MatIterator_;
template<typename _Tp> class MatConstIterator_;

enum { MAGIC_MASK=0xFFFF0000, TYPE_MASK=0x00000FFF, DEPTH_MASK=7 };

static inline size_t getElemSize(int type) { return CV_ELEM_SIZE(type); }

// matrix decomposition types
enum { DECOMP_LU=0, DECOMP_SVD=1, DECOMP_EIG=2, DECOMP_CHOLESKY=3, DECOMP_QR=4, DECOMP_NORMAL=16 };
enum { NORM_INF=1, NORM_L1=2, NORM_L2=4, NORM_TYPE_MASK=7, NORM_RELATIVE=8, NORM_MINMAX=32};
enum { CMP_EQ=0, CMP_GT=1, CMP_GE=2, CMP_LT=3, CMP_LE=4, CMP_NE=5 };
enum { GEMM_1_T=1, GEMM_2_T=2, GEMM_3_T=4 };
enum { DFT_INVERSE=1, DFT_SCALE=2, DFT_ROWS=4, DFT_COMPLEX_OUTPUT=16, DFT_REAL_OUTPUT=32,
    DCT_INVERSE = DFT_INVERSE, DCT_ROWS=DFT_ROWS };

/*!
   The matrix class.
   
   The class represents a 2D numerical array that can act as a matrix, image, optical flow map etc.
   It is very similar to CvMat type from earlier versions of OpenCV, and similarly to CvMat,
   the matrix can be multi-channel. It also fully supports ROI mechanism.

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

     As noted in the introduction of this chapter, \texttt{create()}
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
   if you need to process a whole row of matrix, the most efficient way is to get the pointer to the row first, and then just use plain C operator \texttt{[]}:

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

   The matrix iterators are random-access iterators, so they can be passed to any STL algorithm, including \texttt{std::sort()}.
*/
class CV_EXPORTS Mat
{
public:
    //! default constructor
    Mat();
    //! constructs matrix of the specified size and type
    // (_type is CV_8UC1, CV_64FC3, CV_32SC(12) etc.)
    Mat(int _rows, int _cols, int _type);
    Mat(Size _size, int _type);
    //! constucts matrix and fills it with the specified value _s.
    Mat(int _rows, int _cols, int _type, const Scalar& _s);
    Mat(Size _size, int _type, const Scalar& _s);
    //! copy constructor
    Mat(const Mat& m);
    //! constructor for matrix headers pointing to user-allocated data
    Mat(int _rows, int _cols, int _type, void* _data, size_t _step=AUTO_STEP);
    Mat(Size _size, int _type, void* _data, size_t _step=AUTO_STEP);
    //! creates a matrix header for a part of the bigger matrix
    Mat(const Mat& m, const Range& rowRange, const Range& colRange);
    Mat(const Mat& m, const Rect& roi);
    //! converts old-style CvMat to the new matrix; the data is not copied by default
    Mat(const CvMat* m, bool copyData=false);
    //! converts old-style IplImage to the new matrix; the data is not copied by default
    Mat(const IplImage* img, bool copyData=false);
    //! builds matrix from std::vector with or without copying the data
    template<typename _Tp> explicit Mat(const vector<_Tp>& vec, bool copyData=false);
    //! builds matrix from cv::Vec; the data is copied
    template<typename _Tp, int n> explicit Mat(const Vec<_Tp, n>& vec);
    //! builds matrix from a 2D point
    template<typename _Tp> explicit Mat(const Point_<_Tp>& pt);
    //! builds matrix from a 3D point
    template<typename _Tp> explicit Mat(const Point3_<_Tp>& pt);
    //! helper constructor to compile matrix expressions
    Mat(const MatExpr_Base& expr);
    //! destructor - calls release()
    ~Mat();
    //! assignment operators
    Mat& operator = (const Mat& m);
    Mat& operator = (const MatExpr_Base& expr);

    operator MatExpr_<Mat, Mat>() const;

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

    //! matrix transposition by means of matrix expressions
    MatExpr_<MatExpr_Op2_<Mat, double, Mat, MatOp_T_<Mat> >, Mat>
    t() const;
    //! matrix inversion by means of matrix expressions
    MatExpr_<MatExpr_Op2_<Mat, int, Mat, MatOp_Inv_<Mat> >, Mat>
        inv(int method=DECOMP_LU) const;
    MatExpr_<MatExpr_Op4_<Mat, Mat, double, char, Mat, MatOp_MulDiv_<Mat> >, Mat>
    //! per-element matrix multiplication by means of matrix expressions
    mul(const Mat& m, double scale=1) const;
    MatExpr_<MatExpr_Op4_<Mat, Mat, double, char, Mat, MatOp_MulDiv_<Mat> >, Mat>
    mul(const MatExpr_<MatExpr_Op2_<Mat, double, Mat, MatOp_Scale_<Mat> >, Mat>& m, double scale=1) const;
    MatExpr_<MatExpr_Op4_<Mat, Mat, double, char, Mat, MatOp_MulDiv_<Mat> >, Mat>    
    mul(const MatExpr_<MatExpr_Op2_<Mat, double, Mat, MatOp_DivRS_<Mat> >, Mat>& m, double scale=1) const;
    
    //! computes cross-product of 2 3D vectors
    Mat cross(const Mat& m) const;
    //! computes dot-product
    double dot(const Mat& m) const;

    //! Matlab-style matrix initialization
    static MatExpr_Initializer zeros(int rows, int cols, int type);
    static MatExpr_Initializer zeros(Size size, int type);
    static MatExpr_Initializer ones(int rows, int cols, int type);
    static MatExpr_Initializer ones(Size size, int type);
    static MatExpr_Initializer eye(int rows, int cols, int type);
    static MatExpr_Initializer eye(Size size, int type);

    //! allocates new matrix data unless the matrix already has specified size and type.
    // previous data is unreferenced if needed.
    void create(int _rows, int _cols, int _type);
    void create(Size _size, int _type);
    //! increases the reference counter; use with care to avoid memleaks
    void addref();
    //! decreases reference counter;
    // deallocate the data when reference counter reaches 0.
    void release();

    //! locates matrix header within a parent matrix. See below
    void locateROI( Size& wholeSize, Point& ofs ) const;
    //! moves/resizes the current matrix ROI inside the parent matrix.
    Mat& adjustROI( int dtop, int dbottom, int dleft, int dright );
    //! extracts a rectangular sub-matrix
    // (this is a generalized form of row, rowRange etc.)
    Mat operator()( Range rowRange, Range colRange ) const;
    Mat operator()( const Rect& roi ) const;

    //! converts header to CvMat; no data is copied
    operator CvMat() const;
    //! converts header to IplImage; no data is copied
    operator IplImage() const;
    
    //! returns true iff the matrix data is continuous
    // (i.e. when there are no gaps between successive rows).
    // similar to CV_IS_MAT_CONT(cvmat->type)
    bool isContinuous() const;
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
    size_t step1() const;
    //! returns matrix size:
    // width == number of columns, height == number of rows
    Size size() const;
    //! returns true if matrix data is NULL
    bool empty() const;

    //! returns pointer to y-th row
    uchar* ptr(int y=0);
    const uchar* ptr(int y=0) const;

    //! template version of the above method
    template<typename _Tp> _Tp* ptr(int y=0);
    template<typename _Tp> const _Tp* ptr(int y=0) const;
    
    //! template methods for read-write or read-only element access.
    // note that _Tp must match the actual matrix type -
    // the functions do not do any on-fly type conversion
    template<typename _Tp> _Tp& at(int y, int x);
    template<typename _Tp> _Tp& at(Point pt);
    template<typename _Tp> const _Tp& at(int y, int x) const;
    template<typename _Tp> const _Tp& at(Point pt) const;
    template<typename _Tp> _Tp& at(int i);
    template<typename _Tp> const _Tp& at(int i) const;
    
    //! template methods for iteration over matrix elements.
    // the iterators take care of skipping gaps in the end of rows (if any)
    template<typename _Tp> MatIterator_<_Tp> begin();
    template<typename _Tp> MatIterator_<_Tp> end();
    template<typename _Tp> MatConstIterator_<_Tp> begin() const;
    template<typename _Tp> MatConstIterator_<_Tp> end() const;

    enum { MAGIC_VAL=0x42FF0000, AUTO_STEP=0, CONTINUOUS_FLAG=CV_MAT_CONT_FLAG };

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
    // when matrix points to user-allocated data, the pointer is NULL
    int* refcount;
    
    //! helper fields used in locateROI and adjustROI
    uchar* datastart;
    uchar* dataend;
};


/*!
   Random Number Generator
 
   The class implements RNG using Multiply-with-Carry algorithm
*/
class CV_EXPORTS RNG
{
public:
    enum { A=4164903690U, UNIFORM=0, NORMAL=1 };

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
    void fill( MatND& mat, int distType, const Scalar& a, const Scalar& b );
	//! returns Gaussian random variate with mean zero.
	double gaussian(double sigma);

    uint64 state;
};

class CV_EXPORTS TermCriteria
{
public:
    enum { COUNT=1, MAX_ITER=COUNT, EPS=2 };

    TermCriteria();
    TermCriteria(int _type, int _maxCount, double _epsilon);
    TermCriteria(const CvTermCriteria& criteria);
    operator CvTermCriteria() const;
    
    int type;
    int maxCount;
    double epsilon;
};

//! converts array (CvMat or IplImage) to cv::Mat
CV_EXPORTS Mat cvarrToMat(const CvArr* arr, bool copyData,
                          bool allowND, int coiMode);
//! extracts Channel of Interest from CvMat or IplImage and makes cv::Mat out of it.
CV_EXPORTS void extractImageCOI(const CvArr* arr, Mat& coiimg, int coi=-1);
//! inserts single-channel cv::Mat into a multi-channel CvMat or IplImage
CV_EXPORTS void insertImageCOI(const Mat& coiimg, CvArr* arr, int coi=-1);

//! adds one matrix to another (c = a + b)
CV_EXPORTS void add(const Mat& a, const Mat& b, Mat& c, const Mat& mask);
//! subtracts one matrix from another (c = a - b) 
CV_EXPORTS void subtract(const Mat& a, const Mat& b, Mat& c, const Mat& mask);
//! adds one matrix to another (c = a + b)    
CV_EXPORTS void add(const Mat& a, const Mat& b, Mat& c);
//! subtracts one matrix from another (c = a - b) 
CV_EXPORTS void subtract(const Mat& a, const Mat& b, Mat& c);
//! adds scalar to a matrix (c = a + s)
CV_EXPORTS void add(const Mat& a, const Scalar& s, Mat& c, const Mat& mask=Mat());
//! subtracts scalar from a matrix (c = a - s)    
CV_EXPORTS void subtract(const Mat& a, const Scalar& s, Mat& c, const Mat& mask=Mat());
//! subtracts matrix from a scalar matrix (c = s - a)    
CV_EXPORTS void subtract(const Scalar& s, const Mat& a, Mat& c, const Mat& mask=Mat());

//! computes element-wise weighted product of the two arrays (c = scale*a*b)
CV_EXPORTS void multiply(const Mat& a, const Mat& b, Mat& c, double scale=1);
//! computes element-wise weighted quotient of the two arrays (c = scale*a/b)
CV_EXPORTS void divide(const Mat& a, const Mat& b, Mat& c, double scale=1);
//! computes element-wise weighted reciprocal of an array (c = scale/b)
CV_EXPORTS void divide(double scale, const Mat& b, Mat& c);

//! adds scaled array to another one (c = a*alpha + b)
CV_EXPORTS void scaleAdd(const Mat& a, double alpha, const Mat& b, Mat& c);
//! computes weighted sum of two arrays (c=alpha*a + beta*b + gamma)
CV_EXPORTS void addWeighted(const Mat& a, double alpha, const Mat& b,
                            double beta, double gamma, Mat& c);
//! scales array elements, computes absolute values and converts the results to 8-bit unsigned integers: c(i)=saturate_cast<uchar>abs(a(i)*alpha+beta)
CV_EXPORTS void convertScaleAbs(const Mat& a, Mat& c, double alpha=1, double beta=0);
//! transforms 8-bit unsigned integers using lookup table: b(i)=lut(a(i))
CV_EXPORTS void LUT(const Mat& a, const Mat& lut, Mat& b);

//! computes sum of array elements
CV_EXPORTS Scalar sum(const Mat& m);
//! computes the number of nonzero array elements
CV_EXPORTS int countNonZero( const Mat& m );

//! computes mean value of array elements
CV_EXPORTS Scalar mean(const Mat& m);
//! computes mean value of selected array elements
CV_EXPORTS Scalar mean(const Mat& m, const Mat& mask);
//! computes mean value and standard deviation of all or selected array elements
CV_EXPORTS void meanStdDev(const Mat& m, Scalar& mean, Scalar& stddev, const Mat& mask=Mat());
//! computes norm of array elements
CV_EXPORTS double norm(const Mat& a, int normType=NORM_L2);
//! computes norm of the difference between two arrays
CV_EXPORTS double norm(const Mat& a, const Mat& b, int normType=NORM_L2);
//! computes norm of selected array elements
CV_EXPORTS double norm(const Mat& a, int normType, const Mat& mask);
//! computes norm of selected part of the difference between two arrays
CV_EXPORTS double norm(const Mat& a, const Mat& b,
                       int normType, const Mat& mask);
//! scales and shifts array elements so that either the specified norm (alpha) or the minimum (alpha) and maximum (beta) array values get the specified values 
CV_EXPORTS void normalize( const Mat& a, Mat& b, double alpha=1, double beta=0,
                          int norm_type=NORM_L2, int rtype=-1, const Mat& mask=Mat());

//! computes global minimum and maximum array elements and their locations
CV_EXPORTS void minMaxLoc(const Mat& a, double* minVal,
                          double* maxVal=0, Point* minLoc=0,
                          Point* maxLoc=0, const Mat& mask=Mat());
//! transforms 2D matrix to 1D row or column vector by taking sum, minimum, maximum or mean value over all the rows
CV_EXPORTS void reduce(const Mat& m, Mat& dst, int dim, int rtype, int dtype=-1);
//! makes multi-channel array out of several single channel arrays
CV_EXPORTS void merge(const Mat* mv, size_t count, Mat& dst);
//! copies each plane of a multi-channel array to a dedicated array
CV_EXPORTS void split(const Mat& m, Mat* mvbegin);

//! copies selected channels from the input arrays to the selected channels of the output arrays
CV_EXPORTS void mixChannels(const Mat* src, int nsrcs, Mat* dst, int ndsts,
                            const int* fromTo, size_t npairs);
//! reverses the order of the rows, columns or both in a matrix
CV_EXPORTS void flip(const Mat& a, Mat& b, int flipCode);

//! replicates the input matrix the specified number of times in the horizontal and/or vertical direction
CV_EXPORTS void repeat(const Mat& a, int ny, int nx, Mat& b);
static inline Mat repeat(const Mat& src, int ny, int nx)
{
    if( nx == 1 && ny == 1 ) return src;
    Mat dst; repeat(src, ny, nx, dst); return dst;
}

//! computes bitwise conjunction of the two arrays (c = a & b)
CV_EXPORTS void bitwise_and(const Mat& a, const Mat& b, Mat& c, const Mat& mask=Mat());
//! computes bitwise disjunction of the two arrays (c = a | b)
CV_EXPORTS void bitwise_or(const Mat& a, const Mat& b, Mat& c, const Mat& mask=Mat());
//! computes bitwise exclusive-or of the two arrays (c = a ^ b)
CV_EXPORTS void bitwise_xor(const Mat& a, const Mat& b, Mat& c, const Mat& mask=Mat());
//! computes bitwise conjunction of an array and scalar (c = a & s)
CV_EXPORTS void bitwise_and(const Mat& a, const Scalar& s, Mat& c, const Mat& mask=Mat());
//! computes bitwise disjunction of an array and scalar (c = a | s)
CV_EXPORTS void bitwise_or(const Mat& a, const Scalar& s, Mat& c, const Mat& mask=Mat());
//! computes bitwise exclusive-or of an array and scalar (c = a ^ s)
CV_EXPORTS void bitwise_xor(const Mat& a, const Scalar& s, Mat& c, const Mat& mask=Mat());
//! inverts each bit of a (c = ~a)
CV_EXPORTS void bitwise_not(const Mat& a, Mat& c);
//! computes element-wise absolute difference of two arrays (c = abs(a - b))
CV_EXPORTS void absdiff(const Mat& a, const Mat& b, Mat& c);
//! computes element-wise absolute difference of array and a scalar (c = abs(a - s))
CV_EXPORTS void absdiff(const Mat& a, const Scalar& s, Mat& c);
//! set mask elements for those array elements which are within the element-specific bounding box (dst = lowerb <= src && src < upperb)    
CV_EXPORTS void inRange(const Mat& src, const Mat& lowerb,
                        const Mat& upperb, Mat& dst);
//! set mask elements for those array elements which are within the fixed bounding box (dst = lowerb <= src && src < upperb)    
CV_EXPORTS void inRange(const Mat& src, const Scalar& lowerb,
                        const Scalar& upperb, Mat& dst);
//! compares elements of two arrays (c = a <cmpop> b)
CV_EXPORTS void compare(const Mat& a, const Mat& b, Mat& c, int cmpop);
//! compares elements of array with a scalar (c = a <cmpop> s)
CV_EXPORTS void compare(const Mat& a, double s, Mat& c, int cmpop);
//! computes per-element minimum of two arrays (c = min(a, b))
CV_EXPORTS void min(const Mat& a, const Mat& b, Mat& c);
//! computes per-element minimum of array and a scalar (c = min(a, alpha))
CV_EXPORTS void min(const Mat& a, double alpha, Mat& c);
//! computes per-element maximum of two arrays (c = max(a, b))
CV_EXPORTS void max(const Mat& a, const Mat& b, Mat& c);
//! computes per-element maximum of array and a scalar (c = max(a, alpha))
CV_EXPORTS void max(const Mat& a, double alpha, Mat& c);


CV_EXPORTS void sqrt(const Mat& a, Mat& b);
CV_EXPORTS void pow(const Mat& a, double power, Mat& b);
CV_EXPORTS void exp(const Mat& a, Mat& b);
CV_EXPORTS void log(const Mat& a, Mat& b);
CV_EXPORTS float cubeRoot(float val);
CV_EXPORTS float fastAtan2(float y, float x);
CV_EXPORTS void polarToCart(const Mat& magnitude, const Mat& angle,
                            Mat& x, Mat& y, bool angleInDegrees=false);
CV_EXPORTS void cartToPolar(const Mat& x, const Mat& y,
                            Mat& magnitude, Mat& angle,
                            bool angleInDegrees=false);
CV_EXPORTS void phase(const Mat& x, const Mat& y, Mat& angle,
                            bool angleInDegrees=false);
CV_EXPORTS void magnitude(const Mat& x, const Mat& y, Mat& magnitude);
CV_EXPORTS bool checkRange(const Mat& a, bool quiet=true, Point* pt=0,
                           double minVal=-DBL_MAX, double maxVal=DBL_MAX);

CV_EXPORTS void gemm(const Mat& a, const Mat& b, double alpha,
                     const Mat& c, double gamma, Mat& d, int flags=0);
CV_EXPORTS void mulTransposed( const Mat& a, Mat& c, bool aTa,
                               const Mat& delta=Mat(),
                               double scale=1, int rtype=-1 );
CV_EXPORTS void transpose(const Mat& a, Mat& b);
CV_EXPORTS void transform(const Mat& src, Mat& dst, const Mat& m );
CV_EXPORTS void perspectiveTransform(const Mat& src, Mat& dst, const Mat& m );

CV_EXPORTS void completeSymm(Mat& a, bool lowerToUpper=false);
CV_EXPORTS void setIdentity(Mat& c, const Scalar& s=Scalar(1));
CV_EXPORTS double determinant(const Mat& m);
CV_EXPORTS Scalar trace(const Mat& m);
CV_EXPORTS double invert(const Mat& a, Mat& c, int flags=DECOMP_LU);
CV_EXPORTS bool solve(const Mat& a, const Mat& b, Mat& x, int flags=DECOMP_LU);
CV_EXPORTS void sort(const Mat& a, Mat& b, int flags);
CV_EXPORTS void sortIdx(const Mat& a, Mat& b, int flags);
CV_EXPORTS int solveCubic(const Mat& coeffs, Mat& roots);
CV_EXPORTS double solvePoly(const Mat& coeffs, Mat& roots, int maxIters=300);
CV_EXPORTS bool eigen(const Mat& a, Mat& eigenvalues, int lowindex=-1,
                      int highindex=-1);
CV_EXPORTS bool eigen(const Mat& a, Mat& eigenvalues, Mat& eigenvectors,
                      int lowindex=-1, int highindex=-1);

CV_EXPORTS void calcCovarMatrix( const Mat* samples, int nsamples,
                                 Mat& covar, Mat& mean,
                                 int flags, int ctype=CV_64F);
CV_EXPORTS void calcCovarMatrix( const Mat& samples, Mat& covar, Mat& mean,
                                 int flags, int ctype=CV_64F);

class CV_EXPORTS PCA
{
public:
    PCA();
    PCA(const Mat& data, const Mat& mean, int flags, int maxComponents=0);
    PCA& operator()(const Mat& data, const Mat& mean, int flags, int maxComponents=0);
    Mat project(const Mat& vec) const;
    void project(const Mat& vec, Mat& result) const;
    Mat backProject(const Mat& vec) const;
    void backProject(const Mat& vec, Mat& result) const;

    Mat eigenvectors;
    Mat eigenvalues;
    Mat mean;
};

class CV_EXPORTS SVD
{
public:
    enum { MODIFY_A=1, NO_UV=2, FULL_UV=4 };
    SVD();
    SVD( const Mat& m, int flags=0 );
    SVD& operator ()( const Mat& m, int flags=0 );

    static void solveZ( const Mat& m, Mat& dst );
    void backSubst( const Mat& rhs, Mat& dst ) const;

    Mat u, w, vt;
};

CV_EXPORTS double Mahalanobis(const Mat& v1, const Mat& v2, const Mat& icovar);
static inline double Mahalonobis(const Mat& v1, const Mat& v2, const Mat& icovar)
{ return Mahalanobis(v1, v2, icovar); }

CV_EXPORTS void dft(const Mat& src, Mat& dst, int flags=0, int nonzeroRows=0);
CV_EXPORTS void idft(const Mat& src, Mat& dst, int flags=0, int nonzeroRows=0);
CV_EXPORTS void dct(const Mat& src, Mat& dst, int flags=0);
CV_EXPORTS void idct(const Mat& src, Mat& dst, int flags=0);
CV_EXPORTS void mulSpectrums(const Mat& a, const Mat& b, Mat& c,
                             int flags, bool conjB=false);
CV_EXPORTS int getOptimalDFTSize(int vecsize);

enum { KMEANS_RANDOM_CENTERS=0, KMEANS_PP_CENTERS=2, KMEANS_USE_INITIAL_LABELS=1 };
CV_EXPORTS double kmeans( const Mat& data, int K, Mat& best_labels,
                          TermCriteria criteria, int attempts,
                          int flags, Mat* centers );

CV_EXPORTS RNG& theRNG();
template<typename _Tp> static inline _Tp randu() { return (_Tp)theRNG(); }

static inline void randu(Mat& dst, const Scalar& low, const Scalar& high)
{ theRNG().fill(dst, RNG::UNIFORM, low, high); }
static inline void randn(Mat& dst, const Scalar& mean, const Scalar& stddev)
{ theRNG().fill(dst, RNG::NORMAL, mean, stddev); }
CV_EXPORTS void randShuffle(Mat& dst, double iterFactor=1., RNG* rng=0);


CV_EXPORTS void line(Mat& img, Point pt1, Point pt2, const Scalar& color,
                     int thickness=1, int lineType=8, int shift=0);

CV_EXPORTS void rectangle(Mat& img, Point pt1, Point pt2,
                          const Scalar& color, int thickness=1,
                          int lineType=8, int shift=0);

CV_EXPORTS void rectangle(Mat& img, Rect rec,
                          const Scalar& color, int thickness=1,
                          int lineType=8, int shift=0);

CV_EXPORTS void circle(Mat& img, Point center, int radius,
                       const Scalar& color, int thickness=1,
                       int lineType=8, int shift=0);

CV_EXPORTS void ellipse(Mat& img, Point center, Size axes,
                        double angle, double startAngle, double endAngle,
                        const Scalar& color, int thickness=1,
                        int lineType=8, int shift=0);

CV_EXPORTS void ellipse(Mat& img, const RotatedRect& box, const Scalar& color,
                        int thickness=1, int lineType=8);

CV_EXPORTS void fillConvexPoly(Mat& img, const Point* pts, int npts,
                               const Scalar& color, int lineType=8,
                               int shift=0);

CV_EXPORTS void fillPoly(Mat& img, const Point** pts, const int* npts, int ncontours,
                         const Scalar& color, int lineType=8, int shift=0,
                         Point offset=Point() );

CV_EXPORTS void polylines(Mat& img, const Point** pts, const int* npts, int ncontours, bool isClosed,
                          const Scalar& color, int thickness=1, int lineType=8, int shift=0 );

CV_EXPORTS bool clipLine(Size imgSize, Point& pt1, Point& pt2);
CV_EXPORTS bool clipLine(Rect img_rect, Point& pt1, Point& pt2);

class CV_EXPORTS LineIterator
{
public:
    LineIterator(const Mat& img, Point pt1, Point pt2,
                 int connectivity=8, bool leftToRight=false);
    uchar* operator *();
    LineIterator& operator ++();
    LineIterator operator ++(int);

    uchar* ptr;
    int err, count;
    int minusDelta, plusDelta;
    int minusStep, plusStep;
};

CV_EXPORTS void ellipse2Poly( Point center, Size axes, int angle,
                              int arcStart, int arcEnd, int delta, vector<Point>& pts );

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

CV_EXPORTS void putText( Mat& img, const string& text, Point org,
                         int fontFace, double fontScale, Scalar color,
                         int thickness=1, int linetype=8,
                         bool bottomLeftOrigin=false );

CV_EXPORTS Size getTextSize(const string& text, int fontFace,
                            double fontScale, int thickness,
                            int* baseLine);

///////////////////////////////// Mat_<_Tp> ////////////////////////////////////

template<typename _Tp> class CV_EXPORTS Mat_ : public Mat
{
public:
    typedef _Tp value_type;
    typedef typename DataType<_Tp>::channel_type channel_type;
    typedef MatIterator_<_Tp> iterator;
    typedef MatConstIterator_<_Tp> const_iterator;
    
    Mat_();
    //! equivalent to Mat(_rows, _cols, DataType<_Tp>::type)
    Mat_(int _rows, int _cols);
    //! other forms of the above constructor
    Mat_(int _rows, int _cols, const _Tp& value);
    explicit Mat_(Size _size);
    Mat_(Size _size, const _Tp& value);
    //! copy/conversion contructor. If m is of different type, it's converted
    Mat_(const Mat& m);
    //! copy constructor
    Mat_(const Mat_& m);
    //! construct a matrix on top of user-allocated data.
    // step is in bytes(!!!), regardless of the type
    Mat_(int _rows, int _cols, _Tp* _data, size_t _step=AUTO_STEP);
    // minor selection
    Mat_(const Mat_& m, const Range& rowRange, const Range& colRange);
    Mat_(const Mat_& m, const Rect& roi);
    //! to support complex matrix expressions
    Mat_(const MatExpr_Base& expr);
    //! makes a matrix out of Vec, std::vector, Point_ or Point3_. The matrix will have a single column
    explicit Mat_(const vector<_Tp>& vec, bool copyData=false);
    template<int n> explicit Mat_(const Vec<_Tp, n>& vec);
    explicit Mat_(const Point_<_Tp>& pt);
    explicit Mat_(const Point3_<_Tp>& pt);

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
    void create(Size _size);
    //! cross-product
    Mat_ cross(const Mat_& m) const;
    //! to support complex matrix expressions
    Mat_& operator = (const MatExpr_Base& expr);
    //! data type conversion
    template<typename T2> operator Mat_<T2>() const;
    //! overridden forms of Mat::row() etc.
    Mat_ row(int y) const;
    Mat_ col(int x) const;
    Mat_ diag(int d=0) const;
    Mat_ clone() const;

    //! transposition, inversion, per-element multiplication
    MatExpr_<MatExpr_Op2_<Mat, double, Mat, MatOp_T_<Mat> >, Mat> t() const;
    MatExpr_<MatExpr_Op2_<Mat, int, Mat, MatOp_Inv_<Mat> >, Mat> inv(int method=DECOMP_LU) const;

    MatExpr_<MatExpr_Op4_<Mat, Mat, double, char, Mat, MatOp_MulDiv_<Mat> >, Mat>
    mul(const Mat_& m, double scale=1) const;
    MatExpr_<MatExpr_Op4_<Mat, Mat, double, char, Mat, MatOp_MulDiv_<Mat> >, Mat>
    mul(const MatExpr_<MatExpr_Op2_<Mat, double, Mat,
        MatOp_Scale_<Mat> >, Mat>& m, double scale=1) const;
    MatExpr_<MatExpr_Op4_<Mat, Mat, double, char, Mat, MatOp_MulDiv_<Mat> >, Mat>    
    mul(const MatExpr_<MatExpr_Op2_<Mat, double, Mat,
        MatOp_DivRS_<Mat> >, Mat>& m, double scale=1) const;

    // overridden forms of Mat::elemSize() etc.
    size_t elemSize() const;
    size_t elemSize1() const;
    int type() const;
    int depth() const;
    int channels() const;
    size_t step1() const;
    // returns step()/sizeof(_Tp)
    size_t stepT() const;

    // overridden forms of Mat::zeros() etc. Data type is omitted, of course
    static MatExpr_Initializer zeros(int rows, int cols);
    static MatExpr_Initializer zeros(Size size);
    static MatExpr_Initializer ones(int rows, int cols);
    static MatExpr_Initializer ones(Size size);
    static MatExpr_Initializer eye(int rows, int cols);
    static MatExpr_Initializer eye(Size size);

    // some more overriden methods
    Mat_ reshape(int _rows) const;
    Mat_& adjustROI( int dtop, int dbottom, int dleft, int dright );
    Mat_ operator()( const Range& rowRange, const Range& colRange ) const;
    Mat_ operator()( const Rect& roi ) const;

    // more convenient forms of row and element access operators 
    _Tp* operator [](int y);
    const _Tp* operator [](int y) const;

    _Tp& operator ()(int row, int col);
    const _Tp& operator ()(int row, int col) const;
    _Tp& operator ()(Point pt);
    const _Tp& operator ()(Point pt) const;
    _Tp& operator ()(int i);
    const _Tp& operator ()(int i) const;

    // to support matrix expressions
    operator MatExpr_<Mat, Mat>() const;
    
    // conversion to vector.
    operator vector<_Tp>() const;
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

template<typename _Tp>
class CV_EXPORTS MatConstIterator_
{
public:
    typedef _Tp value_type;
    typedef int difference_type;

    MatConstIterator_();
    MatConstIterator_(const Mat_<_Tp>* _m);
    MatConstIterator_(const Mat_<_Tp>* _m, int _row, int _col=0);
    MatConstIterator_(const Mat_<_Tp>* _m, Point _pt);
    MatConstIterator_(const MatConstIterator_& it);

    MatConstIterator_& operator = (const MatConstIterator_& it );
    _Tp operator *() const;
    _Tp operator [](int i) const;
    
    MatConstIterator_& operator += (int ofs);
    MatConstIterator_& operator -= (int ofs);
    MatConstIterator_& operator --();
    MatConstIterator_ operator --(int);
    MatConstIterator_& operator ++();
    MatConstIterator_ operator ++(int);
    Point pos() const;

    const Mat_<_Tp>* m;
    _Tp* ptr;
    _Tp* sliceEnd;
};


template<typename _Tp>
class CV_EXPORTS MatIterator_ : public MatConstIterator_<_Tp>
{
public:
    typedef _Tp* pointer;
    typedef _Tp& reference;
    typedef std::random_access_iterator_tag iterator_category;

    MatIterator_();
    MatIterator_(Mat_<_Tp>* _m);
    MatIterator_(Mat_<_Tp>* _m, int _row, int _col=0);
    MatIterator_(const Mat_<_Tp>* _m, Point _pt);
    MatIterator_(const MatIterator_& it);
    MatIterator_& operator = (const MatIterator_<_Tp>& it );

    _Tp& operator *() const;
    _Tp& operator [](int i) const;

    MatIterator_& operator += (int ofs);
    MatIterator_& operator -= (int ofs);
    MatIterator_& operator --();
    MatIterator_ operator --(int);
    MatIterator_& operator ++();
    MatIterator_ operator ++(int);
};

template<typename _Tp> class CV_EXPORTS MatOp_Iter_;

template<typename _Tp> class CV_EXPORTS MatCommaInitializer_ :
    public MatExpr_<MatExpr_Op1_<MatIterator_<_Tp>, Mat_<_Tp>, MatOp_Iter_<_Tp> >, Mat_<_Tp> >
{
public:
    MatCommaInitializer_(Mat_<_Tp>* _m);
    template<typename T2> MatCommaInitializer_<_Tp>& operator , (T2 v);
    operator Mat_<_Tp>() const;
    Mat_<_Tp> operator *() const;
    void assignTo(Mat& m, int type=-1) const;
};

#if 0
template<typename _Tp> class VectorCommaInitializer_
{
public:
    VectorCommaInitializer_(vector<_Tp>* _vec);
    template<typename T2> VectorCommaInitializer_<_Tp>& operator , (T2 val);
    operator vector<_Tp>() const;
    vector<_Tp> operator *() const;

    vector<_Tp>* vec;
    int idx;
};
#endif

template<typename _Tp, size_t fixed_size=4096/sizeof(_Tp)+8> class CV_EXPORTS AutoBuffer
{
public:
    typedef _Tp value_type;

    AutoBuffer();
    AutoBuffer(size_t _size);
    ~AutoBuffer();

    void allocate(size_t _size);
    void deallocate();
    operator _Tp* ();
    operator const _Tp* () const;

protected:
    _Tp* ptr;
    size_t size;
    _Tp buf[fixed_size];
};

/////////////////////////// multi-dimensional dense matrix //////////////////////////

class MatND;
class SparseMat;

class CV_EXPORTS MatND
{
public:
    // default constructor
    MatND();
    // constructs array with specific size and data type
    MatND(int _ndims, const int* _sizes, int _type);
    // constructs array and fills it with the specified value
    MatND(int _ndims, const int* _sizes, int _type, const Scalar& _s);
    // copy constructor. only the header is copied.
    MatND(const MatND& m);
    // sub-array selection. only the header is copied
    MatND(const MatND& m, const Range* ranges);
    // converts 2D matrix to ND matrix
    explicit MatND(const Mat& m);
    // converts old-style nd array to MatND; optionally, copies the data
    MatND(const CvMatND* m, bool copyData=false);
    ~MatND();
    MatND& operator = (const MatND& m);
    
    void assignTo( MatND& m, int type ) const;

    // creates a complete copy of the matrix (all the data is copied)
    MatND clone() const;
    // sub-array selection; only the header is copied
    MatND operator()(const Range* ranges) const;

    // copies the data to another matrix.
    // Calls m.create(this->size(), this->type()) prior to
    // copying the data
    void copyTo( MatND& m ) const;
    // copies only the selected elements to another matrix.
    void copyTo( MatND& m, const MatND& mask ) const;
    // converts data to the specified data type.
    // calls m.create(this->size(), rtype) prior to the conversion
    void convertTo( MatND& m, int rtype, double alpha=1, double beta=0 ) const;
    
    // assigns "s" to each array element. 
    MatND& operator = (const Scalar& s);
    // assigns "s" to the selected elements of array
    // (or to all the elements if mask==MatND())
    MatND& setTo(const Scalar& s, const MatND& mask=MatND());
    // modifies geometry of array without copying the data
    MatND reshape(int _newcn, int _newndims=0, const int* _newsz=0) const;

    // allocates a new buffer for the data unless the current one already
    // has the specified size and type.
    void create(int _ndims, const int* _sizes, int _type);
    // manually increment reference counter (use with care !!!)
    void addref();
    // decrements the reference counter. Dealloctes the data when
    // the reference counter reaches zero.
    void release();

    // converts the matrix to 2D Mat or to the old-style CvMatND.
    // In either case the data is not copied.
    operator Mat() const;
    operator CvMatND() const;
    // returns true if the array data is stored continuously 
    bool isContinuous() const;
    // returns size of each element in bytes
    size_t elemSize() const;
    // returns size of each element channel in bytes
    size_t elemSize1() const;
    // returns OpenCV data type id (CV_8UC1, ... CV_64FC4,...)
    int type() const;
    // returns depth (CV_8U ... CV_64F)
    int depth() const;
    // returns the number of channels
    int channels() const;
    // step1() ~ step()/elemSize1()
    size_t step1(int i) const;

    // return pointer to the element (versions for 1D, 2D, 3D and generic nD cases)
    uchar* ptr(int i0);
    const uchar* ptr(int i0) const;
    uchar* ptr(int i0, int i1);
    const uchar* ptr(int i0, int i1) const;
    uchar* ptr(int i0, int i1, int i2);
    const uchar* ptr(int i0, int i1, int i2) const;
    uchar* ptr(const int* idx);
    const uchar* ptr(const int* idx) const;

    // convenient template methods for element access.
    // note that _Tp must match the actual matrix type -
    // the functions do not do any on-fly type conversion
    template<typename _Tp> _Tp& at(int i0);
    template<typename _Tp> const _Tp& at(int i0) const;
    template<typename _Tp> _Tp& at(int i0, int i1);
    template<typename _Tp> const _Tp& at(int i0, int i1) const;
    template<typename _Tp> _Tp& at(int i0, int i1, int i2);
    template<typename _Tp> const _Tp& at(int i0, int i1, int i2) const;
    template<typename _Tp> _Tp& at(const int* idx);
    template<typename _Tp> const _Tp& at(const int* idx) const;

    enum { MAGIC_VAL=0x42FE0000, AUTO_STEP=-1,
        CONTINUOUS_FLAG=CV_MAT_CONT_FLAG, MAX_DIM=CV_MAX_DIM };

    // combines data type, continuity flag, signature (magic value) 
    int flags;
    // the array dimensionality
    int dims;

    // data reference counter
    int* refcount;
    // pointer to the data
    uchar* data;
    // and its actual beginning and end
    uchar* datastart;
    uchar* dataend;

    // step and size for each dimension, MAX_DIM at max
    int size[MAX_DIM];
    size_t step[MAX_DIM];
};

class CV_EXPORTS NAryMatNDIterator
{
public:
    NAryMatNDIterator();
    NAryMatNDIterator(const MatND* arrays, size_t count);
    NAryMatNDIterator(const MatND** arrays, size_t count);
    NAryMatNDIterator(const MatND& m1);
    NAryMatNDIterator(const MatND& m1, const MatND& m2);
    NAryMatNDIterator(const MatND& m1, const MatND& m2, const MatND& m3);
    NAryMatNDIterator(const MatND& m1, const MatND& m2, const MatND& m3, const MatND& m4);
    NAryMatNDIterator(const MatND& m1, const MatND& m2, const MatND& m3,
                      const MatND& m4, const MatND& m5);
    NAryMatNDIterator(const MatND& m1, const MatND& m2, const MatND& m3,
                      const MatND& m4, const MatND& m5, const MatND& m6);
    
    void init(const MatND** arrays, size_t count);

    NAryMatNDIterator& operator ++();
    NAryMatNDIterator operator ++(int);
    
    vector<MatND> arrays;
    vector<Mat> planes;

    int nplanes;
protected:
    int iterdepth, idx;
};

CV_EXPORTS void add(const MatND& a, const MatND& b, MatND& c, const MatND& mask);
CV_EXPORTS void subtract(const MatND& a, const MatND& b, MatND& c, const MatND& mask);
CV_EXPORTS void add(const MatND& a, const MatND& b, MatND& c);
CV_EXPORTS void subtract(const MatND& a, const MatND& b, MatND& c);
CV_EXPORTS void add(const MatND& a, const Scalar& s, MatND& c, const MatND& mask=MatND());

CV_EXPORTS void multiply(const MatND& a, const MatND& b, MatND& c, double scale=1);
CV_EXPORTS void divide(const MatND& a, const MatND& b, MatND& c, double scale=1);
CV_EXPORTS void divide(double scale, const MatND& b, MatND& c);

CV_EXPORTS void subtract(const Scalar& s, const MatND& a, MatND& c, const MatND& mask=MatND());
CV_EXPORTS void scaleAdd(const MatND& a, double alpha, const MatND& b, MatND& c);
CV_EXPORTS void addWeighted(const MatND& a, double alpha, const MatND& b,
                            double beta, double gamma, MatND& c);

CV_EXPORTS Scalar sum(const MatND& m);
CV_EXPORTS int countNonZero( const MatND& m );

CV_EXPORTS Scalar mean(const MatND& m);
CV_EXPORTS Scalar mean(const MatND& m, const MatND& mask);
CV_EXPORTS void meanStdDev(const MatND& m, Scalar& mean, Scalar& stddev, const MatND& mask=MatND());
CV_EXPORTS double norm(const MatND& a, int normType=NORM_L2, const MatND& mask=MatND());
CV_EXPORTS double norm(const MatND& a, const MatND& b,
                       int normType=NORM_L2, const MatND& mask=MatND());
CV_EXPORTS void normalize( const MatND& a, MatND& b, double alpha=1, double beta=0,
                           int norm_type=NORM_L2, int rtype=-1, const MatND& mask=MatND());

CV_EXPORTS void minMaxLoc(const MatND& a, double* minVal,
                       double* maxVal, int* minIdx=0, int* maxIdx=0,
                       const MatND& mask=MatND());

CV_EXPORTS void merge(const MatND* mvbegin, size_t count, MatND& dst);
CV_EXPORTS void split(const MatND& m, MatND* mv);
CV_EXPORTS void mixChannels(const MatND* src, int nsrcs, MatND* dst, int ndsts,
                            const int* fromTo, size_t npairs);

CV_EXPORTS void bitwise_and(const MatND& a, const MatND& b, MatND& c, const MatND& mask=MatND());
CV_EXPORTS void bitwise_or(const MatND& a, const MatND& b, MatND& c, const MatND& mask=MatND());
CV_EXPORTS void bitwise_xor(const MatND& a, const MatND& b, MatND& c, const MatND& mask=MatND());
CV_EXPORTS void bitwise_and(const MatND& a, const Scalar& s, MatND& c, const MatND& mask=MatND());
CV_EXPORTS void bitwise_or(const MatND& a, const Scalar& s, MatND& c, const MatND& mask=MatND());
CV_EXPORTS void bitwise_xor(const MatND& a, const Scalar& s, MatND& c, const MatND& mask=MatND());
CV_EXPORTS void bitwise_not(const MatND& a, MatND& c);
CV_EXPORTS void absdiff(const MatND& a, const MatND& b, MatND& c);
CV_EXPORTS void absdiff(const MatND& a, const Scalar& s, MatND& c);
CV_EXPORTS void inRange(const MatND& src, const MatND& lowerb,
                        const MatND& upperb, MatND& dst);
CV_EXPORTS void inRange(const MatND& src, const Scalar& lowerb,
                        const Scalar& upperb, MatND& dst);
CV_EXPORTS void compare(const MatND& a, const MatND& b, MatND& c, int cmpop);
CV_EXPORTS void compare(const MatND& a, double s, MatND& c, int cmpop);
CV_EXPORTS void min(const MatND& a, const MatND& b, MatND& c);
CV_EXPORTS void min(const MatND& a, double alpha, MatND& c);
CV_EXPORTS void max(const MatND& a, const MatND& b, MatND& c);
CV_EXPORTS void max(const MatND& a, double alpha, MatND& c);

CV_EXPORTS void sqrt(const MatND& a, MatND& b);
CV_EXPORTS void pow(const MatND& a, double power, MatND& b);
CV_EXPORTS void exp(const MatND& a, MatND& b);
CV_EXPORTS void log(const MatND& a, MatND& b);
CV_EXPORTS bool checkRange(const MatND& a, bool quiet=true, int* idx=0,
                           double minVal=-DBL_MAX, double maxVal=DBL_MAX);

typedef void (*ConvertData)(const void* from, void* to, int cn);
typedef void (*ConvertScaleData)(const void* from, void* to, int cn, double alpha, double beta);

CV_EXPORTS ConvertData getConvertElem(int fromType, int toType);
CV_EXPORTS ConvertScaleData getConvertScaleElem(int fromType, int toType);

template<typename _Tp> class CV_EXPORTS MatND_ : public MatND
{
public:
    typedef _Tp value_type;
    typedef typename DataType<_Tp>::channel_type channel_type;

    MatND_();
    MatND_(int dims, const int* _sizes);
    MatND_(int dims, const int* _sizes, const _Tp& _s);
    MatND_(const MatND& m);
    MatND_(const MatND_& m);
    MatND_(const MatND_& m, const Range* ranges);
    MatND_(const CvMatND* m, bool copyData=false);
    MatND_& operator = (const MatND& m);
    MatND_& operator = (const MatND_& m);
    MatND_& operator = (const _Tp& s);

    void create(int dims, const int* _sizes);
    template<typename T2> operator MatND_<T2>() const;
    MatND_ clone() const;
    MatND_ operator()(const Range* ranges) const;

    size_t elemSize() const;
    size_t elemSize1() const;
    int type() const;
    int depth() const;
    int channels() const;
    size_t stepT(int i) const;
    size_t step1(int i) const;

    _Tp& operator ()(const int* idx);
    const _Tp& operator ()(const int* idx) const;

    _Tp& operator ()(int idx0);
    const _Tp& operator ()(int idx0) const;
    _Tp& operator ()(int idx0, int idx1);
    const _Tp& operator ()(int idx0, int idx1) const;
    _Tp& operator ()(int idx0, int idx1, int idx2);
    const _Tp& operator ()(int idx0, int idx1, int idx2) const;
};

/////////////////////////// multi-dimensional sparse matrix //////////////////////////

class SparseMatIterator;
class SparseMatConstIterator;
template<typename _Tp> class SparseMatIterator_;
template<typename _Tp> class SparseMatConstIterator_;

class CV_EXPORTS SparseMat
{
public:
    typedef SparseMatIterator iterator;
    typedef SparseMatConstIterator const_iterator;

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

    // sparse matrix node - element of a hash table
    struct CV_EXPORTS Node
    {
        size_t hashval;
        size_t next;
        int idx[CV_MAX_DIM];
    };

    ////////// constructors and destructor //////////
    // default constructor
    SparseMat();
    // creates matrix of the specified size and type
    SparseMat(int dims, const int* _sizes, int _type);
    // copy constructor
    SparseMat(const SparseMat& m);
    // converts dense 2d matrix to the sparse form,
    // if try1d is true and matrix is a single-column matrix (Nx1),
    // then the sparse matrix will be 1-dimensional.
    SparseMat(const Mat& m, bool try1d=false);
    // converts dense n-d matrix to the sparse form
    SparseMat(const MatND& m);
    // converts old-style sparse matrix to the new-style.
    // all the data is copied, so that "m" can be safely
    // deleted after the conversion
    SparseMat(const CvSparseMat* m);
    // destructor
    ~SparseMat();
    
    ///////// assignment operations /////////// 
    
    // this is O(1) operation; no data is copied
    SparseMat& operator = (const SparseMat& m);
    // (equivalent to the corresponding constructor with try1d=false)
    SparseMat& operator = (const Mat& m);
    SparseMat& operator = (const MatND& m);

    // creates full copy of the matrix
    SparseMat clone() const;
    
    // copy all the data to the destination matrix.
    // the destination will be reallocated if needed.
    void copyTo( SparseMat& m ) const;
    // converts 1D or 2D sparse matrix to dense 2D matrix.
    // If the sparse matrix is 1D, then the result will
    // be a single-column matrix.
    void copyTo( Mat& m ) const;
    // converts arbitrary sparse matrix to dense matrix.
    // watch out the memory!
    void copyTo( MatND& m ) const;
    // multiplies all the matrix elements by the specified scalar
    void convertTo( SparseMat& m, int rtype, double alpha=1 ) const;
    // converts sparse matrix to dense matrix with optional type conversion and scaling.
    // When rtype=-1, the destination element type will be the same
    // as the sparse matrix element type.
    // Otherwise rtype will specify the depth and
    // the number of channels will remain the same is in the sparse matrix
    void convertTo( Mat& m, int rtype, double alpha=1, double beta=0 ) const;
    void convertTo( MatND& m, int rtype, double alpha=1, double beta=0 ) const;

    // not used now
    void assignTo( SparseMat& m, int type=-1 ) const;

    // reallocates sparse matrix. If it was already of the proper size and type,
    // it is simply cleared with clear(), otherwise,
    // the old matrix is released (using release()) and the new one is allocated.
    void create(int dims, const int* _sizes, int _type);
    // sets all the matrix elements to 0, which means clearing the hash table.
    void clear();
    // manually increases reference counter to the header.
    void addref();
    // decreses the header reference counter, when it reaches 0,
    // the header and all the underlying data are deallocated.
    void release();

    // converts sparse matrix to the old-style representation.
    // all the elements are copied.
    operator CvSparseMat*() const;
    // size of each element in bytes
    // (the matrix nodes will be bigger because of
    //  element indices and other SparseMat::Node elements).
    size_t elemSize() const;
    // elemSize()/channels()
    size_t elemSize1() const;
    
    // the same is in Mat and MatND
    int type() const;
    int depth() const;
    int channels() const;
    
    // returns the array of sizes and 0 if the matrix is not allocated
    const int* size() const;
    // returns i-th size (or 0)
    int size(int i) const;
    // returns the matrix dimensionality
    int dims() const;
    // returns the number of non-zero elements
    size_t nzcount() const;
    
    // compute element hash value from the element indices:
    // 1D case
    size_t hash(int i0) const;
    // 2D case
    size_t hash(int i0, int i1) const;
    // 3D case
    size_t hash(int i0, int i1, int i2) const;
    // n-D case
    size_t hash(const int* idx) const;
    
    // low-level element-acccess functions,
    // special variants for 1D, 2D, 3D cases and the generic one for n-D case.
    //
    // return pointer to the matrix element.
    //  if the element is there (it's non-zero), the pointer to it is returned
    //  if it's not there and createMissing=false, NULL pointer is returned
    //  if it's not there and createMissing=true, then the new element
    //    is created and initialized with 0. Pointer to it is returned
    //  If the optional hashval pointer is not NULL, the element hash value is
    //  not computed, but *hashval is taken instead.
    uchar* ptr(int i0, bool createMissing, size_t* hashval=0);
    uchar* ptr(int i0, int i1, bool createMissing, size_t* hashval=0);
    uchar* ptr(int i0, int i1, int i2, bool createMissing, size_t* hashval=0);
    uchar* ptr(const int* idx, bool createMissing, size_t* hashval=0);

    // higher-level element access functions:
    // ref<_Tp>(i0,...[,hashval]) - equivalent to *(_Tp*)ptr(i0,...true[,hashval]).
    //    always return valid reference to the element.
    //    If it's did not exist, it is created.
    // find<_Tp>(i0,...[,hashval]) - equivalent to (_const Tp*)ptr(i0,...false[,hashval]).
    //    return pointer to the element or NULL pointer if the element is not there.
    // value<_Tp>(i0,...[,hashval]) - equivalent to
    //    { const _Tp* p = find<_Tp>(i0,...[,hashval]); return p ? *p : _Tp(); }
    //    that is, 0 is returned when the element is not there.
    // note that _Tp must match the actual matrix type -
    // the functions do not do any on-fly type conversion
    
    // 1D case
    template<typename _Tp> _Tp& ref(int i0, size_t* hashval=0);   
    template<typename _Tp> _Tp value(int i0, size_t* hashval=0) const;
    template<typename _Tp> const _Tp* find(int i0, size_t* hashval=0) const;

    // 2D case
    template<typename _Tp> _Tp& ref(int i0, int i1, size_t* hashval=0);   
    template<typename _Tp> _Tp value(int i0, int i1, size_t* hashval=0) const;
    template<typename _Tp> const _Tp* find(int i0, int i1, size_t* hashval=0) const;
    
    // 3D case
    template<typename _Tp> _Tp& ref(int i0, int i1, int i2, size_t* hashval=0);
    template<typename _Tp> _Tp value(int i0, int i1, int i2, size_t* hashval=0) const;
    template<typename _Tp> const _Tp* find(int i0, int i1, int i2, size_t* hashval=0) const;

    // n-D case
    template<typename _Tp> _Tp& ref(const int* idx, size_t* hashval=0);
    template<typename _Tp> _Tp value(const int* idx, size_t* hashval=0) const;
    template<typename _Tp> const _Tp* find(const int* idx, size_t* hashval=0) const;

    // erase the specified matrix element.
    // When there is no such element, the methods do nothing
    void erase(int i0, int i1, size_t* hashval=0);
    void erase(int i0, int i1, int i2, size_t* hashval=0);
    void erase(const int* idx, size_t* hashval=0);

    // return the matrix iterators,
    //   pointing to the first sparse matrix element,
    SparseMatIterator begin();
    SparseMatConstIterator begin() const;
    //   ... or to the point after the last sparse matrix element
    SparseMatIterator end();
    SparseMatConstIterator end() const;
    
    // and the template forms of the above methods.
    // _Tp must match the actual matrix type.
    template<typename _Tp> SparseMatIterator_<_Tp> begin();
    template<typename _Tp> SparseMatConstIterator_<_Tp> begin() const;
    template<typename _Tp> SparseMatIterator_<_Tp> end();
    template<typename _Tp> SparseMatConstIterator_<_Tp> end() const;

    // return value stored in the sparse martix node
    template<typename _Tp> _Tp& value(Node* n);
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


CV_EXPORTS void minMaxLoc(const SparseMat& a, double* minVal,
                          double* maxVal, int* minIdx=0, int* maxIdx=0);
CV_EXPORTS double norm( const SparseMat& src, int normType );
CV_EXPORTS void normalize( const SparseMat& src, SparseMat& dst, double alpha, int normType );

class CV_EXPORTS SparseMatConstIterator
{
public:
    SparseMatConstIterator();
    SparseMatConstIterator(const SparseMat* _m);
    SparseMatConstIterator(const SparseMatConstIterator& it);

    SparseMatConstIterator& operator = (const SparseMatConstIterator& it);

    template<typename _Tp> const _Tp& value() const;
    const SparseMat::Node* node() const;
    
    SparseMatConstIterator& operator --();
    SparseMatConstIterator operator --(int);
    SparseMatConstIterator& operator ++();
    SparseMatConstIterator operator ++(int);
    
    void seekEnd();

    const SparseMat* m;
    size_t hashidx;
    uchar* ptr;
};

class CV_EXPORTS SparseMatIterator : public SparseMatConstIterator
{
public:
    SparseMatIterator();
    SparseMatIterator(SparseMat* _m);
    SparseMatIterator(SparseMat* _m, const int* idx);
    SparseMatIterator(const SparseMatIterator& it);

    SparseMatIterator& operator = (const SparseMatIterator& it);
    template<typename _Tp> _Tp& value() const;
    SparseMat::Node* node() const;
    
    SparseMatIterator& operator ++();
    SparseMatIterator operator ++(int);
};


template<typename _Tp> class CV_EXPORTS SparseMat_ : public SparseMat
{
public:
    typedef SparseMatIterator_<_Tp> iterator;
    typedef SparseMatConstIterator_<_Tp> const_iterator;

    SparseMat_();
    SparseMat_(int dims, const int* _sizes);
    SparseMat_(const SparseMat& m);
    SparseMat_(const SparseMat_& m);
    SparseMat_(const Mat& m);
    SparseMat_(const MatND& m);
    SparseMat_(const CvSparseMat* m);
    SparseMat_& operator = (const SparseMat& m);
    SparseMat_& operator = (const SparseMat_& m);
    SparseMat_& operator = (const Mat& m);
    SparseMat_& operator = (const MatND& m);

    SparseMat_ clone() const;
    void create(int dims, const int* _sizes);
    operator CvSparseMat*() const;

    int type() const;
    int depth() const;
    int channels() const;
    
    _Tp& ref(int i0, size_t* hashval=0);
    _Tp operator()(int i0, size_t* hashval=0) const;
    _Tp& ref(int i0, int i1, size_t* hashval=0);
    _Tp operator()(int i0, int i1, size_t* hashval=0) const;
    _Tp& ref(int i0, int i1, int i2, size_t* hashval=0);
    _Tp operator()(int i0, int i1, int i2, size_t* hashval=0) const;
    _Tp& ref(const int* idx, size_t* hashval=0);
    _Tp operator()(const int* idx, size_t* hashval=0) const;

    SparseMatIterator_<_Tp> begin();
    SparseMatConstIterator_<_Tp> begin() const;
    SparseMatIterator_<_Tp> end();
    SparseMatConstIterator_<_Tp> end() const;
};

template<typename _Tp> class CV_EXPORTS SparseMatConstIterator_ : public SparseMatConstIterator
{
public:
    typedef std::forward_iterator_tag iterator_category;
    
    SparseMatConstIterator_();
    SparseMatConstIterator_(const SparseMat_<_Tp>* _m);
    SparseMatConstIterator_(const SparseMatConstIterator_& it);

    SparseMatConstIterator_& operator = (const SparseMatConstIterator_& it);
    const _Tp& operator *() const;
    
    SparseMatConstIterator_& operator ++();
    SparseMatConstIterator_ operator ++(int);
};

template<typename _Tp> class CV_EXPORTS SparseMatIterator_ : public SparseMatConstIterator_<_Tp>
{
public:
    typedef std::forward_iterator_tag iterator_category;
    
    SparseMatIterator_();
    SparseMatIterator_(SparseMat_<_Tp>* _m);
    SparseMatIterator_(const SparseMatIterator_& it);

    SparseMatIterator_& operator = (const SparseMatIterator_& it);
    _Tp& operator *() const;
    
    SparseMatIterator_& operator ++();
    SparseMatIterator_ operator ++(int);
};

//////////////////// Fast Nearest-Neighbor Search Structure ////////////////////

class CV_EXPORTS KDTree
{
public:
    struct Node
    {
        Node() : idx(-1), left(-1), right(-1), boundary(0.f) {}
        Node(int _idx, int _left, int _right, float _boundary)
            : idx(_idx), left(_left), right(_right), boundary(_boundary) {}
        int idx;            // split dimension; >=0 for nodes (dim),
                            // < 0 for leaves (index of the point)
        int left, right;    // node indices of left and right branches
        float boundary;     // left if vec[dim]<=boundary, otherwise right
    };

    KDTree();
    KDTree(const Mat& _points, bool copyAndReorderPoints=false);
    void build(const Mat& _points, bool copyAndReorderPoints=false);

    int findNearest(const float* vec, int K, int Emax, int* neighborsIdx,
                    Mat* neighbors=0, float* dist=0) const;
    int findNearest(const float* vec, int K, int Emax,
                    vector<int>* neighborsIdx,
                    Mat* neighbors=0, vector<float>* dist=0) const;
    void findOrthoRange(const float* minBounds, const float* maxBounds,
                        vector<int>* neighborsIdx, Mat* neighbors=0) const;
    void getPoints(const int* idx, size_t nidx, Mat& pts) const;
    void getPoints(const Mat& idxs, Mat& pts) const;
    const float* getPoint(int ptidx) const;
    int dims() const;

    vector<Node> nodes;
    Mat points;
    int maxDepth;
    int normType;
};

//////////////////////////////////////// XML & YAML I/O ////////////////////////////////////

class CV_EXPORTS FileNode;

class CV_EXPORTS FileStorage
{
public:
    enum { READ=0, WRITE=1, APPEND=2 };
    enum { UNDEFINED=0, VALUE_EXPECTED=1, NAME_EXPECTED=2, INSIDE_MAP=4 };
    FileStorage();
    FileStorage(const string& filename, int flags);
    FileStorage(CvFileStorage* fs);
    virtual ~FileStorage();

    virtual bool open(const string& filename, int flags);
    virtual bool isOpened() const;
    virtual void release();

    FileNode getFirstTopLevelNode() const;
    FileNode root(int streamidx=0) const;
    FileNode operator[](const string& nodename) const;
    FileNode operator[](const char* nodename) const;

    CvFileStorage* operator *() { return fs; }
    const CvFileStorage* operator *() const { return fs; }
    void writeRaw( const string& fmt, const uchar* vec, size_t len );
    void writeObj( const string& name, const void* obj );

    static string getDefaultObjectName(const string& filename);

    Ptr<CvFileStorage> fs;
    string elname;
    vector<char> structs;
    int state;
};

class CV_EXPORTS FileNodeIterator;

class CV_EXPORTS FileNode
{
public:
    enum { NONE=0, INT=1, REAL=2, FLOAT=REAL, STR=3, STRING=STR, REF=4, SEQ=5, MAP=6, TYPE_MASK=7,
        FLOW=8, USER=16, EMPTY=32, NAMED=64 };
    FileNode();
    FileNode(const CvFileStorage* fs, const CvFileNode* node);
    FileNode(const FileNode& node);
    FileNode operator[](const string& nodename) const;
    FileNode operator[](const char* nodename) const;
    FileNode operator[](int i) const;
    int type() const;
    int rawDataSize(const string& fmt) const;
    bool empty() const;
    bool isNone() const;
    bool isSeq() const;
    bool isMap() const;
    bool isInt() const;
    bool isReal() const;
    bool isString() const;
    bool isNamed() const;
    string name() const;
    size_t size() const;
    operator int() const;
    operator float() const;
    operator double() const;
    operator string() const;
    
    CvFileNode* operator *();
    const CvFileNode* operator* () const;
    

    FileNodeIterator begin() const;
    FileNodeIterator end() const;

    void readRaw( const string& fmt, uchar* vec, size_t len ) const;
    void* readObj() const;

    // do not use wrapper pointer classes for better efficiency
    const CvFileStorage* fs;
    const CvFileNode* node;
};

class CV_EXPORTS FileNodeIterator
{
public:
    FileNodeIterator();
    FileNodeIterator(const CvFileStorage* fs, const CvFileNode* node, size_t ofs=0);
    FileNodeIterator(const FileNodeIterator& it);
    FileNode operator *() const;
    FileNode operator ->() const;

    FileNodeIterator& operator ++();
    FileNodeIterator operator ++(int);
    FileNodeIterator& operator --();
    FileNodeIterator operator --(int);
    FileNodeIterator& operator += (int);
    FileNodeIterator& operator -= (int);

    FileNodeIterator& readRaw( const string& fmt, uchar* vec,
                               size_t maxCount=(size_t)INT_MAX );

    const CvFileStorage* fs;
    const CvFileNode* container;
    CvSeqReader reader;
    size_t remaining;
};

////////////// convenient wrappers for operating old-style dynamic structures //////////////

// !!! NOTE that the wrappers are "thin", i.e. they do not call
// any element constructors/destructors

template<typename _Tp> class SeqIterator;

typedef Ptr<CvMemStorage> MemStorage;

template<typename _Tp> class CV_EXPORTS Seq
{
public:
    typedef SeqIterator<_Tp> iterator;
    typedef SeqIterator<_Tp> const_iterator;
    
    Seq();
    Seq(const CvSeq* seq);
    Seq(MemStorage& storage, int headerSize = sizeof(CvSeq));
    _Tp& operator [](int idx);
    const _Tp& operator[](int idx) const;
    SeqIterator<_Tp> begin() const;
    SeqIterator<_Tp> end() const;
    size_t size() const;
    int type() const;
    int depth() const;
    int channels() const;
    size_t elemSize() const;
    size_t index(const _Tp& elem) const;
    void push_back(const _Tp& elem);
    void push_front(const _Tp& elem);
    void push_back(const _Tp* elems, size_t count);
    void push_front(const _Tp* elems, size_t count);
    void insert(int idx, const _Tp& elem);
    void insert(int idx, const _Tp* elems, size_t count);
    void remove(int idx);
    void remove(const Range& r);
    
    _Tp& front();
    const _Tp& front() const;
    _Tp& back();
    const _Tp& back() const;
    bool empty() const;

    void clear();
    void pop_front();
    void pop_back();
    void pop_front(_Tp* elems, size_t count);
    void pop_back(_Tp* elems, size_t count);

    void copyTo(vector<_Tp>& vec, const Range& range=Range::all()) const;
    operator vector<_Tp>() const;
    
    CvSeq* seq;
};

template<typename _Tp> class CV_EXPORTS SeqIterator : public CvSeqReader
{
public:
    SeqIterator();
    SeqIterator(const Seq<_Tp>& seq, bool seekEnd=false);
    void seek(size_t pos);
    size_t tell() const;
    _Tp& operator *();
    const _Tp& operator *() const;
    SeqIterator& operator ++();
    SeqIterator operator ++(int) const;
    SeqIterator& operator --();
    SeqIterator operator --(int) const;

    SeqIterator& operator +=(int);
    SeqIterator& operator -=(int);

    // this is index of the current element module seq->total*2
    // (to distinguish between 0 and seq->total)
    int index;
};

}

#endif // __cplusplus

#include "opencv2/core/operations.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/flann.hpp"	// FLANN (Fast Library for Approximate Nearest Neighbors)

#endif /*__OPENCV_CORE_HPP__*/
