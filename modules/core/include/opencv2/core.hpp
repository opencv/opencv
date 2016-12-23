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
// Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2015, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
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

#ifndef OPENCV_CORE_HPP
#define OPENCV_CORE_HPP

#ifndef __cplusplus
#  error core.hpp header must be compiled as C++
#endif

#include "opencv2/core/cvdef.h"
#include "opencv2/core/version.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/core/traits.hpp"
#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/persistence.hpp"

/**
@defgroup core Core functionality
@{
    @defgroup core_basic Basic structures
    @defgroup core_c C structures and operations
    @{
        @defgroup core_c_glue Connections with C++
    @}
    @defgroup core_array Operations on arrays
    @defgroup core_xml XML/YAML Persistence
    @defgroup core_cluster Clustering
    @defgroup core_utils Utility and system functions and macros
    @{
        @defgroup core_utils_sse SSE utilities
        @defgroup core_utils_neon NEON utilities
    @}
    @defgroup core_opengl OpenGL interoperability
    @defgroup core_ipp Intel IPP Asynchronous C/C++ Converters
    @defgroup core_optim Optimization Algorithms
    @defgroup core_directx DirectX interoperability
    @defgroup core_eigen Eigen support
    @defgroup core_opencl OpenCL support
    @defgroup core_va_intel Intel VA-API/OpenCL (CL-VA) interoperability
    @defgroup core_hal Hardware Acceleration Layer
    @{
        @defgroup core_hal_functions Functions
        @defgroup core_hal_interface Interface
        @defgroup core_hal_intrin Universal intrinsics
        @{
            @defgroup core_hal_intrin_impl Private implementation helpers
        @}
    @}
@}
 */

namespace cv {

//! @addtogroup core_utils
//! @{

/*! @brief Class passed to an error.

This class encapsulates all or almost all necessary
information about the error happened in the program. The exception is
usually constructed and thrown implicitly via CV_Error and CV_Error_ macros.
@see error
 */
class CV_EXPORTS Exception : public std::exception
{
public:
    /*!
     Default constructor
     */
    Exception();
    /*!
     Full constructor. Normally the constuctor is not called explicitly.
     Instead, the macros CV_Error(), CV_Error_() and CV_Assert() are used.
    */
    Exception(int _code, const String& _err, const String& _func, const String& _file, int _line);
    virtual ~Exception() throw();

    /*!
     \return the error description and the context as a text string.
    */
    virtual const char *what() const throw();
    void formatMessage();

    String msg; ///< the formatted error message

    int code; ///< error code @see CVStatus
    String err; ///< error description
    String func; ///< function name. Available only when the compiler supports getting it
    String file; ///< source file name where the error has occured
    int line; ///< line number in the source file where the error has occured
};

/*! @brief Signals an error and raises the exception.

By default the function prints information about the error to stderr,
then it either stops if cv::setBreakOnError() had been called before or raises the exception.
It is possible to alternate error processing by using cv::redirectError().
@param exc the exception raisen.
@deprecated drop this version
 */
CV_EXPORTS void error( const Exception& exc );

enum SortFlags { SORT_EVERY_ROW    = 0, //!< each matrix row is sorted independently
                 SORT_EVERY_COLUMN = 1, //!< each matrix column is sorted
                                        //!< independently; this flag and the previous one are
                                        //!< mutually exclusive.
                 SORT_ASCENDING    = 0, //!< each matrix row is sorted in the ascending
                                        //!< order.
                 SORT_DESCENDING   = 16 //!< each matrix row is sorted in the
                                        //!< descending order; this flag and the previous one are also
                                        //!< mutually exclusive.
               };

//! @} core_utils

//! @addtogroup core
//! @{

//! Covariation flags
enum CovarFlags {
    /** The output covariance matrix is calculated as:
       \f[\texttt{scale}   \cdot  [  \texttt{vects}  [0]-  \texttt{mean}  , \texttt{vects}  [1]-  \texttt{mean}  ,...]^T  \cdot  [ \texttt{vects}  [0]- \texttt{mean}  , \texttt{vects}  [1]- \texttt{mean}  ,...],\f]
       The covariance matrix will be nsamples x nsamples. Such an unusual covariance matrix is used
       for fast PCA of a set of very large vectors (see, for example, the EigenFaces technique for
       face recognition). Eigenvalues of this "scrambled" matrix match the eigenvalues of the true
       covariance matrix. The "true" eigenvectors can be easily calculated from the eigenvectors of
       the "scrambled" covariance matrix. */
    COVAR_SCRAMBLED = 0,
    /**The output covariance matrix is calculated as:
        \f[\texttt{scale}   \cdot  [  \texttt{vects}  [0]-  \texttt{mean}  , \texttt{vects}  [1]-  \texttt{mean}  ,...]  \cdot  [ \texttt{vects}  [0]- \texttt{mean}  , \texttt{vects}  [1]- \texttt{mean}  ,...]^T,\f]
        covar will be a square matrix of the same size as the total number of elements in each input
        vector. One and only one of COVAR_SCRAMBLED and COVAR_NORMAL must be specified.*/
    COVAR_NORMAL    = 1,
    /** If the flag is specified, the function does not calculate mean from
        the input vectors but, instead, uses the passed mean vector. This is useful if mean has been
        pre-calculated or known in advance, or if the covariance matrix is calculated by parts. In
        this case, mean is not a mean vector of the input sub-set of vectors but rather the mean
        vector of the whole set.*/
    COVAR_USE_AVG   = 2,
    /** If the flag is specified, the covariance matrix is scaled. In the
        "normal" mode, scale is 1./nsamples . In the "scrambled" mode, scale is the reciprocal of the
        total number of elements in each input vector. By default (if the flag is not specified), the
        covariance matrix is not scaled ( scale=1 ).*/
    COVAR_SCALE     = 4,
    /** If the flag is
        specified, all the input vectors are stored as rows of the samples matrix. mean should be a
        single-row vector in this case.*/
    COVAR_ROWS      = 8,
    /** If the flag is
        specified, all the input vectors are stored as columns of the samples matrix. mean should be a
        single-column vector in this case.*/
    COVAR_COLS      = 16
};

//! k-Means flags
enum KmeansFlags {
    /** Select random initial centers in each attempt.*/
    KMEANS_RANDOM_CENTERS     = 0,
    /** Use kmeans++ center initialization by Arthur and Vassilvitskii [Arthur2007].*/
    KMEANS_PP_CENTERS         = 2,
    /** During the first (and possibly the only) attempt, use the
        user-supplied labels instead of computing them from the initial centers. For the second and
        further attempts, use the random or semi-random centers. Use one of KMEANS_\*_CENTERS flag
        to specify the exact method.*/
    KMEANS_USE_INITIAL_LABELS = 1
};

//! type of line
enum LineTypes {
    FILLED  = -1,
    LINE_4  = 4, //!< 4-connected line
    LINE_8  = 8, //!< 8-connected line
    LINE_AA = 16 //!< antialiased line
};

//! Only a subset of Hershey fonts
//! <http://sources.isc.org/utils/misc/hershey-font.txt> are supported
enum HersheyFonts {
    FONT_HERSHEY_SIMPLEX        = 0, //!< normal size sans-serif font
    FONT_HERSHEY_PLAIN          = 1, //!< small size sans-serif font
    FONT_HERSHEY_DUPLEX         = 2, //!< normal size sans-serif font (more complex than FONT_HERSHEY_SIMPLEX)
    FONT_HERSHEY_COMPLEX        = 3, //!< normal size serif font
    FONT_HERSHEY_TRIPLEX        = 4, //!< normal size serif font (more complex than FONT_HERSHEY_COMPLEX)
    FONT_HERSHEY_COMPLEX_SMALL  = 5, //!< smaller version of FONT_HERSHEY_COMPLEX
    FONT_HERSHEY_SCRIPT_SIMPLEX = 6, //!< hand-writing style font
    FONT_HERSHEY_SCRIPT_COMPLEX = 7, //!< more complex variant of FONT_HERSHEY_SCRIPT_SIMPLEX
    FONT_ITALIC                 = 16 //!< flag for italic font
};

enum ReduceTypes { REDUCE_SUM = 0, //!< the output is the sum of all rows/columns of the matrix.
                   REDUCE_AVG = 1, //!< the output is the mean vector of all rows/columns of the matrix.
                   REDUCE_MAX = 2, //!< the output is the maximum (column/row-wise) of all rows/columns of the matrix.
                   REDUCE_MIN = 3  //!< the output is the minimum (column/row-wise) of all rows/columns of the matrix.
                 };


/** @brief Swaps two matrices
*/
CV_EXPORTS void swap(Mat& a, Mat& b);
/** @overload */
CV_EXPORTS void swap( UMat& a, UMat& b );

//! @} core

//! @addtogroup core_array
//! @{

/** @brief Computes the source location of an extrapolated pixel.

The function computes and returns the coordinate of a donor pixel corresponding to the specified
extrapolated pixel when using the specified extrapolation border mode. For example, if you use
cv::BORDER_WRAP mode in the horizontal direction, cv::BORDER_REFLECT_101 in the vertical direction and
want to compute value of the "virtual" pixel Point(-5, 100) in a floating-point image img , it
looks like:
@code{.cpp}
    float val = img.at<float>(borderInterpolate(100, img.rows, cv::BORDER_REFLECT_101),
                              borderInterpolate(-5, img.cols, cv::BORDER_WRAP));
@endcode
Normally, the function is not called directly. It is used inside filtering functions and also in
copyMakeBorder.
@param p 0-based coordinate of the extrapolated pixel along one of the axes, likely \<0 or \>= len
@param len Length of the array along the corresponding axis.
@param borderType Border type, one of the cv::BorderTypes, except for cv::BORDER_TRANSPARENT and
cv::BORDER_ISOLATED . When borderType==cv::BORDER_CONSTANT , the function always returns -1, regardless
of p and len.

@sa copyMakeBorder
*/
CV_EXPORTS_W int borderInterpolate(int p, int len, int borderType);

/** @brief Forms a border around an image.

The function copies the source image into the middle of the destination image. The areas to the
left, to the right, above and below the copied source image will be filled with extrapolated
pixels. This is not what filtering functions based on it do (they extrapolate pixels on-fly), but
what other more complex functions, including your own, may do to simplify image boundary handling.

The function supports the mode when src is already in the middle of dst . In this case, the
function does not copy src itself but simply constructs the border, for example:

@code{.cpp}
    // let border be the same in all directions
    int border=2;
    // constructs a larger image to fit both the image and the border
    Mat gray_buf(rgb.rows + border*2, rgb.cols + border*2, rgb.depth());
    // select the middle part of it w/o copying data
    Mat gray(gray_canvas, Rect(border, border, rgb.cols, rgb.rows));
    // convert image from RGB to grayscale
    cvtColor(rgb, gray, COLOR_RGB2GRAY);
    // form a border in-place
    copyMakeBorder(gray, gray_buf, border, border,
                   border, border, BORDER_REPLICATE);
    // now do some custom filtering ...
    ...
@endcode
@note When the source image is a part (ROI) of a bigger image, the function will try to use the
pixels outside of the ROI to form a border. To disable this feature and always do extrapolation, as
if src was not a ROI, use borderType | BORDER_ISOLATED.

@param src Source image.
@param dst Destination image of the same type as src and the size Size(src.cols+left+right,
src.rows+top+bottom) .
@param top
@param bottom
@param left
@param right Parameter specifying how many pixels in each direction from the source image rectangle
to extrapolate. For example, top=1, bottom=1, left=1, right=1 mean that 1 pixel-wide border needs
to be built.
@param borderType Border type. See borderInterpolate for details.
@param value Border value if borderType==BORDER_CONSTANT .

@sa  borderInterpolate
*/
CV_EXPORTS_W void copyMakeBorder(InputArray src, OutputArray dst,
                                 int top, int bottom, int left, int right,
                                 int borderType, const Scalar& value = Scalar() );

/** @brief Calculates the per-element sum of two arrays or an array and a scalar.

The function add calculates:
- Sum of two arrays when both input arrays have the same size and the same number of channels:
\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) +  \texttt{src2}(I)) \quad \texttt{if mask}(I) \ne0\f]
- Sum of an array and a scalar when src2 is constructed from Scalar or has the same number of
elements as `src1.channels()`:
\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) +  \texttt{src2} ) \quad \texttt{if mask}(I) \ne0\f]
- Sum of a scalar and an array when src1 is constructed from Scalar or has the same number of
elements as `src2.channels()`:
\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1} +  \texttt{src2}(I) ) \quad \texttt{if mask}(I) \ne0\f]
where `I` is a multi-dimensional index of array elements. In case of multi-channel arrays, each
channel is processed independently.

The first function in the list above can be replaced with matrix expressions:
@code{.cpp}
    dst = src1 + src2;
    dst += src1; // equivalent to add(dst, src1, dst);
@endcode
The input arrays and the output array can all have the same or different depths. For example, you
can add a 16-bit unsigned array to a 8-bit signed array and store the sum as a 32-bit
floating-point array. Depth of the output array is determined by the dtype parameter. In the second
and third cases above, as well as in the first case, when src1.depth() == src2.depth(), dtype can
be set to the default -1. In this case, the output array will have the same depth as the input
array, be it src1, src2 or both.
@note Saturation is not applied when the output array has the depth CV_32S. You may even get
result of an incorrect sign in the case of overflow.
@param src1 first input array or a scalar.
@param src2 second input array or a scalar.
@param dst output array that has the same size and number of channels as the input array(s); the
depth is defined by dtype or src1/src2.
@param mask optional operation mask - 8-bit single channel array, that specifies elements of the
output array to be changed.
@param dtype optional depth of the output array (see the discussion below).
@sa subtract, addWeighted, scaleAdd, Mat::convertTo
*/
CV_EXPORTS_W void add(InputArray src1, InputArray src2, OutputArray dst,
                      InputArray mask = noArray(), int dtype = -1);

/** @brief Calculates the per-element difference between two arrays or array and a scalar.

The function subtract calculates:
- Difference between two arrays, when both input arrays have the same size and the same number of
channels:
    \f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) -  \texttt{src2}(I)) \quad \texttt{if mask}(I) \ne0\f]
- Difference between an array and a scalar, when src2 is constructed from Scalar or has the same
number of elements as `src1.channels()`:
    \f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) -  \texttt{src2} ) \quad \texttt{if mask}(I) \ne0\f]
- Difference between a scalar and an array, when src1 is constructed from Scalar or has the same
number of elements as `src2.channels()`:
    \f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1} -  \texttt{src2}(I) ) \quad \texttt{if mask}(I) \ne0\f]
- The reverse difference between a scalar and an array in the case of `SubRS`:
    \f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src2} -  \texttt{src1}(I) ) \quad \texttt{if mask}(I) \ne0\f]
where I is a multi-dimensional index of array elements. In case of multi-channel arrays, each
channel is processed independently.

The first function in the list above can be replaced with matrix expressions:
@code{.cpp}
    dst = src1 - src2;
    dst -= src1; // equivalent to subtract(dst, src1, dst);
@endcode
The input arrays and the output array can all have the same or different depths. For example, you
can subtract to 8-bit unsigned arrays and store the difference in a 16-bit signed array. Depth of
the output array is determined by dtype parameter. In the second and third cases above, as well as
in the first case, when src1.depth() == src2.depth(), dtype can be set to the default -1. In this
case the output array will have the same depth as the input array, be it src1, src2 or both.
@note Saturation is not applied when the output array has the depth CV_32S. You may even get
result of an incorrect sign in the case of overflow.
@param src1 first input array or a scalar.
@param src2 second input array or a scalar.
@param dst output array of the same size and the same number of channels as the input array.
@param mask optional operation mask; this is an 8-bit single channel array that specifies elements
of the output array to be changed.
@param dtype optional depth of the output array
@sa  add, addWeighted, scaleAdd, Mat::convertTo
  */
CV_EXPORTS_W void subtract(InputArray src1, InputArray src2, OutputArray dst,
                           InputArray mask = noArray(), int dtype = -1);


/** @brief Calculates the per-element scaled product of two arrays.

The function multiply calculates the per-element product of two arrays:

\f[\texttt{dst} (I)= \texttt{saturate} ( \texttt{scale} \cdot \texttt{src1} (I)  \cdot \texttt{src2} (I))\f]

There is also a @ref MatrixExpressions -friendly variant of the first function. See Mat::mul .

For a not-per-element matrix product, see gemm .

@note Saturation is not applied when the output array has the depth
CV_32S. You may even get result of an incorrect sign in the case of
overflow.
@param src1 first input array.
@param src2 second input array of the same size and the same type as src1.
@param dst output array of the same size and type as src1.
@param scale optional scale factor.
@param dtype optional depth of the output array
@sa add, subtract, divide, scaleAdd, addWeighted, accumulate, accumulateProduct, accumulateSquare,
Mat::convertTo
*/
CV_EXPORTS_W void multiply(InputArray src1, InputArray src2,
                           OutputArray dst, double scale = 1, int dtype = -1);

/** @brief Performs per-element division of two arrays or a scalar by an array.

The function cv::divide divides one array by another:
\f[\texttt{dst(I) = saturate(src1(I)*scale/src2(I))}\f]
or a scalar by an array when there is no src1 :
\f[\texttt{dst(I) = saturate(scale/src2(I))}\f]

When src2(I) is zero, dst(I) will also be zero. Different channels of
multi-channel arrays are processed independently.

@note Saturation is not applied when the output array has the depth CV_32S. You may even get
result of an incorrect sign in the case of overflow.
@param src1 first input array.
@param src2 second input array of the same size and type as src1.
@param scale scalar factor.
@param dst output array of the same size and type as src2.
@param dtype optional depth of the output array; if -1, dst will have depth src2.depth(), but in
case of an array-by-array division, you can only pass -1 when src1.depth()==src2.depth().
@sa  multiply, add, subtract
*/
CV_EXPORTS_W void divide(InputArray src1, InputArray src2, OutputArray dst,
                         double scale = 1, int dtype = -1);

/** @overload */
CV_EXPORTS_W void divide(double scale, InputArray src2,
                         OutputArray dst, int dtype = -1);

/** @brief Calculates the sum of a scaled array and another array.

The function scaleAdd is one of the classical primitive linear algebra operations, known as DAXPY
or SAXPY in [BLAS](http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms). It calculates
the sum of a scaled array and another array:
\f[\texttt{dst} (I)= \texttt{scale} \cdot \texttt{src1} (I) +  \texttt{src2} (I)\f]
The function can also be emulated with a matrix expression, for example:
@code{.cpp}
    Mat A(3, 3, CV_64F);
    ...
    A.row(0) = A.row(1)*2 + A.row(2);
@endcode
@param src1 first input array.
@param alpha scale factor for the first array.
@param src2 second input array of the same size and type as src1.
@param dst output array of the same size and type as src1.
@sa add, addWeighted, subtract, Mat::dot, Mat::convertTo
*/
CV_EXPORTS_W void scaleAdd(InputArray src1, double alpha, InputArray src2, OutputArray dst);

/** @brief Calculates the weighted sum of two arrays.

The function addWeighted calculates the weighted sum of two arrays as follows:
\f[\texttt{dst} (I)= \texttt{saturate} ( \texttt{src1} (I)* \texttt{alpha} +  \texttt{src2} (I)* \texttt{beta} +  \texttt{gamma} )\f]
where I is a multi-dimensional index of array elements. In case of multi-channel arrays, each
channel is processed independently.
The function can be replaced with a matrix expression:
@code{.cpp}
    dst = src1*alpha + src2*beta + gamma;
@endcode
@note Saturation is not applied when the output array has the depth CV_32S. You may even get
result of an incorrect sign in the case of overflow.
@param src1 first input array.
@param alpha weight of the first array elements.
@param src2 second input array of the same size and channel number as src1.
@param beta weight of the second array elements.
@param gamma scalar added to each sum.
@param dst output array that has the same size and number of channels as the input arrays.
@param dtype optional depth of the output array; when both input arrays have the same depth, dtype
can be set to -1, which will be equivalent to src1.depth().
@sa  add, subtract, scaleAdd, Mat::convertTo
*/
CV_EXPORTS_W void addWeighted(InputArray src1, double alpha, InputArray src2,
                              double beta, double gamma, OutputArray dst, int dtype = -1);

/** @brief Scales, calculates absolute values, and converts the result to 8-bit.

On each element of the input array, the function convertScaleAbs
performs three operations sequentially: scaling, taking an absolute
value, conversion to an unsigned 8-bit type:
\f[\texttt{dst} (I)= \texttt{saturate\_cast<uchar>} (| \texttt{src} (I)* \texttt{alpha} +  \texttt{beta} |)\f]
In case of multi-channel arrays, the function processes each channel
independently. When the output is not 8-bit, the operation can be
emulated by calling the Mat::convertTo method (or by using matrix
expressions) and then by calculating an absolute value of the result.
For example:
@code{.cpp}
    Mat_<float> A(30,30);
    randu(A, Scalar(-100), Scalar(100));
    Mat_<float> B = A*5 + 3;
    B = abs(B);
    // Mat_<float> B = abs(A*5+3) will also do the job,
    // but it will allocate a temporary matrix
@endcode
@param src input array.
@param dst output array.
@param alpha optional scale factor.
@param beta optional delta added to the scaled values.
@sa  Mat::convertTo, cv::abs(const Mat&)
*/
CV_EXPORTS_W void convertScaleAbs(InputArray src, OutputArray dst,
                                  double alpha = 1, double beta = 0);

/** @brief Converts an array to half precision floating number.

This function converts FP32 (single precision floating point) from/to FP16 (half precision floating point).  The input array has to have type of CV_32F or
CV_16S to represent the bit depth.  If the input array is neither of them, the function will raise an error.
The format of half precision floating point is defined in IEEE 754-2008.

@param src input array.
@param dst output array.
*/
CV_EXPORTS_W void convertFp16(InputArray src, OutputArray dst);

/** @brief Performs a look-up table transform of an array.

The function LUT fills the output array with values from the look-up table. Indices of the entries
are taken from the input array. That is, the function processes each element of src as follows:
\f[\texttt{dst} (I)  \leftarrow \texttt{lut(src(I) + d)}\f]
where
\f[d =  \fork{0}{if \(\texttt{src}\) has depth \(\texttt{CV_8U}\)}{128}{if \(\texttt{src}\) has depth \(\texttt{CV_8S}\)}\f]
@param src input array of 8-bit elements.
@param lut look-up table of 256 elements; in case of multi-channel input array, the table should
either have a single channel (in this case the same table is used for all channels) or the same
number of channels as in the input array.
@param dst output array of the same size and number of channels as src, and the same depth as lut.
@sa  convertScaleAbs, Mat::convertTo
*/
CV_EXPORTS_W void LUT(InputArray src, InputArray lut, OutputArray dst);

/** @brief Calculates the sum of array elements.

The function cv::sum calculates and returns the sum of array elements,
independently for each channel.
@param src input array that must have from 1 to 4 channels.
@sa  countNonZero, mean, meanStdDev, norm, minMaxLoc, reduce
*/
CV_EXPORTS_AS(sumElems) Scalar sum(InputArray src);

/** @brief Counts non-zero array elements.

The function returns the number of non-zero elements in src :
\f[\sum _{I: \; \texttt{src} (I) \ne0 } 1\f]
@param src single-channel array.
@sa  mean, meanStdDev, norm, minMaxLoc, calcCovarMatrix
*/
CV_EXPORTS_W int countNonZero( InputArray src );

/** @brief Returns the list of locations of non-zero pixels

Given a binary matrix (likely returned from an operation such
as threshold(), compare(), >, ==, etc, return all of
the non-zero indices as a cv::Mat or std::vector<cv::Point> (x,y)
For example:
@code{.cpp}
    cv::Mat binaryImage; // input, binary image
    cv::Mat locations;   // output, locations of non-zero pixels
    cv::findNonZero(binaryImage, locations);

    // access pixel coordinates
    Point pnt = locations.at<Point>(i);
@endcode
or
@code{.cpp}
    cv::Mat binaryImage; // input, binary image
    vector<Point> locations;   // output, locations of non-zero pixels
    cv::findNonZero(binaryImage, locations);

    // access pixel coordinates
    Point pnt = locations[i];
@endcode
@param src single-channel array (type CV_8UC1)
@param idx the output array, type of cv::Mat or std::vector<Point>, corresponding to non-zero indices in the input
*/
CV_EXPORTS_W void findNonZero( InputArray src, OutputArray idx );

/** @brief Calculates an average (mean) of array elements.

The function cv::mean calculates the mean value M of array elements,
independently for each channel, and return it:
\f[\begin{array}{l} N =  \sum _{I: \; \texttt{mask} (I) \ne 0} 1 \\ M_c =  \left ( \sum _{I: \; \texttt{mask} (I) \ne 0}{ \texttt{mtx} (I)_c} \right )/N \end{array}\f]
When all the mask elements are 0's, the function returns Scalar::all(0)
@param src input array that should have from 1 to 4 channels so that the result can be stored in
Scalar_ .
@param mask optional operation mask.
@sa  countNonZero, meanStdDev, norm, minMaxLoc
*/
CV_EXPORTS_W Scalar mean(InputArray src, InputArray mask = noArray());

/** Calculates a mean and standard deviation of array elements.

The function cv::meanStdDev calculates the mean and the standard deviation M
of array elements independently for each channel and returns it via the
output parameters:
\f[\begin{array}{l} N =  \sum _{I, \texttt{mask} (I)  \ne 0} 1 \\ \texttt{mean} _c =  \frac{\sum_{ I: \; \texttt{mask}(I) \ne 0} \texttt{src} (I)_c}{N} \\ \texttt{stddev} _c =  \sqrt{\frac{\sum_{ I: \; \texttt{mask}(I) \ne 0} \left ( \texttt{src} (I)_c -  \texttt{mean} _c \right )^2}{N}} \end{array}\f]
When all the mask elements are 0's, the function returns
mean=stddev=Scalar::all(0).
@note The calculated standard deviation is only the diagonal of the
complete normalized covariance matrix. If the full matrix is needed, you
can reshape the multi-channel array M x N to the single-channel array
M\*N x mtx.channels() (only possible when the matrix is continuous) and
then pass the matrix to calcCovarMatrix .
@param src input array that should have from 1 to 4 channels so that the results can be stored in
Scalar_ 's.
@param mean output parameter: calculated mean value.
@param stddev output parameter: calculateded standard deviation.
@param mask optional operation mask.
@sa  countNonZero, mean, norm, minMaxLoc, calcCovarMatrix
*/
CV_EXPORTS_W void meanStdDev(InputArray src, OutputArray mean, OutputArray stddev,
                             InputArray mask=noArray());

/** @brief Calculates an absolute array norm, an absolute difference norm, or a
relative difference norm.

The function cv::norm calculates an absolute norm of src1 (when there is no
src2 ):

\f[norm =  \forkthree{\|\texttt{src1}\|_{L_{\infty}} =  \max _I | \texttt{src1} (I)|}{if  \(\texttt{normType} = \texttt{NORM_INF}\) }
{ \| \texttt{src1} \| _{L_1} =  \sum _I | \texttt{src1} (I)|}{if  \(\texttt{normType} = \texttt{NORM_L1}\) }
{ \| \texttt{src1} \| _{L_2} =  \sqrt{\sum_I \texttt{src1}(I)^2} }{if  \(\texttt{normType} = \texttt{NORM_L2}\) }\f]

or an absolute or relative difference norm if src2 is there:

\f[norm =  \forkthree{\|\texttt{src1}-\texttt{src2}\|_{L_{\infty}} =  \max _I | \texttt{src1} (I) -  \texttt{src2} (I)|}{if  \(\texttt{normType} = \texttt{NORM_INF}\) }
{ \| \texttt{src1} - \texttt{src2} \| _{L_1} =  \sum _I | \texttt{src1} (I) -  \texttt{src2} (I)|}{if  \(\texttt{normType} = \texttt{NORM_L1}\) }
{ \| \texttt{src1} - \texttt{src2} \| _{L_2} =  \sqrt{\sum_I (\texttt{src1}(I) - \texttt{src2}(I))^2} }{if  \(\texttt{normType} = \texttt{NORM_L2}\) }\f]

or

\f[norm =  \forkthree{\frac{\|\texttt{src1}-\texttt{src2}\|_{L_{\infty}}    }{\|\texttt{src2}\|_{L_{\infty}} }}{if  \(\texttt{normType} = \texttt{NORM_RELATIVE_INF}\) }
{ \frac{\|\texttt{src1}-\texttt{src2}\|_{L_1} }{\|\texttt{src2}\|_{L_1}} }{if  \(\texttt{normType} = \texttt{NORM_RELATIVE_L1}\) }
{ \frac{\|\texttt{src1}-\texttt{src2}\|_{L_2} }{\|\texttt{src2}\|_{L_2}} }{if  \(\texttt{normType} = \texttt{NORM_RELATIVE_L2}\) }\f]

The function cv::norm returns the calculated norm.

When the mask parameter is specified and it is not empty, the norm is
calculated only over the region specified by the mask.

A multi-channel input arrays are treated as a single-channel, that is,
the results for all channels are combined.

@param src1 first input array.
@param normType type of the norm (see cv::NormTypes).
@param mask optional operation mask; it must have the same size as src1 and CV_8UC1 type.
*/
CV_EXPORTS_W double norm(InputArray src1, int normType = NORM_L2, InputArray mask = noArray());

/** @overload
@param src1 first input array.
@param src2 second input array of the same size and the same type as src1.
@param normType type of the norm (cv::NormTypes).
@param mask optional operation mask; it must have the same size as src1 and CV_8UC1 type.
*/
CV_EXPORTS_W double norm(InputArray src1, InputArray src2,
                         int normType = NORM_L2, InputArray mask = noArray());
/** @overload
@param src first input array.
@param normType type of the norm (see cv::NormTypes).
*/
CV_EXPORTS double norm( const SparseMat& src, int normType );

/** @brief computes PSNR image/video quality metric

see http://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio for details
@todo document
  */
CV_EXPORTS_W double PSNR(InputArray src1, InputArray src2);

/** @brief naive nearest neighbor finder

see http://en.wikipedia.org/wiki/Nearest_neighbor_search
@todo document
  */
CV_EXPORTS_W void batchDistance(InputArray src1, InputArray src2,
                                OutputArray dist, int dtype, OutputArray nidx,
                                int normType = NORM_L2, int K = 0,
                                InputArray mask = noArray(), int update = 0,
                                bool crosscheck = false);

/** @brief Normalizes the norm or value range of an array.

The function cv::normalize normalizes scale and shift the input array elements so that
\f[\| \texttt{dst} \| _{L_p}= \texttt{alpha}\f]
(where p=Inf, 1 or 2) when normType=NORM_INF, NORM_L1, or NORM_L2, respectively; or so that
\f[\min _I  \texttt{dst} (I)= \texttt{alpha} , \, \, \max _I  \texttt{dst} (I)= \texttt{beta}\f]

when normType=NORM_MINMAX (for dense arrays only). The optional mask specifies a sub-array to be
normalized. This means that the norm or min-n-max are calculated over the sub-array, and then this
sub-array is modified to be normalized. If you want to only use the mask to calculate the norm or
min-max but modify the whole array, you can use norm and Mat::convertTo.

In case of sparse matrices, only the non-zero values are analyzed and transformed. Because of this,
the range transformation for sparse matrices is not allowed since it can shift the zero level.

Possible usage with some positive example data:
@code{.cpp}
    vector<double> positiveData = { 2.0, 8.0, 10.0 };
    vector<double> normalizedData_l1, normalizedData_l2, normalizedData_inf, normalizedData_minmax;

    // Norm to probability (total count)
    // sum(numbers) = 20.0
    // 2.0      0.1     (2.0/20.0)
    // 8.0      0.4     (8.0/20.0)
    // 10.0     0.5     (10.0/20.0)
    normalize(positiveData, normalizedData_l1, 1.0, 0.0, NORM_L1);

    // Norm to unit vector: ||positiveData|| = 1.0
    // 2.0      0.15
    // 8.0      0.62
    // 10.0     0.77
    normalize(positiveData, normalizedData_l2, 1.0, 0.0, NORM_L2);

    // Norm to max element
    // 2.0      0.2     (2.0/10.0)
    // 8.0      0.8     (8.0/10.0)
    // 10.0     1.0     (10.0/10.0)
    normalize(positiveData, normalizedData_inf, 1.0, 0.0, NORM_INF);

    // Norm to range [0.0;1.0]
    // 2.0      0.0     (shift to left border)
    // 8.0      0.75    (6.0/8.0)
    // 10.0     1.0     (shift to right border)
    normalize(positiveData, normalizedData_minmax, 1.0, 0.0, NORM_MINMAX);
@endcode

@param src input array.
@param dst output array of the same size as src .
@param alpha norm value to normalize to or the lower range boundary in case of the range
normalization.
@param beta upper range boundary in case of the range normalization; it is not used for the norm
normalization.
@param norm_type normalization type (see cv::NormTypes).
@param dtype when negative, the output array has the same type as src; otherwise, it has the same
number of channels as src and the depth =CV_MAT_DEPTH(dtype).
@param mask optional operation mask.
@sa norm, Mat::convertTo, SparseMat::convertTo
*/
CV_EXPORTS_W void normalize( InputArray src, InputOutputArray dst, double alpha = 1, double beta = 0,
                             int norm_type = NORM_L2, int dtype = -1, InputArray mask = noArray());

/** @overload
@param src input array.
@param dst output array of the same size as src .
@param alpha norm value to normalize to or the lower range boundary in case of the range
normalization.
@param normType normalization type (see cv::NormTypes).
*/
CV_EXPORTS void normalize( const SparseMat& src, SparseMat& dst, double alpha, int normType );

/** @brief Finds the global minimum and maximum in an array.

The function cv::minMaxLoc finds the minimum and maximum element values and their positions. The
extremums are searched across the whole array or, if mask is not an empty array, in the specified
array region.

The function do not work with multi-channel arrays. If you need to find minimum or maximum
elements across all the channels, use Mat::reshape first to reinterpret the array as
single-channel. Or you may extract the particular channel using either extractImageCOI , or
mixChannels , or split .
@param src input single-channel array.
@param minVal pointer to the returned minimum value; NULL is used if not required.
@param maxVal pointer to the returned maximum value; NULL is used if not required.
@param minLoc pointer to the returned minimum location (in 2D case); NULL is used if not required.
@param maxLoc pointer to the returned maximum location (in 2D case); NULL is used if not required.
@param mask optional mask used to select a sub-array.
@sa max, min, compare, inRange, extractImageCOI, mixChannels, split, Mat::reshape
*/
CV_EXPORTS_W void minMaxLoc(InputArray src, CV_OUT double* minVal,
                            CV_OUT double* maxVal = 0, CV_OUT Point* minLoc = 0,
                            CV_OUT Point* maxLoc = 0, InputArray mask = noArray());


/** @brief Finds the global minimum and maximum in an array

The function cv::minMaxIdx finds the minimum and maximum element values and their positions. The
extremums are searched across the whole array or, if mask is not an empty array, in the specified
array region. The function does not work with multi-channel arrays. If you need to find minimum or
maximum elements across all the channels, use Mat::reshape first to reinterpret the array as
single-channel. Or you may extract the particular channel using either extractImageCOI , or
mixChannels , or split . In case of a sparse matrix, the minimum is found among non-zero elements
only.
@note When minIdx is not NULL, it must have at least 2 elements (as well as maxIdx), even if src is
a single-row or single-column matrix. In OpenCV (following MATLAB) each array has at least 2
dimensions, i.e. single-column matrix is Mx1 matrix (and therefore minIdx/maxIdx will be
(i1,0)/(i2,0)) and single-row matrix is 1xN matrix (and therefore minIdx/maxIdx will be
(0,j1)/(0,j2)).
@param src input single-channel array.
@param minVal pointer to the returned minimum value; NULL is used if not required.
@param maxVal pointer to the returned maximum value; NULL is used if not required.
@param minIdx pointer to the returned minimum location (in nD case); NULL is used if not required;
Otherwise, it must point to an array of src.dims elements, the coordinates of the minimum element
in each dimension are stored there sequentially.
@param maxIdx pointer to the returned maximum location (in nD case). NULL is used if not required.
@param mask specified array region
*/
CV_EXPORTS void minMaxIdx(InputArray src, double* minVal, double* maxVal = 0,
                          int* minIdx = 0, int* maxIdx = 0, InputArray mask = noArray());

/** @overload
@param a input single-channel array.
@param minVal pointer to the returned minimum value; NULL is used if not required.
@param maxVal pointer to the returned maximum value; NULL is used if not required.
@param minIdx pointer to the returned minimum location (in nD case); NULL is used if not required;
Otherwise, it must point to an array of src.dims elements, the coordinates of the minimum element
in each dimension are stored there sequentially.
@param maxIdx pointer to the returned maximum location (in nD case). NULL is used if not required.
*/
CV_EXPORTS void minMaxLoc(const SparseMat& a, double* minVal,
                          double* maxVal, int* minIdx = 0, int* maxIdx = 0);

/** @brief Reduces a matrix to a vector.

The function cv::reduce reduces the matrix to a vector by treating the matrix rows/columns as a set of
1D vectors and performing the specified operation on the vectors until a single row/column is
obtained. For example, the function can be used to compute horizontal and vertical projections of a
raster image. In case of REDUCE_MAX and REDUCE_MIN , the output image should have the same type as the source one.
In case of REDUCE_SUM and REDUCE_AVG , the output may have a larger element bit-depth to preserve accuracy.
And multi-channel arrays are also supported in these two reduction modes.
@param src input 2D matrix.
@param dst output vector. Its size and type is defined by dim and dtype parameters.
@param dim dimension index along which the matrix is reduced. 0 means that the matrix is reduced to
a single row. 1 means that the matrix is reduced to a single column.
@param rtype reduction operation that could be one of cv::ReduceTypes
@param dtype when negative, the output vector will have the same type as the input matrix,
otherwise, its type will be CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), src.channels()).
@sa repeat
*/
CV_EXPORTS_W void reduce(InputArray src, OutputArray dst, int dim, int rtype, int dtype = -1);

/** @brief Creates one multi-channel array out of several single-channel ones.

The function cv::merge merges several arrays to make a single multi-channel array. That is, each
element of the output array will be a concatenation of the elements of the input arrays, where
elements of i-th input array are treated as mv[i].channels()-element vectors.

The function cv::split does the reverse operation. If you need to shuffle channels in some other
advanced way, use cv::mixChannels.
@param mv input array of matrices to be merged; all the matrices in mv must have the same
size and the same depth.
@param count number of input matrices when mv is a plain C array; it must be greater than zero.
@param dst output array of the same size and the same depth as mv[0]; The number of channels will
be equal to the parameter count.
@sa  mixChannels, split, Mat::reshape
*/
CV_EXPORTS void merge(const Mat* mv, size_t count, OutputArray dst);

/** @overload
@param mv input vector of matrices to be merged; all the matrices in mv must have the same
size and the same depth.
@param dst output array of the same size and the same depth as mv[0]; The number of channels will
be the total number of channels in the matrix array.
  */
CV_EXPORTS_W void merge(InputArrayOfArrays mv, OutputArray dst);

/** @brief Divides a multi-channel array into several single-channel arrays.

The function cv::split splits a multi-channel array into separate single-channel arrays:
\f[\texttt{mv} [c](I) =  \texttt{src} (I)_c\f]
If you need to extract a single channel or do some other sophisticated channel permutation, use
mixChannels .
@param src input multi-channel array.
@param mvbegin output array; the number of arrays must match src.channels(); the arrays themselves are
reallocated, if needed.
@sa merge, mixChannels, cvtColor
*/
CV_EXPORTS void split(const Mat& src, Mat* mvbegin);

/** @overload
@param m input multi-channel array.
@param mv output vector of arrays; the arrays themselves are reallocated, if needed.
*/
CV_EXPORTS_W void split(InputArray m, OutputArrayOfArrays mv);

/** @brief Copies specified channels from input arrays to the specified channels of
output arrays.

The function cv::mixChannels provides an advanced mechanism for shuffling image channels.

cv::split,cv::merge,cv::extractChannel,cv::insertChannel and some forms of cv::cvtColor are partial cases of cv::mixChannels.

In the example below, the code splits a 4-channel BGRA image into a 3-channel BGR (with B and R
channels swapped) and a separate alpha-channel image:
@code{.cpp}
    Mat bgra( 100, 100, CV_8UC4, Scalar(255,0,0,255) );
    Mat bgr( bgra.rows, bgra.cols, CV_8UC3 );
    Mat alpha( bgra.rows, bgra.cols, CV_8UC1 );

    // forming an array of matrices is a quite efficient operation,
    // because the matrix data is not copied, only the headers
    Mat out[] = { bgr, alpha };
    // bgra[0] -> bgr[2], bgra[1] -> bgr[1],
    // bgra[2] -> bgr[0], bgra[3] -> alpha[0]
    int from_to[] = { 0,2, 1,1, 2,0, 3,3 };
    mixChannels( &bgra, 1, out, 2, from_to, 4 );
@endcode
@note Unlike many other new-style C++ functions in OpenCV (see the introduction section and
Mat::create ), cv::mixChannels requires the output arrays to be pre-allocated before calling the
function.
@param src input array or vector of matrices; all of the matrices must have the same size and the
same depth.
@param nsrcs number of matrices in `src`.
@param dst output array or vector of matrices; all the matrices **must be allocated**; their size and
depth must be the same as in `src[0]`.
@param ndsts number of matrices in `dst`.
@param fromTo array of index pairs specifying which channels are copied and where; fromTo[k\*2] is
a 0-based index of the input channel in src, fromTo[k\*2+1] is an index of the output channel in
dst; the continuous channel numbering is used: the first input image channels are indexed from 0 to
src[0].channels()-1, the second input image channels are indexed from src[0].channels() to
src[0].channels() + src[1].channels()-1, and so on, the same scheme is used for the output image
channels; as a special case, when fromTo[k\*2] is negative, the corresponding output channel is
filled with zero .
@param npairs number of index pairs in `fromTo`.
@sa split, merge, extractChannel, insertChannel, cvtColor
*/
CV_EXPORTS void mixChannels(const Mat* src, size_t nsrcs, Mat* dst, size_t ndsts,
                            const int* fromTo, size_t npairs);

/** @overload
@param src input array or vector of matrices; all of the matrices must have the same size and the
same depth.
@param dst output array or vector of matrices; all the matrices **must be allocated**; their size and
depth must be the same as in src[0].
@param fromTo array of index pairs specifying which channels are copied and where; fromTo[k\*2] is
a 0-based index of the input channel in src, fromTo[k\*2+1] is an index of the output channel in
dst; the continuous channel numbering is used: the first input image channels are indexed from 0 to
src[0].channels()-1, the second input image channels are indexed from src[0].channels() to
src[0].channels() + src[1].channels()-1, and so on, the same scheme is used for the output image
channels; as a special case, when fromTo[k\*2] is negative, the corresponding output channel is
filled with zero .
@param npairs number of index pairs in fromTo.
*/
CV_EXPORTS void mixChannels(InputArrayOfArrays src, InputOutputArrayOfArrays dst,
                            const int* fromTo, size_t npairs);

/** @overload
@param src input array or vector of matrices; all of the matrices must have the same size and the
same depth.
@param dst output array or vector of matrices; all the matrices **must be allocated**; their size and
depth must be the same as in src[0].
@param fromTo array of index pairs specifying which channels are copied and where; fromTo[k\*2] is
a 0-based index of the input channel in src, fromTo[k\*2+1] is an index of the output channel in
dst; the continuous channel numbering is used: the first input image channels are indexed from 0 to
src[0].channels()-1, the second input image channels are indexed from src[0].channels() to
src[0].channels() + src[1].channels()-1, and so on, the same scheme is used for the output image
channels; as a special case, when fromTo[k\*2] is negative, the corresponding output channel is
filled with zero .
*/
CV_EXPORTS_W void mixChannels(InputArrayOfArrays src, InputOutputArrayOfArrays dst,
                              const std::vector<int>& fromTo);

/** @brief Extracts a single channel from src (coi is 0-based index)
@param src input array
@param dst output array
@param coi index of channel to extract
@sa mixChannels, split
*/
CV_EXPORTS_W void extractChannel(InputArray src, OutputArray dst, int coi);

/** @brief Inserts a single channel to dst (coi is 0-based index)
@param src input array
@param dst output array
@param coi index of channel for insertion
@sa mixChannels, merge
*/
CV_EXPORTS_W void insertChannel(InputArray src, InputOutputArray dst, int coi);

/** @brief Flips a 2D array around vertical, horizontal, or both axes.

The function cv::flip flips the array in one of three different ways (row
and column indices are 0-based):
\f[\texttt{dst} _{ij} =
\left\{
\begin{array}{l l}
\texttt{src} _{\texttt{src.rows}-i-1,j} & if\;  \texttt{flipCode} = 0 \\
\texttt{src} _{i, \texttt{src.cols} -j-1} & if\;  \texttt{flipCode} > 0 \\
\texttt{src} _{ \texttt{src.rows} -i-1, \texttt{src.cols} -j-1} & if\; \texttt{flipCode} < 0 \\
\end{array}
\right.\f]
The example scenarios of using the function are the following:
*   Vertical flipping of the image (flipCode == 0) to switch between
    top-left and bottom-left image origin. This is a typical operation
    in video processing on Microsoft Windows\* OS.
*   Horizontal flipping of the image with the subsequent horizontal
    shift and absolute difference calculation to check for a
    vertical-axis symmetry (flipCode \> 0).
*   Simultaneous horizontal and vertical flipping of the image with
    the subsequent shift and absolute difference calculation to check
    for a central symmetry (flipCode \< 0).
*   Reversing the order of point arrays (flipCode \> 0 or
    flipCode == 0).
@param src input array.
@param dst output array of the same size and type as src.
@param flipCode a flag to specify how to flip the array; 0 means
flipping around the x-axis and positive value (for example, 1) means
flipping around y-axis. Negative value (for example, -1) means flipping
around both axes.
@sa transpose , repeat , completeSymm
*/
CV_EXPORTS_W void flip(InputArray src, OutputArray dst, int flipCode);

enum RotateFlags {
    ROTATE_90_CLOCKWISE = 0, //Rotate 90 degrees clockwise
    ROTATE_180 = 1, //Rotate 180 degrees clockwise
    ROTATE_90_COUNTERCLOCKWISE = 2, //Rotate 270 degrees clockwise
};
/** @brief Rotates a 2D array in multiples of 90 degrees.
The function rotate rotates the array in one of three different ways:
*   Rotate by 90 degrees clockwise (rotateCode = ROTATE_90).
*   Rotate by 180 degrees clockwise (rotateCode = ROTATE_180).
*   Rotate by 270 degrees clockwise (rotateCode = ROTATE_270).
@param src input array.
@param dst output array of the same type as src.  The size is the same with ROTATE_180,
and the rows and cols are switched for ROTATE_90 and ROTATE_270.
@param rotateCode an enum to specify how to rotate the array; see the enum RotateFlags
@sa transpose , repeat , completeSymm, flip, RotateFlags
*/
CV_EXPORTS_W void rotate(InputArray src, OutputArray dst, int rotateCode);

/** @brief Fills the output array with repeated copies of the input array.

The function cv::repeat duplicates the input array one or more times along each of the two axes:
\f[\texttt{dst} _{ij}= \texttt{src} _{i\mod src.rows, \; j\mod src.cols }\f]
The second variant of the function is more convenient to use with @ref MatrixExpressions.
@param src input array to replicate.
@param ny Flag to specify how many times the `src` is repeated along the
vertical axis.
@param nx Flag to specify how many times the `src` is repeated along the
horizontal axis.
@param dst output array of the same type as `src`.
@sa cv::reduce
*/
CV_EXPORTS_W void repeat(InputArray src, int ny, int nx, OutputArray dst);

/** @overload
@param src input array to replicate.
@param ny Flag to specify how many times the `src` is repeated along the
vertical axis.
@param nx Flag to specify how many times the `src` is repeated along the
horizontal axis.
  */
CV_EXPORTS Mat repeat(const Mat& src, int ny, int nx);

/** @brief Applies horizontal concatenation to given matrices.

The function horizontally concatenates two or more cv::Mat matrices (with the same number of rows).
@code{.cpp}
    cv::Mat matArray[] = { cv::Mat(4, 1, CV_8UC1, cv::Scalar(1)),
                           cv::Mat(4, 1, CV_8UC1, cv::Scalar(2)),
                           cv::Mat(4, 1, CV_8UC1, cv::Scalar(3)),};

    cv::Mat out;
    cv::hconcat( matArray, 3, out );
    //out:
    //[1, 2, 3;
    // 1, 2, 3;
    // 1, 2, 3;
    // 1, 2, 3]
@endcode
@param src input array or vector of matrices. all of the matrices must have the same number of rows and the same depth.
@param nsrc number of matrices in src.
@param dst output array. It has the same number of rows and depth as the src, and the sum of cols of the src.
@sa cv::vconcat(const Mat*, size_t, OutputArray), @sa cv::vconcat(InputArrayOfArrays, OutputArray) and @sa cv::vconcat(InputArray, InputArray, OutputArray)
*/
CV_EXPORTS void hconcat(const Mat* src, size_t nsrc, OutputArray dst);
/** @overload
 @code{.cpp}
    cv::Mat_<float> A = (cv::Mat_<float>(3, 2) << 1, 4,
                                                  2, 5,
                                                  3, 6);
    cv::Mat_<float> B = (cv::Mat_<float>(3, 2) << 7, 10,
                                                  8, 11,
                                                  9, 12);

    cv::Mat C;
    cv::hconcat(A, B, C);
    //C:
    //[1, 4, 7, 10;
    // 2, 5, 8, 11;
    // 3, 6, 9, 12]
 @endcode
 @param src1 first input array to be considered for horizontal concatenation.
 @param src2 second input array to be considered for horizontal concatenation.
 @param dst output array. It has the same number of rows and depth as the src1 and src2, and the sum of cols of the src1 and src2.
 */
CV_EXPORTS void hconcat(InputArray src1, InputArray src2, OutputArray dst);
/** @overload
 @code{.cpp}
    std::vector<cv::Mat> matrices = { cv::Mat(4, 1, CV_8UC1, cv::Scalar(1)),
                                      cv::Mat(4, 1, CV_8UC1, cv::Scalar(2)),
                                      cv::Mat(4, 1, CV_8UC1, cv::Scalar(3)),};

    cv::Mat out;
    cv::hconcat( matrices, out );
    //out:
    //[1, 2, 3;
    // 1, 2, 3;
    // 1, 2, 3;
    // 1, 2, 3]
 @endcode
 @param src input array or vector of matrices. all of the matrices must have the same number of rows and the same depth.
 @param dst output array. It has the same number of rows and depth as the src, and the sum of cols of the src.
same depth.
 */
CV_EXPORTS_W void hconcat(InputArrayOfArrays src, OutputArray dst);

/** @brief Applies vertical concatenation to given matrices.

The function vertically concatenates two or more cv::Mat matrices (with the same number of cols).
@code{.cpp}
    cv::Mat matArray[] = { cv::Mat(1, 4, CV_8UC1, cv::Scalar(1)),
                           cv::Mat(1, 4, CV_8UC1, cv::Scalar(2)),
                           cv::Mat(1, 4, CV_8UC1, cv::Scalar(3)),};

    cv::Mat out;
    cv::vconcat( matArray, 3, out );
    //out:
    //[1,   1,   1,   1;
    // 2,   2,   2,   2;
    // 3,   3,   3,   3]
@endcode
@param src input array or vector of matrices. all of the matrices must have the same number of cols and the same depth.
@param nsrc number of matrices in src.
@param dst output array. It has the same number of cols and depth as the src, and the sum of rows of the src.
@sa cv::hconcat(const Mat*, size_t, OutputArray), @sa cv::hconcat(InputArrayOfArrays, OutputArray) and @sa cv::hconcat(InputArray, InputArray, OutputArray)
*/
CV_EXPORTS void vconcat(const Mat* src, size_t nsrc, OutputArray dst);
/** @overload
 @code{.cpp}
    cv::Mat_<float> A = (cv::Mat_<float>(3, 2) << 1, 7,
                                                  2, 8,
                                                  3, 9);
    cv::Mat_<float> B = (cv::Mat_<float>(3, 2) << 4, 10,
                                                  5, 11,
                                                  6, 12);

    cv::Mat C;
    cv::vconcat(A, B, C);
    //C:
    //[1, 7;
    // 2, 8;
    // 3, 9;
    // 4, 10;
    // 5, 11;
    // 6, 12]
 @endcode
 @param src1 first input array to be considered for vertical concatenation.
 @param src2 second input array to be considered for vertical concatenation.
 @param dst output array. It has the same number of cols and depth as the src1 and src2, and the sum of rows of the src1 and src2.
 */
CV_EXPORTS void vconcat(InputArray src1, InputArray src2, OutputArray dst);
/** @overload
 @code{.cpp}
    std::vector<cv::Mat> matrices = { cv::Mat(1, 4, CV_8UC1, cv::Scalar(1)),
                                      cv::Mat(1, 4, CV_8UC1, cv::Scalar(2)),
                                      cv::Mat(1, 4, CV_8UC1, cv::Scalar(3)),};

    cv::Mat out;
    cv::vconcat( matrices, out );
    //out:
    //[1,   1,   1,   1;
    // 2,   2,   2,   2;
    // 3,   3,   3,   3]
 @endcode
 @param src input array or vector of matrices. all of the matrices must have the same number of cols and the same depth
 @param dst output array. It has the same number of cols and depth as the src, and the sum of rows of the src.
same depth.
 */
CV_EXPORTS_W void vconcat(InputArrayOfArrays src, OutputArray dst);

/** @brief computes bitwise conjunction of the two arrays (dst = src1 & src2)
Calculates the per-element bit-wise conjunction of two arrays or an
array and a scalar.

The function cv::bitwise_and calculates the per-element bit-wise logical conjunction for:
*   Two arrays when src1 and src2 have the same size:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  \wedge \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]
*   An array and a scalar when src2 is constructed from Scalar or has
    the same number of elements as `src1.channels()`:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  \wedge \texttt{src2} \quad \texttt{if mask} (I) \ne0\f]
*   A scalar and an array when src1 is constructed from Scalar or has
    the same number of elements as `src2.channels()`:
    \f[\texttt{dst} (I) =  \texttt{src1}  \wedge \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]
In case of floating-point arrays, their machine-specific bit
representations (usually IEEE754-compliant) are used for the operation.
In case of multi-channel arrays, each channel is processed
independently. In the second and third cases above, the scalar is first
converted to the array type.
@param src1 first input array or a scalar.
@param src2 second input array or a scalar.
@param dst output array that has the same size and type as the input
arrays.
@param mask optional operation mask, 8-bit single channel array, that
specifies elements of the output array to be changed.
*/
CV_EXPORTS_W void bitwise_and(InputArray src1, InputArray src2,
                              OutputArray dst, InputArray mask = noArray());

/** @brief Calculates the per-element bit-wise disjunction of two arrays or an
array and a scalar.

The function cv::bitwise_or calculates the per-element bit-wise logical disjunction for:
*   Two arrays when src1 and src2 have the same size:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  \vee \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]
*   An array and a scalar when src2 is constructed from Scalar or has
    the same number of elements as `src1.channels()`:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  \vee \texttt{src2} \quad \texttt{if mask} (I) \ne0\f]
*   A scalar and an array when src1 is constructed from Scalar or has
    the same number of elements as `src2.channels()`:
    \f[\texttt{dst} (I) =  \texttt{src1}  \vee \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]
In case of floating-point arrays, their machine-specific bit
representations (usually IEEE754-compliant) are used for the operation.
In case of multi-channel arrays, each channel is processed
independently. In the second and third cases above, the scalar is first
converted to the array type.
@param src1 first input array or a scalar.
@param src2 second input array or a scalar.
@param dst output array that has the same size and type as the input
arrays.
@param mask optional operation mask, 8-bit single channel array, that
specifies elements of the output array to be changed.
*/
CV_EXPORTS_W void bitwise_or(InputArray src1, InputArray src2,
                             OutputArray dst, InputArray mask = noArray());

/** @brief Calculates the per-element bit-wise "exclusive or" operation on two
arrays or an array and a scalar.

The function cv::bitwise_xor calculates the per-element bit-wise logical "exclusive-or"
operation for:
*   Two arrays when src1 and src2 have the same size:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  \oplus \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]
*   An array and a scalar when src2 is constructed from Scalar or has
    the same number of elements as `src1.channels()`:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  \oplus \texttt{src2} \quad \texttt{if mask} (I) \ne0\f]
*   A scalar and an array when src1 is constructed from Scalar or has
    the same number of elements as `src2.channels()`:
    \f[\texttt{dst} (I) =  \texttt{src1}  \oplus \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]
In case of floating-point arrays, their machine-specific bit
representations (usually IEEE754-compliant) are used for the operation.
In case of multi-channel arrays, each channel is processed
independently. In the 2nd and 3rd cases above, the scalar is first
converted to the array type.
@param src1 first input array or a scalar.
@param src2 second input array or a scalar.
@param dst output array that has the same size and type as the input
arrays.
@param mask optional operation mask, 8-bit single channel array, that
specifies elements of the output array to be changed.
*/
CV_EXPORTS_W void bitwise_xor(InputArray src1, InputArray src2,
                              OutputArray dst, InputArray mask = noArray());

/** @brief  Inverts every bit of an array.

The function cv::bitwise_not calculates per-element bit-wise inversion of the input
array:
\f[\texttt{dst} (I) =  \neg \texttt{src} (I)\f]
In case of a floating-point input array, its machine-specific bit
representation (usually IEEE754-compliant) is used for the operation. In
case of multi-channel arrays, each channel is processed independently.
@param src input array.
@param dst output array that has the same size and type as the input
array.
@param mask optional operation mask, 8-bit single channel array, that
specifies elements of the output array to be changed.
*/
CV_EXPORTS_W void bitwise_not(InputArray src, OutputArray dst,
                              InputArray mask = noArray());

/** @brief Calculates the per-element absolute difference between two arrays or between an array and a scalar.

The function cv::absdiff calculates:
*   Absolute difference between two arrays when they have the same
    size and type:
    \f[\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1}(I) -  \texttt{src2}(I)|)\f]
*   Absolute difference between an array and a scalar when the second
    array is constructed from Scalar or has as many elements as the
    number of channels in `src1`:
    \f[\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1}(I) -  \texttt{src2} |)\f]
*   Absolute difference between a scalar and an array when the first
    array is constructed from Scalar or has as many elements as the
    number of channels in `src2`:
    \f[\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1} -  \texttt{src2}(I) |)\f]
    where I is a multi-dimensional index of array elements. In case of
    multi-channel arrays, each channel is processed independently.
@note Saturation is not applied when the arrays have the depth CV_32S.
You may even get a negative value in the case of overflow.
@param src1 first input array or a scalar.
@param src2 second input array or a scalar.
@param dst output array that has the same size and type as input arrays.
@sa cv::abs(const Mat&)
*/
CV_EXPORTS_W void absdiff(InputArray src1, InputArray src2, OutputArray dst);

/** @brief  Checks if array elements lie between the elements of two other arrays.

The function checks the range as follows:
-   For every element of a single-channel input array:
    \f[\texttt{dst} (I)= \texttt{lowerb} (I)_0  \leq \texttt{src} (I)_0 \leq  \texttt{upperb} (I)_0\f]
-   For two-channel arrays:
    \f[\texttt{dst} (I)= \texttt{lowerb} (I)_0  \leq \texttt{src} (I)_0 \leq  \texttt{upperb} (I)_0  \land \texttt{lowerb} (I)_1  \leq \texttt{src} (I)_1 \leq  \texttt{upperb} (I)_1\f]
-   and so forth.

That is, dst (I) is set to 255 (all 1 -bits) if src (I) is within the
specified 1D, 2D, 3D, ... box and 0 otherwise.

When the lower and/or upper boundary parameters are scalars, the indexes
(I) at lowerb and upperb in the above formulas should be omitted.
@param src first input array.
@param lowerb inclusive lower boundary array or a scalar.
@param upperb inclusive upper boundary array or a scalar.
@param dst output array of the same size as src and CV_8U type.
*/
CV_EXPORTS_W void inRange(InputArray src, InputArray lowerb,
                          InputArray upperb, OutputArray dst);

/** @brief Performs the per-element comparison of two arrays or an array and scalar value.

The function compares:
*   Elements of two arrays when src1 and src2 have the same size:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  \,\texttt{cmpop}\, \texttt{src2} (I)\f]
*   Elements of src1 with a scalar src2 when src2 is constructed from
    Scalar or has a single element:
    \f[\texttt{dst} (I) =  \texttt{src1}(I) \,\texttt{cmpop}\,  \texttt{src2}\f]
*   src1 with elements of src2 when src1 is constructed from Scalar or
    has a single element:
    \f[\texttt{dst} (I) =  \texttt{src1}  \,\texttt{cmpop}\, \texttt{src2} (I)\f]
When the comparison result is true, the corresponding element of output
array is set to 255. The comparison operations can be replaced with the
equivalent matrix expressions:
@code{.cpp}
    Mat dst1 = src1 >= src2;
    Mat dst2 = src1 < 8;
    ...
@endcode
@param src1 first input array or a scalar; when it is an array, it must have a single channel.
@param src2 second input array or a scalar; when it is an array, it must have a single channel.
@param dst output array of type ref CV_8U that has the same size and the same number of channels as
    the input arrays.
@param cmpop a flag, that specifies correspondence between the arrays (cv::CmpTypes)
@sa checkRange, min, max, threshold
*/
CV_EXPORTS_W void compare(InputArray src1, InputArray src2, OutputArray dst, int cmpop);

/** @brief Calculates per-element minimum of two arrays or an array and a scalar.

The function cv::min calculates the per-element minimum of two arrays:
\f[\texttt{dst} (I)= \min ( \texttt{src1} (I), \texttt{src2} (I))\f]
or array and a scalar:
\f[\texttt{dst} (I)= \min ( \texttt{src1} (I), \texttt{value} )\f]
@param src1 first input array.
@param src2 second input array of the same size and type as src1.
@param dst output array of the same size and type as src1.
@sa max, compare, inRange, minMaxLoc
*/
CV_EXPORTS_W void min(InputArray src1, InputArray src2, OutputArray dst);
/** @overload
needed to avoid conflicts with const _Tp& std::min(const _Tp&, const _Tp&, _Compare)
*/
CV_EXPORTS void min(const Mat& src1, const Mat& src2, Mat& dst);
/** @overload
needed to avoid conflicts with const _Tp& std::min(const _Tp&, const _Tp&, _Compare)
*/
CV_EXPORTS void min(const UMat& src1, const UMat& src2, UMat& dst);

/** @brief Calculates per-element maximum of two arrays or an array and a scalar.

The function cv::max calculates the per-element maximum of two arrays:
\f[\texttt{dst} (I)= \max ( \texttt{src1} (I), \texttt{src2} (I))\f]
or array and a scalar:
\f[\texttt{dst} (I)= \max ( \texttt{src1} (I), \texttt{value} )\f]
@param src1 first input array.
@param src2 second input array of the same size and type as src1 .
@param dst output array of the same size and type as src1.
@sa  min, compare, inRange, minMaxLoc, @ref MatrixExpressions
*/
CV_EXPORTS_W void max(InputArray src1, InputArray src2, OutputArray dst);
/** @overload
needed to avoid conflicts with const _Tp& std::min(const _Tp&, const _Tp&, _Compare)
*/
CV_EXPORTS void max(const Mat& src1, const Mat& src2, Mat& dst);
/** @overload
needed to avoid conflicts with const _Tp& std::min(const _Tp&, const _Tp&, _Compare)
*/
CV_EXPORTS void max(const UMat& src1, const UMat& src2, UMat& dst);

/** @brief Calculates a square root of array elements.

The function cv::sqrt calculates a square root of each input array element.
In case of multi-channel arrays, each channel is processed
independently. The accuracy is approximately the same as of the built-in
std::sqrt .
@param src input floating-point array.
@param dst output array of the same size and type as src.
*/
CV_EXPORTS_W void sqrt(InputArray src, OutputArray dst);

/** @brief Raises every array element to a power.

The function cv::pow raises every element of the input array to power :
\f[\texttt{dst} (I) =  \fork{\texttt{src}(I)^{power}}{if \(\texttt{power}\) is integer}{|\texttt{src}(I)|^{power}}{otherwise}\f]

So, for a non-integer power exponent, the absolute values of input array
elements are used. However, it is possible to get true values for
negative values using some extra operations. In the example below,
computing the 5th root of array src shows:
@code{.cpp}
    Mat mask = src < 0;
    pow(src, 1./5, dst);
    subtract(Scalar::all(0), dst, dst, mask);
@endcode
For some values of power, such as integer values, 0.5 and -0.5,
specialized faster algorithms are used.

Special values (NaN, Inf) are not handled.
@param src input array.
@param power exponent of power.
@param dst output array of the same size and type as src.
@sa sqrt, exp, log, cartToPolar, polarToCart
*/
CV_EXPORTS_W void pow(InputArray src, double power, OutputArray dst);

/** @brief Calculates the exponent of every array element.

The function cv::exp calculates the exponent of every element of the input
array:
\f[\texttt{dst} [I] = e^{ src(I) }\f]

The maximum relative error is about 7e-6 for single-precision input and
less than 1e-10 for double-precision input. Currently, the function
converts denormalized values to zeros on output. Special values (NaN,
Inf) are not handled.
@param src input array.
@param dst output array of the same size and type as src.
@sa log , cartToPolar , polarToCart , phase , pow , sqrt , magnitude
*/
CV_EXPORTS_W void exp(InputArray src, OutputArray dst);

/** @brief Calculates the natural logarithm of every array element.

The function cv::log calculates the natural logarithm of every element of the input array:
\f[\texttt{dst} (I) =  \log (\texttt{src}(I)) \f]

Output on zero, negative and special (NaN, Inf) values is undefined.

@param src input array.
@param dst output array of the same size and type as src .
@sa exp, cartToPolar, polarToCart, phase, pow, sqrt, magnitude
*/
CV_EXPORTS_W void log(InputArray src, OutputArray dst);

/** @brief Calculates x and y coordinates of 2D vectors from their magnitude and angle.

The function cv::polarToCart calculates the Cartesian coordinates of each 2D
vector represented by the corresponding elements of magnitude and angle:
\f[\begin{array}{l} \texttt{x} (I) =  \texttt{magnitude} (I) \cos ( \texttt{angle} (I)) \\ \texttt{y} (I) =  \texttt{magnitude} (I) \sin ( \texttt{angle} (I)) \\ \end{array}\f]

The relative accuracy of the estimated coordinates is about 1e-6.
@param magnitude input floating-point array of magnitudes of 2D vectors;
it can be an empty matrix (=Mat()), in this case, the function assumes
that all the magnitudes are =1; if it is not empty, it must have the
same size and type as angle.
@param angle input floating-point array of angles of 2D vectors.
@param x output array of x-coordinates of 2D vectors; it has the same
size and type as angle.
@param y output array of y-coordinates of 2D vectors; it has the same
size and type as angle.
@param angleInDegrees when true, the input angles are measured in
degrees, otherwise, they are measured in radians.
@sa cartToPolar, magnitude, phase, exp, log, pow, sqrt
*/
CV_EXPORTS_W void polarToCart(InputArray magnitude, InputArray angle,
                              OutputArray x, OutputArray y, bool angleInDegrees = false);

/** @brief Calculates the magnitude and angle of 2D vectors.

The function cv::cartToPolar calculates either the magnitude, angle, or both
for every 2D vector (x(I),y(I)):
\f[\begin{array}{l} \texttt{magnitude} (I)= \sqrt{\texttt{x}(I)^2+\texttt{y}(I)^2} , \\ \texttt{angle} (I)= \texttt{atan2} ( \texttt{y} (I), \texttt{x} (I))[ \cdot180 / \pi ] \end{array}\f]

The angles are calculated with accuracy about 0.3 degrees. For the point
(0,0), the angle is set to 0.
@param x array of x-coordinates; this must be a single-precision or
double-precision floating-point array.
@param y array of y-coordinates, that must have the same size and same type as x.
@param magnitude output array of magnitudes of the same size and type as x.
@param angle output array of angles that has the same size and type as
x; the angles are measured in radians (from 0 to 2\*Pi) or in degrees (0 to 360 degrees).
@param angleInDegrees a flag, indicating whether the angles are measured
in radians (which is by default), or in degrees.
@sa Sobel, Scharr
*/
CV_EXPORTS_W void cartToPolar(InputArray x, InputArray y,
                              OutputArray magnitude, OutputArray angle,
                              bool angleInDegrees = false);

/** @brief Calculates the rotation angle of 2D vectors.

The function cv::phase calculates the rotation angle of each 2D vector that
is formed from the corresponding elements of x and y :
\f[\texttt{angle} (I) =  \texttt{atan2} ( \texttt{y} (I), \texttt{x} (I))\f]

The angle estimation accuracy is about 0.3 degrees. When x(I)=y(I)=0 ,
the corresponding angle(I) is set to 0.
@param x input floating-point array of x-coordinates of 2D vectors.
@param y input array of y-coordinates of 2D vectors; it must have the
same size and the same type as x.
@param angle output array of vector angles; it has the same size and
same type as x .
@param angleInDegrees when true, the function calculates the angle in
degrees, otherwise, they are measured in radians.
*/
CV_EXPORTS_W void phase(InputArray x, InputArray y, OutputArray angle,
                        bool angleInDegrees = false);

/** @brief Calculates the magnitude of 2D vectors.

The function cv::magnitude calculates the magnitude of 2D vectors formed
from the corresponding elements of x and y arrays:
\f[\texttt{dst} (I) =  \sqrt{\texttt{x}(I)^2 + \texttt{y}(I)^2}\f]
@param x floating-point array of x-coordinates of the vectors.
@param y floating-point array of y-coordinates of the vectors; it must
have the same size as x.
@param magnitude output array of the same size and type as x.
@sa cartToPolar, polarToCart, phase, sqrt
*/
CV_EXPORTS_W void magnitude(InputArray x, InputArray y, OutputArray magnitude);

/** @brief Checks every element of an input array for invalid values.

The function cv::checkRange checks that every array element is neither NaN nor infinite. When minVal \>
-DBL_MAX and maxVal \< DBL_MAX, the function also checks that each value is between minVal and
maxVal. In case of multi-channel arrays, each channel is processed independently. If some values
are out of range, position of the first outlier is stored in pos (when pos != NULL). Then, the
function either returns false (when quiet=true) or throws an exception.
@param a input array.
@param quiet a flag, indicating whether the functions quietly return false when the array elements
are out of range or they throw an exception.
@param pos optional output parameter, when not NULL, must be a pointer to array of src.dims
elements.
@param minVal inclusive lower boundary of valid values range.
@param maxVal exclusive upper boundary of valid values range.
*/
CV_EXPORTS_W bool checkRange(InputArray a, bool quiet = true, CV_OUT Point* pos = 0,
                            double minVal = -DBL_MAX, double maxVal = DBL_MAX);

/** @brief converts NaN's to the given number
*/
CV_EXPORTS_W void patchNaNs(InputOutputArray a, double val = 0);

/** @brief Performs generalized matrix multiplication.

The function cv::gemm performs generalized matrix multiplication similar to the
gemm functions in BLAS level 3. For example,
`gemm(src1, src2, alpha, src3, beta, dst, GEMM_1_T + GEMM_3_T)`
corresponds to
\f[\texttt{dst} =  \texttt{alpha} \cdot \texttt{src1} ^T  \cdot \texttt{src2} +  \texttt{beta} \cdot \texttt{src3} ^T\f]

In case of complex (two-channel) data, performed a complex matrix
multiplication.

The function can be replaced with a matrix expression. For example, the
above call can be replaced with:
@code{.cpp}
    dst = alpha*src1.t()*src2 + beta*src3.t();
@endcode
@param src1 first multiplied input matrix that could be real(CV_32FC1,
CV_64FC1) or complex(CV_32FC2, CV_64FC2).
@param src2 second multiplied input matrix of the same type as src1.
@param alpha weight of the matrix product.
@param src3 third optional delta matrix added to the matrix product; it
should have the same type as src1 and src2.
@param beta weight of src3.
@param dst output matrix; it has the proper size and the same type as
input matrices.
@param flags operation flags (cv::GemmFlags)
@sa mulTransposed , transform
*/
CV_EXPORTS_W void gemm(InputArray src1, InputArray src2, double alpha,
                       InputArray src3, double beta, OutputArray dst, int flags = 0);

/** @brief Calculates the product of a matrix and its transposition.

The function cv::mulTransposed calculates the product of src and its
transposition:
\f[\texttt{dst} = \texttt{scale} ( \texttt{src} - \texttt{delta} )^T ( \texttt{src} - \texttt{delta} )\f]
if aTa=true , and
\f[\texttt{dst} = \texttt{scale} ( \texttt{src} - \texttt{delta} ) ( \texttt{src} - \texttt{delta} )^T\f]
otherwise. The function is used to calculate the covariance matrix. With
zero delta, it can be used as a faster substitute for general matrix
product A\*B when B=A'
@param src input single-channel matrix. Note that unlike gemm, the
function can multiply not only floating-point matrices.
@param dst output square matrix.
@param aTa Flag specifying the multiplication ordering. See the
description below.
@param delta Optional delta matrix subtracted from src before the
multiplication. When the matrix is empty ( delta=noArray() ), it is
assumed to be zero, that is, nothing is subtracted. If it has the same
size as src , it is simply subtracted. Otherwise, it is "repeated" (see
repeat ) to cover the full src and then subtracted. Type of the delta
matrix, when it is not empty, must be the same as the type of created
output matrix. See the dtype parameter description below.
@param scale Optional scale factor for the matrix product.
@param dtype Optional type of the output matrix. When it is negative,
the output matrix will have the same type as src . Otherwise, it will be
type=CV_MAT_DEPTH(dtype) that should be either CV_32F or CV_64F .
@sa calcCovarMatrix, gemm, repeat, reduce
*/
CV_EXPORTS_W void mulTransposed( InputArray src, OutputArray dst, bool aTa,
                                 InputArray delta = noArray(),
                                 double scale = 1, int dtype = -1 );

/** @brief Transposes a matrix.

The function cv::transpose transposes the matrix src :
\f[\texttt{dst} (i,j) =  \texttt{src} (j,i)\f]
@note No complex conjugation is done in case of a complex matrix. It it
should be done separately if needed.
@param src input array.
@param dst output array of the same type as src.
*/
CV_EXPORTS_W void transpose(InputArray src, OutputArray dst);

/** @brief Performs the matrix transformation of every array element.

The function cv::transform performs the matrix transformation of every
element of the array src and stores the results in dst :
\f[\texttt{dst} (I) =  \texttt{m} \cdot \texttt{src} (I)\f]
(when m.cols=src.channels() ), or
\f[\texttt{dst} (I) =  \texttt{m} \cdot [ \texttt{src} (I); 1]\f]
(when m.cols=src.channels()+1 )

Every element of the N -channel array src is interpreted as N -element
vector that is transformed using the M x N or M x (N+1) matrix m to
M-element vector - the corresponding element of the output array dst .

The function may be used for geometrical transformation of
N -dimensional points, arbitrary linear color space transformation (such
as various kinds of RGB to YUV transforms), shuffling the image
channels, and so forth.
@param src input array that must have as many channels (1 to 4) as
m.cols or m.cols-1.
@param dst output array of the same size and depth as src; it has as
many channels as m.rows.
@param m transformation 2x2 or 2x3 floating-point matrix.
@sa perspectiveTransform, getAffineTransform, estimateAffine2D, warpAffine, warpPerspective
*/
CV_EXPORTS_W void transform(InputArray src, OutputArray dst, InputArray m );

/** @brief Performs the perspective matrix transformation of vectors.

The function cv::perspectiveTransform transforms every element of src by
treating it as a 2D or 3D vector, in the following way:
\f[(x, y, z)  \rightarrow (x'/w, y'/w, z'/w)\f]
where
\f[(x', y', z', w') =  \texttt{mat} \cdot \begin{bmatrix} x & y & z & 1  \end{bmatrix}\f]
and
\f[w =  \fork{w'}{if \(w' \ne 0\)}{\infty}{otherwise}\f]

Here a 3D vector transformation is shown. In case of a 2D vector
transformation, the z component is omitted.

@note The function transforms a sparse set of 2D or 3D vectors. If you
want to transform an image using perspective transformation, use
warpPerspective . If you have an inverse problem, that is, you want to
compute the most probable perspective transformation out of several
pairs of corresponding points, you can use getPerspectiveTransform or
findHomography .
@param src input two-channel or three-channel floating-point array; each
element is a 2D/3D vector to be transformed.
@param dst output array of the same size and type as src.
@param m 3x3 or 4x4 floating-point transformation matrix.
@sa  transform, warpPerspective, getPerspectiveTransform, findHomography
*/
CV_EXPORTS_W void perspectiveTransform(InputArray src, OutputArray dst, InputArray m );

/** @brief Copies the lower or the upper half of a square matrix to another half.

The function cv::completeSymm copies the lower half of a square matrix to
its another half. The matrix diagonal remains unchanged:
*   \f$\texttt{mtx}_{ij}=\texttt{mtx}_{ji}\f$ for \f$i > j\f$ if
    lowerToUpper=false
*   \f$\texttt{mtx}_{ij}=\texttt{mtx}_{ji}\f$ for \f$i < j\f$ if
    lowerToUpper=true
@param mtx input-output floating-point square matrix.
@param lowerToUpper operation flag; if true, the lower half is copied to
the upper half. Otherwise, the upper half is copied to the lower half.
@sa flip, transpose
*/
CV_EXPORTS_W void completeSymm(InputOutputArray mtx, bool lowerToUpper = false);

/** @brief Initializes a scaled identity matrix.

The function cv::setIdentity initializes a scaled identity matrix:
\f[\texttt{mtx} (i,j)= \fork{\texttt{value}}{ if \(i=j\)}{0}{otherwise}\f]

The function can also be emulated using the matrix initializers and the
matrix expressions:
@code
    Mat A = Mat::eye(4, 3, CV_32F)*5;
    // A will be set to [[5, 0, 0], [0, 5, 0], [0, 0, 5], [0, 0, 0]]
@endcode
@param mtx matrix to initialize (not necessarily square).
@param s value to assign to diagonal elements.
@sa Mat::zeros, Mat::ones, Mat::setTo, Mat::operator=
*/
CV_EXPORTS_W void setIdentity(InputOutputArray mtx, const Scalar& s = Scalar(1));

/** @brief Returns the determinant of a square floating-point matrix.

The function cv::determinant calculates and returns the determinant of the
specified matrix. For small matrices ( mtx.cols=mtx.rows\<=3 ), the
direct method is used. For larger matrices, the function uses LU
factorization with partial pivoting.

For symmetric positively-determined matrices, it is also possible to use
eigen decomposition to calculate the determinant.
@param mtx input matrix that must have CV_32FC1 or CV_64FC1 type and
square size.
@sa trace, invert, solve, eigen, @ref MatrixExpressions
*/
CV_EXPORTS_W double determinant(InputArray mtx);

/** @brief Returns the trace of a matrix.

The function cv::trace returns the sum of the diagonal elements of the
matrix mtx .
\f[\mathrm{tr} ( \texttt{mtx} ) =  \sum _i  \texttt{mtx} (i,i)\f]
@param mtx input matrix.
*/
CV_EXPORTS_W Scalar trace(InputArray mtx);

/** @brief Finds the inverse or pseudo-inverse of a matrix.

The function cv::invert inverts the matrix src and stores the result in dst
. When the matrix src is singular or non-square, the function calculates
the pseudo-inverse matrix (the dst matrix) so that norm(src\*dst - I) is
minimal, where I is an identity matrix.

In case of the DECOMP_LU method, the function returns non-zero value if
the inverse has been successfully calculated and 0 if src is singular.

In case of the DECOMP_SVD method, the function returns the inverse
condition number of src (the ratio of the smallest singular value to the
largest singular value) and 0 if src is singular. The SVD method
calculates a pseudo-inverse matrix if src is singular.

Similarly to DECOMP_LU, the method DECOMP_CHOLESKY works only with
non-singular square matrices that should also be symmetrical and
positively defined. In this case, the function stores the inverted
matrix in dst and returns non-zero. Otherwise, it returns 0.

@param src input floating-point M x N matrix.
@param dst output matrix of N x M size and the same type as src.
@param flags inversion method (cv::DecompTypes)
@sa solve, SVD
*/
CV_EXPORTS_W double invert(InputArray src, OutputArray dst, int flags = DECOMP_LU);

/** @brief Solves one or more linear systems or least-squares problems.

The function cv::solve solves a linear system or least-squares problem (the
latter is possible with SVD or QR methods, or by specifying the flag
DECOMP_NORMAL ):
\f[\texttt{dst} =  \arg \min _X \| \texttt{src1} \cdot \texttt{X} -  \texttt{src2} \|\f]

If DECOMP_LU or DECOMP_CHOLESKY method is used, the function returns 1
if src1 (or \f$\texttt{src1}^T\texttt{src1}\f$ ) is non-singular. Otherwise,
it returns 0. In the latter case, dst is not valid. Other methods find a
pseudo-solution in case of a singular left-hand side part.

@note If you want to find a unity-norm solution of an under-defined
singular system \f$\texttt{src1}\cdot\texttt{dst}=0\f$ , the function solve
will not do the work. Use SVD::solveZ instead.

@param src1 input matrix on the left-hand side of the system.
@param src2 input matrix on the right-hand side of the system.
@param dst output solution.
@param flags solution (matrix inversion) method (cv::DecompTypes)
@sa invert, SVD, eigen
*/
CV_EXPORTS_W bool solve(InputArray src1, InputArray src2,
                        OutputArray dst, int flags = DECOMP_LU);

/** @brief Sorts each row or each column of a matrix.

The function cv::sort sorts each matrix row or each matrix column in
ascending or descending order. So you should pass two operation flags to
get desired behaviour. If you want to sort matrix rows or columns
lexicographically, you can use STL std::sort generic function with the
proper comparison predicate.

@param src input single-channel array.
@param dst output array of the same size and type as src.
@param flags operation flags, a combination of cv::SortFlags
@sa sortIdx, randShuffle
*/
CV_EXPORTS_W void sort(InputArray src, OutputArray dst, int flags);

/** @brief Sorts each row or each column of a matrix.

The function cv::sortIdx sorts each matrix row or each matrix column in the
ascending or descending order. So you should pass two operation flags to
get desired behaviour. Instead of reordering the elements themselves, it
stores the indices of sorted elements in the output array. For example:
@code
    Mat A = Mat::eye(3,3,CV_32F), B;
    sortIdx(A, B, SORT_EVERY_ROW + SORT_ASCENDING);
    // B will probably contain
    // (because of equal elements in A some permutations are possible):
    // [[1, 2, 0], [0, 2, 1], [0, 1, 2]]
@endcode
@param src input single-channel array.
@param dst output integer array of the same size as src.
@param flags operation flags that could be a combination of cv::SortFlags
@sa sort, randShuffle
*/
CV_EXPORTS_W void sortIdx(InputArray src, OutputArray dst, int flags);

/** @brief Finds the real roots of a cubic equation.

The function solveCubic finds the real roots of a cubic equation:
-   if coeffs is a 4-element vector:
\f[\texttt{coeffs} [0] x^3 +  \texttt{coeffs} [1] x^2 +  \texttt{coeffs} [2] x +  \texttt{coeffs} [3] = 0\f]
-   if coeffs is a 3-element vector:
\f[x^3 +  \texttt{coeffs} [0] x^2 +  \texttt{coeffs} [1] x +  \texttt{coeffs} [2] = 0\f]

The roots are stored in the roots array.
@param coeffs equation coefficients, an array of 3 or 4 elements.
@param roots output array of real roots that has 1 or 3 elements.
*/
CV_EXPORTS_W int solveCubic(InputArray coeffs, OutputArray roots);

/** @brief Finds the real or complex roots of a polynomial equation.

The function cv::solvePoly finds real and complex roots of a polynomial equation:
\f[\texttt{coeffs} [n] x^{n} +  \texttt{coeffs} [n-1] x^{n-1} + ... +  \texttt{coeffs} [1] x +  \texttt{coeffs} [0] = 0\f]
@param coeffs array of polynomial coefficients.
@param roots output (complex) array of roots.
@param maxIters maximum number of iterations the algorithm does.
*/
CV_EXPORTS_W double solvePoly(InputArray coeffs, OutputArray roots, int maxIters = 300);

/** @brief Calculates eigenvalues and eigenvectors of a symmetric matrix.

The function cv::eigen calculates just eigenvalues, or eigenvalues and eigenvectors of the symmetric
matrix src:
@code
    src*eigenvectors.row(i).t() = eigenvalues.at<srcType>(i)*eigenvectors.row(i).t()
@endcode
@note in the new and the old interfaces different ordering of eigenvalues and eigenvectors
parameters is used.
@param src input matrix that must have CV_32FC1 or CV_64FC1 type, square size and be symmetrical
(src ^T^ == src).
@param eigenvalues output vector of eigenvalues of the same type as src; the eigenvalues are stored
in the descending order.
@param eigenvectors output matrix of eigenvectors; it has the same size and type as src; the
eigenvectors are stored as subsequent matrix rows, in the same order as the corresponding
eigenvalues.
@sa completeSymm , PCA
*/
CV_EXPORTS_W bool eigen(InputArray src, OutputArray eigenvalues,
                        OutputArray eigenvectors = noArray());

/** @brief Calculates the covariance matrix of a set of vectors.

The function cv::calcCovarMatrix calculates the covariance matrix and, optionally, the mean vector of
the set of input vectors.
@param samples samples stored as separate matrices
@param nsamples number of samples
@param covar output covariance matrix of the type ctype and square size.
@param mean input or output (depending on the flags) array as the average value of the input vectors.
@param flags operation flags as a combination of cv::CovarFlags
@param ctype type of the matrixl; it equals 'CV_64F' by default.
@sa PCA, mulTransposed, Mahalanobis
@todo InputArrayOfArrays
*/
CV_EXPORTS void calcCovarMatrix( const Mat* samples, int nsamples, Mat& covar, Mat& mean,
                                 int flags, int ctype = CV_64F);

/** @overload
@note use cv::COVAR_ROWS or cv::COVAR_COLS flag
@param samples samples stored as rows/columns of a single matrix.
@param covar output covariance matrix of the type ctype and square size.
@param mean input or output (depending on the flags) array as the average value of the input vectors.
@param flags operation flags as a combination of cv::CovarFlags
@param ctype type of the matrixl; it equals 'CV_64F' by default.
*/
CV_EXPORTS_W void calcCovarMatrix( InputArray samples, OutputArray covar,
                                   InputOutputArray mean, int flags, int ctype = CV_64F);

/** wrap PCA::operator() */
CV_EXPORTS_W void PCACompute(InputArray data, InputOutputArray mean,
                             OutputArray eigenvectors, int maxComponents = 0);

/** wrap PCA::operator() */
CV_EXPORTS_W void PCACompute(InputArray data, InputOutputArray mean,
                             OutputArray eigenvectors, double retainedVariance);

/** wrap PCA::project */
CV_EXPORTS_W void PCAProject(InputArray data, InputArray mean,
                             InputArray eigenvectors, OutputArray result);

/** wrap PCA::backProject */
CV_EXPORTS_W void PCABackProject(InputArray data, InputArray mean,
                                 InputArray eigenvectors, OutputArray result);

/** wrap SVD::compute */
CV_EXPORTS_W void SVDecomp( InputArray src, OutputArray w, OutputArray u, OutputArray vt, int flags = 0 );

/** wrap SVD::backSubst */
CV_EXPORTS_W void SVBackSubst( InputArray w, InputArray u, InputArray vt,
                               InputArray rhs, OutputArray dst );

/** @brief Calculates the Mahalanobis distance between two vectors.

The function cv::Mahalanobis calculates and returns the weighted distance between two vectors:
\f[d( \texttt{vec1} , \texttt{vec2} )= \sqrt{\sum_{i,j}{\texttt{icovar(i,j)}\cdot(\texttt{vec1}(I)-\texttt{vec2}(I))\cdot(\texttt{vec1(j)}-\texttt{vec2(j)})} }\f]
The covariance matrix may be calculated using the cv::calcCovarMatrix function and then inverted using
the invert function (preferably using the cv::DECOMP_SVD method, as the most accurate).
@param v1 first 1D input vector.
@param v2 second 1D input vector.
@param icovar inverse covariance matrix.
*/
CV_EXPORTS_W double Mahalanobis(InputArray v1, InputArray v2, InputArray icovar);

/** @brief Performs a forward or inverse Discrete Fourier transform of a 1D or 2D floating-point array.

The function cv::dft performs one of the following:
-   Forward the Fourier transform of a 1D vector of N elements:
    \f[Y = F^{(N)}  \cdot X,\f]
    where \f$F^{(N)}_{jk}=\exp(-2\pi i j k/N)\f$ and \f$i=\sqrt{-1}\f$
-   Inverse the Fourier transform of a 1D vector of N elements:
    \f[\begin{array}{l} X'=  \left (F^{(N)} \right )^{-1}  \cdot Y =  \left (F^{(N)} \right )^*  \cdot y  \\ X = (1/N)  \cdot X, \end{array}\f]
    where \f$F^*=\left(\textrm{Re}(F^{(N)})-\textrm{Im}(F^{(N)})\right)^T\f$
-   Forward the 2D Fourier transform of a M x N matrix:
    \f[Y = F^{(M)}  \cdot X  \cdot F^{(N)}\f]
-   Inverse the 2D Fourier transform of a M x N matrix:
    \f[\begin{array}{l} X'=  \left (F^{(M)} \right )^*  \cdot Y  \cdot \left (F^{(N)} \right )^* \\ X =  \frac{1}{M \cdot N} \cdot X' \end{array}\f]

In case of real (single-channel) data, the output spectrum of the forward Fourier transform or input
spectrum of the inverse Fourier transform can be represented in a packed format called *CCS*
(complex-conjugate-symmetrical). It was borrowed from IPL (Intel\* Image Processing Library). Here
is how 2D *CCS* spectrum looks:
\f[\begin{bmatrix} Re Y_{0,0} & Re Y_{0,1} & Im Y_{0,1} & Re Y_{0,2} & Im Y_{0,2} &  \cdots & Re Y_{0,N/2-1} & Im Y_{0,N/2-1} & Re Y_{0,N/2}  \\ Re Y_{1,0} & Re Y_{1,1} & Im Y_{1,1} & Re Y_{1,2} & Im Y_{1,2} &  \cdots & Re Y_{1,N/2-1} & Im Y_{1,N/2-1} & Re Y_{1,N/2}  \\ Im Y_{1,0} & Re Y_{2,1} & Im Y_{2,1} & Re Y_{2,2} & Im Y_{2,2} &  \cdots & Re Y_{2,N/2-1} & Im Y_{2,N/2-1} & Im Y_{1,N/2}  \\ \hdotsfor{9} \\ Re Y_{M/2-1,0} &  Re Y_{M-3,1}  & Im Y_{M-3,1} &  \hdotsfor{3} & Re Y_{M-3,N/2-1} & Im Y_{M-3,N/2-1}& Re Y_{M/2-1,N/2}  \\ Im Y_{M/2-1,0} &  Re Y_{M-2,1}  & Im Y_{M-2,1} &  \hdotsfor{3} & Re Y_{M-2,N/2-1} & Im Y_{M-2,N/2-1}& Im Y_{M/2-1,N/2}  \\ Re Y_{M/2,0}  &  Re Y_{M-1,1} &  Im Y_{M-1,1} &  \hdotsfor{3} & Re Y_{M-1,N/2-1} & Im Y_{M-1,N/2-1}& Re Y_{M/2,N/2} \end{bmatrix}\f]

In case of 1D transform of a real vector, the output looks like the first row of the matrix above.

So, the function chooses an operation mode depending on the flags and size of the input array:
-   If DFT_ROWS is set or the input array has a single row or single column, the function
    performs a 1D forward or inverse transform of each row of a matrix when DFT_ROWS is set.
    Otherwise, it performs a 2D transform.
-   If the input array is real and DFT_INVERSE is not set, the function performs a forward 1D or
    2D transform:
    -   When DFT_COMPLEX_OUTPUT is set, the output is a complex matrix of the same size as
        input.
    -   When DFT_COMPLEX_OUTPUT is not set, the output is a real matrix of the same size as
        input. In case of 2D transform, it uses the packed format as shown above. In case of a
        single 1D transform, it looks like the first row of the matrix above. In case of
        multiple 1D transforms (when using the DFT_ROWS flag), each row of the output matrix
        looks like the first row of the matrix above.
-   If the input array is complex and either DFT_INVERSE or DFT_REAL_OUTPUT are not set, the
    output is a complex array of the same size as input. The function performs a forward or
    inverse 1D or 2D transform of the whole input array or each row of the input array
    independently, depending on the flags DFT_INVERSE and DFT_ROWS.
-   When DFT_INVERSE is set and the input array is real, or it is complex but DFT_REAL_OUTPUT
    is set, the output is a real array of the same size as input. The function performs a 1D or 2D
    inverse transformation of the whole input array or each individual row, depending on the flags
    DFT_INVERSE and DFT_ROWS.

If DFT_SCALE is set, the scaling is done after the transformation.

Unlike dct , the function supports arrays of arbitrary size. But only those arrays are processed
efficiently, whose sizes can be factorized in a product of small prime numbers (2, 3, and 5 in the
current implementation). Such an efficient DFT size can be calculated using the getOptimalDFTSize
method.

The sample below illustrates how to calculate a DFT-based convolution of two 2D real arrays:
@code
    void convolveDFT(InputArray A, InputArray B, OutputArray C)
    {
        // reallocate the output array if needed
        C.create(abs(A.rows - B.rows)+1, abs(A.cols - B.cols)+1, A.type());
        Size dftSize;
        // calculate the size of DFT transform
        dftSize.width = getOptimalDFTSize(A.cols + B.cols - 1);
        dftSize.height = getOptimalDFTSize(A.rows + B.rows - 1);

        // allocate temporary buffers and initialize them with 0's
        Mat tempA(dftSize, A.type(), Scalar::all(0));
        Mat tempB(dftSize, B.type(), Scalar::all(0));

        // copy A and B to the top-left corners of tempA and tempB, respectively
        Mat roiA(tempA, Rect(0,0,A.cols,A.rows));
        A.copyTo(roiA);
        Mat roiB(tempB, Rect(0,0,B.cols,B.rows));
        B.copyTo(roiB);

        // now transform the padded A & B in-place;
        // use "nonzeroRows" hint for faster processing
        dft(tempA, tempA, 0, A.rows);
        dft(tempB, tempB, 0, B.rows);

        // multiply the spectrums;
        // the function handles packed spectrum representations well
        mulSpectrums(tempA, tempB, tempA);

        // transform the product back from the frequency domain.
        // Even though all the result rows will be non-zero,
        // you need only the first C.rows of them, and thus you
        // pass nonzeroRows == C.rows
        dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, C.rows);

        // now copy the result back to C.
        tempA(Rect(0, 0, C.cols, C.rows)).copyTo(C);

        // all the temporary buffers will be deallocated automatically
    }
@endcode
To optimize this sample, consider the following approaches:
-   Since nonzeroRows != 0 is passed to the forward transform calls and since A and B are copied to
    the top-left corners of tempA and tempB, respectively, it is not necessary to clear the whole
    tempA and tempB. It is only necessary to clear the tempA.cols - A.cols ( tempB.cols - B.cols)
    rightmost columns of the matrices.
-   This DFT-based convolution does not have to be applied to the whole big arrays, especially if B
    is significantly smaller than A or vice versa. Instead, you can calculate convolution by parts.
    To do this, you need to split the output array C into multiple tiles. For each tile, estimate
    which parts of A and B are required to calculate convolution in this tile. If the tiles in C are
    too small, the speed will decrease a lot because of repeated work. In the ultimate case, when
    each tile in C is a single pixel, the algorithm becomes equivalent to the naive convolution
    algorithm. If the tiles are too big, the temporary arrays tempA and tempB become too big and
    there is also a slowdown because of bad cache locality. So, there is an optimal tile size
    somewhere in the middle.
-   If different tiles in C can be calculated in parallel and, thus, the convolution is done by
    parts, the loop can be threaded.

All of the above improvements have been implemented in matchTemplate and filter2D . Therefore, by
using them, you can get the performance even better than with the above theoretically optimal
implementation. Though, those two functions actually calculate cross-correlation, not convolution,
so you need to "flip" the second convolution operand B vertically and horizontally using flip .
@note
-   An example using the discrete fourier transform can be found at
    opencv_source_code/samples/cpp/dft.cpp
-   (Python) An example using the dft functionality to perform Wiener deconvolution can be found
    at opencv_source/samples/python/deconvolution.py
-   (Python) An example rearranging the quadrants of a Fourier image can be found at
    opencv_source/samples/python/dft.py
@param src input array that could be real or complex.
@param dst output array whose size and type depends on the flags .
@param flags transformation flags, representing a combination of the cv::DftFlags
@param nonzeroRows when the parameter is not zero, the function assumes that only the first
nonzeroRows rows of the input array (DFT_INVERSE is not set) or only the first nonzeroRows of the
output array (DFT_INVERSE is set) contain non-zeros, thus, the function can handle the rest of the
rows more efficiently and save some time; this technique is very useful for calculating array
cross-correlation or convolution using DFT.
@sa dct , getOptimalDFTSize , mulSpectrums, filter2D , matchTemplate , flip , cartToPolar ,
magnitude , phase
*/
CV_EXPORTS_W void dft(InputArray src, OutputArray dst, int flags = 0, int nonzeroRows = 0);

/** @brief Calculates the inverse Discrete Fourier Transform of a 1D or 2D array.

idft(src, dst, flags) is equivalent to dft(src, dst, flags | DFT_INVERSE) .
@note None of dft and idft scales the result by default. So, you should pass DFT_SCALE to one of
dft or idft explicitly to make these transforms mutually inverse.
@sa dft, dct, idct, mulSpectrums, getOptimalDFTSize
@param src input floating-point real or complex array.
@param dst output array whose size and type depend on the flags.
@param flags operation flags (see dft and cv::DftFlags).
@param nonzeroRows number of dst rows to process; the rest of the rows have undefined content (see
the convolution sample in dft description.
*/
CV_EXPORTS_W void idft(InputArray src, OutputArray dst, int flags = 0, int nonzeroRows = 0);

/** @brief Performs a forward or inverse discrete Cosine transform of 1D or 2D array.

The function cv::dct performs a forward or inverse discrete Cosine transform (DCT) of a 1D or 2D
floating-point array:
-   Forward Cosine transform of a 1D vector of N elements:
    \f[Y = C^{(N)}  \cdot X\f]
    where
    \f[C^{(N)}_{jk}= \sqrt{\alpha_j/N} \cos \left ( \frac{\pi(2k+1)j}{2N} \right )\f]
    and
    \f$\alpha_0=1\f$, \f$\alpha_j=2\f$ for *j \> 0*.
-   Inverse Cosine transform of a 1D vector of N elements:
    \f[X =  \left (C^{(N)} \right )^{-1}  \cdot Y =  \left (C^{(N)} \right )^T  \cdot Y\f]
    (since \f$C^{(N)}\f$ is an orthogonal matrix, \f$C^{(N)} \cdot \left(C^{(N)}\right)^T = I\f$ )
-   Forward 2D Cosine transform of M x N matrix:
    \f[Y = C^{(N)}  \cdot X  \cdot \left (C^{(N)} \right )^T\f]
-   Inverse 2D Cosine transform of M x N matrix:
    \f[X =  \left (C^{(N)} \right )^T  \cdot X  \cdot C^{(N)}\f]

The function chooses the mode of operation by looking at the flags and size of the input array:
-   If (flags & DCT_INVERSE) == 0 , the function does a forward 1D or 2D transform. Otherwise, it
    is an inverse 1D or 2D transform.
-   If (flags & DCT_ROWS) != 0 , the function performs a 1D transform of each row.
-   If the array is a single column or a single row, the function performs a 1D transform.
-   If none of the above is true, the function performs a 2D transform.

@note Currently dct supports even-size arrays (2, 4, 6 ...). For data analysis and approximation, you
can pad the array when necessary.
Also, the function performance depends very much, and not monotonically, on the array size (see
getOptimalDFTSize ). In the current implementation DCT of a vector of size N is calculated via DFT
of a vector of size N/2 . Thus, the optimal DCT size N1 \>= N can be calculated as:
@code
    size_t getOptimalDCTSize(size_t N) { return 2*getOptimalDFTSize((N+1)/2); }
    N1 = getOptimalDCTSize(N);
@endcode
@param src input floating-point array.
@param dst output array of the same size and type as src .
@param flags transformation flags as a combination of cv::DftFlags (DCT_*)
@sa dft , getOptimalDFTSize , idct
*/
CV_EXPORTS_W void dct(InputArray src, OutputArray dst, int flags = 0);

/** @brief Calculates the inverse Discrete Cosine Transform of a 1D or 2D array.

idct(src, dst, flags) is equivalent to dct(src, dst, flags | DCT_INVERSE).
@param src input floating-point single-channel array.
@param dst output array of the same size and type as src.
@param flags operation flags.
@sa  dct, dft, idft, getOptimalDFTSize
*/
CV_EXPORTS_W void idct(InputArray src, OutputArray dst, int flags = 0);

/** @brief Performs the per-element multiplication of two Fourier spectrums.

The function cv::mulSpectrums performs the per-element multiplication of the two CCS-packed or complex
matrices that are results of a real or complex Fourier transform.

The function, together with dft and idft , may be used to calculate convolution (pass conjB=false )
or correlation (pass conjB=true ) of two arrays rapidly. When the arrays are complex, they are
simply multiplied (per element) with an optional conjugation of the second-array elements. When the
arrays are real, they are assumed to be CCS-packed (see dft for details).
@param a first input array.
@param b second input array of the same size and type as src1 .
@param c output array of the same size and type as src1 .
@param flags operation flags; currently, the only supported flag is cv::DFT_ROWS, which indicates that
each row of src1 and src2 is an independent 1D Fourier spectrum. If you do not want to use this flag, then simply add a `0` as value.
@param conjB optional flag that conjugates the second input array before the multiplication (true)
or not (false).
*/
CV_EXPORTS_W void mulSpectrums(InputArray a, InputArray b, OutputArray c,
                               int flags, bool conjB = false);

/** @brief Returns the optimal DFT size for a given vector size.

DFT performance is not a monotonic function of a vector size. Therefore, when you calculate
convolution of two arrays or perform the spectral analysis of an array, it usually makes sense to
pad the input data with zeros to get a bit larger array that can be transformed much faster than the
original one. Arrays whose size is a power-of-two (2, 4, 8, 16, 32, ...) are the fastest to process.
Though, the arrays whose size is a product of 2's, 3's, and 5's (for example, 300 = 5\*5\*3\*2\*2)
are also processed quite efficiently.

The function cv::getOptimalDFTSize returns the minimum number N that is greater than or equal to vecsize
so that the DFT of a vector of size N can be processed efficiently. In the current implementation N
= 2 ^p^ \* 3 ^q^ \* 5 ^r^ for some integer p, q, r.

The function returns a negative number if vecsize is too large (very close to INT_MAX ).

While the function cannot be used directly to estimate the optimal vector size for DCT transform
(since the current DCT implementation supports only even-size vectors), it can be easily processed
as getOptimalDFTSize((vecsize+1)/2)\*2.
@param vecsize vector size.
@sa dft , dct , idft , idct , mulSpectrums
*/
CV_EXPORTS_W int getOptimalDFTSize(int vecsize);

/** @brief Returns the default random number generator.

The function cv::theRNG returns the default random number generator. For each thread, there is a
separate random number generator, so you can use the function safely in multi-thread environments.
If you just need to get a single random number using this generator or initialize an array, you can
use randu or randn instead. But if you are going to generate many random numbers inside a loop, it
is much faster to use this function to retrieve the generator and then use RNG::operator _Tp() .
@sa RNG, randu, randn
*/
CV_EXPORTS RNG& theRNG();

/** @brief Sets state of default random number generator.

The function cv::setRNGSeed sets state of default random number generator to custom value.
@param seed new state for default random number generator
@sa RNG, randu, randn
*/
CV_EXPORTS_W void setRNGSeed(int seed);

/** @brief Generates a single uniformly-distributed random number or an array of random numbers.

Non-template variant of the function fills the matrix dst with uniformly-distributed
random numbers from the specified range:
\f[\texttt{low} _c  \leq \texttt{dst} (I)_c <  \texttt{high} _c\f]
@param dst output array of random numbers; the array must be pre-allocated.
@param low inclusive lower boundary of the generated random numbers.
@param high exclusive upper boundary of the generated random numbers.
@sa RNG, randn, theRNG
*/
CV_EXPORTS_W void randu(InputOutputArray dst, InputArray low, InputArray high);

/** @brief Fills the array with normally distributed random numbers.

The function cv::randn fills the matrix dst with normally distributed random numbers with the specified
mean vector and the standard deviation matrix. The generated random numbers are clipped to fit the
value range of the output array data type.
@param dst output array of random numbers; the array must be pre-allocated and have 1 to 4 channels.
@param mean mean value (expectation) of the generated random numbers.
@param stddev standard deviation of the generated random numbers; it can be either a vector (in
which case a diagonal standard deviation matrix is assumed) or a square matrix.
@sa RNG, randu
*/
CV_EXPORTS_W void randn(InputOutputArray dst, InputArray mean, InputArray stddev);

/** @brief Shuffles the array elements randomly.

The function cv::randShuffle shuffles the specified 1D array by randomly choosing pairs of elements and
swapping them. The number of such swap operations will be dst.rows\*dst.cols\*iterFactor .
@param dst input/output numerical 1D array.
@param iterFactor scale factor that determines the number of random swap operations (see the details
below).
@param rng optional random number generator used for shuffling; if it is zero, theRNG () is used
instead.
@sa RNG, sort
*/
CV_EXPORTS_W void randShuffle(InputOutputArray dst, double iterFactor = 1., RNG* rng = 0);

/** @brief Principal Component Analysis

The class is used to calculate a special basis for a set of vectors. The
basis will consist of eigenvectors of the covariance matrix calculated
from the input set of vectors. The class %PCA can also transform
vectors to/from the new coordinate space defined by the basis. Usually,
in this new coordinate system, each vector from the original set (and
any linear combination of such vectors) can be quite accurately
approximated by taking its first few components, corresponding to the
eigenvectors of the largest eigenvalues of the covariance matrix.
Geometrically it means that you calculate a projection of the vector to
a subspace formed by a few eigenvectors corresponding to the dominant
eigenvalues of the covariance matrix. And usually such a projection is
very close to the original vector. So, you can represent the original
vector from a high-dimensional space with a much shorter vector
consisting of the projected vector's coordinates in the subspace. Such a
transformation is also known as Karhunen-Loeve Transform, or KLT.
See http://en.wikipedia.org/wiki/Principal_component_analysis

The sample below is the function that takes two matrices. The first
function stores a set of vectors (a row per vector) that is used to
calculate PCA. The second function stores another "test" set of vectors
(a row per vector). First, these vectors are compressed with PCA, then
reconstructed back, and then the reconstruction error norm is computed
and printed for each vector. :

@code{.cpp}
using namespace cv;

PCA compressPCA(const Mat& pcaset, int maxComponents,
                const Mat& testset, Mat& compressed)
{
    PCA pca(pcaset, // pass the data
            Mat(), // we do not have a pre-computed mean vector,
                   // so let the PCA engine to compute it
            PCA::DATA_AS_ROW, // indicate that the vectors
                                // are stored as matrix rows
                                // (use PCA::DATA_AS_COL if the vectors are
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
@endcode
@sa calcCovarMatrix, mulTransposed, SVD, dft, dct
*/
class CV_EXPORTS PCA
{
public:
    enum Flags { DATA_AS_ROW = 0, //!< indicates that the input samples are stored as matrix rows
                 DATA_AS_COL = 1, //!< indicates that the input samples are stored as matrix columns
                 USE_AVG     = 2  //!
               };

    /** @brief default constructor

    The default constructor initializes an empty %PCA structure. The other
    constructors initialize the structure and call PCA::operator()().
    */
    PCA();

    /** @overload
    @param data input samples stored as matrix rows or matrix columns.
    @param mean optional mean value; if the matrix is empty (@c noArray()),
    the mean is computed from the data.
    @param flags operation flags; currently the parameter is only used to
    specify the data layout (PCA::Flags)
    @param maxComponents maximum number of components that %PCA should
    retain; by default, all the components are retained.
    */
    PCA(InputArray data, InputArray mean, int flags, int maxComponents = 0);

    /** @overload
    @param data input samples stored as matrix rows or matrix columns.
    @param mean optional mean value; if the matrix is empty (noArray()),
    the mean is computed from the data.
    @param flags operation flags; currently the parameter is only used to
    specify the data layout (PCA::Flags)
    @param retainedVariance Percentage of variance that PCA should retain.
    Using this parameter will let the PCA decided how many components to
    retain but it will always keep at least 2.
    */
    PCA(InputArray data, InputArray mean, int flags, double retainedVariance);

    /** @brief performs %PCA

    The operator performs %PCA of the supplied dataset. It is safe to reuse
    the same PCA structure for multiple datasets. That is, if the structure
    has been previously used with another dataset, the existing internal
    data is reclaimed and the new @ref eigenvalues, @ref eigenvectors and @ref
    mean are allocated and computed.

    The computed @ref eigenvalues are sorted from the largest to the smallest and
    the corresponding @ref eigenvectors are stored as eigenvectors rows.

    @param data input samples stored as the matrix rows or as the matrix
    columns.
    @param mean optional mean value; if the matrix is empty (noArray()),
    the mean is computed from the data.
    @param flags operation flags; currently the parameter is only used to
    specify the data layout. (Flags)
    @param maxComponents maximum number of components that PCA should
    retain; by default, all the components are retained.
    */
    PCA& operator()(InputArray data, InputArray mean, int flags, int maxComponents = 0);

    /** @overload
    @param data input samples stored as the matrix rows or as the matrix
    columns.
    @param mean optional mean value; if the matrix is empty (noArray()),
    the mean is computed from the data.
    @param flags operation flags; currently the parameter is only used to
    specify the data layout. (PCA::Flags)
    @param retainedVariance Percentage of variance that %PCA should retain.
    Using this parameter will let the %PCA decided how many components to
    retain but it will always keep at least 2.
     */
    PCA& operator()(InputArray data, InputArray mean, int flags, double retainedVariance);

    /** @brief Projects vector(s) to the principal component subspace.

    The methods project one or more vectors to the principal component
    subspace, where each vector projection is represented by coefficients in
    the principal component basis. The first form of the method returns the
    matrix that the second form writes to the result. So the first form can
    be used as a part of expression while the second form can be more
    efficient in a processing loop.
    @param vec input vector(s); must have the same dimensionality and the
    same layout as the input data used at %PCA phase, that is, if
    DATA_AS_ROW are specified, then `vec.cols==data.cols`
    (vector dimensionality) and `vec.rows` is the number of vectors to
    project, and the same is true for the PCA::DATA_AS_COL case.
    */
    Mat project(InputArray vec) const;

    /** @overload
    @param vec input vector(s); must have the same dimensionality and the
    same layout as the input data used at PCA phase, that is, if
    DATA_AS_ROW are specified, then `vec.cols==data.cols`
    (vector dimensionality) and `vec.rows` is the number of vectors to
    project, and the same is true for the PCA::DATA_AS_COL case.
    @param result output vectors; in case of PCA::DATA_AS_COL, the
    output matrix has as many columns as the number of input vectors, this
    means that `result.cols==vec.cols` and the number of rows match the
    number of principal components (for example, `maxComponents` parameter
    passed to the constructor).
     */
    void project(InputArray vec, OutputArray result) const;

    /** @brief Reconstructs vectors from their PC projections.

    The methods are inverse operations to PCA::project. They take PC
    coordinates of projected vectors and reconstruct the original vectors.
    Unless all the principal components have been retained, the
    reconstructed vectors are different from the originals. But typically,
    the difference is small if the number of components is large enough (but
    still much smaller than the original vector dimensionality). As a
    result, PCA is used.
    @param vec coordinates of the vectors in the principal component
    subspace, the layout and size are the same as of PCA::project output
    vectors.
     */
    Mat backProject(InputArray vec) const;

    /** @overload
    @param vec coordinates of the vectors in the principal component
    subspace, the layout and size are the same as of PCA::project output
    vectors.
    @param result reconstructed vectors; the layout and size are the same as
    of PCA::project input vectors.
     */
    void backProject(InputArray vec, OutputArray result) const;

    /** @brief write PCA objects

    Writes @ref eigenvalues @ref eigenvectors and @ref mean to specified FileStorage
     */
    void write(FileStorage& fs) const;

    /** @brief load PCA objects

    Loads @ref eigenvalues @ref eigenvectors and @ref mean from specified FileNode
     */
    void read(const FileNode& fn);

    Mat eigenvectors; //!< eigenvectors of the covariation matrix
    Mat eigenvalues; //!< eigenvalues of the covariation matrix
    Mat mean; //!< mean value subtracted before the projection and added after the back projection
};

/** @example pca.cpp
  An example using %PCA for dimensionality reduction while maintaining an amount of variance
 */

/**
   @brief Linear Discriminant Analysis
   @todo document this class
 */
class CV_EXPORTS LDA
{
public:
    /** @brief constructor
    Initializes a LDA with num_components (default 0).
    */
    explicit LDA(int num_components = 0);

    /** Initializes and performs a Discriminant Analysis with Fisher's
     Optimization Criterion on given data in src and corresponding labels
     in labels. If 0 (or less) number of components are given, they are
     automatically determined for given data in computation.
    */
    LDA(InputArrayOfArrays src, InputArray labels, int num_components = 0);

    /** Serializes this object to a given filename.
      */
    void save(const String& filename) const;

    /** Deserializes this object from a given filename.
      */
    void load(const String& filename);

    /** Serializes this object to a given cv::FileStorage.
      */
    void save(FileStorage& fs) const;

    /** Deserializes this object from a given cv::FileStorage.
      */
    void load(const FileStorage& node);

    /** destructor
      */
    ~LDA();

    /** Compute the discriminants for data in src (row aligned) and labels.
      */
    void compute(InputArrayOfArrays src, InputArray labels);

    /** Projects samples into the LDA subspace.
        src may be one or more row aligned samples.
      */
    Mat project(InputArray src);

    /** Reconstructs projections from the LDA subspace.
        src may be one or more row aligned projections.
      */
    Mat reconstruct(InputArray src);

    /** Returns the eigenvectors of this LDA.
      */
    Mat eigenvectors() const { return _eigenvectors; }

    /** Returns the eigenvalues of this LDA.
      */
    Mat eigenvalues() const { return _eigenvalues; }

    static Mat subspaceProject(InputArray W, InputArray mean, InputArray src);
    static Mat subspaceReconstruct(InputArray W, InputArray mean, InputArray src);

protected:
    bool _dataAsRow; // unused, but needed for 3.0 ABI compatibility.
    int _num_components;
    Mat _eigenvectors;
    Mat _eigenvalues;
    void lda(InputArrayOfArrays src, InputArray labels);
};

/** @brief Singular Value Decomposition

Class for computing Singular Value Decomposition of a floating-point
matrix. The Singular Value Decomposition is used to solve least-square
problems, under-determined linear systems, invert matrices, compute
condition numbers, and so on.

If you want to compute a condition number of a matrix or an absolute value of
its determinant, you do not need `u` and `vt`. You can pass
flags=SVD::NO_UV|... . Another flag SVD::FULL_UV indicates that full-size u
and vt must be computed, which is not necessary most of the time.

@sa invert, solve, eigen, determinant
*/
class CV_EXPORTS SVD
{
public:
    enum Flags {
        /** allow the algorithm to modify the decomposed matrix; it can save space and speed up
            processing. currently ignored. */
        MODIFY_A = 1,
        /** indicates that only a vector of singular values `w` is to be processed, while u and vt
            will be set to empty matrices */
        NO_UV    = 2,
        /** when the matrix is not square, by default the algorithm produces u and vt matrices of
            sufficiently large size for the further A reconstruction; if, however, FULL_UV flag is
            specified, u and vt will be full-size square orthogonal matrices.*/
        FULL_UV  = 4
    };

    /** @brief the default constructor

    initializes an empty SVD structure
      */
    SVD();

    /** @overload
    initializes an empty SVD structure and then calls SVD::operator()
    @param src decomposed matrix.
    @param flags operation flags (SVD::Flags)
      */
    SVD( InputArray src, int flags = 0 );

    /** @brief the operator that performs SVD. The previously allocated u, w and vt are released.

    The operator performs the singular value decomposition of the supplied
    matrix. The u,`vt` , and the vector of singular values w are stored in
    the structure. The same SVD structure can be reused many times with
    different matrices. Each time, if needed, the previous u,`vt` , and w
    are reclaimed and the new matrices are created, which is all handled by
    Mat::create.
    @param src decomposed matrix.
    @param flags operation flags (SVD::Flags)
      */
    SVD& operator ()( InputArray src, int flags = 0 );

    /** @brief decomposes matrix and stores the results to user-provided matrices

    The methods/functions perform SVD of matrix. Unlike SVD::SVD constructor
    and SVD::operator(), they store the results to the user-provided
    matrices:

    @code{.cpp}
    Mat A, w, u, vt;
    SVD::compute(A, w, u, vt);
    @endcode

    @param src decomposed matrix
    @param w calculated singular values
    @param u calculated left singular vectors
    @param vt transposed matrix of right singular values
    @param flags operation flags - see SVD::SVD.
      */
    static void compute( InputArray src, OutputArray w,
                         OutputArray u, OutputArray vt, int flags = 0 );

    /** @overload
    computes singular values of a matrix
    @param src decomposed matrix
    @param w calculated singular values
    @param flags operation flags - see SVD::Flags.
      */
    static void compute( InputArray src, OutputArray w, int flags = 0 );

    /** @brief performs back substitution
      */
    static void backSubst( InputArray w, InputArray u,
                           InputArray vt, InputArray rhs,
                           OutputArray dst );

    /** @brief solves an under-determined singular linear system

    The method finds a unit-length solution x of a singular linear system
    A\*x = 0. Depending on the rank of A, there can be no solutions, a
    single solution or an infinite number of solutions. In general, the
    algorithm solves the following problem:
    \f[dst =  \arg \min _{x:  \| x \| =1}  \| src  \cdot x  \|\f]
    @param src left-hand-side matrix.
    @param dst found solution.
      */
    static void solveZ( InputArray src, OutputArray dst );

    /** @brief performs a singular value back substitution.

    The method calculates a back substitution for the specified right-hand
    side:

    \f[\texttt{x} =  \texttt{vt} ^T  \cdot diag( \texttt{w} )^{-1}  \cdot \texttt{u} ^T  \cdot \texttt{rhs} \sim \texttt{A} ^{-1}  \cdot \texttt{rhs}\f]

    Using this technique you can either get a very accurate solution of the
    convenient linear system, or the best (in the least-squares terms)
    pseudo-solution of an overdetermined linear system.

    @param rhs right-hand side of a linear system (u\*w\*v')\*dst = rhs to
    be solved, where A has been previously decomposed.

    @param dst found solution of the system.

    @note Explicit SVD with the further back substitution only makes sense
    if you need to solve many linear systems with the same left-hand side
    (for example, src ). If all you need is to solve a single system
    (possibly with multiple rhs immediately available), simply call solve
    add pass DECOMP_SVD there. It does absolutely the same thing.
      */
    void backSubst( InputArray rhs, OutputArray dst ) const;

    /** @todo document */
    template<typename _Tp, int m, int n, int nm> static
    void compute( const Matx<_Tp, m, n>& a, Matx<_Tp, nm, 1>& w, Matx<_Tp, m, nm>& u, Matx<_Tp, n, nm>& vt );

    /** @todo document */
    template<typename _Tp, int m, int n, int nm> static
    void compute( const Matx<_Tp, m, n>& a, Matx<_Tp, nm, 1>& w );

    /** @todo document */
    template<typename _Tp, int m, int n, int nm, int nb> static
    void backSubst( const Matx<_Tp, nm, 1>& w, const Matx<_Tp, m, nm>& u, const Matx<_Tp, n, nm>& vt, const Matx<_Tp, m, nb>& rhs, Matx<_Tp, n, nb>& dst );

    Mat u, w, vt;
};

/** @brief Random Number Generator

Random number generator. It encapsulates the state (currently, a 64-bit
integer) and has methods to return scalar random values and to fill
arrays with random values. Currently it supports uniform and Gaussian
(normal) distributions. The generator uses Multiply-With-Carry
algorithm, introduced by G. Marsaglia (
<http://en.wikipedia.org/wiki/Multiply-with-carry> ).
Gaussian-distribution random numbers are generated using the Ziggurat
algorithm ( <http://en.wikipedia.org/wiki/Ziggurat_algorithm> ),
introduced by G. Marsaglia and W. W. Tsang.
*/
class CV_EXPORTS RNG
{
public:
    enum { UNIFORM = 0,
           NORMAL  = 1
         };

    /** @brief constructor

    These are the RNG constructors. The first form sets the state to some
    pre-defined value, equal to 2\*\*32-1 in the current implementation. The
    second form sets the state to the specified value. If you passed state=0
    , the constructor uses the above default value instead to avoid the
    singular random number sequence, consisting of all zeros.
    */
    RNG();
    /** @overload
    @param state 64-bit value used to initialize the RNG.
    */
    RNG(uint64 state);
    /**The method updates the state using the MWC algorithm and returns the
    next 32-bit random number.*/
    unsigned next();

    /**Each of the methods updates the state using the MWC algorithm and
    returns the next random number of the specified type. In case of integer
    types, the returned number is from the available value range for the
    specified type. In case of floating-point types, the returned value is
    from [0,1) range.
    */
    operator uchar();
    /** @overload */
    operator schar();
    /** @overload */
    operator ushort();
    /** @overload */
    operator short();
    /** @overload */
    operator unsigned();
    /** @overload */
    operator int();
    /** @overload */
    operator float();
    /** @overload */
    operator double();

    /** @brief returns a random integer sampled uniformly from [0, N).

    The methods transform the state using the MWC algorithm and return the
    next random number. The first form is equivalent to RNG::next . The
    second form returns the random number modulo N , which means that the
    result is in the range [0, N) .
    */
    unsigned operator ()();
    /** @overload
    @param N upper non-inclusive boundary of the returned random number.
    */
    unsigned operator ()(unsigned N);

    /** @brief returns uniformly distributed integer random number from [a,b) range

    The methods transform the state using the MWC algorithm and return the
    next uniformly-distributed random number of the specified type, deduced
    from the input parameter type, from the range [a, b) . There is a nuance
    illustrated by the following sample:

    @code{.cpp}
    RNG rng;

    // always produces 0
    double a = rng.uniform(0, 1);

    // produces double from [0, 1)
    double a1 = rng.uniform((double)0, (double)1);

    // produces float from [0, 1)
    double b = rng.uniform(0.f, 1.f);

    // produces double from [0, 1)
    double c = rng.uniform(0., 1.);

    // may cause compiler error because of ambiguity:
    //  RNG::uniform(0, (int)0.999999)? or RNG::uniform((double)0, 0.99999)?
    double d = rng.uniform(0, 0.999999);
    @endcode

    The compiler does not take into account the type of the variable to
    which you assign the result of RNG::uniform . The only thing that
    matters to the compiler is the type of a and b parameters. So, if you
    want a floating-point random number, but the range boundaries are
    integer numbers, either put dots in the end, if they are constants, or
    use explicit type cast operators, as in the a1 initialization above.
    @param a lower inclusive boundary of the returned random numbers.
    @param b upper non-inclusive boundary of the returned random numbers.
      */
    int uniform(int a, int b);
    /** @overload */
    float uniform(float a, float b);
    /** @overload */
    double uniform(double a, double b);

    /** @brief Fills arrays with random numbers.

    @param mat 2D or N-dimensional matrix; currently matrices with more than
    4 channels are not supported by the methods, use Mat::reshape as a
    possible workaround.
    @param distType distribution type, RNG::UNIFORM or RNG::NORMAL.
    @param a first distribution parameter; in case of the uniform
    distribution, this is an inclusive lower boundary, in case of the normal
    distribution, this is a mean value.
    @param b second distribution parameter; in case of the uniform
    distribution, this is a non-inclusive upper boundary, in case of the
    normal distribution, this is a standard deviation (diagonal of the
    standard deviation matrix or the full standard deviation matrix).
    @param saturateRange pre-saturation flag; for uniform distribution only;
    if true, the method will first convert a and b to the acceptable value
    range (according to the mat datatype) and then will generate uniformly
    distributed random numbers within the range [saturate(a), saturate(b)),
    if saturateRange=false, the method will generate uniformly distributed
    random numbers in the original range [a, b) and then will saturate them,
    it means, for example, that
    <tt>theRNG().fill(mat_8u, RNG::UNIFORM, -DBL_MAX, DBL_MAX)</tt> will likely
    produce array mostly filled with 0's and 255's, since the range (0, 255)
    is significantly smaller than [-DBL_MAX, DBL_MAX).

    Each of the methods fills the matrix with the random values from the
    specified distribution. As the new numbers are generated, the RNG state
    is updated accordingly. In case of multiple-channel images, every
    channel is filled independently, which means that RNG cannot generate
    samples from the multi-dimensional Gaussian distribution with
    non-diagonal covariance matrix directly. To do that, the method
    generates samples from multi-dimensional standard Gaussian distribution
    with zero mean and identity covariation matrix, and then transforms them
    using transform to get samples from the specified Gaussian distribution.
    */
    void fill( InputOutputArray mat, int distType, InputArray a, InputArray b, bool saturateRange = false );

    /** @brief Returns the next random number sampled from the Gaussian distribution
    @param sigma standard deviation of the distribution.

    The method transforms the state using the MWC algorithm and returns the
    next random number from the Gaussian distribution N(0,sigma) . That is,
    the mean value of the returned random numbers is zero and the standard
    deviation is the specified sigma .
    */
    double gaussian(double sigma);

    uint64 state;
};

/** @brief Mersenne Twister random number generator

Inspired by http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/CODES/mt19937ar.c
@todo document
 */
class CV_EXPORTS RNG_MT19937
{
public:
    RNG_MT19937();
    RNG_MT19937(unsigned s);
    void seed(unsigned s);

    unsigned next();

    operator int();
    operator unsigned();
    operator float();
    operator double();

    unsigned operator ()(unsigned N);
    unsigned operator ()();

    /** @brief returns uniformly distributed integer random number from [a,b) range

*/
    int uniform(int a, int b);
    /** @brief returns uniformly distributed floating-point random number from [a,b) range

*/
    float uniform(float a, float b);
    /** @brief returns uniformly distributed double-precision floating-point random number from [a,b) range

*/
    double uniform(double a, double b);

private:
    enum PeriodParameters {N = 624, M = 397};
    unsigned state[N];
    int mti;
};

//! @} core_array

//! @addtogroup core_cluster
//!  @{

/** @example kmeans.cpp
  An example on K-means clustering
*/

/** @brief Finds centers of clusters and groups input samples around the clusters.

The function kmeans implements a k-means algorithm that finds the centers of cluster_count clusters
and groups the input samples around the clusters. As an output, \f$\texttt{labels}_i\f$ contains a
0-based cluster index for the sample stored in the \f$i^{th}\f$ row of the samples matrix.

@note
-   (Python) An example on K-means clustering can be found at
    opencv_source_code/samples/python/kmeans.py
@param data Data for clustering. An array of N-Dimensional points with float coordinates is needed.
Examples of this array can be:
-   Mat points(count, 2, CV_32F);
-   Mat points(count, 1, CV_32FC2);
-   Mat points(1, count, CV_32FC2);
-   std::vector\<cv::Point2f\> points(sampleCount);
@param K Number of clusters to split the set by.
@param bestLabels Input/output integer array that stores the cluster indices for every sample.
@param criteria The algorithm termination criteria, that is, the maximum number of iterations and/or
the desired accuracy. The accuracy is specified as criteria.epsilon. As soon as each of the cluster
centers moves by less than criteria.epsilon on some iteration, the algorithm stops.
@param attempts Flag to specify the number of times the algorithm is executed using different
initial labellings. The algorithm returns the labels that yield the best compactness (see the last
function parameter).
@param flags Flag that can take values of cv::KmeansFlags
@param centers Output matrix of the cluster centers, one row per each cluster center.
@return The function returns the compactness measure that is computed as
\f[\sum _i  \| \texttt{samples} _i -  \texttt{centers} _{ \texttt{labels} _i} \| ^2\f]
after every attempt. The best (minimum) value is chosen and the corresponding labels and the
compactness value are returned by the function. Basically, you can use only the core of the
function, set the number of attempts to 1, initialize labels each time using a custom algorithm,
pass them with the ( flags = KMEANS_USE_INITIAL_LABELS ) flag, and then choose the best
(most-compact) clustering.
*/
CV_EXPORTS_W double kmeans( InputArray data, int K, InputOutputArray bestLabels,
                            TermCriteria criteria, int attempts,
                            int flags, OutputArray centers = noArray() );

//! @} core_cluster

//! @addtogroup core_basic
//! @{

/////////////////////////////// Formatted output of cv::Mat ///////////////////////////

/** @todo document */
class CV_EXPORTS Formatted
{
public:
    virtual const char* next() = 0;
    virtual void reset() = 0;
    virtual ~Formatted();
};

/** @todo document */
class CV_EXPORTS Formatter
{
public:
    enum { FMT_DEFAULT = 0,
           FMT_MATLAB  = 1,
           FMT_CSV     = 2,
           FMT_PYTHON  = 3,
           FMT_NUMPY   = 4,
           FMT_C       = 5
         };

    virtual ~Formatter();

    virtual Ptr<Formatted> format(const Mat& mtx) const = 0;

    virtual void set32fPrecision(int p = 8) = 0;
    virtual void set64fPrecision(int p = 16) = 0;
    virtual void setMultiline(bool ml = true) = 0;

    static Ptr<Formatter> get(int fmt = FMT_DEFAULT);

};

static inline
String& operator << (String& out, Ptr<Formatted> fmtd)
{
    fmtd->reset();
    for(const char* str = fmtd->next(); str; str = fmtd->next())
        out += cv::String(str);
    return out;
}

static inline
String& operator << (String& out, const Mat& mtx)
{
    return out << Formatter::get()->format(mtx);
}

//////////////////////////////////////// Algorithm ////////////////////////////////////

class CV_EXPORTS Algorithm;

template<typename _Tp> struct ParamType {};


/** @brief This is a base class for all more or less complex algorithms in OpenCV

especially for classes of algorithms, for which there can be multiple implementations. The examples
are stereo correspondence (for which there are algorithms like block matching, semi-global block
matching, graph-cut etc.), background subtraction (which can be done using mixture-of-gaussians
models, codebook-based algorithm etc.), optical flow (block matching, Lucas-Kanade, Horn-Schunck
etc.).

Here is example of SIFT use in your application via Algorithm interface:
@code
    #include "opencv2/opencv.hpp"
    #include "opencv2/xfeatures2d.hpp"
    using namespace cv::xfeatures2d;

    Ptr<Feature2D> sift = SIFT::create();
    FileStorage fs("sift_params.xml", FileStorage::READ);
    if( fs.isOpened() ) // if we have file with parameters, read them
    {
        sift->read(fs["sift_params"]);
        fs.release();
    }
    else // else modify the parameters and store them; user can later edit the file to use different parameters
    {
        sift->setContrastThreshold(0.01f); // lower the contrast threshold, compared to the default value
        {
            WriteStructContext ws(fs, "sift_params", CV_NODE_MAP);
            sift->write(fs);
        }
    }
    Mat image = imread("myimage.png", 0), descriptors;
    vector<KeyPoint> keypoints;
    sift->detectAndCompute(image, noArray(), keypoints, descriptors);
@endcode
 */
class CV_EXPORTS_W Algorithm
{
public:
    Algorithm();
    virtual ~Algorithm();

    /** @brief Clears the algorithm state
    */
    CV_WRAP virtual void clear() {}

    /** @brief Stores algorithm parameters in a file storage
    */
    virtual void write(FileStorage& fs) const { (void)fs; }

    /** @brief Reads algorithm parameters from a file storage
    */
    virtual void read(const FileNode& fn) { (void)fn; }

    /** @brief Returns true if the Algorithm is empty (e.g. in the very beginning or after unsuccessful read
     */
    virtual bool empty() const { return false; }

    /** @brief Reads algorithm from the file node

     This is static template method of Algorithm. It's usage is following (in the case of SVM):
     @code
     cv::FileStorage fsRead("example.xml", FileStorage::READ);
     Ptr<SVM> svm = Algorithm::read<SVM>(fsRead.root());
     @endcode
     In order to make this method work, the derived class must overwrite Algorithm::read(const
     FileNode& fn) and also have static create() method without parameters
     (or with all the optional parameters)
     */
    template<typename _Tp> static Ptr<_Tp> read(const FileNode& fn)
    {
        Ptr<_Tp> obj = _Tp::create();
        obj->read(fn);
        return !obj->empty() ? obj : Ptr<_Tp>();
    }

    /** @brief Loads algorithm from the file

     @param filename Name of the file to read.
     @param objname The optional name of the node to read (if empty, the first top-level node will be used)

     This is static template method of Algorithm. It's usage is following (in the case of SVM):
     @code
     Ptr<SVM> svm = Algorithm::load<SVM>("my_svm_model.xml");
     @endcode
     In order to make this method work, the derived class must overwrite Algorithm::read(const
     FileNode& fn).
     */
    template<typename _Tp> static Ptr<_Tp> load(const String& filename, const String& objname=String())
    {
        FileStorage fs(filename, FileStorage::READ);
        FileNode fn = objname.empty() ? fs.getFirstTopLevelNode() : fs[objname];
        if (fn.empty()) return Ptr<_Tp>();
        Ptr<_Tp> obj = _Tp::create();
        obj->read(fn);
        return !obj->empty() ? obj : Ptr<_Tp>();
    }

    /** @brief Loads algorithm from a String

     @param strModel The string variable containing the model you want to load.
     @param objname The optional name of the node to read (if empty, the first top-level node will be used)

     This is static template method of Algorithm. It's usage is following (in the case of SVM):
     @code
     Ptr<SVM> svm = Algorithm::loadFromString<SVM>(myStringModel);
     @endcode
     */
    template<typename _Tp> static Ptr<_Tp> loadFromString(const String& strModel, const String& objname=String())
    {
        FileStorage fs(strModel, FileStorage::READ + FileStorage::MEMORY);
        FileNode fn = objname.empty() ? fs.getFirstTopLevelNode() : fs[objname];
        Ptr<_Tp> obj = _Tp::create();
        obj->read(fn);
        return !obj->empty() ? obj : Ptr<_Tp>();
    }

    /** Saves the algorithm to a file.
     In order to make this method work, the derived class must implement Algorithm::write(FileStorage& fs). */
    CV_WRAP virtual void save(const String& filename) const;

    /** Returns the algorithm string identifier.
     This string is used as top level xml/yml node tag when the object is saved to a file or string. */
    CV_WRAP virtual String getDefaultName() const;

protected:
    void writeFormat(FileStorage& fs) const;
};

struct Param {
    enum { INT=0, BOOLEAN=1, REAL=2, STRING=3, MAT=4, MAT_VECTOR=5, ALGORITHM=6, FLOAT=7,
           UNSIGNED_INT=8, UINT64=9, UCHAR=11 };
};



template<> struct ParamType<bool>
{
    typedef bool const_param_type;
    typedef bool member_type;

    enum { type = Param::BOOLEAN };
};

template<> struct ParamType<int>
{
    typedef int const_param_type;
    typedef int member_type;

    enum { type = Param::INT };
};

template<> struct ParamType<double>
{
    typedef double const_param_type;
    typedef double member_type;

    enum { type = Param::REAL };
};

template<> struct ParamType<String>
{
    typedef const String& const_param_type;
    typedef String member_type;

    enum { type = Param::STRING };
};

template<> struct ParamType<Mat>
{
    typedef const Mat& const_param_type;
    typedef Mat member_type;

    enum { type = Param::MAT };
};

template<> struct ParamType<std::vector<Mat> >
{
    typedef const std::vector<Mat>& const_param_type;
    typedef std::vector<Mat> member_type;

    enum { type = Param::MAT_VECTOR };
};

template<> struct ParamType<Algorithm>
{
    typedef const Ptr<Algorithm>& const_param_type;
    typedef Ptr<Algorithm> member_type;

    enum { type = Param::ALGORITHM };
};

template<> struct ParamType<float>
{
    typedef float const_param_type;
    typedef float member_type;

    enum { type = Param::FLOAT };
};

template<> struct ParamType<unsigned>
{
    typedef unsigned const_param_type;
    typedef unsigned member_type;

    enum { type = Param::UNSIGNED_INT };
};

template<> struct ParamType<uint64>
{
    typedef uint64 const_param_type;
    typedef uint64 member_type;

    enum { type = Param::UINT64 };
};

template<> struct ParamType<uchar>
{
    typedef uchar const_param_type;
    typedef uchar member_type;

    enum { type = Param::UCHAR };
};

//! @} core_basic

} //namespace cv

#include "opencv2/core/operations.hpp"
#include "opencv2/core/cvstd.inl.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/optim.hpp"
#include "opencv2/core/ovx.hpp"

#endif /*OPENCV_CORE_HPP*/
