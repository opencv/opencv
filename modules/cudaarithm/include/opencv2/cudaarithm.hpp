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

#ifndef __OPENCV_CUDAARITHM_HPP__
#define __OPENCV_CUDAARITHM_HPP__

#ifndef __cplusplus
#  error cudaarithm.hpp header must be compiled as C++
#endif

#include "opencv2/core/cuda.hpp"

namespace cv { namespace cuda {

//! adds one matrix to another (dst = src1 + src2)
CV_EXPORTS void add(InputArray src1, InputArray src2, OutputArray dst, InputArray mask = noArray(), int dtype = -1, Stream& stream = Stream::Null());

//! subtracts one matrix from another (dst = src1 - src2)
CV_EXPORTS void subtract(InputArray src1, InputArray src2, OutputArray dst, InputArray mask = noArray(), int dtype = -1, Stream& stream = Stream::Null());

//! computes element-wise weighted product of the two arrays (dst = scale * src1 * src2)
CV_EXPORTS void multiply(InputArray src1, InputArray src2, OutputArray dst, double scale = 1, int dtype = -1, Stream& stream = Stream::Null());

//! computes element-wise weighted quotient of the two arrays (dst = scale * (src1 / src2))
CV_EXPORTS void divide(InputArray src1, InputArray src2, OutputArray dst, double scale = 1, int dtype = -1, Stream& stream = Stream::Null());

//! computes element-wise weighted reciprocal of an array (dst = scale/src2)
static inline void divide(double src1, InputArray src2, OutputArray dst, int dtype = -1, Stream& stream = Stream::Null())
{
    divide(src1, src2, dst, 1.0, dtype, stream);
}

//! computes element-wise absolute difference of two arrays (dst = abs(src1 - src2))
CV_EXPORTS void absdiff(InputArray src1, InputArray src2, OutputArray dst, Stream& stream = Stream::Null());

//! computes absolute value of each matrix element
CV_EXPORTS void abs(InputArray src, OutputArray dst, Stream& stream = Stream::Null());

//! computes square of each pixel in an image
CV_EXPORTS void sqr(InputArray src, OutputArray dst, Stream& stream = Stream::Null());

//! computes square root of each pixel in an image
CV_EXPORTS void sqrt(InputArray src, OutputArray dst, Stream& stream = Stream::Null());

//! computes exponent of each matrix element
CV_EXPORTS void exp(InputArray src, OutputArray dst, Stream& stream = Stream::Null());

//! computes natural logarithm of absolute value of each matrix element
CV_EXPORTS void log(InputArray src, OutputArray dst, Stream& stream = Stream::Null());

//! computes power of each matrix element:
//!    (dst(i,j) = pow(     src(i,j) , power), if src.type() is integer
//!    (dst(i,j) = pow(fabs(src(i,j)), power), otherwise
CV_EXPORTS void pow(InputArray src, double power, OutputArray dst, Stream& stream = Stream::Null());

//! compares elements of two arrays (dst = src1 <cmpop> src2)
CV_EXPORTS void compare(InputArray src1, InputArray src2, OutputArray dst, int cmpop, Stream& stream = Stream::Null());

//! performs per-elements bit-wise inversion
CV_EXPORTS void bitwise_not(InputArray src, OutputArray dst, InputArray mask = noArray(), Stream& stream = Stream::Null());

//! calculates per-element bit-wise disjunction of two arrays
CV_EXPORTS void bitwise_or(InputArray src1, InputArray src2, OutputArray dst, InputArray mask = noArray(), Stream& stream = Stream::Null());

//! calculates per-element bit-wise conjunction of two arrays
CV_EXPORTS void bitwise_and(InputArray src1, InputArray src2, OutputArray dst, InputArray mask = noArray(), Stream& stream = Stream::Null());

//! calculates per-element bit-wise "exclusive or" operation
CV_EXPORTS void bitwise_xor(InputArray src1, InputArray src2, OutputArray dst, InputArray mask = noArray(), Stream& stream = Stream::Null());

//! pixel by pixel right shift of an image by a constant value
//! supports 1, 3 and 4 channels images with integers elements
CV_EXPORTS void rshift(InputArray src, Scalar_<int> val, OutputArray dst, Stream& stream = Stream::Null());

//! pixel by pixel left shift of an image by a constant value
//! supports 1, 3 and 4 channels images with CV_8U, CV_16U or CV_32S depth
CV_EXPORTS void lshift(InputArray src, Scalar_<int> val, OutputArray dst, Stream& stream = Stream::Null());

//! computes per-element minimum of two arrays (dst = min(src1, src2))
CV_EXPORTS void min(InputArray src1, InputArray src2, OutputArray dst, Stream& stream = Stream::Null());

//! computes per-element maximum of two arrays (dst = max(src1, src2))
CV_EXPORTS void max(InputArray src1, InputArray src2, OutputArray dst, Stream& stream = Stream::Null());

//! computes the weighted sum of two arrays (dst = alpha*src1 + beta*src2 + gamma)
CV_EXPORTS void addWeighted(InputArray src1, double alpha, InputArray src2, double beta, double gamma, OutputArray dst,
                            int dtype = -1, Stream& stream = Stream::Null());

//! adds scaled array to another one (dst = alpha*src1 + src2)
static inline void scaleAdd(InputArray src1, double alpha, InputArray src2, OutputArray dst, Stream& stream = Stream::Null())
{
    addWeighted(src1, alpha, src2, 1.0, 0.0, dst, -1, stream);
}

//! applies fixed threshold to the image
CV_EXPORTS double threshold(InputArray src, OutputArray dst, double thresh, double maxval, int type, Stream& stream = Stream::Null());

//! computes magnitude of complex (x(i).re, x(i).im) vector
//! supports only CV_32FC2 type
CV_EXPORTS void magnitude(InputArray xy, OutputArray magnitude, Stream& stream = Stream::Null());

//! computes squared magnitude of complex (x(i).re, x(i).im) vector
//! supports only CV_32FC2 type
CV_EXPORTS void magnitudeSqr(InputArray xy, OutputArray magnitude, Stream& stream = Stream::Null());

//! computes magnitude of each (x(i), y(i)) vector
//! supports only floating-point source
CV_EXPORTS void magnitude(InputArray x, InputArray y, OutputArray magnitude, Stream& stream = Stream::Null());

//! computes squared magnitude of each (x(i), y(i)) vector
//! supports only floating-point source
CV_EXPORTS void magnitudeSqr(InputArray x, InputArray y, OutputArray magnitude, Stream& stream = Stream::Null());

//! computes angle of each (x(i), y(i)) vector
//! supports only floating-point source
CV_EXPORTS void phase(InputArray x, InputArray y, OutputArray angle, bool angleInDegrees = false, Stream& stream = Stream::Null());

//! converts Cartesian coordinates to polar
//! supports only floating-point source
CV_EXPORTS void cartToPolar(InputArray x, InputArray y, OutputArray magnitude, OutputArray angle, bool angleInDegrees = false, Stream& stream = Stream::Null());

//! converts polar coordinates to Cartesian
//! supports only floating-point source
CV_EXPORTS void polarToCart(InputArray magnitude, InputArray angle, OutputArray x, OutputArray y, bool angleInDegrees = false, Stream& stream = Stream::Null());

//! makes multi-channel array out of several single-channel arrays
CV_EXPORTS void merge(const GpuMat* src, size_t n, OutputArray dst, Stream& stream = Stream::Null());
CV_EXPORTS void merge(const std::vector<GpuMat>& src, OutputArray dst, Stream& stream = Stream::Null());

//! copies each plane of a multi-channel array to a dedicated array
CV_EXPORTS void split(InputArray src, GpuMat* dst, Stream& stream = Stream::Null());
CV_EXPORTS void split(InputArray src, std::vector<GpuMat>& dst, Stream& stream = Stream::Null());

//! transposes the matrix
//! supports matrix with element size = 1, 4 and 8 bytes (CV_8UC1, CV_8UC4, CV_16UC2, CV_32FC1, etc)
CV_EXPORTS void transpose(InputArray src1, OutputArray dst, Stream& stream = Stream::Null());

//! reverses the order of the rows, columns or both in a matrix
//! supports 1, 3 and 4 channels images with CV_8U, CV_16U, CV_32S or CV_32F depth
CV_EXPORTS void flip(InputArray src, OutputArray dst, int flipCode, Stream& stream = Stream::Null());

//! transforms 8-bit unsigned integers using lookup table: dst(i)=lut(src(i))
//! destination array will have the depth type as lut and the same channels number as source
//! supports CV_8UC1, CV_8UC3 types
class CV_EXPORTS LookUpTable : public Algorithm
{
public:
    virtual void transform(InputArray src, OutputArray dst, Stream& stream = Stream::Null()) = 0;
};

CV_EXPORTS Ptr<LookUpTable> createLookUpTable(InputArray lut);

//! copies 2D array to a larger destination array and pads borders with user-specifiable constant
CV_EXPORTS void copyMakeBorder(InputArray src, OutputArray dst, int top, int bottom, int left, int right, int borderType,
                               Scalar value = Scalar(), Stream& stream = Stream::Null());

//! computes norm of array
//! supports NORM_INF, NORM_L1, NORM_L2
//! supports all matrices except 64F
CV_EXPORTS double norm(InputArray src1, int normType, InputArray mask, GpuMat& buf);
static inline double norm(InputArray src, int normType)
{
    GpuMat buf;
    return norm(src, normType, GpuMat(), buf);
}
static inline double norm(InputArray src, int normType, GpuMat& buf)
{
    return norm(src, normType, GpuMat(), buf);
}

//! computes norm of the difference between two arrays
//! supports NORM_INF, NORM_L1, NORM_L2
//! supports only CV_8UC1 type
CV_EXPORTS double norm(InputArray src1, InputArray src2, GpuMat& buf, int normType=NORM_L2);
static inline double norm(InputArray src1, InputArray src2, int normType=NORM_L2)
{
    GpuMat buf;
    return norm(src1, src2, buf, normType);
}

//! computes sum of array elements
//! supports only single channel images
CV_EXPORTS Scalar sum(InputArray src, InputArray mask, GpuMat& buf);
static inline Scalar sum(InputArray src)
{
    GpuMat buf;
    return sum(src, GpuMat(), buf);
}
static inline Scalar sum(InputArray src, GpuMat& buf)
{
    return sum(src, GpuMat(), buf);
}

//! computes sum of array elements absolute values
//! supports only single channel images
CV_EXPORTS Scalar absSum(InputArray src, InputArray mask, GpuMat& buf);
static inline Scalar absSum(InputArray src)
{
    GpuMat buf;
    return absSum(src, GpuMat(), buf);
}
static inline Scalar absSum(InputArray src, GpuMat& buf)
{
    return absSum(src, GpuMat(), buf);
}

//! computes squared sum of array elements
//! supports only single channel images
CV_EXPORTS Scalar sqrSum(InputArray src, InputArray mask, GpuMat& buf);
static inline Scalar sqrSum(InputArray src)
{
    GpuMat buf;
    return sqrSum(src, GpuMat(), buf);
}
static inline Scalar sqrSum(InputArray src, GpuMat& buf)
{
    return sqrSum(src, GpuMat(), buf);
}

//! finds global minimum and maximum array elements and returns their values
CV_EXPORTS void minMax(InputArray src, double* minVal, double* maxVal, InputArray mask, GpuMat& buf);
static inline void minMax(InputArray src, double* minVal, double* maxVal=0, InputArray mask=noArray())
{
    GpuMat buf;
    minMax(src, minVal, maxVal, mask, buf);
}

//! finds global minimum and maximum array elements and returns their values with locations
CV_EXPORTS void minMaxLoc(InputArray src, double* minVal, double* maxVal, Point* minLoc, Point* maxLoc,
                          InputArray mask, GpuMat& valbuf, GpuMat& locbuf);
static inline void minMaxLoc(InputArray src, double* minVal, double* maxVal=0, Point* minLoc=0, Point* maxLoc=0,
                             InputArray mask=noArray())
{
    GpuMat valBuf, locBuf;
    minMaxLoc(src, minVal, maxVal, minLoc, maxLoc, mask, valBuf, locBuf);
}

//! counts non-zero array elements
CV_EXPORTS int countNonZero(InputArray src, GpuMat& buf);
static inline int countNonZero(const GpuMat& src)
{
    GpuMat buf;
    return countNonZero(src, buf);
}

//! reduces a matrix to a vector
CV_EXPORTS void reduce(InputArray mtx, OutputArray vec, int dim, int reduceOp, int dtype = -1, Stream& stream = Stream::Null());

//! computes mean value and standard deviation of all or selected array elements
//! supports only CV_8UC1 type
CV_EXPORTS void meanStdDev(InputArray mtx, Scalar& mean, Scalar& stddev, GpuMat& buf);
static inline void meanStdDev(InputArray src, Scalar& mean, Scalar& stddev)
{
    GpuMat buf;
    meanStdDev(src, mean, stddev, buf);
}

//! computes the standard deviation of integral images
//! supports only CV_32SC1 source type and CV_32FC1 sqr type
//! output will have CV_32FC1 type
CV_EXPORTS void rectStdDev(InputArray src, InputArray sqr, OutputArray dst, Rect rect, Stream& stream = Stream::Null());

//! scales and shifts array elements so that either the specified norm (alpha) or the minimum (alpha) and maximum (beta) array values get the specified values
CV_EXPORTS void normalize(InputArray src, OutputArray dst, double alpha, double beta,
                          int norm_type, int dtype, InputArray mask, GpuMat& norm_buf, GpuMat& cvt_buf);
static inline void normalize(InputArray src, OutputArray dst, double alpha = 1, double beta = 0,
                             int norm_type = NORM_L2, int dtype = -1, InputArray mask = noArray())
{
    GpuMat norm_buf;
    GpuMat cvt_buf;
    normalize(src, dst, alpha, beta, norm_type, dtype, mask, norm_buf, cvt_buf);
}

//! computes the integral image
//! sum will have CV_32S type, but will contain unsigned int values
//! supports only CV_8UC1 source type
CV_EXPORTS void integral(InputArray src, OutputArray sum, GpuMat& buffer, Stream& stream = Stream::Null());
static inline void integralBuffered(InputArray src, OutputArray sum, GpuMat& buffer, Stream& stream = Stream::Null())
{
    integral(src, sum, buffer, stream);
}
static inline void integral(InputArray src, OutputArray sum, Stream& stream = Stream::Null())
{
    GpuMat buffer;
    integral(src, sum, buffer, stream);
}

//! computes squared integral image
//! result matrix will have 64F type, but will contain 64U values
//! supports source images of 8UC1 type only
CV_EXPORTS void sqrIntegral(InputArray src, OutputArray sqsum, GpuMat& buf, Stream& stream = Stream::Null());
static inline void sqrIntegral(InputArray src, OutputArray sqsum, Stream& stream = Stream::Null())
{
    GpuMat buffer;
    sqrIntegral(src, sqsum, buffer, stream);
}

CV_EXPORTS void gemm(InputArray src1, InputArray src2, double alpha,
                     InputArray src3, double beta, OutputArray dst, int flags = 0, Stream& stream = Stream::Null());

//! performs per-element multiplication of two full (not packed) Fourier spectrums
//! supports 32FC2 matrixes only (interleaved format)
CV_EXPORTS void mulSpectrums(InputArray src1, InputArray src2, OutputArray dst, int flags, bool conjB=false, Stream& stream = Stream::Null());

//! performs per-element multiplication of two full (not packed) Fourier spectrums
//! supports 32FC2 matrixes only (interleaved format)
CV_EXPORTS void mulAndScaleSpectrums(InputArray src1, InputArray src2, OutputArray dst, int flags, float scale, bool conjB=false, Stream& stream = Stream::Null());

//! Performs a forward or inverse discrete Fourier transform (1D or 2D) of floating point matrix.
//! Param dft_size is the size of DFT transform.
//!
//! If the source matrix is not continous, then additional copy will be done,
//! so to avoid copying ensure the source matrix is continous one. If you want to use
//! preallocated output ensure it is continuous too, otherwise it will be reallocated.
//!
//! Being implemented via CUFFT real-to-complex transform result contains only non-redundant values
//! in CUFFT's format. Result as full complex matrix for such kind of transform cannot be retrieved.
//!
//! For complex-to-real transform it is assumed that the source matrix is packed in CUFFT's format.
CV_EXPORTS void dft(InputArray src, OutputArray dst, Size dft_size, int flags=0, Stream& stream = Stream::Null());

//! computes convolution (or cross-correlation) of two images using discrete Fourier transform
//! supports source images of 32FC1 type only
//! result matrix will have 32FC1 type
class CV_EXPORTS Convolution : public Algorithm
{
public:
    virtual void convolve(InputArray image, InputArray templ, OutputArray result, bool ccorr = false, Stream& stream = Stream::Null()) = 0;
};

CV_EXPORTS Ptr<Convolution> createConvolution(Size user_block_size = Size());

}} // namespace cv { namespace cuda {

#endif /* __OPENCV_CUDAARITHM_HPP__ */
