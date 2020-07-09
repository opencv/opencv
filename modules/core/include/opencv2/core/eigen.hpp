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


#ifndef OPENCV_CORE_EIGEN_HPP
#define OPENCV_CORE_EIGEN_HPP

#ifndef EIGEN_WORLD_VERSION
#error "Wrong usage of OpenCV's Eigen utility header. Include Eigen's headers first. See https://github.com/opencv/opencv/issues/17366"
#endif

#include "opencv2/core.hpp"

#if defined _MSC_VER && _MSC_VER >= 1200
#define NOMINMAX // fix https://github.com/opencv/opencv/issues/17548
#pragma warning( disable: 4714 ) //__forceinline is not inlined
#pragma warning( disable: 4127 ) //conditional expression is constant
#pragma warning( disable: 4244 ) //conversion from '__int64' to 'int', possible loss of data
#endif

#if EIGEN_WORLD_VERSION == 3 && EIGEN_MAJOR_VERSION >= 3 \
    && defined(CV_CXX11) && defined(CV_CXX_STD_ARRAY)
#include <unsupported/Eigen/CXX11/Tensor>
#define OPENCV_EIGEN_TENSOR_SUPPORT
#endif // EIGEN_WORLD_VERSION == 3 && EIGEN_MAJOR_VERSION >= 3

namespace cv
{

/** @addtogroup core_eigen
These functions are provided for OpenCV-Eigen interoperability. They convert `Mat`
objects to corresponding `Eigen::Matrix` objects and vice-versa. Consult the [Eigen
documentation](https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html) for
information about the `Matrix` template type.

@note Using these functions requires the `Eigen/Dense` or similar header to be
included before this header.
*/
//! @{

#if defined(OPENCV_EIGEN_TENSOR_SUPPORT) || defined(CV_DOXYGEN)
/** @brief Converts an Eigen::Tensor to a cv::Mat.

The method converts an Eigen::Tensor with shape (H x W x C) to a cv::Mat where:
 H = number of rows
 W = number of columns
 C = number of channels

Usage:
\code
Eigen::Tensor<float, 3, Eigen::RowMajor> a_tensor(...);
// populate tensor with values
Mat a_mat;
eigen2cv(a_tensor, a_mat);
\endcode
*/
template <typename _Tp, int _layout> static inline
void eigen2cv( const Eigen::Tensor<_Tp, 3, _layout> &src, OutputArray dst )
{
    if( !(_layout & Eigen::RowMajorBit) )
    {
        const std::array<int, 3> shuffle{2, 1, 0};
        Eigen::Tensor<_Tp, 3, !_layout> row_major_tensor = src.swap_layout().shuffle(shuffle);
        Mat _src(src.dimension(0), src.dimension(1), CV_MAKETYPE(DataType<_Tp>::type, src.dimension(2)), row_major_tensor.data());
        _src.copyTo(dst);
    }
    else
    {
        Mat _src(src.dimension(0), src.dimension(1), CV_MAKETYPE(DataType<_Tp>::type, src.dimension(2)), (void *)src.data());
        _src.copyTo(dst);
    }
}

/** @brief Converts a cv::Mat to an Eigen::Tensor.

The method converts a cv::Mat to an Eigen Tensor with shape (H x W x C) where:
 H = number of rows
 W = number of columns
 C = number of channels

Usage:
\code
Mat a_mat(...);
// populate Mat with values
Eigen::Tensor<float, 3, Eigen::RowMajor> a_tensor(...);
cv2eigen(a_mat, a_tensor);
\endcode
*/
template <typename _Tp, int _layout> static inline
void cv2eigen( const Mat &src, Eigen::Tensor<_Tp, 3, _layout> &dst )
{
    if( !(_layout & Eigen::RowMajorBit) )
    {
        Eigen::Tensor<_Tp, 3, !_layout> row_major_tensor(src.rows, src.cols, src.channels());
        Mat _dst(src.rows, src.cols, CV_MAKETYPE(DataType<_Tp>::type, src.channels()), row_major_tensor.data());
        if (src.type() == _dst.type())
            src.copyTo(_dst);
        else
            src.convertTo(_dst, _dst.type());
        const std::array<int, 3> shuffle{2, 1, 0};
        dst = row_major_tensor.swap_layout().shuffle(shuffle);
    }
    else
    {
        dst.resize(src.rows, src.cols, src.channels());
        Mat _dst(src.rows, src.cols, CV_MAKETYPE(DataType<_Tp>::type, src.channels()), dst.data());
        if (src.type() == _dst.type())
            src.copyTo(_dst);
        else
            src.convertTo(_dst, _dst.type());
    }
}

/** @brief Maps cv::Mat data to an Eigen::TensorMap.

The method wraps an existing Mat data array with an Eigen TensorMap of shape (H x W x C) where:
 H = number of rows
 W = number of columns
 C = number of channels

Explicit instantiation of the return type is required.

@note Caller should be aware of the lifetime of the cv::Mat instance and take appropriate safety measures.
The cv::Mat instance will retain ownership of the data and the Eigen::TensorMap will lose access when the cv::Mat data is deallocated.

The example below initializes a cv::Mat and produces an Eigen::TensorMap:
\code
float arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
Mat a_mat(2, 2, CV_32FC3, arr);
Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>> a_tensormap = cv2eigen_tensormap<float>(a_mat);
\endcode
*/
template <typename _Tp> static inline
Eigen::TensorMap<Eigen::Tensor<_Tp, 3, Eigen::RowMajor>> cv2eigen_tensormap(InputArray src)
{
    Mat mat = src.getMat();
    CV_CheckTypeEQ(mat.type(), CV_MAKETYPE(traits::Type<_Tp>::value, mat.channels()), "");
    return Eigen::TensorMap<Eigen::Tensor<_Tp, 3, Eigen::RowMajor>>((_Tp *)mat.data, mat.rows, mat.cols, mat.channels());
}
#endif // OPENCV_EIGEN_TENSOR_SUPPORT

template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols> static inline
void eigen2cv( const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& src, OutputArray dst )
{
    if( !(src.Flags & Eigen::RowMajorBit) )
    {
        Mat _src(src.cols(), src.rows(), traits::Type<_Tp>::value,
              (void*)src.data(), src.outerStride()*sizeof(_Tp));
        transpose(_src, dst);
    }
    else
    {
        Mat _src(src.rows(), src.cols(), traits::Type<_Tp>::value,
                 (void*)src.data(), src.outerStride()*sizeof(_Tp));
        _src.copyTo(dst);
    }
}

// Matx case
template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols> static inline
void eigen2cv( const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& src,
               Matx<_Tp, _rows, _cols>& dst )
{
    if( !(src.Flags & Eigen::RowMajorBit) )
    {
        dst = Matx<_Tp, _cols, _rows>(static_cast<const _Tp*>(src.data())).t();
    }
    else
    {
        dst = Matx<_Tp, _rows, _cols>(static_cast<const _Tp*>(src.data()));
    }
}

template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols> static inline
void cv2eigen( const Mat& src,
               Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& dst )
{
    CV_DbgAssert(src.rows == _rows && src.cols == _cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(src.cols, src.rows, traits::Type<_Tp>::value,
                 dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        if( src.type() == _dst.type() )
            transpose(src, _dst);
        else if( src.cols == src.rows )
        {
            src.convertTo(_dst, _dst.type());
            transpose(_dst, _dst);
        }
        else
            Mat(src.t()).convertTo(_dst, _dst.type());
    }
    else
    {
        const Mat _dst(src.rows, src.cols, traits::Type<_Tp>::value,
                 dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
    }
}

// Matx case
template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols> static inline
void cv2eigen( const Matx<_Tp, _rows, _cols>& src,
               Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& dst )
{
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(_cols, _rows, traits::Type<_Tp>::value,
                 dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        transpose(src, _dst);
    }
    else
    {
        const Mat _dst(_rows, _cols, traits::Type<_Tp>::value,
                 dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        Mat(src).copyTo(_dst);
    }
}

template<typename _Tp>  static inline
void cv2eigen( const Mat& src,
               Eigen::Matrix<_Tp, Eigen::Dynamic, Eigen::Dynamic>& dst )
{
    dst.resize(src.rows, src.cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(src.cols, src.rows, traits::Type<_Tp>::value,
             dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        if( src.type() == _dst.type() )
            transpose(src, _dst);
        else if( src.cols == src.rows )
        {
            src.convertTo(_dst, _dst.type());
            transpose(_dst, _dst);
        }
        else
            Mat(src.t()).convertTo(_dst, _dst.type());
    }
    else
    {
        const Mat _dst(src.rows, src.cols, traits::Type<_Tp>::value,
                 dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
    }
}

// Matx case
template<typename _Tp, int _rows, int _cols> static inline
void cv2eigen( const Matx<_Tp, _rows, _cols>& src,
               Eigen::Matrix<_Tp, Eigen::Dynamic, Eigen::Dynamic>& dst )
{
    dst.resize(_rows, _cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(_cols, _rows, traits::Type<_Tp>::value,
             dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        transpose(src, _dst);
    }
    else
    {
        const Mat _dst(_rows, _cols, traits::Type<_Tp>::value,
                 dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        Mat(src).copyTo(_dst);
    }
}

template<typename _Tp> static inline
void cv2eigen( const Mat& src,
               Eigen::Matrix<_Tp, Eigen::Dynamic, 1>& dst )
{
    CV_Assert(src.cols == 1);
    dst.resize(src.rows);

    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(src.cols, src.rows, traits::Type<_Tp>::value,
                 dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        if( src.type() == _dst.type() )
            transpose(src, _dst);
        else
            Mat(src.t()).convertTo(_dst, _dst.type());
    }
    else
    {
        const Mat _dst(src.rows, src.cols, traits::Type<_Tp>::value,
                 dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
    }
}

// Matx case
template<typename _Tp, int _rows> static inline
void cv2eigen( const Matx<_Tp, _rows, 1>& src,
               Eigen::Matrix<_Tp, Eigen::Dynamic, 1>& dst )
{
    dst.resize(_rows);

    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(1, _rows, traits::Type<_Tp>::value,
                 dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        transpose(src, _dst);
    }
    else
    {
        const Mat _dst(_rows, 1, traits::Type<_Tp>::value,
                 dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        src.copyTo(_dst);
    }
}


template<typename _Tp> static inline
void cv2eigen( const Mat& src,
               Eigen::Matrix<_Tp, 1, Eigen::Dynamic>& dst )
{
    CV_Assert(src.rows == 1);
    dst.resize(src.cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(src.cols, src.rows, traits::Type<_Tp>::value,
                 dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        if( src.type() == _dst.type() )
            transpose(src, _dst);
        else
            Mat(src.t()).convertTo(_dst, _dst.type());
    }
    else
    {
        const Mat _dst(src.rows, src.cols, traits::Type<_Tp>::value,
                 dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
    }
}

//Matx
template<typename _Tp, int _cols> static inline
void cv2eigen( const Matx<_Tp, 1, _cols>& src,
               Eigen::Matrix<_Tp, 1, Eigen::Dynamic>& dst )
{
    dst.resize(_cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(_cols, 1, traits::Type<_Tp>::value,
                 dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        transpose(src, _dst);
    }
    else
    {
        const Mat _dst(1, _cols, traits::Type<_Tp>::value,
                 dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        Mat(src).copyTo(_dst);
    }
}

//! @}

} // cv

#endif
