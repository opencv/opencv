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
// Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#ifndef OPENCV_CORE_IPPASYNC_HPP
#define OPENCV_CORE_IPPASYNC_HPP

#ifdef HAVE_IPP_A  // this file will be removed in OpenCV 4.0

#include "opencv2/core.hpp"
#include <ipp_async_op.h>
#include <ipp_async_accel.h>

namespace cv
{

namespace hpp
{

/** @addtogroup core_ipp
This section describes conversion between OpenCV and [Intel&reg; IPP Asynchronous
C/C++](http://software.intel.com/en-us/intel-ipp-preview) library. [Getting Started
Guide](http://registrationcenter.intel.com/irc_nas/3727/ipp_async_get_started.htm) help you to
install the library, configure header and library build paths.
 */
//! @{

    //! convert OpenCV data type to hppDataType
    inline int toHppType(const int cvType)
    {
        int depth = CV_MAT_DEPTH(cvType);
        int hppType = depth == CV_8U ? HPP_DATA_TYPE_8U :
                     depth == CV_16U ? HPP_DATA_TYPE_16U :
                     depth == CV_16S ? HPP_DATA_TYPE_16S :
                     depth == CV_32S ? HPP_DATA_TYPE_32S :
                     depth == CV_32F ? HPP_DATA_TYPE_32F :
                     depth == CV_64F ? HPP_DATA_TYPE_64F : -1;
        CV_Assert( hppType >= 0 );
        return hppType;
    }

    //! convert hppDataType to OpenCV data type
    inline int toCvType(const int hppType)
    {
        int cvType = hppType == HPP_DATA_TYPE_8U ? CV_8U :
                    hppType == HPP_DATA_TYPE_16U ? CV_16U :
                    hppType == HPP_DATA_TYPE_16S ? CV_16S :
                    hppType == HPP_DATA_TYPE_32S ? CV_32S :
                    hppType == HPP_DATA_TYPE_32F ? CV_32F :
                    hppType == HPP_DATA_TYPE_64F ? CV_64F : -1;
        CV_Assert( cvType >= 0 );
        return cvType;
    }

    /** @brief Convert hppiMatrix to Mat.

    This function allocates and initializes new matrix (if needed) that has the same size and type as
    input matrix. Supports CV_8U, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F.
    @param src input hppiMatrix.
    @param dst output matrix.
    @param accel accelerator instance (see hpp::getHpp for the list of acceleration framework types).
    @param cn number of channels.
     */
    inline void copyHppToMat(hppiMatrix* src, Mat& dst, hppAccel accel, int cn)
    {
        hppDataType type;
        hpp32u width, height;
        hppStatus sts;

        if (src == NULL)
            return dst.release();

        sts = hppiInquireMatrix(src, &type, &width, &height);

        CV_Assert( sts == HPP_STATUS_NO_ERROR);

        int matType = CV_MAKETYPE(toCvType(type), cn);

        CV_Assert(width%cn == 0);

        width /= cn;

        dst.create((int)height, (int)width, (int)matType);

        size_t newSize = (size_t)(height*(hpp32u)(dst.step));

        sts = hppiGetMatrixData(accel,src,(hpp32u)(dst.step),dst.data,&newSize);

        CV_Assert( sts == HPP_STATUS_NO_ERROR);
    }

    /** @brief Create Mat from hppiMatrix.

    This function allocates and initializes the Mat that has the same size and type as input matrix.
    Supports CV_8U, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F.
    @param src input hppiMatrix.
    @param accel accelerator instance (see hpp::getHpp for the list of acceleration framework types).
    @param cn number of channels.
    @sa howToUseIPPAconversion, hpp::copyHppToMat, hpp::getHpp.
     */
    inline Mat getMat(hppiMatrix* src, hppAccel accel, int cn)
    {
        Mat dst;
        copyHppToMat(src, dst, accel, cn);
        return dst;
    }

    /** @brief Create hppiMatrix from Mat.

    This function allocates and initializes the hppiMatrix that has the same size and type as input
    matrix, returns the hppiMatrix*.

    If you want to use zero-copy for GPU you should to have 4KB aligned matrix data. See details
    [hppiCreateSharedMatrix](http://software.intel.com/ru-ru/node/501697).

    Supports CV_8U, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F.

    @note The hppiMatrix pointer to the image buffer in system memory refers to the src.data. Control
    the lifetime of the matrix and don't change its data, if there is no special need.
    @param src input matrix.
    @param accel accelerator instance. Supports type:
    -   **HPP_ACCEL_TYPE_CPU** - accelerated by optimized CPU instructions.
    -   **HPP_ACCEL_TYPE_GPU** - accelerated by GPU programmable units or fixed-function
        accelerators.
    -   **HPP_ACCEL_TYPE_ANY** - any acceleration or no acceleration available.
    @sa howToUseIPPAconversion, hpp::getMat
     */
    inline hppiMatrix* getHpp(const Mat& src, hppAccel accel)
    {
        int htype = toHppType(src.type());
        int cn = src.channels();

        CV_Assert(src.data);
        hppAccelType accelType = hppQueryAccelType(accel);

        if (accelType!=HPP_ACCEL_TYPE_CPU)
        {
            hpp32u pitch, size;
            hppQueryMatrixAllocParams(accel, src.cols*cn, src.rows, htype, &pitch, &size);
            if (pitch!=0 && size!=0)
                if ((int)(src.data)%4096==0 && pitch==(hpp32u)(src.step))
                {
                    return hppiCreateSharedMatrix(htype, src.cols*cn, src.rows, src.data, pitch, size);
                }
        }

        return hppiCreateMatrix(htype, src.cols*cn, src.rows, src.data, (hpp32s)(src.step));;
    }

//! @}
}}

#endif

#endif
