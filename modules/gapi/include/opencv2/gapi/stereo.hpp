// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distereoibution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STEREO_HPP
#define OPENCV_GAPI_STEREO_HPP

#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/gscalar.hpp>
#include <opencv2/gapi/gkernel.hpp>

namespace cv {
namespace gapi {

/// The enum specified format of result that you get from @ref stereo.
enum class StereoOutputFormat {
    DEPTH_FLOAT16, ///< Floating point 16 bit value, CV_16FC1
    DEPTH_FLOAT32, ///< Floating point 32 bit value, CV_32FC1
    DISPARITY_FIXED16_11_5, ///< 16 bit signed. First bit for sign,
                            ///< 10 bits for integer,
                            ///< 5 bits for fractional
    DISPARITY_FIXED16_12_4, ///< 16 bit signed: first bit for sign,
                            ///< 11 bits for integer,
                            ///< 4 bits for fractional
    DEPTH_F16, ///< Floating point 16 bit value, CV_16FC1
    DEPTH_F32, ///< Floating point 32 bit value, CV_32FC1
    DISPARITY_Q16_10_5, ///< 16 bit signed. first bit for sign,
                        ///< 10 bits for integer,
                        ///< 5 bits for fractional
    DISPARITY_Q16_11_4  ///< 16 bit signed: first bit for sign,
                        ///< 11 bits for integer,
                        ///< 4 bits for fractional
};

namespace calib3d {

G_TYPED_KERNEL(GStereo, <GMat(GMat, GMat, const StereoOutputFormat)>, "org.opencv.stereo") {
    static GMatDesc outMeta(const GMatDesc &left, const GMatDesc &right, const StereoOutputFormat of) {
        GAPI_Assert(left.chan == 1);
        GAPI_Assert(left.depth == CV_8U);

        GAPI_Assert(right.chan == 1);
        GAPI_Assert(right.depth == CV_8U);

        switch(of) {
            case StereoOutputFormat::DEPTH_F16:
            case StereoOutputFormat::DEPTH_FLOAT16:
                return left.withDepth(CV_16FC1);
            case StereoOutputFormat::DEPTH_F32:
            case StereoOutputFormat::DEPTH_FLOAT32:
                return left.withDepth(CV_32FC1);
            case StereoOutputFormat::DISPARITY_Q16_10_5:
            case StereoOutputFormat::DISPARITY_FIXED16_11_5:

            case StereoOutputFormat::DISPARITY_Q16_11_4:
            case StereoOutputFormat::DISPARITY_FIXED16_12_4:
                return left.withDepth(CV_16SC1);
            default:
                GAPI_Assert(false && "Unknown output format!");
        }
    }
};

} // namespace calib3d

/** @brief Computes disparity/depth map for the specified stereo-pair.
The function compute disparity or depth map depending on passed StereoOutputFormat argument.

@param left 8-bit single-channel image from left source.
@param right 8-bit single-channel image from right source.
@param of enum to specified output kind: depth or disparity and corresponding type
*/
GAPI_EXPORTS GMat stereo(const GMat& left,
                         const GMat& right,
                         const StereoOutputFormat of = StereoOutputFormat::DEPTH_FLOAT32);
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_STEREO_HPP
