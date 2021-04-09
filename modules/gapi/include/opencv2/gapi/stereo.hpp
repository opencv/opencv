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

enum class StereoOutputFormat {
    DEPTH_FLOAT16,
    DEPTH_FLOAT32,
    DISPARITY_FIXED16_11_5,
    DISPARITY_FIXED16_12_4
};

namespace calib3d {

G_TYPED_KERNEL(GStereo, <GMat(GMat, GMat, const StereoOutputFormat)>, "org.opencv.stereo") {
    static GMatDesc outMeta(const GMatDesc &left, const GMatDesc &right, const StereoOutputFormat of) {
        GAPI_Assert(left.chan == 1);
        GAPI_Assert(left.depth == CV_8U);

        GAPI_Assert(right.chan == 1);
        GAPI_Assert(right.depth == CV_8U);

        switch(of) {
            case StereoOutputFormat::DEPTH_FLOAT16:
                return left.withDepth(CV_16FC1);
            case StereoOutputFormat::DEPTH_FLOAT32:
                return left.withDepth(CV_32FC1);
            case StereoOutputFormat::DISPARITY_FIXED16_11_5:
            case StereoOutputFormat::DISPARITY_FIXED16_12_4:
                return left.withDepth(CV_16SC1);
            default:
                GAPI_Assert(false && "Unknown output format!");
        }
    }
};

} // namespace calib3d

/** @brief Extract disparity/depth information depending on passed StereoOutputFormat argument.
The function extracts disparity/depth information depending on passed StereoOutputFormat argument from
given stereo-pair.

@param left left 8-bit unsigned 1-channel image of @ref CV_8UC1 type
@param right right 8-bit unsigned 1-channel image of @ref CV_8UC1 type
@param of enum to specify output kind: depth or disparity and corresponding type
*/
GAPI_EXPORTS GMat stereo(const GMat& left,
                         const GMat& right,
                         const StereoOutputFormat of = StereoOutputFormat::DEPTH_FLOAT32);
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_STEREO_HPP
