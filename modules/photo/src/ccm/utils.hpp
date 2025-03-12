// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
//
//                       License Agreement
//              For Open Source Computer Vision Library
//
// Copyright(C) 2020, Huawei Technologies Co.,Ltd. All rights reserved.
// Third party copyrights are property of their respective owners.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//             http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#ifndef __OPENCV_CCM_UTILS_HPP__
#define __OPENCV_CCM_UTILS_HPP__

#include <opencv2/core.hpp>

namespace cv {
namespace ccm {

double gammaCorrection_(const double& element, const double& gamma);

/** @brief gamma correction.
           \f[
            C_l=C_n^{\gamma},\qquad C_n\ge0\\
            C_l=-(-C_n)^{\gamma},\qquad C_n<0\\\\
            \f]
    @param src the input array,type of Mat.
    @param gamma a constant for gamma correction.
    @param dst the output array, type of Mat.
 */
Mat gammaCorrection(const Mat& src, const double& gamma, Mat dst=Mat());

/** @brief maskCopyTo a function to delete unsatisfied elementwise.
    @param src the input array, type of Mat.
    @param mask operation mask that used to choose satisfided elementwise.
 */
Mat maskCopyTo(const Mat& src, const Mat& mask);

/** @brief multiple the function used to compute an array with n channels
      mulipied by ccm.
    @param xyz the input array, type of Mat.
    @param ccm the ccm matrix to make color correction.
 */
Mat multiple(const Mat& xyz, const Mat& ccm);

/** @brief multiple the function used to get the mask of saturated colors,
            colors between low and up will be choosed.
    @param src the input array, type of Mat.
    @param low  the threshold to choose saturated colors
    @param up  the threshold to choose saturated colors
*/
Mat saturate(Mat& src, const double& low, const double& up);

/** @brief rgb2gray it is an approximation grayscale function for relative RGB
           color space
    @param  rgb the input array,type of Mat.
 */
Mat rgb2gray(const Mat& rgb);

/** @brief function for elementWise operation
    @param src the input array, type of Mat
    @param lambda a for operation
 */
template <typename F>
Mat elementWise(const Mat& src, F&& lambda, Mat dst=Mat())
{
    if (dst.empty() || !dst.isContinuous() || dst.total() != src.total() || dst.type() != src.type())
        dst = Mat(src.rows, src.cols, src.type());
    const int channel = src.channels();
    if (src.isContinuous()) {
        const int num_elements = (int)src.total()*channel;
        const double *psrc = (double*)src.data;
        double *pdst = (double*)dst.data;
        const int batch = getNumThreads() > 1 ? 128 : num_elements;
        const int N = (num_elements / batch) + ((num_elements % batch) > 0);
        parallel_for_(Range(0, N),[&](const Range& range) {
            const int start = range.start * batch;
            const int end = std::min(range.end*batch, num_elements);
            for (int i = start; i < end; i++) {
                pdst[i] = lambda(psrc[i]);
            }
        });
        return dst;
    }
    switch (channel)
    {
    case 1:
    {

        MatIterator_<double> it, end;
        for (it = dst.begin<double>(), end = dst.end<double>(); it != end; ++it)
        {
            (*it) = lambda((*it));
        }
        break;
    }
    case 3:
    {
        MatIterator_<Vec3d> it, end;
        for (it = dst.begin<Vec3d>(), end = dst.end<Vec3d>(); it != end; ++it)
        {
            for (int j = 0; j < 3; j++)
            {
                (*it)[j] = lambda((*it)[j]);
            }
        }
        break;
    }
    default:
        CV_Error(Error::StsBadArg, "Wrong channel!" );
        break;
    }
    return dst;
}

/** @brief function for channel operation
      @param src the input array, type of Mat
      @param lambda the function for operation
*/
template <typename F>
Mat channelWise(const Mat& src, F&& lambda)
{
    Mat dst = src.clone();
    MatIterator_<Vec3d> it, end;
    for (it = dst.begin<Vec3d>(), end = dst.end<Vec3d>(); it != end; ++it)
    {
        *it = lambda(*it);
    }
    return dst;
}

/** @brief function for distance operation.
    @param src the input array, type of Mat.
    @param ref another input array, type of Mat.
    @param lambda the computing method for distance .
 */
template <typename F>
Mat distanceWise(Mat& src, Mat& ref, F&& lambda)
{
    Mat dst = Mat(src.size(), CV_64FC1);
    MatIterator_<Vec3d> it_src = src.begin<Vec3d>(), end_src = src.end<Vec3d>(),
                        it_ref = ref.begin<Vec3d>();
    MatIterator_<double> it_dst = dst.begin<double>();
    for (; it_src != end_src; ++it_src, ++it_ref, ++it_dst)
    {
        *it_dst = lambda(*it_src, *it_ref);
    }
    return dst;
}

Mat multiple(const Mat& xyz, const Mat& ccm);

}
}  // namespace cv::ccm

#endif