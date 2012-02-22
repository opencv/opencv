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
//     and/or other GpuMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or bpied warranties, including, but not limited to, the bpied
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

#include "precomp.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

#if !defined (HAVE_CUDA)

void cv::gpu::GoodFeaturesToTrackDetector_GPU::operator ()(const GpuMat&, GpuMat&, const GpuMat&) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu { namespace device 
{
    namespace gfft 
    {
        int findCorners_gpu(DevMem2Df eig, float threshold, DevMem2Db mask, float2* corners, int max_count);
        void sortCorners_gpu(DevMem2Df eig, float2* corners, int count);
    }
}}}

void cv::gpu::GoodFeaturesToTrackDetector_GPU::operator ()(const GpuMat& image, GpuMat& corners, const GpuMat& mask)
{
    using namespace cv::gpu::device::gfft;

    CV_Assert(qualityLevel > 0 && minDistance >= 0 && maxCorners >= 0);
    CV_Assert(mask.empty() || (mask.type() == CV_8UC1 && mask.size() == image.size()));
    CV_Assert(TargetArchs::builtWith(GLOBAL_ATOMICS) && DeviceInfo().supports(GLOBAL_ATOMICS));

    ensureSizeIsEnough(image.size(), CV_32F, eig_);

    if (useHarrisDetector)
        cornerHarris(image, eig_, Dx_, Dy_, buf_, blockSize, 3, harrisK);
    else
        cornerMinEigenVal(image, eig_, Dx_, Dy_, buf_, blockSize, 3);

    double maxVal = 0;
    minMax(eig_, 0, &maxVal, GpuMat(), minMaxbuf_);

    ensureSizeIsEnough(1, std::max(1000, static_cast<int>(image.size().area() * 0.05)), CV_32FC2, tmpCorners_);

    int total = findCorners_gpu(eig_, static_cast<float>(maxVal * qualityLevel), mask, tmpCorners_.ptr<float2>(), tmpCorners_.cols);

    sortCorners_gpu(eig_, tmpCorners_.ptr<float2>(), total);

    if (minDistance < 1)
        tmpCorners_.colRange(0, maxCorners > 0 ? std::min(maxCorners, total) : total).copyTo(corners);
    else
    {
        vector<Point2f> tmp(total);
        Mat tmpMat(1, total, CV_32FC2, (void*)&tmp[0]);
        tmpCorners_.colRange(0, total).download(tmpMat);

        vector<Point2f> tmp2;
        tmp2.reserve(total);

        const int cell_size = cvRound(minDistance);
        const int grid_width = (image.cols + cell_size - 1) / cell_size;
        const int grid_height = (image.rows + cell_size - 1) / cell_size;

        std::vector< std::vector<Point2f> > grid(grid_width * grid_height);

        for (int i = 0; i < total; ++i)
        {
            Point2f p = tmp[i];

	        bool good = true;

            int x_cell = static_cast<int>(p.x / cell_size);
            int y_cell = static_cast<int>(p.y / cell_size);

            int x1 = x_cell - 1;
            int y1 = y_cell - 1;
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width - 1, x2);
            y2 = std::min(grid_height - 1, y2);

            for (int yy = y1; yy <= y2; yy++)
            {
                for (int xx = x1; xx <= x2; xx++)
                {   
                    vector<Point2f>& m = grid[yy * grid_width + xx];

                    if (!m.empty())
                    {
                        for(size_t j = 0; j < m.size(); j++)
                        {
                            float dx = p.x - m[j].x;
                            float dy = p.y - m[j].y;

                            if (dx * dx + dy * dy < minDistance * minDistance)
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }                
                }
            }

            break_out:

            if(good)
            {
                grid[y_cell * grid_width + x_cell].push_back(p);

                tmp2.push_back(p);

                if (maxCorners > 0 && tmp2.size() == static_cast<size_t>(maxCorners))
                    break;
            }
        }

        corners.upload(Mat(1, tmp2.size(), CV_32FC2, &tmp2[0]));
    }
}

#endif /* !defined (HAVE_CUDA) */
