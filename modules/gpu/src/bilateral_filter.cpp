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

#include "precomp.hpp"

using namespace cv;
using namespace cv::gpu;
using namespace std;

#if !defined (HAVE_CUDA)

cv::gpu::DisparityBilateralFilter::DisparityBilateralFilter(int, int, int) { throw_nogpu(); }
cv::gpu::DisparityBilateralFilter::DisparityBilateralFilter(int, int, int, float, float, float) { throw_nogpu(); }

void cv::gpu::DisparityBilateralFilter::operator()(const GpuMat&, const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu { namespace device 
{
    namespace bilateral_filter
    {
        void load_constants(float* table_color, DevMem2Df table_space, int ndisp, int radius, short edge_disc, short max_disc);

        void bilateral_filter_gpu(DevMem2Db disp, DevMem2Db img, int channels, int iters, cudaStream_t stream);
        void bilateral_filter_gpu(DevMem2D_<short> disp, DevMem2Db img, int channels, int iters, cudaStream_t stream);
    }
}}}

using namespace ::cv::gpu::device::bilateral_filter;

namespace
{
    const float DEFAULT_EDGE_THRESHOLD = 0.1f;
    const float DEFAULT_MAX_DISC_THRESHOLD = 0.2f;
    const float DEFAULT_SIGMA_RANGE = 10.0f;

    inline void calc_color_weighted_table(GpuMat& table_color, float sigma_range, int len)
    {
        Mat cpu_table_color(1, len, CV_32F);

        float* line = cpu_table_color.ptr<float>();

        for(int i = 0; i < len; i++) 
            line[i] = static_cast<float>(std::exp(-double(i * i) / (2 * sigma_range * sigma_range)));

        table_color.upload(cpu_table_color);
    }

    inline void calc_space_weighted_filter(GpuMat& table_space, int win_size, float dist_space)
    {
        int half = (win_size >> 1);

        Mat cpu_table_space(half + 1, half + 1, CV_32F);

        for (int y = 0; y <= half; ++y)
        {
            float* row = cpu_table_space.ptr<float>(y);
            for (int x = 0; x <= half; ++x)
                row[x] = exp(-sqrt(float(y * y) + float(x * x)) / dist_space);
        }

        table_space.upload(cpu_table_space);
    }

    template <typename T>
    void bilateral_filter_operator(int ndisp, int radius, int iters, float edge_threshold,float max_disc_threshold, 
                                   GpuMat& table_color, GpuMat& table_space, 
                                   const GpuMat& disp, const GpuMat& img, GpuMat& dst, Stream& stream)
    {
        short edge_disc = max<short>(short(1), short(ndisp * edge_threshold + 0.5));
        short max_disc = short(ndisp * max_disc_threshold + 0.5);

        load_constants(table_color.ptr<float>(), table_space, ndisp, radius, edge_disc, max_disc);

        if (&dst != &disp)
        {
            if (stream)
                stream.enqueueCopy(disp, dst);
            else
                disp.copyTo(dst);
        }

        bilateral_filter_gpu((DevMem2D_<T>)dst, img, img.channels(), iters, StreamAccessor::getStream(stream));
    }

    typedef void (*bilateral_filter_operator_t)(int ndisp, int radius, int iters, float edge_threshold, float max_disc_threshold, 
                                                GpuMat& table_color, GpuMat& table_space, 
                                                const GpuMat& disp, const GpuMat& img, GpuMat& dst, Stream& stream);
    
    const bilateral_filter_operator_t operators[] = 
        {bilateral_filter_operator<unsigned char>, 0, 0, bilateral_filter_operator<short>, 0, 0, 0, 0};
}

cv::gpu::DisparityBilateralFilter::DisparityBilateralFilter(int ndisp_, int radius_, int iters_)
    : ndisp(ndisp_), radius(radius_), iters(iters_), edge_threshold(DEFAULT_EDGE_THRESHOLD), max_disc_threshold(DEFAULT_MAX_DISC_THRESHOLD),
      sigma_range(DEFAULT_SIGMA_RANGE)
{
    calc_color_weighted_table(table_color, sigma_range, 255);
    calc_space_weighted_filter(table_space, radius * 2 + 1, radius + 1.0f);
}

cv::gpu::DisparityBilateralFilter::DisparityBilateralFilter(int ndisp_, int radius_, int iters_, float edge_threshold_, 
                                                     float max_disc_threshold_, float sigma_range_)
    : ndisp(ndisp_), radius(radius_), iters(iters_), edge_threshold(edge_threshold_), max_disc_threshold(max_disc_threshold_), 
      sigma_range(sigma_range_)
{
    calc_color_weighted_table(table_color, sigma_range, 255);
    calc_space_weighted_filter(table_space, radius * 2 + 1, radius + 1.0f);
}

void cv::gpu::DisparityBilateralFilter::operator()(const GpuMat& disp, const GpuMat& img, GpuMat& dst, Stream& stream)
{
    CV_DbgAssert(0 < ndisp && 0 < radius && 0 < iters);
    CV_Assert(disp.rows == img.rows && disp.cols == img.cols && (disp.type() == CV_8U || disp.type() == CV_16S) && (img.type() == CV_8UC1 || img.type() == CV_8UC3));
    operators[disp.type()](ndisp, radius, iters, edge_threshold, max_disc_threshold, table_color, table_space, disp, img, dst, stream);
}

#endif /* !defined (HAVE_CUDA) */
