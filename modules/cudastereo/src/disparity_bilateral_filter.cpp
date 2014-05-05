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

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

Ptr<cuda::DisparityBilateralFilter> cv::cuda::createDisparityBilateralFilter(int, int, int) { throw_no_cuda(); return Ptr<cuda::DisparityBilateralFilter>(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace cuda { namespace device
{
    namespace disp_bilateral_filter
    {
        void disp_load_constants(float* table_color, PtrStepSzf table_space, int ndisp, int radius, short edge_disc, short max_disc);

        template<typename T>
        void disp_bilateral_filter(PtrStepSz<T> disp, PtrStepSzb img, int channels, int iters, cudaStream_t stream);
    }
}}}

namespace
{
    class DispBilateralFilterImpl : public cuda::DisparityBilateralFilter
    {
    public:
        DispBilateralFilterImpl(int ndisp, int radius, int iters);

        void apply(InputArray disparity, InputArray image, OutputArray dst, Stream& stream);

        int getNumDisparities() const { return ndisp_; }
        void setNumDisparities(int numDisparities) { ndisp_ = numDisparities; }

        int getRadius() const { return radius_; }
        void setRadius(int radius);

        int getNumIters() const { return iters_; }
        void setNumIters(int iters) { iters_ = iters; }

        double getEdgeThreshold() const { return edge_threshold_; }
        void setEdgeThreshold(double edge_threshold) { edge_threshold_ = (float) edge_threshold; }

        double getMaxDiscThreshold() const { return max_disc_threshold_; }
        void setMaxDiscThreshold(double max_disc_threshold) { max_disc_threshold_ = (float) max_disc_threshold; }

        double getSigmaRange() const { return sigma_range_; }
        void setSigmaRange(double sigma_range);

    private:
        int ndisp_;
        int radius_;
        int iters_;
        float edge_threshold_;
        float max_disc_threshold_;
        float sigma_range_;

        GpuMat table_color_;
        GpuMat table_space_;
    };

    void calc_color_weighted_table(GpuMat& table_color, float sigma_range, int len)
    {
        Mat cpu_table_color(1, len, CV_32F);

        float* line = cpu_table_color.ptr<float>();

        for(int i = 0; i < len; i++)
            line[i] = static_cast<float>(std::exp(-double(i * i) / (2 * sigma_range * sigma_range)));

        table_color.upload(cpu_table_color);
    }

    void calc_space_weighted_filter(GpuMat& table_space, int win_size, float dist_space)
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

    const float DEFAULT_EDGE_THRESHOLD = 0.1f;
    const float DEFAULT_MAX_DISC_THRESHOLD = 0.2f;
    const float DEFAULT_SIGMA_RANGE = 10.0f;

    DispBilateralFilterImpl::DispBilateralFilterImpl(int ndisp, int radius, int iters) :
        ndisp_(ndisp), radius_(radius), iters_(iters),
        edge_threshold_(DEFAULT_EDGE_THRESHOLD), max_disc_threshold_(DEFAULT_MAX_DISC_THRESHOLD),
        sigma_range_(DEFAULT_SIGMA_RANGE)
    {
        calc_color_weighted_table(table_color_, sigma_range_, 255);
        calc_space_weighted_filter(table_space_, radius_ * 2 + 1, radius_ + 1.0f);
    }

    void DispBilateralFilterImpl::setRadius(int radius)
    {
        radius_ = radius;
        calc_space_weighted_filter(table_space_, radius_ * 2 + 1, radius_ + 1.0f);
    }

    void DispBilateralFilterImpl::setSigmaRange(double sigma_range)
    {
        sigma_range_ = (float) sigma_range;
        calc_color_weighted_table(table_color_, sigma_range_, 255);
    }

    template <typename T>
    void disp_bilateral_filter_operator(int ndisp, int radius, int iters, float edge_threshold, float max_disc_threshold,
                                        GpuMat& table_color, GpuMat& table_space,
                                        const GpuMat& disp, const GpuMat& img,
                                        OutputArray _dst, Stream& stream)
    {
        using namespace cv::cuda::device::disp_bilateral_filter;

        const short edge_disc = std::max<short>(short(1), short(ndisp * edge_threshold + 0.5));
        const short max_disc = short(ndisp * max_disc_threshold + 0.5);

        disp_load_constants(table_color.ptr<float>(), table_space, ndisp, radius, edge_disc, max_disc);

        _dst.create(disp.size(), disp.type());
        GpuMat dst = _dst.getGpuMat();

        if (dst.data != disp.data)
            disp.copyTo(dst, stream);

        disp_bilateral_filter<T>(dst, img, img.channels(), iters, StreamAccessor::getStream(stream));
    }

    void DispBilateralFilterImpl::apply(InputArray _disp, InputArray _image, OutputArray dst, Stream& stream)
    {
        typedef void (*bilateral_filter_operator_t)(int ndisp, int radius, int iters, float edge_threshold, float max_disc_threshold,
                                                    GpuMat& table_color, GpuMat& table_space,
                                                    const GpuMat& disp, const GpuMat& img, OutputArray dst, Stream& stream);
        const bilateral_filter_operator_t operators[] =
            {disp_bilateral_filter_operator<unsigned char>, 0, 0, disp_bilateral_filter_operator<short>, 0, 0, 0, 0};

        CV_Assert( 0 < ndisp_ && 0 < radius_ && 0 < iters_ );

        GpuMat disp = _disp.getGpuMat();
        GpuMat img = _image.getGpuMat();

        CV_Assert( disp.type() == CV_8U || disp.type() == CV_16S );
        CV_Assert( img.type() == CV_8UC1 || img.type() == CV_8UC3 );
        CV_Assert( disp.size() == img.size() );

        operators[disp.type()](ndisp_, radius_, iters_, edge_threshold_, max_disc_threshold_,
                               table_color_, table_space_, disp, img, dst, stream);
    }
}

Ptr<cuda::DisparityBilateralFilter> cv::cuda::createDisparityBilateralFilter(int ndisp, int radius, int iters)
{
    return makePtr<DispBilateralFilterImpl>(ndisp, radius, iters);
}

#endif /* !defined (HAVE_CUDA) */
