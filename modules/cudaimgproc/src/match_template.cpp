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

#if !defined (HAVE_CUDA) || !defined (HAVE_OPENCV_CUDAARITHM) || defined (CUDA_DISABLER)

Ptr<cuda::TemplateMatching> cv::cuda::createTemplateMatching(int, int, Size) { throw_no_cuda(); return Ptr<cuda::TemplateMatching>(); }

#else

namespace cv { namespace cuda { namespace device
{
    namespace match_template
    {
        void matchTemplateNaive_CCORR_8U(const PtrStepSzb image, const PtrStepSzb templ, PtrStepSzf result, int cn, cudaStream_t stream);
        void matchTemplateNaive_CCORR_32F(const PtrStepSzb image, const PtrStepSzb templ, PtrStepSzf result, int cn, cudaStream_t stream);

        void matchTemplateNaive_SQDIFF_8U(const PtrStepSzb image, const PtrStepSzb templ, PtrStepSzf result, int cn, cudaStream_t stream);
        void matchTemplateNaive_SQDIFF_32F(const PtrStepSzb image, const PtrStepSzb templ, PtrStepSzf result, int cn, cudaStream_t stream);

        void matchTemplatePrepared_SQDIFF_8U(int w, int h, const PtrStepSz<double> image_sqsum, double templ_sqsum, PtrStepSzf result,
            int cn, cudaStream_t stream);

        void matchTemplatePrepared_SQDIFF_NORMED_8U(int w, int h, const PtrStepSz<double> image_sqsum, double templ_sqsum, PtrStepSzf result,
            int cn, cudaStream_t stream);

        void matchTemplatePrepared_CCOFF_8U(int w, int h, const PtrStepSz<int> image_sum, int templ_sum, PtrStepSzf result, cudaStream_t stream);
        void matchTemplatePrepared_CCOFF_8UC2(
            int w, int h,
            const PtrStepSz<int> image_sum_r,
            const PtrStepSz<int> image_sum_g,
            int templ_sum_r,
            int templ_sum_g,
            PtrStepSzf result, cudaStream_t stream);
        void matchTemplatePrepared_CCOFF_8UC3(
                int w, int h,
                const PtrStepSz<int> image_sum_r,
                const PtrStepSz<int> image_sum_g,
                const PtrStepSz<int> image_sum_b,
                int templ_sum_r,
                int templ_sum_g,
                int templ_sum_b,
                PtrStepSzf result, cudaStream_t stream);
        void matchTemplatePrepared_CCOFF_8UC4(
                int w, int h,
                const PtrStepSz<int> image_sum_r,
                const PtrStepSz<int> image_sum_g,
                const PtrStepSz<int> image_sum_b,
                const PtrStepSz<int> image_sum_a,
                int templ_sum_r,
                int templ_sum_g,
                int templ_sum_b,
                int templ_sum_a,
                PtrStepSzf result, cudaStream_t stream);


        void matchTemplatePrepared_CCOFF_NORMED_8U(
                int w, int h, const PtrStepSz<int> image_sum,
                const PtrStepSz<double> image_sqsum,
                int templ_sum, double templ_sqsum,
                PtrStepSzf result, cudaStream_t stream);
        void matchTemplatePrepared_CCOFF_NORMED_8UC2(
                int w, int h,
                const PtrStepSz<int> image_sum_r, const PtrStepSz<double> image_sqsum_r,
                const PtrStepSz<int> image_sum_g, const PtrStepSz<double> image_sqsum_g,
                int templ_sum_r, double templ_sqsum_r,
                int templ_sum_g, double templ_sqsum_g,
                PtrStepSzf result, cudaStream_t stream);
        void matchTemplatePrepared_CCOFF_NORMED_8UC3(
                int w, int h,
                const PtrStepSz<int> image_sum_r, const PtrStepSz<double> image_sqsum_r,
                const PtrStepSz<int> image_sum_g, const PtrStepSz<double> image_sqsum_g,
                const PtrStepSz<int> image_sum_b, const PtrStepSz<double> image_sqsum_b,
                int templ_sum_r, double templ_sqsum_r,
                int templ_sum_g, double templ_sqsum_g,
                int templ_sum_b, double templ_sqsum_b,
                PtrStepSzf result, cudaStream_t stream);
        void matchTemplatePrepared_CCOFF_NORMED_8UC4(
                int w, int h,
                const PtrStepSz<int> image_sum_r, const PtrStepSz<double> image_sqsum_r,
                const PtrStepSz<int> image_sum_g, const PtrStepSz<double> image_sqsum_g,
                const PtrStepSz<int> image_sum_b, const PtrStepSz<double> image_sqsum_b,
                const PtrStepSz<int> image_sum_a, const PtrStepSz<double> image_sqsum_a,
                int templ_sum_r, double templ_sqsum_r,
                int templ_sum_g, double templ_sqsum_g,
                int templ_sum_b, double templ_sqsum_b,
                int templ_sum_a, double templ_sqsum_a,
                PtrStepSzf result, cudaStream_t stream);

        void normalize_8U(int w, int h, const PtrStepSz<double> image_sqsum,
                          double templ_sqsum, PtrStepSzf result, int cn, cudaStream_t stream);

        void extractFirstChannel_32F(const PtrStepSzb image, PtrStepSzf result, int cn, cudaStream_t stream);
    }
}}}

namespace
{
    // Evaluates optimal template's area threshold. If
    // template's area is less  than the threshold, we use naive match
    // template version, otherwise FFT-based (if available)
    int getTemplateThreshold(int method, int depth)
    {
        switch (method)
        {
        case TM_CCORR:
            if (depth == CV_32F) return 250;
            if (depth == CV_8U) return 300;
            break;

        case TM_SQDIFF:
            if (depth == CV_8U) return 300;
            break;
        }

        CV_Error(Error::StsBadArg, "unsupported match template mode");
        return 0;
    }

    ///////////////////////////////////////////////////////////////
    // CCORR_32F

    class Match_CCORR_32F : public TemplateMatching
    {
    public:
        explicit Match_CCORR_32F(Size user_block_size);

        void match(InputArray image, InputArray templ, OutputArray result, Stream& stream = Stream::Null());

    private:
        Ptr<cuda::Convolution> conv_;
        GpuMat result_;
    };

    Match_CCORR_32F::Match_CCORR_32F(Size user_block_size)
    {
        conv_ = cuda::createConvolution(user_block_size);
    }

    void Match_CCORR_32F::match(InputArray _image, InputArray _templ, OutputArray _result, Stream& _stream)
    {
        using namespace cv::cuda::device::match_template;

        GpuMat image = _image.getGpuMat();
        GpuMat templ = _templ.getGpuMat();

        CV_Assert( image.depth() == CV_32F );
        CV_Assert( image.type() == templ.type() );
        CV_Assert( image.cols >= templ.cols && image.rows >= templ.rows );

        cudaStream_t stream = StreamAccessor::getStream(_stream);

        _result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32FC1);
        GpuMat result = _result.getGpuMat();

        if (templ.size().area() < getTemplateThreshold(TM_CCORR, CV_32F))
        {
            matchTemplateNaive_CCORR_32F(image, templ, result, image.channels(), stream);
            return;
        }

        if (image.channels() == 1)
        {
            conv_->convolve(image.reshape(1), templ.reshape(1), result, true, _stream);
        }
        else
        {
            conv_->convolve(image.reshape(1), templ.reshape(1), result_, true, _stream);
            extractFirstChannel_32F(result_, result, image.channels(), stream);
        }
    }

    ///////////////////////////////////////////////////////////////
    // CCORR_8U

    class Match_CCORR_8U : public TemplateMatching
    {
    public:
        explicit Match_CCORR_8U(Size user_block_size) : match32F_(user_block_size)
        {
        }

        void match(InputArray image, InputArray templ, OutputArray result, Stream& stream = Stream::Null());

    private:
        GpuMat imagef_, templf_;
        Match_CCORR_32F match32F_;
    };

    void Match_CCORR_8U::match(InputArray _image, InputArray _templ, OutputArray _result, Stream& stream)
    {
        using namespace cv::cuda::device::match_template;

        GpuMat image = _image.getGpuMat();
        GpuMat templ = _templ.getGpuMat();

        CV_Assert( image.depth() == CV_8U );
        CV_Assert( image.type() == templ.type() );
        CV_Assert( image.cols >= templ.cols && image.rows >= templ.rows );

        if (templ.size().area() < getTemplateThreshold(TM_CCORR, CV_8U))
        {
            _result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32FC1);
            GpuMat result = _result.getGpuMat();

            matchTemplateNaive_CCORR_8U(image, templ, result, image.channels(), StreamAccessor::getStream(stream));
            return;
        }

        image.convertTo(imagef_, CV_32F, stream);
        templ.convertTo(templf_, CV_32F, stream);

        match32F_.match(imagef_, templf_, _result, stream);
    }

    ///////////////////////////////////////////////////////////////
    // CCORR_NORMED_8U

    class Match_CCORR_NORMED_8U : public TemplateMatching
    {
    public:
        explicit Match_CCORR_NORMED_8U(Size user_block_size) : match_CCORR_(user_block_size)
        {
        }

        void match(InputArray image, InputArray templ, OutputArray result, Stream& stream = Stream::Null());

    private:
        Match_CCORR_8U match_CCORR_;
        GpuMat image_sqsums_;
    };

    void Match_CCORR_NORMED_8U::match(InputArray _image, InputArray _templ, OutputArray _result, Stream& stream)
    {
        using namespace cv::cuda::device::match_template;

        GpuMat image = _image.getGpuMat();
        GpuMat templ = _templ.getGpuMat();

        CV_Assert( image.depth() == CV_8U );
        CV_Assert( image.type() == templ.type() );
        CV_Assert( image.cols >= templ.cols && image.rows >= templ.rows );

        match_CCORR_.match(image, templ, _result, stream);
        GpuMat result = _result.getGpuMat();

        cuda::sqrIntegral(image.reshape(1), image_sqsums_, stream);

        double templ_sqsum = cuda::sqrSum(templ.reshape(1))[0];

        normalize_8U(templ.cols, templ.rows, image_sqsums_, templ_sqsum, result, image.channels(), StreamAccessor::getStream(stream));
    }

    ///////////////////////////////////////////////////////////////
    // SQDIFF_32F

    class Match_SQDIFF_32F : public TemplateMatching
    {
    public:
        void match(InputArray image, InputArray templ, OutputArray result, Stream& stream = Stream::Null());
    };

    void Match_SQDIFF_32F::match(InputArray _image, InputArray _templ, OutputArray _result, Stream& stream)
    {
        using namespace cv::cuda::device::match_template;

        GpuMat image = _image.getGpuMat();
        GpuMat templ = _templ.getGpuMat();

        CV_Assert( image.depth() == CV_32F );
        CV_Assert( image.type() == templ.type() );
        CV_Assert( image.cols >= templ.cols && image.rows >= templ.rows );

        _result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32FC1);
        GpuMat result = _result.getGpuMat();

        matchTemplateNaive_SQDIFF_32F(image, templ, result, image.channels(), StreamAccessor::getStream(stream));
    }

    ///////////////////////////////////////////////////////////////
    // SQDIFF_8U

    class Match_SQDIFF_8U : public TemplateMatching
    {
    public:
        explicit Match_SQDIFF_8U(Size user_block_size) : match_CCORR_(user_block_size)
        {
        }

        void match(InputArray image, InputArray templ, OutputArray result, Stream& stream = Stream::Null());

    private:
        GpuMat image_sqsums_;
        Match_CCORR_8U match_CCORR_;
    };

    void Match_SQDIFF_8U::match(InputArray _image, InputArray _templ, OutputArray _result, Stream& stream)
    {
        using namespace cv::cuda::device::match_template;

        GpuMat image = _image.getGpuMat();
        GpuMat templ = _templ.getGpuMat();

        CV_Assert( image.depth() == CV_8U );
        CV_Assert( image.type() == templ.type() );
        CV_Assert( image.cols >= templ.cols && image.rows >= templ.rows );

        if (templ.size().area() < getTemplateThreshold(TM_SQDIFF, CV_8U))
        {
            _result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32FC1);
            GpuMat result = _result.getGpuMat();

            matchTemplateNaive_SQDIFF_8U(image, templ, result, image.channels(), StreamAccessor::getStream(stream));
            return;
        }

        cuda::sqrIntegral(image.reshape(1), image_sqsums_, stream);

        double templ_sqsum = cuda::sqrSum(templ.reshape(1))[0];

        match_CCORR_.match(image, templ, _result, stream);
        GpuMat result = _result.getGpuMat();

        matchTemplatePrepared_SQDIFF_8U(templ.cols, templ.rows, image_sqsums_, templ_sqsum, result, image.channels(), StreamAccessor::getStream(stream));
    }

    ///////////////////////////////////////////////////////////////
    // SQDIFF_NORMED_8U

    class Match_SQDIFF_NORMED_8U : public TemplateMatching
    {
    public:
        explicit Match_SQDIFF_NORMED_8U(Size user_block_size) : match_CCORR_(user_block_size)
        {
        }

        void match(InputArray image, InputArray templ, OutputArray result, Stream& stream = Stream::Null());

    private:
        GpuMat image_sqsums_;
        Match_CCORR_8U match_CCORR_;
    };

    void Match_SQDIFF_NORMED_8U::match(InputArray _image, InputArray _templ, OutputArray _result, Stream& stream)
    {
        using namespace cv::cuda::device::match_template;

        GpuMat image = _image.getGpuMat();
        GpuMat templ = _templ.getGpuMat();

        CV_Assert( image.depth() == CV_8U );
        CV_Assert( image.type() == templ.type() );
        CV_Assert( image.cols >= templ.cols && image.rows >= templ.rows );

        cuda::sqrIntegral(image.reshape(1), image_sqsums_, stream);

        double templ_sqsum = cuda::sqrSum(templ.reshape(1))[0];

        match_CCORR_.match(image, templ, _result, stream);
        GpuMat result = _result.getGpuMat();

        matchTemplatePrepared_SQDIFF_NORMED_8U(templ.cols, templ.rows, image_sqsums_, templ_sqsum, result, image.channels(), StreamAccessor::getStream(stream));
    }

    ///////////////////////////////////////////////////////////////
    // CCOFF_8U

    class Match_CCOEFF_8U : public TemplateMatching
    {
    public:
        explicit Match_CCOEFF_8U(Size user_block_size) : match_CCORR_(user_block_size)
        {
        }

        void match(InputArray image, InputArray templ, OutputArray result, Stream& stream = Stream::Null());

    private:
        std::vector<GpuMat> images_;
        std::vector<GpuMat> image_sums_;
        Match_CCORR_8U match_CCORR_;
    };

    void Match_CCOEFF_8U::match(InputArray _image, InputArray _templ, OutputArray _result, Stream& stream)
    {
        using namespace cv::cuda::device::match_template;

        GpuMat image = _image.getGpuMat();
        GpuMat templ = _templ.getGpuMat();

        CV_Assert( image.depth() == CV_8U );
        CV_Assert( image.type() == templ.type() );
        CV_Assert( image.cols >= templ.cols && image.rows >= templ.rows );

        match_CCORR_.match(image, templ, _result, stream);
        GpuMat result = _result.getGpuMat();

        if (image.channels() == 1)
        {
            image_sums_.resize(1);
            cuda::integral(image, image_sums_[0], stream);

            int templ_sum = (int) cuda::sum(templ)[0];

            matchTemplatePrepared_CCOFF_8U(templ.cols, templ.rows, image_sums_[0], templ_sum, result, StreamAccessor::getStream(stream));
        }
        else
        {
            cuda::split(image, images_);

            image_sums_.resize(images_.size());
            for (int i = 0; i < image.channels(); ++i)
                cuda::integral(images_[i], image_sums_[i], stream);

            Scalar templ_sum = cuda::sum(templ);

            switch (image.channels())
            {
            case 2:
                matchTemplatePrepared_CCOFF_8UC2(
                        templ.cols, templ.rows, image_sums_[0], image_sums_[1],
                        (int) templ_sum[0], (int) templ_sum[1],
                        result, StreamAccessor::getStream(stream));
                break;
            case 3:
                matchTemplatePrepared_CCOFF_8UC3(
                        templ.cols, templ.rows, image_sums_[0], image_sums_[1], image_sums_[2],
                        (int) templ_sum[0], (int) templ_sum[1], (int) templ_sum[2],
                        result, StreamAccessor::getStream(stream));
                break;
            case 4:
                matchTemplatePrepared_CCOFF_8UC4(
                        templ.cols, templ.rows, image_sums_[0], image_sums_[1], image_sums_[2], image_sums_[3],
                        (int) templ_sum[0], (int) templ_sum[1], (int) templ_sum[2], (int) templ_sum[3],
                        result, StreamAccessor::getStream(stream));
                break;
            default:
                CV_Error(Error::StsBadArg, "unsupported number of channels");
            }
        }
    }

    ///////////////////////////////////////////////////////////////
    // CCOFF_NORMED_8U

    class Match_CCOEFF_NORMED_8U : public TemplateMatching
    {
    public:
        explicit Match_CCOEFF_NORMED_8U(Size user_block_size) : match_CCORR_32F_(user_block_size)
        {
        }

        void match(InputArray image, InputArray templ, OutputArray result, Stream& stream = Stream::Null());

    private:
        GpuMat imagef_, templf_;
        Match_CCORR_32F match_CCORR_32F_;
        std::vector<GpuMat> images_;
        std::vector<GpuMat> image_sums_;
        std::vector<GpuMat> image_sqsums_;
    };

    void Match_CCOEFF_NORMED_8U::match(InputArray _image, InputArray _templ, OutputArray _result, Stream& stream)
    {
        using namespace cv::cuda::device::match_template;

        GpuMat image = _image.getGpuMat();
        GpuMat templ = _templ.getGpuMat();

        CV_Assert( image.depth() == CV_8U );
        CV_Assert( image.type() == templ.type() );
        CV_Assert( image.cols >= templ.cols && image.rows >= templ.rows );

        image.convertTo(imagef_, CV_32F, stream);
        templ.convertTo(templf_, CV_32F, stream);

        match_CCORR_32F_.match(imagef_, templf_, _result, stream);
        GpuMat result = _result.getGpuMat();

        if (image.channels() == 1)
        {
            image_sums_.resize(1);
            cuda::integral(image, image_sums_[0], stream);

            image_sqsums_.resize(1);
            cuda::sqrIntegral(image, image_sqsums_[0], stream);

            int templ_sum = (int) cuda::sum(templ)[0];
            double templ_sqsum = cuda::sqrSum(templ)[0];

            matchTemplatePrepared_CCOFF_NORMED_8U(
                    templ.cols, templ.rows, image_sums_[0], image_sqsums_[0],
                    templ_sum, templ_sqsum, result, StreamAccessor::getStream(stream));
        }
        else
        {
            cuda::split(image, images_);

            image_sums_.resize(images_.size());
            image_sqsums_.resize(images_.size());
            for (int i = 0; i < image.channels(); ++i)
            {
                cuda::integral(images_[i], image_sums_[i], stream);
                cuda::sqrIntegral(images_[i], image_sqsums_[i], stream);
            }

            Scalar templ_sum = cuda::sum(templ);
            Scalar templ_sqsum = cuda::sqrSum(templ);

            switch (image.channels())
            {
            case 2:
                matchTemplatePrepared_CCOFF_NORMED_8UC2(
                        templ.cols, templ.rows,
                        image_sums_[0], image_sqsums_[0],
                        image_sums_[1], image_sqsums_[1],
                        (int)templ_sum[0], templ_sqsum[0],
                        (int)templ_sum[1], templ_sqsum[1],
                        result, StreamAccessor::getStream(stream));
                break;
            case 3:
                matchTemplatePrepared_CCOFF_NORMED_8UC3(
                        templ.cols, templ.rows,
                        image_sums_[0], image_sqsums_[0],
                        image_sums_[1], image_sqsums_[1],
                        image_sums_[2], image_sqsums_[2],
                        (int)templ_sum[0], templ_sqsum[0],
                        (int)templ_sum[1], templ_sqsum[1],
                        (int)templ_sum[2], templ_sqsum[2],
                        result, StreamAccessor::getStream(stream));
                break;
            case 4:
                matchTemplatePrepared_CCOFF_NORMED_8UC4(
                        templ.cols, templ.rows,
                        image_sums_[0], image_sqsums_[0],
                        image_sums_[1], image_sqsums_[1],
                        image_sums_[2], image_sqsums_[2],
                        image_sums_[3], image_sqsums_[3],
                        (int)templ_sum[0], templ_sqsum[0],
                        (int)templ_sum[1], templ_sqsum[1],
                        (int)templ_sum[2], templ_sqsum[2],
                        (int)templ_sum[3], templ_sqsum[3],
                        result, StreamAccessor::getStream(stream));
                break;
            default:
                CV_Error(Error::StsBadArg, "unsupported number of channels");
            }
        }
    }
}

Ptr<cuda::TemplateMatching> cv::cuda::createTemplateMatching(int srcType, int method, Size user_block_size)
{
    const int sdepth = CV_MAT_DEPTH(srcType);

    CV_Assert( sdepth == CV_8U || sdepth == CV_32F );

    if (sdepth == CV_32F)
    {
        switch (method)
        {
        case TM_SQDIFF:
            return makePtr<Match_SQDIFF_32F>();

        case TM_CCORR:
            return makePtr<Match_CCORR_32F>(user_block_size);

        default:
            CV_Error( Error::StsBadFlag, "Unsopported method" );
            return Ptr<cuda::TemplateMatching>();
        }
    }
    else
    {
        switch (method)
        {
        case TM_SQDIFF:
            return makePtr<Match_SQDIFF_8U>(user_block_size);

        case TM_SQDIFF_NORMED:
            return makePtr<Match_SQDIFF_NORMED_8U>(user_block_size);

        case TM_CCORR:
            return makePtr<Match_CCORR_8U>(user_block_size);

        case TM_CCORR_NORMED:
            return makePtr<Match_CCORR_NORMED_8U>(user_block_size);

        case TM_CCOEFF:
            return makePtr<Match_CCOEFF_8U>(user_block_size);

        case TM_CCOEFF_NORMED:
            return makePtr<Match_CCOEFF_NORMED_8U>(user_block_size);

        default:
            CV_Error( Error::StsBadFlag, "Unsopported method" );
            return Ptr<cuda::TemplateMatching>();
        }
    }
}

#endif
