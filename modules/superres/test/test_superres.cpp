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

#include "test_precomp.hpp"

class AllignedFrameSource : public cv::superres::FrameSource
{
public:
    AllignedFrameSource(const cv::Ptr<cv::superres::FrameSource>& base, int scale);

    void nextFrame(cv::OutputArray frame);
    void reset();

private:
    cv::Ptr<cv::superres::FrameSource> base_;
    cv::Mat origFrame_;
    int scale_;
};

AllignedFrameSource::AllignedFrameSource(const cv::Ptr<cv::superres::FrameSource>& base, int scale) :
    base_(base), scale_(scale)
{
    CV_Assert( !base_.empty() );
}

void AllignedFrameSource::nextFrame(cv::OutputArray frame)
{
    base_->nextFrame(origFrame_);

    if (origFrame_.rows % scale_ == 0 && origFrame_.cols % scale_ == 0)
    {
        cv::superres::arrCopy(origFrame_, frame);
    }
    else
    {
        cv::Rect ROI(0, 0, (origFrame_.cols / scale_) * scale_, (origFrame_.rows / scale_) * scale_);
        cv::superres::arrCopy(origFrame_(ROI), frame);
    }
}

void AllignedFrameSource::reset()
{
    base_->reset();
}

class DegradeFrameSource : public cv::superres::FrameSource
{
public:
    DegradeFrameSource(const cv::Ptr<cv::superres::FrameSource>& base, int scale);

    void nextFrame(cv::OutputArray frame);
    void reset();

private:
    cv::Ptr<cv::superres::FrameSource> base_;
    cv::Mat origFrame_;
    cv::Mat blurred_;
    cv::Mat deg_;
    double iscale_;
};

DegradeFrameSource::DegradeFrameSource(const cv::Ptr<cv::superres::FrameSource>& base, int scale) :
    base_(base), iscale_(1.0 / scale)
{
    CV_Assert( !base_.empty() );
}

void addGaussNoise(cv::Mat& image, double sigma)
{
    cv::Mat noise(image.size(), CV_32FC(image.channels()));
    cvtest::TS::ptr()->get_rng().fill(noise, cv::RNG::NORMAL, 0.0, sigma);

    cv::addWeighted(image, 1.0, noise, 1.0, 0.0, image, image.depth());
}

void addSpikeNoise(cv::Mat& image, int frequency)
{
    cv::Mat_<uchar> mask(image.size(), 0);

    for (int y = 0; y < mask.rows; ++y)
    {
        for (int x = 0; x < mask.cols; ++x)
        {
            if (cvtest::TS::ptr()->get_rng().uniform(0, frequency) < 1)
                mask(y, x) = 255;
        }
    }

    image.setTo(cv::Scalar::all(255), mask);
}

void DegradeFrameSource::nextFrame(cv::OutputArray frame)
{
    base_->nextFrame(origFrame_);

    cv::GaussianBlur(origFrame_, blurred_, cv::Size(5, 5), 0);
    cv::resize(blurred_, deg_, cv::Size(), iscale_, iscale_, cv::INTER_NEAREST);

    addGaussNoise(deg_, 10.0);
    addSpikeNoise(deg_, 500);

    cv::superres::arrCopy(deg_, frame);
}

void DegradeFrameSource::reset()
{
    base_->reset();
}

double MSSIM(const cv::Mat& i1, const cv::Mat& i2)
{
    const double C1 = 6.5025;
    const double C2 = 58.5225;

    const int depth = CV_32F;

    cv::Mat I1, I2;
    i1.convertTo(I1, depth);
    i2.convertTo(I2, depth);

    cv::Mat I2_2  = I2.mul(I2); // I2^2
    cv::Mat I1_2  = I1.mul(I1); // I1^2
    cv::Mat I1_I2 = I1.mul(I2); // I1 * I2

    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2   = mu1.mul(mu1);
    cv::Mat mu2_2   = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1, t2;
    cv::Mat numerator;
    cv::Mat denominator;

    // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    numerator = t1.mul(t2);

    // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    denominator = t1.mul(t2);

    // ssim_map =  numerator./denominator;
    cv::Mat ssim_map;
    cv::divide(numerator, denominator, ssim_map);

    // mssim = average of ssim map
    cv::Scalar mssim = cv::mean(ssim_map);

    if (i1.channels() == 1)
        return mssim[0];

    return (mssim[0] + mssim[1] + mssim[3]) / 3;
}

class SuperResolution : public testing::Test
{
public:
    void RunTest(cv::Ptr<cv::superres::SuperResolution> superRes);
};

void SuperResolution::RunTest(cv::Ptr<cv::superres::SuperResolution> superRes)
{
    const std::string inputVideoName = cvtest::TS::ptr()->get_data_path() + "car.avi";
    const int scale = 2;
    const int iterations = 100;
    const int temporalAreaRadius = 2;

    ASSERT_FALSE( superRes.empty() );

    const int btvKernelSize = superRes->getInt("btvKernelSize");

    superRes->set("scale", scale);
    superRes->set("iterations", iterations);
    superRes->set("temporalAreaRadius", temporalAreaRadius);

    cv::Ptr<cv::superres::FrameSource> goldSource(new AllignedFrameSource(cv::superres::createFrameSource_Video(inputVideoName), scale));
    cv::Ptr<cv::superres::FrameSource> lowResSource(new DegradeFrameSource(new AllignedFrameSource(cv::superres::createFrameSource_Video(inputVideoName), scale), scale));

    // skip first frame
    cv::Mat frame;

    lowResSource->nextFrame(frame);
    goldSource->nextFrame(frame);

    cv::Rect inner(btvKernelSize, btvKernelSize, frame.cols - 2 * btvKernelSize, frame.rows - 2 * btvKernelSize);

    superRes->setInput(lowResSource);

    double srAvgMSSIM = 0.0;
    const int count = 10;

    cv::Mat goldFrame, superResFrame;
    for (int i = 0; i < count; ++i)
    {
        goldSource->nextFrame(goldFrame);
        ASSERT_FALSE( goldFrame.empty() );

        superRes->nextFrame(superResFrame);
        ASSERT_FALSE( superResFrame.empty() );

        const double srMSSIM = MSSIM(goldFrame(inner), superResFrame);

        srAvgMSSIM += srMSSIM;
    }

    srAvgMSSIM /= count;

    EXPECT_GE( srAvgMSSIM, 0.5 );
}

TEST_F(SuperResolution, BTVL1)
{
    RunTest(cv::superres::createSuperResolution_BTVL1());
}

#if defined(HAVE_OPENCV_GPU) && defined(HAVE_CUDA)

TEST_F(SuperResolution, BTVL1_GPU)
{
    RunTest(cv::superres::createSuperResolution_BTVL1_GPU());
}
#endif
#if defined(HAVE_OPENCV_OCL) && defined(HAVE_OPENCL)
TEST_F(SuperResolution, BTVL1_OCL)
{
    RunTest(cv::superres::createSuperResolution_BTVL1_OCL());
}
#endif
