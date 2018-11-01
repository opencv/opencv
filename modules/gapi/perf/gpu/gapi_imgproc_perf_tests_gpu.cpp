// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../perf_precomp.hpp"
#include "../common/gapi_imgproc_perf_tests.hpp"
#include "opencv2/gapi/gpu/imgproc.hpp"

#define IMGPROC_GPU cv::gapi::imgproc::gpu::kernels()

namespace opencv_test
{

class AbsExactGPU : public Wrappable<AbsExactGPU>
{
public:
    AbsExactGPU() {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const { return cv::countNonZero(in1 != in2) == 0; }
private:
};

class AbsToleranceGPU : public Wrappable<AbsToleranceGPU>
{
public:
    AbsToleranceGPU(double tol) : _tol(tol) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        cv::Mat absDiff; cv::absdiff(in1, in2, absDiff);
        return cv::countNonZero(absDiff > _tol) == 0;
    }
private:
    double _tol;
};

class AbsTolerance32FGPU : public Wrappable<AbsTolerance32FGPU>
{
public:
    AbsTolerance32FGPU(double tol, double tol8u) : _tol(tol), _tol8u(tol8u) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        if (CV_MAT_DEPTH(in1.type()) == CV_32F)
            return ((cv::countNonZero(cv::abs(in1 - in2) > (_tol)*cv::abs(in2))) ? false : true);
        else
            return ((cv::countNonZero(in1 != in2) <= (_tol8u)* in2.total()) ? true : false);
    }
private:
    double _tol;
    double _tol8u;
};

class AbsToleranceSepFilterGPU : public Wrappable<AbsToleranceSepFilterGPU>
{
public:
    AbsToleranceSepFilterGPU(double tol) : _tol(tol) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        return ((cv::countNonZero(cv::abs(in1 - in2) > (_tol)* cv::abs(in2)) <= 0.01 * in2.total()) ? true : false);
    }
private:
    double _tol;
};

class AbsToleranceGaussianBlurGPU : public Wrappable<AbsToleranceGaussianBlurGPU>
{
public:
    AbsToleranceGaussianBlurGPU(double tol, double tol8u) : _tol(tol), _tol8u(tol8u) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        if (CV_MAT_DEPTH(in1.type()) == CV_32F || CV_MAT_DEPTH(in1.type()) == CV_64F)
        {
            return ((cv::countNonZero(cv::abs(in1 - in2) > (_tol)*cv::abs(in2))) ? false : true);
        }
        else
        {
            if (CV_MAT_DEPTH(in1.type()) == CV_8U)
            {
                bool a = (cv::countNonZero(cv::abs(in1 - in2) > 1) <= _tol8u * in2.total());
                return ((a == 1 ? 0 : 1) && ((cv::countNonZero(cv::abs(in1 - in2) > 2) <= 0) == 1 ? 0 : 1)) == 1 ? false : true;
            }
            else return cv::countNonZero(in1 != in2) == 0;
        }
    }
private:
    double _tol;
    double _tol8u;
};

class ToleranceTripleGPU : public Wrappable<ToleranceTripleGPU>
{
public:
    ToleranceTripleGPU(double tol1, double tol2, double tol3) : _tol1(tol1), _tol2(tol2), _tol3(tol3) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        bool a = (cv::countNonZero((in1 - in2) > 0) <= _tol1 * in2.total());
        return (((a == 1 ? 0 : 1) &&
            ((cv::countNonZero((in1 - in2) > 1) <= _tol2 * in2.total()) == 1 ? 0 : 1) &&
            ((cv::countNonZero((in1 - in2) > 2) <= _tol3 * in2.total()) == 1 ? 0 : 1))) == 1 ? false : true;
    }
private:
    double _tol1, _tol2, _tol3;
};



INSTANTIATE_TEST_CASE_P(SepFilterPerfTestGPU_8U, SepFilterPerfTest,
                        Combine(Values(AbsToleranceSepFilterGPU(1e-4f).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3),
                                Values(3),
                                Values(szVGA, sz720p, sz1080p),
                                Values(-1, CV_16S, CV_32F),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(SepFilterPerfTestGPU_other, SepFilterPerfTest,
                        Combine(Values(AbsToleranceSepFilterGPU(1e-4f).to_compare_f()),
                                Values(CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3),
                                Values(szVGA, sz720p, sz1080p),
                                Values(-1, CV_32F),
                                Values(cv::compile_args(IMGPROC_GPU))));



INSTANTIATE_TEST_CASE_P(Filter2DPerfTestGPU, Filter2DPerfTest,
                        Combine(Values(AbsTolerance32FGPU(1e-5, 1e-3).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3, 4, 5, 7),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::BORDER_DEFAULT),
                                Values(-1, CV_32F),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(BoxFilterPerfTestGPU, BoxFilterPerfTest,
                        Combine(Values(AbsTolerance32FGPU(1e-5, 1e-3).to_compare_f()),
                                Values(/*CV_8UC1,*/ CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3,5),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::BORDER_DEFAULT),
                                Values(-1, CV_32F),
                                Values(cv::compile_args(IMGPROC_GPU)))); //TODO: 8UC1 doesn't work

INSTANTIATE_TEST_CASE_P(BlurPerfTestGPU, BlurPerfTest,
                        Combine(Values(AbsTolerance32FGPU(1e-4, 1e-2).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3, 5),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::BORDER_DEFAULT),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(GaussianBlurPerfTestGPU, GaussianBlurPerfTest,
                        Combine(Values(AbsToleranceGaussianBlurGPU(1e-5, 0.05).to_compare_f()), //TODO: too relaxed?
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3, 5),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(MedianBlurPerfTestGPU, MedianBlurPerfTest,
                         Combine(Values(AbsExactGPU().to_compare_f()),
                                 Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                 Values(3, 5),
                                 Values(szVGA, sz720p, sz1080p),
                                 Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(ErodePerfTestGPU, ErodePerfTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3, 5),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::MorphShapes::MORPH_RECT,
                                       cv::MorphShapes::MORPH_CROSS,
                                       cv::MorphShapes::MORPH_ELLIPSE),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(Erode3x3PerfTestGPU, Erode3x3PerfTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(szVGA, sz720p, sz1080p),
                                Values(1,2,4),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(DilatePerfTestGPU, DilatePerfTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3, 5),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::MorphShapes::MORPH_RECT,
                                       cv::MorphShapes::MORPH_CROSS,
                                       cv::MorphShapes::MORPH_ELLIPSE),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(Dilate3x3PerfTestGPU, Dilate3x3PerfTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(szVGA, sz720p, sz1080p),
                                Values(1,2,4),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(SobelPerfTestGPU, SobelPerfTest,
                        Combine(Values(AbsTolerance32FGPU(1e-4, 1e-4).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1/*, CV_32FC1*/), //TODO: CV_32FC1 fails accuracy
                                Values(3, 5),
                                Values(szVGA, sz720p, sz1080p),
                                Values(-1, CV_32F),
                                Values(0, 1),
                                Values(1, 2),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(CannyPerfTestGPU, CannyPerfTest,
                        Combine(Values(AbsTolerance32FGPU(1e-4, 1e-2).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3),
                                Values(szVGA, sz720p, sz1080p),
                                Values(3.0, 120.0),
                                Values(125.0, 240.0),
                                Values(3, 5),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(EqHistPerfTestGPU, EqHistPerfTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                        Values(szVGA, sz720p, sz1080p),
                        Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(RGB2GrayPerfTestGPU, RGB2GrayPerfTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                        Values(szVGA, sz720p, sz1080p),
                        Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(BGR2GrayPerfTestGPU, BGR2GrayPerfTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                        Values(szVGA, sz720p, sz1080p),
                        Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(RGB2YUVPerfTestGPU, RGB2YUVPerfTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                        Values(szVGA, sz720p, sz1080p),
                        Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(YUV2RGBPerfTestGPU, YUV2RGBPerfTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                        Values(szVGA, sz720p, sz1080p),
                        Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(RGB2LabPerfTestGPU, RGB2LabPerfTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                        Values(szVGA, sz720p, sz1080p),
                        Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(BGR2LUVPerfTestGPU, BGR2LUVPerfTest,
                        Combine(Values(ToleranceTripleGPU(0.25 * 3, 0.01 * 3, 0.0001 * 3).to_compare_f()),
                        Values(szVGA, sz720p, sz1080p),
                        Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(LUV2BGRPerfTestGPU, LUV2BGRPerfTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                        Values(szVGA, sz720p, sz1080p),
                        Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(BGR2YUVPerfTestGPU, BGR2YUVPerfTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                        Values(szVGA, sz720p, sz1080p),
                        Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(YUV2BGRPerfTestGPU, YUV2BGRPerfTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                        Values(szVGA, sz720p, sz1080p),
                        Values(cv::compile_args(IMGPROC_GPU))));

}
