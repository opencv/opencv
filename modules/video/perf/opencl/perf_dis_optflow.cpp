// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

namespace opencv_test { namespace {

#ifdef HAVE_OPENCL

void MakeArtificialExample(UMat &dst_frame1, UMat &dst_frame2);

typedef tuple<String, Size> DISParams;
typedef TestBaseWithParam<DISParams> DenseOpticalFlow_DIS;

OCL_PERF_TEST_P(DenseOpticalFlow_DIS, perf,
                Combine(Values("PRESET_ULTRAFAST", "PRESET_FAST", "PRESET_MEDIUM"), Values(szVGA, sz720p, sz1080p)))
{
    DISParams params = GetParam();

    // use strings to print preset names in the perf test results:
    String preset_string = get<0>(params);
    int preset = DISOpticalFlow::PRESET_FAST;
    if (preset_string == "PRESET_ULTRAFAST")
        preset = DISOpticalFlow::PRESET_ULTRAFAST;
    else if (preset_string == "PRESET_FAST")
        preset = DISOpticalFlow::PRESET_FAST;
    else if (preset_string == "PRESET_MEDIUM")
        preset = DISOpticalFlow::PRESET_MEDIUM;
    Size sz = get<1>(params);

    UMat frame1(sz, CV_8U);
    UMat frame2(sz, CV_8U);
    UMat flow;

    MakeArtificialExample(frame1, frame2);

    Ptr<DenseOpticalFlow> algo = DISOpticalFlow::create(preset);

    PERF_SAMPLE_BEGIN()
    {
        algo->calc(frame1, frame2, flow);
    }
    PERF_SAMPLE_END()

    SANITY_CHECK_NOTHING();
}

void MakeArtificialExample(UMat &dst_frame1, UMat &dst_frame2)
{
    int src_scale = 2;
    int OF_scale = 6;
    double sigma = dst_frame1.cols / 300;

    UMat tmp(Size(dst_frame1.cols / (1 << src_scale), dst_frame1.rows / (1 << src_scale)), CV_8U);
    randu(tmp, 0, 255);
    resize(tmp, dst_frame1, dst_frame1.size(), 0.0, 0.0, INTER_LINEAR_EXACT);
    resize(tmp, dst_frame2, dst_frame2.size(), 0.0, 0.0, INTER_LINEAR_EXACT);

    Mat displacement_field(Size(dst_frame1.cols / (1 << OF_scale), dst_frame1.rows / (1 << OF_scale)),
                           CV_32FC2);
    randn(displacement_field, 0.0, sigma);
    resize(displacement_field, displacement_field, dst_frame2.size(), 0.0, 0.0, INTER_CUBIC);
    for (int i = 0; i < displacement_field.rows; i++)
        for (int j = 0; j < displacement_field.cols; j++)
            displacement_field.at<Vec2f>(i, j) += Vec2f((float)j, (float)i);

    remap(dst_frame2, dst_frame2, displacement_field, Mat(), INTER_LINEAR, BORDER_REPLICATE);
}

#endif // HAVE_OPENCL

}} // namespace
