#ifndef __OPENCV_TEST_PRECOMP_HPP__
#define __OPENCV_TEST_PRECOMP_HPP__

#include "opencv2/ts/ts.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui/highgui.hpp"

static inline bool check_and_treat_gpu_exception(const cv::Exception& e, cvtest::TS* ts)
{
    switch (e.code)
    {
    case CV_GpuNotSupported: 
        ts->printf(cvtest::TS::LOG, "\nGpu not supported by the library"); 
        break;

    case CV_GpuApiCallError: 
        ts->printf(cvtest::TS::LOG, "\nGPU Error: %s", e.what());
        break;

    case CV_GpuNppCallError: 
        ts->printf(cvtest::TS::LOG, "\nNPP Error: %s", e.what());
        break;

    default:
        return false;
    }
    ts->set_failed_test_info(cvtest::TS::FAIL_GENERIC);                        
    return true;
}

#endif
