/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#include "gputest.hpp"
#include <limits>

using namespace cv;
using namespace std;
using namespace gpu;

class CV_GpuMatOpSetToTest : public CvTest
{
public:
    CV_GpuMatOpSetToTest();
    ~CV_GpuMatOpSetToTest() {}

protected:
    void run(int);

    bool testSetTo(cv::Mat& cpumat, gpu::GpuMat& gpumat, const cv::Mat& cpumask = cv::Mat(), const cv::gpu::GpuMat& gpumask = cv::gpu::GpuMat());

private:
    int rows;
    int cols;
    Scalar s;
};

CV_GpuMatOpSetToTest::CV_GpuMatOpSetToTest(): CvTest( "GPU-MatOperatorSetTo", "setTo" )
{
    rows = 35;
    cols = 67;

    s.val[0] = 127.0;
    s.val[1] = 127.0;
    s.val[2] = 127.0;
    s.val[3] = 127.0;
}

bool CV_GpuMatOpSetToTest::testSetTo(cv::Mat& cpumat, gpu::GpuMat& gpumat, const cv::Mat& cpumask, const cv::gpu::GpuMat& gpumask)
{
    cpumat.setTo(s, cpumask);
    gpumat.setTo(s, gpumask);

    double ret = norm(cpumat, gpumat, NORM_INF);

    if (ret < std::numeric_limits<double>::epsilon())
        return true;
    else
    {
        ts->printf(CvTS::LOG, "\nNorm: %f\n", ret);
        return false;
    }
}

void CV_GpuMatOpSetToTest::run( int /* start_from */)
{
    bool is_test_good = true;

    try
    {
        cv::Mat cpumask(rows, cols, CV_8UC1);
        cv::RNG rng(*ts->get_rng());
        rng.fill(cpumask, RNG::UNIFORM, cv::Scalar::all(0.0), cv::Scalar(1.5));
        cv::gpu::GpuMat gpumask(cpumask);

        int lastType = CV_32F;

        if (TargetArchs::builtWith(NATIVE_DOUBLE) && DeviceInfo().supports(NATIVE_DOUBLE))
            lastType = CV_64F;

        for (int i = 0; i <= lastType; i++)
        {
            for (int cn = 1; cn <= 4; ++cn)
            {
                int mat_type = CV_MAKETYPE(i, cn);
                Mat cpumat(rows, cols, mat_type, Scalar::all(0));
                GpuMat gpumat(cpumat);
                is_test_good &= testSetTo(cpumat, gpumat, cpumask, gpumask);
            }
        }
    }
    catch(const cv::Exception& e)
    {
        if (!check_and_treat_gpu_exception(e, ts))
            throw;
        return;
    }

    if (is_test_good == true)
        ts->set_failed_test_info(CvTS::OK);
    else
        ts->set_failed_test_info(CvTS::FAIL_GENERIC);
}


/////////////////////////////////////////////////////////////////////////////
/////////////////// tests registration  /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

CV_GpuMatOpSetToTest CV_GpuMatOpSetTo_test;
