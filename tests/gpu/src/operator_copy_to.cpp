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
#include "highgui.h"
#include "cv.h"

#include <string>
#include <iostream>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <iomanip> // for  cout << setw()

using namespace cv;
using namespace std;
using namespace gpu;

class CV_GpuMatOpCopyToTest : public CvTest
{
    public:
        CV_GpuMatOpCopyToTest();
        ~CV_GpuMatOpCopyToTest();

    protected:
        void run(int);
        template <typename T>
        void print_mat(const T & mat, const std::string & name) const;
        bool compare_matrix(cv::Mat & cpumat, gpu::GpuMat & gpumat);

    private:
        int rows;
        int cols;
};

CV_GpuMatOpCopyToTest::CV_GpuMatOpCopyToTest(): CvTest( "GPU-MatOperatorCopyTo", "copyTo" )
{
    rows = 234;
    cols = 123;

    //#define PRINT_MATRIX
}

CV_GpuMatOpCopyToTest::~CV_GpuMatOpCopyToTest() {}

template<typename T>
void CV_GpuMatOpCopyToTest::print_mat(const T & mat, const std::string & name) const
{
    cv::imshow(name, mat);
}

bool CV_GpuMatOpCopyToTest::compare_matrix(cv::Mat & cpumat, gpu::GpuMat & gpumat)
{
    Mat cmat(cpumat.size(), cpumat.type(), Scalar::all(0));
    GpuMat gmat(cmat);

    Mat cpumask(cpumat.size(), CV_8U);

    cv::RNG rng(*ts->get_rng());

    rng.fill(cpumask, RNG::NORMAL, Scalar::all(0), Scalar::all(127));

    threshold(cpumask, cpumask, 0, 127, THRESH_BINARY);

    GpuMat gpumask(cpumask);

    //int64 time = getTickCount();
    cpumat.copyTo(cmat, cpumask);
    //int64 time1 = getTickCount();
    gpumat.copyTo(gmat, gpumask);
    //int64 time2 = getTickCount();

    //std::cout << "\ntime cpu: " << std::fixed << std::setprecision(12) << 1.0 / double((time1 - time)  / (double)getTickFrequency());
    //std::cout << "\ntime gpu: " << std::fixed << std::setprecision(12) << 1.0 / double((time2 - time1) / (double)getTickFrequency());
    //std::cout << "\n";

#ifdef PRINT_MATRIX
    print_mat(cmat, "cpu mat");
    print_mat(gmat, "gpu mat");
    print_mat(cpumask, "cpu mask");
    print_mat(gpumask, "gpu mask");
    cv::waitKey(0);
#endif

    double ret = norm(cmat, gmat);

    if (ret < 1.0)
        return true;
    else
    {
        ts->printf(CvTS::CONSOLE, "\nNorm: %f\n", ret);
        return false;
    }
}

void CV_GpuMatOpCopyToTest::run( int /* start_from */)
{
    bool is_test_good = true;

    try
    {
        for (int i = 0 ; i < 7; i++)
        {
            Mat cpumat(rows, cols, i);
            cpumat.setTo(Scalar::all(127));

            GpuMat gpumat(cpumat);

            is_test_good &= compare_matrix(cpumat, gpumat);
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

CV_GpuMatOpCopyToTest CV_GpuMatOpCopyTo_test;
