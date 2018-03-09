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
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
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
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

#define DUMP_CONFIG_PROPERTY(propertyName, propertyValue) \
    do { \
        std::stringstream ssName, ssValue;\
        ssName << propertyName;\
        ssValue << (propertyValue); \
        ::testing::Test::RecordProperty(ssName.str(), ssValue.str()); \
    } while (false)

#define DUMP_MESSAGE_STDOUT(msg) \
    do { \
        std::cout << msg << std::endl; \
    } while (false)

#include <opencv2/core/opencl/opencl_info.hpp>

#endif // HAVE_OPENCL

namespace cvtest {
namespace ocl {

using namespace cv;

int test_loop_times = 1; // TODO Read from command line / environment

#ifdef HAVE_OPENCL
void dumpOpenCLDevice()
{
    cv::dumpOpenCLInformation();
}
#endif // HAVE_OPENCL

Mat TestUtils::readImage(const String &fileName, int flags)
{
    return cv::imread(cvtest::TS::ptr()->get_data_path() + fileName, flags);
}

Mat TestUtils::readImageType(const String &fname, int type)
{
    Mat src = readImage(fname, CV_MAT_CN(type) == 1 ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    if (CV_MAT_CN(type) == 4)
    {
        Mat temp;
        cv::cvtColor(src, temp, cv::COLOR_BGR2BGRA);
        swap(src, temp);
    }
    src.convertTo(src, CV_MAT_DEPTH(type));
    return src;
}

double TestUtils::checkNorm1(InputArray m, InputArray mask)
{
    return cvtest::norm(m.getMat(), NORM_INF, mask.getMat());
}

double TestUtils::checkNorm2(InputArray m1, InputArray m2, InputArray mask)
{
    return cvtest::norm(m1.getMat(), m2.getMat(), NORM_INF, mask.getMat());
}

double TestUtils::checkSimilarity(InputArray m1, InputArray m2)
{
    Mat diff;
    matchTemplate(m1.getMat(), m2.getMat(), diff, CV_TM_CCORR_NORMED);
    return std::abs(diff.at<float>(0, 0) - 1.f);
}

double TestUtils::checkRectSimilarity(const Size & sz, std::vector<Rect>& ob1, std::vector<Rect>& ob2)
{
    double final_test_result = 0.0;
    size_t sz1 = ob1.size();
    size_t sz2 = ob2.size();

    if (sz1 != sz2)
        return sz1 > sz2 ? (double)(sz1 - sz2) : (double)(sz2 - sz1);
    else
    {
        if (sz1 == 0 && sz2 == 0)
            return 0;
        cv::Mat cpu_result(sz, CV_8UC1);
        cpu_result.setTo(0);

        for (vector<Rect>::const_iterator r = ob1.begin(); r != ob1.end(); ++r)
        {
            cv::Mat cpu_result_roi(cpu_result, *r);
            cpu_result_roi.setTo(1);
            cpu_result.copyTo(cpu_result);
        }
        int cpu_area = cv::countNonZero(cpu_result > 0);

        cv::Mat gpu_result(sz, CV_8UC1);
        gpu_result.setTo(0);
        for(vector<Rect>::const_iterator r2 = ob2.begin(); r2 != ob2.end(); ++r2)
        {
            cv::Mat gpu_result_roi(gpu_result, *r2);
            gpu_result_roi.setTo(1);
            gpu_result.copyTo(gpu_result);
        }

        cv::Mat result_;
        multiply(cpu_result, gpu_result, result_);
        int result = cv::countNonZero(result_ > 0);
        if (cpu_area!=0 && result!=0)
            final_test_result = 1.0 - (double)result/(double)cpu_area;
        else if(cpu_area==0 && result!=0)
            final_test_result = -1;
    }
    return final_test_result;
}

void TestUtils::showDiff(InputArray _src, InputArray _gold, InputArray _actual, double eps, bool alwaysShow)
{
    Mat src = _src.getMat(), actual = _actual.getMat(), gold = _gold.getMat();

    Mat diff, diff_thresh;
    absdiff(gold, actual, diff);
    diff.convertTo(diff, CV_32F);
    threshold(diff, diff_thresh, eps, 255.0, cv::THRESH_BINARY);

    if (alwaysShow || cv::countNonZero(diff_thresh.reshape(1)) > 0)
    {
#if 0
        std::cout << "Source: " << std::endl << src << std::endl;
        std::cout << "Expected: " << std::endl << gold << std::endl;
        std::cout << "Actual: " << std::endl << actual << std::endl;
#endif

        namedWindow("src", WINDOW_NORMAL);
        namedWindow("gold", WINDOW_NORMAL);
        namedWindow("actual", WINDOW_NORMAL);
        namedWindow("diff", WINDOW_NORMAL);

        imshow("src", src);
        imshow("gold", gold);
        imshow("actual", actual);
        imshow("diff", diff);

        cv::waitKey();
    }
}

} } // namespace cvtest::ocl
