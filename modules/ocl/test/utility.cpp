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

#include "test_precomp.hpp"
#define VARNAME(A) #A
using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace cvtest;


//std::string generateVarList(int first,...)
//{
//	vector<std::string> varname;
//
//	va_list argp;
//	string s;
//	stringstream ss;
//	va_start(argp,first);
//	int i=first;
//	while(i!=-1)
//	{
//		ss<<i<<",";
//		i=va_arg(argp,int);
//	};
//	s=ss.str();
//	va_end(argp);
//	return s;
//};

//std::string generateVarList(int& p1,int& p2)
//{
//	stringstream ss;
//	ss<<VARNAME(p1)<<":"<<src1x<<","<<VARNAME(p2)<<":"<<src1y;
//	return ss.str();
//};

int randomInt(int minVal, int maxVal)
{
    RNG &rng = TS::ptr()->get_rng();
    return rng.uniform(minVal, maxVal);
}

double randomDouble(double minVal, double maxVal)
{
    RNG &rng = TS::ptr()->get_rng();
    return rng.uniform(minVal, maxVal);
}

Size randomSize(int minVal, int maxVal)
{
    return cv::Size(randomInt(minVal, maxVal), randomInt(minVal, maxVal));
}

Scalar randomScalar(double minVal, double maxVal)
{
    return Scalar(randomDouble(minVal, maxVal), randomDouble(minVal, maxVal), randomDouble(minVal, maxVal), randomDouble(minVal, maxVal));
}

Mat randomMat(Size size, int type, double minVal, double maxVal)
{
    return randomMat(TS::ptr()->get_rng(), size, type, minVal, maxVal, false);
}

/*
void showDiff(InputArray gold_, InputArray actual_, double eps)
{
    Mat gold;
    if (gold_.kind() == _InputArray::MAT)
        gold = gold_.getMat();
    else
        gold_.getGpuMat().download(gold);

    Mat actual;
    if (actual_.kind() == _InputArray::MAT)
        actual = actual_.getMat();
    else
        actual_.getGpuMat().download(actual);

    Mat diff;
    absdiff(gold, actual, diff);
    threshold(diff, diff, eps, 255.0, cv::THRESH_BINARY);

    namedWindow("gold", WINDOW_NORMAL);
    namedWindow("actual", WINDOW_NORMAL);
    namedWindow("diff", WINDOW_NORMAL);

    imshow("gold", gold);
    imshow("actual", actual);
    imshow("diff", diff);

    waitKey();
}
*/



vector<MatType> types(int depth_start, int depth_end, int cn_start, int cn_end)
{
    vector<MatType> v;

    v.reserve((depth_end - depth_start + 1) * (cn_end - cn_start + 1));

    for (int depth = depth_start; depth <= depth_end; ++depth)
    {
        for (int cn = cn_start; cn <= cn_end; ++cn)
        {
            v.push_back(CV_MAKETYPE(depth, cn));
        }
    }

    return v;
}

const vector<MatType> &all_types()
{
    static vector<MatType> v = types(CV_8U, CV_64F, 1, 4);

    return v;
}

Mat readImage(const string &fileName, int flags)
{
    return imread(string(cvtest::TS::ptr()->get_data_path()) + fileName, flags);
}

Mat readImageType(const string &fname, int type)
{
    Mat src = readImage(fname, CV_MAT_CN(type) == 1 ? IMREAD_GRAYSCALE : IMREAD_COLOR);
    if (CV_MAT_CN(type) == 4)
    {
        Mat temp;
        cvtColor(src, temp, cv::COLOR_BGR2BGRA);
        swap(src, temp);
    }
    src.convertTo(src, CV_MAT_DEPTH(type));
    return src;
}

double checkNorm(const Mat &m)
{
    return norm(m, NORM_INF);
}

double checkNorm(const Mat &m1, const Mat &m2)
{
    return norm(m1, m2, NORM_INF);
}

double checkSimilarity(const Mat &m1, const Mat &m2)
{
    Mat diff;
    matchTemplate(m1, m2, diff, TM_CCORR_NORMED);
    return std::abs(diff.at<float>(0, 0) - 1.f);
}

/*
void cv::ocl::PrintTo(const DeviceInfo& info, ostream* os)
{
    (*os) << info.name();
}
*/

void PrintTo(const Inverse &inverse, std::ostream *os)
{
    if (inverse)
        (*os) << "inverse";
    else
        (*os) << "direct";
}

double checkRectSimilarity(Size sz, std::vector<Rect>& ob1, std::vector<Rect>& ob2)
{
    double final_test_result = 0.0;
    size_t sz1 = ob1.size();
    size_t sz2 = ob2.size();

    if(sz1 != sz2)
    {
        return sz1 > sz2 ? (double)(sz1 - sz2) : (double)(sz2 - sz1);
    }
    else
    {
        if(sz1==0 && sz2==0)
            return 0;
        cv::Mat cpu_result(sz, CV_8UC1);
        cpu_result.setTo(0);

        for(vector<Rect>::const_iterator r = ob1.begin(); r != ob1.end(); r++)
        {      
            cv::Mat cpu_result_roi(cpu_result, *r);
            cpu_result_roi.setTo(1);
            cpu_result.copyTo(cpu_result);
        }
        int cpu_area = cv::countNonZero(cpu_result > 0);

        cv::Mat gpu_result(sz, CV_8UC1);
        gpu_result.setTo(0);
        for(vector<Rect>::const_iterator r2 = ob2.begin(); r2 != ob2.end(); r2++)
        {
            cv::Mat gpu_result_roi(gpu_result, *r2);
            gpu_result_roi.setTo(1);
            gpu_result.copyTo(gpu_result);
        }

        cv::Mat result_;
        multiply(cpu_result, gpu_result, result_);
        int result = cv::countNonZero(result_ > 0);
        if(cpu_area!=0 && result!=0)
            final_test_result = 1.0 - (double)result/(double)cpu_area;
        else if(cpu_area==0 && result!=0)
            final_test_result = -1;
    }
    return final_test_result;
}

