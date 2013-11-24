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
using namespace cvtest;

namespace cvtest {
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

cv::ocl::oclMat createMat_ocl(cv::RNG& rng, Size size, int type, bool useRoi)
{
    Size size0 = size;

    if (useRoi)
    {
        size0.width += rng.uniform(5, 15);
        size0.height += rng.uniform(5, 15);
    }

    cv::ocl::oclMat d_m(size0, type);

    if (size0 != size)
        d_m = d_m(Rect((size0.width - size.width) / 2, (size0.height - size.height) / 2, size.width, size.height));

    return d_m;
}

cv::ocl::oclMat loadMat_ocl(cv::RNG& rng, const Mat& m, bool useRoi)
{
    CV_Assert(m.type() == CV_8UC1 || m.type() == CV_8UC3);
    cv::ocl::oclMat d_m;
    d_m = createMat_ocl(rng, m.size(), m.type(), useRoi);

    Size ls;
    Point pt;

    d_m.locateROI(ls, pt);

    Rect roi(pt.x, pt.y, d_m.size().width, d_m.size().height);

    cv::ocl::oclMat m_ocl(m);

    cv::ocl::oclMat d_m_roi(d_m, roi);

    m_ocl.copyTo(d_m);
    return d_m;
}

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

void showDiff(const Mat& gold, const Mat& actual, double eps, bool alwaysShow)
{
    Mat diff, diff_thresh;
    absdiff(gold, actual, diff);
    diff.convertTo(diff, CV_32F);
    threshold(diff, diff_thresh, eps, 255.0, cv::THRESH_BINARY);

    if (alwaysShow || cv::countNonZero(diff_thresh.reshape(1)) > 0)
    {
        namedWindow("gold", WINDOW_NORMAL);
        namedWindow("actual", WINDOW_NORMAL);
        namedWindow("diff", WINDOW_NORMAL);

        imshow("gold", gold);
        imshow("actual", actual);
        imshow("diff", diff);

        waitKey();
    }
}

} // namespace cvtest
