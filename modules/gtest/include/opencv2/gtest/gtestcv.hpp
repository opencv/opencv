#ifndef __OPENCV_GTESTCV_HPP__
#define __OPENCV_GTESTCV_HPP__

#include "opencv2/gtest/gtest.h"
#include "opencv2/core/core.hpp"

namespace cvtest
{

using std::vector;
using cv::RNG;
using cv::Mat;
using cv::Scalar;
using cv::Size;
using cv::Point;
using cv::Rect;

enum
{
    TYPE_MASK_8U = 1 << CV_8U,
    TYPE_MASK_8S = 1 << CV_8S,
    TYPE_MASK_16U = 1 << CV_16U,
    TYPE_MASK_16S = 1 << CV_16S,
    TYPE_MASK_32S = 1 << CV_32S,
    TYPE_MASK_32F = 1 << CV_32F,
    TYPE_MASK_64F = 1 << CV_64F,
    TYPE_MASK_ALL = (TYPE_MASK_64F<<1)-1,
    TYPE_MASK_ALL_BUT_8S = TYPE_MASK_ALL & ~TYPE_MASK_8S
};
    
CV_EXPORTS Size randomSize(RNG& rng, double maxSizeLog);
CV_EXPORTS void randomSize(RNG& rng, int minDims, int maxDims, double maxSizeLog, vector<int>& sz);    
CV_EXPORTS int randomType(RNG& rng, int typeMask, int minChannels, int maxChannels);
CV_EXPORTS Mat randomMat(RNG& rng, Size size, int type, bool useRoi);
CV_EXPORTS Mat randomMat(RNG& rng, const vector<int>& size, int type, bool useRoi);
CV_EXPORTS void add(const Mat& a, double alpha, const Mat& b, double beta,
                      Scalar gamma, Mat& c, int ctype, bool calcAbs);
CV_EXPORTS void convert(const Mat& src, Mat& dst, int dtype, double alpha, double beta);
CV_EXPORTS void copy(const Mat& src, Mat& dst, const Mat& mask=Mat());
CV_EXPORTS void set(Mat& dst, const Scalar& gamma, const Mat& mask=Mat());
CV_EXPORTS void minMaxFilter(const Mat& a, Mat& maxresult, const Mat& minresult, const Mat& kernel, Point anchor);
CV_EXPORTS void filter2D(const Mat& src, Mat& dst, int ddepth, const Mat& kernel, Point anchor, double delta, int borderType);
CV_EXPORTS void copyMakeBorder(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int borderType, Scalar borderValue);
CV_EXPORTS void minMaxLoc(const Mat& src, double* maxval, double* minval,
                          vector<int>* maxloc, vector<int>* minloc, const Mat& mask=Mat());
CV_EXPORTS double norm(const Mat& src, int normType, const Mat& mask=Mat());
CV_EXPORTS double norm(const Mat& src1, const Mat& src2, int normType, const Mat& mask=Mat());
CV_EXPORTS bool cmpEps(const Mat& src1, const Mat& src2, double maxDiff, vector<int>* loc);
CV_EXPORTS void logicOp(const Mat& src1, const Mat& src2, Mat& dst, char c);
CV_EXPORTS void logicOp(const Mat& src, const Scalar& s, Mat& dst, char c);
CV_EXPORTS void compare(const Mat& src1, const Mat& src2, Mat& dst, int cmpop);
CV_EXPORTS void compare(const Mat& src, const Scalar& s, Mat& dst, int cmpop);    
CV_EXPORTS void gemm(const Mat& src1, const Mat& src2, double alpha,
                     const Mat& src3, double beta, Mat& dst, int flags);
CV_EXPORTS void crosscorr(const Mat& src1, const Mat& src2, Mat& dst, int dtype);

}

#endif

