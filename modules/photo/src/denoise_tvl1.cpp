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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
#include <vector>
#include <algorithm>

#define ABSCLIP(val,threshold) MIN(MAX((val),-(threshold)),(threshold))

namespace cv{

    class AddFloatToCharScaled{
        public:
            AddFloatToCharScaled(double scale):_scale(scale){}
            inline double operator()(double a,uchar b){
                return a+_scale*((double)b);
            }
        private:
            double _scale;
    };

#ifndef OPENCV_NOSTL
    using std::transform;
#else
    template <class InputIterator, class InputIterator2, class OutputIterator, class BinaryOperator>
    static OutputIterator transform (InputIterator first1, InputIterator last1, InputIterator2 first2,
                                     OutputIterator result, BinaryOperator binary_op)
    {
        while (first1 != last1)
        {
            *result = binary_op(*first1, *first2);
            ++result; ++first1; ++first2;
        }
        return result;
    }
#endif
    void denoise_TVL1(const std::vector<Mat>& observations,Mat& result, double lambda, int niters){

        CV_Assert(observations.size()>0 && niters>0 && lambda>0);

        const double L2 = 8.0, tau = 0.02, sigma = 1./(L2*tau), theta = 1.0;
        double clambda = (double)lambda;
        double s=0;
        const int workdepth = CV_64F;

        int i, x, y, rows=observations[0].rows, cols=observations[0].cols,count;
        for(i=1;i<(int)observations.size();i++){
            CV_Assert(observations[i].rows==rows && observations[i].cols==cols);
        }

        Mat X, P = Mat::zeros(rows, cols, CV_MAKETYPE(workdepth, 2));
        observations[0].convertTo(X, workdepth, 1./255);
        std::vector< Mat_<double> > Rs(observations.size());
        for(count=0;count<(int)Rs.size();count++){
            Rs[count]=Mat::zeros(rows,cols,workdepth);
        }

        for( i = 0; i < niters; i++ )
        {
            double currsigma = i == 0 ? 1 + sigma : sigma;

            // P_ = P + sigma*nabla(X)
            // P(x,y) = P_(x,y)/max(||P(x,y)||,1)
            for( y = 0; y < rows; y++ )
            {
                const double* x_curr = X.ptr<double>(y);
                const double* x_next = X.ptr<double>(std::min(y+1, rows-1));
                Point2d* p_curr = P.ptr<Point2d>(y);
                double dx, dy, m;
                for( x = 0; x < cols-1; x++ )
                {
                    dx = (x_curr[x+1] - x_curr[x])*currsigma + p_curr[x].x;
                    dy = (x_next[x] - x_curr[x])*currsigma + p_curr[x].y;
                    m = 1.0/std::max(std::sqrt(dx*dx + dy*dy), 1.0);
                    p_curr[x].x = dx*m;
                    p_curr[x].y = dy*m;
                }
                dy = (x_next[x] - x_curr[x])*currsigma + p_curr[x].y;
                m = 1.0/std::max(std::abs(dy), 1.0);
                p_curr[x].x = 0.0;
                p_curr[x].y = dy*m;
            }


            //Rs = clip(Rs + sigma*(X-imgs), -clambda, clambda)
            for(count=0;count<(int)Rs.size();count++){
                transform<MatIterator_<double>,MatConstIterator_<uchar>,MatIterator_<double>,AddFloatToCharScaled>(
                        Rs[count].begin(),Rs[count].end(),observations[count].begin<uchar>(),
                        Rs[count].begin(),AddFloatToCharScaled(-sigma/255.0));
                Rs[count]+=sigma*X;
                min(Rs[count],clambda,Rs[count]);
                max(Rs[count],-clambda,Rs[count]);
            }

            for( y = 0; y < rows; y++ )
            {
                double* x_curr = X.ptr<double>(y);
                const Point2d* p_curr = P.ptr<Point2d>(y);
                const Point2d* p_prev = P.ptr<Point2d>(std::max(y - 1, 0));

                // X1 = X + tau*(-nablaT(P))
                x = 0;
                s=0.0;
                for(count=0;count<(int)Rs.size();count++){
                    s=s+Rs[count](y,x);
                }
                double x_new = x_curr[x] + tau*(p_curr[x].y - p_prev[x].y)-tau*s;
                    // X = X2 + theta*(X2 - X)
                x_curr[x] = x_new + theta*(x_new - x_curr[x]);


                for(x = 1; x < cols; x++ )
                {
                    s=0.0;
                    for(count=0;count<(int)Rs.size();count++){
                        s+=Rs[count](y,x);
                    }
                        // X1 = X + tau*(-nablaT(P))
                    x_new = x_curr[x] + tau*(p_curr[x].x - p_curr[x-1].x + p_curr[x].y - p_prev[x].y)-tau*s;
                        // X = X2 + theta*(X2 - X)
                    x_curr[x] = x_new + theta*(x_new - x_curr[x]);
                }
            }
        }

        result.create(X.rows,X.cols,CV_8U);
        X.convertTo(result, CV_8U, 255);
    }
}
