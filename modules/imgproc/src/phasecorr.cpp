/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2008-2011, William Lucas
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

#include "precomp.hpp"
#include <vector>

namespace cv
{

static void divComplex(InputArray _src1, InputArray _src2, OutputArray _dst)
{
    Mat src1 = _src1.getMat();
    Mat src2 = _src2.getMat();
    
    CV_Assert( src1.type() == src2.type() && src1.size() == src2.size());
    CV_Assert( src1.type() == CV_32FC2 || src1.type() == CV_64FC2 );
    
    _dst.create(src1.size(), src1.type());
    Mat dst  = _dst.getMat();
    
    int length = src1.rows*src1.cols;
    
    if(src1.depth() == CV_32F)
    {
        const float* dataA = (const float*)src1.data;
        const float* dataB = (const float*)src2.data;
        float* dataC = (float*)dst.data;
        float eps = FLT_EPSILON; // prevent div0 problems
        
        for(int j = 0; j < length - 1; j += 2)
        {
            double denom = (double)(dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps);
            double re = (double)(dataA[j]*dataB[j] + dataA[j+1]*dataB[j+1]);
            double im = (double)(dataA[j+1]*dataB[j] - dataA[j]*dataB[j+1]);
            dataC[j] = (float)(re / denom);
            dataC[j+1] = (float)(im / denom);
        }
    }
    else
    {
        const double* dataA = (const double*)src1.data;
        const double* dataB = (const double*)src2.data;
        double* dataC = (double*)dst.data;
        double eps = DBL_EPSILON; // prevent div0 problems
        
        for(int j = 0; j < length - 1; j += 2)
        {
            double denom = dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps;
            double re = dataA[j]*dataB[j] + dataA[j+1]*dataB[j+1];
            double im = dataA[j+1]*dataB[j] - dataA[j]*dataB[j+1];
            dataC[j] = re / denom;
            dataC[j+1] = im / denom;
        }
    }
}

static void absComplex(InputArray _src, OutputArray _dst)
{
    Mat src = _src.getMat();
    
    CV_Assert( src.type() == CV_32FC2 || src.type() == CV_64FC2 );
    
    vector<Mat> planes;
    split(src, planes);
    
    magnitude(planes[0], planes[1], planes[0]);
    planes[1] = Mat::zeros(planes[0].size(), planes[0].type());
    
    merge(planes, _dst);
}

static void fftShift(InputOutputArray _out)
{
    Mat out = _out.getMat();
    
    vector<Mat> planes;
    split(out, planes);
    
    int xMid = out.cols >> 1;
    int yMid = out.rows >> 1;
    
    for(size_t i = 0; i < planes.size(); i++)
    {
        // perform quadrant swaps...
        Mat tmp;
        Mat q0(planes[i], Rect(0,    0,    xMid, yMid));
        Mat q1(planes[i], Rect(xMid, 0,    xMid, yMid));
        Mat q2(planes[i], Rect(0,    yMid, xMid, yMid));
        Mat q3(planes[i], Rect(xMid, yMid, xMid, yMid));
        
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);
        
        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);
    }
    
    merge(planes, out);
}

Point2d weightedCentroid(InputArray _src, cv::Point peakLocation, cv::Size weightBoxSize)
{
    Mat src = _src.getMat();
    
    int type = src.type();
    CV_Assert( type == CV_32FC1 || type == CV_64FC1 );
    
    int minr = peakLocation.y - (weightBoxSize.height >> 1);
    int maxr = peakLocation.y + (weightBoxSize.height >> 1);
    int minc = peakLocation.x - (weightBoxSize.width  >> 1);
    int maxc = peakLocation.x + (weightBoxSize.width  >> 1);
    
    Point2d centroid;
    double sumIntensity = 0.0;
    
    // clamp the values to min and max if needed.
    if(minr < 0)
    {
        minr = 0;
    }
    
    if(minc < 0)
    {
        minc = 0;
    }
    
    if(maxr > src.rows - 1)
    {
        maxr = src.rows - 1;
    }
    
    if(maxc > src.cols - 1)
    {
        maxc = src.cols - 1;
    }
    
    if(type == CV_32FC1)
    {
        const float* dataIn = (const float*)src.data;
        dataIn += minr*src.cols;
        for(int y = minr; y <= maxr; y++)
        {
            for(int x = minc; x <= maxc; x++)
            {
                centroid.x   += (double)x*dataIn[x];
                centroid.y   += (double)y*dataIn[x];
                sumIntensity += (double)dataIn[x];
            }
            
            dataIn += src.cols;
        }
    }
    else
    {
        const double* dataIn = (const double*)src.data;
        dataIn += minr*src.cols;
        for(int y = minr; y <= maxr; y++)
        {
            for(int x = minc; x <= maxc; x++)
            {
                centroid.x   += (double)x*dataIn[x];
                centroid.y   += (double)y*dataIn[x];
                sumIntensity += dataIn[x];
            }
            
            dataIn += src.cols;
        }
    }
    
    sumIntensity += DBL_EPSILON; // prevent div0 problems...
    
    centroid.x /= sumIntensity;
    centroid.y /= sumIntensity;
    
    return centroid;
}
    
}

cv::Point2d cv::phaseCorrelate(InputArray _src1, InputArray _src2, InputArray _window)
{
    Mat src1 = _src1.getMat();
    Mat src2 = _src2.getMat();
    Mat window = _window.getMat();

    CV_Assert( src1.type() == src2.type());
    CV_Assert( src1.type() == CV_32FC1 || src1.type() == CV_64FC1 );
    CV_Assert( src1.size == src2.size);

    if(!window.empty())
    {
        CV_Assert( src1.type() == window.type());
        CV_Assert( src1.size == window.size);
    }

    int M = getOptimalDFTSize(src1.rows);
    int N = getOptimalDFTSize(src1.cols);

    Mat padded1, padded2, paddedWin;

    if(M != src1.rows || N != src1.cols)
    {
        copyMakeBorder(src1, padded1, 0, M - src1.rows, 0, N - src1.cols, BORDER_CONSTANT, Scalar::all(0));
        copyMakeBorder(src2, padded2, 0, M - src2.rows, 0, N - src2.cols, BORDER_CONSTANT, Scalar::all(0));

        if(!window.empty())
        {
            copyMakeBorder(window, paddedWin, 0, M - window.rows, 0, N - window.cols, BORDER_CONSTANT, Scalar::all(0));
        }
    }
    else
    {
        padded1 = src1;
        padded2 = src2;
        paddedWin = window;
    }

    Mat FFT1, FFT2, P, Pm, C;

    // perform window multiplication if available
    if(!paddedWin.empty())
    {
        // apply window to both images before proceeding...
        multiply(paddedWin, padded1, padded1);
        multiply(paddedWin, padded2, padded2);
    }

    // TODO should be able to improve speed by switching to CCS packed matrices
    vector<Mat> cplx1, cplx2;
    cplx1.push_back(padded1);
    cplx1.push_back(Mat::zeros(padded1.size(), padded1.type()));
    merge(cplx1, FFT1);

    cplx2.push_back(padded2);
    cplx2.push_back(Mat::zeros(padded2.size(), padded2.type()));
    merge(cplx2, FFT2);

    // execute phase correlation equation
    // Reference: http://en.wikipedia.org/wiki/Phase_correlation
    dft(FFT1, FFT1);
    dft(FFT2, FFT2);

    mulSpectrums(FFT1, FFT2, P, 0, true);

    // TODO these two functions need to be changed to work with CCS packed matrices...
    absComplex(P, Pm);
    divComplex(P, Pm, C); // FF* / |FF*| (phase correlation equation completed here...)

    idft(C, C); // gives us the nice peak shift location...

    vector<Mat> Cplanes;
    split(C, Cplanes);
    C = Cplanes[0]; // use only the real plane since that's all that's left...

    fftShift(C); // shift the energy to the center of the frame.

    // locate the highest peak
    Point peakLoc;
    minMaxLoc(C, NULL, NULL, NULL, &peakLoc);

    // get the phase shift with sub-pixel accuracy, 5x5 window seems about right here...
    Point2d t;
    t = weightedCentroid(C, peakLoc, Size(5, 5));

    // adjust shift relative to image center...
    Point2d center((double)src1.cols / 2.0, (double)src1.rows / 2.0);

    return (center - t);
}


void cv::createHanningWindow(OutputArray _dst, cv::Size winSize, int type)
{
    CV_Assert( type == CV_32FC1 || type == CV_64FC1 );

    _dst.create(winSize, type);
    Mat dst = _dst.getMat();

    int rows = dst.rows;
    int cols = dst.cols;

    if(dst.depth() == CV_32F)
    {
        float* dstData = (float*)dst.data;

        for(int i = 0; i < rows; i++)
        {
            double wr = 0.5 * (1.0f - cos(2.0f * CV_PI * (double)i / (double)(rows - 1)));
            for(int j = 0; j < cols; j++)
            {
                double wc = 0.5 * (1.0f - cos(2.0f * CV_PI * (double)j / (double)(cols - 1)));
                dstData[i*cols + j] = wr * wc;
            }
        }

        // perform batch sqrt for SSE performance gains
        cv::sqrt(dst, dst);
    }
    else
    {
        double* dstData = (double*)dst.data;

        for(int i = 0; i < rows; i++)
        {
            double wr = 0.5 * (1.0 - cos(2.0 * CV_PI * (double)i / (double)(rows - 1)));
            for(int j = 0; j < cols; j++)
            {
                double wc = 0.5 * (1.0 - cos(2.0 * CV_PI * (double)j / (double)(cols - 1)));
                dstData[i*cols + j] = wr * wc;
            }
        }

        // perform batch sqrt for SSE performance gains
        cv::sqrt(dst, dst);
    }
}
