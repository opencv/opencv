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
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "opencv2/photo.hpp"
#include <iostream>
#include <stdlib.h>
#include <limits>
#include <math.h>


using namespace std;
using namespace cv;

double myinf = std::numeric_limits<double>::infinity();

class Domain_Filter
{
    public:
        Mat ct_H, ct_V, horiz, vert, O, O_t, lower_idx, upper_idx;
        void init(const Mat &img, int flags, float sigma_s, float sigma_r);
        void getGradientx( const Mat &img, Mat &gx);
        void getGradienty( const Mat &img, Mat &gy);
        void diffx(const Mat &img, Mat &temp);
        void diffy(const Mat &img, Mat &temp);
        void find_magnitude(Mat &img, Mat &mag);
        void compute_boxfilter(Mat &output, Mat &hz, Mat &psketch, float radius);
        void compute_Rfilter(Mat &O, Mat &horiz, float sigma_h);
        void compute_NCfilter(Mat &O, Mat &horiz, Mat &psketch, float radius);
        void filter(const Mat &img, Mat &res, float sigma_s, float sigma_r, int flags);
        void pencil_sketch(const Mat &img, Mat &sketch, Mat &color_res, float sigma_s, float sigma_r, float shade_factor);
        void Depth_of_field(const Mat &img, Mat &img1, float sigma_s, float sigma_r);
};

void Domain_Filter::diffx(const Mat &img, Mat &temp)
{
    int channel = img.channels();
    int h = img.size().height;
    int w = img.size().width;

    for(int i = 0; i < h; i++)
    {
        const float* src = img.ptr<float>(i);
        float* dst = temp.ptr<float>(i);
        for(int j = 0; j < w-1; j++)
            for(int c = 0; c < channel; c++)
                dst[j*channel+c] = src[(j+1)*channel+c] - src[j*channel+c];
    }
}

void Domain_Filter::diffy(const Mat &img, Mat &temp)
{
    int channel = img.channels();
    int h = img.size().height;
    int w = img.size().width;

    for(int i = 0; i < h-1; i++)
    {
        const float* src0 = img.ptr<float>(i);
        const float* src1 = img.ptr<float>(i+1);
        float* dst = temp.ptr<float>(i);
        for(int j = 0; j < w; j++)
            for(int c = 0; c < channel; c++)
                dst[j*channel+c] = src1[j*channel+c] - src0[j*channel+c];
    }
}

void Domain_Filter::getGradientx( const Mat &img, Mat &gx)
{
    int w = img.cols;
    int h = img.rows;
    int channel = img.channels();

    for(int i=0;i<h;i++)
    {
        const float* src = img.ptr<float>(i);
        float* dst = gx.ptr<float>(i);
        for(int j=0;j<w;j++)
            for(int c=0;c<channel;++c)
                dst[j*channel+c] = src[(j+1)*channel+c] - src[j*channel+c];
    }
}

void Domain_Filter::getGradienty( const Mat &img, Mat &gy)
{
    int w = img.cols;
    int h = img.rows;
    int channel = img.channels();

    for(int i=0;i<h;i++)
    {
        const float* src0 = img.ptr<float>(i);
        const float* src1 = img.ptr<float>(i+1);
        float* dst = gy.ptr<float>(i);
        for(int j=0;j<w;j++)
            for(int c=0;c<channel;++c)
                dst[j*channel+c] = src1[j*channel+c] - src0[j*channel+c];
    }
}

void Domain_Filter::find_magnitude(Mat &img, Mat &mag)
{
    int h = img.rows;
    int w = img.cols;

    vector <Mat> planes;
    split(img, planes);

    Mat magXR = Mat(h, w, CV_32FC1);
    Mat magYR = Mat(h, w, CV_32FC1);

    Mat magXG = Mat(h, w, CV_32FC1);
    Mat magYG = Mat(h, w, CV_32FC1);

    Mat magXB = Mat(h, w, CV_32FC1);
    Mat magYB = Mat(h, w, CV_32FC1);

    Sobel(planes[0], magXR, CV_32FC1, 1, 0, 3);
    Sobel(planes[0], magYR, CV_32FC1, 0, 1, 3);

    Sobel(planes[1], magXG, CV_32FC1, 1, 0, 3);
    Sobel(planes[1], magYG, CV_32FC1, 0, 1, 3);

    Sobel(planes[2], magXB, CV_32FC1, 1, 0, 3);
    Sobel(planes[2], magYB, CV_32FC1, 0, 1, 3);

    Mat mag1 = Mat(h,w,CV_32FC1);
    Mat mag2 = Mat(h,w,CV_32FC1);
    Mat mag3 = Mat(h,w,CV_32FC1);

    magnitude(magXR,magYR,mag1);
    magnitude(magXG,magYG,mag2);
    magnitude(magXB,magYB,mag3);

    mag = mag1 + mag2 + mag3;
    mag = 1.0f - mag;
}

void Domain_Filter::compute_Rfilter(Mat &output, Mat &hz, float sigma_h)
{
    int h = output.rows;
    int w = output.cols;
    int channel = output.channels();

    float a = (float) exp((-1.0 * sqrt(2.0)) / sigma_h);

    Mat temp = Mat(h,w,CV_32FC3);
    output.copyTo(temp);

    parallel_for_(Range(0, h), [&](const Range& range) {
        std::vector<float> Vrow(w);
        for(int i = range.start; i < range.end; i++)
        {
            const float* hzRow = hz.ptr<float>(i);
            float* tRow = temp.ptr<float>(i);

            for(int j = 0; j < w; j++)
                Vrow[j] = std::pow(a, hzRow[j]);

            for(int j = 1; j < w; j++)
            {
                float v = Vrow[j];
                for(int c = 0; c < channel; c++)
                    tRow[j*channel+c] += (tRow[(j-1)*channel+c] - tRow[j*channel+c]) * v;
            }

            for(int j = w-2; j >= 0; j--)
            {
                float v = Vrow[j+1];
                for(int c = 0; c < channel; c++)
                    tRow[j*channel+c] += (tRow[(j+1)*channel+c] - tRow[j*channel+c]) * v;
            }
        }
    });

    temp.copyTo(output);
}

void Domain_Filter::compute_boxfilter(Mat &output, Mat &hz, Mat &psketch, float radius)
{
    int h = output.rows;
    int w = output.cols;
    Mat lower_pos = Mat(h,w,CV_32FC1);
    Mat upper_pos = Mat(h,w,CV_32FC1);

    lower_pos = hz - radius;
    upper_pos = hz + radius;

    lower_idx = Mat::zeros(h,w,CV_32FC1);
    upper_idx = Mat::zeros(h,w,CV_32FC1);

    std::vector<float> domain_row_buf(w+1);
    std::vector<float> lower_pos_row(w);
    std::vector<float> upper_pos_row(w);
    std::vector<float> temp_lower_idx(w);
    std::vector<float> temp_upper_idx(w);

    for(int i=0;i<h;i++)
    {
        const float* hzRow = hz.ptr<float>(i);
        const float* lpRow = lower_pos.ptr<float>(i);
        const float* upRow = upper_pos.ptr<float>(i);
        float* liRow = lower_idx.ptr<float>(i);
        float* uiRow = upper_idx.ptr<float>(i);

        for(int j=0;j<w;j++)
            domain_row_buf[j] = hzRow[j];
        domain_row_buf[w] = (float) myinf;

        for(int j=0;j<w;j++)
        {
            lower_pos_row[j] = lpRow[j];
            upper_pos_row[j] = upRow[j];
        }

        std::fill(temp_lower_idx.begin(), temp_lower_idx.end(), 0.0f);
        std::fill(temp_upper_idx.begin(), temp_upper_idx.end(), 0.0f);

        for(int j=0;j<w;j++)
        {
            if(domain_row_buf[j] > lower_pos_row[0])
            {
                temp_lower_idx[0] = (float) j;
                break;
            }
        }
        for(int j=0;j<w;j++)
        {
            if(domain_row_buf[j] > upper_pos_row[0])
            {
                temp_upper_idx[0] = (float) j;
                break;
            }
        }

        int temp_val = 0;
        for(int j=1;j<w;j++)
        {
            int count=0;
            for(int k=(int) temp_lower_idx[j-1];k<w+1;k++)
            {
                if(domain_row_buf[k] > lower_pos_row[j])
                {
                    temp_val = count;
                    break;
                }
                count++;
            }

            temp_lower_idx[j] = temp_lower_idx[j-1] + temp_val;

            count = 0;
            for(int k=(int) temp_upper_idx[j-1];k<w+1;k++)
            {
                if(domain_row_buf[k] > upper_pos_row[j])
                {
                    temp_val = count;
                    break;
                }
                count++;
            }

            temp_upper_idx[j] = temp_upper_idx[j-1] + temp_val;
        }

        for(int j=0;j<w;j++)
        {
            liRow[j] = temp_lower_idx[j] + 1;
            uiRow[j] = temp_upper_idx[j] + 1;
        }
    }
    psketch = upper_idx - lower_idx;
}

void Domain_Filter::compute_NCfilter(Mat &output, Mat &hz, Mat &psketch, float radius)
{
    int h = output.rows;
    int w = output.cols;
    int channel = output.channels();

    compute_boxfilter(output,hz,psketch,radius);

    Mat box_filter = Mat::zeros(h,w+1,CV_32FC3);

    for(int i = 0; i < h; i++)
    {
        const float* outRow = output.ptr<float>(i);
        float* bfRow = box_filter.ptr<float>(i);
        bfRow[1*channel+0] = outRow[0*channel+0];
        bfRow[1*channel+1] = outRow[0*channel+1];
        bfRow[1*channel+2] = outRow[0*channel+2];
        for(int j = 2; j < w+1; j++)
            for(int c=0;c<channel;c++)
                bfRow[j*channel+c] = outRow[(j-1)*channel+c] + bfRow[(j-1)*channel+c];
    }

    Mat final = Mat::zeros(h,w,CV_32FC3);

    // Compute the box filter using a summed area table.
    int hw1 = h * (w+1);
    for(int c=0;c<channel;c++)
    {
        int channel_offset = c * hw1;

        for(int i=0;i<h;i++)
        {
            const float* liRow  = lower_idx.ptr<float>(i);
            const float* uiRow  = upper_idx.ptr<float>(i);
            float* finRow = final.ptr<float>(i);
            int row_idx = i + 1;

            for(int j=0;j<w;j++)
            {
                int aval = channel_offset + ((int)liRow[j] - 1) * h + row_idx;
                int bval = channel_offset + ((int)uiRow[j] - 1) * h + row_idx;

                int r   = bval / hw1;
                int rem = bval - r * hw1;
                int q   = rem / h;
                int p   = rem - q * h;
                if(q==0) { p=h; q=w; r=r-1; }
                if(p==0) { p=h; q=q-1; }

                int r1   = aval / hw1;
                int rem1 = aval - r1 * hw1;
                int q1   = rem1 / h;
                int p1   = rem1 - q1 * h;
                if(p1==0) { p1=h; q1=q1-1; }

                const float* bfp  = box_filter.ptr<float>(p-1);
                const float* bfp1 = box_filter.ptr<float>(p1-1);
                float denom = uiRow[j] - liRow[j];
                finRow[j*channel+2-c] = (bfp[q*channel+(2-r)] - bfp1[q1*channel+(2-r1)]) / denom;
            }
        }
    }

    final.copyTo(output);
}

void Domain_Filter::init(const Mat &img, int flags, float sigma_s, float sigma_r)
{
    int h = img.size().height;
    int w = img.size().width;
    int channel = img.channels();

    ////////////////////////////////////     horizontal and vertical partial derivatives /////////////////////////////////

    Mat derivx = Mat::zeros(h,w-1,CV_32FC3);
    Mat derivy = Mat::zeros(h-1,w,CV_32FC3);

    diffx(img,derivx);
    diffy(img,derivy);

    Mat distx = Mat::zeros(h,w,CV_32FC1);
    Mat disty = Mat::zeros(h,w,CV_32FC1);

    //////////////////////// Compute the l1-norm distance of neighbor pixels ////////////////////////////////////////////////

    for(int i = 0; i < h; i++)
    {
        const float* dxRow   = derivx.ptr<float>(i);
        float* dstxRow = distx.ptr<float>(i);
        for(int j = 0, k = 1; j < w-1; j++, k++)
            for(int c = 0; c < channel; c++)
                dstxRow[k] += std::abs(dxRow[j*channel+c]);
    }

    for(int i = 0, k = 1; i < h-1; i++, k++)
    {
        const float* dyRow = derivy.ptr<float>(i);
        float* dstyRow = disty.ptr<float>(k);
        for(int j = 0; j < w; j++)
            for(int c = 0; c < channel; c++)
                dstyRow[j] += std::abs(dyRow[j*channel+c]);
    }

    ////////////////////// Compute the derivatives of the horizontal and vertical domain transforms. /////////////////////////////

    horiz = Mat(h,w,CV_32FC1);
    vert = Mat(h,w,CV_32FC1);

    Mat final = Mat(h,w,CV_32FC3);

    Mat tempx,tempy;
    multiply(distx,sigma_s/sigma_r,tempx);
    multiply(disty,sigma_s/sigma_r,tempy);

    horiz = 1.0f + tempx;
    vert = 1.0f + tempy;

    O = Mat(h,w,CV_32FC3);
    img.copyTo(O);

    O_t = Mat(w,h,CV_32FC3);

    if(flags == 2)
    {
        ct_H = Mat(h,w,CV_32FC1);
        ct_V = Mat(h,w,CV_32FC1);

        for(int i = 0; i < h; i++)
        {
            const float* hRow  = horiz.ptr<float>(i);
            float* ctRow = ct_H.ptr<float>(i);
            ctRow[0] = hRow[0];
            for(int j = 1; j < w; j++)
                ctRow[j] = hRow[j] + ctRow[j-1];
        }

        {
            const float* v0 = vert.ptr<float>(0);
            float* c0 = ct_V.ptr<float>(0);
            for(int j = 0; j < w; j++)
                c0[j] = v0[j];
            for(int i = 1; i < h; i++)
            {
                const float* vRow = vert.ptr<float>(i);
                const float* prevRow = ct_V.ptr<float>(i-1);
                float* ctRow = ct_V.ptr<float>(i);
                for(int j = 0; j < w; j++)
                    ctRow[j] = vRow[j] + prevRow[j];
            }
        }
    }
}

void Domain_Filter::filter(const Mat &img, Mat &res, float sigma_s = 60, float sigma_r = 0.4, int flags = 1)
{
    int no_of_iter = 3;
    int h = img.size().height;
    int w = img.size().width;
    float sigma_h = sigma_s;

    init(img,flags,sigma_s,sigma_r);

    if(flags == 1)
    {
        Mat vert_t = vert.t();

        for(int i=0;i<no_of_iter;i++)
        {
            sigma_h = (float) (sigma_s * sqrt(3.0) * std::pow(2,(no_of_iter - (i+1))) / sqrt(std::pow(4,no_of_iter) -1));

            compute_Rfilter(O, horiz, sigma_h);

            O_t = O.t();

            compute_Rfilter(O_t, vert_t, sigma_h);

            O = O_t.t();
        }
    }
    else if(flags == 2)
    {
        Mat vert_t = ct_V.t();
        Mat temp = Mat(h,w,CV_32FC1);
        Mat temp1 = Mat(w,h,CV_32FC1);

        float radius;

        for(int i=0;i<no_of_iter;i++)
        {
            sigma_h = (float) (sigma_s * sqrt(3.0) * std::pow(2,(no_of_iter - (i+1))) / sqrt(std::pow(4,no_of_iter) -1));

            radius = (float) sqrt(3.0) * sigma_h;

            compute_NCfilter(O, ct_H, temp,radius);

            O_t = O.t();

            compute_NCfilter(O_t, vert_t, temp1, radius);

            O = O_t.t();
        }
    }

    res = O.clone();
}

void Domain_Filter::pencil_sketch(const Mat &img, Mat &sketch, Mat &color_res, float sigma_s, float sigma_r, float shade_factor)
{
    int no_of_iter = 3;
    init(img,2,sigma_s,sigma_r);
    int h = img.size().height;
    int w = img.size().width;

    /////////////////////// convert to YCBCR model for color pencil drawing //////////////////////////////////////////////////////

    Mat color_sketch = Mat(h,w,CV_32FC3);

    cvtColor(img,color_sketch,COLOR_BGR2YCrCb);

    vector <Mat> YUV_channel;
    Mat vert_t = ct_V.t();

    float sigma_h = sigma_s;

    Mat penx = Mat(h,w,CV_32FC1);

    Mat pen_res = Mat::zeros(h,w,CV_32FC1);
    Mat peny = Mat(w,h,CV_32FC1);

    Mat peny_t;

    float radius;

    for(int i=0;i<no_of_iter;i++)
    {
        sigma_h = (float) (sigma_s * sqrt(3.0) * std::pow(2,(no_of_iter - (i+1))) / sqrt(std::pow(4,no_of_iter) -1));

        radius = (float) sqrt(3.0) * sigma_h;

        compute_boxfilter(O, ct_H, penx, radius);

        O_t = O.t();

        compute_boxfilter(O_t, vert_t, peny, radius);

        O = O_t.t();

        peny_t = peny.t();

        for(int k=0;k<h;k++)
        {
            const float* pxRow = penx.ptr<float>(k);
            const float* pyRow = peny_t.ptr<float>(k);
            float* prRow = pen_res.ptr<float>(k);
            for(int j=0;j<w;j++)
                prRow[j] = shade_factor * (pxRow[j] + pyRow[j]);
        }

        if(i==0)
        {
            sketch = pen_res.clone();
            split(color_sketch,YUV_channel);
            pen_res.copyTo(YUV_channel[0]);
            merge(YUV_channel,color_sketch);
            cvtColor(color_sketch,color_res,COLOR_YCrCb2BGR);
        }
    }
}
