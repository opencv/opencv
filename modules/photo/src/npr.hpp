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

// Columns processed together by the passes that run down the image: they are copied to contiguous
// lines, so the image is never transposed.
static const int NPR_COLUMN_BLOCK = 64;

class Domain_Filter
{
    public:
        Mat ct_H, ct_V, horiz, vert;
        void init(const Mat &img, int flags, float sigma_s, float sigma_r);
        void getGradientx( const Mat &img, Mat &gx);
        void getGradienty( const Mat &img, Mat &gy);
        static void find_magnitude(const Mat &img, Mat &mag);
        void compute_Rfilter(Mat &O, Mat &horiz, float sigma_h);
        void compute_Rfilter_vertical(Mat &O, Mat &vert, float sigma_h);
        void compute_NCfilter(Mat &O, float radius);
        void filter(Mat &img, float sigma_s, float sigma_r, int flags);
        void pencil_sketch(Mat &img, Mat &sketch, Mat &color_res, float sigma_s, float sigma_r, float shade_factor);
        void Depth_of_field(const Mat &img, Mat &img1, float sigma_s, float sigma_r);
};

// Box limits of the normalized convolution on one line of a cumulative domain transform. domain holds
// the n transform values followed by +inf; lo[j] and up[j] are the first indices whose transform
// exceeds domain[j] - radius and domain[j] + radius, each search starting at the previous limit.
static void boxfilter_limits(const float* domain, int n, float radius, int* lo, int* up)
{
    float lower = domain[0] - radius;
    float upper = domain[0] + radius;
    lo[0] = up[0] = 0;
    for(int k = 0; k < n; k++)
    {
        if(domain[k] > lower)
        {
            lo[0] = k;
            break;
        }
    }
    // the +inf at n ends this search when the whole line lies within the radius
    for(int k = 0; k <= n; k++)
    {
        if(domain[k] > upper)
        {
            up[0] = k;
            break;
        }
    }

    // a search that finds nothing reuses the last step found, as in the original algorithm
    int step = 0;
    for(int j = 1; j < n; j++)
    {
        lower = domain[j] - radius;
        upper = domain[j] + radius;
        for(int k = lo[j-1]; k <= n; k++)
        {
            if(domain[k] > lower)
            {
                step = k - lo[j-1];
                break;
            }
        }
        lo[j] = lo[j-1] + step;
        for(int k = up[j-1]; k <= n; k++)
        {
            if(domain[k] > upper)
            {
                step = k - up[j-1];
                break;
            }
        }
        up[j] = up[j-1] + step;
    }
}

// Normalized convolution of one line, in place: line holds n pixels of cn floats, domain the n values
// of the cumulative domain transform followed by +inf. sum, lo and up are scratch buffers of
// (n + 1) * cn floats, n ints and n ints.
static void nc_filter_line(float* line, int n, int cn, const float* domain, float radius,
                           float* sum, int* lo, int* up)
{
    boxfilter_limits(domain, n, radius, lo, up);

    // summed table along the line: sum[j] = line[0] + ... + line[j-1]
    for(int c = 0; c < cn; c++)
    {
        sum[c] = 0.f;
        sum[cn+c] = line[c];
    }
    for(int j = 2; j <= n; j++)
        for(int c = 0; c < cn; c++)
            sum[j*cn+c] = line[(j-1)*cn+c] + sum[(j-1)*cn+c];

    // box averages; the integer limits are exact for any image size
    for(int j = 0; j < n; j++)
    {
        const float* su = sum + up[j]*cn;
        const float* sl = sum + lo[j]*cn;
        float denom = (float)(up[j] - lo[j]);
        for(int c = 0; c < cn; c++)
            line[j*cn+c] = (su[c] - sl[c]) / denom;
    }
}

// Copies columns x0 .. x0+bw-1 of a single-channel transform into bw lines of ct.rows values, each
// followed by +inf.
static void gather_domain_columns(const Mat &ct, int x0, int bw, float* domain)
{
    int h = ct.rows;
    for(int i = 0; i < h; i++)
    {
        const float* src = ct.ptr<float>(i) + x0;
        for(int b = 0; b < bw; b++)
            domain[b*(h+1)+i] = src[b];
    }
    for(int b = 0; b < bw; b++)
        domain[b*(h+1)+h] = (float) myinf;
}

void Domain_Filter::getGradientx( const Mat &img, Mat &gx)
{
    int w = img.cols;
    int h = img.rows;
    int channel = img.channels();

    for(int i=0;i<h;i++)
        for(int j=0;j<w;j++)
            for(int c=0;c<channel;++c)
            {
                gx.at<float>(i,j*channel+c) =
                    img.at<float>(i,(j+1)*channel+c) - img.at<float>(i,j*channel+c);
            }
}

void Domain_Filter::getGradienty( const Mat &img, Mat &gy)
{
    int w = img.cols;
    int h = img.rows;
    int channel = img.channels();

    for(int i=0;i<h;i++)
        for(int j=0;j<w;j++)
            for(int c=0;c<channel;++c)
            {
                gy.at<float>(i,j*channel+c) =
                    img.at<float>(i+1,j*channel+c) - img.at<float>(i,j*channel+c);

            }
}

void Domain_Filter::find_magnitude(const Mat &img, Mat &mag)
{
    // sum of the Sobel gradient magnitudes of the first three channels, one channel at a time
    Mat plane, gx, gy, m;
    for(int c = 0; c < 3; c++)
    {
        extractChannel(img, plane, c);
        Sobel(plane, gx, CV_32FC1, 1, 0, 3);
        Sobel(plane, gy, CV_32FC1, 0, 1, 3);
        if(c == 0)
            magnitude(gx, gy, mag);
        else
        {
            magnitude(gx, gy, m);
            mag += m;
        }
    }

    mag = 1.0f - mag;
}

void Domain_Filter::compute_Rfilter(Mat &output, Mat &hz, float sigma_h)
{
    int h = output.rows;
    int w = output.cols;
    int channel = output.channels();

    float a = (float) exp((-1.0 * sqrt(2.0)) / sigma_h);

    // Each row is independent: filter it left to right, then right to left.
    AutoBuffer<float> V(w);
    float* v = V.data();
    for(int i = 0; i < h; i++)
    {
        const float* hzr = hz.ptr<float>(i);
        for(int j = 0; j < w; j++)
            v[j] = std::pow(a, hzr[j]);

        float* t = output.ptr<float>(i);
        for(int j = 1; j < w; j++)
            for(int c = 0; c < channel; c++)
                t[j*channel+c] = t[j*channel+c] + (t[(j-1)*channel+c] - t[j*channel+c]) * v[j];
        for(int j = w-2; j >= 0; j--)
            for(int c = 0; c < channel; c++)
                t[j*channel+c] = t[j*channel+c] + (t[(j+1)*channel+c] - t[j*channel+c]) * v[j+1];
    }
}

// Vertical pass of the recursive filter: the same arithmetic per pixel as compute_Rfilter() on the
// transposed image, run down blocks of columns instead of transposing.
void Domain_Filter::compute_Rfilter_vertical(Mat &output, Mat &vert_hz, float sigma_h)
{
    int h = output.rows;
    int w = output.cols;
    int channel = output.channels();
    const int B = NPR_COLUMN_BLOCK;

    float a = (float) exp((-1.0 * sqrt(2.0)) / sigma_h);

    AutoBuffer<float> Vbuf((size_t)h * B);
    float* V = Vbuf.data();
    for(int x0 = 0; x0 < w; x0 += B)
    {
        int bw = std::min(B, w - x0);
        for(int i = 0; i < h; i++)
        {
            const float* hzr = vert_hz.ptr<float>(i) + x0;
            for(int j = 0; j < bw; j++)
                V[i*B+j] = std::pow(a, hzr[j]);
        }

        for(int i = 1; i < h; i++)
        {
            float* t = output.ptr<float>(i) + x0*channel;
            const float* tp = output.ptr<float>(i-1) + x0*channel;
            const float* v = V + i*B;
            for(int j = 0; j < bw; j++)
                for(int c = 0; c < channel; c++)
                    t[j*channel+c] = t[j*channel+c] + (tp[j*channel+c] - t[j*channel+c]) * v[j];
        }
        for(int i = h-2; i >= 0; i--)
        {
            float* t = output.ptr<float>(i) + x0*channel;
            const float* tn = output.ptr<float>(i+1) + x0*channel;
            const float* v = V + (i+1)*B;
            for(int j = 0; j < bw; j++)
                for(int c = 0; c < channel; c++)
                    t[j*channel+c] = t[j*channel+c] + (tn[j*channel+c] - t[j*channel+c]) * v[j];
        }
    }
}

// One iteration of the normalized convolution filter: a pass along the rows with ct_H, then a pass
// down the columns with ct_V, both line by line in place.
void Domain_Filter::compute_NCfilter(Mat &output, float radius)
{
    int h = output.rows;
    int w = output.cols;
    int cn = output.channels();
    int n = std::max(h, w);
    const int B = NPR_COLUMN_BLOCK;

    AutoBuffer<float> sumbuf((size_t)(n + 1) * cn);
    AutoBuffer<int> idxbuf((size_t)n * 2);
    AutoBuffer<float> domainbuf(std::max((size_t)w + 1, (size_t)B * (h + 1)));
    AutoBuffer<float> colbuf((size_t)B * h * cn);
    float* sum = sumbuf.data();
    int* lo = idxbuf.data();
    int* up = lo + n;
    float* domain = domainbuf.data();
    float* col = colbuf.data();

    for(int i = 0; i < h; i++)
    {
        const float* ct = ct_H.ptr<float>(i);
        std::copy(ct, ct + w, domain);
        domain[w] = (float) myinf;
        nc_filter_line(output.ptr<float>(i), w, cn, domain, radius, sum, lo, up);
    }

    for(int x0 = 0; x0 < w; x0 += B)
    {
        int bw = std::min(B, w - x0);
        gather_domain_columns(ct_V, x0, bw, domain);
        for(int i = 0; i < h; i++)
        {
            const float* src = output.ptr<float>(i) + x0*cn;
            for(int b = 0; b < bw; b++)
                for(int c = 0; c < cn; c++)
                    col[(b*h+i)*cn+c] = src[b*cn+c];
        }
        for(int b = 0; b < bw; b++)
            nc_filter_line(col + b*h*cn, h, cn, domain + b*(h+1), radius, sum, lo, up);
        for(int i = 0; i < h; i++)
        {
            float* dst = output.ptr<float>(i) + x0*cn;
            for(int b = 0; b < bw; b++)
                for(int c = 0; c < cn; c++)
                    dst[b*cn+c] = col[(b*h+i)*cn+c];
        }
    }
}

void Domain_Filter::init(const Mat &img, int flags, float sigma_s, float sigma_r)
{
    int h = img.size().height;
    int w = img.size().width;
    int channel = img.channels();

    //////////////////////// Compute the l1-norm distance of neighbor pixels ////////////////////////////////////////////////

    horiz = Mat(h,w,CV_32FC1);
    vert = Mat(h,w,CV_32FC1);

    for(int i = 0; i < h; i++)
    {
        const float* src = img.ptr<float>(i);
        float* d = horiz.ptr<float>(i);
        d[0] = 0.f;
        for(int j = 1; j < w; j++)
        {
            float s = 0.f;
            for(int c = 0; c < channel; c++)
                s = s + abs(src[j*channel+c] - src[(j-1)*channel+c]);
            d[j] = s;
        }
    }

    vert.row(0).setTo(Scalar::all(0));
    for(int i = 1; i < h; i++)
    {
        const float* src = img.ptr<float>(i);
        const float* prev = img.ptr<float>(i-1);
        float* d = vert.ptr<float>(i);
        for(int j = 0; j < w; j++)
        {
            float s = 0.f;
            for(int c = 0; c < channel; c++)
                s = s + abs(src[j*channel+c] - prev[j*channel+c]);
            d[j] = s;
        }
    }

    ////////////////////// Compute the derivatives of the horizontal and vertical domain transforms. /////////////////////////////

    multiply(horiz,sigma_s/sigma_r,horiz);
    multiply(vert,sigma_s/sigma_r,vert);

    horiz = 1.0f + horiz;
    vert = 1.0f + vert;

    if(flags == 2)
    {
        // the cumulative transforms, summed in place
        for(int i = 0; i < h; i++)
        {
            float* hr = horiz.ptr<float>(i);
            for(int j = 1; j < w; j++)
                hr[j] = hr[j] + hr[j-1];
        }

        for(int i = 1; i < h; i++)
        {
            const float* vp = vert.ptr<float>(i-1);
            float* vr = vert.ptr<float>(i);
            for(int j = 0; j < w; j++)
                vr[j] = vr[j] + vp[j];
        }

        ct_H = horiz;
        ct_V = vert;
        horiz.release();
        vert.release();
    }

}

// Filters img in place.
void Domain_Filter::filter(Mat &img, float sigma_s = 60, float sigma_r = 0.4, int flags = 1)
{
    int no_of_iter = 3;
    float sigma_h = sigma_s;

    init(img,flags,sigma_s,sigma_r);

    if(flags == 1)
    {
        for(int i=0;i<no_of_iter;i++)
        {
            sigma_h = (float) (sigma_s * sqrt(3.0) * std::pow(2,(no_of_iter - (i+1))) / sqrt(std::pow(4,no_of_iter) -1));

            compute_Rfilter(img, horiz, sigma_h);

            compute_Rfilter_vertical(img, vert, sigma_h);
        }
    }
    else if(flags == 2)
    {
        float radius;

        for(int i=0;i<no_of_iter;i++)
        {
            sigma_h = (float) (sigma_s * sqrt(3.0) * std::pow(2,(no_of_iter - (i+1))) / sqrt(std::pow(4,no_of_iter) -1));

            radius = (float) sqrt(3.0) * sigma_h;

            compute_NCfilter(img, radius);
        }
    }
}

// img is released once it is no longer needed.
void Domain_Filter::pencil_sketch(Mat &img, Mat &sketch, Mat &color_res, float sigma_s, float sigma_r, float shade_factor)
{
    int no_of_iter = 3;
    init(img,2,sigma_s,sigma_r);
    int h = img.size().height;
    int w = img.size().width;
    const int B = NPR_COLUMN_BLOCK;

    // The original algorithm loops over no_of_iter passes, but only the first pass reaches sketch and
    // color_res; the later passes were computed and discarded.
    float sigma_h = (float) (sigma_s * sqrt(3.0) * std::pow(2,(no_of_iter - 1)) / sqrt(std::pow(4,no_of_iter) -1));
    float radius = (float) sqrt(3.0) * sigma_h;

    // pen_res = shade_factor * (box width along the row + box width down the column)
    Mat pen_res = Mat(h,w,CV_32FC1);
    int n = std::max(h, w);
    AutoBuffer<int> idxbuf((size_t)n * 2);
    AutoBuffer<float> domainbuf(std::max((size_t)w + 1, (size_t)B * (h + 1)));
    int* lo = idxbuf.data();
    int* up = lo + n;
    float* domain = domainbuf.data();

    for(int i = 0; i < h; i++)
    {
        const float* ct = ct_H.ptr<float>(i);
        std::copy(ct, ct + w, domain);
        domain[w] = (float) myinf;
        boxfilter_limits(domain, w, radius, lo, up);
        float* pr = pen_res.ptr<float>(i);
        for(int j = 0; j < w; j++)
            pr[j] = (float)(up[j] - lo[j]);
    }

    for(int x0 = 0; x0 < w; x0 += B)
    {
        int bw = std::min(B, w - x0);
        gather_domain_columns(ct_V, x0, bw, domain);
        for(int b = 0; b < bw; b++)
        {
            boxfilter_limits(domain + b*(h+1), h, radius, lo, up);
            for(int i = 0; i < h; i++)
            {
                float* pr = pen_res.ptr<float>(i) + x0 + b;
                *pr = shade_factor * (*pr + (float)(up[i] - lo[i]));
            }
        }
    }
    ct_H.release();
    ct_V.release();

    sketch = pen_res;

    /////////////////////// convert to YCBCR model for color pencil drawing //////////////////////////////////////////////////////

    Mat color_sketch;
    cvtColor(img,color_sketch,COLOR_BGR2YCrCb);
    img.release();

    insertChannel(pen_res, color_sketch, 0);
    cvtColor(color_sketch,color_res,COLOR_YCrCb2BGR);
}
