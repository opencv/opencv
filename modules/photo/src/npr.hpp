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
// The normalized convolution copies its block of columns: narrower blocks keep that copy small for
// each thread, at the same speed.
static const int NPR_NC_COLUMN_BLOCK = 16;

// Lines are independent, so the passes run lines or blocks of columns in parallel; each line is
// computed by the same code whatever the split.
static double npr_stripes()
{
    return getNumThreads() * 4.0;
}

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

// Recursive filter of one row of w pixels of cn floats in place, left to right, then right to left.
// v is scratch for w floats.
static void rf_filter_row(float* t, const float* hz, int w, int cn, float a, float* v)
{
    for(int j = 0; j < w; j++)
        v[j] = std::pow(a, hz[j]);

    for(int j = 1; j < w; j++)
        for(int c = 0; c < cn; c++)
            t[j*cn+c] = t[j*cn+c] + (t[(j-1)*cn+c] - t[j*cn+c]) * v[j];
    for(int j = w-2; j >= 0; j--)
        for(int c = 0; c < cn; c++)
            t[j*cn+c] = t[j*cn+c] + (t[(j+1)*cn+c] - t[j*cn+c]) * v[j+1];
}

// The same arithmetic per pixel down columns x0 .. x0+bw-1, top to bottom, then bottom to top.
// V is scratch for output.rows * NPR_COLUMN_BLOCK floats.
static void rf_filter_columns(Mat &output, const Mat &vert_hz, int x0, int bw, float a, float* V)
{
    const int h = output.rows;
    const int cn = output.channels();
    const int B = NPR_COLUMN_BLOCK;

    for(int i = 0; i < h; i++)
    {
        const float* hzr = vert_hz.ptr<float>(i) + x0;
        for(int j = 0; j < bw; j++)
            V[i*B+j] = std::pow(a, hzr[j]);
    }

    for(int i = 1; i < h; i++)
    {
        float* t = output.ptr<float>(i) + x0*cn;
        const float* tp = output.ptr<float>(i-1) + x0*cn;
        const float* v = V + i*B;
        for(int j = 0; j < bw; j++)
            for(int c = 0; c < cn; c++)
                t[j*cn+c] = t[j*cn+c] + (tp[j*cn+c] - t[j*cn+c]) * v[j];
    }
    for(int i = h-2; i >= 0; i--)
    {
        float* t = output.ptr<float>(i) + x0*cn;
        const float* tn = output.ptr<float>(i+1) + x0*cn;
        const float* v = V + (i+1)*B;
        for(int j = 0; j < bw; j++)
            for(int c = 0; c < cn; c++)
                t[j*cn+c] = t[j*cn+c] + (tn[j*cn+c] - t[j*cn+c]) * v[j];
    }
}

void Domain_Filter::compute_Rfilter(Mat &output, Mat &hz, float sigma_h)
{
    int h = output.rows;
    int w = output.cols;
    int channel = output.channels();

    float a = (float) exp((-1.0 * sqrt(2.0)) / sigma_h);

    parallel_for_(Range(0, h), [&](const Range& range) {
        AutoBuffer<float> V(w);
        for(int i = range.start; i < range.end; i++)
            rf_filter_row(output.ptr<float>(i), hz.ptr<float>(i), w, channel, a, V.data());
    }, npr_stripes());
}

// Vertical pass of the recursive filter: the same arithmetic per pixel as compute_Rfilter() on the
// transposed image, run down blocks of columns instead of transposing.
void Domain_Filter::compute_Rfilter_vertical(Mat &output, Mat &vert_hz, float sigma_h)
{
    int h = output.rows;
    int w = output.cols;
    const int B = NPR_COLUMN_BLOCK;

    float a = (float) exp((-1.0 * sqrt(2.0)) / sigma_h);

    int nblocks = (w + B - 1) / B;
    parallel_for_(Range(0, nblocks), [&](const Range& range) {
        AutoBuffer<float> V((size_t)h * B);
        for(int x0 = range.start * B; x0 < std::min(w, range.end * B); x0 += B)
            rf_filter_columns(output, vert_hz, x0, std::min(B, w - x0), a, V.data());
    }, std::min((double)nblocks, npr_stripes()));
}

// One iteration of the normalized convolution filter: a pass along the rows with ct_H, then a pass
// down the columns with ct_V, both line by line in place.
void Domain_Filter::compute_NCfilter(Mat &output, float radius)
{
    int h = output.rows;
    int w = output.cols;
    int cn = output.channels();
    const int B = NPR_NC_COLUMN_BLOCK;

    parallel_for_(Range(0, h), [&](const Range& range) {
        AutoBuffer<float> sumbuf((size_t)(w + 1) * cn);
        AutoBuffer<int> idxbuf((size_t)w * 2);
        AutoBuffer<float> domainbuf((size_t)w + 1);
        float* sum = sumbuf.data();
        int* lo = idxbuf.data();
        int* up = lo + w;
        float* domain = domainbuf.data();
        for(int i = range.start; i < range.end; i++)
        {
            const float* ct = ct_H.ptr<float>(i);
            std::copy(ct, ct + w, domain);
            domain[w] = (float) myinf;
            nc_filter_line(output.ptr<float>(i), w, cn, domain, radius, sum, lo, up);
        }
    }, npr_stripes());

    int nblocks = (w + B - 1) / B;
    parallel_for_(Range(0, nblocks), [&](const Range& range) {
        AutoBuffer<float> sumbuf((size_t)(h + 1) * cn);
        AutoBuffer<int> idxbuf((size_t)h * 2);
        AutoBuffer<float> domainbuf((size_t)B * (h + 1));
        AutoBuffer<float> colbuf((size_t)B * h * cn);
        float* sum = sumbuf.data();
        int* lo = idxbuf.data();
        int* up = lo + h;
        float* domain = domainbuf.data();
        float* col = colbuf.data();
        for(int x0 = range.start * B; x0 < std::min(w, range.end * B); x0 += B)
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
    }, std::min((double)nblocks, npr_stripes()));
}

void Domain_Filter::init(const Mat &img, int flags, float sigma_s, float sigma_r)
{
    int h = img.size().height;
    int w = img.size().width;
    int channel = img.channels();

    //////////////////////// Compute the l1-norm distance of neighbor pixels ////////////////////////////////////////////////

    horiz = Mat(h,w,CV_32FC1);
    vert = Mat(h,w,CV_32FC1);

    // the l1 distances, then the derivatives of the horizontal and vertical domain transforms
    parallel_for_(Range(0, h), [&](const Range& range) {
        for(int i = range.start; i < range.end; i++)
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

            d = vert.ptr<float>(i);
            if(i == 0)
            {
                std::fill(d, d + w, 0.f);
                continue;
            }
            const float* prev = img.ptr<float>(i-1);
            for(int j = 0; j < w; j++)
            {
                float s = 0.f;
                for(int c = 0; c < channel; c++)
                    s = s + abs(src[j*channel+c] - prev[j*channel+c]);
                d[j] = s;
            }
        }

        Mat hr = horiz.rowRange(range), vr = vert.rowRange(range);
        multiply(hr,sigma_s/sigma_r,hr);
        multiply(vr,sigma_s/sigma_r,vr);
        hr = 1.0f + hr;
        vr = 1.0f + vr;
    }, npr_stripes());

    if(flags == 2)
    {
        // the cumulative transforms, summed in place: along each row, and down blocks of columns
        parallel_for_(Range(0, h), [&](const Range& range) {
            for(int i = range.start; i < range.end; i++)
            {
                float* hr = horiz.ptr<float>(i);
                for(int j = 1; j < w; j++)
                    hr[j] = hr[j] + hr[j-1];
            }
        }, npr_stripes());

        const int B = NPR_COLUMN_BLOCK;
        int nblocks = (w + B - 1) / B;
        parallel_for_(Range(0, nblocks), [&](const Range& range) {
            int x0 = range.start * B, x1 = std::min(w, range.end * B);
            for(int i = 1; i < h; i++)
            {
                const float* vp = vert.ptr<float>(i-1);
                float* vr = vert.ptr<float>(i);
                for(int j = x0; j < x1; j++)
                    vr[j] = vr[j] + vp[j];
            }
        }, std::min((double)nblocks, npr_stripes()));

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

    parallel_for_(Range(0, h), [&](const Range& range) {
        AutoBuffer<int> idxbuf((size_t)w * 2);
        AutoBuffer<float> domainbuf((size_t)w + 1);
        int* lo = idxbuf.data();
        int* up = lo + w;
        float* domain = domainbuf.data();
        for(int i = range.start; i < range.end; i++)
        {
            const float* ct = ct_H.ptr<float>(i);
            std::copy(ct, ct + w, domain);
            domain[w] = (float) myinf;
            boxfilter_limits(domain, w, radius, lo, up);
            float* pr = pen_res.ptr<float>(i);
            for(int j = 0; j < w; j++)
                pr[j] = (float)(up[j] - lo[j]);
        }
    }, npr_stripes());

    int nblocks = (w + B - 1) / B;
    parallel_for_(Range(0, nblocks), [&](const Range& range) {
        AutoBuffer<int> idxbuf((size_t)h * 2);
        AutoBuffer<float> domainbuf((size_t)B * (h + 1));
        int* lo = idxbuf.data();
        int* up = lo + h;
        float* domain = domainbuf.data();
        for(int x0 = range.start * B; x0 < std::min(w, range.end * B); x0 += B)
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
    }, std::min((double)nblocks, npr_stripes()));
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
