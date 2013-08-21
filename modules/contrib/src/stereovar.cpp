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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

/*
 This is a modification of the variational stereo correspondence algorithm, described in:
 S. Kosov, T. Thormaehlen, H.-P. Seidel "Accurate Real-Time Disparity Estimation with Variational Methods"
 Proceedings of the 5th International Symposium on Visual Computing, Vegas, USA

 This code is written by Sergey G. Kosov for "Visir PX" application as part of Project X (www.project-10.de)
 */

#include "precomp.hpp"
#include <limits.h>

namespace cv
{
StereoVar::StereoVar() : levels(3), pyrScale(0.5), nIt(5), minDisp(0), maxDisp(16), poly_n(3), poly_sigma(0), fi(25.0f), lambda(0.03f), penalization(PENALIZATION_TICHONOV), cycle(CYCLE_V), flags(USE_SMART_ID | USE_AUTO_PARAMS)
{
}

StereoVar::StereoVar(int _levels, double _pyrScale, int _nIt, int _minDisp, int _maxDisp, int _poly_n, double _poly_sigma, float _fi, float _lambda, int _penalization, int _cycle, int _flags) : levels(_levels), pyrScale(_pyrScale), nIt(_nIt), minDisp(_minDisp), maxDisp(_maxDisp), poly_n(_poly_n), poly_sigma(_poly_sigma), fi(_fi), lambda(_lambda), penalization(_penalization), cycle(_cycle), flags(_flags)
{ // No Parameters check, since they are all public
}

StereoVar::~StereoVar()
{
}

static Mat diffX(Mat &src)
{
    int cols = src.cols - 1;
    Mat dst(src.size(), src.type());
    for(int y = 0; y < src.rows; y++){
        const float* pSrc = src.ptr<float>(y);
        float* pDst = dst.ptr<float>(y);
        int x = 0;
#if CV_SSE2
        for (x = 0; x <= cols - 8; x += 8) {
            __m128 a0 = _mm_loadu_ps(pSrc + x);
            __m128 b0 = _mm_loadu_ps(pSrc + x + 1);
            __m128 a1 = _mm_loadu_ps(pSrc + x + 4);
            __m128 b1 = _mm_loadu_ps(pSrc + x + 5);
            b0 = _mm_sub_ps(b0, a0);
            b1 = _mm_sub_ps(b1, a1);
            _mm_storeu_ps(pDst + x, b0);
            _mm_storeu_ps(pDst + x + 4, b1);
        }
#endif
        for( ; x < cols; x++) pDst[x] = pSrc[x+1] - pSrc[x];
        pDst[cols] = 0.f;
    }
    return dst;
}

static Mat getGradient(Mat &src)
{
    register int x, y;
    Mat dst(src.size(), src.type());
    dst.setTo(0);
    for (y = 0; y < src.rows - 1; y++) {
        float *pSrc = src.ptr<float>(y);
        float *pSrcF = src.ptr<float>(y + 1);
        float *pDst = dst.ptr<float>(y);
        for (x = 0; x < src.cols - 1; x++)
            pDst[x] = fabs(pSrc[x + 1] - pSrc[x]) + fabs(pSrcF[x] - pSrc[x]);
    }
    return dst;
}

static Mat getG_c(Mat &src, float l)
{
    Mat dst(src.size(), src.type());
    for (register int y = 0; y < src.rows; y++) {
        float *pSrc = src.ptr<float>(y);
        float *pDst = dst.ptr<float>(y);
        for (register int x = 0; x < src.cols; x++)
            pDst[x] = 0.5f*l / sqrtf(l*l + pSrc[x]*pSrc[x]);
    }
    return dst;
}

static Mat getG_p(Mat &src, float l)
{
    Mat dst(src.size(), src.type());
    for (register int y = 0; y < src.rows; y++) {
        float *pSrc = src.ptr<float>(y);
        float *pDst = dst.ptr<float>(y);
        for (register int x = 0; x < src.cols; x++)
            pDst[x] = 0.5f*l*l / (l*l + pSrc[x]*pSrc[x]);
    }
    return dst;
}

void StereoVar::VariationalSolver(Mat &I1, Mat &I2, Mat &I2x, Mat &u, int level)
{
    register int n, x, y;
    float gl = 1, gr = 1, gu = 1, gd = 1, gc = 4;
    Mat g_c, g_p;
    Mat U;
    u.copyTo(U);

    int     N = nIt;
    float   l = lambda;
    float   Fi = fi;


    if (flags & USE_SMART_ID) {
        double scale = pow(pyrScale, (double) level) * (1 + pyrScale);
        N = (int) (N / scale);
    }

    double scale = pow(pyrScale, (double) level);
    Fi /= (float) scale;
    l *= (float) scale;

    int width   = u.cols - 1;
    int height  = u.rows - 1;
    for (n = 0; n < N; n++) {
        if (penalization != PENALIZATION_TICHONOV) {
            Mat gradient = getGradient(U);
            switch (penalization) {
                case PENALIZATION_CHARBONNIER:  g_c = getG_c(gradient, l); break;
                case PENALIZATION_PERONA_MALIK: g_p = getG_p(gradient, l); break;
            }
            gradient.release();
        }
        for (y = 1 ; y < height; y++) {
            float *pU   = U.ptr<float>(y);
            float *pUu  = U.ptr<float>(y + 1);
            float *pUd  = U.ptr<float>(y - 1);
            float *pu   = u.ptr<float>(y);
            float *pI1  = I1.ptr<float>(y);
            float *pI2  = I2.ptr<float>(y);
            float *pI2x = I2x.ptr<float>(y);
            float *pG_c = NULL, *pG_cu = NULL, *pG_cd = NULL;
            float *pG_p = NULL, *pG_pu = NULL, *pG_pd = NULL;
            switch (penalization) {
                case PENALIZATION_CHARBONNIER:
                    pG_c    = g_c.ptr<float>(y);
                    pG_cu   = g_c.ptr<float>(y + 1);
                    pG_cd   = g_c.ptr<float>(y - 1);
                    break;
                case PENALIZATION_PERONA_MALIK:
                    pG_p    = g_p.ptr<float>(y);
                    pG_pu   = g_p.ptr<float>(y + 1);
                    pG_pd   = g_p.ptr<float>(y - 1);
                    break;
            }
            for (x = 1; x < width; x++) {
                switch (penalization) {
                    case PENALIZATION_CHARBONNIER:
                        gc = pG_c[x];
                        gl = gc + pG_c[x - 1];
                        gr = gc + pG_c[x + 1];
                        gu = gc + pG_cu[x];
                        gd = gc + pG_cd[x];
                        gc = gl + gr + gu + gd;
                        break;
                    case PENALIZATION_PERONA_MALIK:
                        gc = pG_p[x];
                        gl = gc + pG_p[x - 1];
                        gr = gc + pG_p[x + 1];
                        gu = gc + pG_pu[x];
                        gd = gc + pG_pd[x];
                        gc = gl + gr + gu + gd;
                        break;
                }

                float _fi = Fi;
                if (maxDisp > minDisp) {
                    if (pU[x] > maxDisp * scale) {_fi *= 1000; pU[x] = static_cast<float>(maxDisp * scale);}
                    if (pU[x] < minDisp * scale) {_fi *= 1000; pU[x] = static_cast<float>(minDisp * scale);}
                }

                int A = static_cast<int>(pU[x]);
                int neg = 0; if (pU[x] <= 0) neg = -1;

                if (x + A > width)
                    pu[x] = pU[width - A];
                else if (x + A + neg < 0)
                    pu[x] = pU[- A + 2];
                else {
                    pu[x] = A + (pI2x[x + A + neg] * (pI1[x] - pI2[x + A])
                              + _fi * (gr * pU[x + 1] + gl * pU[x - 1] + gu * pUu[x] + gd * pUd[x] - gc * A))
                              / (pI2x[x + A + neg] * pI2x[x + A + neg] + gc * _fi) ;
                }
            }// x
            pu[0] = pu[1];
            pu[width] = pu[width - 1];
        }// y
        for (x = 0; x <= width; x++) {
            u.at<float>(0, x) = u.at<float>(1, x);
            u.at<float>(height, x) = u.at<float>(height - 1, x);
        }
        u.copyTo(U);
        if (!g_c.empty()) g_c.release();
        if (!g_p.empty()) g_p.release();
    }//n
}

void StereoVar::VCycle_MyFAS(Mat &I1, Mat &I2, Mat &I2x, Mat &_u, int level)
{
    CvSize imgSize = _u.size();
    CvSize frmSize = cvSize((int) (imgSize.width * pyrScale + 0.5), (int) (imgSize.height * pyrScale + 0.5));
    Mat I1_h, I2_h, I2x_h, u_h, U, U_h;

    //PRE relaxation
    VariationalSolver(I1, I2, I2x, _u, level);

    if (level >= levels - 1) return;
    level ++;

    //scaling DOWN
    resize(I1, I1_h, frmSize, 0, 0, INTER_AREA);
    resize(I2, I2_h, frmSize, 0, 0, INTER_AREA);
    resize(_u, u_h, frmSize, 0, 0, INTER_AREA);
    u_h.convertTo(u_h, u_h.type(), pyrScale);
    I2x_h = diffX(I2_h);

    //Next level
    U_h = u_h.clone();
    VCycle_MyFAS(I1_h, I2_h, I2x_h, U_h, level);

    subtract(U_h, u_h, U_h);
    U_h.convertTo(U_h, U_h.type(), 1.0 / pyrScale);

    //scaling UP
    resize(U_h, U, imgSize);

    //correcting the solution
    add(_u, U, _u);

    //POST relaxation
    VariationalSolver(I1, I2, I2x, _u, level - 1);

    if (flags & USE_MEDIAN_FILTERING) medianBlur(_u, _u, 3);

    I1_h.release();
    I2_h.release();
    I2x_h.release();
    u_h.release();
    U.release();
    U_h.release();
}

void StereoVar::FMG(Mat &I1, Mat &I2, Mat &I2x, Mat &u, int level)
{
    double  scale = pow(pyrScale, (double) level);
    CvSize  frmSize = cvSize((int) (u.cols * scale + 0.5), (int) (u.rows * scale + 0.5));
    Mat I1_h, I2_h, I2x_h, u_h;

    //scaling DOWN
    resize(I1, I1_h, frmSize, 0, 0, INTER_AREA);
    resize(I2, I2_h, frmSize, 0, 0, INTER_AREA);
    resize(u, u_h, frmSize, 0, 0, INTER_AREA);
    u_h.convertTo(u_h, u_h.type(), scale);
    I2x_h = diffX(I2_h);

    switch (cycle) {
        case CYCLE_O:
            VariationalSolver(I1_h, I2_h, I2x_h, u_h, level);
            break;
        case CYCLE_V:
            VCycle_MyFAS(I1_h, I2_h, I2x_h, u_h, level);
            break;
    }

    u_h.convertTo(u_h, u_h.type(), 1.0 / scale);

    //scaling UP
    resize(u_h, u, u.size(), 0, 0, INTER_CUBIC);

    I1_h.release();
    I2_h.release();
    I2x_h.release();
    u_h.release();

    level--;
    if ((flags & USE_AUTO_PARAMS) && (level < levels / 3)) {
        penalization = PENALIZATION_PERONA_MALIK;
        fi *= 100;
        flags -= USE_AUTO_PARAMS;
        autoParams();
    }
    if (flags & USE_MEDIAN_FILTERING) medianBlur(u, u, 3);
    if (level >= 0) FMG(I1, I2, I2x, u, level);
}

void StereoVar::autoParams()
{
    int maxD = MAX(labs(maxDisp), labs(minDisp));

    if (!maxD) pyrScale = 0.85;
    else if (maxD < 8) pyrScale = 0.5;
    else if (maxD < 64) pyrScale = 0.5 + static_cast<double>(maxD - 8) * 0.00625;
    else pyrScale = 0.85;

    if (maxD) {
        levels = 0;
        while ( pow(pyrScale, levels) * maxD > 1.5) levels ++;
        levels++;
    }

    switch(penalization) {
        case PENALIZATION_TICHONOV: cycle = CYCLE_V; break;
        case PENALIZATION_CHARBONNIER: cycle = CYCLE_O; break;
        case PENALIZATION_PERONA_MALIK: cycle = CYCLE_O; break;
    }
}

void StereoVar::operator ()( const Mat& left, const Mat& right, Mat& disp )
{
    CV_Assert(left.size() == right.size() && left.type() == right.type());
    CvSize imgSize = left.size();
    int MaxD = MAX(labs(minDisp), labs(maxDisp));
    int SignD = 1; if (MIN(minDisp, maxDisp) < 0) SignD = -1;
    if (minDisp >= maxDisp) {MaxD = 256; SignD = 1;}

    Mat u;
    if ((flags & USE_INITIAL_DISPARITY) && (!disp.empty())) {
        CV_Assert(disp.size() == left.size() && disp.type() == CV_8UC1);
        disp.convertTo(u, CV_32FC1, static_cast<double>(SignD * MaxD) / 256);
    } else {
        u.create(imgSize, CV_32FC1);
        u.setTo(0);
    }

    // Preprocessing
    Mat leftgray, rightgray;
    if (left.type() != CV_8UC1) {
        cvtColor(left, leftgray, CV_BGR2GRAY);
        cvtColor(right, rightgray, CV_BGR2GRAY);
    } else {
        left.copyTo(leftgray);
        right.copyTo(rightgray);
    }
    if (flags & USE_EQUALIZE_HIST) {
        equalizeHist(leftgray, leftgray);
        equalizeHist(rightgray, rightgray);
    }
    if (poly_sigma > 0.0001) {
        GaussianBlur(leftgray, leftgray, cvSize(poly_n, poly_n), poly_sigma);
        GaussianBlur(rightgray, rightgray, cvSize(poly_n, poly_n), poly_sigma);
    }

    if (flags & USE_AUTO_PARAMS) {
        penalization = PENALIZATION_TICHONOV;
        autoParams();
    }

    Mat I1, I2;
    leftgray.convertTo(I1, CV_32FC1);
    rightgray.convertTo(I2, CV_32FC1);
    leftgray.release();
    rightgray.release();

    Mat I2x = diffX(I2);

    FMG(I1, I2, I2x, u, levels - 1);

    I1.release();
    I2.release();
    I2x.release();


    disp.create( left.size(), CV_8UC1 );
    u = abs(u);
    u.convertTo(disp, disp.type(), 256 / MaxD, 0);

    u.release();
}
} // namespace
