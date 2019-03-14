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

#include "precomp.hpp"
#include "opencl_kernels_video.hpp"
#include "opencv2/core/hal/intrin.hpp"

#if defined __APPLE__ || defined __ANDROID__
#define SMALL_LOCALSIZE
#endif

//
// 2D dense optical flow algorithm from the following paper:
// Gunnar Farneback. "Two-Frame Motion Estimation Based on Polynomial Expansion".
// Proceedings of the 13th Scandinavian Conference on Image Analysis, Gothenburg, Sweden
//

namespace cv
{

static void
FarnebackPrepareGaussian(int n, double sigma, float *g, float *xg, float *xxg,
                         double &ig11, double &ig03, double &ig33, double &ig55)
{
    if( sigma < FLT_EPSILON )
        sigma = n*0.3;

    double s = 0.;
    for (int x = -n; x <= n; x++)
    {
        g[x] = (float)std::exp(-x*x/(2*sigma*sigma));
        s += g[x];
    }

    s = 1./s;
    for (int x = -n; x <= n; x++)
    {
        g[x] = (float)(g[x]*s);
        xg[x] = (float)(x*g[x]);
        xxg[x] = (float)(x*x*g[x]);
    }

    Mat_<double> G(6, 6);
    G.setTo(0);

    for (int y = -n; y <= n; y++)
    {
        for (int x = -n; x <= n; x++)
        {
            G(0,0) += g[y]*g[x];
            G(1,1) += g[y]*g[x]*x*x;
            G(3,3) += g[y]*g[x]*x*x*x*x;
            G(5,5) += g[y]*g[x]*x*x*y*y;
        }
    }

    //G[0][0] = 1.;
    G(2,2) = G(0,3) = G(0,4) = G(3,0) = G(4,0) = G(1,1);
    G(4,4) = G(3,3);
    G(3,4) = G(4,3) = G(5,5);

    // invG:
    // [ x        e  e    ]
    // [    y             ]
    // [       y          ]
    // [ e        z       ]
    // [ e           z    ]
    // [                u ]
    Mat_<double> invG = G.inv(DECOMP_CHOLESKY);

    ig11 = invG(1,1);
    ig03 = invG(0,3);
    ig33 = invG(3,3);
    ig55 = invG(5,5);
}

static void
FarnebackPolyExp( const Mat& src, Mat& dst, int n, double sigma )
{
    int k, x, y;

    CV_Assert( src.type() == CV_32FC1 );
    int width = src.cols;
    int height = src.rows;
    AutoBuffer<float> kbuf(n*6 + 3), _row((width + n*2)*3);
    float* g = kbuf.data() + n;
    float* xg = g + n*2 + 1;
    float* xxg = xg + n*2 + 1;
    float *row = _row.data() + n*3;
    double ig11, ig03, ig33, ig55;

    FarnebackPrepareGaussian(n, sigma, g, xg, xxg, ig11, ig03, ig33, ig55);

    dst.create( height, width, CV_32FC(5));

    for( y = 0; y < height; y++ )
    {
        float g0 = g[0], g1, g2;
        const float *srow0 = src.ptr<float>(y), *srow1 = 0;
        float *drow = dst.ptr<float>(y);

        // vertical part of convolution
        for( x = 0; x < width; x++ )
        {
            row[x*3] = srow0[x]*g0;
            row[x*3+1] = row[x*3+2] = 0.f;
        }

        for( k = 1; k <= n; k++ )
        {
            g0 = g[k]; g1 = xg[k]; g2 = xxg[k];
            srow0 = src.ptr<float>(std::max(y-k,0));
            srow1 = src.ptr<float>(std::min(y+k,height-1));

            for( x = 0; x < width; x++ )
            {
                float p = srow0[x] + srow1[x];
                float t0 = row[x*3] + g0*p;
                float t1 = row[x*3+1] + g1*(srow1[x] - srow0[x]);
                float t2 = row[x*3+2] + g2*p;

                row[x*3] = t0;
                row[x*3+1] = t1;
                row[x*3+2] = t2;
            }
        }

        // horizontal part of convolution
        for( x = 0; x < n*3; x++ )
        {
            row[-1-x] = row[2-x];
            row[width*3+x] = row[width*3+x-3];
        }

        for( x = 0; x < width; x++ )
        {
            g0 = g[0];
            // r1 ~ 1, r2 ~ x, r3 ~ y, r4 ~ x^2, r5 ~ y^2, r6 ~ xy
            double b1 = row[x*3]*g0, b2 = 0, b3 = row[x*3+1]*g0,
                b4 = 0, b5 = row[x*3+2]*g0, b6 = 0;

            for( k = 1; k <= n; k++ )
            {
                double tg = row[(x+k)*3] + row[(x-k)*3];
                g0 = g[k];
                b1 += tg*g0;
                b4 += tg*xxg[k];
                b2 += (row[(x+k)*3] - row[(x-k)*3])*xg[k];
                b3 += (row[(x+k)*3+1] + row[(x-k)*3+1])*g0;
                b6 += (row[(x+k)*3+1] - row[(x-k)*3+1])*xg[k];
                b5 += (row[(x+k)*3+2] + row[(x-k)*3+2])*g0;
            }

            // do not store r1
            drow[x*5+1] = (float)(b2*ig11);
            drow[x*5] = (float)(b3*ig11);
            drow[x*5+3] = (float)(b1*ig03 + b4*ig33);
            drow[x*5+2] = (float)(b1*ig03 + b5*ig33);
            drow[x*5+4] = (float)(b6*ig55);
        }
    }

    row -= n*3;
}


/*static void
FarnebackPolyExpPyr( const Mat& src0, Vector<Mat>& pyr, int maxlevel, int n, double sigma )
{
    Vector<Mat> imgpyr;
    buildPyramid( src0, imgpyr, maxlevel );

    for( int i = 0; i <= maxlevel; i++ )
        FarnebackPolyExp( imgpyr[i], pyr[i], n, sigma );
}*/


static void
FarnebackUpdateMatrices( const Mat& _R0, const Mat& _R1, const Mat& _flow, Mat& matM, int _y0, int _y1 )
{
    const int BORDER = 5;
    static const float border[BORDER] = {0.14f, 0.14f, 0.4472f, 0.4472f, 0.4472f};

    int x, y, width = _flow.cols, height = _flow.rows;
    const float* R1 = _R1.ptr<float>();
    size_t step1 = _R1.step/sizeof(R1[0]);

    matM.create(height, width, CV_32FC(5));

    for( y = _y0; y < _y1; y++ )
    {
        const float* flow = _flow.ptr<float>(y);
        const float* R0 = _R0.ptr<float>(y);
        float* M = matM.ptr<float>(y);

        for( x = 0; x < width; x++ )
        {
            float dx = flow[x*2], dy = flow[x*2+1];
            float fx = x + dx, fy = y + dy;

#if 1
            int x1 = cvFloor(fx), y1 = cvFloor(fy);
            const float* ptr = R1 + y1*step1 + x1*5;
            float r2, r3, r4, r5, r6;

            fx -= x1; fy -= y1;

            if( (unsigned)x1 < (unsigned)(width-1) &&
                (unsigned)y1 < (unsigned)(height-1) )
            {
                float a00 = (1.f-fx)*(1.f-fy), a01 = fx*(1.f-fy),
                      a10 = (1.f-fx)*fy, a11 = fx*fy;

                r2 = a00*ptr[0] + a01*ptr[5] + a10*ptr[step1] + a11*ptr[step1+5];
                r3 = a00*ptr[1] + a01*ptr[6] + a10*ptr[step1+1] + a11*ptr[step1+6];
                r4 = a00*ptr[2] + a01*ptr[7] + a10*ptr[step1+2] + a11*ptr[step1+7];
                r5 = a00*ptr[3] + a01*ptr[8] + a10*ptr[step1+3] + a11*ptr[step1+8];
                r6 = a00*ptr[4] + a01*ptr[9] + a10*ptr[step1+4] + a11*ptr[step1+9];

                r4 = (R0[x*5+2] + r4)*0.5f;
                r5 = (R0[x*5+3] + r5)*0.5f;
                r6 = (R0[x*5+4] + r6)*0.25f;
            }
#else
            int x1 = cvRound(fx), y1 = cvRound(fy);
            const float* ptr = R1 + y1*step1 + x1*5;
            float r2, r3, r4, r5, r6;

            if( (unsigned)x1 < (unsigned)width &&
                (unsigned)y1 < (unsigned)height )
            {
                r2 = ptr[0];
                r3 = ptr[1];
                r4 = (R0[x*5+2] + ptr[2])*0.5f;
                r5 = (R0[x*5+3] + ptr[3])*0.5f;
                r6 = (R0[x*5+4] + ptr[4])*0.25f;
            }
#endif
            else
            {
                r2 = r3 = 0.f;
                r4 = R0[x*5+2];
                r5 = R0[x*5+3];
                r6 = R0[x*5+4]*0.5f;
            }

            r2 = (R0[x*5] - r2)*0.5f;
            r3 = (R0[x*5+1] - r3)*0.5f;

            r2 += r4*dy + r6*dx;
            r3 += r6*dy + r5*dx;

            if( (unsigned)(x - BORDER) >= (unsigned)(width - BORDER*2) ||
                (unsigned)(y - BORDER) >= (unsigned)(height - BORDER*2))
            {
                float scale = (x < BORDER ? border[x] : 1.f)*
                    (x >= width - BORDER ? border[width - x - 1] : 1.f)*
                    (y < BORDER ? border[y] : 1.f)*
                    (y >= height - BORDER ? border[height - y - 1] : 1.f);

                r2 *= scale; r3 *= scale; r4 *= scale;
                r5 *= scale; r6 *= scale;
            }

            M[x*5]   = r4*r4 + r6*r6; // G(1,1)
            M[x*5+1] = (r4 + r5)*r6;  // G(1,2)=G(2,1)
            M[x*5+2] = r5*r5 + r6*r6; // G(2,2)
            M[x*5+3] = r4*r2 + r6*r3; // h(1)
            M[x*5+4] = r6*r2 + r5*r3; // h(2)
        }
    }
}


static void
FarnebackUpdateFlow_Blur( const Mat& _R0, const Mat& _R1,
                          Mat& _flow, Mat& matM, int block_size,
                          bool update_matrices )
{
    int x, y, width = _flow.cols, height = _flow.rows;
    int m = block_size/2;
    int y0 = 0, y1;
    int min_update_stripe = std::max((1 << 10)/width, block_size);
    double scale = 1./(block_size*block_size);

    AutoBuffer<double> _vsum((width+m*2+2)*5);
    double* vsum = _vsum.data() + (m+1)*5;

    // init vsum
    const float* srow0 = matM.ptr<float>();
    for( x = 0; x < width*5; x++ )
        vsum[x] = srow0[x]*(m+2);

    for( y = 1; y < m; y++ )
    {
        srow0 = matM.ptr<float>(std::min(y,height-1));
        for( x = 0; x < width*5; x++ )
            vsum[x] += srow0[x];
    }

    // compute blur(G)*flow=blur(h)
    for( y = 0; y < height; y++ )
    {
        double g11, g12, g22, h1, h2;
        float* flow = _flow.ptr<float>(y);

        srow0 = matM.ptr<float>(std::max(y-m-1,0));
        const float* srow1 = matM.ptr<float>(std::min(y+m,height-1));

        // vertical blur
        for( x = 0; x < width*5; x++ )
            vsum[x] += srow1[x] - srow0[x];

        // update borders
        for( x = 0; x < (m+1)*5; x++ )
        {
            vsum[-1-x] = vsum[4-x];
            vsum[width*5+x] = vsum[width*5+x-5];
        }

        // init g** and h*
        g11 = vsum[0]*(m+2);
        g12 = vsum[1]*(m+2);
        g22 = vsum[2]*(m+2);
        h1 = vsum[3]*(m+2);
        h2 = vsum[4]*(m+2);

        for( x = 1; x < m; x++ )
        {
            g11 += vsum[x*5];
            g12 += vsum[x*5+1];
            g22 += vsum[x*5+2];
            h1 += vsum[x*5+3];
            h2 += vsum[x*5+4];
        }

        // horizontal blur
        for( x = 0; x < width; x++ )
        {
            g11 += vsum[(x+m)*5] - vsum[(x-m)*5 - 5];
            g12 += vsum[(x+m)*5 + 1] - vsum[(x-m)*5 - 4];
            g22 += vsum[(x+m)*5 + 2] - vsum[(x-m)*5 - 3];
            h1 += vsum[(x+m)*5 + 3] - vsum[(x-m)*5 - 2];
            h2 += vsum[(x+m)*5 + 4] - vsum[(x-m)*5 - 1];

            double g11_ = g11*scale;
            double g12_ = g12*scale;
            double g22_ = g22*scale;
            double h1_ = h1*scale;
            double h2_ = h2*scale;

            double idet = 1./(g11_*g22_ - g12_*g12_+1e-3);

            flow[x*2] = (float)((g11_*h2_-g12_*h1_)*idet);
            flow[x*2+1] = (float)((g22_*h1_-g12_*h2_)*idet);
        }

        y1 = y == height - 1 ? height : y - block_size;
        if( update_matrices && (y1 == height || y1 >= y0 + min_update_stripe) )
        {
            FarnebackUpdateMatrices( _R0, _R1, _flow, matM, y0, y1 );
            y0 = y1;
        }
    }
}


static void
FarnebackUpdateFlow_GaussianBlur( const Mat& _R0, const Mat& _R1,
                                  Mat& _flow, Mat& matM, int block_size,
                                  bool update_matrices )
{
    int x, y, i, width = _flow.cols, height = _flow.rows;
    int m = block_size/2;
    int y0 = 0, y1;
    int min_update_stripe = std::max((1 << 10)/width, block_size);
    double sigma = m*0.3, s = 1;

    AutoBuffer<float> _vsum((width+m*2+2)*5 + 16), _hsum(width*5 + 16);
    AutoBuffer<float> _kernel((m+1)*5 + 16);
    AutoBuffer<const float*> _srow(m*2+1);
    float *vsum = alignPtr(_vsum.data() + (m+1)*5, 16), *hsum = alignPtr(_hsum.data(), 16);
    float* kernel = _kernel.data();
    const float** srow = _srow.data();
    kernel[0] = (float)s;

    for( i = 1; i <= m; i++ )
    {
        float t = (float)std::exp(-i*i/(2*sigma*sigma) );
        kernel[i] = t;
        s += t*2;
    }

    s = 1./s;
    for( i = 0; i <= m; i++ )
        kernel[i] = (float)(kernel[i]*s);

#if CV_SIMD128
    float* simd_kernel = alignPtr(kernel + m+1, 16);
    {
        for( i = 0; i <= m; i++ )
            v_store(simd_kernel + i*4, v_setall_f32(kernel[i]));
    }
#endif

    // compute blur(G)*flow=blur(h)
    for( y = 0; y < height; y++ )
    {
        double g11, g12, g22, h1, h2;
        float* flow = _flow.ptr<float>(y);

        // vertical blur
        for( i = 0; i <= m; i++ )
        {
            srow[m-i] = matM.ptr<float>(std::max(y-i,0));
            srow[m+i] = matM.ptr<float>(std::min(y+i,height-1));
        }

        x = 0;
#if CV_SIMD128
        {
            for( ; x <= width*5 - 16; x += 16 )
            {
                const float *sptr0 = srow[m], *sptr1;
                v_float32x4 g4 = v_load(simd_kernel);
                v_float32x4 s0, s1, s2, s3;
                s0 = v_load(sptr0 + x) * g4;
                s1 = v_load(sptr0 + x + 4) * g4;
                s2 = v_load(sptr0 + x + 8) * g4;
                s3 = v_load(sptr0 + x + 12) * g4;

                for( i = 1; i <= m; i++ )
                {
                    v_float32x4 x0, x1;
                    sptr0 = srow[m+i], sptr1 = srow[m-i];
                    g4 = v_load(simd_kernel + i*4);
                    x0 = v_load(sptr0 + x) + v_load(sptr1 + x);
                    x1 = v_load(sptr0 + x + 4) + v_load(sptr1 + x + 4);
                    s0 = v_muladd(x0, g4, s0);
                    s1 = v_muladd(x1, g4, s1);
                    x0 = v_load(sptr0 + x + 8) + v_load(sptr1 + x + 8);
                    x1 = v_load(sptr0 + x + 12) + v_load(sptr1 + x + 12);
                    s2 = v_muladd(x0, g4, s2);
                    s3 = v_muladd(x1, g4, s3);
                }

                v_store(vsum + x, s0);
                v_store(vsum + x + 4, s1);
                v_store(vsum + x + 8, s2);
                v_store(vsum + x + 12, s3);
            }

            for( ; x <= width*5 - 4; x += 4 )
            {
                const float *sptr0 = srow[m], *sptr1;
                v_float32x4 g4 = v_load(simd_kernel);
                v_float32x4 s0 = v_load(sptr0 + x) * g4;

                for( i = 1; i <= m; i++ )
                {
                    sptr0 = srow[m+i], sptr1 = srow[m-i];
                    g4 = v_load(simd_kernel + i*4);
                    v_float32x4 x0 = v_load(sptr0 + x) + v_load(sptr1 + x);
                    s0 = v_muladd(x0, g4, s0);
                }
                v_store(vsum + x, s0);
            }
        }
#endif
        for( ; x < width*5; x++ )
        {
            float s0 = srow[m][x]*kernel[0];
            for( i = 1; i <= m; i++ )
                s0 += (srow[m+i][x] + srow[m-i][x])*kernel[i];
            vsum[x] = s0;
        }

        // update borders
        for( x = 0; x < m*5; x++ )
        {
            vsum[-1-x] = vsum[4-x];
            vsum[width*5+x] = vsum[width*5+x-5];
        }

        // horizontal blur
        x = 0;
#if CV_SIMD128
        {
            for( ; x <= width*5 - 8; x += 8 )
            {
                v_float32x4 g4 = v_load(simd_kernel);
                v_float32x4 s0 = v_load(vsum + x) * g4;
                v_float32x4 s1 = v_load(vsum + x + 4) * g4;

                for( i = 1; i <= m; i++ )
                {
                    g4 = v_load(simd_kernel + i*4);
                    v_float32x4 x0 = v_load(vsum + x - i*5) + v_load(vsum + x+ i*5);
                    v_float32x4 x1 = v_load(vsum + x - i*5 + 4) + v_load(vsum + x+ i*5 + 4);
                    s0 = v_muladd(x0, g4, s0);
                    s1 = v_muladd(x1, g4, s1);
                }

                v_store(hsum + x, s0);
                v_store(hsum + x + 4, s1);
            }
        }
#endif
        for( ; x < width*5; x++ )
        {
            float sum = vsum[x]*kernel[0];
            for( i = 1; i <= m; i++ )
                sum += kernel[i]*(vsum[x - i*5] + vsum[x + i*5]);
            hsum[x] = sum;
        }

        for( x = 0; x < width; x++ )
        {
            g11 = hsum[x*5];
            g12 = hsum[x*5+1];
            g22 = hsum[x*5+2];
            h1 = hsum[x*5+3];
            h2 = hsum[x*5+4];

            double idet = 1./(g11*g22 - g12*g12 + 1e-3);

            flow[x*2] = (float)((g11*h2-g12*h1)*idet);
            flow[x*2+1] = (float)((g22*h1-g12*h2)*idet);
        }

        y1 = y == height - 1 ? height : y - block_size;
        if( update_matrices && (y1 == height || y1 >= y0 + min_update_stripe) )
        {
            FarnebackUpdateMatrices( _R0, _R1, _flow, matM, y0, y1 );
            y0 = y1;
        }
    }
}

}

namespace cv
{
namespace
{
class FarnebackOpticalFlowImpl : public FarnebackOpticalFlow
{
public:
    FarnebackOpticalFlowImpl(int numLevels=5, double pyrScale=0.5, bool fastPyramids=false, int winSize=13,
                             int numIters=10, int polyN=5, double polySigma=1.1, int flags=0) :
        numLevels_(numLevels), pyrScale_(pyrScale), fastPyramids_(fastPyramids), winSize_(winSize),
        numIters_(numIters), polyN_(polyN), polySigma_(polySigma), flags_(flags)
    {
    }

    virtual int getNumLevels() const CV_OVERRIDE { return numLevels_; }
    virtual void setNumLevels(int numLevels) CV_OVERRIDE { numLevels_ = numLevels; }

    virtual double getPyrScale() const CV_OVERRIDE { return pyrScale_; }
    virtual void setPyrScale(double pyrScale) CV_OVERRIDE { pyrScale_ = pyrScale; }

    virtual bool getFastPyramids() const CV_OVERRIDE { return fastPyramids_; }
    virtual void setFastPyramids(bool fastPyramids) CV_OVERRIDE { fastPyramids_ = fastPyramids; }

    virtual int getWinSize() const CV_OVERRIDE { return winSize_; }
    virtual void setWinSize(int winSize) CV_OVERRIDE { winSize_ = winSize; }

    virtual int getNumIters() const CV_OVERRIDE { return numIters_; }
    virtual void setNumIters(int numIters) CV_OVERRIDE { numIters_ = numIters; }

    virtual int getPolyN() const CV_OVERRIDE { return polyN_; }
    virtual void setPolyN(int polyN) CV_OVERRIDE { polyN_ = polyN; }

    virtual double getPolySigma() const CV_OVERRIDE { return polySigma_; }
    virtual void setPolySigma(double polySigma) CV_OVERRIDE { polySigma_ = polySigma; }

    virtual int getFlags() const CV_OVERRIDE { return flags_; }
    virtual void setFlags(int flags) CV_OVERRIDE { flags_ = flags; }

    virtual void calc(InputArray I0, InputArray I1, InputOutputArray flow) CV_OVERRIDE;

private:
    int numLevels_;
    double pyrScale_;
    bool fastPyramids_;
    int winSize_;
    int numIters_;
    int polyN_;
    double polySigma_;
    int flags_;

#ifdef HAVE_OPENCL
    bool operator ()(const UMat &frame0, const UMat &frame1, UMat &flowx, UMat &flowy)
    {
        CV_Assert(frame0.channels() == 1 && frame1.channels() == 1);
        CV_Assert(frame0.size() == frame1.size());
        CV_Assert(polyN_ == 5 || polyN_ == 7);
        CV_Assert(!fastPyramids_ || std::abs(pyrScale_ - 0.5) < 1e-6);

        const int min_size = 32;

        Size size = frame0.size();
        UMat prevFlowX, prevFlowY, curFlowX, curFlowY;

        UMat flowx0 = flowx;
        UMat flowy0 = flowy;

        // Crop unnecessary levels
        double scale = 1;
        int numLevelsCropped = 0;
        for (; numLevelsCropped < numLevels_; numLevelsCropped++)
        {
            scale *= pyrScale_;
            if (size.width*scale < min_size || size.height*scale < min_size)
                break;
        }

        frame0.convertTo(frames_[0], CV_32F);
        frame1.convertTo(frames_[1], CV_32F);

        if (fastPyramids_)
        {
            // Build Gaussian pyramids using pyrDown()
            pyramid0_.resize(numLevelsCropped + 1);
            pyramid1_.resize(numLevelsCropped + 1);
            pyramid0_[0] = frames_[0];
            pyramid1_[0] = frames_[1];
            for (int i = 1; i <= numLevelsCropped; ++i)
            {
                pyrDown(pyramid0_[i - 1], pyramid0_[i]);
                pyrDown(pyramid1_[i - 1], pyramid1_[i]);
            }
        }

        setPolynomialExpansionConsts(polyN_, polySigma_);

        for (int k = numLevelsCropped; k >= 0; k--)
        {
            scale = 1;
            for (int i = 0; i < k; i++)
                scale *= pyrScale_;

            double sigma = (1./scale - 1) * 0.5;
            int smoothSize = cvRound(sigma*5) | 1;
            smoothSize = std::max(smoothSize, 3);

            int width = cvRound(size.width*scale);
            int height = cvRound(size.height*scale);

            if (fastPyramids_)
            {
                width = pyramid0_[k].cols;
                height = pyramid0_[k].rows;
            }

            if (k > 0)
            {
                curFlowX.create(height, width, CV_32F);
                curFlowY.create(height, width, CV_32F);
            }
            else
            {
                curFlowX = flowx0;
                curFlowY = flowy0;
            }

            if (prevFlowX.empty())
            {
                if (flags_ & cv::OPTFLOW_USE_INITIAL_FLOW)
                {
                    resize(flowx0, curFlowX, Size(width, height), 0, 0, INTER_LINEAR);
                    resize(flowy0, curFlowY, Size(width, height), 0, 0, INTER_LINEAR);
                    multiply(scale, curFlowX, curFlowX);
                    multiply(scale, curFlowY, curFlowY);
                }
                else
                {
                    curFlowX.setTo(0);
                    curFlowY.setTo(0);
                }
            }
            else
            {
                resize(prevFlowX, curFlowX, Size(width, height), 0, 0, INTER_LINEAR);
                resize(prevFlowY, curFlowY, Size(width, height), 0, 0, INTER_LINEAR);
                multiply(1./pyrScale_, curFlowX, curFlowX);
                multiply(1./pyrScale_, curFlowY, curFlowY);
            }

            UMat M = allocMatFromBuf(5*height, width, CV_32F, M_);
            UMat bufM = allocMatFromBuf(5*height, width, CV_32F, bufM_);
            UMat R[2] =
            {
                allocMatFromBuf(5*height, width, CV_32F, R_[0]),
                allocMatFromBuf(5*height, width, CV_32F, R_[1])
            };

            if (fastPyramids_)
            {
                if (!polynomialExpansionOcl(pyramid0_[k], R[0]))
                    return false;
                if (!polynomialExpansionOcl(pyramid1_[k], R[1]))
                    return false;
            }
            else
            {
                UMat blurredFrame[2] =
                {
                    allocMatFromBuf(size.height, size.width, CV_32F, blurredFrame_[0]),
                    allocMatFromBuf(size.height, size.width, CV_32F, blurredFrame_[1])
                };
                UMat pyrLevel[2] =
                {
                    allocMatFromBuf(height, width, CV_32F, pyrLevel_[0]),
                    allocMatFromBuf(height, width, CV_32F, pyrLevel_[1])
                };

                setGaussianBlurKernel(smoothSize, sigma);

                for (int i = 0; i < 2; i++)
                {
                    if (!gaussianBlurOcl(frames_[i], smoothSize/2, blurredFrame[i]))
                        return false;
                    resize(blurredFrame[i], pyrLevel[i], Size(width, height), INTER_LINEAR);
                    if (!polynomialExpansionOcl(pyrLevel[i], R[i]))
                        return false;
                }
            }

            if (!updateMatricesOcl(curFlowX, curFlowY, R[0], R[1], M))
                return false;

            if (flags_ & OPTFLOW_FARNEBACK_GAUSSIAN)
                setGaussianBlurKernel(winSize_, winSize_/2*0.3f);
            for (int i = 0; i < numIters_; i++)
            {
                if (flags_ & OPTFLOW_FARNEBACK_GAUSSIAN)
                {
                    if (!updateFlow_gaussianBlur(R[0], R[1], curFlowX, curFlowY, M, bufM, winSize_, i < numIters_-1))
                        return false;
                }
                else
                {
                    if (!updateFlow_boxFilter(R[0], R[1], curFlowX, curFlowY, M, bufM, winSize_, i < numIters_-1))
                        return false;
                }
            }

            prevFlowX = curFlowX;
            prevFlowY = curFlowY;
        }

        flowx = curFlowX;
        flowy = curFlowY;
        return true;
    }
    virtual void collectGarbage() CV_OVERRIDE {
        releaseMemory();
    }
    void releaseMemory()
    {
        frames_[0].release();
        frames_[1].release();
        pyrLevel_[0].release();
        pyrLevel_[1].release();
        M_.release();
        bufM_.release();
        R_[0].release();
        R_[1].release();
        blurredFrame_[0].release();
        blurredFrame_[1].release();
        pyramid0_.clear();
        pyramid1_.clear();
    }
private:
    UMat m_g;
    UMat m_xg;
    UMat m_xxg;

    double m_igd[4];
    float  m_ig[4];
    void setPolynomialExpansionConsts(int n, double sigma)
    {
        std::vector<float> buf(n*6 + 3);
        float* g = &buf[0] + n;
        float* xg = g + n*2 + 1;
        float* xxg = xg + n*2 + 1;

        FarnebackPrepareGaussian(n, sigma, g, xg, xxg, m_igd[0], m_igd[1], m_igd[2], m_igd[3]);

        cv::Mat t_g(1, n + 1, CV_32FC1, g);     t_g.copyTo(m_g);
        cv::Mat t_xg(1, n + 1, CV_32FC1, xg);   t_xg.copyTo(m_xg);
        cv::Mat t_xxg(1, n + 1, CV_32FC1, xxg); t_xxg.copyTo(m_xxg);

        m_ig[0] = static_cast<float>(m_igd[0]);
        m_ig[1] = static_cast<float>(m_igd[1]);
        m_ig[2] = static_cast<float>(m_igd[2]);
        m_ig[3] = static_cast<float>(m_igd[3]);
    }
private:
    UMat m_gKer;
    inline void setGaussianBlurKernel(int smoothSize, double sigma)
    {
        Mat g = getGaussianKernel(smoothSize, sigma, CV_32F);
        Mat gKer(1, smoothSize/2 + 1, CV_32FC1, g.ptr<float>(smoothSize/2));
        gKer.copyTo(m_gKer);
    }
private:
    UMat frames_[2];
    UMat pyrLevel_[2], M_, bufM_, R_[2], blurredFrame_[2];
    std::vector<UMat> pyramid0_, pyramid1_;

    static UMat allocMatFromBuf(int rows, int cols, int type, UMat &mat)
    {
        if (!mat.empty() && mat.type() == type && mat.rows >= rows && mat.cols >= cols)
            return mat(Rect(0, 0, cols, rows));
        return mat = UMat(rows, cols, type);
    }
private:
#define DIVUP(total, grain) (((total) + (grain) - 1) / (grain))

    bool gaussianBlurOcl(const UMat &src, int ksizeHalf, UMat &dst)
    {
#ifdef SMALL_LOCALSIZE
        size_t localsize[2] = { 128, 1};
#else
        size_t localsize[2] = { 256, 1};
#endif
        size_t globalsize[2] = { (size_t)src.cols, (size_t)src.rows};
        int smem_size = (int)((localsize[0] + 2*ksizeHalf) * sizeof(float));
        ocl::Kernel kernel;
        if (!kernel.create("gaussianBlur", cv::ocl::video::optical_flow_farneback_oclsrc, ""))
            return false;

        CV_Assert(dst.size() == src.size());
        int idxArg = 0;
        idxArg = kernel.set(idxArg, ocl::KernelArg::PtrReadOnly(src));
        idxArg = kernel.set(idxArg, (int)(src.step / src.elemSize()));
        idxArg = kernel.set(idxArg, ocl::KernelArg::PtrWriteOnly(dst));
        idxArg = kernel.set(idxArg, (int)(dst.step / dst.elemSize()));
        idxArg = kernel.set(idxArg, dst.rows);
        idxArg = kernel.set(idxArg, dst.cols);
        idxArg = kernel.set(idxArg, ocl::KernelArg::PtrReadOnly(m_gKer));
        idxArg = kernel.set(idxArg, (int)ksizeHalf);
        kernel.set(idxArg, (void *)NULL, smem_size);
        return kernel.run(2, globalsize, localsize, false);
    }
    bool gaussianBlur5Ocl(const UMat &src, int ksizeHalf, UMat &dst)
    {
        int height = src.rows / 5;
#ifdef SMALL_LOCALSIZE
        size_t localsize[2] = { 128, 1};
#else
        size_t localsize[2] = { 256, 1};
#endif
        size_t globalsize[2] = { (size_t)src.cols, (size_t)height};
        int smem_size = (int)((localsize[0] + 2*ksizeHalf) * 5 * sizeof(float));
        ocl::Kernel kernel;
        if (!kernel.create("gaussianBlur5", cv::ocl::video::optical_flow_farneback_oclsrc, ""))
            return false;

        int idxArg = 0;
        idxArg = kernel.set(idxArg, ocl::KernelArg::PtrReadOnly(src));
        idxArg = kernel.set(idxArg, (int)(src.step / src.elemSize()));
        idxArg = kernel.set(idxArg, ocl::KernelArg::PtrWriteOnly(dst));
        idxArg = kernel.set(idxArg, (int)(dst.step / dst.elemSize()));
        idxArg = kernel.set(idxArg, height);
        idxArg = kernel.set(idxArg, src.cols);
        idxArg = kernel.set(idxArg, ocl::KernelArg::PtrReadOnly(m_gKer));
        idxArg = kernel.set(idxArg, (int)ksizeHalf);
        kernel.set(idxArg, (void *)NULL, smem_size);
        return kernel.run(2, globalsize, localsize, false);
    }
    bool polynomialExpansionOcl(const UMat &src, UMat &dst)
    {
#ifdef SMALL_LOCALSIZE
        size_t localsize[2] = { 128, 1};
#else
        size_t localsize[2] = { 256, 1};
#endif
        size_t globalsize[2] = { DIVUP((size_t)src.cols, localsize[0] - 2*polyN_) * localsize[0], (size_t)src.rows};

#if 0
        const cv::ocl::Device &device = cv::ocl::Device::getDefault();
        bool useDouble = (0 != device.doubleFPConfig());

        cv::String build_options = cv::format("-D polyN=%d -D USE_DOUBLE=%d", polyN_, useDouble ? 1 : 0);
#else
        cv::String build_options = cv::format("-D polyN=%d", polyN_);
#endif
        ocl::Kernel kernel;
        if (!kernel.create("polynomialExpansion", cv::ocl::video::optical_flow_farneback_oclsrc, build_options))
            return false;

        int smem_size = (int)(3 * localsize[0] * sizeof(float));
        int idxArg = 0;
        idxArg = kernel.set(idxArg, ocl::KernelArg::PtrReadOnly(src));
        idxArg = kernel.set(idxArg, (int)(src.step / src.elemSize()));
        idxArg = kernel.set(idxArg, ocl::KernelArg::PtrWriteOnly(dst));
        idxArg = kernel.set(idxArg, (int)(dst.step / dst.elemSize()));
        idxArg = kernel.set(idxArg, src.rows);
        idxArg = kernel.set(idxArg, src.cols);
        idxArg = kernel.set(idxArg, ocl::KernelArg::PtrReadOnly(m_g));
        idxArg = kernel.set(idxArg, ocl::KernelArg::PtrReadOnly(m_xg));
        idxArg = kernel.set(idxArg, ocl::KernelArg::PtrReadOnly(m_xxg));
        idxArg = kernel.set(idxArg, (void *)NULL, smem_size);
        kernel.set(idxArg, (void *)m_ig, 4 * sizeof(float));
        return kernel.run(2, globalsize, localsize, false);
    }
    bool boxFilter5Ocl(const UMat &src, int ksizeHalf, UMat &dst)
    {
        int height = src.rows / 5;
#ifdef SMALL_LOCALSIZE
        size_t localsize[2] = { 128, 1};
#else
        size_t localsize[2] = { 256, 1};
#endif
        size_t globalsize[2] = { (size_t)src.cols, (size_t)height};

        ocl::Kernel kernel;
        if (!kernel.create("boxFilter5", cv::ocl::video::optical_flow_farneback_oclsrc, ""))
            return false;

        int smem_size = (int)((localsize[0] + 2*ksizeHalf) * 5 * sizeof(float));

        int idxArg = 0;
        idxArg = kernel.set(idxArg, ocl::KernelArg::PtrReadOnly(src));
        idxArg = kernel.set(idxArg, (int)(src.step / src.elemSize()));
        idxArg = kernel.set(idxArg, ocl::KernelArg::PtrWriteOnly(dst));
        idxArg = kernel.set(idxArg, (int)(dst.step / dst.elemSize()));
        idxArg = kernel.set(idxArg, height);
        idxArg = kernel.set(idxArg, src.cols);
        idxArg = kernel.set(idxArg, (int)ksizeHalf);
        kernel.set(idxArg, (void *)NULL, smem_size);
        return kernel.run(2, globalsize, localsize, false);
    }

    bool updateFlowOcl(const UMat &M, UMat &flowx, UMat &flowy)
    {
#ifdef SMALL_LOCALSIZE
        size_t localsize[2] = { 32, 4};
#else
        size_t localsize[2] = { 32, 8};
#endif
        size_t globalsize[2] = { (size_t)flowx.cols, (size_t)flowx.rows};

        ocl::Kernel kernel;
        if (!kernel.create("updateFlow", cv::ocl::video::optical_flow_farneback_oclsrc, ""))
            return false;

        int idxArg = 0;
        idxArg = kernel.set(idxArg, ocl::KernelArg::PtrWriteOnly(M));
        idxArg = kernel.set(idxArg, (int)(M.step / M.elemSize()));
        idxArg = kernel.set(idxArg, ocl::KernelArg::PtrReadOnly(flowx));
        idxArg = kernel.set(idxArg, (int)(flowx.step / flowx.elemSize()));
        idxArg = kernel.set(idxArg, ocl::KernelArg::PtrReadOnly(flowy));
        idxArg = kernel.set(idxArg, (int)(flowy.step / flowy.elemSize()));
        idxArg = kernel.set(idxArg, (int)flowy.rows);
        kernel.set(idxArg, (int)flowy.cols);
        return kernel.run(2, globalsize, localsize, false);
    }
    bool updateMatricesOcl(const UMat &flowx, const UMat &flowy, const UMat &R0, const UMat &R1, UMat &M)
    {
#ifdef SMALL_LOCALSIZE
        size_t localsize[2] = { 32, 4};
#else
        size_t localsize[2] = { 32, 8};
#endif
        size_t globalsize[2] = { (size_t)flowx.cols, (size_t)flowx.rows};

        ocl::Kernel kernel;
        if (!kernel.create("updateMatrices", cv::ocl::video::optical_flow_farneback_oclsrc, ""))
            return false;

        int idxArg = 0;
        idxArg = kernel.set(idxArg, ocl::KernelArg::PtrReadOnly(flowx));
        idxArg = kernel.set(idxArg, (int)(flowx.step / flowx.elemSize()));
        idxArg = kernel.set(idxArg, ocl::KernelArg::PtrReadOnly(flowy));
        idxArg = kernel.set(idxArg, (int)(flowy.step / flowy.elemSize()));
        idxArg = kernel.set(idxArg, (int)flowx.rows);
        idxArg = kernel.set(idxArg, (int)flowx.cols);
        idxArg = kernel.set(idxArg, ocl::KernelArg::PtrReadOnly(R0));
        idxArg = kernel.set(idxArg, (int)(R0.step / R0.elemSize()));
        idxArg = kernel.set(idxArg, ocl::KernelArg::PtrReadOnly(R1));
        idxArg = kernel.set(idxArg, (int)(R1.step / R1.elemSize()));
        idxArg = kernel.set(idxArg, ocl::KernelArg::PtrWriteOnly(M));
        kernel.set(idxArg, (int)(M.step / M.elemSize()));
        return kernel.run(2, globalsize, localsize, false);
    }

    bool updateFlow_boxFilter(
        const UMat& R0, const UMat& R1, UMat& flowx, UMat &flowy,
        UMat& M, UMat &bufM, int blockSize, bool updateMatrices)
    {
        if (!boxFilter5Ocl(M, blockSize/2, bufM))
            return false;
        swap(M, bufM);
        if (!updateFlowOcl(M, flowx, flowy))
            return false;
        if (updateMatrices)
            if (!updateMatricesOcl(flowx, flowy, R0, R1, M))
                return false;
        return true;
    }
    bool updateFlow_gaussianBlur(
        const UMat& R0, const UMat& R1, UMat& flowx, UMat& flowy,
        UMat& M, UMat &bufM, int blockSize, bool updateMatrices)
    {
        if (!gaussianBlur5Ocl(M, blockSize/2, bufM))
            return false;
        swap(M, bufM);
        if (!updateFlowOcl(M, flowx, flowy))
            return false;
        if (updateMatrices)
            if (!updateMatricesOcl(flowx, flowy, R0, R1, M))
                return false;
        return true;
    }
    bool calc_ocl( InputArray _prev0, InputArray _next0,
                   InputOutputArray _flow0)
    {
        if ((5 != polyN_) && (7 != polyN_))
            return false;
        if (_next0.size() != _prev0.size())
            return false;
        int typePrev = _prev0.type();
        int typeNext = _next0.type();
        if ((1 != CV_MAT_CN(typePrev)) || (1 != CV_MAT_CN(typeNext)))
            return false;

        std::vector<UMat> flowar;

        // If flag is set, check for integrity; if not set, allocate memory space
        if (flags_ & OPTFLOW_USE_INITIAL_FLOW)
        {
            if (_flow0.empty() || _flow0.size() != _prev0.size() || _flow0.channels() != 2 ||
                _flow0.depth() != CV_32F)
                return false;
            split(_flow0, flowar);
        }
        else
        {
            flowar.push_back(UMat(_prev0.size(), CV_32FC1));
            flowar.push_back(UMat(_prev0.size(), CV_32FC1));
        }
        if(!this->operator()(_prev0.getUMat(), _next0.getUMat(), flowar[0], flowar[1])){
            return false;
        }
        merge(flowar, _flow0);
        return true;
    }
#else // HAVE_OPENCL
    virtual void collectGarbage() CV_OVERRIDE {}
#endif
};

void FarnebackOpticalFlowImpl::calc(InputArray _prev0, InputArray _next0,
                                    InputOutputArray _flow0)
{
    CV_INSTRUMENT_REGION();

    CV_OCL_RUN(_flow0.isUMat() &&
               ocl::Image2D::isFormatSupported(CV_32F, 1, false),
               calc_ocl(_prev0,_next0,_flow0))
    Mat prev0 = _prev0.getMat(), next0 = _next0.getMat();
    const int min_size = 32;
    const Mat* img[2] = { &prev0, &next0 };

    int i, k;
    double scale;
    Mat prevFlow, flow, fimg;
    int levels = numLevels_;

    CV_Assert( prev0.size() == next0.size() && prev0.channels() == next0.channels() &&
               prev0.channels() == 1 && pyrScale_ < 1 );

    // If flag is set, check for integrity; if not set, allocate memory space
    if( flags_ & OPTFLOW_USE_INITIAL_FLOW )
        CV_Assert( _flow0.size() == prev0.size() && _flow0.channels() == 2 &&
                   _flow0.depth() == CV_32F );
    else
        _flow0.create( prev0.size(), CV_32FC2 );

    Mat flow0 = _flow0.getMat();

    for( k = 0, scale = 1; k < levels; k++ )
    {
        scale *= pyrScale_;
        if( prev0.cols*scale < min_size || prev0.rows*scale < min_size )
            break;
    }

    levels = k;

    for( k = levels; k >= 0; k-- )
    {
        for( i = 0, scale = 1; i < k; i++ )
            scale *= pyrScale_;

        double sigma = (1./scale-1)*0.5;
        int smooth_sz = cvRound(sigma*5)|1;
        smooth_sz = std::max(smooth_sz, 3);

        int width = cvRound(prev0.cols*scale);
        int height = cvRound(prev0.rows*scale);

        if( k > 0 )
            flow.create( height, width, CV_32FC2 );
        else
            flow = flow0;

        if( prevFlow.empty() )
        {
            if( flags_ & OPTFLOW_USE_INITIAL_FLOW )
            {
                resize( flow0, flow, Size(width, height), 0, 0, INTER_AREA );
                flow *= scale;
            }
            else
                flow = Mat::zeros( height, width, CV_32FC2 );
        }
        else
        {
            resize( prevFlow, flow, Size(width, height), 0, 0, INTER_LINEAR );
            flow *= 1./pyrScale_;
        }

        Mat R[2], I, M;
        for( i = 0; i < 2; i++ )
        {
            img[i]->convertTo(fimg, CV_32F);
            GaussianBlur(fimg, fimg, Size(smooth_sz, smooth_sz), sigma, sigma);
            resize( fimg, I, Size(width, height), INTER_LINEAR );
            FarnebackPolyExp( I, R[i], polyN_, polySigma_ );
        }

        FarnebackUpdateMatrices( R[0], R[1], flow, M, 0, flow.rows );

        for( i = 0; i < numIters_; i++ )
        {
            if( flags_ & OPTFLOW_FARNEBACK_GAUSSIAN )
                FarnebackUpdateFlow_GaussianBlur( R[0], R[1], flow, M, winSize_, i < numIters_ - 1 );
            else
                FarnebackUpdateFlow_Blur( R[0], R[1], flow, M, winSize_, i < numIters_ - 1 );
        }

        prevFlow = flow;
    }
}
} // namespace
} // namespace cv

void cv::calcOpticalFlowFarneback( InputArray _prev0, InputArray _next0,
                               InputOutputArray _flow0, double pyr_scale, int levels, int winsize,
                               int iterations, int poly_n, double poly_sigma, int flags )
{
    CV_INSTRUMENT_REGION();

    Ptr<cv::FarnebackOpticalFlow> optflow;
    optflow = makePtr<FarnebackOpticalFlowImpl>(levels,pyr_scale,false,winsize,iterations,poly_n,poly_sigma,flags);
    optflow->calc(_prev0,_next0,_flow0);
}


cv::Ptr<cv::FarnebackOpticalFlow> cv::FarnebackOpticalFlow::create(int numLevels, double pyrScale, bool fastPyramids, int winSize,
                                                               int numIters, int polyN, double polySigma, int flags)
{
    return makePtr<FarnebackOpticalFlowImpl>(numLevels, pyrScale, fastPyramids, winSize,
                                             numIters, polyN, polySigma, flags);
}
