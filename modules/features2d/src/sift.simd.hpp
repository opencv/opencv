// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2020, Intel Corporation, all rights reserved.

/**********************************************************************************************\
 Implementation of SIFT is based on the code from http://blogs.oregonstate.edu/hess/code/sift/
 Below is the original copyright.
 Patent US6711293 expired in March 2020.

//    Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
//    All rights reserved.

//    The following patent has been issued for methods embodied in this
//    software: "Method and apparatus for identifying scale invariant features
//    in an image and use of same for locating an object in an image," David
//    G. Lowe, US Patent 6,711,293 (March 23, 2004). Provisional application
//    filed March 8, 1999. Asignee: The University of British Columbia. For
//    further details, contact David Lowe (lowe@cs.ubc.ca) or the
//    University-Industry Liaison Office of the University of British
//    Columbia.

//    Note that restrictions imposed by this patent (and possibly others)
//    exist independently of and may be in conflict with the freedoms granted
//    in this license, which refers to copyright of the program, not patents
//    for any methods that it implements.  Both copyright and patent law must
//    be obeyed to legally use and redistribute this program and it is not the
//    purpose of this license to induce you to infringe any patents or other
//    property right claims or to contest validity of any such claims.  If you
//    redistribute or use the program, then this license merely protects you
//    from committing copyright infringement.  It does not protect you from
//    committing patent infringement.  So, before you do anything with this
//    program, make sure that you have permission to do so not merely in terms
//    of copyright, but also in terms of patent law.

//    Please note that this license is not to be understood as a guarantee
//    either.  If you use the program according to this license, but in
//    conflict with patent law, it does not mean that the licensor will refund
//    you for any losses that you incur if you are sued for your patent
//    infringement.

//    Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are
//    met:
//        * Redistributions of source code must retain the above copyright and
//          patent notices, this list of conditions and the following
//          disclaimer.
//        * Redistributions in binary form must reproduce the above copyright
//          notice, this list of conditions and the following disclaimer in
//          the documentation and/or other materials provided with the
//          distribution.
//        * Neither the name of Oregon State University nor the names of its
//          contributors may be used to endorse or promote products derived
//          from this software without specific prior written permission.

//    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
//    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
//    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//    HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\**********************************************************************************************/

#include "precomp.hpp"

#include <opencv2/core/hal/hal.hpp>
#include "opencv2/core/hal/intrin.hpp"
#include <opencv2/core/utils/buffer_area.private.hpp>

namespace cv {

#if !defined(CV_CPU_DISPATCH_MODE) || !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY)
/******************************* Defs and macros *****************************/

// default width of descriptor histogram array
static const int SIFT_DESCR_WIDTH = 4;

// default number of bins per histogram in descriptor array
static const int SIFT_DESCR_HIST_BINS = 8;

// assumed gaussian blur for input image
static const float SIFT_INIT_SIGMA = 0.5f;

// width of border in which to ignore keypoints
static const int SIFT_IMG_BORDER = 5;

// maximum steps of keypoint interpolation before failure
static const int SIFT_MAX_INTERP_STEPS = 5;

// default number of bins in histogram for orientation assignment
static const int SIFT_ORI_HIST_BINS = 36;

// determines gaussian sigma for orientation assignment
static const float SIFT_ORI_SIG_FCTR = 1.5f;

// determines the radius of the region used in orientation assignment
static const float SIFT_ORI_RADIUS = 4.5f; // 3 * SIFT_ORI_SIG_FCTR;

// orientation magnitude relative to max that results in new feature
static const float SIFT_ORI_PEAK_RATIO = 0.8f;

// determines the size of a single descriptor orientation histogram
static const float SIFT_DESCR_SCL_FCTR = 3.f;

// threshold on magnitude of elements of descriptor vector
static const float SIFT_DESCR_MAG_THR = 0.2f;

// factor used to convert floating-point descriptor to unsigned char
static const float SIFT_INT_DESCR_FCTR = 512.f;

#define DoG_TYPE_SHORT 0
#if DoG_TYPE_SHORT
// intermediate type used for DoG pyramids
typedef short sift_wt;
static const int SIFT_FIXPT_SCALE = 48;
#else
// intermediate type used for DoG pyramids
typedef float sift_wt;
static const int SIFT_FIXPT_SCALE = 1;
#endif

#endif  // definitions and macros


CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void findScaleSpaceExtrema(
    int octave,
    int layer,
    int threshold,
    int idx,
    int step,
    int cols,
    int nOctaveLayers,
    double contrastThreshold,
    double edgeThreshold,
    double sigma,
    const std::vector<Mat>& gauss_pyr,
    const std::vector<Mat>& dog_pyr,
    std::vector<KeyPoint>& kpts,
    const cv::Range& range);

void calcSIFTDescriptor(
        const Mat& img, Point2f ptf, float ori, float scl,
        int d, int n, Mat& dst, int row
);


#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

// Computes a gradient orientation histogram at a specified pixel
static
float calcOrientationHist(
        const Mat& img, Point pt, int radius,
        float sigma, float* hist, int n
)
{
    CV_TRACE_FUNCTION();

    int i, j, k, len = (radius*2+1)*(radius*2+1);

    float expf_scale = -1.f/(2.f * sigma * sigma);

    cv::utils::BufferArea area;
    float *X = 0, *Y = 0, *Mag, *Ori = 0, *W = 0, *temphist = 0;
    area.allocate(X, len, CV_SIMD_WIDTH);
    area.allocate(Y, len, CV_SIMD_WIDTH);
    area.allocate(Ori, len, CV_SIMD_WIDTH);
    area.allocate(W, len, CV_SIMD_WIDTH);
    area.allocate(temphist, n+4, CV_SIMD_WIDTH);
    area.commit();
    temphist += 2;
    Mag = X;

    for( i = 0; i < n; i++ )
        temphist[i] = 0.f;

    for( i = -radius, k = 0; i <= radius; i++ )
    {
        int y = pt.y + i;
        if( y <= 0 || y >= img.rows - 1 )
            continue;
        for( j = -radius; j <= radius; j++ )
        {
            int x = pt.x + j;
            if( x <= 0 || x >= img.cols - 1 )
                continue;

            float dx = (float)(img.at<sift_wt>(y, x+1) - img.at<sift_wt>(y, x-1));
            float dy = (float)(img.at<sift_wt>(y-1, x) - img.at<sift_wt>(y+1, x));

            X[k] = dx; Y[k] = dy; W[k] = (i*i + j*j)*expf_scale;
            k++;
        }
    }

    len = k;

    // compute gradient values, orientations and the weights over the pixel neighborhood
    cv::hal::exp32f(W, W, len);
    cv::hal::fastAtan2(Y, X, Ori, len, true);
    cv::hal::magnitude32f(X, Y, Mag, len);

    k = 0;
#if CV_SIMD
    const int vecsize = v_float32::nlanes;
    v_float32 nd360 = vx_setall_f32(n/360.f);
    v_int32 __n = vx_setall_s32(n);
    int CV_DECL_ALIGNED(CV_SIMD_WIDTH) bin_buf[vecsize];
    float CV_DECL_ALIGNED(CV_SIMD_WIDTH) w_mul_mag_buf[vecsize];

    for( ; k <= len - vecsize; k += vecsize )
    {
        v_float32 w = vx_load_aligned( W + k );
        v_float32 mag = vx_load_aligned( Mag + k );
        v_float32 ori = vx_load_aligned( Ori + k );
        v_int32 bin = v_round( nd360 * ori );

        bin = v_select(bin >= __n, bin - __n, bin);
        bin = v_select(bin < vx_setzero_s32(), bin + __n, bin);

        w = w * mag;
        v_store_aligned(bin_buf, bin);
        v_store_aligned(w_mul_mag_buf, w);
        for(int vi = 0; vi < vecsize; vi++)
        {
            temphist[bin_buf[vi]] += w_mul_mag_buf[vi];
        }
    }
#endif
    for( ; k < len; k++ )
    {
        int bin = cvRound((n/360.f)*Ori[k]);
        if( bin >= n )
            bin -= n;
        if( bin < 0 )
            bin += n;
        temphist[bin] += W[k]*Mag[k];
    }

    // smooth the histogram
    temphist[-1] = temphist[n-1];
    temphist[-2] = temphist[n-2];
    temphist[n] = temphist[0];
    temphist[n+1] = temphist[1];

    i = 0;
#if CV_SIMD
    v_float32 d_1_16 = vx_setall_f32(1.f/16.f);
    v_float32 d_4_16 = vx_setall_f32(4.f/16.f);
    v_float32 d_6_16 = vx_setall_f32(6.f/16.f);
    for( ; i <= n - v_float32::nlanes; i += v_float32::nlanes )
    {
        v_float32 tn2 = vx_load_aligned(temphist + i-2);
        v_float32 tn1 = vx_load(temphist + i-1);
        v_float32 t0 = vx_load(temphist + i);
        v_float32 t1 = vx_load(temphist + i+1);
        v_float32 t2 = vx_load(temphist + i+2);
        v_float32 _hist = v_fma(tn2 + t2, d_1_16,
            v_fma(tn1 + t1, d_4_16, t0 * d_6_16));
        v_store(hist + i, _hist);
    }
#endif
    for( ; i < n; i++ )
    {
        hist[i] = (temphist[i-2] + temphist[i+2])*(1.f/16.f) +
            (temphist[i-1] + temphist[i+1])*(4.f/16.f) +
            temphist[i]*(6.f/16.f);
    }

    float maxval = hist[0];
    for( i = 1; i < n; i++ )
        maxval = std::max(maxval, hist[i]);

    return maxval;
}


//
// Interpolates a scale-space extremum's location and scale to subpixel
// accuracy to form an image feature. Rejects features with low contrast.
// Based on Section 4 of Lowe's paper.
static
bool adjustLocalExtrema(
        const std::vector<Mat>& dog_pyr, KeyPoint& kpt, int octv,
        int& layer, int& r, int& c, int nOctaveLayers,
        float contrastThreshold, float edgeThreshold, float sigma
)
{
    CV_TRACE_FUNCTION();

    const float img_scale = 1.f/(255*SIFT_FIXPT_SCALE);
    const float deriv_scale = img_scale*0.5f;
    const float second_deriv_scale = img_scale;
    const float cross_deriv_scale = img_scale*0.25f;

    float xi=0, xr=0, xc=0, contr=0;
    int i = 0;

    for( ; i < SIFT_MAX_INTERP_STEPS; i++ )
    {
        int idx = octv*(nOctaveLayers+2) + layer;
        const Mat& img = dog_pyr[idx];
        const Mat& prev = dog_pyr[idx-1];
        const Mat& next = dog_pyr[idx+1];

        Vec3f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
                 (img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
                 (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);

        float v2 = (float)img.at<sift_wt>(r, c)*2;
        float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
        float dyy = (img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
        float dss = (next.at<sift_wt>(r, c) + prev.at<sift_wt>(r, c) - v2)*second_deriv_scale;
        float dxy = (img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
                     img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1))*cross_deriv_scale;
        float dxs = (next.at<sift_wt>(r, c+1) - next.at<sift_wt>(r, c-1) -
                     prev.at<sift_wt>(r, c+1) + prev.at<sift_wt>(r, c-1))*cross_deriv_scale;
        float dys = (next.at<sift_wt>(r+1, c) - next.at<sift_wt>(r-1, c) -
                     prev.at<sift_wt>(r+1, c) + prev.at<sift_wt>(r-1, c))*cross_deriv_scale;

        Matx33f H(dxx, dxy, dxs,
                  dxy, dyy, dys,
                  dxs, dys, dss);

        Vec3f X = H.solve(dD, DECOMP_LU);

        xi = -X[2];
        xr = -X[1];
        xc = -X[0];

        if( std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f )
            break;

        if( std::abs(xi) > (float)(INT_MAX/3) ||
            std::abs(xr) > (float)(INT_MAX/3) ||
            std::abs(xc) > (float)(INT_MAX/3) )
            return false;

        c += cvRound(xc);
        r += cvRound(xr);
        layer += cvRound(xi);

        if( layer < 1 || layer > nOctaveLayers ||
            c < SIFT_IMG_BORDER || c >= img.cols - SIFT_IMG_BORDER  ||
            r < SIFT_IMG_BORDER || r >= img.rows - SIFT_IMG_BORDER )
            return false;
    }

    // ensure convergence of interpolation
    if( i >= SIFT_MAX_INTERP_STEPS )
        return false;

    {
        int idx = octv*(nOctaveLayers+2) + layer;
        const Mat& img = dog_pyr[idx];
        const Mat& prev = dog_pyr[idx-1];
        const Mat& next = dog_pyr[idx+1];
        Matx31f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
                   (img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
                   (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);
        float t = dD.dot(Matx31f(xc, xr, xi));

        contr = img.at<sift_wt>(r, c)*img_scale + t * 0.5f;
        if( std::abs( contr ) * nOctaveLayers < contrastThreshold )
            return false;

        // principal curvatures are computed using the trace and det of Hessian
        float v2 = img.at<sift_wt>(r, c)*2.f;
        float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
        float dyy = (img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
        float dxy = (img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
                     img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1)) * cross_deriv_scale;
        float tr = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;

        if( det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det )
            return false;
    }

    kpt.pt.x = (c + xc) * (1 << octv);
    kpt.pt.y = (r + xr) * (1 << octv);
    kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5)*255) << 16);
    kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv)*2;
    kpt.response = std::abs(contr);

    return true;
}

namespace {

class findScaleSpaceExtremaT
{
public:
    findScaleSpaceExtremaT(
        int _o,
        int _i,
        int _threshold,
        int _idx,
        int _step,
        int _cols,
        int _nOctaveLayers,
        double _contrastThreshold,
        double _edgeThreshold,
        double _sigma,
        const std::vector<Mat>& _gauss_pyr,
        const std::vector<Mat>& _dog_pyr,
        std::vector<KeyPoint>& kpts)

        : o(_o),
          i(_i),
          threshold(_threshold),
          idx(_idx),
          step(_step),
          cols(_cols),
          nOctaveLayers(_nOctaveLayers),
          contrastThreshold(_contrastThreshold),
          edgeThreshold(_edgeThreshold),
          sigma(_sigma),
          gauss_pyr(_gauss_pyr),
          dog_pyr(_dog_pyr),
          kpts_(kpts)
    {
        // nothing
    }
    void process(const cv::Range& range)
    {
        CV_TRACE_FUNCTION();

        const int begin = range.start;
        const int end = range.end;

        static const int n = SIFT_ORI_HIST_BINS;
        float CV_DECL_ALIGNED(CV_SIMD_WIDTH) hist[n];

        const Mat& img = dog_pyr[idx];
        const Mat& prev = dog_pyr[idx-1];
        const Mat& next = dog_pyr[idx+1];

        for( int r = begin; r < end; r++)
        {
            const sift_wt* currptr = img.ptr<sift_wt>(r);
            const sift_wt* prevptr = prev.ptr<sift_wt>(r);
            const sift_wt* nextptr = next.ptr<sift_wt>(r);

            for( int c = SIFT_IMG_BORDER; c < cols-SIFT_IMG_BORDER; c++)
            {
                sift_wt val = currptr[c];

                // find local extrema with pixel accuracy
                if( std::abs(val) > threshold &&
                   ((val > 0 && val >= currptr[c-1] && val >= currptr[c+1] &&
                     val >= currptr[c-step-1] && val >= currptr[c-step] && val >= currptr[c-step+1] &&
                     val >= currptr[c+step-1] && val >= currptr[c+step] && val >= currptr[c+step+1] &&
                     val >= nextptr[c] && val >= nextptr[c-1] && val >= nextptr[c+1] &&
                     val >= nextptr[c-step-1] && val >= nextptr[c-step] && val >= nextptr[c-step+1] &&
                     val >= nextptr[c+step-1] && val >= nextptr[c+step] && val >= nextptr[c+step+1] &&
                     val >= prevptr[c] && val >= prevptr[c-1] && val >= prevptr[c+1] &&
                     val >= prevptr[c-step-1] && val >= prevptr[c-step] && val >= prevptr[c-step+1] &&
                     val >= prevptr[c+step-1] && val >= prevptr[c+step] && val >= prevptr[c+step+1]) ||
                    (val < 0 && val <= currptr[c-1] && val <= currptr[c+1] &&
                     val <= currptr[c-step-1] && val <= currptr[c-step] && val <= currptr[c-step+1] &&
                     val <= currptr[c+step-1] && val <= currptr[c+step] && val <= currptr[c+step+1] &&
                     val <= nextptr[c] && val <= nextptr[c-1] && val <= nextptr[c+1] &&
                     val <= nextptr[c-step-1] && val <= nextptr[c-step] && val <= nextptr[c-step+1] &&
                     val <= nextptr[c+step-1] && val <= nextptr[c+step] && val <= nextptr[c+step+1] &&
                     val <= prevptr[c] && val <= prevptr[c-1] && val <= prevptr[c+1] &&
                     val <= prevptr[c-step-1] && val <= prevptr[c-step] && val <= prevptr[c-step+1] &&
                     val <= prevptr[c+step-1] && val <= prevptr[c+step] && val <= prevptr[c+step+1])))
                {
                    CV_TRACE_REGION("pixel_candidate");

                    KeyPoint kpt;
                    int r1 = r, c1 = c, layer = i;
                    if( !adjustLocalExtrema(dog_pyr, kpt, o, layer, r1, c1,
                                            nOctaveLayers, (float)contrastThreshold,
                                            (float)edgeThreshold, (float)sigma) )
                        continue;
                    float scl_octv = kpt.size*0.5f/(1 << o);
                    float omax = calcOrientationHist(gauss_pyr[o*(nOctaveLayers+3) + layer],
                                                     Point(c1, r1),
                                                     cvRound(SIFT_ORI_RADIUS * scl_octv),
                                                     SIFT_ORI_SIG_FCTR * scl_octv,
                                                     hist, n);
                    float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);
                    for( int j = 0; j < n; j++ )
                    {
                        int l = j > 0 ? j - 1 : n - 1;
                        int r2 = j < n-1 ? j + 1 : 0;

                        if( hist[j] > hist[l]  &&  hist[j] > hist[r2]  &&  hist[j] >= mag_thr )
                        {
                            float bin = j + 0.5f * (hist[l]-hist[r2]) / (hist[l] - 2*hist[j] + hist[r2]);
                            bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
                            kpt.angle = 360.f - (float)((360.f/n) * bin);
                            if(std::abs(kpt.angle - 360.f) < FLT_EPSILON)
                                kpt.angle = 0.f;

                            kpts_.push_back(kpt);
                        }
                    }
                }
            }
        }
    }
private:
    int o, i;
    int threshold;
    int idx, step, cols;
    int nOctaveLayers;
    double contrastThreshold;
    double edgeThreshold;
    double sigma;
    const std::vector<Mat>& gauss_pyr;
    const std::vector<Mat>& dog_pyr;
    std::vector<KeyPoint>& kpts_;
};

}  // namespace


void findScaleSpaceExtrema(
    int octave,
    int layer,
    int threshold,
    int idx,
    int step,
    int cols,
    int nOctaveLayers,
    double contrastThreshold,
    double edgeThreshold,
    double sigma,
    const std::vector<Mat>& gauss_pyr,
    const std::vector<Mat>& dog_pyr,
    std::vector<KeyPoint>& kpts,
    const cv::Range& range)
{
    CV_TRACE_FUNCTION();

    findScaleSpaceExtremaT(octave, layer, threshold, idx,
            step, cols,
            nOctaveLayers, contrastThreshold, edgeThreshold, sigma,
            gauss_pyr, dog_pyr,
            kpts)
        .process(range);
}

void calcSIFTDescriptor(
        const Mat& img, Point2f ptf, float ori, float scl,
        int d, int n, Mat& dstMat, int row
)
{
    CV_TRACE_FUNCTION();

    Point pt(cvRound(ptf.x), cvRound(ptf.y));
    float cos_t = cosf(ori*(float)(CV_PI/180));
    float sin_t = sinf(ori*(float)(CV_PI/180));
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f/(d * d * 0.5f);
    float hist_width = SIFT_DESCR_SCL_FCTR * scl;
    int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
    // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
    radius = std::min(radius, (int)std::sqrt(((double) img.cols)*img.cols + ((double) img.rows)*img.rows));
    cos_t /= hist_width;
    sin_t /= hist_width;

    int i, j, k, len = (radius*2+1)*(radius*2+1), histlen = (d+2)*(d+2)*(n+2);
    int rows = img.rows, cols = img.cols;

    cv::utils::BufferArea area;
    float *X = 0, *Y = 0, *Mag, *Ori = 0, *W = 0, *RBin = 0, *CBin = 0, *hist = 0, *rawDst = 0;
    area.allocate(X, len, CV_SIMD_WIDTH);
    area.allocate(Y, len, CV_SIMD_WIDTH);
    area.allocate(Ori, len, CV_SIMD_WIDTH);
    area.allocate(W, len, CV_SIMD_WIDTH);
    area.allocate(RBin, len, CV_SIMD_WIDTH);
    area.allocate(CBin, len, CV_SIMD_WIDTH);
    area.allocate(hist, histlen, CV_SIMD_WIDTH);
    area.allocate(rawDst, len, CV_SIMD_WIDTH);
    area.commit();
    Mag = Y;

    for( i = 0; i < d+2; i++ )
    {
        for( j = 0; j < d+2; j++ )
            for( k = 0; k < n+2; k++ )
                hist[(i*(d+2) + j)*(n+2) + k] = 0.;
    }

    for( i = -radius, k = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            // Calculate sample's histogram array coords rotated relative to ori.
            // Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
            // r_rot = 1.5) have full weight placed in row 1 after interpolation.
            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d/2 - 0.5f;
            float cbin = c_rot + d/2 - 0.5f;
            int r = pt.y + i, c = pt.x + j;

            if( rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
                r > 0 && r < rows - 1 && c > 0 && c < cols - 1 )
            {
                float dx = (float)(img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1));
                float dy = (float)(img.at<sift_wt>(r-1, c) - img.at<sift_wt>(r+1, c));
                X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
                W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
                k++;
            }
        }

    len = k;
    cv::hal::fastAtan2(Y, X, Ori, len, true);
    cv::hal::magnitude32f(X, Y, Mag, len);
    cv::hal::exp32f(W, W, len);

    k = 0;
#if CV_SIMD
    {
        const int vecsize = v_float32::nlanes;
        int CV_DECL_ALIGNED(CV_SIMD_WIDTH) idx_buf[vecsize];
        float CV_DECL_ALIGNED(CV_SIMD_WIDTH) rco_buf[8*vecsize];
        const v_float32 __ori  = vx_setall_f32(ori);
        const v_float32 __bins_per_rad = vx_setall_f32(bins_per_rad);
        const v_int32 __n = vx_setall_s32(n);
        const v_int32 __1 = vx_setall_s32(1);
        const v_int32 __d_plus_2 = vx_setall_s32(d+2);
        const v_int32 __n_plus_2 = vx_setall_s32(n+2);
        for( ; k <= len - vecsize; k += vecsize )
        {
            v_float32 rbin = vx_load_aligned(RBin + k);
            v_float32 cbin = vx_load_aligned(CBin + k);
            v_float32 obin = (vx_load_aligned(Ori + k) - __ori) * __bins_per_rad;
            v_float32 mag = vx_load_aligned(Mag + k) * vx_load_aligned(W + k);

            v_int32 r0 = v_floor(rbin);
            v_int32 c0 = v_floor(cbin);
            v_int32 o0 = v_floor(obin);
            rbin -= v_cvt_f32(r0);
            cbin -= v_cvt_f32(c0);
            obin -= v_cvt_f32(o0);

            o0 = v_select(o0 < vx_setzero_s32(), o0 + __n, o0);
            o0 = v_select(o0 >= __n, o0 - __n, o0);

            v_float32 v_r1 = mag*rbin, v_r0 = mag - v_r1;
            v_float32 v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
            v_float32 v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
            v_float32 v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
            v_float32 v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
            v_float32 v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
            v_float32 v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

            v_int32 idx = v_muladd(v_muladd(r0+__1, __d_plus_2, c0+__1), __n_plus_2, o0);
            v_store_aligned(idx_buf, idx);

            v_store_aligned(rco_buf,           v_rco000);
            v_store_aligned(rco_buf+vecsize,   v_rco001);
            v_store_aligned(rco_buf+vecsize*2, v_rco010);
            v_store_aligned(rco_buf+vecsize*3, v_rco011);
            v_store_aligned(rco_buf+vecsize*4, v_rco100);
            v_store_aligned(rco_buf+vecsize*5, v_rco101);
            v_store_aligned(rco_buf+vecsize*6, v_rco110);
            v_store_aligned(rco_buf+vecsize*7, v_rco111);

            for(int id = 0; id < vecsize; id++)
            {
                hist[idx_buf[id]] += rco_buf[id];
                hist[idx_buf[id]+1] += rco_buf[vecsize + id];
                hist[idx_buf[id]+(n+2)] += rco_buf[2*vecsize + id];
                hist[idx_buf[id]+(n+3)] += rco_buf[3*vecsize + id];
                hist[idx_buf[id]+(d+2)*(n+2)] += rco_buf[4*vecsize + id];
                hist[idx_buf[id]+(d+2)*(n+2)+1] += rco_buf[5*vecsize + id];
                hist[idx_buf[id]+(d+3)*(n+2)] += rco_buf[6*vecsize + id];
                hist[idx_buf[id]+(d+3)*(n+2)+1] += rco_buf[7*vecsize + id];
            }
        }
    }
#endif
    for( ; k < len; k++ )
    {
        float rbin = RBin[k], cbin = CBin[k];
        float obin = (Ori[k] - ori)*bins_per_rad;
        float mag = Mag[k]*W[k];

        int r0 = cvFloor( rbin );
        int c0 = cvFloor( cbin );
        int o0 = cvFloor( obin );
        rbin -= r0;
        cbin -= c0;
        obin -= o0;

        if( o0 < 0 )
            o0 += n;
        if( o0 >= n )
            o0 -= n;

        // histogram update using tri-linear interpolation
        float v_r1 = mag*rbin, v_r0 = mag - v_r1;
        float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
        float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
        float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
        float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
        float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
        float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

        int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;
        hist[idx] += v_rco000;
        hist[idx+1] += v_rco001;
        hist[idx+(n+2)] += v_rco010;
        hist[idx+(n+3)] += v_rco011;
        hist[idx+(d+2)*(n+2)] += v_rco100;
        hist[idx+(d+2)*(n+2)+1] += v_rco101;
        hist[idx+(d+3)*(n+2)] += v_rco110;
        hist[idx+(d+3)*(n+2)+1] += v_rco111;
    }

    // finalize histogram, since the orientation histograms are circular
    for( i = 0; i < d; i++ )
        for( j = 0; j < d; j++ )
        {
            int idx = ((i+1)*(d+2) + (j+1))*(n+2);
            hist[idx] += hist[idx+n];
            hist[idx+1] += hist[idx+n+1];
            for( k = 0; k < n; k++ )
                rawDst[(i*d + j)*n + k] = hist[idx+k];
        }
    // copy histogram to the descriptor,
    // apply hysteresis thresholding
    // and scale the result, so that it can be easily converted
    // to byte array
    float nrm2 = 0;
    len = d*d*n;
    k = 0;
#if CV_SIMD
    {
        v_float32 __nrm2 = vx_setzero_f32();
        v_float32 __rawDst;
        for( ; k <= len - v_float32::nlanes; k += v_float32::nlanes )
        {
            __rawDst = vx_load_aligned(rawDst + k);
            __nrm2 = v_fma(__rawDst, __rawDst, __nrm2);
        }
        nrm2 = (float)v_reduce_sum(__nrm2);
    }
#endif
    for( ; k < len; k++ )
        nrm2 += rawDst[k]*rawDst[k];

    float thr = std::sqrt(nrm2)*SIFT_DESCR_MAG_THR;

    i = 0, nrm2 = 0;
#if 0 //CV_AVX2
    // This code cannot be enabled because it sums nrm2 in a different order,
    // thus producing slightly different results
    {
        float CV_DECL_ALIGNED(CV_SIMD_WIDTH) nrm2_buf[8];
        __m256 __dst;
        __m256 __nrm2 = _mm256_setzero_ps();
        __m256 __thr = _mm256_set1_ps(thr);
        for( ; i <= len - 8; i += 8 )
        {
            __dst = _mm256_loadu_ps(&rawDst[i]);
            __dst = _mm256_min_ps(__dst, __thr);
            _mm256_storeu_ps(&rawDst[i], __dst);
#if CV_FMA3
            __nrm2 = _mm256_fmadd_ps(__dst, __dst, __nrm2);
#else
            __nrm2 = _mm256_add_ps(__nrm2, _mm256_mul_ps(__dst, __dst));
#endif
        }
        _mm256_store_ps(nrm2_buf, __nrm2);
        nrm2 = nrm2_buf[0] + nrm2_buf[1] + nrm2_buf[2] + nrm2_buf[3] +
               nrm2_buf[4] + nrm2_buf[5] + nrm2_buf[6] + nrm2_buf[7];
    }
#endif
    for( ; i < len; i++ )
    {
        float val = std::min(rawDst[i], thr);
        rawDst[i] = val;
        nrm2 += val*val;
    }
    nrm2 = SIFT_INT_DESCR_FCTR/std::max(std::sqrt(nrm2), FLT_EPSILON);

#if 1
    k = 0;
if( dstMat.type() == CV_32F )
{
    float* dst = dstMat.ptr<float>(row);
#if CV_SIMD
    v_float32 __dst;
    v_float32 __min = vx_setzero_f32();
    v_float32 __max = vx_setall_f32(255.0f); // max of uchar
    v_float32 __nrm2 = vx_setall_f32(nrm2);
    for( k = 0; k <= len - v_float32::nlanes; k += v_float32::nlanes )
    {
        __dst = vx_load_aligned(rawDst + k);
        __dst = v_min(v_max(v_cvt_f32(v_round(__dst * __nrm2)), __min), __max);
        v_store(dst + k, __dst);
    }
#endif
    for( ; k < len; k++ )
    {
        dst[k] = saturate_cast<uchar>(rawDst[k]*nrm2);
    }
}
else // CV_8U
{
    uint8_t* dst = dstMat.ptr<uint8_t>(row);
#if CV_SIMD
    v_float32 __dst0, __dst1;
    v_uint16 __pack01;
    v_float32 __nrm2 = vx_setall_f32(nrm2);
    for( k = 0; k <= len - v_float32::nlanes * 2; k += v_float32::nlanes * 2 )
    {
        __dst0 = vx_load_aligned(rawDst + k);
        __dst1 = vx_load_aligned(rawDst + k + v_float32::nlanes);

        __pack01 = v_pack_u(v_round(__dst0 * __nrm2), v_round(__dst1 * __nrm2));
        v_pack_store(dst + k, __pack01);
    }
#endif
    for( ; k < len; k++ )
    {
        dst[k] = saturate_cast<uchar>(rawDst[k]*nrm2);
    }
}
#else
    float* dst = dstMat.ptr<float>(row);
    float nrm1 = 0;
    for( k = 0; k < len; k++ )
    {
        rawDst[k] *= nrm2;
        nrm1 += rawDst[k];
    }
    nrm1 = 1.f/std::max(nrm1, FLT_EPSILON);
if( dstMat.type() == CV_32F )
{
    for( k = 0; k < len; k++ )
    {
        dst[k] = std::sqrt(rawDst[k] * nrm1);
    }
}
else // CV_8U
{
    for( k = 0; k < len; k++ )
    {
        dst[k] = saturate_cast<uchar>(std::sqrt(rawDst[k] * nrm1)*SIFT_INT_DESCR_FCTR);
    }
}
#endif
}

#endif
CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace
