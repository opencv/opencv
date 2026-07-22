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
//    filed March 8, 1999. Assignee: The University of British Columbia. For
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
#include <opencv2/core/utils/tls.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "sift.simd.hpp"
#include "sift.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content
#include "opencl_kernels_features.hpp"

namespace cv {

/*!
 SIFT implementation.

 The class implements SIFT algorithm by D. Lowe.
 */
class SIFT_Impl : public SIFT
{
public:
    explicit SIFT_Impl( int nfeatures = 0, int nOctaveLayers = 3,
                          double contrastThreshold = 0.04, double edgeThreshold = 10,
                          double sigma = 1.6, int descriptorType = CV_32F,
                          bool enable_precise_upscale = true );

    //! returns the descriptor size in floats (128)
    int descriptorSize() const CV_OVERRIDE;

    //! returns the descriptor type
    int descriptorType() const CV_OVERRIDE;

    //! returns the default norm type
    int defaultNorm() const CV_OVERRIDE;

    //! finds the keypoints and computes descriptors for them using SIFT algorithm.
    //! Optionally it can compute descriptors for the user-provided keypoints
    void detectAndCompute(InputArray img, InputArray mask,
                    std::vector<KeyPoint>& keypoints,
                    OutputArray descriptors,
                    bool useProvidedKeypoints = false) CV_OVERRIDE;

    void buildGaussianPyramid( const Mat& base, std::vector<Mat>& pyr, int nOctaves ) const;
    void buildDoGPyramid( const std::vector<Mat>& pyr, std::vector<Mat>& dogpyr ) const;
    void findScaleSpaceExtrema( const std::vector<Mat>& gauss_pyr, const std::vector<Mat>& dog_pyr,
                               std::vector<KeyPoint>& keypoints ) const;

    void read( const FileNode& fn) CV_OVERRIDE;
    void write( FileStorage& fs) const CV_OVERRIDE;

    void setNFeatures(int maxFeatures) CV_OVERRIDE { nfeatures = maxFeatures; }
    int getNFeatures() const CV_OVERRIDE { return nfeatures; }

    void setNOctaveLayers(int nOctaveLayers_) CV_OVERRIDE { nOctaveLayers = nOctaveLayers_; }
    int getNOctaveLayers() const CV_OVERRIDE { return nOctaveLayers; }

    void setContrastThreshold(double contrastThreshold_) CV_OVERRIDE  { contrastThreshold = contrastThreshold_; }
    double getContrastThreshold() const CV_OVERRIDE { return contrastThreshold; }

    void setEdgeThreshold(double edgeThreshold_) CV_OVERRIDE  { edgeThreshold = edgeThreshold_; }
    double getEdgeThreshold() const CV_OVERRIDE { return edgeThreshold; }

    void setSigma(double sigma_) CV_OVERRIDE  { sigma = sigma_; }
    double getSigma() const CV_OVERRIDE { return sigma; }

private:
#ifdef HAVE_OPENCL
    bool ocl_detectAndCompute(InputArray _image, InputArray _mask,
                              std::vector<KeyPoint>& keypoints,
                              OutputArray _descriptors, bool useProvidedKeypoints);
#endif

protected:
    CV_PROP_RW int nfeatures;
    CV_PROP_RW int nOctaveLayers;
    CV_PROP_RW double contrastThreshold;
    CV_PROP_RW double edgeThreshold;
    CV_PROP_RW double sigma;
    CV_PROP_RW int descriptor_type;
    CV_PROP_RW bool enable_precise_upscale;
};

Ptr<SIFT> SIFT::create( int _nfeatures, int _nOctaveLayers,
                     double _contrastThreshold, double _edgeThreshold, double _sigma, bool enable_precise_upscale )
{
    CV_TRACE_FUNCTION();

    return makePtr<SIFT_Impl>(_nfeatures, _nOctaveLayers, _contrastThreshold, _edgeThreshold, _sigma, CV_32F, enable_precise_upscale);
}

Ptr<SIFT> SIFT::create( int _nfeatures, int _nOctaveLayers,
                     double _contrastThreshold, double _edgeThreshold, double _sigma, int _descriptorType, bool enable_precise_upscale )
{
    CV_TRACE_FUNCTION();

    // SIFT descriptor supports 32bit floating point and 8bit unsigned int.
    CV_Assert(_descriptorType == CV_32F || _descriptorType == CV_8U);
    return makePtr<SIFT_Impl>(_nfeatures, _nOctaveLayers, _contrastThreshold, _edgeThreshold, _sigma, _descriptorType, enable_precise_upscale);
}

String SIFT::getDefaultName() const
{
    return (Feature2D::getDefaultName() + ".SIFT");
}

static inline void
unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale)
{
    octave = kpt.octave & 255;
    layer = (kpt.octave >> 8) & 255;
    octave = octave < 128 ? octave : (-128 | octave);
    scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);
}

static Mat createInitialImage( const Mat& img, bool doubleImageSize, float sigma, bool enable_precise_upscale )
{
    CV_TRACE_FUNCTION();

    Mat gray, gray_fpt;
    if( img.channels() == 3 || img.channels() == 4 )
    {
        cvtColor(img, gray, COLOR_BGR2GRAY);
        gray.convertTo(gray_fpt, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);
    }
    else
        img.convertTo(gray_fpt, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);

    float sig_diff;

    if( doubleImageSize )
    {
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f) );

        Mat dbl;
        if (enable_precise_upscale) {
            dbl.create(Size(gray_fpt.cols*2, gray_fpt.rows*2), gray_fpt.type());
            Mat H = Mat::zeros(2, 3, CV_32F);
            H.at<float>(0, 0) = 0.5f;
            H.at<float>(1, 1) = 0.5f;

            cv::warpAffine(gray_fpt, dbl, H, dbl.size(), INTER_LINEAR | WARP_INVERSE_MAP, BORDER_REFLECT);
        } else {
#if DoG_TYPE_SHORT
            resize(gray_fpt, dbl, Size(gray_fpt.cols*2, gray_fpt.rows*2), 0, 0, INTER_LINEAR_EXACT);
#else
            resize(gray_fpt, dbl, Size(gray_fpt.cols*2, gray_fpt.rows*2), 0, 0, INTER_LINEAR);
#endif
        }
        Mat result;
        GaussianBlur(dbl, result, Size(), sig_diff, sig_diff);
        return result;
    }
    else
    {
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f) );
        Mat result;
        GaussianBlur(gray_fpt, result, Size(), sig_diff, sig_diff);
        return result;
    }
}


void SIFT_Impl::buildGaussianPyramid( const Mat& base, std::vector<Mat>& pyr, int nOctaves ) const
{
    CV_TRACE_FUNCTION();

    AutoBuffer<double> sig(nOctaveLayers + 3);
    pyr.resize(nOctaves*(nOctaveLayers + 3));

    // precompute Gaussian sigmas using the following formula:
    //  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    sig[0] = sigma;
    double k = std::pow( 2, 1. / nOctaveLayers );
    for( int i = 1; i < nOctaveLayers + 3; i++ )
    {
        double sig_prev = (double)std::pow(k, i-1)*sigma;
        double sig_total = sig_prev*k;
        sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
    }

    for( int o = 0; o < nOctaves; o++ )
    {
        for( int i = 0; i < nOctaveLayers + 3; i++ )
        {
            Mat& dst = pyr[o*(nOctaveLayers + 3) + i];
            if( o == 0  &&  i == 0 )
                dst = base;
            // base of new octave is halved image from end of previous octave
            else if( i == 0 )
            {
                const Mat& src = pyr[(o-1)*(nOctaveLayers + 3) + nOctaveLayers];
                resize(src, dst, Size(src.cols/2, src.rows/2),
                       0, 0, INTER_NEAREST);
            }
            else
            {
                const Mat& src = pyr[o*(nOctaveLayers + 3) + i-1];
                GaussianBlur(src, dst, Size(), sig[i], sig[i]);
            }
        }
    }
}


class buildDoGPyramidComputer : public ParallelLoopBody
{
public:
    buildDoGPyramidComputer(
        int _nOctaveLayers,
        const std::vector<Mat>& _gpyr,
        std::vector<Mat>& _dogpyr)
        : nOctaveLayers(_nOctaveLayers),
          gpyr(_gpyr),
          dogpyr(_dogpyr) { }

    void operator()( const cv::Range& range ) const CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();

        const int begin = range.start;
        const int end = range.end;

        for( int a = begin; a < end; a++ )
        {
            const int o = a / (nOctaveLayers + 2);
            const int i = a % (nOctaveLayers + 2);

            const Mat& src1 = gpyr[o*(nOctaveLayers + 3) + i];
            const Mat& src2 = gpyr[o*(nOctaveLayers + 3) + i + 1];
            Mat& dst = dogpyr[o*(nOctaveLayers + 2) + i];
            subtract(src2, src1, dst, noArray(), DataType<sift_wt>::type);
        }
    }

private:
    int nOctaveLayers;
    const std::vector<Mat>& gpyr;
    std::vector<Mat>& dogpyr;
};

void SIFT_Impl::buildDoGPyramid( const std::vector<Mat>& gpyr, std::vector<Mat>& dogpyr ) const
{
    CV_TRACE_FUNCTION();

    int nOctaves = (int)gpyr.size()/(nOctaveLayers + 3);
    dogpyr.resize( nOctaves*(nOctaveLayers + 2) );

    parallel_for_(Range(0, nOctaves * (nOctaveLayers + 2)), buildDoGPyramidComputer(nOctaveLayers, gpyr, dogpyr));
}

class findScaleSpaceExtremaComputer : public ParallelLoopBody
{
public:
    findScaleSpaceExtremaComputer(
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
        TLSData<std::vector<KeyPoint> > &_tls_kpts_struct)

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
          tls_kpts_struct(_tls_kpts_struct) { }
    void operator()( const cv::Range& range ) const CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();

        std::vector<KeyPoint>& kpts = tls_kpts_struct.getRef();

        CV_CPU_DISPATCH(findScaleSpaceExtrema, (o, i, threshold, idx, step, cols, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, gauss_pyr, dog_pyr, kpts, range),
            CV_CPU_DISPATCH_MODES_ALL);
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
    TLSData<std::vector<KeyPoint> > &tls_kpts_struct;
};

//
// Detects features at extrema in DoG scale space.  Bad features are discarded
// based on contrast and ratio of principal curvatures.
void SIFT_Impl::findScaleSpaceExtrema( const std::vector<Mat>& gauss_pyr, const std::vector<Mat>& dog_pyr,
                                  std::vector<KeyPoint>& keypoints ) const
{
    CV_TRACE_FUNCTION();

    const int nOctaves = (int)gauss_pyr.size()/(nOctaveLayers + 3);
    const int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);

    keypoints.clear();
    TLSDataAccumulator<std::vector<KeyPoint> > tls_kpts_struct;

    for( int o = 0; o < nOctaves; o++ )
        for( int i = 1; i <= nOctaveLayers; i++ )
        {
            const int idx = o*(nOctaveLayers+2)+i;
            const Mat& img = dog_pyr[idx];
            const int step = (int)img.step1();
            const int rows = img.rows, cols = img.cols;

            parallel_for_(Range(SIFT_IMG_BORDER, rows-SIFT_IMG_BORDER),
                findScaleSpaceExtremaComputer(
                    o, i, threshold, idx, step, cols,
                    nOctaveLayers,
                    contrastThreshold,
                    edgeThreshold,
                    sigma,
                    gauss_pyr, dog_pyr, tls_kpts_struct));
        }

    std::vector<std::vector<KeyPoint>*> kpt_vecs;
    tls_kpts_struct.gather(kpt_vecs);
    for (size_t i = 0; i < kpt_vecs.size(); ++i) {
        keypoints.insert(keypoints.end(), kpt_vecs[i]->begin(), kpt_vecs[i]->end());
    }
}


static
void calcSIFTDescriptor(
        const Mat& img, Point2f ptf, float ori, float scl,
        int d, int n, Mat& dst, int row
)
{
    CV_TRACE_FUNCTION();

    CV_CPU_DISPATCH(calcSIFTDescriptor, (img, ptf, ori, scl, d, n, dst, row),
        CV_CPU_DISPATCH_MODES_ALL);
}

class calcDescriptorsComputer : public ParallelLoopBody
{
public:
    calcDescriptorsComputer(const std::vector<Mat>& _gpyr,
                            const std::vector<KeyPoint>& _keypoints,
                            Mat& _descriptors,
                            int _nOctaveLayers,
                            int _firstOctave)
        : gpyr(_gpyr),
          keypoints(_keypoints),
          descriptors(_descriptors),
          nOctaveLayers(_nOctaveLayers),
          firstOctave(_firstOctave) { }

    void operator()( const cv::Range& range ) const CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();

        const int begin = range.start;
        const int end = range.end;

        static const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;

        for ( int i = begin; i<end; i++ )
        {
            KeyPoint kpt = keypoints[i];
            int octave, layer;
            float scale;
            unpackOctave(kpt, octave, layer, scale);
            CV_Assert(octave >= firstOctave && layer <= nOctaveLayers+2);
            float size=kpt.size*scale;
            Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);
            const Mat& img = gpyr[(octave - firstOctave)*(nOctaveLayers + 3) + layer];

            float angle = 360.f - kpt.angle;
            if(std::abs(angle - 360.f) < FLT_EPSILON)
                angle = 0.f;
            calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors, i);
        }
    }
private:
    const std::vector<Mat>& gpyr;
    const std::vector<KeyPoint>& keypoints;
    Mat& descriptors;
    int nOctaveLayers;
    int firstOctave;
};

static void calcDescriptors(const std::vector<Mat>& gpyr, const std::vector<KeyPoint>& keypoints,
                            Mat& descriptors, int nOctaveLayers, int firstOctave )
{
    CV_TRACE_FUNCTION();
    parallel_for_(Range(0, static_cast<int>(keypoints.size())), calcDescriptorsComputer(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave));
}

//////////////////////////////////////////////////////////////////////////////////////////

SIFT_Impl::SIFT_Impl( int _nfeatures, int _nOctaveLayers,
           double _contrastThreshold, double _edgeThreshold, double _sigma, int _descriptorType, bool _enable_precise_upscale)
    : nfeatures(_nfeatures), nOctaveLayers(_nOctaveLayers),
    contrastThreshold(_contrastThreshold), edgeThreshold(_edgeThreshold), sigma(_sigma), descriptor_type(_descriptorType),
    enable_precise_upscale(_enable_precise_upscale)
{
    if (!enable_precise_upscale) {
        CV_LOG_ONCE_INFO(NULL, "precise upscale disabled, this is now deprecated as it was found to induce a location bias");
    }
}

int SIFT_Impl::descriptorSize() const
{
    return SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
}

int SIFT_Impl::descriptorType() const
{
    return descriptor_type;
}

int SIFT_Impl::defaultNorm() const
{
    return NORM_L2;
}

#ifdef HAVE_OPENCL

namespace {

static void computeGaussianKernel1D(double sigma, std::vector<float>& coeffs, int& radius)
{
    radius = (int)std::ceil(sigma * 3.0);
    coeffs.resize(2 * radius + 1);
    double sum = 0.0;
    for (int i = -radius; i <= radius; i++)
    {
        double v = std::exp(-0.5 * (double)(i*i) / (sigma * sigma));
        coeffs[i + radius] = (float)v;
        sum += v;
    }
    for (int i = 0; i < (int)coeffs.size(); i++)
        coeffs[i] = (float)(coeffs[i] / sum);
}

static bool ocl_gaussianBlurSep(const UMat& src, UMat& dst, double sigma)
{
    std::vector<float> coeffs;
    int radius;
    computeGaussianKernel1D(sigma, coeffs, radius);
    UMat uc = Mat(coeffs, true).getUMat(ACCESS_READ);
    dst.create(src.size(), src.type());
    UMat tmp(src.size(), src.type());
    size_t localSize[2] = {16, 16};
    size_t globalSize[2] = {
        (size_t)((src.cols + 15) / 16) * 16,
        (size_t)((src.rows + 15) / 16) * 16
    };

    ocl::Kernel kerH("SIFT_gaussian_blur_h", ocl::features::sift_oclsrc);
    if (kerH.empty())
        return false;
    if (!kerH.args(
        ocl::KernelArg::ReadOnlyNoSize(src),
        ocl::KernelArg::WriteOnlyNoSize(tmp),
        src.cols, src.rows,
        ocl::KernelArg::PtrReadOnly(uc), radius
    ).run(2, globalSize, localSize, false))
        return false;

    ocl::Kernel kerV("SIFT_gaussian_blur_v", ocl::features::sift_oclsrc);
    if (kerV.empty())
        return false;
    return kerV.args(
        ocl::KernelArg::ReadOnlyNoSize(tmp),
        ocl::KernelArg::WriteOnlyNoSize(dst),
        src.cols, src.rows,
        ocl::KernelArg::PtrReadOnly(uc), radius
    ).run(2, globalSize, localSize, false);
}

static UMat ocl_createInitialImage(const UMat& img, bool doubleImageSize, float sigma, bool enable_precise_upscale)
{
    UMat gray, gray_fpt;
    if (img.channels() == 3 || img.channels() == 4)
    {
        cvtColor(img, gray, COLOR_BGR2GRAY);
        gray.convertTo(gray_fpt, CV_32F, SIFT_FIXPT_SCALE, 0);
    }
    else
        img.convertTo(gray_fpt, CV_32F, SIFT_FIXPT_SCALE, 0);

    float sig_diff;
    if (doubleImageSize)
    {
        sig_diff = sqrtf(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f));
        UMat dbl;
        if (enable_precise_upscale) {
            dbl.create(Size(gray_fpt.cols*2, gray_fpt.rows*2), gray_fpt.type());
            Mat H = Mat::zeros(2, 3, CV_32F);
            H.at<float>(0, 0) = 0.5f;
            H.at<float>(1, 1) = 0.5f;
            warpAffine(gray_fpt, dbl, H, dbl.size(), INTER_LINEAR | WARP_INVERSE_MAP, BORDER_REFLECT);
        } else {
            resize(gray_fpt, dbl, Size(gray_fpt.cols*2, gray_fpt.rows*2), 0, 0, INTER_LINEAR);
        }
        UMat result;
        if (!ocl_gaussianBlurSep(dbl, result, (double)sig_diff))
            return UMat();
        return result;
    }
    else
    {
        sig_diff = sqrtf(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f));
        UMat result;
        if (!ocl_gaussianBlurSep(gray_fpt, result, (double)sig_diff))
            return UMat();
        return result;
    }
}

static bool ocl_buildGaussianPyramid(const UMat& base, std::vector<UMat>& gauss_packs,
                                     int nOctaves, int nOctaveLayers, double sigma)
{
    const int levelsPerOctave = nOctaveLayers + 3;
    std::vector<double> sig(nOctaveLayers + 3);
    gauss_packs.resize(nOctaves);

    sig[0] = sigma;
    double k = std::pow(2., 1. / nOctaveLayers);
    for (int i = 1; i < nOctaveLayers + 3; i++)
    {
        double sig_prev = std::pow(k, i-1)*sigma;
        double sig_total = sig_prev*k;
        sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
    }

    int n_kernels = nOctaveLayers + 2;
    std::vector<std::vector<float> > gk_coeffs(n_kernels);
    std::vector<int> gk_radius(n_kernels);
    for (int i = 0; i < n_kernels; i++)
        computeGaussianKernel1D(sig[i + 1], gk_coeffs[i], gk_radius[i]);

    std::vector<UMat> uc(n_kernels);
    for (int i = 0; i < n_kernels; i++)
        uc[i] = Mat(gk_coeffs[i], true).getUMat(ACCESS_READ);

    for (int o = 0; o < nOctaves; o++)
    {
        int levelW = base.cols >> o;
        int levelH = base.rows >> o;
        gauss_packs[o].create(levelsPerOctave * levelH, levelW, base.type());
    }

    for (int o = 0; o < nOctaves; o++)
    {
        int img_cols = gauss_packs[o].cols;
        int img_rows = gauss_packs[o].rows / levelsPerOctave;

        UMat level0 = gauss_packs[o].rowRange(0, img_rows);
        if (o == 0)
        {
            base.copyTo(level0);
        }
        else
        {
            int prev_rows = gauss_packs[o-1].rows / levelsPerOctave;
            UMat src = gauss_packs[o-1].rowRange(nOctaveLayers * prev_rows,
                                                  (nOctaveLayers + 1) * prev_rows);
            resize(src, level0, Size(img_cols, img_rows), 0, 0, INTER_NEAREST);
        }

        UMat tmp(Size(img_cols, img_rows), base.type());
        UMat prev = level0;
        size_t localSize[2] = {16, 16};

        for (int i = 1; i < levelsPerOctave; i++)
        {
            int lev_idx = i - 1;
            UMat dst = gauss_packs[o].rowRange(i * img_rows, (i + 1) * img_rows);

            size_t globalSize[2] = {
                (size_t)((img_cols + 15) / 16) * 16,
                (size_t)((img_rows + 15) / 16) * 16
            };

            ocl::Kernel kerH("SIFT_gaussian_blur_h", ocl::features::sift_oclsrc);
            if (kerH.empty())
                return false;
            if (!kerH.args(
                ocl::KernelArg::ReadOnlyNoSize(prev),
                ocl::KernelArg::WriteOnlyNoSize(tmp),
                img_cols, img_rows,
                ocl::KernelArg::PtrReadOnly(uc[lev_idx]), gk_radius[lev_idx]
            ).run(2, globalSize, localSize, false))
                return false;

            ocl::Kernel kerV("SIFT_gaussian_blur_v", ocl::features::sift_oclsrc);
            if (kerV.empty())
                return false;
            if (!kerV.args(
                ocl::KernelArg::ReadOnlyNoSize(tmp),
                ocl::KernelArg::WriteOnlyNoSize(dst),
                img_cols, img_rows,
                ocl::KernelArg::PtrReadOnly(uc[lev_idx]), gk_radius[lev_idx]
            ).run(2, globalSize, localSize, false))
                return false;

            prev = dst;
        }
    }
    return true;
}

static bool ocl_detectAndOrient(
    const std::vector<UMat>& gauss_packs,
    std::vector<KeyPoint>& keypoints,
    int nOctaves, int nOctaveLayers,
    double contrastThreshold, double edgeThreshold, double sigma)
{
    const int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);
    const int max_kpts_per_call = 100000;

    UMat count_umat(1, 1, CV_32S, Scalar::all(0));
    UMat kp_buf(1, max_kpts_per_call * 6, CV_32F);

    for (int o = 0; o < nOctaves; o++)
    {
        int dog_cols = gauss_packs[o].cols;
        int dog_rows = gauss_packs[o].rows / (nOctaveLayers + 3);

        for (int i = 1; i <= nOctaveLayers; i++)
        {
            size_t globalSize[2] = {(size_t)dog_cols, (size_t)dog_rows};

            ocl::Kernel ker("SIFT_detect_and_orient", ocl::features::sift_oclsrc);
            if (ker.empty())
                return false;

            bool ok = ker.args(
                ocl::KernelArg::ReadOnlyNoSize(gauss_packs[o]),
                dog_cols, dog_rows,
                threshold,
                (float)contrastThreshold, (float)edgeThreshold, (float)sigma,
                nOctaveLayers, o, i,
                ocl::KernelArg::PtrWriteOnly(count_umat),
                ocl::KernelArg::PtrWriteOnly(kp_buf),
                max_kpts_per_call
            ).run(2, globalSize, 0, false);
            if (!ok)
                return false;
        }
    }

    ocl::finish();

    Mat count_mat = count_umat.getMat(ACCESS_READ);
    int total = count_mat.at<int>(0, 0);
    if (total > max_kpts_per_call)
        total = max_kpts_per_call;

    Mat kp = kp_buf.getMat(ACCESS_READ);
    const float* kdata = (const float*)kp.data;

    keypoints.resize(total);
    for (int i = 0; i < total; i++)
    {
        KeyPoint& kpt = keypoints[i];
        int base = i * 6;
        kpt.pt.x = kdata[base + 0];
        kpt.pt.y = kdata[base + 1];
        kpt.angle = kdata[base + 2];
        kpt.size = kdata[base + 3];
        kpt.response = kdata[base + 4];
        memcpy(&kpt.octave, &kdata[base + 5], sizeof(int));
        kpt.class_id = -1;
    }

    return true;
}

static bool ocl_computeSIFTDescriptors(
    const std::vector<UMat>& gauss_packs,
    const std::vector<KeyPoint>& keypoints,
    OutputArray _descriptors,
    int nOctaveLayers, int firstOctave, int descriptor_type)
{
    const int dsize = SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
    const int nkp = (int)keypoints.size();
    if (nkp == 0)
    {
        _descriptors.create(0, dsize, descriptor_type);
        return true;
    }

    const int levelsPerOctave = nOctaveLayers + 3;
    const int nOctaves = (int)gauss_packs.size();

    struct KptGroup { int start; int count; };
    std::vector<KptGroup> groups(nOctaves, {0, 0});

    for (int i = 0; i < nkp; i++)
    {
        int octave, layer; float scale;
        unpackOctave(keypoints[i], octave, layer, scale);
        int oct_idx = octave - firstOctave;
        if (oct_idx < 0 || oct_idx >= nOctaves)
            return false;
        groups[oct_idx].count++;
    }

    int offset = 0;
    for (int o = 0; o < nOctaves; o++)
    {
        groups[o].start = offset;
        offset += groups[o].count;
    }

    Mat all_kpts(nkp, 5, CV_32F);
    Mat all_rows(nkp, 1, CV_32S);
    std::vector<int> cursors(nOctaves, 0);

    for (int i = 0; i < nkp; i++)
    {
        int octave, layer; float scale;
        unpackOctave(keypoints[i], octave, layer, scale);
        int oct_idx = octave - firstOctave;
        float size = keypoints[i].size * scale;
        Point2f ptf(keypoints[i].pt.x * scale, keypoints[i].pt.y * scale);
        float angle = 360.f - keypoints[i].angle;
        if (std::abs(angle - 360.f) < FLT_EPSILON)
            angle = 0.f;
        float scl = size * 0.5f;
        int pos = groups[oct_idx].start + cursors[oct_idx]++;
        float* dst = all_kpts.ptr<float>(pos);
        dst[0] = ptf.x; dst[1] = ptf.y; dst[2] = angle; dst[3] = scl;
        dst[4] = (float)layer;
        all_rows.ptr<int>(pos)[0] = i;
    }

    UMat kpts_buf; all_kpts.copyTo(kpts_buf);
    UMat rows_buf; all_rows.copyTo(rows_buf);

    _descriptors.create(nkp, dsize, descriptor_type);
    UMat desc_out = _descriptors.getUMat();

    for (int o = 0; o < nOctaves; o++)
    {
        int nGroup = groups[o].count;
        if (nGroup == 0)
            continue;

        const UMat& packed = gauss_packs[o];
        int img_cols = packed.cols;
        int img_rows = packed.rows / levelsPerOctave;
        float diag = sqrt((float)img_cols * img_cols + (float)img_rows * img_rows);

        UMat kpts_roi = kpts_buf.rowRange(groups[o].start,
                                          groups[o].start + nGroup);

        ocl::Kernel ker("SIFT_compute_descriptor", ocl::features::sift_oclsrc);
        if (ker.empty())
            return false;

        const int KP_PER_WG = 4;
        const int WG_SIZE = KP_PER_WG * 16;
        size_t localsz = WG_SIZE;
        size_t globalsz = (size_t)(((nGroup + KP_PER_WG - 1) / KP_PER_WG) * WG_SIZE);
        bool ok = ker.args(
            ocl::KernelArg::ReadOnlyNoSize(packed), img_cols, img_rows, diag,
            ocl::KernelArg::ReadOnlyNoSize(kpts_roi),
            ocl::KernelArg::PtrReadOnly(rows_buf),
            groups[o].start,
            nGroup,
            ocl::KernelArg::WriteOnlyNoSize(desc_out),
            descriptor_type
        ).run(1, &globalsz, &localsz, false);
        if (!ok)
            return false;
    }

    return true;
}

} // namespace

bool SIFT_Impl::ocl_detectAndCompute(InputArray _image, InputArray _mask,
                                     std::vector<KeyPoint>& keypoints,
                                     OutputArray _descriptors, bool useProvidedKeypoints)
{
    UMat image = _image.getUMat();
    if (image.empty() || image.depth() != CV_8U)
        return false;

    int firstOctave = -1;
    int nOctaves = 0;
    if (useProvidedKeypoints)
    {
        firstOctave = 0;
        int maxOctave = INT_MIN;
        int actualNLayers = 0;
        for (size_t i = 0; i < keypoints.size(); i++)
        {
            int octave, layer; float scale;
            unpackOctave(keypoints[i], octave, layer, scale);
            firstOctave = std::min(firstOctave, octave);
            maxOctave = std::max(maxOctave, octave);
            actualNLayers = std::max(actualNLayers, layer-2);
        }
        firstOctave = std::min(firstOctave, 0);
        if (firstOctave < -1 || actualNLayers > nOctaveLayers)
            return false;
        nOctaves = maxOctave - firstOctave + 1;
    }

    UMat base = ocl_createInitialImage(image, firstOctave < 0, (float)sigma, enable_precise_upscale);
    if (base.empty())
        return false;
    if (!useProvidedKeypoints)
        nOctaves = cvRound(std::log((double)std::min(base.cols, base.rows)) / std::log(2.) - 2) - firstOctave;
    if (nOctaves <= 0)
        return false;

    std::vector<UMat> gauss_packs;
    if (!ocl_buildGaussianPyramid(base, gauss_packs, nOctaves, nOctaveLayers, sigma))
        return false;

    if (!useProvidedKeypoints)
    {
        if (!ocl_detectAndOrient(gauss_packs, keypoints, nOctaves, nOctaveLayers,
                                 contrastThreshold, edgeThreshold, sigma))
            return false;

        KeyPointsFilter::removeDuplicatedSorted(keypoints);
        if (nfeatures > 0)
            KeyPointsFilter::retainBest(keypoints, nfeatures);

        const float scale = 1.f/(float)(1 << -firstOctave);
        for (size_t i = 0; i < keypoints.size(); i++)
        {
            KeyPoint& kpt = keypoints[i];
            kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
            kpt.pt *= scale;
            kpt.size *= scale;
        }

        if (!_mask.empty())
        {
            Mat mask = _mask.getMat();
            KeyPointsFilter::runByPixelsMask(keypoints, mask);
        }

        if (_descriptors.needed())
        {
            if (!ocl_computeSIFTDescriptors(gauss_packs, keypoints, _descriptors, nOctaveLayers, firstOctave, descriptor_type))
                return false;
        }
    }
    else
    {
        if (_descriptors.needed())
        {
            if (!ocl_computeSIFTDescriptors(gauss_packs, keypoints, _descriptors, nOctaveLayers, firstOctave, descriptor_type))
                return false;
        }
    }
    return true;
}

#endif // HAVE_OPENCL

void SIFT_Impl::detectAndCompute(InputArray _image, InputArray _mask,
                      std::vector<KeyPoint>& keypoints,
                      OutputArray _descriptors,
                      bool useProvidedKeypoints)
{
    CV_TRACE_FUNCTION();

#ifdef HAVE_OPENCL
    if (ocl::isOpenCLActivated() && _image.isUMat() && sizeof(void*) > 4)
    {
        ocl::Device d = ocl::Device::getDefault();
        bool isDiscreteGPU = (d.type() == ocl::Device::TYPE_GPU) && !d.hostUnifiedMemory();
        bool ocl11OrLater = (d.deviceVersionMajor() > 1) || (d.deviceVersionMajor() == 1 && d.deviceVersionMinor() >= 1);
        if (isDiscreteGPU && ocl11OrLater && ocl_detectAndCompute(_image, _mask, keypoints, _descriptors, useProvidedKeypoints))
        {
            CV_IMPL_ADD(CV_IMPL_OCL);
            return;
        }
    }
#endif

    int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
    Mat image = _image.getMat(), mask = _mask.getMat();

    if( image.empty() || image.depth() != CV_8U )
        CV_Error( Error::StsBadArg, "image is empty or has incorrect depth (!=CV_8U)" );

    if( !mask.empty() && mask.type() != CV_8UC1 )
        CV_Error( Error::StsBadArg, "mask has incorrect type (!=CV_8UC1)" );

    if( useProvidedKeypoints )
    {
        firstOctave = 0;
        int maxOctave = INT_MIN;
        for( size_t i = 0; i < keypoints.size(); i++ )
        {
            int octave, layer;
            float scale;
            unpackOctave(keypoints[i], octave, layer, scale);
            firstOctave = std::min(firstOctave, octave);
            maxOctave = std::max(maxOctave, octave);
            actualNLayers = std::max(actualNLayers, layer-2);
        }

        firstOctave = std::min(firstOctave, 0);
        CV_Assert( firstOctave >= -1 && actualNLayers <= nOctaveLayers );
        actualNOctaves = maxOctave - firstOctave + 1;
    }

    Mat base = createInitialImage(image, firstOctave < 0, (float)sigma, enable_precise_upscale);
    std::vector<Mat> gpyr;
    int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(std::log( (double)std::min( base.cols, base.rows ) ) / std::log(2.) - 2) - firstOctave;

    //double t, tf = getTickFrequency();
    //t = (double)getTickCount();
    buildGaussianPyramid(base, gpyr, nOctaves);

    //t = (double)getTickCount() - t;
    //printf("pyramid construction time: %g\n", t*1000./tf);

    if( !useProvidedKeypoints )
    {
        std::vector<Mat> dogpyr;
        buildDoGPyramid(gpyr, dogpyr);
        //t = (double)getTickCount();
        findScaleSpaceExtrema(gpyr, dogpyr, keypoints);
        KeyPointsFilter::removeDuplicatedSorted( keypoints );

        if( nfeatures > 0 )
            KeyPointsFilter::retainBest(keypoints, nfeatures);
        //t = (double)getTickCount() - t;
        //printf("keypoint detection time: %g\n", t*1000./tf);

        if( firstOctave < 0 )
            for( size_t i = 0; i < keypoints.size(); i++ )
            {
                KeyPoint& kpt = keypoints[i];
                float scale = 1.f/(float)(1 << -firstOctave);
                kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
                kpt.pt *= scale;
                kpt.size *= scale;
            }

        if( !mask.empty() )
            KeyPointsFilter::runByPixelsMask( keypoints, mask );
    }
    else
    {
        // filter keypoints by mask
        //KeyPointsFilter::runByPixelsMask( keypoints, mask );
    }

    if( _descriptors.needed() )
    {
        //t = (double)getTickCount();
        int dsize = descriptorSize();
        _descriptors.create((int)keypoints.size(), dsize, descriptor_type);

        Mat descriptors = _descriptors.getMat();
        calcDescriptors(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
        //t = (double)getTickCount() - t;
        //printf("descriptor extraction time: %g\n", t*1000./tf);
    }
}

void SIFT_Impl::read( const FileNode& fn)
{
  // if node is empty, keep previous value
  if (!fn["nfeatures"].empty())
    fn["nfeatures"] >> nfeatures;
  if (!fn["nOctaveLayers"].empty())
    fn["nOctaveLayers"] >> nOctaveLayers;
  if (!fn["contrastThreshold"].empty())
    fn["contrastThreshold"] >> contrastThreshold;
  if (!fn["edgeThreshold"].empty())
    fn["edgeThreshold"] >> edgeThreshold;
  if (!fn["sigma"].empty())
    fn["sigma"] >> sigma;
  if (!fn["descriptorType"].empty())
    fn["descriptorType"] >> descriptor_type;
}
void SIFT_Impl::write( FileStorage& fs) const
{
  if(fs.isOpened())
  {
    fs << "name" << getDefaultName();
    fs << "nfeatures" << nfeatures;
    fs << "nOctaveLayers" << nOctaveLayers;
    fs << "contrastThreshold" << contrastThreshold;
    fs << "edgeThreshold" << edgeThreshold;
    fs << "sigma" << sigma;
    fs << "descriptorType" << descriptor_type;
  }
}

}
