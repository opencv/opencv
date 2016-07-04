/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective icvers.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#ifndef __OPENCV_BM3D_DENOISING_INVOKER_HPP__
#define __OPENCV_BM3D_DENOISING_INVOKER_HPP__

#include "precomp.hpp"
#include <limits>

#include "bm3d_denoising_invoker_commons.hpp"
#include "arrays.hpp"

using namespace cv;

#define BM3D_BLOCK_SIZE 4
#define BM3D_BLOCK_SIZE_SQ 16
#define BM3D_MAX_3D_SIZE 8
const float sqrt2 = std::sqrt(2);

static float kThrMap2D[BM3D_BLOCK_SIZE_SQ] = {
    0,          0.5,        1 / sqrt2,  1 / sqrt2,
    0.5,        1,          sqrt2,      sqrt2,
    1 / sqrt2,  sqrt2,      2,          2,
    1 / sqrt2,  sqrt2,      2,          2
};

static const float kThrMap1D[(BM3D_MAX_3D_SIZE << 1) - 1] = {
    1.0, //1 element
    1 / sqrt2,  sqrt2, // 2 elements
    0.5, 1.0, sqrt2, sqrt2,
    sqrt2 / 4, 1.0 / sqrt2, 1, 1, sqrt2, sqrt2, sqrt2, sqrt2
};

static const float kCoeff[4] = {
    1,
    std::sqrt(2 * std::log(2)),
    std::sqrt(2 * std::log(4)),
    std::sqrt(2 * std::log(8))
};

const float kCoeff2D = std::sqrt(2 * std::log(BM3D_BLOCK_SIZE * BM3D_BLOCK_SIZE));


template <typename T, typename IT, typename UIT, typename D, typename WT>
struct Bm3dDenoisingInvoker :
    public ParallelLoopBody
{
public:
    Bm3dDenoisingInvoker(const Mat& src, Mat& dst,
        int templateWindowSize, int searchWindowSize, const float &h);

    void operator() (const Range& range) const;

private:
    void operator= (const Bm3dDenoisingInvoker&);

    const Mat& src_;
    Mat& dst_;
    Mat srcExtended_;
    Mat dstExtended_;
    Mat weights_;

    int borderSize_;

    int templateWindowSize_;
    int searchWindowSize_;

    int halfTemplateWindowSize_;
    int halfSearchWindowSize_;

    int templateWindowSizeSq_;
    int searchWindowSizeSq_;

    typename pixelInfo<WT>::sampleType fixed_point_mult_;
    int almost_templateWindowSizeSq_bin_shift_;
    std::vector<WT> almost_dist2weight_;

    // Function pointers
    void(*haarTransform2D)(T *ptr, short *dst, const short &step);
    void(*inverseHaar2D)(short *src);

    // Threshold maps
    short *thrMap2D;
    short *thrMap2Dpre;
    short *thrMap1D;

    //void calcDistSumsForFirstElementInRow(
    //	int i, Array2d<int>& dist_sums,
    //	Array3d<int>& col_dist_sums,
    //	Array3d<int>& up_col_dist_sums) const;

    //void calcDistSumsForElementInFirstRow(
    //	int i, int j, int first_col_num,
    //	Array2d<int>& dist_sums,
    //	Array3d<int>& col_dist_sums,
    //	Array3d<int>& up_col_dist_sums) const;
};

inline int getNearestPowerOf2(int value)
{
    int p = 0;
    while (1 << p < value)
        ++p;
    return p;
}

template <typename T, typename IT, typename UIT, typename D, typename WT>
Bm3dDenoisingInvoker<T, IT, UIT, D, WT>::Bm3dDenoisingInvoker(
    const Mat& src, Mat& dst,
    int templateWindowSize,
    int searchWindowSize,
    const float &h) :
    src_(src), dst_(dst)
{
    CV_Assert(src.channels() == pixelInfo<T>::channels);

    halfTemplateWindowSize_ = templateWindowSize >> 1;
    halfSearchWindowSize_ = searchWindowSize >> 1;
    templateWindowSize_ = halfTemplateWindowSize_ << 1;
    searchWindowSize_ = halfSearchWindowSize_ << 1 + 1;
    templateWindowSizeSq_ = templateWindowSize_ * templateWindowSize_;

    borderSize_ = halfSearchWindowSize_ + halfTemplateWindowSize_;
    copyMakeBorder(src_, srcExtended_, borderSize_, borderSize_, borderSize_, borderSize_, BORDER_DEFAULT);

    const IT max_estimate_sum_value =
        (IT)searchWindowSize_ * (IT)searchWindowSize_ * (IT)pixelInfo<T>::sampleMax();
    fixed_point_mult_ = (int)std::min<IT>(std::numeric_limits<IT>::max() / max_estimate_sum_value,
        pixelInfo<WT>::sampleMax());

    // precalc weight for every possible l2 dist between blocks
    // additional optimization of precalced weights to replace division(averaging) by binary shift
    CV_Assert(templateWindowSize_ <= 46340); // sqrt(INT_MAX)
    almost_templateWindowSizeSq_bin_shift_ = getNearestPowerOf2(templateWindowSizeSq_);
    double almost_dist2actual_dist_multiplier = ((double)(1 << almost_templateWindowSizeSq_bin_shift_)) / templateWindowSizeSq_;

    int max_dist = D::template maxDist<T>();
    int almost_max_dist = (int)(max_dist / almost_dist2actual_dist_multiplier + 1);
    almost_dist2weight_.resize(almost_max_dist);

    for (int almost_dist = 0; almost_dist < almost_max_dist; almost_dist++)
    {
        double dist = almost_dist * almost_dist2actual_dist_multiplier;
        almost_dist2weight_[almost_dist] =
            D::template calcWeight<T, WT>(dist, h, fixed_point_mult_);
    }

    // additional optimization init end
    if (dst_.empty())
        dst_ = Mat::zeros(src_.size(), src_.type());

    //const int step = srcExtended_.step / sizeof(T);
    //const int cstep = step - templateWindowSize_;
    //const int csstep = step - searchWindowSize_;

    ////cv::Mat dstExteded = cv::Mat::zeros(srcExtended_.size(), CV_32FC1);
    ////cv::Mat weights = cv::Mat::zeros(srcExtended_.size(), CV_32FC1);

    //const int dstStep = dstExtended_.step / sizeof(float);
    //const int weiStep = dstExtended_.step / sizeof(float);
    //const int dstcstep = dstStep - templateWindowSize_;
    //const int weicstep = weiStep - templateWindowSize_;

    // Precompute thresholds
    const int hardThrPre2D = 0;
    const int hardThr1D = h;
    const int hardThr2D = h;
    const int hardThrDC = 0.25;

    // Threshold maps for 2D filtering
    thrMap2D = new short[templateWindowSizeSq_];
    thrMap2Dpre = new short[templateWindowSizeSq_];

    ComputeThresholdMap2D(thrMap2D, kThrMap2D, hardThr2D, kCoeff2D, templateWindowSizeSq_, true);
    ComputeThresholdMap2D(thrMap2Dpre, kThrMap2D, hardThrPre2D, kCoeff2D, templateWindowSizeSq_, false);

    // Set DC components filtering
    kThrMap2D[0] = hardThrDC;

    // Threshold map for 1D filtering
    thrMap1D = new short[templateWindowSizeSq_ * ((BM3D_MAX_3D_SIZE << 1) - 1)];
    ComputeThresholdMap1D(thrMap1D, kThrMap1D, kThrMap2D, hardThr1D, kCoeff, templateWindowSizeSq_);

    switch (templateWindowSize_)
    {
    case 4:
        haarTransform2D = Haar4x4;
        inverseHaar2D = InvHaar4x4;
        break;
    default:
        CV_Error(Error::StsBadArg, "Unsupported template size! Currently supported is only size of 4.");
    }
}

template <typename T, typename IT, typename UIT, typename D, typename WT>
void Bm3dDenoisingInvoker<T, IT, UIT, D, WT>::operator() (const Range& range) const
{
    const int step = srcExtended_.step / sizeof(T);
    const int cstep = step - templateWindowSize_;
    const int csstep = step - searchWindowSize_;

    const int dstStep = dstExtended_.step / sizeof(float);
    const int weiStep = dstExtended_.step / sizeof(float);
    const int dstcstep = dstStep - templateWindowSize_;
    const int weicstep = weiStep - templateWindowSize_;

    int row_from = range.start;
    int row_to = range.end - 1;

    for (int j = row_from; j <= row_to; ++j)
    {
        // For shirnkage
        short *r = new short[templateWindowSizeSq_];  // reference block
        short **z = new short*[searchWindowSizeSq_];  // 3D array
        for (int i = 0; i < searchWindowSizeSq_; ++i)
            z[i] = new short[templateWindowSizeSq_];
        int *dist = new short[templateWindowSizeSq_];
        short *coords_x = new short[searchWindowSizeSq_];
        short *coords_y = new short[searchWindowSizeSq_];

        for (int i = 0; i < src_.cols; ++i)
        {
            T *referencePatch = srcExtended_.ptr<T>(0) + step*(halfSearchWindowSize_ + j) + (halfSearchWindowSize_ + i);
            T *currentPixel = srcExtended_.ptr<T>(0) + step*j + i;

            haarTransform2D(referencePatch, r, step);
            HardThreshold2D(r, thrMap2Dpre);

            int elementSize = 0;
            for (int l = 0; l < searchWindowSize; ++l)
            {
                T *candidatePatch = currentPixel + step*l;
                for (int k = 0; k < searchWindowSize; ++k)
                {
                    haarTransform2D(candidatePatch + k, z[elementSize], step);
                    HardThreshold2D(z[elementSize], thrMap2Dpre);

                    // Calc distance
                    int e = 0;
                    for (int n = templateWindowSizeSq_; n--;)
                        e += (z[elementSize][n] - r[n]) * (z[elementSize][n] - r[n]);
                    e /= templateWindowSizeSq_;

                    // Increase the counter and save the distance
                    if (e < bmThr)
                    {
                        dist[elementSize] = 0;

                        // Save coords
                        coords_x[elementSize] = k;
                        coords_y[elementSize] = l;
                        ++elementSize;
                    }
                }
            }

            if (elementSize == 0)
                continue;

            // Sort z by distance
            for (int k = 0; k < elementSize; ++k)
            {
                for (int l = k + 1; l < elementSize; ++l)
                {
                    if (dist[l] < dist[k])
                    {
                        // swap pointers in z
                        short *temp = z[k];
                        z[k] = z[l];
                        z[l] = temp;

                        // swap dist
                        std::swap(dist[k], dist[l]);

                        // swap coords
                        std::swap(coords_x[k], coords_x[l]);
                        std::swap(coords_y[k], coords_y[l]);
                    }
                }
            }

            // Find the nearest power of 2 and cap the group size from the top
            elementSize = getNearestPowerOf2(elementSize);
            if (elementSize > BM3D_MAX_3D_SIZE)
                elementSize = BM3D_MAX_3D_SIZE;

            // Shrink in 2D
            for (int k = 0; k < elementSize; ++k)
                HardThreshold2D(z[k], thrMap2D);

            // Transform and shrink 1D columns
            short sumNonZero = 0;
            short *thrMapPtr1D = thrMap1D + (elementSize - 1) * templateWindowSizeSq_;
            switch (elementSize)
            {
            case 8:
                for (int n = 0; n < templateWindowSizeSq_; n++)
                {
                    sumNonZero += HaarTransformShrink8(z, n, thrMapPtr1D);
                    InverseHaarTransform8(z, n);
                }
                break;

            case 4:
                for (int n = 0; n < templateWindowSizeSq_; n++)
                {
                    sumNonZero += HaarTransformShrink4(z, n, thrMapPtr1D);
                    InverseHaarTransform4(z, n);
                }
                break;

            case 2:
                for (int n = 0; n < templateWindowSizeSq_; n++)
                {
                    sumNonZero += HaarTransformShrink2(z, n, thrMapPtr1D);
                    InverseHaarTransform2(z, n);
                }
                break;

            case 1:
                for (int n = 0; n < templateWindowSizeSq_; n++)
                {
                    shrink(z[0][n], sumNonZero, *thrMapPtr1D++);
                }
                break;
            }


            // Aggregate the results
            ++sumNonZero;
            float weight = 1.0 / (float)sumNonZero;

            // Scale weight by element size
            weight *= elementSize;
            weight /= BM3D_MAX_3D_SIZE;

            // Put patches back to their original positions
            float *dstPtr = dstExtended_.ptr<float < (j)+i;
            float *weiPtr = weights.ptr<float < (j)+i;

            for (int l = 0; l < elementSize; ++l)
            {
                const int offset = coords_y[l] * dststep + coords_x[l];
                float *d = dstPtr + offset;
                float *dw = weiPtr + offset;

                for (int n = 0; n < templateWindowSize_; ++n)
                {
                    for (int m = 0; m < templateWindowSize_; ++m)
                    {
                        float curWeight = weight;// *kernel[n * templateWindowSize_ + m];
                        *d += z[l][n * templateWindowSize_ + m] * curWeight;
                        *dw += curWeight;
                        ++d, ++dw;
                    }
                    d += dstcstep;
                    dw += weicstep;
                }
            } // i

            delete[] r;
            delete[] dist;
            for (int i = 0; i < searchWindowSizeSq_; ++i)
                delete[] z[i];
            delete[] coords_x;
            delete[] coords_y;
        }
    }
}

#endif