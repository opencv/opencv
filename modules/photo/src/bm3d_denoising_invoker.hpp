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

#define BM3D_HALF_BLOCK_SIZE 2
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
    Bm3dDenoisingInvoker(
        const Mat& src,
        Mat& dst,
        int templateWindowSize,
        int searchWindowSize,
        const float &h,
        Mutex &lock);

    void operator() (const Range& range) const;

private:
    //void operator= (const Bm3dDenoisingInvoker&);

    const Mat& src_;
    Mat& dst_;
    Mat srcExtended_;
    Mat *dstExtended_;
    Mat *weights_;

    // Shared lock for thread safety
    Mutex &lock_;

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
    void(*haarTransform2D)(const T *ptr, short *dst, const short &step);
    void(*inverseHaar2D)(short *src);

    // Threshold maps
    short *thrMap2D;
    short *thrMap2Dpre;
    short *thrMap1D;
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
    const Mat& src,
    Mat& dst,
    int templateWindowSize,
    int searchWindowSize,
    const float &h,
    Mutex &lock) :
    src_(src), dst_(dst), dstExtended_(new cv::Mat), weights_(new cv::Mat), lock_(lock)
{
    printf("Inside Bm3dDenoisingInvoker...\n");

    CV_Assert(src.channels() == pixelInfo<T>::channels);

    printf("templateWIndowSize: %d\n", templateWindowSize);
    halfTemplateWindowSize_ = templateWindowSize >> 1;
    halfSearchWindowSize_ = searchWindowSize >> 1;
    templateWindowSize_ = halfTemplateWindowSize_ << 1;
    printf("templateWindowSize_: %d\n", templateWindowSize_);
    searchWindowSize_ = halfSearchWindowSize_ << 1 + 1;
    templateWindowSizeSq_ = templateWindowSize_ * templateWindowSize_;
    searchWindowSizeSq_ = searchWindowSize_ * searchWindowSize_;

    borderSize_ = halfSearchWindowSize_ + halfTemplateWindowSize_;
    copyMakeBorder(src_, srcExtended_, borderSize_, borderSize_, borderSize_, borderSize_, BORDER_DEFAULT);

    printf("Assigning dstExt and weights.\n");
    *dstExtended_ = Mat::zeros(srcExtended_.size(), CV_32FC1);
    *weights_ = Mat::zeros(srcExtended_.size(), CV_32FC1);
    printf("Assigning dstExt and weights... done.\n");

    // Precompute threshold maps
    printf("Precomputing threshold maps...\n");
    const float hardThrPre2D = 0;
    const float hardThr1D = h;
    const float hardThr2D = h;
    const float hardThrDC = 0.25;

    // Threshold maps for 2D filtering
    thrMap2D = new short[BM3D_BLOCK_SIZE_SQ];
    thrMap2Dpre = new short[BM3D_BLOCK_SIZE_SQ];

    printf("2D...\n");
    ComputeThresholdMap2D(thrMap2D, kThrMap2D, hardThr2D, kCoeff2D, BM3D_BLOCK_SIZE_SQ, true);
    ComputeThresholdMap2D(thrMap2Dpre, kThrMap2D, hardThrPre2D, kCoeff2D, BM3D_BLOCK_SIZE_SQ, false);
    printf("2D... Done.\n");

    // Set DC components filtering
    kThrMap2D[0] = hardThrDC;

    // Threshold map for 1D filtering
    thrMap1D = new short[BM3D_BLOCK_SIZE_SQ * ((BM3D_MAX_3D_SIZE << 1) - 1)];
    printf("1D...\n");
    ComputeThresholdMap1D(thrMap1D, kThrMap1D, kThrMap2D, hardThr1D, kCoeff, BM3D_BLOCK_SIZE_SQ);
    printf("1D... Done.\n");

    switch (templateWindowSize_)
    {
    case 4:
        haarTransform2D = Haar4x4;
        inverseHaar2D = InvHaar4x4;
        break;
    default:
        printf("templateWindowSize_: %d\n", templateWindowSize_);
        CV_Error(Error::StsBadArg, "Unsupported template size! Currently supported is only size of 4.");
    }

    printf("Bm3dDenoisingInvoker done.\n");
}

template <typename T, typename IT, typename UIT, typename D, typename WT>
void Bm3dDenoisingInvoker<T, IT, UIT, D, WT>::operator() (const Range& range) const
{
    const short bmThr = 1;

    const int step = srcExtended_.step / sizeof(T);
    const int cstep = step - BM3D_BLOCK_SIZE;
    const int csstep = step - searchWindowSize_;

    const int dstStep = dstExtended_->step / sizeof(float);
    const int weiStep = weights_->step / sizeof(float);
    const int dstcstep = dstStep - BM3D_BLOCK_SIZE;
    const int weicstep = weiStep - BM3D_BLOCK_SIZE;

    int row_from = range.start;
    int row_to = range.end - 1;

    printf("from %d to %d\n", row_from, row_to);

    // Local vars for faster processing
    const int searchWindowSizeSq = searchWindowSizeSq_;
    const int halfSearchWindowSize = halfSearchWindowSize_;

    printf("Entering the loop...\n");

    for (int j = row_from; j <= row_to; ++j)
    {
        if (j % 50 == 0)
            printf("%d / %d\n", j, row_to);

        // For shirnkage
        short *r = new short[BM3D_BLOCK_SIZE_SQ];  // reference block
        short **z = new short*[searchWindowSizeSq];  // 3D array
        for (int i = 0; i < searchWindowSizeSq; ++i)
            z[i] = new short[BM3D_BLOCK_SIZE_SQ];
        short *dist = new short[searchWindowSizeSq];
        short *coords_x = new short[searchWindowSizeSq];
        short *coords_y = new short[searchWindowSizeSq];

        for (int i = 0; i < src_.cols; ++i)
        {
            const T *referencePatch = srcExtended_.ptr<T>(0) + step*(halfSearchWindowSize + j) + (halfSearchWindowSize + i);
            const T *currentPixel = srcExtended_.ptr<T>(0) + step*j + i;

            haarTransform2D(referencePatch, r, step);
            hardThreshold2D(r, thrMap2Dpre, BM3D_BLOCK_SIZE_SQ);

            int elementSize = 0;
            for (int l = 0; l < searchWindowSize_; ++l)
            {
                const T *candidatePatch = currentPixel + step*l;
                for (int k = 0; k < searchWindowSize_; ++k)
                {
                    haarTransform2D(candidatePatch + k, z[elementSize], step);
                    hardThreshold2D(z[elementSize], thrMap2Dpre, BM3D_BLOCK_SIZE_SQ);

                    // Calc distance
                    int e = 0;
                    for (int n = BM3D_BLOCK_SIZE_SQ; n--;)
                        e += (z[elementSize][n] - r[n]) * (z[elementSize][n] - r[n]);
                    e /= BM3D_BLOCK_SIZE_SQ;

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
            //elementSize = getNearestPowerOf2(elementSize);
            if (elementSize > BM3D_MAX_3D_SIZE)
                elementSize = BM3D_MAX_3D_SIZE;
            else if (elementSize > 4)
                elementSize = 4;
            else if (elementSize > 2)
                elementSize = 2;

            // Shrink in 2D
            for (int k = 0; k < elementSize; ++k)
                hardThreshold2D(z[k], thrMap2D, BM3D_BLOCK_SIZE_SQ);

            // Transform and shrink 1D columns
            short sumNonZero = 0;
            short *thrMapPtr1D = thrMap1D + (elementSize - 1) * BM3D_BLOCK_SIZE_SQ;
            switch (elementSize)
            {
            case 8:
                for (int n = 0; n < BM3D_BLOCK_SIZE_SQ; n++)
                {
                    sumNonZero += HaarTransformShrink8(z, n, thrMapPtr1D);
                    InverseHaarTransform8(z, n);
                }
                break;

            case 4:
                for (int n = 0; n < BM3D_BLOCK_SIZE_SQ; n++)
                {
                    sumNonZero += HaarTransformShrink4(z, n, thrMapPtr1D);
                    InverseHaarTransform4(z, n);
                }
                break;

            case 2:
                for (int n = 0; n < BM3D_BLOCK_SIZE_SQ; n++)
                {
                    sumNonZero += HaarTransformShrink2(z, n, thrMapPtr1D);
                    InverseHaarTransform2(z, n);
                }
                break;

            case 1:
                for (int n = 0; n < BM3D_BLOCK_SIZE_SQ; n++)
                {
                    shrink(z[0][n], sumNonZero, *thrMapPtr1D++);
                }
                break;
            case 0:
            default:
                continue;
            }

            // Inverse 2D transform
            for (int n = elementSize; n--;)
                inverseHaar2D(z[n]);

            // Aggregate the results
            ++sumNonZero;
            float weight = 1.0 / (float)sumNonZero;

            // Scale weight by element size
            weight *= elementSize;
            weight /= BM3D_MAX_3D_SIZE;

            // Put patches back to their original positions
            float *dstPtr = dstExtended_->ptr<float>(j) + i;
            float *weiPtr = weights_->ptr<float>(j) + i;

            for (int l = 0; l < elementSize; ++l)
            {
                int offset = coords_y[l] * dstStep + coords_x[l];
                float *d = dstPtr + offset;
                float *dw = weiPtr + offset;

                for (int n = 0; n < BM3D_BLOCK_SIZE; ++n)
                {
                    for (int m = 0; m < BM3D_BLOCK_SIZE; ++m)
                    {
                        float curWeight = weight;// *kernel[n * BM3D_BLOCK_SIZE + m];
                        *d += z[l][n * BM3D_BLOCK_SIZE + m] * curWeight;
                        *dw += curWeight;
                        ++d, ++dw;
                    }
                    d += dstcstep;
                    dw += weicstep;
                }
            }
        } // i

        delete[] r;
        delete[] dist;
        for (int i = 0; i < searchWindowSizeSq; ++i)
            delete[] z[i];
        delete[] coords_x;
        delete[] coords_y;
    } // j

    printf("Aggregate results...\n");
    // Divide accumulation buffer by the corresponding weights
    for (int i = row_from; i <= row_to; ++i)
    {
        T *d = dst_.ptr<T>(i);
        float *dE = dstExtended_->ptr<float>(i + halfSearchWindowSize + BM3D_HALF_BLOCK_SIZE) + halfSearchWindowSize;
        float *dw = weights_->ptr<float>(i + halfSearchWindowSize + BM3D_HALF_BLOCK_SIZE) + halfSearchWindowSize;
        for (int j = 0; j < dst_.cols; ++j)
            d[j] = cv::saturate_cast<T>(dE[j + BM3D_HALF_BLOCK_SIZE] / dw[j + BM3D_HALF_BLOCK_SIZE]);
    }

    printf("All done.\n");
}

#endif