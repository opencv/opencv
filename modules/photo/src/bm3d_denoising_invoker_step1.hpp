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

#ifndef __OPENCV_BM3D_DENOISING_INVOKER_STEP1_HPP__
#define __OPENCV_BM3D_DENOISING_INVOKER_STEP1_HPP__

#include "precomp.hpp"
#include <limits>

#include "bm3d_denoising_invoker_commons.hpp"
#include "bm3d_denoising_transforms.hpp"
#include "arrays.hpp"

using namespace cv;

template <typename T, typename IT, typename UIT, typename D, typename WT, typename TT>
struct Bm3dDenoisingInvokerStep1 : public ParallelLoopBody
{
public:
    Bm3dDenoisingInvokerStep1(
        const Mat& src,
        Mat& dst,
        const int &templateWindowSize,
        const int &searchWindowSize,
        const float &h,
        const int &hBM,
        const int &groupSize);

    virtual ~Bm3dDenoisingInvokerStep1();
    void operator() (const Range& range) const;

private:
    void operator= (const Bm3dDenoisingInvokerStep1&);

    const Mat& src_;
    Mat& dst_;
    Mat srcExtended_;

    int borderSize_;

    int templateWindowSize_;
    int searchWindowSize_;

    int halfTemplateWindowSize_;
    int halfSearchWindowSize_;

    int templateWindowSizeSq_;
    int searchWindowSizeSq_;

    // Block matching threshold
    int hBM_;

    // Maximum size of 3D group
    int groupSize_;

    // Function pointers
    void(*haarTransform2D)(const T *ptr, TT *dst, const int &step);
    void(*inverseHaar2D)(TT *src);

    // Threshold map
    TT *thrMap_;
};

template <typename T, typename IT, typename UIT, typename D, typename WT, typename TT>
Bm3dDenoisingInvokerStep1<T, IT, UIT, D, WT, TT>::Bm3dDenoisingInvokerStep1(
    const Mat& src,
    Mat& dst,
    const int &templateWindowSize,
    const int &searchWindowSize,
    const float &h,
    const int &hBM,
    const int &groupSize) :
    src_(src), dst_(dst), groupSize_(groupSize), thrMap_(NULL)
{
    groupSize_ = getLargestPowerOf2SmallerThan(groupSize);
    CV_Assert(groupSize <= BM3D_MAX_3D_SIZE && groupSize > 0);

    halfTemplateWindowSize_ = templateWindowSize >> 1;
    halfSearchWindowSize_ = searchWindowSize >> 1;
    templateWindowSize_ = halfTemplateWindowSize_ << 1;
    searchWindowSize_ = (halfSearchWindowSize_ << 1);
    templateWindowSizeSq_ = templateWindowSize_ * templateWindowSize_;
    searchWindowSizeSq_ = searchWindowSize_ * searchWindowSize_;

    // Extend image to avoid border problem
    borderSize_ = halfSearchWindowSize_ + halfTemplateWindowSize_;
    copyMakeBorder(src_, srcExtended_, borderSize_, borderSize_, borderSize_, borderSize_, BORDER_DEFAULT);

    // Calculate block matching threshold
    hBM_ = D::template calcBlockMatchingThreshold<int>(hBM, templateWindowSizeSq_);

    // Select transforms depending on the template size
    switch (templateWindowSize_)
    {
    case 4:
        haarTransform2D = Haar4x4;
        inverseHaar2D = InvHaar4x4;
        break;
    case 8:
        haarTransform2D = Haar8x8;
        inverseHaar2D = InvHaar8x8;
        break;
    default:
        CV_Error(Error::StsBadArg,
            "Unsupported template size! Only 4 and 8 are supported currently.");
    }

    // Precompute threshold map
    calcHaarThresholdMap3D(thrMap_, h, templateWindowSize_, groupSize_);
}

template<typename T, typename IT, typename UIT, typename D, typename WT, typename TT>
inline Bm3dDenoisingInvokerStep1<T, IT, UIT, D, WT, TT>::~Bm3dDenoisingInvokerStep1()
{
    delete[] thrMap_;
}

template <typename T, typename IT, typename UIT, typename D, typename WT, typename TT>
void Bm3dDenoisingInvokerStep1<T, IT, UIT, D, WT, TT>::operator() (const Range& range) const
{
    const int size = (range.size() + 2 * borderSize_) * srcExtended_.cols;
    std::vector<WT> weightedSum(size, 0.0);
    std::vector<WT> weights(size, 0.0);
    int row_from = range.start;
    int row_to = range.end - 1;

    // Local vars for faster processing
    const int blockSize = templateWindowSize_;
    const int blockSizeSq = templateWindowSizeSq_;
    const int halfBlockSize = halfTemplateWindowSize_;
    const int searchWindowSize = searchWindowSize_;
    const int searchWindowSizeSq = searchWindowSizeSq_;
    const short halfSearchWindowSize = (short)halfSearchWindowSize_;
    const int hBM = hBM_;
    const int groupSize = groupSize_;

    const int step = srcExtended_.cols;
    const int cstep = step - templateWindowSize_;

    const int dstStep = srcExtended_.cols;
    const int weiStep = srcExtended_.cols;
    const int dstcstep = dstStep - blockSize;
    const int weicstep = weiStep - blockSize;

    // Buffer to store 3D group
    BlockMatch<TT, int, TT> *bm = new BlockMatch<TT, int, TT>[searchWindowSizeSq];
    for (int i = 0; i < searchWindowSizeSq; ++i)
        bm[i].init(blockSizeSq);

    // First element in a group is always the reference patch. Hence distance is 0.
    bm[0](0, halfSearchWindowSize, halfSearchWindowSize);

    for (int j = row_from, jj = 0; j <= row_to; ++j, ++jj)
    {
        for (int i = 0; i < src_.cols; ++i)
        {
            const T *referencePatch = srcExtended_.ptr<T>(0) + step*(halfSearchWindowSize + j) + (halfSearchWindowSize + i);
            const T *currentPixel = srcExtended_.ptr<T>(0) + step*j + i;

            int elementSize = 1;
            for (short l = 0; l < searchWindowSize; ++l)
            {
                const T *candidatePatch = currentPixel + step*l;
                for (short k = 0; k < searchWindowSize; ++k)
                {
                    if (l == halfSearchWindowSize && k == halfSearchWindowSize)
                        continue;

                    // Calc distance
                    int e = 0;
                    const T *canPtr = candidatePatch + k;
                    const T *refPtr = referencePatch;
                    for (int n = 0; n < blockSize; ++n)
                    {
                        for (int m = 0; m < blockSize; ++m)
                            e += D::template calcDist<TT>(*canPtr++, *refPtr++);
                        canPtr += cstep;
                        refPtr += cstep;
                    }

                    // Save the distance, coordinate and increase the counter
                    if (e < hBM)
                        bm[elementSize++](e, k, l);
                }
            }

            // Sort bm by distance (first element is already sorted)
            std::sort(bm + 1, bm + elementSize);

            // Find the nearest power of 2 and cap the group size from the top
            elementSize = getLargestPowerOf2SmallerThan(elementSize);
            if (elementSize > groupSize)
                elementSize = groupSize;

            // Transform 2D patches
            for (int n = 0; n < elementSize; ++n)
            {
                const T *candidatePatch = currentPixel + step * bm[n].coord_y + bm[n].coord_x;
                haarTransform2D(candidatePatch, bm[n].data(), step);
            }

            // Transform and shrink 1D columns
            short sumNonZero = 0;
            TT *thrMapPtr1D = thrMap_ + (elementSize - 1) * blockSizeSq;
            switch (elementSize)
            {
            case 16:
                for (int n = 0; n < blockSizeSq; n++)
                {
                    ForwardHaarTransform16(bm, n);
                    sumNonZero += HardThreshold<16>(bm, n, thrMapPtr1D);
                    InverseHaarTransform16(bm, n);
                }
                break;
            case 8:
                for (int n = 0; n < blockSizeSq; n++)
                {
                    ForwardHaarTransform8(bm, n);
                    sumNonZero += HardThreshold<8>(bm, n, thrMapPtr1D);
                    InverseHaarTransform8(bm, n);
                }
                break;
            case 4:
                for (int n = 0; n < blockSizeSq; n++)
                {
                    ForwardHaarTransform4(bm, n);
                    sumNonZero += HardThreshold<4>(bm, n, thrMapPtr1D);
                    InverseHaarTransform4(bm, n);
                }
                break;
            case 2:
                for (int n = 0; n < blockSizeSq; n++)
                {
                    ForwardHaarTransform2(bm, n);
                    sumNonZero += HardThreshold<2>(bm, n, thrMapPtr1D);
                    InverseHaarTransform2(bm, n);
                }
                break;
            case 1:
                {
                    TT *block = bm[0].data();
                    for (int n = 0; n < blockSizeSq; n++)
                        shrink(block[n], sumNonZero, *thrMapPtr1D++);
                }
                break;
            case 0:
            default:
                continue;
            }

            // Inverse 2D transform
            for (int n = 0; n < elementSize; ++n)
                inverseHaar2D(bm[n].data());

            // Aggregate the results (increase sumNonZero to avoid division by zero)
            float weight = 1.0f / (float)(++sumNonZero);

            // Scale weight by element size
            weight *= elementSize;
            weight /= groupSize;

            // Put patches back to their original positions
            WT *dstPtr = weightedSum.data() + jj * dstStep + i;
            WT *weiPtr = weights.data() + jj * dstStep + i;

            for (int l = 0; l < elementSize; ++l)
            {
                const TT *block = bm[l].data();
                int offset = bm[l].coord_y * dstStep + bm[l].coord_x;
                WT *d = dstPtr + offset;
                WT *dw = weiPtr + offset;

                for (int n = 0; n < blockSize; ++n)
                {
                    for (int m = 0; m < blockSize; ++m)
                    {
                        *d += block[n * blockSize + m] * weight;
                        *dw += weight;
                        ++d, ++dw;
                    }
                    d += dstcstep;
                    dw += weicstep;
                }
            }
        } // i
    } // j

    // Cleanup
    for (int i = 0; i < searchWindowSizeSq; ++i)
        bm[i].release();
    delete[] bm;

    // Divide accumulation buffer by the corresponding weights
    for (int i = row_from, ii = 0; i <= row_to; ++i, ++ii)
    {
        T *d = dst_.ptr<T>(i);
        float *dE = weightedSum.data() + (ii + halfSearchWindowSize + halfBlockSize) * dstStep + halfSearchWindowSize;
        float *dw = weights.data() + (ii + halfSearchWindowSize + halfBlockSize) * dstStep + halfSearchWindowSize;
        for (int j = 0; j < dst_.cols; ++j)
            d[j] = cv::saturate_cast<T>(dE[j + halfBlockSize] / dw[j + halfBlockSize]);
    }
}

#endif