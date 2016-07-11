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
#include "bm3d_denoising_transforms.hpp"
#include "arrays.hpp"

using namespace cv;

template <typename T, typename IT, typename UIT, typename D, typename WT>
struct Bm3dDenoisingInvoker :
    public ParallelLoopBody
{
public:
    Bm3dDenoisingInvoker(
        const Mat& src,
        Mat& dst,
        const int &templateWindowSize,
        const int &searchWindowSize,
        const float &h,
        const int &hBM,
        const int &groupSize);

    void operator() (const Range& range) const;
    virtual ~Bm3dDenoisingInvoker();

private:
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
    void(*haarTransform2D)(const T *ptr, short *dst, const int &step);
    void(*inverseHaar2D)(short *src);

    // Threshold map
    short *thrMap_;
};

template <typename T, typename IT, typename UIT, typename D, typename WT>
Bm3dDenoisingInvoker<T, IT, UIT, D, WT>::Bm3dDenoisingInvoker(
    const Mat& src,
    Mat& dst,
    const int &templateWindowSize,
    const int &searchWindowSize,
    const float &h,
    const int &hBM,
    const int &groupSize) :
    src_(src), dst_(dst), groupSize_(groupSize)
{
    groupSize_ = getLargestPowerOf2SmallerThan(groupSize);
    CV_Assert(groupSize <= BM3D_MAX_3D_SIZE && groupSize > 0);

    halfTemplateWindowSize_ = templateWindowSize >> 1;
    halfSearchWindowSize_ = searchWindowSize >> 1;
    templateWindowSize_ = halfTemplateWindowSize_ << 1;
    searchWindowSize_ = (halfSearchWindowSize_ << 1);
    templateWindowSizeSq_ = templateWindowSize_ * templateWindowSize_;
    searchWindowSizeSq_ = searchWindowSize_ * searchWindowSize_;

#ifdef DEBUG_PRINT
    printf("Inside Bm3dDenoisingInvoker...\n");
    printf("groupSize: %d\n", groupSize);
    printf("groupSize_: %d\n", groupSize_);
    printf("templateWIndowSize: %d\n", templateWindowSize);
    printf("searchWindowSize: %d\n", searchWindowSize);
    printf("templateWindowSize_: %d\n", templateWindowSize_);
    printf("templateWindowSizeSq_: %d\n", templateWindowSizeSq_);
    printf("searchWindowSize_: %d\n", searchWindowSize_);
    printf("searchWindowSizeSq_: %d\n", searchWindowSizeSq_);
#endif

    // Extend image to avoid border problem
    borderSize_ = halfSearchWindowSize_ + halfTemplateWindowSize_;
    copyMakeBorder(src_, srcExtended_, borderSize_, borderSize_, borderSize_, borderSize_, BORDER_DEFAULT);

    // Allocate memory for threshold map
    thrMap_ = new short[templateWindowSizeSq_ * ((BM3D_MAX_3D_SIZE << 1) - 1)];

    // Calculate block matching threshold
    hBM_ = D::template calcBlockMatchingThreshold<int>(hBM, templateWindowSizeSq_);

    switch (templateWindowSize_)
    {
    case 4:
        // Precompute threshold map
        ComputeThresholdMap1D(thrMap_, kThrMap1D, kThrMap4x4, h, kCoeff, templateWindowSizeSq_);

        // Select transforms
        haarTransform2D = Haar4x4;
        inverseHaar2D = InvHaar4x4;
        break;
    default:
        CV_Error(Error::StsBadArg,
            "Unsupported template size! Only 1, 2 and 4 are supported currently.");
    }
}

template<typename T, typename IT, typename UIT, typename D, typename WT>
inline Bm3dDenoisingInvoker<T, IT, UIT, D, WT>::~Bm3dDenoisingInvoker()
{
    delete[] thrMap_;
}

template <typename T, typename IT, typename UIT, typename D, typename WT>
void Bm3dDenoisingInvoker<T, IT, UIT, D, WT>::operator() (const Range& range) const
{
    const int size = (range.size() + 2 * borderSize_) * srcExtended_.cols;
    std::vector<float> weightedSum(size, 0.0);
    std::vector<float> weights(size, 0.0);
    int row_from = range.start;
    int row_to = range.end - 1;

#ifdef DEBUG_PRINT
    printf("Size of the weights is: %d\n", size);
    printf("srcExtended_.total(): %d\n", srcExtended_.total());
    printf("range.size(): %d\n", range.size());
    printf("borderSize_: %d\n", borderSize_);
    printf("srcExtended_.cols: %d\n", srcExtended_.cols);
    printf("sizeof(T): %d\n", sizeof(T));
    printf("from %d to %d\n", row_from, row_to);
#endif

    // Local vars for faster processing
    const int blockSize = templateWindowSize_;
    const int blockSizeSq = templateWindowSizeSq_;
    const int halfBlockSize = halfTemplateWindowSize_;
    const int searchWindowSize = searchWindowSize_;
    const int searchWindowSizeSq = searchWindowSizeSq_;
    const int halfSearchWindowSize = halfSearchWindowSize_;
    const int hBM = hBM_;
    const int groupSize = groupSize_;

    const int step = srcExtended_.step / sizeof(T);
    const int cstep = step - templateWindowSize_;
    const int csstep = step - searchWindowSize_;

    const int dstStep = srcExtended_.cols;
    const int weiStep = srcExtended_.cols;
    const int dstcstep = dstStep - blockSize;
    const int weicstep = weiStep - blockSize;

    // Buffers
    short *r = new short[blockSizeSq];    // reference block
    short **z = new short*[searchWindowSizeSq];  // 3D array
    for (int i = 0; i < searchWindowSizeSq; ++i)
        z[i] = new short[blockSizeSq];
    int *dist = new int[searchWindowSizeSq];

    // Relative coordinates to the current search window
    short *coords_x = new short[searchWindowSizeSq];
    short *coords_y = new short[searchWindowSizeSq];

#ifdef DEBUG_PRINT
    printf("Entering the loop...\n");
#endif

    for (int j = row_from, jj = 0; j <= row_to; ++j, ++jj)
    {
#ifdef DEBUG_PRINT
        if (j % 50 == 0)
            printf("%d / %d\n", j, row_to);
#endif

        for (int i = 0; i < src_.cols; ++i)
        {
            const T *referencePatch = srcExtended_.ptr<T>(0) + step*(halfSearchWindowSize + j) + (halfSearchWindowSize + i);
            const T *currentPixel = srcExtended_.ptr<T>(0) + step*j + i;

            haarTransform2D(referencePatch, r, step);

            int elementSize = 0;
            for (short l = 0; l < searchWindowSize; ++l)
            {
                const T *candidatePatch = currentPixel + step*l;
                for (short k = 0; k < searchWindowSize; ++k)
                {
                    haarTransform2D(candidatePatch + k, z[elementSize], step);

                    // Calc distance
                    int e = 0;
                    for (int n = blockSizeSq; n--;)
                        e += D::template calcDist<short>(z[elementSize][n], r[n]);

                    // Increase the counter and save the distance
                    if (e < hBM)
                    {
                        dist[elementSize] = e;

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
            elementSize = getLargestPowerOf2SmallerThan(elementSize);
            if (elementSize > groupSize)
                elementSize = groupSize;

            // Transform and shrink 1D columns
            short sumNonZero = 0;
            short *thrMapPtr1D = thrMap_ + (elementSize - 1) * blockSizeSq;
            switch (elementSize)
            {
            case 8:
                for (int n = 0; n < blockSizeSq; n++)
                {
                    sumNonZero += HaarTransformShrink8(z, n, thrMapPtr1D);
                    InverseHaarTransform8(z, n);
                }
                break;

            case 4:
                for (int n = 0; n < blockSizeSq; n++)
                {
                    sumNonZero += HaarTransformShrink4(z, n, thrMapPtr1D);
                    InverseHaarTransform4(z, n);
                }
                break;

            case 2:
                for (int n = 0; n < blockSizeSq; n++)
                {
                    sumNonZero += HaarTransformShrink2(z, n, thrMapPtr1D);
                    InverseHaarTransform2(z, n);
                }
                break;

            case 1:
                for (int n = 0; n < blockSizeSq; n++)
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
            float weight = 1.0f / (float)sumNonZero;

            // Scale weight by element size
            weight *= elementSize;
            weight /= BM3D_MAX_3D_SIZE;

            // Put patches back to their original positions
            float *dstPtr = weightedSum.data() + jj * dstStep + i;
            float *weiPtr = weights.data() + jj * dstStep + i;

            for (int l = 0; l < elementSize; ++l)
            {
                int offset = coords_y[l] * dstStep + coords_x[l];
                float *d = dstPtr + offset;
                float *dw = weiPtr + offset;

                for (int n = 0; n < blockSize; ++n)
                {
                    for (int m = 0; m < blockSize; ++m)
                    {
                        *d += z[l][n * blockSize + m] * weight;
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
    delete[] r;
    for (int i = 0; i < searchWindowSizeSq; ++i)
        delete[] z[i];
    delete[] z;
    delete[] dist;
    delete[] coords_x;
    delete[] coords_y;

#ifdef DEBUG_PRINT
    printf("Aggregate results...\n");
#endif

    // Divide accumulation buffer by the corresponding weights
    for (int i = row_from, ii = 0; i <= row_to; ++i, ++ii)
    {
        T *d = dst_.ptr<T>(i);
        float *dE = weightedSum.data() + (ii + halfSearchWindowSize + halfBlockSize) * dstStep + halfSearchWindowSize;
        float *dw = weights.data() + (ii + halfSearchWindowSize + halfBlockSize) * dstStep + halfSearchWindowSize;
        for (int j = 0; j < dst_.cols; ++j)
            d[j] = cv::saturate_cast<T>(dE[j + halfBlockSize] / dw[j + halfBlockSize]);
    }

#ifdef DEBUG_PRINT
    printf("All done.\n");
#endif
}

#endif