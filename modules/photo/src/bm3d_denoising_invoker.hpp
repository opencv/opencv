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

//#define VERIFY_TRANSFORMS

#define BM3D_HALF_BLOCK_SIZE 2
#define BM3D_BLOCK_SIZE 4
#define BM3D_BLOCK_SIZE_SQ 16
#define BM3D_MAX_3D_SIZE 8

const float sqrt2 = std::sqrt(2);

static float kThrMap4x4[BM3D_BLOCK_SIZE_SQ] = {
    0.25,       0.5,        1 / sqrt2,  1 / sqrt2,
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
        const int &templateWindowSize,
        const int &searchWindowSize,
        const float &h,
        const int &hBM,
        const int &groupSize);

    void operator() (const Range& range) const;
    virtual ~Bm3dDenoisingInvoker();

private:
    void operator= (const Bm3dDenoisingInvoker&);

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

    // Constant values
    const int hBM_;
    int groupSize_;

    // Function pointers
    void(*haarTransform2D)(const T *ptr, short *dst, const short &step);
    void(*inverseHaar2D)(short *src);

    // Threshold map
    short *thrMap_;
};

/// Round up to next higher power of 2 (return x if it's already a power
/// of 2).
inline int getLargestPowerOf2SmallerThan(int x)
{
    if (x > 8)
        return 8;
    else if (x > 4)
        return 4;
    else if (x > 2)
        return 2;
    else
        return x;
}

template <typename T, typename IT, typename UIT, typename D, typename WT>
Bm3dDenoisingInvoker<T, IT, UIT, D, WT>::Bm3dDenoisingInvoker(
    const Mat& src,
    Mat& dst,
    const int &templateWindowSize,
    const int &searchWindowSize,
    const float &h,
    const int &hBM,
    const int &groupSize) :
    src_(src), dst_(dst), hBM_(hBM), groupSize_(groupSize)
{
    CV_Assert(src.channels() == pixelInfo<T>::channels);
    CV_Assert(groupSize <= BM3D_MAX_3D_SIZE && groupSize > 0);

    groupSize_ = getLargestPowerOf2SmallerThan(groupSize);

    halfTemplateWindowSize_ = templateWindowSize >> 1;
    halfSearchWindowSize_ = searchWindowSize >> 1;
    templateWindowSize_ = halfTemplateWindowSize_ << 1;
    searchWindowSize_ = (halfSearchWindowSize_ << 1);// +1;
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

    // Precompute threshold map
    thrMap_ = new short[templateWindowSizeSq_ * ((BM3D_MAX_3D_SIZE << 1) - 1)];

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

#ifdef DEBUG_PRINT
static void Display3D(short **z, const int &groupSize)
{
    std::cout << "groupSize: " << groupSize << std::endl;

    for (int n = 0; n < groupSize; ++n)
    {
        for (int m = 0; m < BM3D_BLOCK_SIZE_SQ; ++m)
        {
            if (m % BM3D_BLOCK_SIZE == 0)
                std::cout << std::endl;
            std::cout << z[n][m] << "\t";
        }
        std::cout << std::endl;
    }
}
#endif

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
    const int searchWindowSize = searchWindowSize_;
    const int searchWindowSizeSq = searchWindowSizeSq_;
    const int halfSearchWindowSize = halfSearchWindowSize_;
    const int hBM = hBM_;

    const int step = srcExtended_.step / sizeof(T);
    const int cstep = step - templateWindowSize_;
    const int csstep = step - searchWindowSize_;

    const int dstStep = srcExtended_.cols;// dstExtended.step / sizeof(float);
    const int weiStep = srcExtended_.cols;// weights.step / sizeof(float);
    const int dstcstep = dstStep - BM3D_BLOCK_SIZE;
    const int weicstep = weiStep - BM3D_BLOCK_SIZE;

    // Buffers
    short *r = new short[BM3D_BLOCK_SIZE_SQ];    // reference block
    short **z = new short*[searchWindowSizeSq];  // 3D array
    for (int i = 0; i < searchWindowSizeSq; ++i)
        z[i] = new short[BM3D_BLOCK_SIZE_SQ];
    short *dist = new short[searchWindowSizeSq];

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
            //hardThreshold2D(r, thrMap2Dpre, BM3D_BLOCK_SIZE_SQ);

            int elementSize = 0;
            for (int l = 0; l < searchWindowSize; ++l)
            {
                const T *candidatePatch = currentPixel + step*l;
                for (int k = 0; k < searchWindowSize; ++k)
                {
                    haarTransform2D(candidatePatch + k, z[elementSize], step);
                    //hardThreshold2D(z[elementSize], thrMap2Dpre, BM3D_BLOCK_SIZE_SQ);

                    // Calc distance
                    int e = 0;
                    for (int n = BM3D_BLOCK_SIZE_SQ; n--;)
                        e += (z[elementSize][n] - r[n]) * (z[elementSize][n] - r[n]);
                    e /= BM3D_BLOCK_SIZE_SQ;

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
            if (elementSize > BM3D_MAX_3D_SIZE)
                elementSize = BM3D_MAX_3D_SIZE;

            //// Shrink in 2D
            //for (int k = 0; k < elementSize; ++k)
            //    hardThreshold2D(z[k], thrMap2D, BM3D_BLOCK_SIZE_SQ);

#if defined(DEBUG_PRINT) && defined(VERIFY_TRANSFORMS)
            std::cout << "z before transform:" << std::endl;
            for (int l = 0; l < elementSize; ++l)
            {
                const int offset = coords_y[l] * step + coords_x[l];
                const T *t = currentPixel + offset;
                for (int n = 0; n < BM3D_BLOCK_SIZE; ++n)
                {
                    for (int m = 0; m < BM3D_BLOCK_SIZE; ++m)
                    {
                        std::cout << (int)*t << " ";
                        ++t;
                    }
                    t += cstep;
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }

            std::cout << "z after transform:" << std::endl;
            Display3D(z, elementSize);
#endif

            // Transform and shrink 1D columns
            short sumNonZero = 0;
            short *thrMapPtr1D = thrMap_ + (elementSize - 1) * BM3D_BLOCK_SIZE_SQ;
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

#if defined(DEBUG_PRINT) && defined(VERIFY_TRANSFORMS)
            std::cout << "z after shrinkage:" << std::endl;
            Display3D(z, elementSize);
#endif

            // Inverse 2D transform
            for (int n = elementSize; n--;)
                inverseHaar2D(z[n]);

#if defined(DEBUG_PRINT) && defined(VERIFY_TRANSFORMS)
            std::cout << "z after inverse:" << std::endl;
            Display3D(z, elementSize);
#endif

            // Aggregate the results
            ++sumNonZero;
            float weight = 1.0 / (float)sumNonZero;

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

#ifdef DEBUG_PRINT
                int idx = jj * dstStep + i;
                if (idx + offset + BM3D_BLOCK_SIZE * dstcstep + BM3D_BLOCK_SIZE >= size)
                {
                    printf("j = %d, i = %d, idx: %d\n", j, i, idx);
                    printf("coords_x: %d, coords_y: %d\n", coords_x[l], coords_y[l]);
                }
#endif

                for (int n = 0; n < BM3D_BLOCK_SIZE; ++n)
                {
                    for (int m = 0; m < BM3D_BLOCK_SIZE; ++m)
                    {
                        *d += z[l][n * BM3D_BLOCK_SIZE + m] * weight;
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
        float *dE = weightedSum.data() + (ii + halfSearchWindowSize + BM3D_HALF_BLOCK_SIZE) * dstStep + halfSearchWindowSize;
        float *dw = weights.data() + (ii + halfSearchWindowSize + BM3D_HALF_BLOCK_SIZE) * dstStep + halfSearchWindowSize;
        for (int j = 0; j < dst_.cols; ++j)
            d[j] = cv::saturate_cast<T>(dE[j + BM3D_HALF_BLOCK_SIZE] / dw[j + BM3D_HALF_BLOCK_SIZE]);
    }

#ifdef DEBUG_PRINT
    printf("All done.\n");
#endif
}

#endif