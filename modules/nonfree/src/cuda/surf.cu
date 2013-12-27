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

#include "opencv2/opencv_modules.hpp"

#if defined(HAVE_OPENCV_GPU) && !defined(ANDROID)

#include "opencv2/gpu/device/common.hpp"
#include "opencv2/gpu/device/limits.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"
#include "opencv2/gpu/device/reduce.hpp"
#include "opencv2/gpu/device/utility.hpp"
#include "opencv2/gpu/device/functional.hpp"
#include "opencv2/gpu/device/filters.hpp"

namespace cv { namespace gpu { namespace device
{
    namespace surf
    {
        void loadGlobalConstants(int maxCandidates, int maxFeatures, int img_rows, int img_cols, int nOctaveLayers, float hessianThreshold);
        void loadOctaveConstants(int octave, int layer_rows, int layer_cols);

        void bindImgTex(PtrStepSzb img);
        size_t bindSumTex(PtrStepSz<unsigned int> sum);
        size_t bindMaskSumTex(PtrStepSz<unsigned int> maskSum);

        void icvCalcLayerDetAndTrace_gpu(const PtrStepf& det, const PtrStepf& trace, int img_rows, int img_cols,
            int octave, int nOctaveLayer);

        void icvFindMaximaInLayer_gpu(const PtrStepf& det, const PtrStepf& trace, int4* maxPosBuffer, unsigned int* maxCounter,
            int img_rows, int img_cols, int octave, bool use_mask, int nLayers);

        void icvInterpolateKeypoint_gpu(const PtrStepf& det, const int4* maxPosBuffer, unsigned int maxCounter,
            float* featureX, float* featureY, int* featureLaplacian, int* featureOctave, float* featureSize, float* featureHessian,
            unsigned int* featureCounter);

        void icvCalcOrientation_gpu(const float* featureX, const float* featureY, const float* featureSize, float* featureDir, int nFeatures);

        void compute_descriptors_gpu(PtrStepSz<float4> descriptors, const float* featureX, const float* featureY, const float* featureSize, const float* featureDir, int nFeatures);
    }
}}}

namespace cv { namespace gpu { namespace device
{
    namespace surf
    {
        ////////////////////////////////////////////////////////////////////////
        // Global parameters

        // The maximum number of features (before subpixel interpolation) that memory is reserved for.
        __constant__ int c_max_candidates;
        // The maximum number of features that memory is reserved for.
        __constant__ int c_max_features;
        // The image size.
        __constant__ int c_img_rows;
        __constant__ int c_img_cols;
        // The number of layers.
        __constant__ int c_nOctaveLayers;
        // The hessian threshold.
        __constant__ float c_hessianThreshold;

        // The current octave.
        __constant__ int c_octave;
        // The current layer size.
        __constant__ int c_layer_rows;
        __constant__ int c_layer_cols;

        void loadGlobalConstants(int maxCandidates, int maxFeatures, int img_rows, int img_cols, int nOctaveLayers, float hessianThreshold)
        {
            cudaSafeCall( cudaMemcpyToSymbol(c_max_candidates, &maxCandidates, sizeof(maxCandidates)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_max_features, &maxFeatures, sizeof(maxFeatures)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_img_rows, &img_rows, sizeof(img_rows)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_img_cols, &img_cols, sizeof(img_cols)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_nOctaveLayers, &nOctaveLayers, sizeof(nOctaveLayers)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_hessianThreshold, &hessianThreshold, sizeof(hessianThreshold)) );
        }

        void loadOctaveConstants(int octave, int layer_rows, int layer_cols)
        {
            cudaSafeCall( cudaMemcpyToSymbol(c_octave, &octave, sizeof(octave)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_layer_rows, &layer_rows, sizeof(layer_rows)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_layer_cols, &layer_cols, sizeof(layer_cols)) );
        }

        ////////////////////////////////////////////////////////////////////////
        // Integral image texture

        texture<unsigned char, 2, cudaReadModeElementType> imgTex(0, cudaFilterModePoint, cudaAddressModeClamp);
        texture<unsigned int, 2, cudaReadModeElementType> sumTex(0, cudaFilterModePoint, cudaAddressModeClamp);
        texture<unsigned int, 2, cudaReadModeElementType> maskSumTex(0, cudaFilterModePoint, cudaAddressModeClamp);

        void bindImgTex(PtrStepSzb img)
        {
            bindTexture(&imgTex, img);
        }

        size_t bindSumTex(PtrStepSz<uint> sum)
        {
            size_t offset;
            cudaChannelFormatDesc desc_sum = cudaCreateChannelDesc<uint>();
            cudaSafeCall( cudaBindTexture2D(&offset, sumTex, sum.data, desc_sum, sum.cols, sum.rows, sum.step));
            return offset / sizeof(uint);
        }
        size_t bindMaskSumTex(PtrStepSz<uint> maskSum)
        {
            size_t offset;
            cudaChannelFormatDesc desc_sum = cudaCreateChannelDesc<uint>();
            cudaSafeCall( cudaBindTexture2D(&offset, maskSumTex, maskSum.data, desc_sum, maskSum.cols, maskSum.rows, maskSum.step));
            return offset / sizeof(uint);
        }

        template <int N> __device__ float icvCalcHaarPatternSum(const float src[][5], int oldSize, int newSize, int y, int x)
        {
        #if __CUDA_ARCH__ && __CUDA_ARCH__ >= 200
            typedef double real_t;
        #else
            typedef float  real_t;
        #endif

            float ratio = (float)newSize / oldSize;

            real_t d = 0;

            #pragma unroll
            for (int k = 0; k < N; ++k)
            {
                int dx1 = __float2int_rn(ratio * src[k][0]);
                int dy1 = __float2int_rn(ratio * src[k][1]);
                int dx2 = __float2int_rn(ratio * src[k][2]);
                int dy2 = __float2int_rn(ratio * src[k][3]);

                real_t t = 0;
                t += tex2D(sumTex, x + dx1, y + dy1);
                t -= tex2D(sumTex, x + dx1, y + dy2);
                t -= tex2D(sumTex, x + dx2, y + dy1);
                t += tex2D(sumTex, x + dx2, y + dy2);

                d += t * src[k][4] / ((dx2 - dx1) * (dy2 - dy1));
            }

            return (float)d;
        }

        ////////////////////////////////////////////////////////////////////////
        // Hessian

        __constant__ float c_DX [3][5] = { {0, 2, 3, 7, 1}, {3, 2, 6, 7, -2}, {6, 2, 9, 7, 1} };
        __constant__ float c_DY [3][5] = { {2, 0, 7, 3, 1}, {2, 3, 7, 6, -2}, {2, 6, 7, 9, 1} };
        __constant__ float c_DXY[4][5] = { {1, 1, 4, 4, 1}, {5, 1, 8, 4, -1}, {1, 5, 4, 8, -1}, {5, 5, 8, 8, 1} };

        __host__ __device__ __forceinline__ int calcSize(int octave, int layer)
        {
            /* Wavelet size at first layer of first octave. */
            const int HAAR_SIZE0 = 9;

            /* Wavelet size increment between layers. This should be an even number,
             such that the wavelet sizes in an octave are either all even or all odd.
             This ensures that when looking for the neighbours of a sample, the layers
             above and below are aligned correctly. */
            const int HAAR_SIZE_INC = 6;

            return (HAAR_SIZE0 + HAAR_SIZE_INC * layer) << octave;
        }

        __global__ void icvCalcLayerDetAndTrace(PtrStepf det, PtrStepf trace)
        {
            // Determine the indices
            const int gridDim_y = gridDim.y / (c_nOctaveLayers + 2);
            const int blockIdx_y = blockIdx.y % gridDim_y;
            const int blockIdx_z = blockIdx.y / gridDim_y;

            const int j = threadIdx.x + blockIdx.x * blockDim.x;
            const int i = threadIdx.y + blockIdx_y * blockDim.y;
            const int layer = blockIdx_z;

            const int size = calcSize(c_octave, layer);

            const int samples_i = 1 + ((c_img_rows - size) >> c_octave);
            const int samples_j = 1 + ((c_img_cols - size) >> c_octave);

            // Ignore pixels where some of the kernel is outside the image
            const int margin = (size >> 1) >> c_octave;

            if (size <= c_img_rows && size <= c_img_cols && i < samples_i && j < samples_j)
            {
                const float dx  = icvCalcHaarPatternSum<3>(c_DX , 9, size, (i << c_octave), (j << c_octave));
                const float dy  = icvCalcHaarPatternSum<3>(c_DY , 9, size, (i << c_octave), (j << c_octave));
                const float dxy = icvCalcHaarPatternSum<4>(c_DXY, 9, size, (i << c_octave), (j << c_octave));

                det.ptr(layer * c_layer_rows + i + margin)[j + margin] = dx * dy - 0.81f * dxy * dxy;
                trace.ptr(layer * c_layer_rows + i + margin)[j + margin] = dx + dy;
            }
        }

        void icvCalcLayerDetAndTrace_gpu(const PtrStepf& det, const PtrStepf& trace, int img_rows, int img_cols,
            int octave, int nOctaveLayers)
        {
            const int min_size = calcSize(octave, 0);
            const int max_samples_i = 1 + ((img_rows - min_size) >> octave);
            const int max_samples_j = 1 + ((img_cols - min_size) >> octave);

            dim3 threads(16, 16);

            dim3 grid;
            grid.x = divUp(max_samples_j, threads.x);
            grid.y = divUp(max_samples_i, threads.y) * (nOctaveLayers + 2);

            icvCalcLayerDetAndTrace<<<grid, threads>>>(det, trace);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        ////////////////////////////////////////////////////////////////////////
        // NONMAX

        __constant__ float c_DM[5] = {0, 0, 9, 9, 1};

        struct WithMask
        {
            static __device__ bool check(int sum_i, int sum_j, int size)
            {
                float ratio = (float)size / 9.0f;

                float d = 0;

                int dx1 = __float2int_rn(ratio * c_DM[0]);
                int dy1 = __float2int_rn(ratio * c_DM[1]);
                int dx2 = __float2int_rn(ratio * c_DM[2]);
                int dy2 = __float2int_rn(ratio * c_DM[3]);

                float t = 0;
                t += tex2D(maskSumTex, sum_j + dx1, sum_i + dy1);
                t -= tex2D(maskSumTex, sum_j + dx1, sum_i + dy2);
                t -= tex2D(maskSumTex, sum_j + dx2, sum_i + dy1);
                t += tex2D(maskSumTex, sum_j + dx2, sum_i + dy2);

                d += t * c_DM[4] / ((dx2 - dx1) * (dy2 - dy1));

                return (d >= 0.5f);
            }
        };

        template <typename Mask>
        __global__ void icvFindMaximaInLayer(const PtrStepf det, const PtrStepf trace, int4* maxPosBuffer,
            unsigned int* maxCounter)
        {
            #if __CUDA_ARCH__ && __CUDA_ARCH__ >= 110

            extern __shared__ float N9[];

            // The hidx variables are the indices to the hessian buffer.
            const int gridDim_y = gridDim.y / c_nOctaveLayers;
            const int blockIdx_y = blockIdx.y % gridDim_y;
            const int blockIdx_z = blockIdx.y / gridDim_y;

            const int layer = blockIdx_z + 1;

            const int size = calcSize(c_octave, layer);

            // Ignore pixels without a 3x3x3 neighbourhood in the layer above
            const int margin = ((calcSize(c_octave, layer + 1) >> 1) >> c_octave) + 1;

            const int j = threadIdx.x + blockIdx.x * (blockDim.x - 2) + margin - 1;
            const int i = threadIdx.y + blockIdx_y * (blockDim.y - 2) + margin - 1;

            // Is this thread within the hessian buffer?
            const int zoff = blockDim.x * blockDim.y;
            const int localLin = threadIdx.x + threadIdx.y * blockDim.x + zoff;
            N9[localLin - zoff] = det.ptr(c_layer_rows * (layer - 1) + ::min(::max(i, 0), c_img_rows - 1))[::min(::max(j, 0), c_img_cols - 1)];
            N9[localLin       ] = det.ptr(c_layer_rows * (layer    ) + ::min(::max(i, 0), c_img_rows - 1))[::min(::max(j, 0), c_img_cols - 1)];
            N9[localLin + zoff] = det.ptr(c_layer_rows * (layer + 1) + ::min(::max(i, 0), c_img_rows - 1))[::min(::max(j, 0), c_img_cols - 1)];
            __syncthreads();

            if (i < c_layer_rows - margin && j < c_layer_cols - margin && threadIdx.x > 0 && threadIdx.x < blockDim.x - 1 && threadIdx.y > 0 && threadIdx.y < blockDim.y - 1)
            {
                float val0 = N9[localLin];

                if (val0 > c_hessianThreshold)
                {
                    // Coordinates for the start of the wavelet in the sum image. There
                    // is some integer division involved, so don't try to simplify this
                    // (cancel out sampleStep) without checking the result is the same
                    const int sum_i = (i - ((size >> 1) >> c_octave)) << c_octave;
                    const int sum_j = (j - ((size >> 1) >> c_octave)) << c_octave;

                    if (Mask::check(sum_i, sum_j, size))
                    {
                        // Check to see if we have a max (in its 26 neighbours)
                        const bool condmax = val0 > N9[localLin - 1 - blockDim.x - zoff]
                        &&                   val0 > N9[localLin     - blockDim.x - zoff]
                        &&                   val0 > N9[localLin + 1 - blockDim.x - zoff]
                        &&                   val0 > N9[localLin - 1              - zoff]
                        &&                   val0 > N9[localLin                  - zoff]
                        &&                   val0 > N9[localLin + 1              - zoff]
                        &&                   val0 > N9[localLin - 1 + blockDim.x - zoff]
                        &&                   val0 > N9[localLin     + blockDim.x - zoff]
                        &&                   val0 > N9[localLin + 1 + blockDim.x - zoff]

                        &&                   val0 > N9[localLin - 1 - blockDim.x]
                        &&                   val0 > N9[localLin     - blockDim.x]
                        &&                   val0 > N9[localLin + 1 - blockDim.x]
                        &&                   val0 > N9[localLin - 1             ]
                        &&                   val0 > N9[localLin + 1             ]
                        &&                   val0 > N9[localLin - 1 + blockDim.x]
                        &&                   val0 > N9[localLin     + blockDim.x]
                        &&                   val0 > N9[localLin + 1 + blockDim.x]

                        &&                   val0 > N9[localLin - 1 - blockDim.x + zoff]
                        &&                   val0 > N9[localLin     - blockDim.x + zoff]
                        &&                   val0 > N9[localLin + 1 - blockDim.x + zoff]
                        &&                   val0 > N9[localLin - 1              + zoff]
                        &&                   val0 > N9[localLin                  + zoff]
                        &&                   val0 > N9[localLin + 1              + zoff]
                        &&                   val0 > N9[localLin - 1 + blockDim.x + zoff]
                        &&                   val0 > N9[localLin     + blockDim.x + zoff]
                        &&                   val0 > N9[localLin + 1 + blockDim.x + zoff]
                        ;

                        if(condmax)
                        {
                            unsigned int ind = atomicInc(maxCounter,(unsigned int) -1);

                            if (ind < c_max_candidates)
                            {
                                const int laplacian = (int) copysignf(1.0f, trace.ptr(layer * c_layer_rows + i)[j]);

                                maxPosBuffer[ind] = make_int4(j, i, layer, laplacian);
                            }
                        }
                    }
                }
            }

            #endif
        }

        void icvFindMaximaInLayer_gpu(const PtrStepf& det, const PtrStepf& trace, int4* maxPosBuffer, unsigned int* maxCounter,
            int img_rows, int img_cols, int octave, bool use_mask, int nOctaveLayers)
        {
            const int layer_rows = img_rows >> octave;
            const int layer_cols = img_cols >> octave;

            const int min_margin = ((calcSize(octave, 2) >> 1) >> octave) + 1;

            dim3 threads(16, 16);

            dim3 grid;
            grid.x = divUp(layer_cols - 2 * min_margin, threads.x - 2);
            grid.y = divUp(layer_rows - 2 * min_margin, threads.y - 2) * nOctaveLayers;

            const size_t smem_size = threads.x * threads.y * 3 * sizeof(float);

            if (use_mask)
                icvFindMaximaInLayer<WithMask><<<grid, threads, smem_size>>>(det, trace, maxPosBuffer, maxCounter);
            else
                icvFindMaximaInLayer<WithOutMask><<<grid, threads, smem_size>>>(det, trace, maxPosBuffer, maxCounter);

            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        ////////////////////////////////////////////////////////////////////////
        // INTERPOLATION

        __global__ void icvInterpolateKeypoint(const PtrStepf det, const int4* maxPosBuffer,
            float* featureX, float* featureY, int* featureLaplacian, int* featureOctave, float* featureSize, float* featureHessian,
            unsigned int* featureCounter)
        {
            #if __CUDA_ARCH__ && __CUDA_ARCH__ >= 110

            const int4 maxPos = maxPosBuffer[blockIdx.x];

            const int j = maxPos.x - 1 + threadIdx.x;
            const int i = maxPos.y - 1 + threadIdx.y;
            const int layer = maxPos.z - 1 + threadIdx.z;

            __shared__ float N9[3][3][3];

            N9[threadIdx.z][threadIdx.y][threadIdx.x] = det.ptr(c_layer_rows * layer + i)[j];
            __syncthreads();

            if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            {
                __shared__ float dD[3];

                //dx
                dD[0] = -0.5f * (N9[1][1][2] - N9[1][1][0]);
                //dy
                dD[1] = -0.5f * (N9[1][2][1] - N9[1][0][1]);
                //ds
                dD[2] = -0.5f * (N9[2][1][1] - N9[0][1][1]);

                __shared__ float H[3][3];

                //dxx
                H[0][0] = N9[1][1][0] - 2.0f * N9[1][1][1] + N9[1][1][2];
                //dxy
                H[0][1]= 0.25f * (N9[1][2][2] - N9[1][2][0] - N9[1][0][2] + N9[1][0][0]);
                //dxs
                H[0][2]= 0.25f * (N9[2][1][2] - N9[2][1][0] - N9[0][1][2] + N9[0][1][0]);
                //dyx = dxy
                H[1][0] = H[0][1];
                //dyy
                H[1][1] = N9[1][0][1] - 2.0f * N9[1][1][1] + N9[1][2][1];
                //dys
                H[1][2]= 0.25f * (N9[2][2][1] - N9[2][0][1] - N9[0][2][1] + N9[0][0][1]);
                //dsx = dxs
                H[2][0] = H[0][2];
                //dsy = dys
                H[2][1] = H[1][2];
                //dss
                H[2][2] = N9[0][1][1] - 2.0f * N9[1][1][1] + N9[2][1][1];

                __shared__ float x[3];

                if (solve3x3(H, dD, x))
                {
                    if (::fabs(x[0]) <= 1.f && ::fabs(x[1]) <= 1.f && ::fabs(x[2]) <= 1.f)
                    {
                        // if the step is within the interpolation region, perform it

                        const int size = calcSize(c_octave, maxPos.z);

                        const int sum_i = (maxPos.y - ((size >> 1) >> c_octave)) << c_octave;
                        const int sum_j = (maxPos.x - ((size >> 1) >> c_octave)) << c_octave;

                        const float center_i = sum_i + (float)(size - 1) / 2;
                        const float center_j = sum_j + (float)(size - 1) / 2;

                        const float px = center_j + x[0] * (1 << c_octave);
                        const float py = center_i + x[1] * (1 << c_octave);

                        const int ds = size - calcSize(c_octave, maxPos.z - 1);
                        const float psize = roundf(size + x[2] * ds);

                        /* The sampling intervals and wavelet sized for selecting an orientation
                         and building the keypoint descriptor are defined relative to 's' */
                        const float s = psize * 1.2f / 9.0f;

                        /* To find the dominant orientation, the gradients in x and y are
                         sampled in a circle of radius 6s using wavelets of size 4s.
                         We ensure the gradient wavelet size is even to ensure the
                         wavelet pattern is balanced and symmetric around its center */
                        const int grad_wav_size = 2 * __float2int_rn(2.0f * s);

                        // check when grad_wav_size is too big
                        if ((c_img_rows + 1) >= grad_wav_size && (c_img_cols + 1) >= grad_wav_size)
                        {
                            // Get a new feature index.
                            unsigned int ind = atomicInc(featureCounter, (unsigned int)-1);

                            if (ind < c_max_features)
                            {
                                featureX[ind] = px;
                                featureY[ind] = py;
                                featureLaplacian[ind] = maxPos.w;
                                featureOctave[ind] = c_octave;
                                featureSize[ind] = psize;
                                featureHessian[ind] = N9[1][1][1];
                            }
                        } // grad_wav_size check
                    } // If the subpixel interpolation worked
                }
            } // If this is thread 0.

            #endif
        }

        void icvInterpolateKeypoint_gpu(const PtrStepf& det, const int4* maxPosBuffer, unsigned int maxCounter,
            float* featureX, float* featureY, int* featureLaplacian, int* featureOctave, float* featureSize, float* featureHessian,
            unsigned int* featureCounter)
        {
            dim3 threads;
            threads.x = 3;
            threads.y = 3;
            threads.z = 3;

            dim3 grid;
            grid.x = maxCounter;

            icvInterpolateKeypoint<<<grid, threads>>>(det, maxPosBuffer, featureX, featureY, featureLaplacian, featureOctave, featureSize, featureHessian, featureCounter);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        ////////////////////////////////////////////////////////////////////////
        // Orientation

        #define ORI_SEARCH_INC 5
        #define ORI_WIN        60
        #define ORI_SAMPLES    113

        __constant__ float c_aptX[ORI_SAMPLES] = {-6, -5, -5, -5, -5, -5, -5, -5, -4, -4, -4, -4, -4, -4, -4, -4, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6};
        __constant__ float c_aptY[ORI_SAMPLES] = {0, -3, -2, -1, 0, 1, 2, 3, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -4, -3, -2, -1, 0, 1, 2, 3, 4, -3, -2, -1, 0, 1, 2, 3, 0};
        __constant__ float c_aptW[ORI_SAMPLES] = {0.001455130288377404f, 0.001707611023448408f, 0.002547456417232752f, 0.003238451667129993f, 0.0035081731621176f, 0.003238451667129993f, 0.002547456417232752f, 0.001707611023448408f, 0.002003900473937392f, 0.0035081731621176f, 0.005233579315245152f, 0.00665318313986063f, 0.00720730796456337f, 0.00665318313986063f, 0.005233579315245152f, 0.0035081731621176f, 0.002003900473937392f, 0.001707611023448408f, 0.0035081731621176f, 0.006141661666333675f, 0.009162282571196556f, 0.01164754293859005f, 0.01261763460934162f, 0.01164754293859005f, 0.009162282571196556f, 0.006141661666333675f, 0.0035081731621176f, 0.001707611023448408f, 0.002547456417232752f, 0.005233579315245152f, 0.009162282571196556f, 0.01366852037608624f, 0.01737609319388866f, 0.0188232995569706f, 0.01737609319388866f, 0.01366852037608624f, 0.009162282571196556f, 0.005233579315245152f, 0.002547456417232752f, 0.003238451667129993f, 0.00665318313986063f, 0.01164754293859005f, 0.01737609319388866f, 0.02208934165537357f, 0.02392910048365593f, 0.02208934165537357f, 0.01737609319388866f, 0.01164754293859005f, 0.00665318313986063f, 0.003238451667129993f, 0.001455130288377404f, 0.0035081731621176f, 0.00720730796456337f, 0.01261763460934162f, 0.0188232995569706f, 0.02392910048365593f, 0.02592208795249462f, 0.02392910048365593f, 0.0188232995569706f, 0.01261763460934162f, 0.00720730796456337f, 0.0035081731621176f, 0.001455130288377404f, 0.003238451667129993f, 0.00665318313986063f, 0.01164754293859005f, 0.01737609319388866f, 0.02208934165537357f, 0.02392910048365593f, 0.02208934165537357f, 0.01737609319388866f, 0.01164754293859005f, 0.00665318313986063f, 0.003238451667129993f, 0.002547456417232752f, 0.005233579315245152f, 0.009162282571196556f, 0.01366852037608624f, 0.01737609319388866f, 0.0188232995569706f, 0.01737609319388866f, 0.01366852037608624f, 0.009162282571196556f, 0.005233579315245152f, 0.002547456417232752f, 0.001707611023448408f, 0.0035081731621176f, 0.006141661666333675f, 0.009162282571196556f, 0.01164754293859005f, 0.01261763460934162f, 0.01164754293859005f, 0.009162282571196556f, 0.006141661666333675f, 0.0035081731621176f, 0.001707611023448408f, 0.002003900473937392f, 0.0035081731621176f, 0.005233579315245152f, 0.00665318313986063f, 0.00720730796456337f, 0.00665318313986063f, 0.005233579315245152f, 0.0035081731621176f, 0.002003900473937392f, 0.001707611023448408f, 0.002547456417232752f, 0.003238451667129993f, 0.0035081731621176f, 0.003238451667129993f, 0.002547456417232752f, 0.001707611023448408f, 0.001455130288377404f};

        __constant__ float c_NX[2][5] = {{0, 0, 2, 4, -1}, {2, 0, 4, 4, 1}};
        __constant__ float c_NY[2][5] = {{0, 0, 4, 2, 1}, {0, 2, 4, 4, -1}};

        __global__ void icvCalcOrientation(const float* featureX, const float* featureY, const float* featureSize, float* featureDir)
        {
            __shared__ float s_X[128];
            __shared__ float s_Y[128];
            __shared__ float s_angle[128];

            __shared__ float s_sumx[32 * 4];
            __shared__ float s_sumy[32 * 4];

            /* The sampling intervals and wavelet sized for selecting an orientation
             and building the keypoint descriptor are defined relative to 's' */
            const float s = featureSize[blockIdx.x] * 1.2f / 9.0f;

            /* To find the dominant orientation, the gradients in x and y are
             sampled in a circle of radius 6s using wavelets of size 4s.
             We ensure the gradient wavelet size is even to ensure the
             wavelet pattern is balanced and symmetric around its center */
            const int grad_wav_size = 2 * __float2int_rn(2.0f * s);

            // check when grad_wav_size is too big
            if ((c_img_rows + 1) < grad_wav_size || (c_img_cols + 1) < grad_wav_size)
                return;

            // Calc X, Y, angle and store it to shared memory
            const int tid = threadIdx.y * blockDim.x + threadIdx.x;

            float X = 0.0f, Y = 0.0f, angle = 0.0f;

            if (tid < ORI_SAMPLES)
            {
                const float margin = (float)(grad_wav_size - 1) / 2.0f;
                const int x = __float2int_rn(featureX[blockIdx.x] + c_aptX[tid] * s - margin);
                const int y = __float2int_rn(featureY[blockIdx.x] + c_aptY[tid] * s - margin);

                if (y >= 0 && y < (c_img_rows + 1) - grad_wav_size &&
                    x >= 0 && x < (c_img_cols + 1) - grad_wav_size)
                {
                    X = c_aptW[tid] * icvCalcHaarPatternSum<2>(c_NX, 4, grad_wav_size, y, x);
                    Y = c_aptW[tid] * icvCalcHaarPatternSum<2>(c_NY, 4, grad_wav_size, y, x);

                    angle = atan2f(Y, X);
                    if (angle < 0)
                        angle += 2.0f * CV_PI_F;
                    angle *= 180.0f / CV_PI_F;
                }
            }
            s_X[tid] = X;
            s_Y[tid] = Y;
            s_angle[tid] = angle;
            __syncthreads();

            float bestx = 0, besty = 0, best_mod = 0;

        #if __CUDA_ARCH__ >= 200
            #pragma unroll
        #endif
            for (int i = 0; i < 18; ++i)
            {
                const int dir = (i * 4 + threadIdx.y) * ORI_SEARCH_INC;

                float sumx = 0.0f, sumy = 0.0f;
                int d = ::abs(__float2int_rn(s_angle[threadIdx.x]) - dir);
                if (d < ORI_WIN / 2 || d > 360 - ORI_WIN / 2)
                {
                    sumx = s_X[threadIdx.x];
                    sumy = s_Y[threadIdx.x];
                }
                d = ::abs(__float2int_rn(s_angle[threadIdx.x + 32]) - dir);
                if (d < ORI_WIN / 2 || d > 360 - ORI_WIN / 2)
                {
                    sumx += s_X[threadIdx.x + 32];
                    sumy += s_Y[threadIdx.x + 32];
                }
                d = ::abs(__float2int_rn(s_angle[threadIdx.x + 64]) - dir);
                if (d < ORI_WIN / 2 || d > 360 - ORI_WIN / 2)
                {
                    sumx += s_X[threadIdx.x + 64];
                    sumy += s_Y[threadIdx.x + 64];
                }
                d = ::abs(__float2int_rn(s_angle[threadIdx.x + 96]) - dir);
                if (d < ORI_WIN / 2 || d > 360 - ORI_WIN / 2)
                {
                    sumx += s_X[threadIdx.x + 96];
                    sumy += s_Y[threadIdx.x + 96];
                }

                plus<float> op;
                device::reduce<32>(smem_tuple(s_sumx + threadIdx.y * 32, s_sumy + threadIdx.y * 32),
                                   thrust::tie(sumx, sumy), threadIdx.x, thrust::make_tuple(op, op));

                const float temp_mod = sumx * sumx + sumy * sumy;
                if (temp_mod > best_mod)
                {
                    best_mod = temp_mod;
                    bestx = sumx;
                    besty = sumy;
                }

                __syncthreads();
            }

            if (threadIdx.x == 0)
            {
                s_X[threadIdx.y] = bestx;
                s_Y[threadIdx.y] = besty;
                s_angle[threadIdx.y] = best_mod;
            }
            __syncthreads();

            if (threadIdx.x == 0 && threadIdx.y == 0)
            {
                int bestIdx = 0;

                if (s_angle[1] > s_angle[bestIdx])
                    bestIdx = 1;
                if (s_angle[2] > s_angle[bestIdx])
                    bestIdx = 2;
                if (s_angle[3] > s_angle[bestIdx])
                    bestIdx = 3;

                float kp_dir = atan2f(s_Y[bestIdx], s_X[bestIdx]);
                if (kp_dir < 0)
                    kp_dir += 2.0f * CV_PI_F;
                kp_dir *= 180.0f / CV_PI_F;

                kp_dir = 360.0f - kp_dir;
                if (::fabsf(kp_dir - 360.f) < numeric_limits<float>::epsilon())
                    kp_dir = 0.f;

                featureDir[blockIdx.x] = kp_dir;
            }
        }

        #undef ORI_SEARCH_INC
        #undef ORI_WIN
        #undef ORI_SAMPLES

        void icvCalcOrientation_gpu(const float* featureX, const float* featureY, const float* featureSize, float* featureDir, int nFeatures)
        {
            dim3 threads;
            threads.x = 32;
            threads.y = 4;

            dim3 grid;
            grid.x = nFeatures;

            icvCalcOrientation<<<grid, threads>>>(featureX, featureY, featureSize, featureDir);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        ////////////////////////////////////////////////////////////////////////
        // Descriptors

        #define PATCH_SZ 20

        __constant__ float c_DW[PATCH_SZ * PATCH_SZ] =
        {
            3.695352233989979e-006f, 8.444558261544444e-006f, 1.760426494001877e-005f, 3.34794785885606e-005f, 5.808438800158911e-005f, 9.193058212986216e-005f, 0.0001327334757661447f, 0.0001748319627949968f, 0.0002100782439811155f, 0.0002302826324012131f, 0.0002302826324012131f, 0.0002100782439811155f, 0.0001748319627949968f, 0.0001327334757661447f, 9.193058212986216e-005f, 5.808438800158911e-005f, 3.34794785885606e-005f, 1.760426494001877e-005f, 8.444558261544444e-006f, 3.695352233989979e-006f,
            8.444558261544444e-006f, 1.929736572492402e-005f, 4.022897701361217e-005f, 7.650675252079964e-005f, 0.0001327334903180599f, 0.0002100782585330308f, 0.0003033203829545528f, 0.0003995231236331165f, 0.0004800673632416874f, 0.0005262381164357066f, 0.0005262381164357066f, 0.0004800673632416874f, 0.0003995231236331165f, 0.0003033203829545528f, 0.0002100782585330308f, 0.0001327334903180599f, 7.650675252079964e-005f, 4.022897701361217e-005f, 1.929736572492402e-005f, 8.444558261544444e-006f,
            1.760426494001877e-005f, 4.022897701361217e-005f, 8.386484114453197e-005f, 0.0001594926579855382f, 0.0002767078403849155f, 0.0004379475140012801f, 0.0006323281559161842f, 0.0008328808471560478f, 0.001000790391117334f, 0.001097041997127235f, 0.001097041997127235f, 0.001000790391117334f, 0.0008328808471560478f, 0.0006323281559161842f, 0.0004379475140012801f, 0.0002767078403849155f, 0.0001594926579855382f, 8.386484114453197e-005f, 4.022897701361217e-005f, 1.760426494001877e-005f,
            3.34794785885606e-005f, 7.650675252079964e-005f, 0.0001594926579855382f, 0.0003033203247468919f, 0.0005262380582280457f, 0.0008328807889483869f, 0.001202550483867526f, 0.001583957928232849f, 0.001903285388834775f, 0.002086334861814976f, 0.002086334861814976f, 0.001903285388834775f, 0.001583957928232849f, 0.001202550483867526f, 0.0008328807889483869f, 0.0005262380582280457f, 0.0003033203247468919f, 0.0001594926579855382f, 7.650675252079964e-005f, 3.34794785885606e-005f,
            5.808438800158911e-005f, 0.0001327334903180599f, 0.0002767078403849155f, 0.0005262380582280457f, 0.0009129836107604206f, 0.001444985857233405f, 0.002086335094645619f, 0.002748048631474376f, 0.00330205773934722f, 0.003619635012000799f, 0.003619635012000799f, 0.00330205773934722f, 0.002748048631474376f, 0.002086335094645619f, 0.001444985857233405f, 0.0009129836107604206f, 0.0005262380582280457f, 0.0002767078403849155f, 0.0001327334903180599f, 5.808438800158911e-005f,
            9.193058212986216e-005f, 0.0002100782585330308f, 0.0004379475140012801f, 0.0008328807889483869f, 0.001444985857233405f, 0.002286989474669099f, 0.00330205773934722f, 0.004349356517195702f, 0.00522619066759944f, 0.005728822201490402f, 0.005728822201490402f, 0.00522619066759944f, 0.004349356517195702f, 0.00330205773934722f, 0.002286989474669099f, 0.001444985857233405f, 0.0008328807889483869f, 0.0004379475140012801f, 0.0002100782585330308f, 9.193058212986216e-005f,
            0.0001327334757661447f, 0.0003033203829545528f, 0.0006323281559161842f, 0.001202550483867526f, 0.002086335094645619f, 0.00330205773934722f, 0.004767658654600382f, 0.006279794964939356f, 0.007545807864516974f, 0.008271530270576477f, 0.008271530270576477f, 0.007545807864516974f, 0.006279794964939356f, 0.004767658654600382f, 0.00330205773934722f, 0.002086335094645619f, 0.001202550483867526f, 0.0006323281559161842f, 0.0003033203829545528f, 0.0001327334757661447f,
            0.0001748319627949968f, 0.0003995231236331165f, 0.0008328808471560478f, 0.001583957928232849f, 0.002748048631474376f, 0.004349356517195702f, 0.006279794964939356f, 0.008271529339253902f, 0.009939077310264111f, 0.01089497376233339f, 0.01089497376233339f, 0.009939077310264111f, 0.008271529339253902f, 0.006279794964939356f, 0.004349356517195702f, 0.002748048631474376f, 0.001583957928232849f, 0.0008328808471560478f, 0.0003995231236331165f, 0.0001748319627949968f,
            0.0002100782439811155f, 0.0004800673632416874f, 0.001000790391117334f, 0.001903285388834775f, 0.00330205773934722f, 0.00522619066759944f, 0.007545807864516974f, 0.009939077310264111f, 0.01194280479103327f, 0.01309141051024199f, 0.01309141051024199f, 0.01194280479103327f, 0.009939077310264111f, 0.007545807864516974f, 0.00522619066759944f, 0.00330205773934722f, 0.001903285388834775f, 0.001000790391117334f, 0.0004800673632416874f, 0.0002100782439811155f,
            0.0002302826324012131f, 0.0005262381164357066f, 0.001097041997127235f, 0.002086334861814976f, 0.003619635012000799f, 0.005728822201490402f, 0.008271530270576477f, 0.01089497376233339f, 0.01309141051024199f, 0.01435048412531614f, 0.01435048412531614f, 0.01309141051024199f, 0.01089497376233339f, 0.008271530270576477f, 0.005728822201490402f, 0.003619635012000799f, 0.002086334861814976f, 0.001097041997127235f, 0.0005262381164357066f, 0.0002302826324012131f,
            0.0002302826324012131f, 0.0005262381164357066f, 0.001097041997127235f, 0.002086334861814976f, 0.003619635012000799f, 0.005728822201490402f, 0.008271530270576477f, 0.01089497376233339f, 0.01309141051024199f, 0.01435048412531614f, 0.01435048412531614f, 0.01309141051024199f, 0.01089497376233339f, 0.008271530270576477f, 0.005728822201490402f, 0.003619635012000799f, 0.002086334861814976f, 0.001097041997127235f, 0.0005262381164357066f, 0.0002302826324012131f,
            0.0002100782439811155f, 0.0004800673632416874f, 0.001000790391117334f, 0.001903285388834775f, 0.00330205773934722f, 0.00522619066759944f, 0.007545807864516974f, 0.009939077310264111f, 0.01194280479103327f, 0.01309141051024199f, 0.01309141051024199f, 0.01194280479103327f, 0.009939077310264111f, 0.007545807864516974f, 0.00522619066759944f, 0.00330205773934722f, 0.001903285388834775f, 0.001000790391117334f, 0.0004800673632416874f, 0.0002100782439811155f,
            0.0001748319627949968f, 0.0003995231236331165f, 0.0008328808471560478f, 0.001583957928232849f, 0.002748048631474376f, 0.004349356517195702f, 0.006279794964939356f, 0.008271529339253902f, 0.009939077310264111f, 0.01089497376233339f, 0.01089497376233339f, 0.009939077310264111f, 0.008271529339253902f, 0.006279794964939356f, 0.004349356517195702f, 0.002748048631474376f, 0.001583957928232849f, 0.0008328808471560478f, 0.0003995231236331165f, 0.0001748319627949968f,
            0.0001327334757661447f, 0.0003033203829545528f, 0.0006323281559161842f, 0.001202550483867526f, 0.002086335094645619f, 0.00330205773934722f, 0.004767658654600382f, 0.006279794964939356f, 0.007545807864516974f, 0.008271530270576477f, 0.008271530270576477f, 0.007545807864516974f, 0.006279794964939356f, 0.004767658654600382f, 0.00330205773934722f, 0.002086335094645619f, 0.001202550483867526f, 0.0006323281559161842f, 0.0003033203829545528f, 0.0001327334757661447f,
            9.193058212986216e-005f, 0.0002100782585330308f, 0.0004379475140012801f, 0.0008328807889483869f, 0.001444985857233405f, 0.002286989474669099f, 0.00330205773934722f, 0.004349356517195702f, 0.00522619066759944f, 0.005728822201490402f, 0.005728822201490402f, 0.00522619066759944f, 0.004349356517195702f, 0.00330205773934722f, 0.002286989474669099f, 0.001444985857233405f, 0.0008328807889483869f, 0.0004379475140012801f, 0.0002100782585330308f, 9.193058212986216e-005f,
            5.808438800158911e-005f, 0.0001327334903180599f, 0.0002767078403849155f, 0.0005262380582280457f, 0.0009129836107604206f, 0.001444985857233405f, 0.002086335094645619f, 0.002748048631474376f, 0.00330205773934722f, 0.003619635012000799f, 0.003619635012000799f, 0.00330205773934722f, 0.002748048631474376f, 0.002086335094645619f, 0.001444985857233405f, 0.0009129836107604206f, 0.0005262380582280457f, 0.0002767078403849155f, 0.0001327334903180599f, 5.808438800158911e-005f,
            3.34794785885606e-005f, 7.650675252079964e-005f, 0.0001594926579855382f, 0.0003033203247468919f, 0.0005262380582280457f, 0.0008328807889483869f, 0.001202550483867526f, 0.001583957928232849f, 0.001903285388834775f, 0.002086334861814976f, 0.002086334861814976f, 0.001903285388834775f, 0.001583957928232849f, 0.001202550483867526f, 0.0008328807889483869f, 0.0005262380582280457f, 0.0003033203247468919f, 0.0001594926579855382f, 7.650675252079964e-005f, 3.34794785885606e-005f,
            1.760426494001877e-005f, 4.022897701361217e-005f, 8.386484114453197e-005f, 0.0001594926579855382f, 0.0002767078403849155f, 0.0004379475140012801f, 0.0006323281559161842f, 0.0008328808471560478f, 0.001000790391117334f, 0.001097041997127235f, 0.001097041997127235f, 0.001000790391117334f, 0.0008328808471560478f, 0.0006323281559161842f, 0.0004379475140012801f, 0.0002767078403849155f, 0.0001594926579855382f, 8.386484114453197e-005f, 4.022897701361217e-005f, 1.760426494001877e-005f,
            8.444558261544444e-006f, 1.929736572492402e-005f, 4.022897701361217e-005f, 7.650675252079964e-005f, 0.0001327334903180599f, 0.0002100782585330308f, 0.0003033203829545528f, 0.0003995231236331165f, 0.0004800673632416874f, 0.0005262381164357066f, 0.0005262381164357066f, 0.0004800673632416874f, 0.0003995231236331165f, 0.0003033203829545528f, 0.0002100782585330308f, 0.0001327334903180599f, 7.650675252079964e-005f, 4.022897701361217e-005f, 1.929736572492402e-005f, 8.444558261544444e-006f,
            3.695352233989979e-006f, 8.444558261544444e-006f, 1.760426494001877e-005f, 3.34794785885606e-005f, 5.808438800158911e-005f, 9.193058212986216e-005f, 0.0001327334757661447f, 0.0001748319627949968f, 0.0002100782439811155f, 0.0002302826324012131f, 0.0002302826324012131f, 0.0002100782439811155f, 0.0001748319627949968f, 0.0001327334757661447f, 9.193058212986216e-005f, 5.808438800158911e-005f, 3.34794785885606e-005f, 1.760426494001877e-005f, 8.444558261544444e-006f, 3.695352233989979e-006f
        };

        struct WinReader
        {
            typedef uchar elem_type;

            __device__ __forceinline__ uchar operator ()(int i, int j) const
            {
                float pixel_x = centerX + (win_offset + j) * cos_dir + (win_offset + i) * sin_dir;
                float pixel_y = centerY - (win_offset + j) * sin_dir + (win_offset + i) * cos_dir;

                return tex2D(imgTex, pixel_x, pixel_y);
            }

            float centerX;
            float centerY;
            float win_offset;
            float cos_dir;
            float sin_dir;
            int width;
            int height;
        };

        __device__ void calc_dx_dy(const float* featureX, const float* featureY, const float* featureSize, const float* featureDir,
                                   float& dx, float& dy);

        __device__ void calc_dx_dy(const float* featureX, const float* featureY, const float* featureSize, const float* featureDir,
                                   float& dx, float& dy)
        {
            __shared__ float s_PATCH[PATCH_SZ + 1][PATCH_SZ + 1];

            dx = dy = 0.0f;

            WinReader win;

            win.centerX = featureX[blockIdx.x];
            win.centerY = featureY[blockIdx.x];

            // The sampling intervals and wavelet sized for selecting an orientation
            // and building the keypoint descriptor are defined relative to 's'
            const float s = featureSize[blockIdx.x] * 1.2f / 9.0f;

            // Extract a window of pixels around the keypoint of size 20s
            const int win_size = (int)((PATCH_SZ + 1) * s);

            win.width = win.height = win_size;

            // Nearest neighbour version (faster)
            win.win_offset = -(win_size - 1.0f) / 2.0f;

            float descriptor_dir = 360.0f - featureDir[blockIdx.x];
            if (::fabsf(descriptor_dir - 360.f) < numeric_limits<float>::epsilon())
                descriptor_dir = 0.f;
            descriptor_dir *= CV_PI_F / 180.0f;
            sincosf(descriptor_dir, &win.sin_dir, &win.cos_dir);

            const int tid = threadIdx.y * blockDim.x + threadIdx.x;

            const int xLoadInd = tid % (PATCH_SZ + 1);
            const int yLoadInd = tid / (PATCH_SZ + 1);

            if (yLoadInd < (PATCH_SZ + 1))
            {
                if (s > 1)
                {
                    AreaFilter<WinReader> filter(win, s, s);
                    s_PATCH[yLoadInd][xLoadInd] = filter(yLoadInd, xLoadInd);
                }
                else
                {
                    LinearFilter<WinReader> filter(win);
                    s_PATCH[yLoadInd][xLoadInd] = filter(yLoadInd * s, xLoadInd * s);
                }
            }

            __syncthreads();

            const int xPatchInd = threadIdx.x % 5;
            const int yPatchInd = threadIdx.x / 5;

            if (yPatchInd < 5)
            {
                const int xBlockInd = threadIdx.y % 4;
                const int yBlockInd = threadIdx.y / 4;

                const int xInd = xBlockInd * 5 + xPatchInd;
                const int yInd = yBlockInd * 5 + yPatchInd;

                const float dw = c_DW[yInd * PATCH_SZ + xInd];

                dx = (s_PATCH[yInd    ][xInd + 1] - s_PATCH[yInd][xInd] + s_PATCH[yInd + 1][xInd + 1] - s_PATCH[yInd + 1][xInd    ]) * dw;
                dy = (s_PATCH[yInd + 1][xInd    ] - s_PATCH[yInd][xInd] + s_PATCH[yInd + 1][xInd + 1] - s_PATCH[yInd    ][xInd + 1]) * dw;
            }
        }

        __global__ void compute_descriptors_64(PtrStep<float4> descriptors, const float* featureX, const float* featureY, const float* featureSize, const float* featureDir)
        {
            __shared__ float smem[32 * 16];

            float* sRow = smem + threadIdx.y * 32;

            float dx, dy;
            calc_dx_dy(featureX, featureY, featureSize, featureDir, dx, dy);

            float dxabs = ::fabsf(dx);
            float dyabs = ::fabsf(dy);

            plus<float> op;

            reduce<32>(sRow, dx, threadIdx.x, op);
            reduce<32>(sRow, dy, threadIdx.x, op);
            reduce<32>(sRow, dxabs, threadIdx.x, op);
            reduce<32>(sRow, dyabs, threadIdx.x, op);

            float4* descriptors_block = descriptors.ptr(blockIdx.x) + threadIdx.y;

            // write dx, dy, |dx|, |dy|
            if (threadIdx.x == 0)
                *descriptors_block = make_float4(dx, dy, dxabs, dyabs);
        }

        __global__ void compute_descriptors_128(PtrStep<float4> descriptors, const float* featureX, const float* featureY, const float* featureSize, const float* featureDir)
        {
            __shared__ float smem[32 * 16];

            float* sRow = smem + threadIdx.y * 32;

            float dx, dy;
            calc_dx_dy(featureX, featureY, featureSize, featureDir, dx, dy);

            float4* descriptors_block = descriptors.ptr(blockIdx.x) + threadIdx.y * 2;

            plus<float> op;

            float d1 = 0.0f;
            float d2 = 0.0f;
            float abs1 = 0.0f;
            float abs2 = 0.0f;

            if (dy >= 0)
            {
                d1 = dx;
                abs1 = ::fabsf(dx);
            }
            else
            {
                d2 = dx;
                abs2 = ::fabsf(dx);
            }

            reduce<32>(sRow, d1, threadIdx.x, op);
            reduce<32>(sRow, d2, threadIdx.x, op);
            reduce<32>(sRow, abs1, threadIdx.x, op);
            reduce<32>(sRow, abs2, threadIdx.x, op);

            // write dx (dy >= 0), |dx| (dy >= 0), dx (dy < 0), |dx| (dy < 0)
            if (threadIdx.x == 0)
                descriptors_block[0] = make_float4(d1, abs1, d2, abs2);

            if (dx >= 0)
            {
                d1 = dy;
                abs1 = ::fabsf(dy);
                d2 = 0.0f;
                abs2 = 0.0f;
            }
            else
            {
                d1 = 0.0f;
                abs1 = 0.0f;
                d2 = dy;
                abs2 = ::fabsf(dy);
            }

            reduce<32>(sRow, d1, threadIdx.x, op);
            reduce<32>(sRow, d2, threadIdx.x, op);
            reduce<32>(sRow, abs1, threadIdx.x, op);
            reduce<32>(sRow, abs2, threadIdx.x, op);

            // write dy (dx >= 0), |dy| (dx >= 0), dy (dx < 0), |dy| (dx < 0)
            if (threadIdx.x == 0)
                descriptors_block[1] = make_float4(d1, abs1, d2, abs2);
        }

        template <int BLOCK_DIM_X> __global__ void normalize_descriptors(PtrStepf descriptors)
        {
            __shared__ float smem[BLOCK_DIM_X];
            __shared__ float s_len;

            // no need for thread ID
            float* descriptor_base = descriptors.ptr(blockIdx.x);

            // read in the unnormalized descriptor values (squared)
            const float val = descriptor_base[threadIdx.x];

            float len = val * val;
            reduce<BLOCK_DIM_X>(smem, len, threadIdx.x, plus<float>());

            if (threadIdx.x == 0)
                s_len = ::sqrtf(len);

            __syncthreads();

            // normalize and store in output
            descriptor_base[threadIdx.x] = val / s_len;
        }

        void compute_descriptors_gpu(PtrStepSz<float4> descriptors, const float* featureX, const float* featureY, const float* featureSize, const float* featureDir, int nFeatures)
        {
            // compute unnormalized descriptors, then normalize them - odd indexing since grid must be 2D

            if (descriptors.cols == 64)
            {
                compute_descriptors_64<<<nFeatures, dim3(32, 16)>>>(descriptors, featureX, featureY, featureSize, featureDir);
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );

                normalize_descriptors<64><<<nFeatures, 64>>>((PtrStepSzf) descriptors);
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );
            }
            else
            {
                compute_descriptors_128<<<nFeatures, dim3(32, 16)>>>(descriptors, featureX, featureY, featureSize, featureDir);
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );

                normalize_descriptors<128><<<nFeatures, 128>>>((PtrStepSzf) descriptors);
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );
            }
        }
    } // namespace surf
}}} // namespace cv { namespace gpu { namespace device


#endif /* HAVE_OPENCV_GPU */
