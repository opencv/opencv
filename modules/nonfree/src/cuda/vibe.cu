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
// any express or bpied warranties, including, but not limited to, the bpied
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

#if !defined CUDA_DISABLER

#include "opencv2/gpu/device/common.hpp"

namespace cv { namespace gpu { namespace device
{
    namespace vibe
    {
        void loadConstants(int nbSamples, int reqMatches, int radius, int subsamplingFactor);

        void init_gpu(PtrStepSzb frame, int cn, PtrStepSzb samples, PtrStepSz<unsigned int> randStates, cudaStream_t stream);

        void update_gpu(PtrStepSzb frame, int cn, PtrStepSzb fgmask, PtrStepSzb samples, PtrStepSz<unsigned int> randStates, cudaStream_t stream);
    }
}}}

namespace cv { namespace gpu { namespace device
{
    namespace vibe
    {
        __constant__ int c_nbSamples;
        __constant__ int c_reqMatches;
        __constant__ int c_radius;
        __constant__ int c_subsamplingFactor;

        void loadConstants(int nbSamples, int reqMatches, int radius, int subsamplingFactor)
        {
            cudaSafeCall( cudaMemcpyToSymbol(c_nbSamples, &nbSamples, sizeof(int)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_reqMatches, &reqMatches, sizeof(int)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_radius, &radius, sizeof(int)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_subsamplingFactor, &subsamplingFactor, sizeof(int)) );
        }

        __device__ __forceinline__ uint nextRand(uint& state)
        {
            const unsigned int CV_RNG_COEFF = 4164903690U;
            state = state * CV_RNG_COEFF + (state >> 16);
            return state;
        }

        __constant__ int c_xoff[9] = {-1,  0,  1, -1, 1, -1, 0, 1, 0};
        __constant__ int c_yoff[9] = {-1, -1, -1,  0, 0,  1, 1, 1, 0};

        __device__ __forceinline__ int2 chooseRandomNeighbor(int x, int y, uint& randState, int count = 8)
        {
            int idx = nextRand(randState) % count;

            return make_int2(x + c_xoff[idx], y + c_yoff[idx]);
        }

        __device__ __forceinline__ uchar cvt(uchar val)
        {
            return val;
        }
        __device__ __forceinline__ uchar4 cvt(const uchar3& val)
        {
            return make_uchar4(val.x, val.y, val.z, 0);
        }
        __device__ __forceinline__ uchar4 cvt(const uchar4& val)
        {
            return val;
        }

        template <typename SrcT, typename SampleT>
        __global__ void init(const PtrStepSz<SrcT> frame, PtrStep<SampleT> samples, PtrStep<uint> randStates)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= frame.cols || y >= frame.rows)
                return;

            uint localState = randStates(y, x);

            for (int k = 0; k < c_nbSamples; ++k)
            {
                int2 np = chooseRandomNeighbor(x, y, localState, 9);

                np.x = ::max(0, ::min(np.x, frame.cols - 1));
                np.y = ::max(0, ::min(np.y, frame.rows - 1));

                SrcT pix = frame(np.y, np.x);

                samples(k * frame.rows + y, x) = cvt(pix);
            }

            randStates(y, x) = localState;
        }

        template <typename SrcT, typename SampleT>
        void init_caller(PtrStepSzb frame, PtrStepSzb samples, PtrStepSz<uint> randStates, cudaStream_t stream)
        {
            dim3 block(32, 8);
            dim3 grid(divUp(frame.cols, block.x), divUp(frame.rows, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(init<SrcT, SampleT>, cudaFuncCachePreferL1) );

            init<SrcT, SampleT><<<grid, block, 0, stream>>>((PtrStepSz<SrcT>) frame, (PtrStepSz<SampleT>) samples, randStates);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        void init_gpu(PtrStepSzb frame, int cn, PtrStepSzb samples, PtrStepSz<uint> randStates, cudaStream_t stream)
        {
            typedef void (*func_t)(PtrStepSzb frame, PtrStepSzb samples, PtrStepSz<uint> randStates, cudaStream_t stream);
            static const func_t funcs[] =
            {
                0, init_caller<uchar, uchar>, 0, init_caller<uchar3, uchar4>, init_caller<uchar4, uchar4>
            };

            funcs[cn](frame, samples, randStates, stream);
        }

        __device__ __forceinline__ int calcDist(uchar a, uchar b)
        {
            return ::abs(a - b);
        }
        __device__ __forceinline__ int calcDist(const uchar3& a, const uchar4& b)
        {
            return (::abs(a.x - b.x) + ::abs(a.y - b.y) + ::abs(a.z - b.z)) / 3;
        }
        __device__ __forceinline__ int calcDist(const uchar4& a, const uchar4& b)
        {
            return (::abs(a.x - b.x) + ::abs(a.y - b.y) + ::abs(a.z - b.z)) / 3;
        }

        template <typename SrcT, typename SampleT>
        __global__ void update(const PtrStepSz<SrcT> frame, PtrStepb fgmask, PtrStep<SampleT> samples, PtrStep<uint> randStates)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= frame.cols || y >= frame.rows)
                return;

            uint localState = randStates(y, x);

            SrcT imgPix = frame(y, x);

            // comparison with the model

            int count = 0;
            for (int k = 0; (count < c_reqMatches) && (k < c_nbSamples); ++k)
            {
                SampleT samplePix = samples(k * frame.rows + y, x);

                int distance = calcDist(imgPix, samplePix);

                if (distance < c_radius)
                    ++count;
            }

            // pixel classification according to reqMatches

            fgmask(y, x) = (uchar) (-(count < c_reqMatches));

            if (count >= c_reqMatches)
            {
                // the pixel belongs to the background

                // gets a random number between 0 and subsamplingFactor-1
                int randomNumber = nextRand(localState) % c_subsamplingFactor;

                // update of the current pixel model
                if (randomNumber == 0)
                {
                    // random subsampling

                    int k = nextRand(localState) % c_nbSamples;

                    samples(k * frame.rows + y, x) = cvt(imgPix);
                }

                // update of a neighboring pixel model
                randomNumber = nextRand(localState) % c_subsamplingFactor;

                if (randomNumber == 0)
                {
                    // random subsampling

                    // chooses a neighboring pixel randomly
                    int2 np = chooseRandomNeighbor(x, y, localState);

                    np.x = ::max(0, ::min(np.x, frame.cols - 1));
                    np.y = ::max(0, ::min(np.y, frame.rows - 1));

                    // chooses the value to be replaced randomly
                    int k = nextRand(localState) % c_nbSamples;

                    samples(k * frame.rows + np.y, np.x) = cvt(imgPix);
                }
            }

            randStates(y, x) = localState;
        }

        template <typename SrcT, typename SampleT>
        void update_caller(PtrStepSzb frame, PtrStepSzb fgmask, PtrStepSzb samples, PtrStepSz<uint> randStates, cudaStream_t stream)
        {
            dim3 block(32, 8);
            dim3 grid(divUp(frame.cols, block.x), divUp(frame.rows, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(update<SrcT, SampleT>, cudaFuncCachePreferL1) );

            update<SrcT, SampleT><<<grid, block, 0, stream>>>((PtrStepSz<SrcT>) frame, fgmask, (PtrStepSz<SampleT>) samples, randStates);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        void update_gpu(PtrStepSzb frame, int cn, PtrStepSzb fgmask, PtrStepSzb samples, PtrStepSz<uint> randStates, cudaStream_t stream)
        {
            typedef void (*func_t)(PtrStepSzb frame, PtrStepSzb fgmask, PtrStepSzb samples, PtrStepSz<uint> randStates, cudaStream_t stream);
            static const func_t funcs[] =
            {
                0, update_caller<uchar, uchar>, 0, update_caller<uchar3, uchar4>, update_caller<uchar4, uchar4>
            };

            funcs[cn](frame, fgmask, samples, randStates, stream);
        }
    }
}}}


#endif /* CUDA_DISABLER */
