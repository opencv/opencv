#if !defined CUDA_DISABLER

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/types.hpp"

namespace cv { namespace cuda { namespace device
{
    namespace blend
    {
        __global__ void addSrcWeightKernel16S(const PtrStep<short> src, const PtrStep<short> src_weight,
            PtrStep<short> dst, PtrStep<short> dst_weight, int rows, int cols)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (y < rows && x < cols)
            {
                const short3 v = ((const short3*)src.ptr(y))[x];
                short w = src_weight.ptr(y)[x];
                ((short3*)dst.ptr(y))[x].x += short((v.x * w) >> 8);
                ((short3*)dst.ptr(y))[x].y += short((v.y * w) >> 8);
                ((short3*)dst.ptr(y))[x].z += short((v.z * w) >> 8);
                dst_weight.ptr(y)[x] += w;
            }
        }

        void addSrcWeightGpu16S(const PtrStep<short> src, const PtrStep<short> src_weight,
            PtrStep<short> dst, PtrStep<short> dst_weight, cv::Rect &rc)
        {
            dim3 threads(16, 16);
            dim3 grid(divUp(rc.width, threads.x), divUp(rc.height, threads.y));
            addSrcWeightKernel16S<<<grid, threads>>>(src, src_weight, dst, dst_weight, rc.height, rc.width);
            cudaSafeCall(cudaGetLastError());
        }

        __global__ void addSrcWeightKernel32F(const PtrStep<short> src, const PtrStepf src_weight,
            PtrStep<short> dst, PtrStepf dst_weight, int rows, int cols)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (y < rows && x < cols)
            {
                const short3 v = ((const short3*)src.ptr(y))[x];
                float w = src_weight.ptr(y)[x];
                ((short3*)dst.ptr(y))[x].x += static_cast<short>(v.x * w);
                ((short3*)dst.ptr(y))[x].y += static_cast<short>(v.y * w);
                ((short3*)dst.ptr(y))[x].z += static_cast<short>(v.z * w);
                dst_weight.ptr(y)[x] += w;
            }
        }

        void addSrcWeightGpu32F(const PtrStep<short> src, const PtrStepf src_weight,
            PtrStep<short> dst, PtrStepf dst_weight, cv::Rect &rc)
        {
            dim3 threads(16, 16);
            dim3 grid(divUp(rc.width, threads.x), divUp(rc.height, threads.y));
            addSrcWeightKernel32F<<<grid, threads>>>(src, src_weight, dst, dst_weight, rc.height, rc.width);
            cudaSafeCall(cudaGetLastError());
        }

        __global__ void normalizeUsingWeightKernel16S(const PtrStep<short> weight, PtrStep<short> src,
            const int width, const int height)
        {
            int x = (blockIdx.x * blockDim.x) + threadIdx.x;
            int y = (blockIdx.y * blockDim.y) + threadIdx.y;

            if (x < width && y < height)
            {
                const short3 v = ((short3*)src.ptr(y))[x];
                short w = weight.ptr(y)[x];
                ((short3*)src.ptr(y))[x] = make_short3(short((v.x << 8) / w),
                    short((v.y << 8) / w), short((v.z << 8) / w));
            }
        }

        void normalizeUsingWeightMapGpu16S(const PtrStep<short> weight, PtrStep<short> src,
                                           const int width, const int height)
        {
            dim3 threads(16, 16);
            dim3 grid(divUp(width, threads.x), divUp(height, threads.y));
            normalizeUsingWeightKernel16S<<<grid, threads>>> (weight, src, width, height);
        }

        __global__ void normalizeUsingWeightKernel32F(const PtrStepf weight, PtrStep<short> src,
            const int width, const int height)
        {
            int x = (blockIdx.x * blockDim.x) + threadIdx.x;
            int y = (blockIdx.y * blockDim.y) + threadIdx.y;

            if (x < width && y < height)
            {
                const float WEIGHT_EPS = 1e-5f;
                const short3 v = ((short3*)src.ptr(y))[x];
                float w = weight.ptr(y)[x];
                ((short3*)src.ptr(y))[x] = make_short3(static_cast<short>(v.x / (w + WEIGHT_EPS)),
                    static_cast<short>(v.y / (w + WEIGHT_EPS)),
                    static_cast<short>(v.z / (w + WEIGHT_EPS)));
            }
        }

        void normalizeUsingWeightMapGpu32F(const PtrStepf weight, PtrStep<short> src,
                                           const int width, const int height)
        {
            dim3 threads(16, 16);
            dim3 grid(divUp(width, threads.x), divUp(height, threads.y));
            normalizeUsingWeightKernel32F<<<grid, threads>>> (weight, src, width, height);
        }
    }
}}}

#endif
