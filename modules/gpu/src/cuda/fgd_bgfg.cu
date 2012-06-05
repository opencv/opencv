#include "opencv2/gpu/device/common.hpp"
#include "opencv2/gpu/device/vec_math.hpp"
#include "opencv2/gpu/device/limits.hpp"
#include "opencv2/gpu/device/utility.hpp"
#include "fgd_bgfg_common.hpp"

using namespace cv::gpu;
using namespace cv::gpu::device;

namespace bgfg
{
    ////////////////////////////////////////////////////////////////////////////
    // calcDiffHistogram

    const unsigned int UINT_BITS = 32U;
    const int LOG_WARP_SIZE = 5;
    const int WARP_SIZE = 1 << LOG_WARP_SIZE;
#if (__CUDA_ARCH__ < 120)
    const unsigned int TAG_MASK = (1U << (UINT_BITS - LOG_WARP_SIZE)) - 1U;
#endif

    const int MERGE_THREADBLOCK_SIZE = 256;

    __device__ __forceinline__ void addByte(unsigned int* s_WarpHist_, unsigned int data, unsigned int threadTag)
    {
        #if (__CUDA_ARCH__ < 120)
            volatile unsigned int* s_WarpHist = s_WarpHist_;
            unsigned int count;
            do
            {
                count = s_WarpHist[data] & TAG_MASK;
                count = threadTag | (count + 1);
                s_WarpHist[data] = count;
            } while (s_WarpHist[data] != count);
        #else
            atomicInc(s_WarpHist_ + data, (unsigned int)(-1));
        #endif
    }


    template <typename PT, typename CT>
    __global__ void calcPartialHistogram(const DevMem2D_<PT> prevFrame, const PtrStep_<CT> curFrame, unsigned int* partialBuf0, unsigned int* partialBuf1, unsigned int* partialBuf2)
    {
#if (__CUDA_ARCH__ < 200)
        const int HISTOGRAM_WARP_COUNT = 4;
#else
        const int HISTOGRAM_WARP_COUNT = 6;
#endif
        const int HISTOGRAM_THREADBLOCK_SIZE = HISTOGRAM_WARP_COUNT * WARP_SIZE;
        const int HISTOGRAM_THREADBLOCK_MEMORY = HISTOGRAM_WARP_COUNT * HISTOGRAM_BIN_COUNT;

        //Per-warp subhistogram storage
        __shared__ unsigned int s_Hist0[HISTOGRAM_THREADBLOCK_MEMORY];
        __shared__ unsigned int s_Hist1[HISTOGRAM_THREADBLOCK_MEMORY];
        __shared__ unsigned int s_Hist2[HISTOGRAM_THREADBLOCK_MEMORY];

        //Clear shared memory storage for current threadblock before processing
        #pragma unroll
        for (int i = 0; i < (HISTOGRAM_THREADBLOCK_MEMORY / HISTOGRAM_THREADBLOCK_SIZE); ++i)
        {
           s_Hist0[threadIdx.x + i * HISTOGRAM_THREADBLOCK_SIZE] = 0;
           s_Hist1[threadIdx.x + i * HISTOGRAM_THREADBLOCK_SIZE] = 0;
           s_Hist2[threadIdx.x + i * HISTOGRAM_THREADBLOCK_SIZE] = 0;
        }
        __syncthreads();

        const unsigned int warpId = threadIdx.x >> LOG_WARP_SIZE;

        unsigned int* s_WarpHist0 = s_Hist0 + warpId * HISTOGRAM_BIN_COUNT;
        unsigned int* s_WarpHist1 = s_Hist1 + warpId * HISTOGRAM_BIN_COUNT;
        unsigned int* s_WarpHist2 = s_Hist2 + warpId * HISTOGRAM_BIN_COUNT;

        const unsigned int tag = threadIdx.x << (UINT_BITS - LOG_WARP_SIZE);
        const int dataCount = prevFrame.rows * prevFrame.cols;
        for (unsigned int pos = blockIdx.x * HISTOGRAM_THREADBLOCK_SIZE + threadIdx.x; pos < dataCount; pos += HISTOGRAM_THREADBLOCK_SIZE * PARTIAL_HISTOGRAM_COUNT)
        {
            const unsigned int y = pos / prevFrame.cols;
            const unsigned int x = pos % prevFrame.cols;

            PT prevVal = prevFrame(y, x);
            CT curVal = curFrame(y, x);

            int3 diff = make_int3(
                ::abs(curVal.x - prevVal.x),
                ::abs(curVal.y - prevVal.y),
                ::abs(curVal.z - prevVal.z)
            );

            addByte(s_WarpHist0, diff.x, tag);
            addByte(s_WarpHist1, diff.y, tag);
            addByte(s_WarpHist2, diff.z, tag);
        }
        __syncthreads();

        //Merge per-warp histograms into per-block and write to global memory
        for (unsigned int bin = threadIdx.x; bin < HISTOGRAM_BIN_COUNT; bin += HISTOGRAM_THREADBLOCK_SIZE)
        {
            unsigned int sum0 = 0;
            unsigned int sum1 = 0;
            unsigned int sum2 = 0;

            #pragma unroll
            for (int i = 0; i < HISTOGRAM_WARP_COUNT; ++i)
            {
                #if (__CUDA_ARCH__ < 120)
                    sum0 += s_Hist0[bin + i * HISTOGRAM_BIN_COUNT] & TAG_MASK;
                    sum1 += s_Hist1[bin + i * HISTOGRAM_BIN_COUNT] & TAG_MASK;
                    sum2 += s_Hist2[bin + i * HISTOGRAM_BIN_COUNT] & TAG_MASK;
                #else
                    sum0 += s_Hist0[bin + i * HISTOGRAM_BIN_COUNT];
                    sum1 += s_Hist1[bin + i * HISTOGRAM_BIN_COUNT];
                    sum2 += s_Hist2[bin + i * HISTOGRAM_BIN_COUNT];
                #endif
            }

            partialBuf0[blockIdx.x * HISTOGRAM_BIN_COUNT + bin] = sum0;
            partialBuf1[blockIdx.x * HISTOGRAM_BIN_COUNT + bin] = sum1;
            partialBuf2[blockIdx.x * HISTOGRAM_BIN_COUNT + bin] = sum2;
        }
    }

    __global__ void mergeHistogram(const unsigned int* partialBuf0, const unsigned int* partialBuf1, const unsigned int* partialBuf2, unsigned int* hist0, unsigned int* hist1, unsigned int* hist2)
    {
        unsigned int sum0 = 0;
        unsigned int sum1 = 0;
        unsigned int sum2 = 0;

        #pragma unroll
        for (unsigned int i = threadIdx.x; i < PARTIAL_HISTOGRAM_COUNT; i += MERGE_THREADBLOCK_SIZE)
        {
            sum0 += partialBuf0[blockIdx.x + i * HISTOGRAM_BIN_COUNT];
            sum1 += partialBuf1[blockIdx.x + i * HISTOGRAM_BIN_COUNT];
            sum2 += partialBuf2[blockIdx.x + i * HISTOGRAM_BIN_COUNT];
        }

        __shared__ unsigned int data0[MERGE_THREADBLOCK_SIZE];
        __shared__ unsigned int data1[MERGE_THREADBLOCK_SIZE];
        __shared__ unsigned int data2[MERGE_THREADBLOCK_SIZE];

        data0[threadIdx.x] = sum0;
        data1[threadIdx.x] = sum1;
        data2[threadIdx.x] = sum2;
        __syncthreads();

        if (threadIdx.x < 128)
        {
            data0[threadIdx.x] = sum0 += data0[threadIdx.x + 128];
            data1[threadIdx.x] = sum1 += data1[threadIdx.x + 128];
            data2[threadIdx.x] = sum2 += data2[threadIdx.x + 128];
        }
        __syncthreads();

        if (threadIdx.x < 64)
        {
            data0[threadIdx.x] = sum0 += data0[threadIdx.x + 64];
            data1[threadIdx.x] = sum1 += data1[threadIdx.x + 64];
            data2[threadIdx.x] = sum2 += data2[threadIdx.x + 64];
        }
        __syncthreads();

        if (threadIdx.x < 32)
        {
            volatile unsigned int* vdata0 = data0;
            volatile unsigned int* vdata1 = data1;
            volatile unsigned int* vdata2 = data2;

            vdata0[threadIdx.x] = sum0 += vdata0[threadIdx.x + 32];
            vdata1[threadIdx.x] = sum1 += vdata1[threadIdx.x + 32];
            vdata2[threadIdx.x] = sum2 += vdata2[threadIdx.x + 32];

            vdata0[threadIdx.x] = sum0 += vdata0[threadIdx.x + 16];
            vdata1[threadIdx.x] = sum1 += vdata1[threadIdx.x + 16];
            vdata2[threadIdx.x] = sum2 += vdata2[threadIdx.x + 16];

            vdata0[threadIdx.x] = sum0 += vdata0[threadIdx.x + 8];
            vdata1[threadIdx.x] = sum1 += vdata1[threadIdx.x + 8];
            vdata2[threadIdx.x] = sum2 += vdata2[threadIdx.x + 8];

            vdata0[threadIdx.x] = sum0 += vdata0[threadIdx.x + 4];
            vdata1[threadIdx.x] = sum1 += vdata1[threadIdx.x + 4];
            vdata2[threadIdx.x] = sum2 += vdata2[threadIdx.x + 4];

            vdata0[threadIdx.x] = sum0 += vdata0[threadIdx.x + 2];
            vdata1[threadIdx.x] = sum1 += vdata1[threadIdx.x + 2];
            vdata2[threadIdx.x] = sum2 += vdata2[threadIdx.x + 2];

            vdata0[threadIdx.x] = sum0 += vdata0[threadIdx.x + 1];
            vdata1[threadIdx.x] = sum1 += vdata1[threadIdx.x + 1];
            vdata2[threadIdx.x] = sum2 += vdata2[threadIdx.x + 1];
        }

        if(threadIdx.x == 0)
        {
            hist0[blockIdx.x] = sum0;
            hist1[blockIdx.x] = sum1;
            hist2[blockIdx.x] = sum2;
        }
    }

    template <typename PT, typename CT>
    void calcDiffHistogram_gpu(DevMem2Db prevFrame, DevMem2Db curFrame,
                               unsigned int* hist0, unsigned int* hist1, unsigned int* hist2,
                               unsigned int* partialBuf0, unsigned int* partialBuf1, unsigned int* partialBuf2,
                               int cc, cudaStream_t stream)
    {
        const int HISTOGRAM_WARP_COUNT = cc < 20 ? 4 : 6;
        const int HISTOGRAM_THREADBLOCK_SIZE = HISTOGRAM_WARP_COUNT * WARP_SIZE;

        calcPartialHistogram<PT, CT><<<PARTIAL_HISTOGRAM_COUNT, HISTOGRAM_THREADBLOCK_SIZE, 0, stream>>>(
                (DevMem2D_<PT>)prevFrame, (DevMem2D_<CT>)curFrame, partialBuf0, partialBuf1, partialBuf2);
        cudaSafeCall( cudaGetLastError() );

        mergeHistogram<<<HISTOGRAM_BIN_COUNT, MERGE_THREADBLOCK_SIZE, 0, stream>>>(partialBuf0, partialBuf1, partialBuf2, hist0, hist1, hist2);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template void calcDiffHistogram_gpu<uchar3, uchar3>(DevMem2Db prevFrame, DevMem2Db curFrame, unsigned int* hist0, unsigned int* hist1, unsigned int* hist2, unsigned int* partialBuf0, unsigned int* partialBuf1, unsigned int* partialBuf2, int cc, cudaStream_t stream);
    template void calcDiffHistogram_gpu<uchar3, uchar4>(DevMem2Db prevFrame, DevMem2Db curFrame, unsigned int* hist0, unsigned int* hist1, unsigned int* hist2, unsigned int* partialBuf0, unsigned int* partialBuf1, unsigned int* partialBuf2, int cc, cudaStream_t stream);
    template void calcDiffHistogram_gpu<uchar4, uchar3>(DevMem2Db prevFrame, DevMem2Db curFrame, unsigned int* hist0, unsigned int* hist1, unsigned int* hist2, unsigned int* partialBuf0, unsigned int* partialBuf1, unsigned int* partialBuf2, int cc, cudaStream_t stream);
    template void calcDiffHistogram_gpu<uchar4, uchar4>(DevMem2Db prevFrame, DevMem2Db curFrame, unsigned int* hist0, unsigned int* hist1, unsigned int* hist2, unsigned int* partialBuf0, unsigned int* partialBuf1, unsigned int* partialBuf2, int cc, cudaStream_t stream);

    /////////////////////////////////////////////////////////////////////////
    // calcDiffThreshMask

    template <typename PT, typename CT>
    __global__ void calcDiffThreshMask(const DevMem2D_<PT> prevFrame, const PtrStep_<CT> curFrame, uchar3 bestThres, PtrStepb changeMask)
    {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int x = blockIdx.x * blockDim.x + threadIdx.x;

        if (y > prevFrame.rows || x > prevFrame.cols)
            return;

        PT prevVal = prevFrame(y, x);
        CT curVal = curFrame(y, x);

        int3 diff = make_int3(
            ::abs(curVal.x - prevVal.x),
            ::abs(curVal.y - prevVal.y),
            ::abs(curVal.z - prevVal.z)
        );

        if (diff.x > bestThres.x || diff.y > bestThres.y || diff.z > bestThres.z)
            changeMask(y, x) = 255;
    }

    template <typename PT, typename CT>
    void calcDiffThreshMask_gpu(DevMem2Db prevFrame, DevMem2Db curFrame, uchar3 bestThres, DevMem2Db changeMask, cudaStream_t stream)
    {
        dim3 block(32, 8);
        dim3 grid(divUp(prevFrame.cols, block.x), divUp(prevFrame.rows, block.y));

        calcDiffThreshMask<PT, CT><<<grid, block, 0, stream>>>((DevMem2D_<PT>)prevFrame, (DevMem2D_<CT>)curFrame, bestThres, changeMask);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template void calcDiffThreshMask_gpu<uchar3, uchar3>(DevMem2Db prevFrame, DevMem2Db curFrame, uchar3 bestThres, DevMem2Db changeMask, cudaStream_t stream);
    template void calcDiffThreshMask_gpu<uchar3, uchar4>(DevMem2Db prevFrame, DevMem2Db curFrame, uchar3 bestThres, DevMem2Db changeMask, cudaStream_t stream);
    template void calcDiffThreshMask_gpu<uchar4, uchar3>(DevMem2Db prevFrame, DevMem2Db curFrame, uchar3 bestThres, DevMem2Db changeMask, cudaStream_t stream);
    template void calcDiffThreshMask_gpu<uchar4, uchar4>(DevMem2Db prevFrame, DevMem2Db curFrame, uchar3 bestThres, DevMem2Db changeMask, cudaStream_t stream);

    /////////////////////////////////////////////////////////////////////////
    // bgfgClassification

    __constant__ BGPixelStat c_stat;

    void setBGPixelStat(const BGPixelStat& stat)
    {
        cudaSafeCall( cudaMemcpyToSymbol(c_stat, &stat, sizeof(BGPixelStat)) );
    }

    template <typename T> struct Output;
    template <> struct Output<uchar3>
    {
        static __device__ __forceinline__ uchar3 make(uchar v0, uchar v1, uchar v2)
        {
            return make_uchar3(v0, v1, v2);
        }
    };
    template <> struct Output<uchar4>
    {
        static __device__ __forceinline__ uchar4 make(uchar v0, uchar v1, uchar v2)
        {
            return make_uchar4(v0, v1, v2, 255);
        }
    };

    template <typename PT, typename CT, typename OT>
    __global__ void bgfgClassification(const DevMem2D_<PT> prevFrame, const PtrStep_<CT> curFrame,
                                       const PtrStepb Ftd, const PtrStepb Fbd, PtrStepb foreground,
                                       int deltaC, int deltaCC, float alpha2, int N1c, int N1cc)
    {
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        const int j = blockIdx.x * blockDim.x + threadIdx.x;

        if (i > prevFrame.rows || j > prevFrame.cols)
            return;

        if (Fbd(i, j) || Ftd(i, j))
        {
            float Pb  = 0.0f;
            float Pv  = 0.0f;
            float Pvb = 0.0f;

            int val = 0;

            // Is it a motion pixel?
            if (Ftd(i, j))
            {
                if (!c_stat.is_trained_dyn_model(i, j))
                    val = 1;
                else
                {
                    PT prevVal = prevFrame(i, j);
                    CT curVal = curFrame(i, j);

                    // Compare with stored CCt vectors:
                    for (int k = 0; k < N1cc && c_stat.PV_CC(i, j, k) > alpha2; ++k)
                    {
                        OT v1 = c_stat.V1_CC<OT>(i, j, k);
                        OT v2 = c_stat.V2_CC<OT>(i, j, k);

                        if (::abs(v1.x - prevVal.x) <= deltaCC &&
                            ::abs(v1.y - prevVal.y) <= deltaCC &&
                            ::abs(v1.z - prevVal.z) <= deltaCC &&
                            ::abs(v2.x - curVal.x) <= deltaCC &&
                            ::abs(v2.y - curVal.y) <= deltaCC &&
                            ::abs(v2.z - curVal.z) <= deltaCC)
                        {
                            Pv += c_stat.PV_CC(i, j, k);
                            Pvb += c_stat.PVB_CC(i, j, k);
                        }
                    }

                    Pb = c_stat.Pbcc(i, j);
                    if (2 * Pvb * Pb <= Pv)
                        val = 1;
                }
            }
            else if(c_stat.is_trained_st_model(i, j))
            {
                CT curVal = curFrame(i, j);

                // Compare with stored Ct vectors:
                for (int k = 0; k < N1c && c_stat.PV_C(i, j, k) > alpha2; ++k)
                {
                    OT v = c_stat.V_C<OT>(i, j, k);

                    if (::abs(v.x - curVal.x) <= deltaC &&
                        ::abs(v.y - curVal.y) <= deltaC &&
                        ::abs(v.z - curVal.z) <= deltaC)
                    {
                        Pv += c_stat.PV_C(i, j, k);
                        Pvb += c_stat.PVB_C(i, j, k);
                    }
                }
                Pb = c_stat.Pbc(i, j);
                if (2 * Pvb * Pb <= Pv)
                    val = 1;
            }

            // Update foreground:
            foreground(i, j) = static_cast<uchar>(val);
        } // end if( change detection...
    }

    template <typename PT, typename CT, typename OT>
    void bgfgClassification_gpu(DevMem2Db prevFrame, DevMem2Db curFrame, DevMem2Db Ftd, DevMem2Db Fbd, DevMem2Db foreground,
                                int deltaC, int deltaCC, float alpha2, int N1c, int N1cc, cudaStream_t stream)
    {
        dim3 block(32, 8);
        dim3 grid(divUp(prevFrame.cols, block.x), divUp(prevFrame.rows, block.y));

        cudaSafeCall( cudaFuncSetCacheConfig(bgfgClassification<PT, CT, OT>, cudaFuncCachePreferL1) );

        bgfgClassification<PT, CT, OT><<<grid, block, 0, stream>>>((DevMem2D_<PT>)prevFrame, (DevMem2D_<CT>)curFrame,
                                                                   Ftd, Fbd, foreground,
                                                                   deltaC, deltaCC, alpha2, N1c, N1cc);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template void bgfgClassification_gpu<uchar3, uchar3, uchar3>(DevMem2Db prevFrame, DevMem2Db curFrame, DevMem2Db Ftd, DevMem2Db Fbd, DevMem2Db foreground, int deltaC, int deltaCC, float alpha2, int N1c, int N1cc, cudaStream_t stream);
    template void bgfgClassification_gpu<uchar3, uchar3, uchar4>(DevMem2Db prevFrame, DevMem2Db curFrame, DevMem2Db Ftd, DevMem2Db Fbd, DevMem2Db foreground, int deltaC, int deltaCC, float alpha2, int N1c, int N1cc, cudaStream_t stream);
    template void bgfgClassification_gpu<uchar3, uchar4, uchar3>(DevMem2Db prevFrame, DevMem2Db curFrame, DevMem2Db Ftd, DevMem2Db Fbd, DevMem2Db foreground, int deltaC, int deltaCC, float alpha2, int N1c, int N1cc, cudaStream_t stream);
    template void bgfgClassification_gpu<uchar3, uchar4, uchar4>(DevMem2Db prevFrame, DevMem2Db curFrame, DevMem2Db Ftd, DevMem2Db Fbd, DevMem2Db foreground, int deltaC, int deltaCC, float alpha2, int N1c, int N1cc, cudaStream_t stream);
    template void bgfgClassification_gpu<uchar4, uchar3, uchar3>(DevMem2Db prevFrame, DevMem2Db curFrame, DevMem2Db Ftd, DevMem2Db Fbd, DevMem2Db foreground, int deltaC, int deltaCC, float alpha2, int N1c, int N1cc, cudaStream_t stream);
    template void bgfgClassification_gpu<uchar4, uchar3, uchar4>(DevMem2Db prevFrame, DevMem2Db curFrame, DevMem2Db Ftd, DevMem2Db Fbd, DevMem2Db foreground, int deltaC, int deltaCC, float alpha2, int N1c, int N1cc, cudaStream_t stream);
    template void bgfgClassification_gpu<uchar4, uchar4, uchar3>(DevMem2Db prevFrame, DevMem2Db curFrame, DevMem2Db Ftd, DevMem2Db Fbd, DevMem2Db foreground, int deltaC, int deltaCC, float alpha2, int N1c, int N1cc, cudaStream_t stream);
    template void bgfgClassification_gpu<uchar4, uchar4, uchar4>(DevMem2Db prevFrame, DevMem2Db curFrame, DevMem2Db Ftd, DevMem2Db Fbd, DevMem2Db foreground, int deltaC, int deltaCC, float alpha2, int N1c, int N1cc, cudaStream_t stream);

    ////////////////////////////////////////////////////////////////////////////
    // updateBackgroundModel

    template <typename PT, typename CT, typename OT, class PrevFramePtr2D, class CurFramePtr2D, class FtdPtr2D, class FbdPtr2D>
    __global__ void updateBackgroundModel(int cols, int rows, const PrevFramePtr2D prevFrame, const CurFramePtr2D curFrame, const FtdPtr2D Ftd, const FbdPtr2D Fbd,
                                          PtrStepb foreground, PtrStep_<OT> background,
                                          int deltaC, int deltaCC, float alpha1, float alpha2, float alpha3, int N1c, int N1cc, int N2c, int N2cc, float T)
    {
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        const int j = blockIdx.x * blockDim.x + threadIdx.x;

        if (i > rows || j > cols)
            return;

        const float MIN_PV = 1e-10f;

        const uchar is_trained_dyn_model = c_stat.is_trained_dyn_model(i, j);
        if (Ftd(i, j) || !is_trained_dyn_model)
        {
            const float alpha = is_trained_dyn_model ? alpha2 : alpha3;

            float Pbcc = c_stat.Pbcc(i, j);

            //update Pb
            Pbcc *= (1.0f - alpha);
            if (!foreground(i, j))
            {
                Pbcc += alpha;
            }

            int min_dist = numeric_limits<int>::max();
            int indx = -1;

            PT prevVal = prevFrame(i, j);
            CT curVal = curFrame(i, j);

            // Find best Vi match:
            for (int k = 0; k < N2cc; ++k)
            {
                float PV_CC = c_stat.PV_CC(i, j, k);
                if (!PV_CC)
                    break;

                if (PV_CC < MIN_PV)
                {
                    c_stat.PV_CC(i, j, k) = 0;
                    c_stat.PVB_CC(i, j, k) = 0;
                    continue;
                }

                c_stat.PV_CC(i, j, k) = PV_CC * (1.0f - alpha);
                c_stat.PVB_CC(i, j, k) = c_stat.PVB_CC(i, j, k) * (1.0f - alpha);

                OT v1 = c_stat.V1_CC<OT>(i, j, k);

                int3 val1 = make_int3(
                    ::abs(v1.x - prevVal.x),
                    ::abs(v1.y - prevVal.y),
                    ::abs(v1.z - prevVal.z)
                );

                OT v2 = c_stat.V2_CC<OT>(i, j, k);

                int3 val2 = make_int3(
                    ::abs(v2.x - curVal.x),
                    ::abs(v2.y - curVal.y),
                    ::abs(v2.z - curVal.z)
                );

                int dist = val1.x + val1.y + val1.z + val2.x + val2.y + val2.z;

                if (dist < min_dist &&
                    val1.x <= deltaCC && val1.y <= deltaCC && val1.z <= deltaCC &&
                    val2.x <= deltaCC && val2.y <= deltaCC && val2.z <= deltaCC)
                {
                    min_dist = dist;
                    indx = k;
                }
            }

            if (indx < 0)
            {
                // Replace N2th elem in the table by new feature:
                indx = N2cc - 1;
                c_stat.PV_CC(i, j, indx) = alpha;
                c_stat.PVB_CC(i, j, indx) = alpha;

                //udate Vt
                c_stat.V1_CC<OT>(i, j, indx) = Output<OT>::make(prevVal.x, prevVal.y, prevVal.z);
                c_stat.V2_CC<OT>(i, j, indx) = Output<OT>::make(curVal.x, curVal.y, curVal.z);
            }
            else
            {
                // Update:
                c_stat.PV_CC(i, j, indx) += alpha;

                if (!foreground(i, j))
                {
                    c_stat.PVB_CC(i, j, indx) += alpha;
                }
            }

            //re-sort CCt table by Pv
            const float PV_CC_indx = c_stat.PV_CC(i, j, indx);
            const float PVB_CC_indx = c_stat.PVB_CC(i, j, indx);
            const OT V1_CC_indx = c_stat.V1_CC<OT>(i, j, indx);
            const OT V2_CC_indx = c_stat.V2_CC<OT>(i, j, indx);
            for (int k = 0; k < indx; ++k)
            {
                if (c_stat.PV_CC(i, j, k) <= PV_CC_indx)
                {
                    //shift elements
                    float Pv_tmp1;
                    float Pv_tmp2 = PV_CC_indx;

                    float Pvb_tmp1;
                    float Pvb_tmp2 = PVB_CC_indx;

                    OT v1_tmp1;
                    OT v1_tmp2 = V1_CC_indx;

                    OT v2_tmp1;
                    OT v2_tmp2 = V2_CC_indx;

                    for (int l = k; l <= indx; ++l)
                    {
                        Pv_tmp1 = c_stat.PV_CC(i, j, l);
                        c_stat.PV_CC(i, j, l) = Pv_tmp2;
                        Pv_tmp2 = Pv_tmp1;

                        Pvb_tmp1 = c_stat.PVB_CC(i, j, l);
                        c_stat.PVB_CC(i, j, l) = Pvb_tmp2;
                        Pvb_tmp2 = Pvb_tmp1;

                        v1_tmp1 = c_stat.V1_CC<OT>(i, j, l);
                        c_stat.V1_CC<OT>(i, j, l) = v1_tmp2;
                        v1_tmp2 = v1_tmp1;

                        v2_tmp1 = c_stat.V2_CC<OT>(i, j, l);
                        c_stat.V2_CC<OT>(i, j, l) = v2_tmp2;
                        v2_tmp2 = v2_tmp1;
                    }

                    break;
                }
            }

            float sum1 = 0.0f;
            float sum2 = 0.0f;

            //check "once-off" changes
            for (int k = 0; k < N1cc; ++k)
            {
                const float PV_CC = c_stat.PV_CC(i, j, k);
                if (!PV_CC)
                    break;

                sum1 += PV_CC;
                sum2 += c_stat.PVB_CC(i, j, k);
            }

            if (sum1 > T)
                c_stat.is_trained_dyn_model(i, j) = 1;

            float diff = sum1 - Pbcc * sum2;

            // Update stat table:
            if (diff > T)
            {
                //new BG features are discovered
                for (int k = 0; k < N1cc; ++k)
                {
                    const float PV_CC = c_stat.PV_CC(i, j, k);
                    if (!PV_CC)
                        break;

                    c_stat.PVB_CC(i, j, k) = (PV_CC - Pbcc * c_stat.PVB_CC(i, j, k)) / (1.0f - Pbcc);
                }
            }

            c_stat.Pbcc(i, j) = Pbcc;
        }

        // Handle "stationary" pixel:
        if (!Ftd(i, j))
        {
            const float alpha = c_stat.is_trained_st_model(i, j) ? alpha2 : alpha3;

            float Pbc = c_stat.Pbc(i, j);

            //update Pb
            Pbc *= (1.0f - alpha);
            if (!foreground(i, j))
            {
                Pbc += alpha;
            }

            int min_dist = numeric_limits<int>::max();
            int indx = -1;

            CT curVal = curFrame(i, j);

            //find best Vi match
            for (int k = 0; k < N2c; ++k)
            {
                float PV_C = c_stat.PV_C(i, j, k);

                if (PV_C < MIN_PV)
                {
                    c_stat.PV_C(i, j, k) = 0;
                    c_stat.PVB_C(i, j, k) = 0;
                    continue;
                }

                // Exponential decay of memory
                c_stat.PV_C(i, j, k) = PV_C * (1.0f - alpha);
                c_stat.PVB_C(i, j, k) = c_stat.PVB_C(i, j, k) * (1.0f - alpha);

                OT v = c_stat.V_C<OT>(i, j, k);
                int3 val = make_int3(
                    ::abs(v.x - curVal.x),
                    ::abs(v.y - curVal.y),
                    ::abs(v.z - curVal.z)
                );

                int dist = val.x + val.y + val.z;

                if (dist < min_dist && val.x <= deltaC && val.y <= deltaC && val.z <= deltaC)
                {
                    min_dist = dist;
                    indx = k;
                }
            }

            if (indx < 0)
            {
                //N2th elem in the table is replaced by a new features
                indx = N2c - 1;

                c_stat.PV_C(i, j, indx) = alpha;
                c_stat.PVB_C(i, j, indx) = alpha;

                //udate Vt
                c_stat.V_C<OT>(i, j, indx) = Output<OT>::make(curVal.x, curVal.y, curVal.z);
            }
            else
            {
                //update
                c_stat.PV_C(i, j, indx) += alpha;

                if (!foreground(i, j))
                {
                    c_stat.PVB_C(i, j, indx) += alpha;
                }
            }

            //re-sort Ct table by Pv
            const float PV_C_indx = c_stat.PV_C(i, j, indx);
            const float PVB_C_indx = c_stat.PVB_C(i, j, indx);
            OT V_C_indx = c_stat.V_C<OT>(i, j, indx);
            for (int k = 0; k < indx; ++k)
            {
                if (c_stat.PV_C(i, j, k) <= PV_C_indx)
                {
                    //shift elements
                    float Pv_tmp1;
                    float Pv_tmp2 = PV_C_indx;

                    float Pvb_tmp1;
                    float Pvb_tmp2 = PVB_C_indx;

                    OT v_tmp1;
                    OT v_tmp2 = V_C_indx;

                    for (int l = k; l <= indx; ++l)
                    {
                        Pv_tmp1 = c_stat.PV_C(i, j, l);
                        c_stat.PV_C(i, j, l) = Pv_tmp2;
                        Pv_tmp2 = Pv_tmp1;

                        Pvb_tmp1 = c_stat.PVB_C(i, j, l);
                        c_stat.PVB_C(i, j, l) = Pvb_tmp2;
                        Pvb_tmp2 = Pvb_tmp1;

                        v_tmp1 = c_stat.V_C<OT>(i, j, l);
                        c_stat.V_C<OT>(i, j, l) = v_tmp2;
                        v_tmp2 = v_tmp1;
                    }

                    break;
                }
            }

            // Check "once-off" changes:
            float sum1 = 0.0f;
            float sum2 = 0.0f;
            for (int k = 0; k < N1c; ++k)
            {
                const float PV_C = c_stat.PV_C(i, j, k);
                if (!PV_C)
                    break;

                sum1 += PV_C;
                sum2 += c_stat.PVB_C(i, j, k);
            }

            if (sum1 > T)
                c_stat.is_trained_st_model(i, j) = 1;

            float diff = sum1 - Pbc * sum2;

            // Update stat table:
            if (diff > T)
            {
                //new BG features are discovered
                for (int k = 0; k < N1c; ++k)
                {
                    const float PV_C = c_stat.PV_C(i, j, k);
                    if (!PV_C)
                        break;

                    c_stat.PVB_C(i, j, k) = (PV_C - Pbc * c_stat.PVB_C(i, j, k)) / (1.0f - Pbc);
                }

                c_stat.Pbc(i, j) = 1.0f - Pbc;
            }
            else
            {
                c_stat.Pbc(i, j) = Pbc;
            }
        } // if !(change detection) at pixel (i,j)

        // Update the reference BG image:
        if (!foreground(i, j))
        {
            CT curVal = curFrame(i, j);

            if (!Ftd(i, j) && !Fbd(i, j))
            {
                // Apply IIR filter:
                OT oldVal = background(i, j);

                int3 newVal = make_int3(
                    __float2int_rn(oldVal.x * (1.0f - alpha1) + curVal.x * alpha1),
                    __float2int_rn(oldVal.y * (1.0f - alpha1) + curVal.y * alpha1),
                    __float2int_rn(oldVal.z * (1.0f - alpha1) + curVal.z * alpha1)
                );

                background(i, j) = Output<OT>::make(
                    static_cast<uchar>(newVal.x),
                    static_cast<uchar>(newVal.y),
                    static_cast<uchar>(newVal.z)
                );
            }
            else
            {
                background(i, j) = Output<OT>::make(curVal.x, curVal.y, curVal.z);
            }
        }
    }

    template <typename PT, typename CT, typename OT>
    struct UpdateBackgroundModel
    {
        static void call(DevMem2D_<PT> prevFrame, DevMem2D_<CT> curFrame, DevMem2Db Ftd, DevMem2Db Fbd, DevMem2Db foreground, DevMem2D_<OT> background,
                         int deltaC, int deltaCC, float alpha1, float alpha2, float alpha3, int N1c, int N1cc, int N2c, int N2cc, float T,
                         cudaStream_t stream)
        {
            dim3 block(32, 8);
            dim3 grid(divUp(prevFrame.cols, block.x), divUp(prevFrame.rows, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(updateBackgroundModel<PT, CT, OT, PtrStep_<PT>, PtrStep_<CT>, PtrStepb, PtrStepb>, cudaFuncCachePreferL1) );

            updateBackgroundModel<PT, CT, OT, PtrStep_<PT>, PtrStep_<CT>, PtrStepb, PtrStepb><<<grid, block, 0, stream>>>(
                prevFrame.cols, prevFrame.rows,
                prevFrame, curFrame,
                Ftd, Fbd, foreground, background,
                deltaC, deltaCC, alpha1, alpha2, alpha3, N1c, N1cc, N2c, N2cc, T);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };

    template <typename PT, typename CT, typename OT>
    void updateBackgroundModel_gpu(DevMem2Db prevFrame, DevMem2Db curFrame, DevMem2Db Ftd, DevMem2Db Fbd, DevMem2Db foreground, DevMem2Db background,
                                   int deltaC, int deltaCC, float alpha1, float alpha2, float alpha3, int N1c, int N1cc, int N2c, int N2cc, float T,
                                   cudaStream_t stream)
    {
        UpdateBackgroundModel<PT, CT, OT>::call(DevMem2D_<PT>(prevFrame), DevMem2D_<CT>(curFrame), Ftd, Fbd, foreground, DevMem2D_<OT>(background),
                                                deltaC, deltaCC, alpha1, alpha2, alpha3, N1c, N1cc, N2c, N2cc, T, stream);
    }

    template void updateBackgroundModel_gpu<uchar3, uchar3, uchar3>(DevMem2Db prevFrame, DevMem2Db curFrame, DevMem2Db Ftd, DevMem2Db Fbd, DevMem2Db foreground, DevMem2Db background, int deltaC, int deltaCC, float alpha1, float alpha2, float alpha3, int N1c, int N1cc, int N2c, int N2cc, float T, cudaStream_t stream);
    template void updateBackgroundModel_gpu<uchar3, uchar3, uchar4>(DevMem2Db prevFrame, DevMem2Db curFrame, DevMem2Db Ftd, DevMem2Db Fbd, DevMem2Db foreground, DevMem2Db background, int deltaC, int deltaCC, float alpha1, float alpha2, float alpha3, int N1c, int N1cc, int N2c, int N2cc, float T, cudaStream_t stream);
    template void updateBackgroundModel_gpu<uchar3, uchar4, uchar3>(DevMem2Db prevFrame, DevMem2Db curFrame, DevMem2Db Ftd, DevMem2Db Fbd, DevMem2Db foreground, DevMem2Db background, int deltaC, int deltaCC, float alpha1, float alpha2, float alpha3, int N1c, int N1cc, int N2c, int N2cc, float T, cudaStream_t stream);
    template void updateBackgroundModel_gpu<uchar3, uchar4, uchar4>(DevMem2Db prevFrame, DevMem2Db curFrame, DevMem2Db Ftd, DevMem2Db Fbd, DevMem2Db foreground, DevMem2Db background, int deltaC, int deltaCC, float alpha1, float alpha2, float alpha3, int N1c, int N1cc, int N2c, int N2cc, float T, cudaStream_t stream);
    template void updateBackgroundModel_gpu<uchar4, uchar3, uchar3>(DevMem2Db prevFrame, DevMem2Db curFrame, DevMem2Db Ftd, DevMem2Db Fbd, DevMem2Db foreground, DevMem2Db background, int deltaC, int deltaCC, float alpha1, float alpha2, float alpha3, int N1c, int N1cc, int N2c, int N2cc, float T, cudaStream_t stream);
    template void updateBackgroundModel_gpu<uchar4, uchar3, uchar4>(DevMem2Db prevFrame, DevMem2Db curFrame, DevMem2Db Ftd, DevMem2Db Fbd, DevMem2Db foreground, DevMem2Db background, int deltaC, int deltaCC, float alpha1, float alpha2, float alpha3, int N1c, int N1cc, int N2c, int N2cc, float T, cudaStream_t stream);
    template void updateBackgroundModel_gpu<uchar4, uchar4, uchar3>(DevMem2Db prevFrame, DevMem2Db curFrame, DevMem2Db Ftd, DevMem2Db Fbd, DevMem2Db foreground, DevMem2Db background, int deltaC, int deltaCC, float alpha1, float alpha2, float alpha3, int N1c, int N1cc, int N2c, int N2cc, float T, cudaStream_t stream);
    template void updateBackgroundModel_gpu<uchar4, uchar4, uchar4>(DevMem2Db prevFrame, DevMem2Db curFrame, DevMem2Db Ftd, DevMem2Db Fbd, DevMem2Db foreground, DevMem2Db background, int deltaC, int deltaCC, float alpha1, float alpha2, float alpha3, int N1c, int N1cc, int N2c, int N2cc, float T, cudaStream_t stream);
}
