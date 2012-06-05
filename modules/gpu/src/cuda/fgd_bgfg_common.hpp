#ifndef __FGD_BGFG_COMMON_HPP__
#define __FGD_BGFG_COMMON_HPP__

#include "opencv2/core/devmem2d.hpp"

namespace bgfg
{
    struct BGPixelStat
    {
    public:
#ifdef __CUDACC__
        __device__ float& Pbc(int i, int j);
        __device__ float& Pbcc(int i, int j);

        __device__ unsigned char& is_trained_st_model(int i, int j);
        __device__ unsigned char& is_trained_dyn_model(int i, int j);

        __device__ float& PV_C(int i, int j, int k);
        __device__ float& PVB_C(int i, int j, int k);
        template <typename T> __device__ T& V_C(int i, int j, int k);

        __device__ float& PV_CC(int i, int j, int k);
        __device__ float& PVB_CC(int i, int j, int k);
        template <typename T> __device__ T& V1_CC(int i, int j, int k);
        template <typename T> __device__ T& V2_CC(int i, int j, int k);
#endif

        int rows_;

        unsigned char* Pbc_data_;
        size_t Pbc_step_;

        unsigned char* Pbcc_data_;
        size_t Pbcc_step_;

        unsigned char* is_trained_st_model_data_;
        size_t is_trained_st_model_step_;

        unsigned char* is_trained_dyn_model_data_;
        size_t is_trained_dyn_model_step_;

        unsigned char* ctable_Pv_data_;
        size_t ctable_Pv_step_;

        unsigned char* ctable_Pvb_data_;
        size_t ctable_Pvb_step_;

        unsigned char* ctable_v_data_;
        size_t ctable_v_step_;

        unsigned char* cctable_Pv_data_;
        size_t cctable_Pv_step_;

        unsigned char* cctable_Pvb_data_;
        size_t cctable_Pvb_step_;

        unsigned char* cctable_v1_data_;
        size_t cctable_v1_step_;

        unsigned char* cctable_v2_data_;
        size_t cctable_v2_step_;
    };

#ifdef __CUDACC__
    __device__ __forceinline__ float& BGPixelStat::Pbc(int i, int j)
    {
        return *((float*)(Pbc_data_ + i * Pbc_step_) + j);
    }

    __device__ __forceinline__ float& BGPixelStat::Pbcc(int i, int j)
    {
        return *((float*)(Pbcc_data_ + i * Pbcc_step_) + j);
    }

    __device__ __forceinline__ unsigned char& BGPixelStat::is_trained_st_model(int i, int j)
    {
        return *((unsigned char*)(is_trained_st_model_data_ + i * is_trained_st_model_step_) + j);
    }

    __device__ __forceinline__ unsigned char& BGPixelStat::is_trained_dyn_model(int i, int j)
    {
        return *((unsigned char*)(is_trained_dyn_model_data_ + i * is_trained_dyn_model_step_) + j);
    }

    __device__ __forceinline__ float& BGPixelStat::PV_C(int i, int j, int k)
    {
        return *((float*)(ctable_Pv_data_ + ((k * rows_) + i) * ctable_Pv_step_) + j);
    }

    __device__ __forceinline__ float& BGPixelStat::PVB_C(int i, int j, int k)
    {
        return *((float*)(ctable_Pvb_data_ + ((k * rows_) + i) * ctable_Pvb_step_) + j);
    }

    template <typename T> __device__ __forceinline__ T& BGPixelStat::V_C(int i, int j, int k)
    {
        return *((T*)(ctable_v_data_ + ((k * rows_) + i) * ctable_v_step_) + j);
    }

    __device__ __forceinline__ float& BGPixelStat::PV_CC(int i, int j, int k)
    {
        return *((float*)(cctable_Pv_data_ + ((k * rows_) + i) * cctable_Pv_step_) + j);
    }

    __device__ __forceinline__ float& BGPixelStat::PVB_CC(int i, int j, int k)
    {
        return *((float*)(cctable_Pvb_data_ + ((k * rows_) + i) * cctable_Pvb_step_) + j);
    }

    template <typename T> __device__ __forceinline__ T& BGPixelStat::V1_CC(int i, int j, int k)
    {
        return *((T*)(cctable_v1_data_ + ((k * rows_) + i) * cctable_v1_step_) + j);
    }

    template <typename T> __device__ __forceinline__ T& BGPixelStat::V2_CC(int i, int j, int k)
    {
        return *((T*)(cctable_v2_data_ + ((k * rows_) + i) * cctable_v2_step_) + j);
    }
#endif

    const int PARTIAL_HISTOGRAM_COUNT = 240;
    const int HISTOGRAM_BIN_COUNT = 256;

    template <typename PT, typename CT>
    void calcDiffHistogram_gpu(cv::gpu::DevMem2Db prevFrame, cv::gpu::DevMem2Db curFrame,
                               unsigned int* hist0, unsigned int* hist1, unsigned int* hist2,
                               unsigned int* partialBuf0, unsigned int* partialBuf1, unsigned int* partialBuf2,
                               int cc, cudaStream_t stream);

    template <typename PT, typename CT>
    void calcDiffThreshMask_gpu(cv::gpu::DevMem2Db prevFrame, cv::gpu::DevMem2Db curFrame, uchar3 bestThres, cv::gpu::DevMem2Db changeMask, cudaStream_t stream);

    void setBGPixelStat(const BGPixelStat& stat);

    template <typename PT, typename CT, typename OT>
    void bgfgClassification_gpu(cv::gpu::DevMem2Db prevFrame, cv::gpu::DevMem2Db curFrame,
                                cv::gpu::DevMem2Db Ftd, cv::gpu::DevMem2Db Fbd, cv::gpu::DevMem2Db foreground,
                                int deltaC, int deltaCC, float alpha2, int N1c, int N1cc, cudaStream_t stream);

    template <typename PT, typename CT, typename OT>
    void updateBackgroundModel_gpu(cv::gpu::DevMem2Db prevFrame, cv::gpu::DevMem2Db curFrame,
                                   cv::gpu::DevMem2Db Ftd, cv::gpu::DevMem2Db Fbd, cv::gpu::DevMem2Db foreground, cv::gpu::DevMem2Db background,
                                   int deltaC, int deltaCC, float alpha1, float alpha2, float alpha3, int N1c, int N1cc, int N2c, int N2cc, float T,
                                   cudaStream_t stream);
}

#endif // __FGD_BGFG_COMMON_HPP__
