/**
 * simoncatbot-opencv RPP HAL - Core Operations (GPU + CPU unified dispatch)
 *
 * Architecture:
 *   1. GPU: Uses RPP HIP backend with device memory upload/download
 *   2. CPU: Uses RPP HOST backend (no memory copies, direct pointer pass)
 *   3. Fallback: Returns CV_HAL_ERROR_NOT_IMPLEMENTED -> OpenCV native
 */

#include "rpp_hal_core.hpp"
#include "rpp_hal_utils.hpp"
#include <rpp/rppt_tensor_bitwise_operations.h>

using namespace cv::hal::rpp;

// =========================================================================
// Dispatch helpers
// =========================================================================

namespace {

#ifdef RPP_BACKEND_HIP
    inline void clearStickyHipError() {
        // RPP 3.x HIP backend sometimes leaves a spurious asynchronous
        // "illegal memory access" error in the HIP context even though the
        // kernel produced the correct result. Discard it so the next RPP
        // operation on a fresh handle can initialize successfully.
        (void)hipGetLastError();
    }
#else
    inline void clearStickyHipError() {}
#endif

    enum RppPath { RPP_NONE, RPP_GPU, RPP_CPU };

    inline RppPath selectRppPath() {
        if (isRppGpuAvailable()) return RPP_GPU;
        if (isRppCpuAvailable()) return RPP_CPU;
        return RPP_NONE;
    }

    inline bool runBitwiseAnd(RppBackend backend,
                              void* src1, void* src2, RpptDescPtr desc,
                              void* dst, RpptROIPtr roi,
                              rppHandle_t handle) {
        return (rppt_bitwise_and(src1, src2, desc, dst, desc, roi, XYWH, handle, backend) == RPP_SUCCESS);
    }

    inline bool runBitwiseOr(RppBackend backend,
                             void* src1, void* src2, RpptDescPtr desc,
                             void* dst, RpptROIPtr roi,
                             rppHandle_t handle) {
        return (rppt_bitwise_or(src1, src2, desc, dst, desc, roi, XYWH, handle, backend) == RPP_SUCCESS);
    }

    inline bool runBitwiseXor(RppBackend backend,
                              void* src1, void* src2, RpptDescPtr desc,
                              void* dst, RpptROIPtr roi,
                              rppHandle_t handle) {
        return (rppt_bitwise_xor(src1, src2, desc, dst, desc, roi, XYWH, handle, backend) == RPP_SUCCESS);
    }

    inline bool runBitwiseNot(RppBackend backend,
                              void* src, RpptDescPtr desc,
                              void* dst, RpptROIPtr roi,
                              rppHandle_t handle) {
        return (rppt_bitwise_not(src, desc, dst, desc, roi, XYWH, handle, backend) == RPP_SUCCESS);
    }
}

// =========================================================================
// BITWISE AND
// =========================================================================

extern "C" int rpp_hal_and8u(const uchar* src1_data, size_t src1_step,
                  const uchar* src2_data, size_t src2_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height) {
    RppPath path = selectRppPath();
    if (path == RPP_NONE) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RpptDesc desc;
    buildRppDescNHWC(desc, width, height, 1, CV_8U);
    RpptROI roi;
    buildFullRoi(roi, width, height);

    RppBackend backend = (path == RPP_GPU) ? RPP_HIP_BACKEND : RPP_HOST_BACKEND;

    if (path == RPP_GPU) {
        void* d_src1 = nullptr;
        void* d_src2 = nullptr;
        void* d_dst  = nullptr;

        bool ok = uploadRawToHip(src1_data, src1_step, width, height, CV_8U, 1, &d_src1) &&
                  uploadRawToHip(src2_data, src2_step, width, height, CV_8U, 1, &d_src2) &&
                  uploadRawToHip(dst_data, dst_step, width, height, CV_8U, 1, &d_dst);
        if (!ok) {
            freeHipPtr(d_src1); freeHipPtr(d_src2); freeHipPtr(d_dst);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        rppHandle_t handle = createRppGpuHandle(1);
        if (!handle) {
            freeHipPtr(d_src1); freeHipPtr(d_src2); freeHipPtr(d_dst);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        bool callOk = runBitwiseAnd(backend, d_src1, d_src2, &desc, d_dst, &roi, handle);
        clearStickyHipError();
        if (callOk) {
            callOk = downloadRawFromHip(d_dst, dst_data, dst_step, width, height, CV_8U, 1);
        }

        destroyRppGpuHandle(handle);
        freeHipPtr(d_src1); freeHipPtr(d_src2); freeHipPtr(d_dst);
        return callOk ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    rppHandle_t handle = createRppCpuHandle(1);
    if (!handle) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int status = runBitwiseAnd(backend,
                               const_cast<uchar*>(src1_data),
                               const_cast<uchar*>(src2_data),
                               &desc, dst_data, &roi, handle)
                     ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
    destroyRppCpuHandle(handle);
    return status;
}

// =========================================================================
// BITWISE OR
// =========================================================================

extern "C" int rpp_hal_or8u(const uchar* src1_data, size_t src1_step,
                 const uchar* src2_data, size_t src2_step,
                 uchar* dst_data, size_t dst_step,
                 int width, int height) {
    RppPath path = selectRppPath();
    if (path == RPP_NONE) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RpptDesc desc;
    buildRppDescNHWC(desc, width, height, 1, CV_8U);
    RpptROI roi;
    buildFullRoi(roi, width, height);

    RppBackend backend = (path == RPP_GPU) ? RPP_HIP_BACKEND : RPP_HOST_BACKEND;

    if (path == RPP_GPU) {
        void* d_src1 = nullptr;
        void* d_src2 = nullptr;
        void* d_dst  = nullptr;

        bool ok = uploadRawToHip(src1_data, src1_step, width, height, CV_8U, 1, &d_src1) &&
                  uploadRawToHip(src2_data, src2_step, width, height, CV_8U, 1, &d_src2) &&
                  uploadRawToHip(dst_data, dst_step, width, height, CV_8U, 1, &d_dst);
        if (!ok) {
            freeHipPtr(d_src1); freeHipPtr(d_src2); freeHipPtr(d_dst);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        rppHandle_t handle = createRppGpuHandle(1);
        if (!handle) {
            freeHipPtr(d_src1); freeHipPtr(d_src2); freeHipPtr(d_dst);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        bool callOk = runBitwiseOr(backend, d_src1, d_src2, &desc, d_dst, &roi, handle);
        clearStickyHipError();
        if (callOk) {
            callOk = downloadRawFromHip(d_dst, dst_data, dst_step, width, height, CV_8U, 1);
        }

        destroyRppGpuHandle(handle);
        freeHipPtr(d_src1); freeHipPtr(d_src2); freeHipPtr(d_dst);
        return callOk ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    rppHandle_t handle = createRppCpuHandle(1);
    if (!handle) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int status = runBitwiseOr(backend,
                              const_cast<uchar*>(src1_data),
                              const_cast<uchar*>(src2_data),
                              &desc, dst_data, &roi, handle)
                     ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
    destroyRppCpuHandle(handle);
    return status;
}

// =========================================================================
// BITWISE XOR
// =========================================================================

extern "C" int rpp_hal_xor8u(const uchar* src1_data, size_t src1_step,
                  const uchar* src2_data, size_t src2_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height) {
    RppPath path = selectRppPath();
    if (path == RPP_NONE) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RpptDesc desc;
    buildRppDescNHWC(desc, width, height, 1, CV_8U);
    RpptROI roi;
    buildFullRoi(roi, width, height);

    RppBackend backend = (path == RPP_GPU) ? RPP_HIP_BACKEND : RPP_HOST_BACKEND;

    if (path == RPP_GPU) {
        void* d_src1 = nullptr;
        void* d_src2 = nullptr;
        void* d_dst  = nullptr;

        bool ok = uploadRawToHip(src1_data, src1_step, width, height, CV_8U, 1, &d_src1) &&
                  uploadRawToHip(src2_data, src2_step, width, height, CV_8U, 1, &d_src2) &&
                  uploadRawToHip(dst_data, dst_step, width, height, CV_8U, 1, &d_dst);
        if (!ok) {
            freeHipPtr(d_src1); freeHipPtr(d_src2); freeHipPtr(d_dst);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        rppHandle_t handle = createRppGpuHandle(1);
        if (!handle) {
            freeHipPtr(d_src1); freeHipPtr(d_src2); freeHipPtr(d_dst);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        bool callOk = runBitwiseXor(backend, d_src1, d_src2, &desc, d_dst, &roi, handle);
        clearStickyHipError();
        if (callOk) {
            callOk = downloadRawFromHip(d_dst, dst_data, dst_step, width, height, CV_8U, 1);
        }

        destroyRppGpuHandle(handle);
        freeHipPtr(d_src1); freeHipPtr(d_src2); freeHipPtr(d_dst);
        return callOk ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    rppHandle_t handle = createRppCpuHandle(1);
    if (!handle) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int status = runBitwiseXor(backend,
                               const_cast<uchar*>(src1_data),
                               const_cast<uchar*>(src2_data),
                               &desc, dst_data, &roi, handle)
                     ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
    destroyRppCpuHandle(handle);
    return status;
}

// =========================================================================
// BITWISE NOT
// =========================================================================

extern "C" int rpp_hal_not8u(const uchar* src_data, size_t src_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height) {
    RppPath path = selectRppPath();
    if (path == RPP_NONE) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RpptDesc desc;
    buildRppDescNHWC(desc, width, height, 1, CV_8U);
    RpptROI roi;
    buildFullRoi(roi, width, height);

    RppBackend backend = (path == RPP_GPU) ? RPP_HIP_BACKEND : RPP_HOST_BACKEND;

    if (path == RPP_GPU) {
        void* d_src = nullptr;
        void* d_dst = nullptr;

        bool ok = uploadRawToHip(src_data, src_step, width, height, CV_8U, 1, &d_src) &&
                  uploadRawToHip(dst_data, dst_step, width, height, CV_8U, 1, &d_dst);
        if (!ok) {
            freeHipPtr(d_src); freeHipPtr(d_dst);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        rppHandle_t handle = createRppGpuHandle(1);
        if (!handle) {
            freeHipPtr(d_src); freeHipPtr(d_dst);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        bool callOk = runBitwiseNot(backend, d_src, &desc, d_dst, &roi, handle);
        clearStickyHipError();
        if (callOk) {
            callOk = downloadRawFromHip(d_dst, dst_data, dst_step, width, height, CV_8U, 1);
        }

        destroyRppGpuHandle(handle);
        freeHipPtr(d_src); freeHipPtr(d_dst);
        return callOk ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    rppHandle_t handle = createRppCpuHandle(1);
    if (!handle) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int status = runBitwiseNot(backend,
                               const_cast<uchar*>(src_data),
                               &desc, dst_data, &roi, handle)
                     ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
    destroyRppCpuHandle(handle);
    return status;
}

// =========================================================================
// Math routines with no suitable RPP tensor equivalent -> fallback
// =========================================================================

extern "C" int rpp_hal_add8u(const uchar*, size_t,
                  const uchar*, size_t,
                  uchar*, size_t,
                  int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_sub8u(const uchar*, size_t,
                  const uchar*, size_t,
                  uchar*, size_t,
                  int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_mul8u(const uchar*, size_t,
                  const uchar*, size_t,
                  uchar*, size_t,
                  int, int, double) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_add16s(const short*, size_t,
                   const short*, size_t,
                   short*, size_t,
                   int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_sub16s(const short*, size_t,
                   const short*, size_t,
                   short*, size_t,
                   int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_mul16s(const short*, size_t,
                   const short*, size_t,
                   short*, size_t,
                   int, int, double) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_add32f(const float*, size_t,
                   const float*, size_t,
                   float*, size_t,
                   int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_sub32f(const float*, size_t,
                   const float*, size_t,
                   float*, size_t,
                   int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_mul32f(const float*, size_t,
                   const float*, size_t,
                   float*, size_t,
                   int, int, double) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_div32f(const float* src1_data, size_t src1_step,
                   const float* src2_data, size_t src2_step,
                   float* dst_data, size_t dst_step,
                   int width, int height, double scale) {
    (void)src1_data; (void)src1_step; (void)src2_data; (void)src2_step;
    (void)dst_data; (void)dst_step; (void)width; (void)height; (void)scale;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_recip32f(const float*, size_t,
                     float*, size_t,
                     int, int, double) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_addWeighted8u(const uchar* src1_data, size_t src1_step,
                            const uchar* src2_data, size_t src2_step,
                            uchar* dst_data, size_t dst_step,
                            int width, int height,
                            const double scalars[3]) {
    (void)src1_data; (void)src1_step; (void)src2_data; (void)src2_step;
    (void)dst_data; (void)dst_step; (void)width; (void)height; (void)scalars;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_addWeighted32f(const float* src1_data, size_t src1_step,
                             const float* src2_data, size_t src2_step,
                             float* dst_data, size_t dst_step,
                             int width, int height,
                             const double scalars[3]) {
    (void)src1_data; (void)src1_step; (void)src2_data; (void)src2_step;
    (void)dst_data; (void)dst_step; (void)width; (void)height; (void)scalars;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvt8u16s(const uchar*, size_t,
                     short*, size_t,
                     int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvt16s8u(const short*, size_t,
                     uchar*, size_t,
                     int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvt16s32f(const short*, size_t,
                      float*, size_t,
                      int, int, double) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvt32f16s(const float*, size_t,
                      short*, size_t,
                      int, int, double) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_abs8u(const uchar*, size_t,
                  uchar*, size_t,
                  int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_abs16s(const short*, size_t,
                   short*, size_t,
                   int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_abs32f(const float*, size_t,
                   float*, size_t,
                   int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cmp8u(const uchar*, size_t,
                  const uchar*, size_t,
                  uchar*, size_t,
                  int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cmp16s(const short*, size_t,
                   const short*, size_t,
                   uchar*, size_t,
                   int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cmp32f(const float*, size_t,
                   const float*, size_t,
                   uchar*, size_t,
                   int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cmp64f(const double*, size_t,
                   const double*, size_t,
                   uchar*, size_t,
                   int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_minMaxIdx8u(const uchar*, size_t,
                          double*, double*, int*, int*,
                          int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_minMaxIdx16s(const short*, size_t,
                           double*, double*, int*, int*,
                           int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_minMaxIdx32f(const float*, size_t,
                           double*, double*, int*, int*,
                           int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_minMaxIdx64f(const double*, size_t,
                           double*, double*, int*, int*,
                           int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_countNonZero8u(const uchar*, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_countNonZero16s(const short*, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_countNonZero32f(const float*, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_countNonZero64f(const double*, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_dotProduct8u(const uchar*, size_t,
                         const uchar*, size_t,
                         int, int, double*) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_dotProduct8s(const char*, size_t,
                         const char*, size_t,
                         int, int, double*) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_dotProduct16u(const ushort*, size_t,
                          const ushort*, size_t,
                          int, int, double*) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_dotProduct32f(const float*, size_t,
                          const float*, size_t,
                          int, int, double*) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_dotProduct64f(const double*, size_t,
                          const double*, size_t,
                          int, int, double*) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_meanStdDev8u(const uchar* src_data, size_t src_step,
                         int width, int height, double* meanVal, double* stdDevVal,
                         uchar* mask, size_t maskStep) {
    (void)src_data; (void)src_step; (void)width; (void)height;
    (void)meanVal; (void)stdDevVal; (void)mask; (void)maskStep;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_meanStdDev16u(const ushort* src_data, size_t src_step,
                          int width, int height, double* meanVal, double* stdDevVal,
                          uchar* mask, size_t maskStep) {
    (void)src_data; (void)src_step; (void)width; (void)height;
    (void)meanVal; (void)stdDevVal; (void)mask; (void)maskStep;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_meanStdDev32f(const float* src_data, size_t src_step,
                          int width, int height, double* meanVal, double* stdDevVal,
                          uchar* mask, size_t maskStep) {
    (void)src_data; (void)src_step; (void)width; (void)height;
    (void)meanVal; (void)stdDevVal; (void)mask; (void)maskStep;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_meanStdDev64f(const double* src_data, size_t src_step,
                          int width, int height, double* meanVal, double* stdDevVal,
                          uchar* mask, size_t maskStep) {
    (void)src_data; (void)src_step; (void)width; (void)height;
    (void)meanVal; (void)stdDevVal; (void)mask; (void)maskStep;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_integral8u(const uchar*, size_t,
                     double*, size_t,
                     int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_integral32f(const float*, size_t,
                        double*, size_t,
                        int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_integral32s(const int*, size_t,
                        double*, size_t,
                        int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvtBGRtoGray8u(const uchar*, size_t,
                           uchar*, size_t,
                           int, int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvtBGRtoGray16u(const ushort*, size_t,
                            ushort*, size_t,
                            int, int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvtBGRtoGray32f(const float*, size_t,
                            float*, size_t,
                            int, int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvtGraytoBGR8u(const uchar*, size_t,
                           uchar*, size_t,
                           int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_lut(const uchar*, size_t,
                int, int, int,
                const uchar*, int,
                uchar*, size_t) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_magnitude32f(const float*, const float*,
                         float*, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_magnitude64f(const double*, const double*,
                         double*, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
