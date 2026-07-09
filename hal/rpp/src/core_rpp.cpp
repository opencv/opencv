/**
 * simoncatbot-opencv RPP HAL - Core Operations (GPU + CPU unified dispatch)
 *
 * Architecture:
 *   1. GPU: Uses RPP HIP backend with device memory upload/download
 *   2. CPU: Uses RPP HOST backend (no memory copies, direct pointer pass)
 *   3. Fallback: Returns CV_HAL_ERROR_NOT_IMPLEMENTED → OpenCV native
 */

#include "rpp_hal_core.hpp"
#include "rpp_hal_utils.hpp"
#include <rpp/rppt_tensor_bitwise_operations.h>

using namespace cv::hal::rpp;

// =========================================================================
// Dispatch macros
// =========================================================================

// Unified dispatch: GPU → CPU → NOT_IMPLEMENTED
#define RPP_DISPATCH_START \
    bool useGpu = isRppGpuAvailable(); \
    bool useCpu = !useGpu && isRppCpuAvailable(); \
    if (!useGpu && !useCpu) return CV_HAL_ERROR_NOT_IMPLEMENTED;

// =========================================================================
// BITWISE AND
// =========================================================================

extern "C" int rpp_hal_and8u(const uchar* src1_data, size_t src1_step,
                  const uchar* src2_data, size_t src2_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height) {
    RPP_DISPATCH_START;

    RpptDesc desc;
    buildRppDescNHWC(desc, width, height, 1, CV_8U);
    RpptROI roi;
    buildFullRoi(roi, width, height);

    if (useGpu) {
        void* d_src1 = nullptr;
        void* d_src2 = nullptr;
        void* d_dst  = nullptr;

        if (!uploadRawToHip(src1_data, src1_step, width, height, CV_8U, 1, &d_src1) ||
            !uploadRawToHip(src2_data, src2_step, width, height, CV_8U, 1, &d_src2) ||
            !uploadRawToHip(src1_data, src1_step, width, height, CV_8U, 1, &d_dst)) {
            freeHipPtr(d_src1); freeHipPtr(d_src2); freeHipPtr(d_dst);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        rppHandle_t handle = createRppGpuHandle(1);
        if (!handle) { freeHipPtr(d_src1); freeHipPtr(d_src2); freeHipPtr(d_dst); return CV_HAL_ERROR_NOT_IMPLEMENTED; }

        RppStatus status = rppt_bitwise_and_gpu(
            d_src1, d_src2, &desc, d_dst, &desc,
            &roi, XYWH, handle);

        bool ok = (status == RPP_SUCCESS);
        if (ok) {
            downloadRawFromHip(d_dst, dst_data, dst_step, width, height, CV_8U, 1);
        }

        freeHipPtr(d_src1); freeHipPtr(d_src2); freeHipPtr(d_dst);
        destroyRppGpuHandle(handle);
        return ok ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    // CPU path: direct pointer pass, no copies
    rppHandle_t handle = createRppCpuHandle(1);
    if (!handle) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RppStatus status = rppt_bitwise_and_host(
        const_cast<uchar*>(src1_data), const_cast<uchar*>(src2_data), &desc,
        dst_data, &desc, &roi, XYWH, handle);

    destroyRppCpuHandle(handle);
    return (status == RPP_SUCCESS) ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// =========================================================================
// BITWISE OR
// =========================================================================

extern "C" int rpp_hal_or8u(const uchar* src1_data, size_t src1_step,
                 const uchar* src2_data, size_t src2_step,
                 uchar* dst_data, size_t dst_step,
                 int width, int height) {
    RPP_DISPATCH_START;

    RpptDesc desc;
    buildRppDescNHWC(desc, width, height, 1, CV_8U);
    RpptROI roi;
    buildFullRoi(roi, width, height);

    if (useGpu) {
        void* d_src1 = nullptr;
        void* d_src2 = nullptr;
        void* d_dst  = nullptr;

        if (!uploadRawToHip(src1_data, src1_step, width, height, CV_8U, 1, &d_src1) ||
            !uploadRawToHip(src2_data, src2_step, width, height, CV_8U, 1, &d_src2) ||
            !uploadRawToHip(src1_data, src1_step, width, height, CV_8U, 1, &d_dst)) {
            freeHipPtr(d_src1); freeHipPtr(d_src2); freeHipPtr(d_dst);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        rppHandle_t handle = createRppGpuHandle(1);
        if (!handle) { freeHipPtr(d_src1); freeHipPtr(d_src2); freeHipPtr(d_dst); return CV_HAL_ERROR_NOT_IMPLEMENTED; }

        RppStatus status = rppt_bitwise_or_gpu(
            d_src1, d_src2, &desc, d_dst, &desc,
            &roi, XYWH, handle);

        bool ok = (status == RPP_SUCCESS);
        if (ok) downloadRawFromHip(d_dst, dst_data, dst_step, width, height, CV_8U, 1);

        freeHipPtr(d_src1); freeHipPtr(d_src2); freeHipPtr(d_dst);
        destroyRppGpuHandle(handle);
        return ok ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    rppHandle_t handle = createRppCpuHandle(1);
    if (!handle) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RppStatus status = rppt_bitwise_or_host(
        const_cast<uchar*>(src1_data), const_cast<uchar*>(src2_data), &desc,
        dst_data, &desc, &roi, XYWH, handle);

    destroyRppCpuHandle(handle);
    return (status == RPP_SUCCESS) ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// =========================================================================
// BITWISE XOR
// =========================================================================

extern "C" int rpp_hal_xor8u(const uchar* src1_data, size_t src1_step,
                  const uchar* src2_data, size_t src2_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height) {
    RPP_DISPATCH_START;

    RpptDesc desc;
    buildRppDescNHWC(desc, width, height, 1, CV_8U);
    RpptROI roi;
    buildFullRoi(roi, width, height);

    if (useGpu) {
        void* d_src1 = nullptr;
        void* d_src2 = nullptr;
        void* d_dst  = nullptr;

        if (!uploadRawToHip(src1_data, src1_step, width, height, CV_8U, 1, &d_src1) ||
            !uploadRawToHip(src2_data, src2_step, width, height, CV_8U, 1, &d_src2) ||
            !uploadRawToHip(src1_data, src1_step, width, height, CV_8U, 1, &d_dst)) {
            freeHipPtr(d_src1); freeHipPtr(d_src2); freeHipPtr(d_dst);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        rppHandle_t handle = createRppGpuHandle(1);
        if (!handle) { freeHipPtr(d_src1); freeHipPtr(d_src2); freeHipPtr(d_dst); return CV_HAL_ERROR_NOT_IMPLEMENTED; }

        RppStatus status = rppt_bitwise_xor_gpu(
            d_src1, d_src2, &desc, d_dst, &desc,
            &roi, XYWH, handle);

        bool ok = (status == RPP_SUCCESS);
        if (ok) downloadRawFromHip(d_dst, dst_data, dst_step, width, height, CV_8U, 1);

        freeHipPtr(d_src1); freeHipPtr(d_src2); freeHipPtr(d_dst);
        destroyRppGpuHandle(handle);
        return ok ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    rppHandle_t handle = createRppCpuHandle(1);
    if (!handle) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RppStatus status = rppt_bitwise_xor_host(
        const_cast<uchar*>(src1_data), const_cast<uchar*>(src2_data), &desc,
        dst_data, &desc, &roi, XYWH, handle);

    destroyRppCpuHandle(handle);
    return (status == RPP_SUCCESS) ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// =========================================================================
// BITWISE NOT
// =========================================================================

extern "C" int rpp_hal_not8u(const uchar* src_data, size_t src_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height) {
    RPP_DISPATCH_START;

    RpptDesc desc;
    buildRppDescNHWC(desc, width, height, 1, CV_8U);
    RpptROI roi;
    buildFullRoi(roi, width, height);

    if (useGpu) {
        void* d_src = nullptr;
        void* d_dst = nullptr;

        if (!uploadRawToHip(src_data, src_step, width, height, CV_8U, 1, &d_src) ||
            !uploadRawToHip(src_data, src_step, width, height, CV_8U, 1, &d_dst)) {
            freeHipPtr(d_src); freeHipPtr(d_dst);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        rppHandle_t handle = createRppGpuHandle(1);
        if (!handle) { freeHipPtr(d_src); freeHipPtr(d_dst); return CV_HAL_ERROR_NOT_IMPLEMENTED; }

        RppStatus status = rppt_bitwise_not_gpu(
            d_src, &desc, d_dst, &desc,
            &roi, XYWH, handle);

        bool ok = (status == RPP_SUCCESS);
        if (ok) downloadRawFromHip(d_dst, dst_data, dst_step, width, height, CV_8U, 1);

        freeHipPtr(d_src); freeHipPtr(d_dst);
        destroyRppGpuHandle(handle);
        return ok ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    rppHandle_t handle = createRppCpuHandle(1);
    if (!handle) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RppStatus status = rppt_bitwise_not_host(
        const_cast<uchar*>(src_data), &desc,
        dst_data, &desc, &roi, XYWH, handle);

    destroyRppCpuHandle(handle);
    return (status == RPP_SUCCESS) ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// =========================================================================
// ADD (No RPP equivalent — NOT_IMPLEMENTED)
// =========================================================================

extern "C" int rpp_hal_add8u(const uchar* src1_data, size_t src1_step,
                  const uchar* src2_data, size_t src2_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height) {
    (void)src1_data; (void)src1_step; (void)src2_data; (void)src2_step;
    (void)dst_data; (void)dst_step; (void)width; (void)height;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_add32f(const float* src1_data, size_t src1_step,
                   const float* src2_data, size_t src2_step,
                   float* dst_data, size_t dst_step,
                   int width, int height) {
    (void)src1_data; (void)src1_step; (void)src2_data; (void)src2_step;
    (void)dst_data; (void)dst_step; (void)width; (void)height;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// =========================================================================
// MULTIPLY (No RPP equivalent — NOT_IMPLEMENTED)
// =========================================================================

extern "C" int rpp_hal_mul8u(const uchar* src1_data, size_t src1_step,
                  const uchar* src2_data, size_t src2_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height, double scale) {
    (void)src1_data; (void)src1_step; (void)src2_data; (void)src2_step;
    (void)dst_data; (void)dst_step; (void)width; (void)height; (void)scale;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_mul32f(const float* src1_data, size_t src1_step,
                   const float* src2_data, size_t src2_step,
                   float* dst_data, size_t dst_step,
                   int width, int height, double scale) {
    (void)src1_data; (void)src1_step; (void)src2_data; (void)src2_step;
    (void)dst_data; (void)dst_step; (void)width; (void)height; (void)scale;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// =========================================================================
// LUT (No RPP equivalent — NOT_IMPLEMENTED)
// =========================================================================

extern "C" int rpp_hal_lut(const uchar* src_data, size_t src_step,
                int width, int height, int cn,
                const uchar* lut_data, int lut_cn,
                uchar* dst_data, size_t dst_step) {
    (void)src_data; (void)src_step; (void)lut_data; (void)lut_cn;
    (void)dst_data; (void)dst_step; (void)width; (void)height; (void)cn;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// =========================================================================
// MAGNITUDE (No RPP equivalent — NOT_IMPLEMENTED)
// =========================================================================

extern "C" int rpp_hal_magnitude32f(const float* x_data, const float* y_data,
                         float* dst_data, int len) {
    (void)x_data; (void)y_data; (void)dst_data; (void)len;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_magnitude64f(const double* x_data, const double* y_data,
                         double* dst_data, int len) {
    (void)x_data; (void)y_data; (void)dst_data; (void)len;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
