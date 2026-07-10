/**
 * simoncatbot-opencv RPP HAL - Imgproc Operations (Unified dispatch)
 *
 * Architecture:
 *   1. GPU: Uses RPP HIP backend with device memory upload/download
 *   2. CPU: Uses RPP HOST backend (no memory copies, direct pointer pass)
 *   3. Fallback: Returns CV_HAL_ERROR_NOT_IMPLEMENTED -> OpenCV native
 */

#include "rpp_hal_imgproc.hpp"
#include "rpp_hal_utils.hpp"
#include <rpp/rppt_tensor_geometric_augmentations.h>
#include <rpp/rppt_tensor_filter_augmentations.h>

using namespace cv::hal::rpp;

namespace {

    enum RppPath { RPP_NONE, RPP_GPU, RPP_CPU };

    inline RppPath selectRppPath() {
        const char* disable = getenv("OPENCV_RPP_DISABLE");
        if (disable && (strcmp(disable, "1") == 0 || strcmp(disable, "yes") == 0 || strcmp(disable, "true") == 0)) {
            return RPP_NONE;
        }
        if (isRppGpuAvailable()) return RPP_GPU;
        if (isRppCpuAvailable()) return RPP_CPU;
        return RPP_NONE;
    }

    inline bool checkHip(hipError_t err) { return err == hipSuccess; }

    inline RpptInterpolationType cvInterpolationToRpp(int interpolation) {
        switch (interpolation) {
            case 0: return NEAREST_NEIGHBOR; // INTER_NEAREST
            case 1: return BILINEAR;         // INTER_LINEAR
            case 2: return BICUBIC;          // INTER_CUBIC
            case 3: return LANCZOS;            // INTER_LANCZOS4 (closest)
            default: return BILINEAR;
        }
    }

    inline int cvDepthToChannels(int src_type) {
        return CV_MAT_CN(src_type);
    }

    inline bool supportedDepth(int depth) {
        return depth == CV_8U || depth == CV_32F || depth == CV_8S;
    }

    inline void buildDesc(RpptDesc& desc, int w, int h, int c, int depth) {
        buildRppDescNHWC(desc, w, h, c, depth);
    }

    inline void buildRoi(RpptROI& roi, int w, int h) {
        buildFullRoi(roi, w, h);
    }

}

// =========================================================================
// FLIP
// =========================================================================

extern "C" int rpp_hal_flip(int src_type,
                            const uchar* src_data, size_t src_step,
                            int src_width, int src_height,
                            uchar* dst_data, size_t dst_step,
                            int flip_mode) {
    RppPath path = selectRppPath();
    if (path == RPP_NONE) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int cn = cvDepthToChannels(src_type);
    int depth = CV_MAT_DEPTH(src_type);
    if (!supportedDepth(depth)) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RpptDesc srcDesc;
    buildDesc(srcDesc, src_width, src_height, cn, depth);
    RpptDesc dstDesc;
    buildDesc(dstDesc, src_width, src_height, cn, depth);
    RpptROI roi;
    buildRoi(roi, src_width, src_height);

    Rpp32u horizontal = (flip_mode == 1 || flip_mode == -1) ? 1 : 0;
    Rpp32u vertical   = (flip_mode == 0 || flip_mode == -1) ? 1 : 0;

    RppBackend backend = (path == RPP_GPU) ? RPP_HIP_BACKEND : RPP_HOST_BACKEND;

    if (path == RPP_GPU) {
        void* d_src = nullptr;
        void* d_dst = nullptr;
        if (!uploadRawToHip(src_data, src_step, src_width, src_height, depth, cn, &d_src) ||
            !uploadRawToHip(dst_data, dst_step, src_width, src_height, depth, cn, &d_dst)) {
            freeHipPtr(d_src); freeHipPtr(d_dst);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        rppHandle_t handle = createRppGpuHandle(1);
        if (!handle) {
            freeHipPtr(d_src); freeHipPtr(d_dst);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        RppStatus status = rppt_flip(d_src, &srcDesc, d_dst, &dstDesc,
                                     &horizontal, &vertical,
                                     &roi, XYWH, handle, backend);

        bool ok = (status == RPP_SUCCESS);
        if (ok) {
            ok = downloadRawFromHip(d_dst, dst_data, dst_step, src_width, src_height, depth, cn);
        }
        destroyRppGpuHandle(handle);
        freeHipPtr(d_src); freeHipPtr(d_dst);
        return ok ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    rppHandle_t handle = createRppCpuHandle(1);
    if (!handle) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RppStatus status = rppt_flip(const_cast<uchar*>(src_data), &srcDesc,
                                 dst_data, &dstDesc,
                                 &horizontal, &vertical,
                                 &roi, XYWH, handle, backend);
    destroyRppCpuHandle(handle);
    return (status == RPP_SUCCESS) ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// =========================================================================
// RESIZE
// =========================================================================

extern "C" int rpp_hal_resize(int src_type,
                                const uchar* src_data, size_t src_step,
                                int src_width, int src_height,
                                uchar* dst_data, size_t dst_step,
                                int dst_width, int dst_height,
                                double inv_scale_x, double inv_scale_y,
                                int interpolation) {
    (void)inv_scale_x; (void)inv_scale_y;
    RppPath path = selectRppPath();
    if (path == RPP_NONE) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int cn = cvDepthToChannels(src_type);
    int depth = CV_MAT_DEPTH(src_type);
    if (!supportedDepth(depth)) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RpptDesc srcDesc;
    buildDesc(srcDesc, src_width, src_height, cn, depth);
    RpptDesc dstDesc;
    buildDesc(dstDesc, dst_width, dst_height, cn, depth);
    RpptROI srcRoi;
    buildRoi(srcRoi, src_width, src_height);
    RpptImagePatch dstSize;
    dstSize.width = static_cast<Rpp32u>(dst_width);
    dstSize.height = static_cast<Rpp32u>(dst_height);

    RpptInterpolationType interp = cvInterpolationToRpp(interpolation);
    RppBackend backend = (path == RPP_GPU) ? RPP_HIP_BACKEND : RPP_HOST_BACKEND;

    if (path == RPP_GPU) {
        void* d_src = nullptr;
        void* d_dst = nullptr;
        if (!uploadRawToHip(src_data, src_step, src_width, src_height, depth, cn, &d_src) ||
            !uploadRawToHip(dst_data, dst_step, dst_width, dst_height, depth, cn, &d_dst)) {
            freeHipPtr(d_src); freeHipPtr(d_dst);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        rppHandle_t handle = createRppGpuHandle(1);
        if (!handle) {
            freeHipPtr(d_src); freeHipPtr(d_dst);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        RppStatus status = rppt_resize(d_src, &srcDesc, d_dst, &dstDesc,
                                       &dstSize, interp,
                                       &srcRoi, XYWH, handle, backend);

        bool ok = (status == RPP_SUCCESS);
        if (ok) {
            ok = downloadRawFromHip(d_dst, dst_data, dst_step, dst_width, dst_height, depth, cn);
        }
        destroyRppGpuHandle(handle);
        freeHipPtr(d_src); freeHipPtr(d_dst);
        return ok ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    rppHandle_t handle = createRppCpuHandle(1);
    if (!handle) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RppStatus status = rppt_resize(const_cast<uchar*>(src_data), &srcDesc,
                                   dst_data, &dstDesc,
                                   &dstSize, interp,
                                   &srcRoi, XYWH, handle, backend);
    destroyRppCpuHandle(handle);
    return (status == RPP_SUCCESS) ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// =========================================================================
// WARP AFFINE
// =========================================================================

extern "C" int rpp_hal_warpAffine(int src_type,
                                    const uchar* src_data, size_t src_step,
                                    int src_width, int src_height,
                                    uchar* dst_data, size_t dst_step,
                                    int dst_width, int dst_height,
                                    const double M[6], int interpolation,
                                    int borderType, const double borderValue[4]) {
    (void)borderType; (void)borderValue;
    RppPath path = selectRppPath();
    if (path == RPP_NONE) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int cn = cvDepthToChannels(src_type);
    int depth = CV_MAT_DEPTH(src_type);
    if (!supportedDepth(depth)) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // OpenCV HAL passes the forward affine matrix. RPP expects the same
    // 2x3 matrix layout [m0 m1 m2; m3 m4 m5].
    float affine[6];
    for (int i = 0; i < 6; ++i) affine[i] = static_cast<float>(M[i]);

    RpptDesc srcDesc;
    buildDesc(srcDesc, src_width, src_height, cn, depth);
    RpptDesc dstDesc;
    buildDesc(dstDesc, dst_width, dst_height, cn, depth);
    RpptROI srcRoi;
    buildRoi(srcRoi, src_width, src_height);

    RpptInterpolationType interp = cvInterpolationToRpp(interpolation);
    RppBackend backend = (path == RPP_GPU) ? RPP_HIP_BACKEND : RPP_HOST_BACKEND;

    if (path == RPP_GPU) {
        void* d_src = nullptr;
        void* d_dst = nullptr;
        void* d_affine = nullptr;
        if (!uploadRawToHip(src_data, src_step, src_width, src_height, depth, cn, &d_src) ||
            !uploadRawToHip(dst_data, dst_step, dst_width, dst_height, depth, cn, &d_dst) ||
            hipMalloc(&d_affine, sizeof(affine)) != hipSuccess) {
            freeHipPtr(d_src); freeHipPtr(d_dst); freeHipPtr(d_affine);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
        if (!checkHip(hipMemcpy(d_affine, affine, sizeof(affine), hipMemcpyHostToDevice))) {
            freeHipPtr(d_src); freeHipPtr(d_dst); freeHipPtr(d_affine);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        rppHandle_t handle = createRppGpuHandle(1);
        if (!handle) {
            freeHipPtr(d_src); freeHipPtr(d_dst); freeHipPtr(d_affine);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        RppStatus status = rppt_warp_affine(d_src, &srcDesc, d_dst, &dstDesc,
                                            static_cast<Rpp32f*>(d_affine),
                                            interp, &srcRoi, XYWH, handle, backend);

        bool ok = (status == RPP_SUCCESS);
        if (ok) {
            ok = downloadRawFromHip(d_dst, dst_data, dst_step, dst_width, dst_height, depth, cn);
        }
        destroyRppGpuHandle(handle);
        freeHipPtr(d_src); freeHipPtr(d_dst); freeHipPtr(d_affine);
        return ok ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    rppHandle_t handle = createRppCpuHandle(1);
    if (!handle) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RppStatus status = rppt_warp_affine(const_cast<uchar*>(src_data), &srcDesc,
                                        dst_data, &dstDesc,
                                        affine, interp,
                                        &srcRoi, XYWH, handle, backend);
    destroyRppCpuHandle(handle);
    return (status == RPP_SUCCESS) ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// =========================================================================
// FILTERING (stubs)
// =========================================================================

extern "C" int rpp_hal_boxFilter(const uchar* src_data, size_t src_step,
                                 uchar* dst_data, size_t dst_step,
                                 int width, int height, int src_depth, int dst_depth, int cn,
                                 int margin_left, int margin_top, int margin_right, int margin_bottom,
                                 size_t ksize_width, size_t ksize_height,
                                 int anchor_x, int anchor_y,
                                 bool normalize, int border_type) {
    (void)margin_left; (void)margin_top; (void)margin_right; (void)margin_bottom;
    (void)anchor_x; (void)anchor_y; (void)normalize;

    // RPP box_filter requires square kernel and only supports REPLICATE border.
    if (ksize_width != ksize_height) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (border_type != cv::BORDER_REPLICATE) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (src_depth != dst_depth) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (!supportedDepth(src_depth)) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (cn != 1 && cn != 3) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RppPath path = selectRppPath();
    if (path == RPP_NONE) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RpptDesc srcDesc;
    buildDesc(srcDesc, width, height, cn, src_depth);
    RpptDesc dstDesc;
    buildDesc(dstDesc, width, height, cn, dst_depth);
    RpptROI roi;
    buildRoi(roi, width, height);

    RppBackend backend = (path == RPP_GPU) ? RPP_HIP_BACKEND : RPP_HOST_BACKEND;
    Rpp32u kernelSize = static_cast<Rpp32u>(ksize_width);
    RpptImageBorderType border = REPLICATE;

    if (path == RPP_GPU) {
        void* d_src = nullptr;
        void* d_dst = nullptr;
        if (!uploadRawToHip(src_data, src_step, width, height, src_depth, cn, &d_src) ||
            !uploadRawToHip(dst_data, dst_step, width, height, dst_depth, cn, &d_dst)) {
            freeHipPtr(d_src); freeHipPtr(d_dst);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        rppHandle_t handle = createRppGpuHandle(1);
        if (!handle) {
            freeHipPtr(d_src); freeHipPtr(d_dst);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        RppStatus status = rppt_box_filter(d_src, &srcDesc, d_dst, &dstDesc,
                                           kernelSize, border, &roi, XYWH, handle, backend);

        bool ok = (status == RPP_SUCCESS);
        if (ok) {
            ok = downloadRawFromHip(d_dst, dst_data, dst_step, width, height, dst_depth, cn);
        }
        destroyRppGpuHandle(handle);
        freeHipPtr(d_src); freeHipPtr(d_dst);
        return ok ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    rppHandle_t handle = createRppCpuHandle(1);
    if (!handle) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RppStatus status = rppt_box_filter(const_cast<uchar*>(src_data), &srcDesc,
                                       dst_data, &dstDesc,
                                       kernelSize, border, &roi, XYWH, handle, backend);
    destroyRppCpuHandle(handle);
    return (status == RPP_SUCCESS) ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_gaussianBlur(const uchar* src_data, size_t src_step,
                                    uchar* dst_data, size_t dst_step,
                                    int width, int height, int depth, int cn,
                                    size_t margin_left, size_t margin_top, size_t margin_right, size_t margin_bottom,
                                    size_t ksize_width, size_t ksize_height,
                                    double sigmaX, double sigmaY, int border_type) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)depth; (void)cn;
    (void)margin_left; (void)margin_top; (void)margin_right; (void)margin_bottom;
    (void)ksize_width; (void)ksize_height; (void)sigmaX; (void)sigmaY; (void)border_type;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_medianBlur(const uchar* src_data, size_t src_step,
                                  uchar* dst_data, size_t dst_step,
                                  int width, int height, int depth, int cn, int ksize) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)depth; (void)cn; (void)ksize;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// =========================================================================
// DERIVATIVES
// =========================================================================

extern "C" int rpp_hal_sobel(const uchar* src_data, size_t src_step,
                             uchar* dst_data, size_t dst_step,
                             int width, int height, int src_depth, int dst_depth, int cn,
                             int margin_left, int margin_top, int margin_right, int margin_bottom,
                             int dx, int dy, int ksize, double scale, double delta, int border_type) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)src_depth; (void)dst_depth; (void)cn;
    (void)margin_left; (void)margin_top; (void)margin_right; (void)margin_bottom;
    (void)dx; (void)dy; (void)ksize; (void)scale; (void)delta; (void)border_type;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// =========================================================================
// FEATURES
// =========================================================================

extern "C" int rpp_hal_canny(const uchar* src_data, size_t src_step,
                             uchar* dst_data, size_t dst_step,
                             int width, int height, int cn,
                             double lowThreshold, double highThreshold, int ksize, bool L2gradient) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)cn;
    (void)lowThreshold; (void)highThreshold; (void)ksize; (void)L2gradient;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// =========================================================================
// GEOMETRY (remaining stubs)
// =========================================================================

extern "C" int rpp_hal_warpPerspective(int src_type,
                                       const uchar* src_data, size_t src_step,
                                       int src_width, int src_height,
                                       uchar* dst_data, size_t dst_step,
                                       int dst_width, int dst_height,
                                       const double M[9], int interpolation,
                                       int borderType, const double borderValue[4]) {
    (void)src_type; (void)src_data; (void)src_step; (void)src_width; (void)src_height;
    (void)dst_data; (void)dst_step; (void)dst_width; (void)dst_height;
    (void)M; (void)interpolation; (void)borderType; (void)borderValue;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// =========================================================================
// COLOR CONVERSIONS
// =========================================================================

extern "C" int rpp_hal_cvtBGRtoBGR(const uchar* src_data, size_t src_step,
                                   uchar* dst_data, size_t dst_step,
                                   int width, int height, int depth,
                                   int scn, int dcn, bool swapBlue) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)depth; (void)scn; (void)dcn; (void)swapBlue;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvtBGRtoGray(const uchar* src_data, size_t src_step,
                                    uchar* dst_data, size_t dst_step,
                                    int width, int height, int depth,
                                    int scn, bool swapBlue) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)depth; (void)scn; (void)swapBlue;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvtGraytoBGR(const uchar* src_data, size_t src_step,
                                    uchar* dst_data, size_t dst_step,
                                    int width, int height, int depth, int dcn) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)depth; (void)dcn;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvtBGRtoHSV(const uchar* src_data, size_t src_step,
                                   uchar* dst_data, size_t dst_step,
                                   int width, int height, int depth,
                                   int scn, bool swapBlue, bool isFullRange, bool isHSV) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)depth; (void)scn; (void)swapBlue;
    (void)isFullRange; (void)isHSV;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvtHSVtoBGR(const uchar* src_data, size_t src_step,
                                   uchar* dst_data, size_t dst_step,
                                   int width, int height, int depth,
                                   int dcn, bool swapBlue, bool isFullRange, bool isHSV) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)depth; (void)dcn; (void)swapBlue;
    (void)isFullRange; (void)isHSV;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
