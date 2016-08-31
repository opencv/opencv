#ifndef OPENCV_OPENVX_HAL_HPP_INCLUDED
#define OPENCV_OPENVX_HAL_HPP_INCLUDED

#include "opencv2/core/hal/interface.h"

#include "VX/vx.h"
#include "VX/vxu.h"

#include <string>
#include <vector>

//==================================================================================================
// utility
// ...

#if 0
#include <cstdio>
#define PRINT(...) printf(__VA_ARGS__)
#else
#define PRINT(...)
#endif

#if __cplusplus >= 201103L
#include <chrono>
struct Tick
{
    typedef std::chrono::time_point<std::chrono::steady_clock> point_t;
    point_t start;
    point_t point;
    Tick()
    {
        start = std::chrono::steady_clock::now();
        point = std::chrono::steady_clock::now();
    }
    inline int one()
    {
        point_t old = point;
        point = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(point - old).count();
    }
    inline int total()
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    }
};
#endif

//==================================================================================================
// One more OpenVX C++ binding :-)
// ...

template <typename T>
struct VX_Traits
{
    enum {
        ImgType = 0,
        DataType = 0
    };
};

template <>
struct VX_Traits<uchar>
{
    enum {
        ImgType = VX_DF_IMAGE_U8,
        DataType = VX_TYPE_UINT8
    };
};

template <>
struct VX_Traits<ushort>
{
    enum {
        ImgType = VX_DF_IMAGE_U16,
        DataType = VX_TYPE_UINT16
    };
};

template <>
struct VX_Traits<short>
{
    enum {
        ImgType = VX_DF_IMAGE_S16,
        DataType = VX_TYPE_INT16
    };
};

template <>
struct VX_Traits<float>
{
    enum {
        ImgType = 0,
        DataType = VX_TYPE_FLOAT32
    };
};


struct vxContext;
struct vxImage;
struct vxErr;


struct vxErr
{
    vx_status status;
    std::string msg;
    vxErr(vx_status status_, const std::string & msg_) : status(status_), msg(msg_) {}
    void check()
    {
        if (status != VX_SUCCESS)
            throw *this;
    }
    void print()
    {
        PRINT("OpenVX HAL impl error: %d (%s)\n", status, msg.c_str());
    }
    static void check(vx_context ctx)
    {
        vxErr(vxGetStatus((vx_reference)ctx), "context check").check();
    }
    static void check(vx_image img)
    {
        vxErr(vxGetStatus((vx_reference)img), "image check").check();
    }
    static void check(vx_matrix mtx)
    {
        vxErr(vxGetStatus((vx_reference)mtx), "matrix check").check();
    }
    static void check(vx_convolution cnv)
    {
        vxErr(vxGetStatus((vx_reference)cnv), "convolution check").check();
    }
    static void check(vx_status s)
    {
        vxErr(s, "status check").check();
    }
};


struct vxContext
{
    vx_context ctx;
    static vxContext * getContext();
private:
    vxContext()
    {
        ctx = vxCreateContext();
        vxErr::check(ctx);
    }
    ~vxContext()
    {
        vxReleaseContext(&ctx);
    }
};


struct vxImage
{
    vx_image img;

    template <typename T>
    vxImage(vxContext &ctx, const T *data, size_t step, int w, int h)
    {
        if (h == 1)
            step = w * sizeof(T);
        vx_imagepatch_addressing_t addr;
        addr.dim_x = w;
        addr.dim_y = h;
        addr.stride_x = sizeof(T);
        addr.stride_y = step;
        addr.scale_x = VX_SCALE_UNITY;
        addr.scale_y = VX_SCALE_UNITY;
        addr.step_x = 1;
        addr.step_y = 1;
        void *ptrs[] = { (void*)data };
        img = vxCreateImageFromHandle(ctx.ctx, VX_Traits<T>::ImgType, &addr, ptrs, VX_MEMORY_TYPE_HOST);
        vxErr::check(img);
    }
    ~vxImage()
    {
        vxErr::check(vxSwapImageHandle(img, NULL, NULL, 1));
        vxReleaseImage(&img);
    }
};

struct vxMatrix
{
    vx_matrix mtx;

    template <typename T>
    vxMatrix(vxContext &ctx, const T *data, int w, int h)
    {
        mtx = vxCreateMatrix(ctx.ctx, VX_Traits<T>::DataType, w, h);
        vxErr::check(mtx);
        vxErr::check(vxCopyMatrix(mtx, data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    }
    ~vxMatrix()
    {
        vxReleaseMatrix(&mtx);
    }
};

struct vxConvolution
{
    vx_convolution cnv;

    vxConvolution(vxContext &ctx, const short *data, int w, int h)
    {
        cnv = vxCreateConvolution(ctx.ctx, w, h);
        vxErr::check(cnv);
        vxErr::check(vxCopyConvolutionCoefficients(cnv, (void*)data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    }
    ~vxConvolution()
    {
        vxReleaseConvolution(&cnv);
    }
};

//==================================================================================================
// real code starts here
// ...

#define OVX_BINARY_OP(hal_func, ovx_call, ...)                                                                                   \
template <typename T>                                                                                                            \
inline int ovx_hal_##hal_func(const T *a, size_t astep, const T *b, size_t bstep, T *c, size_t cstep, int w, int h, __VA_ARGS__) \
{                                                                                                                                \
    try                                                                                                                          \
    {                                                                                                                            \
        vxContext * ctx = vxContext::getContext();                                                                               \
        vxImage ia(*ctx, a, astep, w, h);                                                                                        \
        vxImage ib(*ctx, b, bstep, w, h);                                                                                        \
        vxImage ic(*ctx, c, cstep, w, h);                                                                                        \
        ovx_call                                                                                                                 \
    }                                                                                                                            \
    catch (vxErr & e)                                                                                                            \
    {                                                                                                                            \
        e.print();                                                                                                               \
        return CV_HAL_ERROR_UNKNOWN;                                                                                             \
    }                                                                                                                            \
    return CV_HAL_ERROR_OK;                                                                                                      \
}

OVX_BINARY_OP(add, {vxErr::check(vxuAdd(ctx->ctx, ia.img, ib.img, VX_CONVERT_POLICY_SATURATE, ic.img));})
OVX_BINARY_OP(sub, {vxErr::check(vxuSubtract(ctx->ctx, ia.img, ib.img, VX_CONVERT_POLICY_SATURATE, ic.img));})

OVX_BINARY_OP(absdiff, {vxErr::check(vxuAbsDiff(ctx->ctx, ia.img, ib.img, ic.img));})

OVX_BINARY_OP(and, {vxErr::check(vxuAnd(ctx->ctx, ia.img, ib.img, ic.img));})
OVX_BINARY_OP(or, {vxErr::check(vxuOr(ctx->ctx, ia.img, ib.img, ic.img));})
OVX_BINARY_OP(xor, {vxErr::check(vxuXor(ctx->ctx, ia.img, ib.img, ic.img));})

OVX_BINARY_OP(mul, {vxErr::check(vxuMultiply(ctx->ctx, ia.img, ib.img, (float)scale, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_ZERO, ic.img));}, double scale)

inline int ovx_hal_not(const uchar *a, size_t astep, uchar *c, size_t cstep, int w, int h)
{
    try
    {
        vxContext * ctx = vxContext::getContext();
        vxImage ia(*ctx, a, astep, w, h);
        vxImage ic(*ctx, c, cstep, w, h);
        vxErr::check(vxuNot(ctx->ctx, ia.img, ic.img));
    }
    catch (vxErr & e)
    {
        e.print();
        return CV_HAL_ERROR_UNKNOWN;
    }
    return CV_HAL_ERROR_OK;
}

#if defined OPENCV_IMGPROC_HAL_INTERFACE_H
#define CV_HAL_INTER_NEAREST 0
#define CV_HAL_INTER_LINEAR 1
#define CV_HAL_INTER_CUBIC 2
#define CV_HAL_INTER_AREA 3
#define CV_HAL_INTER_LANCZOS4 4
#define MORPH_ERODE 0
#define MORPH_DILATE 1

inline int ovx_hal_resize(int atype, const uchar *a, size_t astep, int aw, int ah, uchar *b, size_t bstep, int bw, int bh, double inv_scale_x, double inv_scale_y, int interpolation)
{
    try
    {
        vxContext * ctx = vxContext::getContext();
        vxImage ia(*ctx, a, astep, aw, ah);
        vxImage ib(*ctx, b, bstep, bw, bh);

        if(!((atype == CV_8UC1 || atype == CV_8SC1) &&
             inv_scale_x > 0 && inv_scale_y > 0 &&
             (bw - 0.5) / inv_scale_x - 0.5 < aw && (bh - 0.5) / inv_scale_y - 0.5 < ah &&
             (bw + 0.5) / inv_scale_x + 0.5 >= aw && (bh + 0.5) / inv_scale_y + 0.5 >= ah &&
             std::abs(bw / inv_scale_x - aw) < 0.1 && std::abs(bh / inv_scale_y - ah) < 0.1 ))
            vxErr(VX_ERROR_INVALID_PARAMETERS, "Bad scale").check();

        int mode;
        if (interpolation == CV_HAL_INTER_LINEAR)
            mode = VX_INTERPOLATION_BILINEAR;
        else if (interpolation == CV_HAL_INTER_AREA)
            mode = VX_INTERPOLATION_AREA;
        else if (interpolation == CV_HAL_INTER_NEAREST)
            mode = VX_INTERPOLATION_NEAREST_NEIGHBOR;
        else
            vxErr(VX_ERROR_INVALID_PARAMETERS, "Bad interpolation mode").check();

        vxErr::check( vxuScaleImage(ctx->ctx, ia.img, ib.img, mode));
    }
    catch (vxErr & e)
    {
        e.print();
        return CV_HAL_ERROR_UNKNOWN;
    }
    return CV_HAL_ERROR_OK;
}

inline int ovx_hal_warpAffine(int atype, const uchar *a, size_t astep, int aw, int ah, uchar *b, size_t bstep, int bw, int bh, const double M[6], int interpolation, int, const double*)
{
    try
    {
        vxContext * ctx = vxContext::getContext();
        vxImage ia(*ctx, a, astep, aw, ah);
        vxImage ib(*ctx, b, bstep, bw, bh);

        if (!(atype == CV_8UC1 || atype == CV_8SC1))
            vxErr(VX_ERROR_INVALID_PARAMETERS, "Bad input type").check();

        // It make sense to check border mode as well, but it's impossible to set border mode for immediate OpenVX calls
        // So the only supported modes should be UNDEFINED(there is no such for HAL) and probably ISOLATED

        int mode;
        if (interpolation == CV_HAL_INTER_LINEAR)
            mode = VX_INTERPOLATION_BILINEAR;
        else if (interpolation == CV_HAL_INTER_AREA)
            mode = VX_INTERPOLATION_AREA;
        else if (interpolation == CV_HAL_INTER_NEAREST)
            mode = VX_INTERPOLATION_NEAREST_NEIGHBOR;
        else
            vxErr(VX_ERROR_INVALID_PARAMETERS, "Bad interpolation mode").check();

        vxMatrix mtx(*ctx, std::vector<float>(M, M + 6).data(), 2, 3);
        vxErr::check(vxuWarpAffine(ctx->ctx, ia.img, mtx.mtx, mode, ib.img));
    }
    catch (vxErr & e)
    {
        e.print();
        return CV_HAL_ERROR_UNKNOWN;
    }
    return CV_HAL_ERROR_OK;
}

inline int ovx_hal_warpPerspectve(int atype, const uchar *a, size_t astep, int aw, int ah, uchar *b, size_t bstep, int bw, int bh, const double M[9], int interpolation, int, const double*)
{
    try
    {
        vxContext * ctx = vxContext::getContext();
        vxImage ia(*ctx, a, astep, aw, ah);
        vxImage ib(*ctx, b, bstep, bw, bh);

        if (!(atype == CV_8UC1 || atype == CV_8SC1))
            vxErr(VX_ERROR_INVALID_PARAMETERS, "Bad input type").check();

        // It make sense to check border mode as well, but it's impossible to set border mode for immediate OpenVX calls
        // So the only supported modes should be UNDEFINED(there is no such for HAL) and probably ISOLATED

        int mode;
        if (interpolation == CV_HAL_INTER_LINEAR)
            mode = VX_INTERPOLATION_BILINEAR;
        else if (interpolation == CV_HAL_INTER_AREA)
            mode = VX_INTERPOLATION_AREA;
        else if (interpolation == CV_HAL_INTER_NEAREST)
            mode = VX_INTERPOLATION_NEAREST_NEIGHBOR;
        else
            vxErr(VX_ERROR_INVALID_PARAMETERS, "Bad interpolation mode").check();

        vxMatrix mtx(*ctx, std::vector<float>(M, M + 9).data(), 3, 3);
        vxErr::check(vxuWarpAffine(ctx->ctx, ia.img, mtx.mtx, mode, ib.img));
    }
    catch (vxErr & e)
    {
        e.print();
        return CV_HAL_ERROR_UNKNOWN;
    }
    return CV_HAL_ERROR_OK;
}

struct cvhalFilter2D;

struct FilterCtx
{
    vxConvolution cnv;
    int dst_type;
    FilterCtx(vxContext &ctx, const short *data, int w, int h, int _dst_type) :
        cnv(ctx, data, w, h), dst_type(_dst_type) {}
};

inline int ovx_hal_filterInit(cvhalFilter2D **filter_context, uchar *kernel_data, size_t kernel_step, int kernel_type, int kernel_width, int kernel_height,
    int max_width, int max_height, int src_type, int dst_type, int borderType, double delta, int anchor_x, int anchor_y, bool allowSubmatrix, bool allowInplace)
{
    if (!filter_context || !kernel_data || allowSubmatrix || allowInplace || delta != 0 ||
        src_type != CV_8UC1 || (dst_type != CV_8UC1 && dst_type != CV_16SC1) ||
        kernel_width % 2 == 0 || kernel_height % 2 == 0 || anchor_x != kernel_width / 2 || anchor_y != kernel_height / 2)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // It make sense to check border mode as well, but it's impossible to set border mode for immediate OpenVX calls
    // So the only supported modes should be UNDEFINED(there is no such for HAL) and probably ISOLATED

    vxContext * ctx = vxContext::getContext();

    std::vector<short> data;
    data.reserve(kernel_width*kernel_height);
    switch (kernel_type)
    {
    case CV_8UC1:
        for (int j = 0; j < kernel_height; ++j)
        {
            uchar * row = (uchar*)(kernel_data + kernel_step*j);
            for (int i = 0; i < kernel_width; ++i)
                data.push_back(row[i]);
        }
        break;
    case CV_8SC1:
        for (int j = 0; j < kernel_height; ++j)
        {
            schar * row = (schar*)(kernel_data + kernel_step*j);
            for (int i = 0; i < kernel_width; ++i)
                data.push_back(row[i]);
        }
        break;
    case CV_16SC1:
        for (int j = 0; j < kernel_height; ++j)
        {
            short * row = (short*)(kernel_data + kernel_step*j);
            for (int i = 0; i < kernel_width; ++i)
                data.push_back(row[i]);
        }
    default:
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    FilterCtx* cnv = new FilterCtx(*ctx, data.data(), kernel_width, kernel_height, dst_type);
    if (!cnv)
        return CV_HAL_ERROR_UNKNOWN;

    *filter_context = (cvhalFilter2D*)(cnv);
    return CV_HAL_ERROR_OK;
}

inline int ovx_hal_filterFree(cvhalFilter2D *filter_context)
{
    if (filter_context)
    {
        delete (FilterCtx*)filter_context;
        return CV_HAL_ERROR_OK;
    }
    else
    {
        return CV_HAL_ERROR_UNKNOWN;
    }
}

inline int ovx_hal_filter(cvhalFilter2D *filter_context, uchar *a, size_t astep, uchar *b, size_t bstep, int w, int h, int full_w, int full_h, int offset_x, int offset_y)
{
    try
    {
        FilterCtx* cnv = (FilterCtx*)filter_context;
        if(cnv)
            vxErr::check(cnv->cnv.cnv);
        else
            vxErr(VX_ERROR_INVALID_PARAMETERS, "Bad HAL context").check();

        vxContext * ctx = vxContext::getContext();
        vxImage ia(*ctx, a, astep, w, h);

        if (cnv->dst_type == CV_16SC1)
        {
            vxImage ib(*ctx, (short*)b, bstep, w, h);
            vxErr::check(vxuConvolve(ctx->ctx, ia.img, cnv->cnv.cnv, ib.img));
        }
        else
        {
            vxImage ib(*ctx, b, bstep, w, h);
            vxErr::check(vxuConvolve(ctx->ctx, ia.img, cnv->cnv.cnv, ib.img));
        }
    }
    catch (vxErr & e)
    {
        e.print();
        return CV_HAL_ERROR_UNKNOWN;
    }
    return CV_HAL_ERROR_OK;
}

#endif

//==================================================================================================
// functions redefinition
// ...

#undef cv_hal_add8u
#define cv_hal_add8u ovx_hal_add<uchar>
#undef cv_hal_add16s
#define cv_hal_add16s ovx_hal_add<short>
#undef cv_hal_sub8u
#define cv_hal_sub8u ovx_hal_sub<uchar>
#undef cv_hal_sub16s
#define cv_hal_sub16s ovx_hal_sub<short>

#undef cv_hal_absdiff8u
#define cv_hal_absdiff8u ovx_hal_absdiff<uchar>
#undef cv_hal_absdiff16s
#define cv_hal_absdiff16s ovx_hal_absdiff<short>

#undef cv_hal_and8u
#define cv_hal_and8u ovx_hal_and<uchar>
#undef cv_hal_or8u
#define cv_hal_or8u ovx_hal_or<uchar>
#undef cv_hal_xor8u
#define cv_hal_xor8u ovx_hal_xor<uchar>
#undef cv_hal_not8u
#define cv_hal_not8u ovx_hal_not

#undef cv_hal_mul8u
#define cv_hal_mul8u ovx_hal_mul<uchar>
#undef cv_hal_mul16s
#define cv_hal_mul16s ovx_hal_mul<short>

#if defined OPENCV_IMGPROC_HAL_INTERFACE_H

#undef cv_hal_resize
#define cv_hal_resize ovx_hal_resize
#undef cv_hal_warpAffine
#define cv_hal_warpAffine ovx_hal_warpAffine
#undef cv_hal_warpPerspective
#define cv_hal_warpPerspective ovx_hal_warpPerspectve

#undef cv_hal_filterInit
#define cv_hal_filterInit ovx_hal_filterInit
#undef cv_hal_filter
#define cv_hal_filter ovx_hal_filter
#undef cv_hal_filterFree
#define cv_hal_filterFree ovx_hal_filterFree

#endif

#endif
