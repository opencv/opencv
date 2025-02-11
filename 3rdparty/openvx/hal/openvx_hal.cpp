#include "openvx_hal.hpp"
#include "opencv2/imgproc/hal/interface.h"

#define IVX_HIDE_INFO_WARNINGS
#include "ivx.hpp"

#include <string>
#include <vector>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstring>

//==================================================================================================
// utility
// ...

#if 0
#include <cstdio>
#define PRINT(...) printf(__VA_ARGS__)
#define PRINT_HALERR_MSG(type) PRINT("OpenVX HAL impl "#type" error: %s\n", e.what())
#else
#define PRINT(...)
#define PRINT_HALERR_MSG(type) (void)e
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

inline ivx::Context& getOpenVXHALContext()
{
#if __cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER >= 1800)
    //CXX11
    static thread_local ivx::Context instance = ivx::Context::create();
#else //__cplusplus >= 201103L || _MSC_VER >= 1800
    //CXX98
#ifdef _WIN32
        static __declspec(thread) ivx::Context instance = ivx::Context::create();
#else
        static __thread ivx::Context instance = ivx::Context::create();
#endif
#endif
    return instance;
}

inline bool dimTooBig(int size)
{
    static vx_uint16 current_vendor = getOpenVXHALContext().vendorID();

    if (current_vendor == VX_ID_KHRONOS || current_vendor == VX_ID_DEFAULT)
    {
        //OpenVX use uint32_t for image addressing
        return ((unsigned)size > (UINT_MAX / VX_SCALE_UNITY));
    }
    else
        return false;
}

//OpenVX calls have essential overhead so it make sense to skip them for small images
template <int kernel_id> inline bool                  skipSmallImages(int w, int h) { return w*h < 7680 * 4320; }
template <> inline bool           skipSmallImages<VX_KERNEL_MULTIPLY>(int w, int h) { return w*h <  640 *  480; }
template <> inline bool      skipSmallImages<VX_KERNEL_COLOR_CONVERT>(int w, int h) { return w*h < 2048 * 1536; }
template <> inline bool     skipSmallImages<VX_KERNEL_INTEGRAL_IMAGE>(int w, int h) { return w*h <  640 *  480; }
template <> inline bool        skipSmallImages<VX_KERNEL_WARP_AFFINE>(int w, int h) { return w*h < 1280 *  720; }
template <> inline bool   skipSmallImages<VX_KERNEL_WARP_PERSPECTIVE>(int w, int h) { return w*h <  320 *  240; }
template <> inline bool skipSmallImages<VX_KERNEL_CUSTOM_CONVOLUTION>(int w, int h) { return w*h <  320 *  240; }

inline void setConstantBorder(ivx::border_t &border, vx_uint8 val)
{
    border.mode = VX_BORDER_CONSTANT;
#if VX_VERSION > VX_VERSION_1_0
    border.constant_value.U8 = val;
#else
    border.constant_value = val;
#endif
}

inline void refineStep(int w, int h, int imgType, size_t& step)
{
    if (h == 1)
        step = w * ((imgType == VX_DF_IMAGE_RGBX ||
                     imgType == VX_DF_IMAGE_U32 || imgType == VX_DF_IMAGE_S32) ? 4 :
                     imgType == VX_DF_IMAGE_RGB ? 3 :
                    (imgType == VX_DF_IMAGE_U16 || imgType == VX_DF_IMAGE_S16 ||
                     imgType == VX_DF_IMAGE_UYVY || imgType == VX_DF_IMAGE_YUYV) ? 2 : 1);
}

//==================================================================================================
// ivx::Image wrapped to simplify call to swapHandle prior to release
// TODO update ivx::Image to handle swapHandle prior to release on the own

class vxImage: public ivx::Image
{
public:
    vxImage(const ivx::Image &_img) : ivx::Image(_img) {}

    ~vxImage()
    {
#if VX_VERSION > VX_VERSION_1_0
        swapHandle();
#endif
    }
};

//==================================================================================================
// real code starts here
// ...

#define OVX_BINARY_OP(hal_func, ovx_call, kernel_id)                                                                \
template <typename T>                                                                                               \
int ovx_hal_##hal_func(const T *a, size_t astep, const T *b, size_t bstep, T *c, size_t cstep, int w, int h)        \
{                                                                                                                   \
    if(skipSmallImages<kernel_id>(w, h))                                                                            \
        return CV_HAL_ERROR_NOT_IMPLEMENTED;                                                                        \
    if(dimTooBig(w) || dimTooBig(h))                                                                                \
        return CV_HAL_ERROR_NOT_IMPLEMENTED;                                                                        \
    refineStep(w, h, ivx::TypeToEnum<T>::imgType, astep);                                                           \
    refineStep(w, h, ivx::TypeToEnum<T>::imgType, bstep);                                                           \
    refineStep(w, h, ivx::TypeToEnum<T>::imgType, cstep);                                                           \
    try                                                                                                             \
    {                                                                                                               \
        ivx::Context ctx = getOpenVXHALContext();                                                                   \
        vxImage                                                                                                     \
            ia = ivx::Image::createFromHandle(ctx, ivx::TypeToEnum<T>::imgType,                                     \
                ivx::Image::createAddressing(w, h, sizeof(T), (vx_int32)(astep)), (void*)a),                        \
            ib = ivx::Image::createFromHandle(ctx, ivx::TypeToEnum<T>::imgType,                                     \
                ivx::Image::createAddressing(w, h, sizeof(T), (vx_int32)(bstep)), (void*)b),                        \
            ic = ivx::Image::createFromHandle(ctx, ivx::TypeToEnum<T>::imgType,                                     \
                ivx::Image::createAddressing(w, h, sizeof(T), (vx_int32)(cstep)), (void*)c);                        \
        ovx_call                                                                                                    \
    }                                                                                                               \
    catch (ivx::RuntimeError & e)                                                                                   \
    {                                                                                                               \
        PRINT_HALERR_MSG(runtime);                                                                                  \
        return CV_HAL_ERROR_UNKNOWN;                                                                                \
    }                                                                                                               \
    catch (ivx::WrapperError & e)                                                                                   \
    {                                                                                                               \
        PRINT_HALERR_MSG(wrapper);                                                                                  \
        return CV_HAL_ERROR_UNKNOWN;                                                                                \
    }                                                                                                               \
    return CV_HAL_ERROR_OK;                                                                                         \
}

OVX_BINARY_OP(add, { ivx::IVX_CHECK_STATUS(vxuAdd(ctx, ia, ib, VX_CONVERT_POLICY_SATURATE, ic)); }, VX_KERNEL_ADD)
OVX_BINARY_OP(sub, { ivx::IVX_CHECK_STATUS(vxuSubtract(ctx, ia, ib, VX_CONVERT_POLICY_SATURATE, ic)); }, VX_KERNEL_SUBTRACT)

OVX_BINARY_OP(absdiff, { ivx::IVX_CHECK_STATUS(vxuAbsDiff(ctx, ia, ib, ic)); }, VX_KERNEL_ABSDIFF)

OVX_BINARY_OP(and, { ivx::IVX_CHECK_STATUS(vxuAnd(ctx, ia, ib, ic)); }, VX_KERNEL_AND)
OVX_BINARY_OP(or , { ivx::IVX_CHECK_STATUS(vxuOr(ctx, ia, ib, ic)); }, VX_KERNEL_OR)
OVX_BINARY_OP(xor, { ivx::IVX_CHECK_STATUS(vxuXor(ctx, ia, ib, ic)); }, VX_KERNEL_XOR)

template <typename T>
int ovx_hal_mul(const T *a, size_t astep, const T *b, size_t bstep, T *c, size_t cstep, int w, int h, double scale)
{
    if(scale == 1.0 || sizeof(T) > 1 ?
       skipSmallImages<VX_KERNEL_ADD>(w, h) : /*actually it could be any kernel with generic minimum size*/
       skipSmallImages<VX_KERNEL_MULTIPLY>(w, h) )
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (dimTooBig(w) || dimTooBig(h))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    refineStep(w, h, ivx::TypeToEnum<T>::imgType, astep);
    refineStep(w, h, ivx::TypeToEnum<T>::imgType, bstep);
    refineStep(w, h, ivx::TypeToEnum<T>::imgType, cstep);
#ifdef _WIN32
    const float MAGIC_SCALE = 0x0.01010102p0;
#else
    const float MAGIC_SCALE = 0.003922; // 0x1.010102p-8;
#endif
    try
    {
        int rounding_policy = VX_ROUND_POLICY_TO_ZERO;
        float fscale = (float)scale;
        if (fabs(fscale - MAGIC_SCALE) > FLT_EPSILON)
        {
            int exp = 0;
            double significand = frexp(fscale, &exp);
            if ((significand != 0.5) || (exp > 1) || (exp < -14))
                return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
        else
        {
            fscale = MAGIC_SCALE;
            rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;// That's the only rounding that MUST be supported for 1/255 scale
        }
        ivx::Context ctx = getOpenVXHALContext();
        vxImage
            ia = ivx::Image::createFromHandle(ctx, ivx::TypeToEnum<T>::imgType,
                ivx::Image::createAddressing(w, h, sizeof(T), (vx_int32)(astep)), (void*)a),
            ib = ivx::Image::createFromHandle(ctx, ivx::TypeToEnum<T>::imgType,
                ivx::Image::createAddressing(w, h, sizeof(T), (vx_int32)(bstep)), (void*)b),
            ic = ivx::Image::createFromHandle(ctx, ivx::TypeToEnum<T>::imgType,
                ivx::Image::createAddressing(w, h, sizeof(T), (vx_int32)(cstep)), (void*)c);
        ivx::IVX_CHECK_STATUS(vxuMultiply(ctx, ia, ib, fscale, VX_CONVERT_POLICY_SATURATE, rounding_policy, ic));
    }
    catch (ivx::RuntimeError & e)
    {
        PRINT_HALERR_MSG(runtime);
        return CV_HAL_ERROR_UNKNOWN;
    }
    catch (ivx::WrapperError & e)
    {
        PRINT_HALERR_MSG(wrapper);
        return CV_HAL_ERROR_UNKNOWN;
    }
    return CV_HAL_ERROR_OK;
}

template int ovx_hal_add<uchar>(const uchar *a, size_t astep, const uchar *b, size_t bstep, uchar *c, size_t cstep, int w, int h);
template int ovx_hal_add<short>(const short *a, size_t astep, const short *b, size_t bstep, short *c, size_t cstep, int w, int h);
template int ovx_hal_sub<uchar>(const uchar *a, size_t astep, const uchar *b, size_t bstep, uchar *c, size_t cstep, int w, int h);
template int ovx_hal_sub<short>(const short *a, size_t astep, const short *b, size_t bstep, short *c, size_t cstep, int w, int h);

template int ovx_hal_absdiff<uchar>(const uchar *a, size_t astep, const uchar *b, size_t bstep, uchar *c, size_t cstep, int w, int h);
template int ovx_hal_absdiff<short>(const short *a, size_t astep, const short *b, size_t bstep, short *c, size_t cstep, int w, int h);

template int ovx_hal_and<uchar>(const uchar *a, size_t astep, const uchar *b, size_t bstep, uchar *c, size_t cstep, int w, int h);
template int ovx_hal_or<uchar>(const uchar *a, size_t astep, const uchar *b, size_t bstep, uchar *c, size_t cstep, int w, int h);
template int ovx_hal_xor<uchar>(const uchar *a, size_t astep, const uchar *b, size_t bstep, uchar *c, size_t cstep, int w, int h);

template int ovx_hal_mul<uchar>(const uchar *a, size_t astep, const uchar *b, size_t bstep, uchar *c, size_t cstep, int w, int h, double scale);
template int ovx_hal_mul<short>(const short *a, size_t astep, const short *b, size_t bstep, short *c, size_t cstep, int w, int h, double scale);

int ovx_hal_not(const uchar *a, size_t astep, uchar *c, size_t cstep, int w, int h)
{
    if (skipSmallImages<VX_KERNEL_NOT>(w, h))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (dimTooBig(w) || dimTooBig(h))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    refineStep(w, h, VX_DF_IMAGE_U8, astep);
    refineStep(w, h, VX_DF_IMAGE_U8, cstep);
    try
    {
        ivx::Context ctx = getOpenVXHALContext();
        vxImage
            ia = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(w, h, 1, (vx_int32)(astep)), (void*)a),
            ic = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(w, h, 1, (vx_int32)(cstep)), (void*)c);
        ivx::IVX_CHECK_STATUS(vxuNot(ctx, ia, ic));
    }
    catch (ivx::RuntimeError & e)
    {
        PRINT_HALERR_MSG(runtime);
        return CV_HAL_ERROR_UNKNOWN;
    }
    catch (ivx::WrapperError & e)
    {
        PRINT_HALERR_MSG(wrapper);
        return CV_HAL_ERROR_UNKNOWN;
    }
    return CV_HAL_ERROR_OK;
}

int ovx_hal_merge8u(const uchar **src_data, uchar *dst_data, int len, int cn)
{
    if (skipSmallImages<VX_KERNEL_CHANNEL_COMBINE>(len, 1))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (dimTooBig(len))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (cn != 3 && cn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    try
    {
        ivx::Context ctx = getOpenVXHALContext();
        vxImage
            ia = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(len, 1, 1, (vx_int32)(len)), (void*)src_data[0]),
            ib = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(len, 1, 1, (vx_int32)(len)), (void*)src_data[1]),
            ic = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(len, 1, 1, (vx_int32)(len)), (void*)src_data[2]),
            id = ivx::Image::createFromHandle(ctx, cn == 4 ? VX_DF_IMAGE_RGBX : VX_DF_IMAGE_RGB,
                ivx::Image::createAddressing(len, 1, cn, (vx_int32)(len*cn)), (void*)dst_data);
        ivx::IVX_CHECK_STATUS(vxuChannelCombine(ctx, ia, ib, ic,
            cn == 4 ? (vx_image)(ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                                     ivx::Image::createAddressing(len, 1, 1, (vx_int32)(len)), (void*)src_data[3])) : NULL,
            id));
    }
    catch (ivx::RuntimeError & e)
    {
        PRINT_HALERR_MSG(runtime);
        return CV_HAL_ERROR_UNKNOWN;
    }
    catch (ivx::WrapperError & e)
    {
        PRINT_HALERR_MSG(wrapper);
        return CV_HAL_ERROR_UNKNOWN;
    }
    return CV_HAL_ERROR_OK;
}

int ovx_hal_resize(int atype, const uchar *a, size_t astep, int aw, int ah, uchar *b, size_t bstep, int bw, int bh, double inv_scale_x, double inv_scale_y, int interpolation)
{
    if (skipSmallImages<VX_KERNEL_SCALE_IMAGE>(aw, ah))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (dimTooBig(aw) || dimTooBig(ah) || dimTooBig(bw) || dimTooBig(bh))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    refineStep(aw, ah, VX_DF_IMAGE_U8, astep);
    refineStep(bw, bh, VX_DF_IMAGE_U8, bstep);
    try
    {
        ivx::Context ctx = getOpenVXHALContext();
        vxImage
            ia = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(aw, ah, 1, (vx_int32)(astep)), (void*)a),
            ib = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(bw, bh, 1, (vx_int32)(bstep)), (void*)b);

        if (!((atype == CV_8UC1 || atype == CV_8SC1) &&
            inv_scale_x > 0 && inv_scale_y > 0 &&
            (bw - 0.5) / inv_scale_x - 0.5 < aw && (bh - 0.5) / inv_scale_y - 0.5 < ah &&
            (bw + 0.5) / inv_scale_x + 0.5 >= aw && (bh + 0.5) / inv_scale_y + 0.5 >= ah &&
            std::abs(bw / inv_scale_x - aw) < 0.1 && std::abs(bh / inv_scale_y - ah) < 0.1))
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        int mode;
        if (interpolation == CV_HAL_INTER_LINEAR)
        {
            mode = VX_INTERPOLATION_BILINEAR;
            if (inv_scale_x > 1 || inv_scale_y > 1)
                return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
        else if (interpolation == CV_HAL_INTER_AREA)
            return CV_HAL_ERROR_NOT_IMPLEMENTED; //mode = VX_INTERPOLATION_AREA;
        else if (interpolation == CV_HAL_INTER_NEAREST)
            return CV_HAL_ERROR_NOT_IMPLEMENTED; //mode = VX_INTERPOLATION_NEAREST_NEIGHBOR;
        else
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        ivx::IVX_CHECK_STATUS(vxuScaleImage(ctx, ia, ib, mode));
    }
    catch (ivx::RuntimeError & e)
    {
        PRINT_HALERR_MSG(runtime);
        return CV_HAL_ERROR_UNKNOWN;
    }
    catch (ivx::WrapperError & e)
    {
        PRINT_HALERR_MSG(wrapper);
        return CV_HAL_ERROR_UNKNOWN;
    }
    return CV_HAL_ERROR_OK;
}

int ovx_hal_warpAffine(int atype, const uchar *a, size_t astep, int aw, int ah, uchar *b, size_t bstep, int bw, int bh, const double M[6], int interpolation, int borderType, const double borderValue[4])
{
    if (skipSmallImages<VX_KERNEL_WARP_AFFINE>(aw, ah))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (dimTooBig(aw) || dimTooBig(ah) || dimTooBig(bw) || dimTooBig(bh))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    refineStep(aw, ah, VX_DF_IMAGE_U8, astep);
    refineStep(bw, bh, VX_DF_IMAGE_U8, bstep);
    try
    {
        ivx::Context ctx = getOpenVXHALContext();
        vxImage
            ia = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(aw, ah, 1, (vx_int32)(astep)), (void*)a),
            ib = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(bw, bh, 1, (vx_int32)(bstep)), (void*)b);

        if (!(atype == CV_8UC1 || atype == CV_8SC1))
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        if(borderType != CV_HAL_BORDER_CONSTANT) // Neither 1.0 nor 1.1 OpenVX support BORDER_REPLICATE for warpings
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        int mode;
        if (interpolation == CV_HAL_INTER_LINEAR)
            mode = VX_INTERPOLATION_BILINEAR;
        //AREA interpolation is unsupported
        //else if (interpolation == CV_HAL_INTER_AREA)
        //    mode = VX_INTERPOLATION_AREA;
        else if (interpolation == CV_HAL_INTER_NEAREST)
            mode = VX_INTERPOLATION_NEAREST_NEIGHBOR;
        else
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        std::vector<float> data;
        data.reserve(6);
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 2; ++i)
                data.push_back((float)(M[i * 3 + j]));

        ivx::Matrix mtx = ivx::Matrix::create(ctx, VX_TYPE_FLOAT32, 2, 3);
        mtx.copyFrom(data);
        //ATTENTION: VX_CONTEXT_IMMEDIATE_BORDER attribute change could lead to strange issues in multi-threaded environments
        //since OpenVX standart says nothing about thread-safety for now
        ivx::border_t prevBorder = ctx.immediateBorder();
        ctx.setImmediateBorder(VX_BORDER_CONSTANT, (vx_uint8)borderValue[0]);
        ivx::IVX_CHECK_STATUS(vxuWarpAffine(ctx, ia, mtx, mode, ib));
        ctx.setImmediateBorder(prevBorder);
    }
    catch (ivx::RuntimeError & e)
    {
        PRINT_HALERR_MSG(runtime);
        return CV_HAL_ERROR_UNKNOWN;
    }
    catch (ivx::WrapperError & e)
    {
        PRINT_HALERR_MSG(wrapper);
        return CV_HAL_ERROR_UNKNOWN;
    }
    return CV_HAL_ERROR_OK;
}

int ovx_hal_warpPerspective(int atype, const uchar *a, size_t astep, int aw, int ah, uchar *b, size_t bstep, int bw, int bh, const double M[9], int interpolation, int borderType, const double borderValue[4])
{
    if (skipSmallImages<VX_KERNEL_WARP_PERSPECTIVE>(aw, ah))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (dimTooBig(aw) || dimTooBig(ah) || dimTooBig(bw) || dimTooBig(bh))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    refineStep(aw, ah, VX_DF_IMAGE_U8, astep);
    refineStep(bw, bh, VX_DF_IMAGE_U8, bstep);
    try
    {
        ivx::Context ctx = getOpenVXHALContext();
        vxImage
            ia = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(aw, ah, 1, (vx_int32)(astep)), (void*)a),
            ib = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(bw, bh, 1, (vx_int32)(bstep)), (void*)b);

        if (!(atype == CV_8UC1 || atype == CV_8SC1))
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        if (borderType != CV_HAL_BORDER_CONSTANT) // Neither 1.0 nor 1.1 OpenVX support BORDER_REPLICATE for warpings
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        int mode;
        if (interpolation == CV_HAL_INTER_LINEAR)
            mode = VX_INTERPOLATION_BILINEAR;
        //AREA interpolation is unsupported
        //else if (interpolation == CV_HAL_INTER_AREA)
        //    mode = VX_INTERPOLATION_AREA;
        else if (interpolation == CV_HAL_INTER_NEAREST)
            mode = VX_INTERPOLATION_NEAREST_NEIGHBOR;
        else
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        std::vector<float> data;
        data.reserve(9);
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                data.push_back((float)(M[i * 3 + j]));

        ivx::Matrix mtx = ivx::Matrix::create(ctx, VX_TYPE_FLOAT32, 3, 3);
        mtx.copyFrom(data);
        //ATTENTION: VX_CONTEXT_IMMEDIATE_BORDER attribute change could lead to strange issues in multi-threaded environments
        //since OpenVX standart says nothing about thread-safety for now
        ivx::border_t prevBorder = ctx.immediateBorder();
        ctx.setImmediateBorder(VX_BORDER_CONSTANT, (vx_uint8)borderValue[0]);
        ivx::IVX_CHECK_STATUS(vxuWarpPerspective(ctx, ia, mtx, mode, ib));
        ctx.setImmediateBorder(prevBorder);
    }
    catch (ivx::RuntimeError & e)
    {
        PRINT_HALERR_MSG(runtime);
        return CV_HAL_ERROR_UNKNOWN;
    }
    catch (ivx::WrapperError & e)
    {
        PRINT_HALERR_MSG(wrapper);
        return CV_HAL_ERROR_UNKNOWN;
    }
    return CV_HAL_ERROR_OK;
}

struct cvhalFilter2D;

struct FilterCtx
{
    ivx::Convolution cnv;
    int dst_type;
    ivx::border_t border;
    FilterCtx(ivx::Context &ctx, const std::vector<short> data, int w, int h, int _dst_type, ivx::border_t & _border) :
        cnv(ivx::Convolution::create(ctx, w, h)), dst_type(_dst_type), border(_border) {
        cnv.copyFrom(data);
    }
};

int ovx_hal_filterInit(cvhalFilter2D **filter_context, uchar *kernel_data, size_t kernel_step, int kernel_type, int kernel_width, int kernel_height,
    int, int, int src_type, int dst_type, int borderType, double delta, int anchor_x, int anchor_y, bool allowSubmatrix, bool allowInplace)
{
    if (!filter_context || !kernel_data || allowSubmatrix || allowInplace || delta != 0 ||
        src_type != CV_8UC1 || (dst_type != CV_8UC1 && dst_type != CV_16SC1) ||
        kernel_width % 2 == 0 || kernel_height % 2 == 0 || anchor_x != kernel_width / 2 || anchor_y != kernel_height / 2)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    ivx::border_t border;
    switch (borderType)
    {
    case CV_HAL_BORDER_CONSTANT:
        setConstantBorder(border, 0);
        break;
    case CV_HAL_BORDER_REPLICATE:
        border.mode = VX_BORDER_REPLICATE;
        break;
    default:
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    ivx::Context ctx = getOpenVXHALContext();

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
        break;
    default:
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    FilterCtx* cnv = new FilterCtx(ctx, data, kernel_width, kernel_height, dst_type, border);
    if (!cnv)
        return CV_HAL_ERROR_UNKNOWN;

    *filter_context = (cvhalFilter2D*)(cnv);
    return CV_HAL_ERROR_OK;
}

int ovx_hal_filterFree(cvhalFilter2D *filter_context)
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

int ovx_hal_filter(cvhalFilter2D *filter_context, uchar *a, size_t astep, uchar *b, size_t bstep, int w, int h, int, int, int, int)
{
    if (skipSmallImages<VX_KERNEL_CUSTOM_CONVOLUTION>(w, h))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (dimTooBig(w) || dimTooBig(h))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    try
    {
        FilterCtx* cnv = (FilterCtx*)filter_context;
        if (!cnv)
            throw ivx::WrapperError("Bad HAL context");
        refineStep(w, h, VX_DF_IMAGE_U8, astep);
        refineStep(w, h, cnv->dst_type == CV_16SC1 ? VX_DF_IMAGE_S16 : VX_DF_IMAGE_U8, bstep);

        ivx::Context ctx = getOpenVXHALContext();
        vxImage
            ia = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(w, h, 1, (vx_int32)(astep)), (void*)a),
            ib = ivx::Image::createFromHandle(ctx, cnv->dst_type == CV_16SC1 ? VX_DF_IMAGE_S16 : VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(w, h, cnv->dst_type == CV_16SC1 ? 2 : 1, (vx_int32)(bstep)), (void*)b);

        //ATTENTION: VX_CONTEXT_IMMEDIATE_BORDER attribute change could lead to strange issues in multi-threaded environments
        //since OpenVX standart says nothing about thread-safety for now
        ivx::border_t prevBorder = ctx.immediateBorder();
        ctx.setImmediateBorder(cnv->border);
        ivx::IVX_CHECK_STATUS(vxuConvolve(ctx, ia, cnv->cnv, ib));
        ctx.setImmediateBorder(prevBorder);
    }
    catch (ivx::RuntimeError & e)
    {
        PRINT_HALERR_MSG(runtime);
        return CV_HAL_ERROR_UNKNOWN;
    }
    catch (ivx::WrapperError & e)
    {
        PRINT_HALERR_MSG(wrapper);
        return CV_HAL_ERROR_UNKNOWN;
    }
    return CV_HAL_ERROR_OK;
}

int ovx_hal_sepFilterInit(cvhalFilter2D **filter_context, int src_type, int dst_type,
    int kernel_type, uchar *kernelx_data, int kernelx_length, uchar *kernely_data, int kernely_length,
    int anchor_x, int anchor_y, double delta, int borderType)
{
    if (!filter_context || !kernelx_data || !kernely_data || delta != 0 ||
        src_type != CV_8UC1 || (dst_type != CV_8UC1 && dst_type != CV_16SC1) ||
        kernelx_length != 3 || kernely_length != 3 || anchor_x != 1 || anchor_y != 1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    ivx::border_t border;
    switch (borderType)
    {
    case CV_HAL_BORDER_CONSTANT:
        setConstantBorder(border, 0);
        break;
    case CV_HAL_BORDER_REPLICATE:
        border.mode = VX_BORDER_REPLICATE;
        break;
    default:
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    ivx::Context ctx = getOpenVXHALContext();

    //At the moment OpenVX doesn't support separable filters natively so combine kernels to generic convolution
    std::vector<short> data;
    data.reserve(kernelx_length*kernely_length);
    switch (kernel_type)
    {
    case CV_8UC1:
        for (int j = 0; j < kernely_length; ++j)
            for (int i = 0; i < kernelx_length; ++i)
                data.push_back((short)(kernely_data[j]) * kernelx_data[i]);
        break;
    case CV_8SC1:
        for (int j = 0; j < kernely_length; ++j)
            for (int i = 0; i < kernelx_length; ++i)
                data.push_back((short)(((schar*)kernely_data)[j]) * ((schar*)kernelx_data)[i]);
        break;
    default:
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    FilterCtx* cnv = new FilterCtx(ctx, data, kernelx_length, kernely_length, dst_type, border);
    if (!cnv)
        return CV_HAL_ERROR_UNKNOWN;

    *filter_context = (cvhalFilter2D*)(cnv);
    return CV_HAL_ERROR_OK;
}

#if VX_VERSION > VX_VERSION_1_0

struct MorphCtx
{
    ivx::Matrix mask;
    int operation;
    ivx::border_t border;
    MorphCtx(ivx::Context &ctx, const std::vector<vx_uint8> data, int w, int h, int _operation, ivx::border_t & _border) :
        mask(ivx::Matrix::create(ctx, ivx::TypeToEnum<vx_uint8>::value, w, h)), operation(_operation), border(_border) {
        mask.copyFrom(data);
    }
};

int ovx_hal_morphInit(cvhalFilter2D **filter_context, int operation, int src_type, int dst_type, int, int,
    int kernel_type, uchar *kernel_data, size_t kernel_step, int kernel_width, int kernel_height, int anchor_x, int anchor_y,
    int borderType, const double borderValue[4], int iterations, bool allowSubmatrix, bool allowInplace)
{
    if (!filter_context || !kernel_data || allowSubmatrix || allowInplace || iterations != 1 ||
        src_type != CV_8UC1 || dst_type != CV_8UC1 ||
        kernel_width % 2 == 0 || kernel_height % 2 == 0 || anchor_x != kernel_width / 2 || anchor_y != kernel_height / 2)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    ivx::border_t border;
    switch (borderType)
    {
    case CV_HAL_BORDER_CONSTANT:
        if (borderValue[0] == DBL_MAX && borderValue[1] == DBL_MAX && borderValue[2] == DBL_MAX && borderValue[3] == DBL_MAX)
        {
            if (operation == CV_HAL_MORPH_ERODE)
                setConstantBorder(border, UCHAR_MAX);
            else
                setConstantBorder(border, 0);
        }
        else
        {
            int rounded = (int)round(borderValue[0]);
            setConstantBorder(border, (vx_uint8)((unsigned)rounded <= UCHAR_MAX ? rounded : rounded > 0 ? UCHAR_MAX : 0));
        }
        break;
    case CV_HAL_BORDER_REPLICATE:
        border.mode = VX_BORDER_REPLICATE;
        break;
    default:
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    ivx::Context ctx = getOpenVXHALContext();

    vx_size maxKernelDim = ctx.nonlinearMaxDimension();
    if ((vx_size)kernel_width > maxKernelDim || (vx_size)kernel_height > maxKernelDim)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    std::vector<vx_uint8> kernel_mat;
    kernel_mat.reserve(kernel_width * kernel_height);
    switch (CV_MAT_DEPTH(kernel_type))
    {
    case CV_8U:
    case CV_8S:
        for (int j = 0; j < kernel_height; ++j)
        {
            uchar * kernel_row = kernel_data + j * kernel_step;
            for (int i = 0; i < kernel_width; ++i)
                kernel_mat.push_back(kernel_row[i] ? 255 : 0);
        }
        break;
    case CV_16U:
    case CV_16S:
        for (int j = 0; j < kernel_height; ++j)
        {
            short * kernel_row = (short*)(kernel_data + j * kernel_step);
            for (int i = 0; i < kernel_width; ++i)
                kernel_mat.push_back(kernel_row[i] ? 255 : 0);
        }
        break;
    case CV_32S:
        for (int j = 0; j < kernel_height; ++j)
        {
            int * kernel_row = (int*)(kernel_data + j * kernel_step);
            for (int i = 0; i < kernel_width; ++i)
                kernel_mat.push_back(kernel_row[i] ? 255 : 0);
        }
        break;
    case CV_32F:
        for (int j = 0; j < kernel_height; ++j)
        {
            float * kernel_row = (float*)(kernel_data + j * kernel_step);
            for (int i = 0; i < kernel_width; ++i)
                kernel_mat.push_back(kernel_row[i] ? 255 : 0);
        }
        break;
    case CV_64F:
        for (int j = 0; j < kernel_height; ++j)
        {
            double * kernel_row = (double*)(kernel_data + j * kernel_step);
            for (int i = 0; i < kernel_width; ++i)
                kernel_mat.push_back(kernel_row[i] ? 255 : 0);
        }
        break;
    default:
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    MorphCtx* mat;
    switch (operation)
    {
    case CV_HAL_MORPH_ERODE:
        mat = new MorphCtx(ctx, kernel_mat, kernel_width, kernel_height, VX_NONLINEAR_FILTER_MIN, border);
        break;
    case CV_HAL_MORPH_DILATE:
        mat = new MorphCtx(ctx, kernel_mat, kernel_width, kernel_height, VX_NONLINEAR_FILTER_MAX, border);
        break;
    default:
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }
    if (!mat)
        return CV_HAL_ERROR_UNKNOWN;

    *filter_context = (cvhalFilter2D*)(mat);
    return CV_HAL_ERROR_OK;
}

int ovx_hal_morphFree(cvhalFilter2D *filter_context)
{
    if (filter_context)
    {
        delete (MorphCtx*)filter_context;
        return CV_HAL_ERROR_OK;
    }
    else
    {
        return CV_HAL_ERROR_UNKNOWN;
    }
}

int ovx_hal_morph(cvhalFilter2D *filter_context, uchar *a, size_t astep, uchar *b, size_t bstep, int w, int h, int, int, int, int, int, int, int, int)
{
    if (skipSmallImages<VX_KERNEL_DILATE_3x3>(w, h))//Actually it make sense to separate checks if implementations of dilation and erosion have different performance gain
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (dimTooBig(w) || dimTooBig(h))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    refineStep(w, h, VX_DF_IMAGE_U8, astep);
    refineStep(w, h, VX_DF_IMAGE_U8, bstep);
    try
    {
        MorphCtx* mat = (MorphCtx*)filter_context;
        if (!mat)
            throw ivx::WrapperError("Bad HAL context");

        ivx::Context ctx = getOpenVXHALContext();
        vxImage
            ia = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(w, h, 1, (vx_int32)(astep)), (void*)a),
            ib = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(w, h, 1, (vx_int32)(bstep)), (void*)b);

        //ATTENTION: VX_CONTEXT_IMMEDIATE_BORDER attribute change could lead to strange issues in multi-threaded environments
        //since OpenVX standart says nothing about thread-safety for now
        ivx::border_t prevBorder = ctx.immediateBorder();
        ctx.setImmediateBorder(mat->border);
        ivx::IVX_CHECK_STATUS(vxuNonLinearFilter(ctx, mat->operation, ia, mat->mask, ib));
        ctx.setImmediateBorder(prevBorder);
    }
    catch (ivx::RuntimeError & e)
    {
        PRINT_HALERR_MSG(runtime);
        return CV_HAL_ERROR_UNKNOWN;
    }
    catch (ivx::WrapperError & e)
    {
        PRINT_HALERR_MSG(wrapper);
        return CV_HAL_ERROR_UNKNOWN;
    }
    return CV_HAL_ERROR_OK;
}

#endif // 1.0 guard

int ovx_hal_cvtBGRtoBGR(const uchar * a, size_t astep, uchar * b, size_t bstep, int w, int h, int depth, int acn, int bcn, bool swapBlue)
{
    if (skipSmallImages<VX_KERNEL_COLOR_CONVERT>(w, h))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (dimTooBig(w) || dimTooBig(h))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (depth != CV_8U || swapBlue || acn == bcn || (acn != 3 && acn != 4) || (bcn != 3 && bcn != 4))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (w & 1 || h & 1) // It's strange but sample implementation unable to convert odd sized images
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    refineStep(w, h, acn == 3 ? VX_DF_IMAGE_RGB : VX_DF_IMAGE_RGBX, astep);
    refineStep(w, h, bcn == 3 ? VX_DF_IMAGE_RGB : VX_DF_IMAGE_RGBX, bstep);
    try
    {
        ivx::Context ctx = getOpenVXHALContext();
        vxImage
            ia = ivx::Image::createFromHandle(ctx, acn == 3 ? VX_DF_IMAGE_RGB : VX_DF_IMAGE_RGBX,
                ivx::Image::createAddressing(w, h, acn, (vx_int32)astep), (void*)a),
            ib = ivx::Image::createFromHandle(ctx, bcn == 3 ? VX_DF_IMAGE_RGB : VX_DF_IMAGE_RGBX,
                ivx::Image::createAddressing(w, h, bcn, (vx_int32)bstep), b);
        ivx::IVX_CHECK_STATUS(vxuColorConvert(ctx, ia, ib));
    }
    catch (ivx::RuntimeError & e)
    {
        PRINT_HALERR_MSG(runtime);
        return CV_HAL_ERROR_UNKNOWN;
    }
    catch (ivx::WrapperError & e)
    {
        PRINT_HALERR_MSG(wrapper);
        return CV_HAL_ERROR_UNKNOWN;
    }
    return CV_HAL_ERROR_OK;
}

int ovx_hal_cvtGraytoBGR(const uchar * a, size_t astep, uchar * b, size_t bstep, int w, int h, int depth, int bcn)
{
    if (skipSmallImages<VX_KERNEL_CHANNEL_COMBINE>(w, h))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (dimTooBig(w) || dimTooBig(h))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (depth != CV_8U || (bcn != 3 && bcn != 4))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    refineStep(w, h, VX_DF_IMAGE_U8, astep);
    refineStep(w, h, bcn == 3 ? VX_DF_IMAGE_RGB : VX_DF_IMAGE_RGBX, bstep);
    try
    {
        ivx::Context ctx = getOpenVXHALContext();
        ivx::Image
            ia = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(w, h, 1, (vx_int32)astep), const_cast<uchar*>(a)),
            ib = ivx::Image::createFromHandle(ctx, bcn == 3 ? VX_DF_IMAGE_RGB : VX_DF_IMAGE_RGBX,
                ivx::Image::createAddressing(w, h, bcn, (vx_int32)bstep), b);
        ivx::IVX_CHECK_STATUS(vxuChannelCombine(ctx, ia, ia, ia,
        bcn == 4 ? (vx_image)(ivx::Image::createUniform(ctx, w, h, VX_DF_IMAGE_U8, vx_uint8(255))) : NULL,
            ib));
    }
    catch (ivx::RuntimeError & e)
    {
        PRINT_HALERR_MSG(runtime);
        return CV_HAL_ERROR_UNKNOWN;
    }
    catch (ivx::WrapperError & e)
    {
        PRINT_HALERR_MSG(wrapper);
        return CV_HAL_ERROR_UNKNOWN;
    }
    return CV_HAL_ERROR_OK;
}

int ovx_hal_cvtTwoPlaneYUVtoBGR(const uchar * a, size_t astep, uchar * b, size_t bstep, int w, int h, int bcn, bool swapBlue, int uIdx)
{
    return ovx_hal_cvtTwoPlaneYUVtoBGREx(a, astep, a + h * astep, astep, b, bstep, w, h, bcn, swapBlue, uIdx);
}

int ovx_hal_cvtTwoPlaneYUVtoBGREx(const uchar * a, size_t astep, const uchar * b, size_t bstep, uchar * c, size_t cstep, int w, int h, int bcn, bool swapBlue, int uIdx)
{
    if (skipSmallImages<VX_KERNEL_COLOR_CONVERT>(w, h))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (dimTooBig(w) || dimTooBig(h))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (!swapBlue || (bcn != 3 && bcn != 4))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (w & 1 || h & 1) // It's not described in spec but sample implementation unable to convert odd sized images
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    try
    {
        ivx::Context ctx = getOpenVXHALContext();

        std::vector<vx_imagepatch_addressing_t> addr;
        std::vector<void *> ptrs;
            addr.push_back(ivx::Image::createAddressing(w, h, 1, (vx_int32)astep));
            ptrs.push_back((void*)a);
            addr.push_back(ivx::Image::createAddressing(w / 2, h / 2, 2, (vx_int32)bstep));
            ptrs.push_back((void*)b);

        vxImage
            ia = ivx::Image::createFromHandle(ctx, uIdx ? VX_DF_IMAGE_NV21 : VX_DF_IMAGE_NV12, addr, ptrs);
        if (ia.range() == VX_CHANNEL_RANGE_FULL)
            return CV_HAL_ERROR_NOT_IMPLEMENTED; // OpenCV store NV12/NV21 as RANGE_RESTRICTED while OpenVX expect RANGE_FULL
        vxImage
            ib = ivx::Image::createFromHandle(ctx, bcn == 3 ? VX_DF_IMAGE_RGB : VX_DF_IMAGE_RGBX,
                ivx::Image::createAddressing(w, h, bcn, (vx_int32)cstep), c);
        ivx::IVX_CHECK_STATUS(vxuColorConvert(ctx, ia, ib));
    }
    catch (ivx::RuntimeError & e)
    {
        PRINT_HALERR_MSG(runtime);
        return CV_HAL_ERROR_UNKNOWN;
    }
    catch (ivx::WrapperError & e)
    {
        PRINT_HALERR_MSG(wrapper);
        return CV_HAL_ERROR_UNKNOWN;
    }
    return CV_HAL_ERROR_OK;
}

int ovx_hal_cvtThreePlaneYUVtoBGR(const uchar * a, size_t astep, uchar * b, size_t bstep, int w, int h, int bcn, bool swapBlue, int uIdx)
{
    if (skipSmallImages<VX_KERNEL_COLOR_CONVERT>(w, h))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (dimTooBig(w) || dimTooBig(h))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (!swapBlue || (bcn != 3 && bcn != 4) || uIdx || (size_t)w / 2 != astep - (size_t)w / 2)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (w & 1 || h & 1) // It's not described in spec but sample implementation unable to convert odd sized images
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    refineStep(w, h, VX_DF_IMAGE_IYUV, astep);
    refineStep(w, h, bcn == 3 ? VX_DF_IMAGE_RGB : VX_DF_IMAGE_RGBX, bstep);
    try
    {
        ivx::Context ctx = getOpenVXHALContext();

        std::vector<vx_imagepatch_addressing_t> addr;
        std::vector<void *> ptrs;
        addr.push_back(ivx::Image::createAddressing(w, h, 1, (vx_int32)astep));
        ptrs.push_back((void*)a);
        addr.push_back(ivx::Image::createAddressing(w / 2, h / 2, 1, w / 2));
        ptrs.push_back((void*)(a + h * astep));
        if (addr[1].dim_x != (astep - addr[1].dim_x))
            throw ivx::WrapperError("UV planes use variable stride");
        addr.push_back(ivx::Image::createAddressing(w / 2, h / 2, 1, w / 2));
        ptrs.push_back((void*)(a + h * astep + addr[1].dim_y * addr[1].stride_y));

        vxImage
            ia = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_IYUV, addr, ptrs);
        if (ia.range() == VX_CHANNEL_RANGE_FULL)
            return CV_HAL_ERROR_NOT_IMPLEMENTED; // OpenCV store NV12/NV21 as RANGE_RESTRICTED while OpenVX expect RANGE_FULL
        vxImage
            ib = ivx::Image::createFromHandle(ctx, bcn == 3 ? VX_DF_IMAGE_RGB : VX_DF_IMAGE_RGBX,
                ivx::Image::createAddressing(w, h, bcn, (vx_int32)bstep), b);
        ivx::IVX_CHECK_STATUS(vxuColorConvert(ctx, ia, ib));
    }
    catch (ivx::RuntimeError & e)
    {
        PRINT_HALERR_MSG(runtime);
        return CV_HAL_ERROR_UNKNOWN;
    }
    catch (ivx::WrapperError & e)
    {
        PRINT_HALERR_MSG(wrapper);
        return CV_HAL_ERROR_UNKNOWN;
    }
    return CV_HAL_ERROR_OK;
}

int ovx_hal_cvtBGRtoThreePlaneYUV(const uchar * a, size_t astep, uchar * b, size_t bstep, int w, int h, int acn, bool swapBlue, int uIdx)
{
    if (skipSmallImages<VX_KERNEL_COLOR_CONVERT>(w, h))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (dimTooBig(w) || dimTooBig(h))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (!swapBlue || (acn != 3 && acn != 4) || uIdx || (size_t)w / 2 != bstep - (size_t)w / 2)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (w & 1 || h & 1) // It's not described in spec but sample implementation unable to convert odd sized images
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    refineStep(w, h, acn == 3 ? VX_DF_IMAGE_RGB : VX_DF_IMAGE_RGBX, astep);
    refineStep(w, h, VX_DF_IMAGE_IYUV, bstep);
    try
    {
        ivx::Context ctx = getOpenVXHALContext();
        vxImage
            ia = ivx::Image::createFromHandle(ctx, acn == 3 ? VX_DF_IMAGE_RGB : VX_DF_IMAGE_RGBX,
                ivx::Image::createAddressing(w, h, acn, (vx_int32)astep), (void*)a);

        std::vector<vx_imagepatch_addressing_t> addr;
        std::vector<void *> ptrs;
        addr.push_back(ivx::Image::createAddressing(w, h, 1, (vx_int32)bstep));
        ptrs.push_back((void*)b);
        addr.push_back(ivx::Image::createAddressing(w / 2, h / 2, 1, w / 2));
        ptrs.push_back((void*)(b + h * bstep));
        if (addr[1].dim_x != (bstep - addr[1].dim_x))
            throw ivx::WrapperError("UV planes use variable stride");
        addr.push_back(ivx::Image::createAddressing(w / 2, h / 2, 1, w / 2));
        ptrs.push_back((void*)(b + h * bstep + addr[1].dim_y * addr[1].stride_y));

        vxImage
            ib = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_IYUV, addr, ptrs);
        ivx::IVX_CHECK_STATUS(vxuColorConvert(ctx, ia, ib));
    }
    catch (ivx::RuntimeError & e)
    {
        PRINT_HALERR_MSG(runtime);
        return CV_HAL_ERROR_UNKNOWN;
    }
    catch (ivx::WrapperError & e)
    {
        PRINT_HALERR_MSG(wrapper);
        return CV_HAL_ERROR_UNKNOWN;
    }
    return CV_HAL_ERROR_OK;
}

int ovx_hal_cvtOnePlaneYUVtoBGR(const uchar * a, size_t astep, uchar * b, size_t bstep, int w, int h, int bcn, bool swapBlue, int uIdx, int ycn)
{
    if (skipSmallImages<VX_KERNEL_COLOR_CONVERT>(w, h))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (dimTooBig(w) || dimTooBig(h))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (!swapBlue || (bcn != 3 && bcn != 4) || uIdx)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (w & 1) // It's not described in spec but sample implementation unable to convert odd sized images
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    refineStep(w, h, ycn ? VX_DF_IMAGE_UYVY : VX_DF_IMAGE_YUYV, astep);
    refineStep(w, h, bcn == 3 ? VX_DF_IMAGE_RGB : VX_DF_IMAGE_RGBX, bstep);
    try
    {
        ivx::Context ctx = getOpenVXHALContext();
        vxImage
            ia = ivx::Image::createFromHandle(ctx, ycn ? VX_DF_IMAGE_UYVY : VX_DF_IMAGE_YUYV,
                ivx::Image::createAddressing(w, h, 2, (vx_int32)astep), (void*)a);
        if (ia.range() == VX_CHANNEL_RANGE_FULL)
            return CV_HAL_ERROR_NOT_IMPLEMENTED; // OpenCV store NV12/NV21 as RANGE_RESTRICTED while OpenVX expect RANGE_FULL
        vxImage
            ib = ivx::Image::createFromHandle(ctx, bcn == 3 ? VX_DF_IMAGE_RGB : VX_DF_IMAGE_RGBX,
                ivx::Image::createAddressing(w, h, bcn, (vx_int32)bstep), b);
        ivx::IVX_CHECK_STATUS(vxuColorConvert(ctx, ia, ib));
    }
    catch (ivx::RuntimeError & e)
    {
        PRINT_HALERR_MSG(runtime);
        return CV_HAL_ERROR_UNKNOWN;
    }
    catch (ivx::WrapperError & e)
    {
        PRINT_HALERR_MSG(wrapper);
        return CV_HAL_ERROR_UNKNOWN;
    }
    return CV_HAL_ERROR_OK;
}

int ovx_hal_integral(int depth, int sdepth, int, const uchar * a, size_t astep, uchar * b, size_t bstep, uchar * c, size_t, uchar * d, size_t, int w, int h, int cn)
{
    if (skipSmallImages<VX_KERNEL_INTEGRAL_IMAGE>(w, h))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (depth != CV_8U || sdepth != CV_32S || c != NULL || d != NULL || cn != 1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    refineStep(w, h, VX_DF_IMAGE_U8, astep);
    try
    {
        ivx::Context ctx = getOpenVXHALContext();
        ivx::Image
            ia = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(w, h, 1, (vx_int32)astep), const_cast<uchar*>(a)),
            ib = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U32,
                ivx::Image::createAddressing(w, h, 4, (vx_int32)bstep), (unsigned int *)(b + bstep + sizeof(unsigned int)));
        ivx::IVX_CHECK_STATUS(vxuIntegralImage(ctx, ia, ib));
        std::memset(b, 0, (w + 1) * sizeof(unsigned int));
        b += bstep;
        for (int i = 0; i < h; i++, b += bstep)
        {
            *((unsigned int*)b) = 0;
        }
    }
    catch (ivx::RuntimeError & e)
    {
        PRINT_HALERR_MSG(runtime);
        return CV_HAL_ERROR_UNKNOWN;
    }
    catch (ivx::WrapperError & e)
    {
        PRINT_HALERR_MSG(wrapper);
        return CV_HAL_ERROR_UNKNOWN;
    }

    return CV_HAL_ERROR_OK;
}
