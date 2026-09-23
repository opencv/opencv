// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "rvv_hal.hpp"

namespace cv { namespace rvv_hal { namespace core {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

#define CV_HAL_RVV_ROTATE_CREATE_x4(suffix, width, v0, v1, v2, v3) \
    __riscv_vset_v_##suffix##m##width##_##suffix##m##width##x4(v, 0, v0); \
    v = __riscv_vset(v, 1, v1); \
    v = __riscv_vset(v, 2, v2); \
    v = __riscv_vset(v, 3, v3);

#define CV_HAL_RVV_ROTATE_CREATE_x8(suffix, width, v0, v1, v2, v3, v4, v5, v6, v7) \
    __riscv_vset_v_##suffix##m##width##_##suffix##m##width##x8(v, 0, v0); \
    v = __riscv_vset(v, 1, v1); \
    v = __riscv_vset(v, 2, v2); \
    v = __riscv_vset(v, 3, v3); \
    v = __riscv_vset(v, 4, v4); \
    v = __riscv_vset(v, 5, v5); \
    v = __riscv_vset(v, 6, v6); \
    v = __riscv_vset(v, 7, v7);

#if defined(__clang__) && __clang_major__ < 18
#define __riscv_vcreate_v_u8m1x8(v0, v1, v2, v3, v4, v5, v6, v7) \
    CV_HAL_RVV_ROTATE_CREATE_x8(u8, 1, v0, v1, v2, v3, v4, v5, v6, v7)
#define __riscv_vcreate_v_u16m1x8(v0, v1, v2, v3, v4, v5, v6, v7) \
    CV_HAL_RVV_ROTATE_CREATE_x8(u16, 1, v0, v1, v2, v3, v4, v5, v6, v7)
#define __riscv_vcreate_v_u32m1x4(v0, v1, v2, v3) \
    CV_HAL_RVV_ROTATE_CREATE_x4(u32, 1, v0, v1, v2, v3)
#define __riscv_vcreate_v_u64m1x8(v0, v1, v2, v3, v4, v5, v6, v7) \
    CV_HAL_RVV_ROTATE_CREATE_x8(u64, 1, v0, v1, v2, v3, v4, v5, v6, v7)
#endif

#define CV_HAL_RVV_ROTATE_8_ROWS(func, T, RVV, suffix, bits) \
static void func(const uchar* src_data, size_t src_step, int src_width, int src_height, \
                 uchar* dst_data, size_t dst_step, int angle) \
{ \
    const size_t src_step_elems = src_step / sizeof(T); \
    int y = 0; \
    for (; y <= src_height - 8; y += 8) \
    { \
        const T* src = reinterpret_cast<const T*>(src_data) + y * src_step_elems; \
        for (int x = 0; x < src_width; ) \
        { \
            const size_t vl = __riscv_vsetvl_e##bits##m1(src_width - x); \
            auto v0 = __riscv_vle##bits##_v_##suffix##m1(src + x, vl); \
            auto v1 = __riscv_vle##bits##_v_##suffix##m1(src + src_step_elems + x, vl); \
            auto v2 = __riscv_vle##bits##_v_##suffix##m1(src + 2 * src_step_elems + x, vl); \
            auto v3 = __riscv_vle##bits##_v_##suffix##m1(src + 3 * src_step_elems + x, vl); \
            auto v4 = __riscv_vle##bits##_v_##suffix##m1(src + 4 * src_step_elems + x, vl); \
            auto v5 = __riscv_vle##bits##_v_##suffix##m1(src + 5 * src_step_elems + x, vl); \
            auto v6 = __riscv_vle##bits##_v_##suffix##m1(src + 6 * src_step_elems + x, vl); \
            auto v7 = __riscv_vle##bits##_v_##suffix##m1(src + 7 * src_step_elems + x, vl); \
            vuint##bits##m1x8_t v; \
            T* dst; \
            ptrdiff_t stride; \
            if (angle == 90) \
            { \
                v = __riscv_vcreate_v_##suffix##m1x8(v7, v6, v5, v4, v3, v2, v1, v0); \
                dst = reinterpret_cast<T*>(dst_data + static_cast<size_t>(x) * dst_step) \
                    + src_height - y - 8; \
                stride = static_cast<ptrdiff_t>(dst_step); \
            } \
            else \
            { \
                v = __riscv_vcreate_v_##suffix##m1x8(v0, v1, v2, v3, v4, v5, v6, v7); \
                dst = reinterpret_cast<T*>(dst_data + static_cast<size_t>(src_width - 1 - x) * dst_step) + y; \
                stride = -static_cast<ptrdiff_t>(dst_step); \
            } \
            __riscv_vssseg8e##bits(dst, stride, v, vl); \
            x += static_cast<int>(vl); \
        } \
    } \
    for (; y < src_height; ++y) \
    { \
        const T* src = reinterpret_cast<const T*>(src_data + static_cast<size_t>(y) * src_step); \
        for (int x = 0; x < src_width; ) \
        { \
            const size_t vl = RVV::setvl(src_width - x); \
            auto v = RVV::vload(src + x, vl); \
            T* dst; \
            ptrdiff_t stride; \
            if (angle == 90) \
            { \
                dst = reinterpret_cast<T*>(dst_data + static_cast<size_t>(x) * dst_step) \
                    + src_height - 1 - y; \
                stride = static_cast<ptrdiff_t>(dst_step); \
            } \
            else \
            { \
                dst = reinterpret_cast<T*>(dst_data + static_cast<size_t>(src_width - 1 - x) * dst_step) + y; \
                stride = -static_cast<ptrdiff_t>(dst_step); \
            } \
            RVV::vstore_stride(dst, stride, v, vl); \
            x += static_cast<int>(vl); \
        } \
    } \
}

CV_HAL_RVV_ROTATE_8_ROWS(rotate90_8u, uint8_t, RVV_U8M8, u8, 8)
CV_HAL_RVV_ROTATE_8_ROWS(rotate90_16u, uint16_t, RVV_U16M8, u16, 16)
CV_HAL_RVV_ROTATE_8_ROWS(rotate90_64u, uint64_t, RVV_U64M8, u64, 64)

static void rotate180_8u(const uchar* src_data, size_t src_step, int src_width, int src_height,
                         uchar* dst_data, size_t dst_step, int)
{
    for (int y = 0; y < src_height; ++y)
    {
        const uint8_t* src = src_data + static_cast<size_t>(y) * src_step;
        uint8_t* dst = dst_data + static_cast<size_t>(src_height - 1 - y) * dst_step;
        for (int x = 0; x < src_width; )
        {
            const size_t vl = RVV_U8M4::setvl(src_width - x);
            auto indices = __riscv_vrsub(RVV_U16M8::vid(vl), vl - 1, vl);
            auto v = RVV_U8M4::vload(src + x, vl);
            RVV_U8M4::vstore(dst + src_width - x - vl,
                             __riscv_vrgatherei16(v, indices, vl), vl);
            x += static_cast<int>(vl);
        }
    }
}

static void rotate90_32u(const uchar* src_data, size_t src_step, int src_width, int src_height,
                         uchar* dst_data, size_t dst_step, int angle)
{
    const size_t src_step_elems = src_step / sizeof(uint32_t);
    int y = 0;
    for (; y <= src_height - 4; y += 4)
    {
        const uint32_t* src = reinterpret_cast<const uint32_t*>(src_data) + y * src_step_elems;
        for (int x = 0; x < src_width; )
        {
            const size_t vl = __riscv_vsetvl_e32m1(src_width - x);
            auto v0 = __riscv_vle32_v_u32m1(src + x, vl);
            auto v1 = __riscv_vle32_v_u32m1(src + src_step_elems + x, vl);
            auto v2 = __riscv_vle32_v_u32m1(src + 2 * src_step_elems + x, vl);
            auto v3 = __riscv_vle32_v_u32m1(src + 3 * src_step_elems + x, vl);
            vuint32m1x4_t v;
            uint32_t* dst;
            ptrdiff_t stride;
            if (angle == 90)
            {
                v = __riscv_vcreate_v_u32m1x4(v3, v2, v1, v0);
                dst = reinterpret_cast<uint32_t*>(dst_data + static_cast<size_t>(x) * dst_step)
                    + src_height - y - 4;
                stride = static_cast<ptrdiff_t>(dst_step);
            }
            else
            {
                v = __riscv_vcreate_v_u32m1x4(v0, v1, v2, v3);
                dst = reinterpret_cast<uint32_t*>(dst_data + static_cast<size_t>(src_width - 1 - x) * dst_step) + y;
                stride = -static_cast<ptrdiff_t>(dst_step);
            }
            __riscv_vssseg4e32(dst, stride, v, vl);
            x += static_cast<int>(vl);
        }
    }
    for (; y < src_height; ++y)
    {
        const uint32_t* src = reinterpret_cast<const uint32_t*>(src_data + static_cast<size_t>(y) * src_step);
        for (int x = 0; x < src_width; )
        {
            const size_t vl = RVV_U32M8::setvl(src_width - x);
            auto v = RVV_U32M8::vload(src + x, vl);
            uint32_t* dst;
            ptrdiff_t stride;
            if (angle == 90)
            {
                dst = reinterpret_cast<uint32_t*>(dst_data + static_cast<size_t>(x) * dst_step)
                    + src_height - 1 - y;
                stride = static_cast<ptrdiff_t>(dst_step);
            }
            else
            {
                dst = reinterpret_cast<uint32_t*>(dst_data + static_cast<size_t>(src_width - 1 - x) * dst_step) + y;
                stride = -static_cast<ptrdiff_t>(dst_step);
            }
            RVV_U32M8::vstore_stride(dst, stride, v, vl);
            x += static_cast<int>(vl);
        }
    }
}

#undef CV_HAL_RVV_ROTATE_8_ROWS
#undef CV_HAL_RVV_ROTATE_CREATE_x4
#undef CV_HAL_RVV_ROTATE_CREATE_x8

} // namespace

int rotate90(int src_type, const uchar* src_data, size_t src_step, int src_width, int src_height,
             uchar* dst_data, size_t dst_step, int angle)
{
    if (src_data == dst_data || (angle != 90 && angle != 180 && angle != 270))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    const size_t element_size = CV_ELEM_SIZE(src_type);
    if (angle == 180)
    {
        // flip's 8-bit gather uses 8-bit indices, which wrap when VLMAX is greater than 256.
        if (element_size != 1 || RVV_U8M8::setvlmax() <= 256)
            return flip(src_type, src_data, src_step, src_width, src_height,
                        dst_data, dst_step, -1);
        rotate180_8u(src_data, src_step, src_width, src_height, dst_data, dst_step, angle);
        return CV_HAL_ERROR_OK;
    }

    using RotateFunc = void (*)(const uchar*, size_t, int, int, uchar*, size_t, int);
    static RotateFunc rotate90Funcs[] = {
        nullptr, rotate90_8u, rotate90_16u, nullptr, rotate90_32u,
        nullptr, nullptr, nullptr, rotate90_64u
    };
    RotateFunc func = element_size < sizeof(rotate90Funcs) / sizeof(rotate90Funcs[0])
        ? rotate90Funcs[element_size] : nullptr;
    if (!func)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    func(src_data, src_step, src_width, src_height, dst_data, dst_step, angle);
    return CV_HAL_ERROR_OK;
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::core
