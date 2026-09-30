// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "rvv_hal.hpp"
#include "rvv_vcreate.hpp"

namespace cv { namespace rvv_hal { namespace core {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

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
CV_HAL_RVV_ROTATE_8_ROWS(rotate90_32u, uint32_t, RVV_U32M8, u32, 32)
CV_HAL_RVV_ROTATE_8_ROWS(rotate90_64u, uint64_t, RVV_U64M8, u64, 64)

#undef CV_HAL_RVV_ROTATE_8_ROWS

} // namespace

int rotate90(int src_type, const uchar* src_data, size_t src_step, int src_width, int src_height,
             uchar* dst_data, size_t dst_step, int angle)
{
    if (src_data == dst_data || (angle != 90 && angle != 270))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    const size_t element_size = CV_ELEM_SIZE(src_type);
    if (src_step % element_size != 0 || dst_step % element_size != 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

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
