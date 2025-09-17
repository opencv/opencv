// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#include "rvv_hal.hpp"
#include "common.hpp"

namespace cv { namespace rvv_hal { namespace imgproc {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

struct Morph2D
{
    int operation;
    int src_type;
    int dst_type;
    int kernel_type;
    uchar* kernel_data;
    size_t kernel_step;
    int kernel_width;
    int kernel_height;
    int anchor_x;
    int anchor_y;
    int borderType;
    const uchar* borderValue;
};

template<int op> struct rvv;
template<> struct rvv<CV_HAL_MORPH_ERODE>
{
    static inline uchar init() { return std::numeric_limits<uchar>::max(); }
    static inline uchar mop(uchar a, uchar b) { return a < b ? a : b; }
    static inline vuint8m4_t vop(vuint8m4_t a, vuint8m4_t b, size_t c) { return __riscv_vminu(a, b, c); }
    static inline vuint8m4_t vop(vuint8m4_t a, uchar b, size_t c) { return __riscv_vminu(a, b, c); }
};
template<> struct rvv<CV_HAL_MORPH_DILATE>
{
    static inline uchar init() { return std::numeric_limits<uchar>::min(); }
    static inline uchar mop(uchar a, uchar b) { return a > b ? a : b; }
    static inline vuint8m4_t vop(vuint8m4_t a, vuint8m4_t b, size_t c) { return __riscv_vmaxu(a, b, c); }
    static inline vuint8m4_t vop(vuint8m4_t a, uchar b, size_t c) { return __riscv_vmaxu(a, b, c); }
};

// the algorithm is copied from 3rdparty/carotene/src/morph.cpp,
// in the function template void morph3x3
template<int op>
static inline int morph(int start, int end, Morph2D* data, const uchar* src_data, size_t src_step, uchar* dst_data, int width, int height, int full_width, int full_height, int offset_x, int offset_y)
{
    bool kernel[9];
    for (int i = 0; i < 9; i++)
    {
        kernel[i] = data->kernel_data[(i / 3) * data->kernel_step + i % 3] != 0;
    }

    constexpr int noval = std::numeric_limits<int>::max();
    auto access = [&](int x, int y) {
        int pi, pj;
        if (data->borderType & BORDER_ISOLATED)
        {
            pi = common::borderInterpolate(x - data->anchor_y, height, data->borderType & ~BORDER_ISOLATED);
            pj = common::borderInterpolate(y - data->anchor_x, width , data->borderType & ~BORDER_ISOLATED);
            pi = pi < 0 ? noval : pi;
            pj = pj < 0 ? noval : pj;
        }
        else
        {
            pi = common::borderInterpolate(offset_y + x - data->anchor_y, full_height, data->borderType);
            pj = common::borderInterpolate(offset_x + y - data->anchor_x, full_width , data->borderType);
            pi = pi < 0 ? noval : pi - offset_y;
            pj = pj < 0 ? noval : pj - offset_x;
        }
        return std::make_pair(pi, pj);
    };

    auto process = [&](int x, int y) {
        if (data->src_type == CV_8UC1)
        {
            uchar val = rvv<op>::init();
            for (int i = 0; i < 9; i++)
            {
                if (kernel[i])
                {
                    auto p = access(x + i / 3, y + i % 3);
                    if (p.first != noval && p.second != noval)
                    {
                        val = rvv<op>::mop(val, src_data[p.first * src_step + p.second]);
                    }
                    else
                    {
                        val = rvv<op>::mop(val, data->borderValue[0]);
                    }
                }
            }
            dst_data[x * width + y] = val;
        }
        else
        {
            uchar val0, val1, val2, val3;
            val0 = val1 = val2 = val3 = rvv<op>::init();
            for (int i = 0; i < 9; i++)
            {
                if (kernel[i])
                {
                    auto p = access(x + i / 3, y + i % 3);
                    if (p.first != noval && p.second != noval)
                    {
                        val0 = rvv<op>::mop(val0, src_data[p.first * src_step + p.second * 4    ]);
                        val1 = rvv<op>::mop(val1, src_data[p.first * src_step + p.second * 4 + 1]);
                        val2 = rvv<op>::mop(val2, src_data[p.first * src_step + p.second * 4 + 2]);
                        val3 = rvv<op>::mop(val3, src_data[p.first * src_step + p.second * 4 + 3]);
                    }
                    else
                    {
                        val0 = rvv<op>::mop(val0, data->borderValue[0]);
                        val1 = rvv<op>::mop(val1, data->borderValue[1]);
                        val2 = rvv<op>::mop(val2, data->borderValue[2]);
                        val3 = rvv<op>::mop(val3, data->borderValue[3]);
                    }
                }
            }
            dst_data[(x * width + y) * 4    ] = val0;
            dst_data[(x * width + y) * 4 + 1] = val1;
            dst_data[(x * width + y) * 4 + 2] = val2;
            dst_data[(x * width + y) * 4 + 3] = val3;
        }
    };

    const int left = data->anchor_x, right = width - (2 - data->anchor_x);
    for (int i = start; i < end; i++)
    {
        if (left >= right)
        {
            for (int j = 0; j < width; j++)
                process(i, j);
        }
        else
        {
            for (int j = 0; j < left; j++)
                process(i, j);
            for (int j = right; j < width; j++)
                process(i, j);

            const uchar* row0 = access(i    , 0).first == noval ? nullptr : src_data + access(i    , 0).first * src_step;
            const uchar* row1 = access(i + 1, 0).first == noval ? nullptr : src_data + access(i + 1, 0).first * src_step;
            const uchar* row2 = access(i + 2, 0).first == noval ? nullptr : src_data + access(i + 2, 0).first * src_step;
            if (data->src_type == CV_8UC1)
            {
                int vl;
                for (int j = left; j < right; j += vl)
                {
                    vl = __riscv_vsetvl_e8m4(right - j);
                    auto m0 = __riscv_vmv_v_x_u8m4(rvv<op>::init(), vl);
                    auto loadsrc = [&](const uchar* row, bool k0, bool k1, bool k2) {
                        if (!row)
                        {
                            m0 = rvv<op>::vop(m0, data->borderValue[0], vl);
                            return;
                        }

                        const uchar* extra = row + j - data->anchor_x;
                        auto v0 = __riscv_vle8_v_u8m4(extra, vl);

                        if (k0) m0 = rvv<op>::vop(m0, v0, vl);
                        v0 = __riscv_vslide1down(v0, extra[vl], vl);
                        if (k1) m0 = rvv<op>::vop(m0, v0, vl);
                        if (!k2) return;
                        v0 = __riscv_vslide1down(v0, extra[vl + 1], vl);
                        m0 = rvv<op>::vop(m0, v0, vl);
                    };

                    loadsrc(row0, kernel[0], kernel[1], kernel[2]);
                    loadsrc(row1, kernel[3], kernel[4], kernel[5]);
                    loadsrc(row2, kernel[6], kernel[7], kernel[8]);
                    __riscv_vse8(dst_data + i * width + j, m0, vl);
                }
            }
            else
            {
                int vl, vl0, vl1;
                for (int j = left; j < right; j += vl)
                {
                    vl = __riscv_vsetvl_e8m4(right - j);
                    vl0 = std::min(vl, (int)__riscv_vlenb() * 2);
                    vl1 = vl - vl0;
                    auto m0 = __riscv_vmv_v_x_u8m4(rvv<op>::init(), vl);
                    auto m1 = __riscv_vmv_v_x_u8m4(rvv<op>::init(), vl);
                    auto m2 = __riscv_vmv_v_x_u8m4(rvv<op>::init(), vl);
                    auto m3 = __riscv_vmv_v_x_u8m4(rvv<op>::init(), vl);

                    auto opshift = [&](vuint8m4_t a, vuint8m4_t b, bool k0, bool k1, bool k2, uchar r1, uchar r2) {
                        if (k0) a = rvv<op>::vop(a, b, vl);
                        b = __riscv_vslide1down(b, r1, vl);
                        if (k1) a = rvv<op>::vop(a, b, vl);
                        if (!k2) return a;
                        b = __riscv_vslide1down(b, r2, vl);
                        return rvv<op>::vop(a, b, vl);
                    };
                    auto loadsrc = [&](const uchar* row, bool k0, bool k1, bool k2) {
                        if (!row)
                        {
                            m0 = rvv<op>::vop(m0, data->borderValue[0], vl);
                            m1 = rvv<op>::vop(m1, data->borderValue[1], vl);
                            m2 = rvv<op>::vop(m2, data->borderValue[2], vl);
                            m3 = rvv<op>::vop(m3, data->borderValue[3], vl);
                            return;
                        }

                        vuint8m4_t v0{}, v1{}, v2{}, v3{};
                        const uchar* extra = row + (j - data->anchor_x) * 4;
                        auto src = __riscv_vlseg4e8_v_u8m2x4(extra, vl0);
                        v0 = __riscv_vset_v_u8m2_u8m4(v0, 0, __riscv_vget_v_u8m2x4_u8m2(src, 0));
                        v1 = __riscv_vset_v_u8m2_u8m4(v1, 0, __riscv_vget_v_u8m2x4_u8m2(src, 1));
                        v2 = __riscv_vset_v_u8m2_u8m4(v2, 0, __riscv_vget_v_u8m2x4_u8m2(src, 2));
                        v3 = __riscv_vset_v_u8m2_u8m4(v3, 0, __riscv_vget_v_u8m2x4_u8m2(src, 3));
                        src = __riscv_vlseg4e8_v_u8m2x4(extra + vl0 * 4, vl1);
                        v0 = __riscv_vset_v_u8m2_u8m4(v0, 1, __riscv_vget_v_u8m2x4_u8m2(src, 0));
                        v1 = __riscv_vset_v_u8m2_u8m4(v1, 1, __riscv_vget_v_u8m2x4_u8m2(src, 1));
                        v2 = __riscv_vset_v_u8m2_u8m4(v2, 1, __riscv_vget_v_u8m2x4_u8m2(src, 2));
                        v3 = __riscv_vset_v_u8m2_u8m4(v3, 1, __riscv_vget_v_u8m2x4_u8m2(src, 3));

                        extra += vl * 4;
                        m0 = opshift(m0, v0, k0, k1, k2, extra[0], extra[4]);
                        m1 = opshift(m1, v1, k0, k1, k2, extra[1], extra[5]);
                        m2 = opshift(m2, v2, k0, k1, k2, extra[2], extra[6]);
                        m3 = opshift(m3, v3, k0, k1, k2, extra[3], extra[7]);
                    };

                    loadsrc(row0, kernel[0], kernel[1], kernel[2]);
                    loadsrc(row1, kernel[3], kernel[4], kernel[5]);
                    loadsrc(row2, kernel[6], kernel[7], kernel[8]);
                    vuint8m2x4_t val{};
                    val = __riscv_vset_v_u8m2_u8m2x4(val, 0, __riscv_vget_v_u8m4_u8m2(m0, 0));
                    val = __riscv_vset_v_u8m2_u8m2x4(val, 1, __riscv_vget_v_u8m4_u8m2(m1, 0));
                    val = __riscv_vset_v_u8m2_u8m2x4(val, 2, __riscv_vget_v_u8m4_u8m2(m2, 0));
                    val = __riscv_vset_v_u8m2_u8m2x4(val, 3, __riscv_vget_v_u8m4_u8m2(m3, 0));
                    __riscv_vsseg4e8(dst_data + (i * width + j) * 4, val, vl0);
                    val = __riscv_vset_v_u8m2_u8m2x4(val, 0, __riscv_vget_v_u8m4_u8m2(m0, 1));
                    val = __riscv_vset_v_u8m2_u8m2x4(val, 1, __riscv_vget_v_u8m4_u8m2(m1, 1));
                    val = __riscv_vset_v_u8m2_u8m2x4(val, 2, __riscv_vget_v_u8m4_u8m2(m2, 1));
                    val = __riscv_vset_v_u8m2_u8m2x4(val, 3, __riscv_vget_v_u8m4_u8m2(m3, 1));
                    __riscv_vsseg4e8(dst_data + (i * width + j + vl0) * 4, val, vl1);
                }
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

} // anonymous

int morphInit(cvhalFilter2D** context, int operation, int src_type, int dst_type, int /*max_width*/, int /*max_height*/, int kernel_type, uchar* kernel_data, size_t kernel_step, int kernel_width, int kernel_height, int anchor_x, int anchor_y, int borderType, const double borderValue[4], int iterations, bool /*allowSubmatrix*/, bool /*allowInplace*/)
{
    if (kernel_type != CV_8UC1 || src_type != dst_type)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (src_type != CV_8UC1 && src_type != CV_8UC4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (kernel_width != kernel_height || kernel_width != 3)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (iterations != 1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (operation != CV_HAL_MORPH_ERODE && operation != CV_HAL_MORPH_DILATE)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if ((borderType & ~BORDER_ISOLATED) == BORDER_WRAP)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    uchar* borderV;
    if (src_type == CV_8UC1)
    {
        borderV = new uchar{static_cast<uchar>(borderValue[0])};
        if (operation == CV_HAL_MORPH_DILATE && borderValue[0] == DBL_MAX)
            borderV[0] = 0;
    }
    else
    {
        borderV = new uchar[4]{static_cast<uchar>(borderValue[0]), static_cast<uchar>(borderValue[1]), static_cast<uchar>(borderValue[2]), static_cast<uchar>(borderValue[3])};
        if (operation == CV_HAL_MORPH_DILATE)
        {
            if (borderValue[0] == DBL_MAX)
                borderV[0] = 0;
            if (borderValue[1] == DBL_MAX)
                borderV[1] = 0;
            if (borderValue[2] == DBL_MAX)
                borderV[2] = 0;
            if (borderValue[3] == DBL_MAX)
                borderV[3] = 0;
        }
    }

    anchor_x = anchor_x < 0 ? kernel_width  / 2 : anchor_x;
    anchor_y = anchor_y < 0 ? kernel_height / 2 : anchor_y;
    *context = reinterpret_cast<cvhalFilter2D*>(new Morph2D{operation, src_type, dst_type, kernel_type, kernel_data, kernel_step, kernel_width, kernel_height, anchor_x, anchor_y, borderType, borderV});
    return CV_HAL_ERROR_OK;
}

int morph(cvhalFilter2D* context, uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int src_full_width, int src_full_height, int src_roi_x, int src_roi_y, int /*dst_full_width*/, int /*dst_full_height*/, int /*dst_roi_x*/, int /*dst_roi_y*/)
{
    Morph2D* data = reinterpret_cast<Morph2D*>(context);
    int cn = data->src_type == CV_8UC1 ? 1 : 4;
    std::vector<uchar> dst(width * height * cn);

    int res = CV_HAL_ERROR_NOT_IMPLEMENTED;
    switch (data->operation)
    {
    case CV_HAL_MORPH_ERODE:
        res = common::invoke(height, {morph<CV_HAL_MORPH_ERODE>}, data, src_data, src_step, dst.data(), width, height, src_full_width, src_full_height, src_roi_x, src_roi_y);
        break;
    case CV_HAL_MORPH_DILATE:
        res = common::invoke(height, {morph<CV_HAL_MORPH_DILATE>}, data, src_data, src_step, dst.data(), width, height, src_full_width, src_full_height, src_roi_x, src_roi_y);
        break;
    }

    for (int i = 0; i < height; i++)
        memcpy(dst_data + i * dst_step, dst.data() + i * width * cn, width * cn);
    return res;
}

int morphFree(cvhalFilter2D* context)
{
    delete reinterpret_cast<Morph2D*>(context)->borderValue;
    delete reinterpret_cast<Morph2D*>(context);
    return CV_HAL_ERROR_OK;
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::imgproc
