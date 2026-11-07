// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#include "rvv_hal.hpp"
#include "common.hpp"

namespace cv { namespace rvv_hal { namespace imgproc {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

// the algorithm is copied from imgproc/src/median_blur.simd.cpp
// in the function template static void medianBlur_SortNet
template<int ksize, typename helper>
static inline int medianBlurC1(int start, int end, const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height)
{
    using T = typename helper::ElemType;
    using VT = typename helper::VecType;

    for (int i = start; i < end; i++)
    {
        const T* row0 = reinterpret_cast<const T*>(src_data + std::min(std::max(i     - ksize / 2, 0), height - 1) * src_step);
        const T* row1 = reinterpret_cast<const T*>(src_data + std::min(std::max(i + 1 - ksize / 2, 0), height - 1) * src_step);
        const T* row2 = reinterpret_cast<const T*>(src_data + std::min(std::max(i + 2 - ksize / 2, 0), height - 1) * src_step);
        const T* row3 = reinterpret_cast<const T*>(src_data + std::min(std::max(i + 3 - ksize / 2, 0), height - 1) * src_step);
        const T* row4 = reinterpret_cast<const T*>(src_data + std::min(std::max(i + 4 - ksize / 2, 0), height - 1) * src_step);
        int vl;
        auto vop = [&vl](VT& a, VT& b) {
            auto t = a;
            a = helper::vmin(a, b, vl);
            b = helper::vmax(t, b, vl);
        };

        for (int j = 0; j < width; j += vl)
        {
            vl = helper::setvl(width - j);
            if (ksize == 3)
            {
                VT p0, p1, p2;
                VT p3, p4, p5;
                VT p6, p7, p8;
                if (j != 0)
                {
                    p0 = helper::vload(row0 + j - 1, vl);
                    p3 = helper::vload(row1 + j - 1, vl);
                    p6 = helper::vload(row2 + j - 1, vl);
                }
                else
                {
                    p0 = helper::vslide1up(helper::vload(row0, vl), row0[0], vl);
                    p3 = helper::vslide1up(helper::vload(row1, vl), row1[0], vl);
                    p6 = helper::vslide1up(helper::vload(row2, vl), row2[0], vl);
                }
                p1 = helper::vslide1down(p0, row0[j + vl - 1], vl);
                p4 = helper::vslide1down(p3, row1[j + vl - 1], vl);
                p7 = helper::vslide1down(p6, row2[j + vl - 1], vl);
                p2 = helper::vslide1down(p1, row0[std::min(width - 1, j + vl)], vl);
                p5 = helper::vslide1down(p4, row1[std::min(width - 1, j + vl)], vl);
                p8 = helper::vslide1down(p7, row2[std::min(width - 1, j + vl)], vl);

                vop(p1, p2); vop(p4, p5); vop(p7, p8); vop(p0, p1);
                vop(p3, p4); vop(p6, p7); vop(p1, p2); vop(p4, p5);
                vop(p7, p8); vop(p0, p3); vop(p5, p8); vop(p4, p7);
                vop(p3, p6); vop(p1, p4); vop(p2, p5); vop(p4, p7);
                vop(p4, p2); vop(p6, p4); vop(p4, p2);
                helper::vstore(reinterpret_cast<T*>(dst_data + i * dst_step) + j, p4, vl);
            }
            else
            {
                VT p0, p1, p2, p3, p4;
                VT p5, p6, p7, p8, p9;
                VT p10, p11, p12, p13, p14;
                VT p15, p16, p17, p18, p19;
                VT p20, p21, p22, p23, p24;
                if (j >= 2)
                {
                    p0 = helper::vload(row0 + j - 2, vl);
                    p5 = helper::vload(row1 + j - 2, vl);
                    p10 = helper::vload(row2 + j - 2, vl);
                    p15 = helper::vload(row3 + j - 2, vl);
                    p20 = helper::vload(row4 + j - 2, vl);
                }
                else
                {
                    p0 = helper::vslide1up(helper::vload(row0, vl), row0[0], vl);
                    p5 = helper::vslide1up(helper::vload(row1, vl), row1[0], vl);
                    p10 = helper::vslide1up(helper::vload(row2, vl), row2[0], vl);
                    p15 = helper::vslide1up(helper::vload(row3, vl), row3[0], vl);
                    p20 = helper::vslide1up(helper::vload(row4, vl), row4[0], vl);
                    if (j == 0)
                    {
                        p0 = helper::vslide1up(p0, row0[0], vl);
                        p5 = helper::vslide1up(p5, row1[0], vl);
                        p10 = helper::vslide1up(p10, row2[0], vl);
                        p15 = helper::vslide1up(p15, row3[0], vl);
                        p20 = helper::vslide1up(p20, row4[0], vl);
                    }
                }
                p1 = helper::vslide1down(p0, row0[j + vl - 2], vl);
                p6 = helper::vslide1down(p5, row1[j + vl - 2], vl);
                p11 = helper::vslide1down(p10, row2[j + vl - 2], vl);
                p16 = helper::vslide1down(p15, row3[j + vl - 2], vl);
                p21 = helper::vslide1down(p20, row4[j + vl - 2], vl);
                p2 = helper::vslide1down(p1, row0[j + vl - 1], vl);
                p7 = helper::vslide1down(p6, row1[j + vl - 1], vl);
                p12 = helper::vslide1down(p11, row2[j + vl - 1], vl);
                p17 = helper::vslide1down(p16, row3[j + vl - 1], vl);
                p22 = helper::vslide1down(p21, row4[j + vl - 1], vl);
                p3 = helper::vslide1down(p2, row0[std::min(width - 1, j + vl)], vl);
                p8 = helper::vslide1down(p7, row1[std::min(width - 1, j + vl)], vl);
                p13 = helper::vslide1down(p12, row2[std::min(width - 1, j + vl)], vl);
                p18 = helper::vslide1down(p17, row3[std::min(width - 1, j + vl)], vl);
                p23 = helper::vslide1down(p22, row4[std::min(width - 1, j + vl)], vl);
                p4 = helper::vslide1down(p3, row0[std::min(width - 1, j + vl + 1)], vl);
                p9 = helper::vslide1down(p8, row1[std::min(width - 1, j + vl + 1)], vl);
                p14 = helper::vslide1down(p13, row2[std::min(width - 1, j + vl + 1)], vl);
                p19 = helper::vslide1down(p18, row3[std::min(width - 1, j + vl + 1)], vl);
                p24 = helper::vslide1down(p23, row4[std::min(width - 1, j + vl + 1)], vl);

                vop(p1, p2); vop(p0, p1); vop(p1, p2); vop(p4, p5); vop(p3, p4);
                vop(p4, p5); vop(p0, p3); vop(p2, p5); vop(p2, p3); vop(p1, p4);
                vop(p1, p2); vop(p3, p4); vop(p7, p8); vop(p6, p7); vop(p7, p8);
                vop(p10, p11); vop(p9, p10); vop(p10, p11); vop(p6, p9); vop(p8, p11);
                vop(p8, p9); vop(p7, p10); vop(p7, p8); vop(p9, p10); vop(p0, p6);
                vop(p4, p10); vop(p4, p6); vop(p2, p8); vop(p2, p4); vop(p6, p8);
                vop(p1, p7); vop(p5, p11); vop(p5, p7); vop(p3, p9); vop(p3, p5);
                vop(p7, p9); vop(p1, p2); vop(p3, p4); vop(p5, p6); vop(p7, p8);
                vop(p9, p10); vop(p13, p14); vop(p12, p13); vop(p13, p14); vop(p16, p17);
                vop(p15, p16); vop(p16, p17); vop(p12, p15); vop(p14, p17); vop(p14, p15);
                vop(p13, p16); vop(p13, p14); vop(p15, p16); vop(p19, p20); vop(p18, p19);
                vop(p19, p20); vop(p21, p22); vop(p23, p24); vop(p21, p23); vop(p22, p24);
                vop(p22, p23); vop(p18, p21); vop(p20, p23); vop(p20, p21); vop(p19, p22);
                vop(p22, p24); vop(p19, p20); vop(p21, p22); vop(p23, p24); vop(p12, p18);
                vop(p16, p22); vop(p16, p18); vop(p14, p20); vop(p20, p24); vop(p14, p16);
                vop(p18, p20); vop(p22, p24); vop(p13, p19); vop(p17, p23); vop(p17, p19);
                vop(p15, p21); vop(p15, p17); vop(p19, p21); vop(p13, p14); vop(p15, p16);
                vop(p17, p18); vop(p19, p20); vop(p21, p22); vop(p23, p24); vop(p0, p12);
                vop(p8, p20); vop(p8, p12); vop(p4, p16); vop(p16, p24); vop(p12, p16);
                vop(p2, p14); vop(p10, p22); vop(p10, p14); vop(p6, p18); vop(p6, p10);
                vop(p10, p12); vop(p1, p13); vop(p9, p21); vop(p9, p13); vop(p5, p17);
                vop(p13, p17); vop(p3, p15); vop(p11, p23); vop(p11, p15); vop(p7, p19);
                vop(p7, p11); vop(p11, p13); vop(p11, p12);
                helper::vstore(reinterpret_cast<T*>(dst_data + i * dst_step) + j, p12, vl);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

template<int ksize>
static inline int medianBlurC4(int start, int end, const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height)
{
    for (int i = start; i < end; i++)
    {
        const uchar* row0 = src_data + std::min(std::max(i     - ksize / 2, 0), height - 1) * src_step;
        const uchar* row1 = src_data + std::min(std::max(i + 1 - ksize / 2, 0), height - 1) * src_step;
        const uchar* row2 = src_data + std::min(std::max(i + 2 - ksize / 2, 0), height - 1) * src_step;
        const uchar* row3 = src_data + std::min(std::max(i + 3 - ksize / 2, 0), height - 1) * src_step;
        const uchar* row4 = src_data + std::min(std::max(i + 4 - ksize / 2, 0), height - 1) * src_step;
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            if (ksize == 3)
            {
                vl = __riscv_vsetvl_e8m1(width - j);
                vuint8m1_t p00, p01, p02;
                vuint8m1_t p03, p04, p05;
                vuint8m1_t p06, p07, p08;
                vuint8m1_t p10, p11, p12;
                vuint8m1_t p13, p14, p15;
                vuint8m1_t p16, p17, p18;
                vuint8m1_t p20, p21, p22;
                vuint8m1_t p23, p24, p25;
                vuint8m1_t p26, p27, p28;
                vuint8m1_t p30, p31, p32;
                vuint8m1_t p33, p34, p35;
                vuint8m1_t p36, p37, p38;
                auto loadsrc = [&vl](const uchar* row, vuint8m1_t& p0, vuint8m1_t& p1, vuint8m1_t& p2, vuint8m1_t& p3) {
                    auto src = __riscv_vlseg4e8_v_u8m1x4(row, vl);
                    p0 = __riscv_vget_v_u8m1x4_u8m1(src, 0);
                    p1 = __riscv_vget_v_u8m1x4_u8m1(src, 1);
                    p2 = __riscv_vget_v_u8m1x4_u8m1(src, 2);
                    p3 = __riscv_vget_v_u8m1x4_u8m1(src, 3);
                };
                if (j != 0)
                {
                    loadsrc(row0 + (j - 1) * 4, p00, p10, p20, p30);
                    loadsrc(row1 + (j - 1) * 4, p03, p13, p23, p33);
                    loadsrc(row2 + (j - 1) * 4, p06, p16, p26, p36);
                }
                else
                {
                    loadsrc(row0, p00, p10, p20, p30);
                    loadsrc(row1, p03, p13, p23, p33);
                    loadsrc(row2, p06, p16, p26, p36);
                    p00 = __riscv_vslide1up(p00, row0[0], vl);
                    p10 = __riscv_vslide1up(p10, row0[1], vl);
                    p20 = __riscv_vslide1up(p20, row0[2], vl);
                    p30 = __riscv_vslide1up(p30, row0[3], vl);
                    p03 = __riscv_vslide1up(p03, row1[0], vl);
                    p13 = __riscv_vslide1up(p13, row1[1], vl);
                    p23 = __riscv_vslide1up(p23, row1[2], vl);
                    p33 = __riscv_vslide1up(p33, row1[3], vl);
                    p06 = __riscv_vslide1up(p06, row2[0], vl);
                    p16 = __riscv_vslide1up(p16, row2[1], vl);
                    p26 = __riscv_vslide1up(p26, row2[2], vl);
                    p36 = __riscv_vslide1up(p36, row2[3], vl);
                }
                p01 = __riscv_vslide1down(p00, row0[(j + vl - 1) * 4    ], vl);
                p11 = __riscv_vslide1down(p10, row0[(j + vl - 1) * 4 + 1], vl);
                p21 = __riscv_vslide1down(p20, row0[(j + vl - 1) * 4 + 2], vl);
                p31 = __riscv_vslide1down(p30, row0[(j + vl - 1) * 4 + 3], vl);
                p04 = __riscv_vslide1down(p03, row1[(j + vl - 1) * 4    ], vl);
                p14 = __riscv_vslide1down(p13, row1[(j + vl - 1) * 4 + 1], vl);
                p24 = __riscv_vslide1down(p23, row1[(j + vl - 1) * 4 + 2], vl);
                p34 = __riscv_vslide1down(p33, row1[(j + vl - 1) * 4 + 3], vl);
                p07 = __riscv_vslide1down(p06, row2[(j + vl - 1) * 4    ], vl);
                p17 = __riscv_vslide1down(p16, row2[(j + vl - 1) * 4 + 1], vl);
                p27 = __riscv_vslide1down(p26, row2[(j + vl - 1) * 4 + 2], vl);
                p37 = __riscv_vslide1down(p36, row2[(j + vl - 1) * 4 + 3], vl);
                p02 = __riscv_vslide1down(p01, row0[std::min(width - 1, j + vl) * 4    ], vl);
                p12 = __riscv_vslide1down(p11, row0[std::min(width - 1, j + vl) * 4 + 1], vl);
                p22 = __riscv_vslide1down(p21, row0[std::min(width - 1, j + vl) * 4 + 2], vl);
                p32 = __riscv_vslide1down(p31, row0[std::min(width - 1, j + vl) * 4 + 3], vl);
                p05 = __riscv_vslide1down(p04, row1[std::min(width - 1, j + vl) * 4    ], vl);
                p15 = __riscv_vslide1down(p14, row1[std::min(width - 1, j + vl) * 4 + 1], vl);
                p25 = __riscv_vslide1down(p24, row1[std::min(width - 1, j + vl) * 4 + 2], vl);
                p35 = __riscv_vslide1down(p34, row1[std::min(width - 1, j + vl) * 4 + 3], vl);
                p08 = __riscv_vslide1down(p07, row2[std::min(width - 1, j + vl) * 4    ], vl);
                p18 = __riscv_vslide1down(p17, row2[std::min(width - 1, j + vl) * 4 + 1], vl);
                p28 = __riscv_vslide1down(p27, row2[std::min(width - 1, j + vl) * 4 + 2], vl);
                p38 = __riscv_vslide1down(p37, row2[std::min(width - 1, j + vl) * 4 + 3], vl);

                auto vop = [&vl](vuint8m1_t& a, vuint8m1_t& b) {
                    auto t = a;
                    a = __riscv_vminu(a, b, vl);
                    b = __riscv_vmaxu(t, b, vl);
                };
                vuint8m1x4_t dst{};
                vop(p01, p02); vop(p04, p05); vop(p07, p08); vop(p00, p01);
                vop(p03, p04); vop(p06, p07); vop(p01, p02); vop(p04, p05);
                vop(p07, p08); vop(p00, p03); vop(p05, p08); vop(p04, p07);
                vop(p03, p06); vop(p01, p04); vop(p02, p05); vop(p04, p07);
                vop(p04, p02); vop(p06, p04); vop(p04, p02);
                dst = __riscv_vset_v_u8m1_u8m1x4(dst, 0, p04);
                vop(p11, p12); vop(p14, p15); vop(p17, p18); vop(p10, p11);
                vop(p13, p14); vop(p16, p17); vop(p11, p12); vop(p14, p15);
                vop(p17, p18); vop(p10, p13); vop(p15, p18); vop(p14, p17);
                vop(p13, p16); vop(p11, p14); vop(p12, p15); vop(p14, p17);
                vop(p14, p12); vop(p16, p14); vop(p14, p12);
                dst = __riscv_vset_v_u8m1_u8m1x4(dst, 1, p14);
                vop(p21, p22); vop(p24, p25); vop(p27, p28); vop(p20, p21);
                vop(p23, p24); vop(p26, p27); vop(p21, p22); vop(p24, p25);
                vop(p27, p28); vop(p20, p23); vop(p25, p28); vop(p24, p27);
                vop(p23, p26); vop(p21, p24); vop(p22, p25); vop(p24, p27);
                vop(p24, p22); vop(p26, p24); vop(p24, p22);
                dst = __riscv_vset_v_u8m1_u8m1x4(dst, 2, p24);
                vop(p31, p32); vop(p34, p35); vop(p37, p38); vop(p30, p31);
                vop(p33, p34); vop(p36, p37); vop(p31, p32); vop(p34, p35);
                vop(p37, p38); vop(p30, p33); vop(p35, p38); vop(p34, p37);
                vop(p33, p36); vop(p31, p34); vop(p32, p35); vop(p34, p37);
                vop(p34, p32); vop(p36, p34); vop(p34, p32);
                dst = __riscv_vset_v_u8m1_u8m1x4(dst, 3, p34);
                __riscv_vsseg4e8(dst_data + i * dst_step + j * 4, dst, vl);
            }
            else
            {
                vl = __riscv_vsetvl_e8m2(width - j);
                vuint8m2_t p00, p01, p02, p03, p04;
                vuint8m2_t p05, p06, p07, p08, p09;
                vuint8m2_t p010, p011, p012, p013, p014;
                vuint8m2_t p015, p016, p017, p018, p019;
                vuint8m2_t p020, p021, p022, p023, p024;
                vuint8m2_t p10, p11, p12, p13, p14;
                vuint8m2_t p15, p16, p17, p18, p19;
                vuint8m2_t p110, p111, p112, p113, p114;
                vuint8m2_t p115, p116, p117, p118, p119;
                vuint8m2_t p120, p121, p122, p123, p124;
                vuint8m2_t p20, p21, p22, p23, p24;
                vuint8m2_t p25, p26, p27, p28, p29;
                vuint8m2_t p210, p211, p212, p213, p214;
                vuint8m2_t p215, p216, p217, p218, p219;
                vuint8m2_t p220, p221, p222, p223, p224;
                vuint8m2_t p30, p31, p32, p33, p34;
                vuint8m2_t p35, p36, p37, p38, p39;
                vuint8m2_t p310, p311, p312, p313, p314;
                vuint8m2_t p315, p316, p317, p318, p319;
                vuint8m2_t p320, p321, p322, p323, p324;
                auto loadsrc = [&vl](const uchar* row, vuint8m2_t& p0, vuint8m2_t& p1, vuint8m2_t& p2, vuint8m2_t& p3) {
                    auto src = __riscv_vlseg4e8_v_u8m2x4(row, vl);
                    p0 = __riscv_vget_v_u8m2x4_u8m2(src, 0);
                    p1 = __riscv_vget_v_u8m2x4_u8m2(src, 1);
                    p2 = __riscv_vget_v_u8m2x4_u8m2(src, 2);
                    p3 = __riscv_vget_v_u8m2x4_u8m2(src, 3);
                };
                if (j >= 2)
                {
                    loadsrc(row0 + (j - 2) * 4, p00, p10, p20, p30);
                    loadsrc(row1 + (j - 2) * 4, p05, p15, p25, p35);
                    loadsrc(row2 + (j - 2) * 4, p010, p110, p210, p310);
                    loadsrc(row3 + (j - 2) * 4, p015, p115, p215, p315);
                    loadsrc(row4 + (j - 2) * 4, p020, p120, p220, p320);
                }
                else
                {
                    loadsrc(row0, p00, p10, p20, p30);
                    loadsrc(row1, p05, p15, p25, p35);
                    loadsrc(row2, p010, p110, p210, p310);
                    loadsrc(row3, p015, p115, p215, p315);
                    loadsrc(row4, p020, p120, p220, p320);
                    auto slideup = [&] {
                        p00 = __riscv_vslide1up(p00, row0[0], vl);
                        p10 = __riscv_vslide1up(p10, row0[1], vl);
                        p20 = __riscv_vslide1up(p20, row0[2], vl);
                        p30 = __riscv_vslide1up(p30, row0[3], vl);
                        p05 = __riscv_vslide1up(p05, row1[0], vl);
                        p15 = __riscv_vslide1up(p15, row1[1], vl);
                        p25 = __riscv_vslide1up(p25, row1[2], vl);
                        p35 = __riscv_vslide1up(p35, row1[3], vl);
                        p010 = __riscv_vslide1up(p010, row2[0], vl);
                        p110 = __riscv_vslide1up(p110, row2[1], vl);
                        p210 = __riscv_vslide1up(p210, row2[2], vl);
                        p310 = __riscv_vslide1up(p310, row2[3], vl);
                        p015 = __riscv_vslide1up(p015, row3[0], vl);
                        p115 = __riscv_vslide1up(p115, row3[1], vl);
                        p215 = __riscv_vslide1up(p215, row3[2], vl);
                        p315 = __riscv_vslide1up(p315, row3[3], vl);
                        p020 = __riscv_vslide1up(p020, row4[0], vl);
                        p120 = __riscv_vslide1up(p120, row4[1], vl);
                        p220 = __riscv_vslide1up(p220, row4[2], vl);
                        p320 = __riscv_vslide1up(p320, row4[3], vl);
                    };
                    slideup();
                    if (j == 0)
                    {
                        slideup();
                    }
                }
                p01 = __riscv_vslide1down(p00, row0[(j + vl - 2) * 4    ], vl);
                p11 = __riscv_vslide1down(p10, row0[(j + vl - 2) * 4 + 1], vl);
                p21 = __riscv_vslide1down(p20, row0[(j + vl - 2) * 4 + 2], vl);
                p31 = __riscv_vslide1down(p30, row0[(j + vl - 2) * 4 + 3], vl);
                p06 = __riscv_vslide1down(p05, row1[(j + vl - 2) * 4    ], vl);
                p16 = __riscv_vslide1down(p15, row1[(j + vl - 2) * 4 + 1], vl);
                p26 = __riscv_vslide1down(p25, row1[(j + vl - 2) * 4 + 2], vl);
                p36 = __riscv_vslide1down(p35, row1[(j + vl - 2) * 4 + 3], vl);
                p011 = __riscv_vslide1down(p010, row2[(j + vl - 2) * 4    ], vl);
                p111 = __riscv_vslide1down(p110, row2[(j + vl - 2) * 4 + 1], vl);
                p211 = __riscv_vslide1down(p210, row2[(j + vl - 2) * 4 + 2], vl);
                p311 = __riscv_vslide1down(p310, row2[(j + vl - 2) * 4 + 3], vl);
                p016 = __riscv_vslide1down(p015, row3[(j + vl - 2) * 4    ], vl);
                p116 = __riscv_vslide1down(p115, row3[(j + vl - 2) * 4 + 1], vl);
                p216 = __riscv_vslide1down(p215, row3[(j + vl - 2) * 4 + 2], vl);
                p316 = __riscv_vslide1down(p315, row3[(j + vl - 2) * 4 + 3], vl);
                p021 = __riscv_vslide1down(p020, row4[(j + vl - 2) * 4    ], vl);
                p121 = __riscv_vslide1down(p120, row4[(j + vl - 2) * 4 + 1], vl);
                p221 = __riscv_vslide1down(p220, row4[(j + vl - 2) * 4 + 2], vl);
                p321 = __riscv_vslide1down(p320, row4[(j + vl - 2) * 4 + 3], vl);
                p02 = __riscv_vslide1down(p01, row0[(j + vl - 1) * 4    ], vl);
                p12 = __riscv_vslide1down(p11, row0[(j + vl - 1) * 4 + 1], vl);
                p22 = __riscv_vslide1down(p21, row0[(j + vl - 1) * 4 + 2], vl);
                p32 = __riscv_vslide1down(p31, row0[(j + vl - 1) * 4 + 3], vl);
                p07 = __riscv_vslide1down(p06, row1[(j + vl - 1) * 4    ], vl);
                p17 = __riscv_vslide1down(p16, row1[(j + vl - 1) * 4 + 1], vl);
                p27 = __riscv_vslide1down(p26, row1[(j + vl - 1) * 4 + 2], vl);
                p37 = __riscv_vslide1down(p36, row1[(j + vl - 1) * 4 + 3], vl);
                p012 = __riscv_vslide1down(p011, row2[(j + vl - 1) * 4    ], vl);
                p112 = __riscv_vslide1down(p111, row2[(j + vl - 1) * 4 + 1], vl);
                p212 = __riscv_vslide1down(p211, row2[(j + vl - 1) * 4 + 2], vl);
                p312 = __riscv_vslide1down(p311, row2[(j + vl - 1) * 4 + 3], vl);
                p017 = __riscv_vslide1down(p016, row3[(j + vl - 1) * 4    ], vl);
                p117 = __riscv_vslide1down(p116, row3[(j + vl - 1) * 4 + 1], vl);
                p217 = __riscv_vslide1down(p216, row3[(j + vl - 1) * 4 + 2], vl);
                p317 = __riscv_vslide1down(p316, row3[(j + vl - 1) * 4 + 3], vl);
                p022 = __riscv_vslide1down(p021, row4[(j + vl - 1) * 4    ], vl);
                p122 = __riscv_vslide1down(p121, row4[(j + vl - 1) * 4 + 1], vl);
                p222 = __riscv_vslide1down(p221, row4[(j + vl - 1) * 4 + 2], vl);
                p322 = __riscv_vslide1down(p321, row4[(j + vl - 1) * 4 + 3], vl);
                p03 = __riscv_vslide1down(p02, row0[std::min(width - 1, j + vl) * 4    ], vl);
                p13 = __riscv_vslide1down(p12, row0[std::min(width - 1, j + vl) * 4 + 1], vl);
                p23 = __riscv_vslide1down(p22, row0[std::min(width - 1, j + vl) * 4 + 2], vl);
                p33 = __riscv_vslide1down(p32, row0[std::min(width - 1, j + vl) * 4 + 3], vl);
                p08 = __riscv_vslide1down(p07, row1[std::min(width - 1, j + vl) * 4    ], vl);
                p18 = __riscv_vslide1down(p17, row1[std::min(width - 1, j + vl) * 4 + 1], vl);
                p28 = __riscv_vslide1down(p27, row1[std::min(width - 1, j + vl) * 4 + 2], vl);
                p38 = __riscv_vslide1down(p37, row1[std::min(width - 1, j + vl) * 4 + 3], vl);
                p013 = __riscv_vslide1down(p012, row2[std::min(width - 1, j + vl) * 4    ], vl);
                p113 = __riscv_vslide1down(p112, row2[std::min(width - 1, j + vl) * 4 + 1], vl);
                p213 = __riscv_vslide1down(p212, row2[std::min(width - 1, j + vl) * 4 + 2], vl);
                p313 = __riscv_vslide1down(p312, row2[std::min(width - 1, j + vl) * 4 + 3], vl);
                p018 = __riscv_vslide1down(p017, row3[std::min(width - 1, j + vl) * 4    ], vl);
                p118 = __riscv_vslide1down(p117, row3[std::min(width - 1, j + vl) * 4 + 1], vl);
                p218 = __riscv_vslide1down(p217, row3[std::min(width - 1, j + vl) * 4 + 2], vl);
                p318 = __riscv_vslide1down(p317, row3[std::min(width - 1, j + vl) * 4 + 3], vl);
                p023 = __riscv_vslide1down(p022, row4[std::min(width - 1, j + vl) * 4    ], vl);
                p123 = __riscv_vslide1down(p122, row4[std::min(width - 1, j + vl) * 4 + 1], vl);
                p223 = __riscv_vslide1down(p222, row4[std::min(width - 1, j + vl) * 4 + 2], vl);
                p323 = __riscv_vslide1down(p322, row4[std::min(width - 1, j + vl) * 4 + 3], vl);
                p04 = __riscv_vslide1down(p03, row0[std::min(width - 1, j + vl + 1) * 4    ], vl);
                p14 = __riscv_vslide1down(p13, row0[std::min(width - 1, j + vl + 1) * 4 + 1], vl);
                p24 = __riscv_vslide1down(p23, row0[std::min(width - 1, j + vl + 1) * 4 + 2], vl);
                p34 = __riscv_vslide1down(p33, row0[std::min(width - 1, j + vl + 1) * 4 + 3], vl);
                p09 = __riscv_vslide1down(p08, row1[std::min(width - 1, j + vl + 1) * 4    ], vl);
                p19 = __riscv_vslide1down(p18, row1[std::min(width - 1, j + vl + 1) * 4 + 1], vl);
                p29 = __riscv_vslide1down(p28, row1[std::min(width - 1, j + vl + 1) * 4 + 2], vl);
                p39 = __riscv_vslide1down(p38, row1[std::min(width - 1, j + vl + 1) * 4 + 3], vl);
                p014 = __riscv_vslide1down(p013, row2[std::min(width - 1, j + vl + 1) * 4    ], vl);
                p114 = __riscv_vslide1down(p113, row2[std::min(width - 1, j + vl + 1) * 4 + 1], vl);
                p214 = __riscv_vslide1down(p213, row2[std::min(width - 1, j + vl + 1) * 4 + 2], vl);
                p314 = __riscv_vslide1down(p313, row2[std::min(width - 1, j + vl + 1) * 4 + 3], vl);
                p019 = __riscv_vslide1down(p018, row3[std::min(width - 1, j + vl + 1) * 4    ], vl);
                p119 = __riscv_vslide1down(p118, row3[std::min(width - 1, j + vl + 1) * 4 + 1], vl);
                p219 = __riscv_vslide1down(p218, row3[std::min(width - 1, j + vl + 1) * 4 + 2], vl);
                p319 = __riscv_vslide1down(p318, row3[std::min(width - 1, j + vl + 1) * 4 + 3], vl);
                p024 = __riscv_vslide1down(p023, row4[std::min(width - 1, j + vl + 1) * 4    ], vl);
                p124 = __riscv_vslide1down(p123, row4[std::min(width - 1, j + vl + 1) * 4 + 1], vl);
                p224 = __riscv_vslide1down(p223, row4[std::min(width - 1, j + vl + 1) * 4 + 2], vl);
                p324 = __riscv_vslide1down(p323, row4[std::min(width - 1, j + vl + 1) * 4 + 3], vl);

                auto vop = [&vl](vuint8m2_t& a, vuint8m2_t& b) {
                    auto t = a;
                    a = __riscv_vminu(a, b, vl);
                    b = __riscv_vmaxu(t, b, vl);
                };
                vuint8m2x4_t dst{};
                vop(p01, p02); vop(p00, p01); vop(p01, p02); vop(p04, p05); vop(p03, p04);
                vop(p04, p05); vop(p00, p03); vop(p02, p05); vop(p02, p03); vop(p01, p04);
                vop(p01, p02); vop(p03, p04); vop(p07, p08); vop(p06, p07); vop(p07, p08);
                vop(p010, p011); vop(p09, p010); vop(p010, p011); vop(p06, p09); vop(p08, p011);
                vop(p08, p09); vop(p07, p010); vop(p07, p08); vop(p09, p010); vop(p00, p06);
                vop(p04, p010); vop(p04, p06); vop(p02, p08); vop(p02, p04); vop(p06, p08);
                vop(p01, p07); vop(p05, p011); vop(p05, p07); vop(p03, p09); vop(p03, p05);
                vop(p07, p09); vop(p01, p02); vop(p03, p04); vop(p05, p06); vop(p07, p08);
                vop(p09, p010); vop(p013, p014); vop(p012, p013); vop(p013, p014); vop(p016, p017);
                vop(p015, p016); vop(p016, p017); vop(p012, p015); vop(p014, p017); vop(p014, p015);
                vop(p013, p016); vop(p013, p014); vop(p015, p016); vop(p019, p020); vop(p018, p019);
                vop(p019, p020); vop(p021, p022); vop(p023, p024); vop(p021, p023); vop(p022, p024);
                vop(p022, p023); vop(p018, p021); vop(p020, p023); vop(p020, p021); vop(p019, p022);
                vop(p022, p024); vop(p019, p020); vop(p021, p022); vop(p023, p024); vop(p012, p018);
                vop(p016, p022); vop(p016, p018); vop(p014, p020); vop(p020, p024); vop(p014, p016);
                vop(p018, p020); vop(p022, p024); vop(p013, p019); vop(p017, p023); vop(p017, p019);
                vop(p015, p021); vop(p015, p017); vop(p019, p021); vop(p013, p014); vop(p015, p016);
                vop(p017, p018); vop(p019, p020); vop(p021, p022); vop(p023, p024); vop(p00, p012);
                vop(p08, p020); vop(p08, p012); vop(p04, p016); vop(p016, p024); vop(p012, p016);
                vop(p02, p014); vop(p010, p022); vop(p010, p014); vop(p06, p018); vop(p06, p010);
                vop(p010, p012); vop(p01, p013); vop(p09, p021); vop(p09, p013); vop(p05, p017);
                vop(p013, p017); vop(p03, p015); vop(p011, p023); vop(p011, p015); vop(p07, p019);
                vop(p07, p011); vop(p011, p013); vop(p011, p012);
                dst = __riscv_vset_v_u8m2_u8m2x4(dst, 0, p012);
                vop(p11, p12); vop(p10, p11); vop(p11, p12); vop(p14, p15); vop(p13, p14);
                vop(p14, p15); vop(p10, p13); vop(p12, p15); vop(p12, p13); vop(p11, p14);
                vop(p11, p12); vop(p13, p14); vop(p17, p18); vop(p16, p17); vop(p17, p18);
                vop(p110, p111); vop(p19, p110); vop(p110, p111); vop(p16, p19); vop(p18, p111);
                vop(p18, p19); vop(p17, p110); vop(p17, p18); vop(p19, p110); vop(p10, p16);
                vop(p14, p110); vop(p14, p16); vop(p12, p18); vop(p12, p14); vop(p16, p18);
                vop(p11, p17); vop(p15, p111); vop(p15, p17); vop(p13, p19); vop(p13, p15);
                vop(p17, p19); vop(p11, p12); vop(p13, p14); vop(p15, p16); vop(p17, p18);
                vop(p19, p110); vop(p113, p114); vop(p112, p113); vop(p113, p114); vop(p116, p117);
                vop(p115, p116); vop(p116, p117); vop(p112, p115); vop(p114, p117); vop(p114, p115);
                vop(p113, p116); vop(p113, p114); vop(p115, p116); vop(p119, p120); vop(p118, p119);
                vop(p119, p120); vop(p121, p122); vop(p123, p124); vop(p121, p123); vop(p122, p124);
                vop(p122, p123); vop(p118, p121); vop(p120, p123); vop(p120, p121); vop(p119, p122);
                vop(p122, p124); vop(p119, p120); vop(p121, p122); vop(p123, p124); vop(p112, p118);
                vop(p116, p122); vop(p116, p118); vop(p114, p120); vop(p120, p124); vop(p114, p116);
                vop(p118, p120); vop(p122, p124); vop(p113, p119); vop(p117, p123); vop(p117, p119);
                vop(p115, p121); vop(p115, p117); vop(p119, p121); vop(p113, p114); vop(p115, p116);
                vop(p117, p118); vop(p119, p120); vop(p121, p122); vop(p123, p124); vop(p10, p112);
                vop(p18, p120); vop(p18, p112); vop(p14, p116); vop(p116, p124); vop(p112, p116);
                vop(p12, p114); vop(p110, p122); vop(p110, p114); vop(p16, p118); vop(p16, p110);
                vop(p110, p112); vop(p11, p113); vop(p19, p121); vop(p19, p113); vop(p15, p117);
                vop(p113, p117); vop(p13, p115); vop(p111, p123); vop(p111, p115); vop(p17, p119);
                vop(p17, p111); vop(p111, p113); vop(p111, p112);
                dst = __riscv_vset_v_u8m2_u8m2x4(dst, 1, p112);
                vop(p21, p22); vop(p20, p21); vop(p21, p22); vop(p24, p25); vop(p23, p24);
                vop(p24, p25); vop(p20, p23); vop(p22, p25); vop(p22, p23); vop(p21, p24);
                vop(p21, p22); vop(p23, p24); vop(p27, p28); vop(p26, p27); vop(p27, p28);
                vop(p210, p211); vop(p29, p210); vop(p210, p211); vop(p26, p29); vop(p28, p211);
                vop(p28, p29); vop(p27, p210); vop(p27, p28); vop(p29, p210); vop(p20, p26);
                vop(p24, p210); vop(p24, p26); vop(p22, p28); vop(p22, p24); vop(p26, p28);
                vop(p21, p27); vop(p25, p211); vop(p25, p27); vop(p23, p29); vop(p23, p25);
                vop(p27, p29); vop(p21, p22); vop(p23, p24); vop(p25, p26); vop(p27, p28);
                vop(p29, p210); vop(p213, p214); vop(p212, p213); vop(p213, p214); vop(p216, p217);
                vop(p215, p216); vop(p216, p217); vop(p212, p215); vop(p214, p217); vop(p214, p215);
                vop(p213, p216); vop(p213, p214); vop(p215, p216); vop(p219, p220); vop(p218, p219);
                vop(p219, p220); vop(p221, p222); vop(p223, p224); vop(p221, p223); vop(p222, p224);
                vop(p222, p223); vop(p218, p221); vop(p220, p223); vop(p220, p221); vop(p219, p222);
                vop(p222, p224); vop(p219, p220); vop(p221, p222); vop(p223, p224); vop(p212, p218);
                vop(p216, p222); vop(p216, p218); vop(p214, p220); vop(p220, p224); vop(p214, p216);
                vop(p218, p220); vop(p222, p224); vop(p213, p219); vop(p217, p223); vop(p217, p219);
                vop(p215, p221); vop(p215, p217); vop(p219, p221); vop(p213, p214); vop(p215, p216);
                vop(p217, p218); vop(p219, p220); vop(p221, p222); vop(p223, p224); vop(p20, p212);
                vop(p28, p220); vop(p28, p212); vop(p24, p216); vop(p216, p224); vop(p212, p216);
                vop(p22, p214); vop(p210, p222); vop(p210, p214); vop(p26, p218); vop(p26, p210);
                vop(p210, p212); vop(p21, p213); vop(p29, p221); vop(p29, p213); vop(p25, p217);
                vop(p213, p217); vop(p23, p215); vop(p211, p223); vop(p211, p215); vop(p27, p219);
                vop(p27, p211); vop(p211, p213); vop(p211, p212);
                dst = __riscv_vset_v_u8m2_u8m2x4(dst, 2, p212);
                vop(p31, p32); vop(p30, p31); vop(p31, p32); vop(p34, p35); vop(p33, p34);
                vop(p34, p35); vop(p30, p33); vop(p32, p35); vop(p32, p33); vop(p31, p34);
                vop(p31, p32); vop(p33, p34); vop(p37, p38); vop(p36, p37); vop(p37, p38);
                vop(p310, p311); vop(p39, p310); vop(p310, p311); vop(p36, p39); vop(p38, p311);
                vop(p38, p39); vop(p37, p310); vop(p37, p38); vop(p39, p310); vop(p30, p36);
                vop(p34, p310); vop(p34, p36); vop(p32, p38); vop(p32, p34); vop(p36, p38);
                vop(p31, p37); vop(p35, p311); vop(p35, p37); vop(p33, p39); vop(p33, p35);
                vop(p37, p39); vop(p31, p32); vop(p33, p34); vop(p35, p36); vop(p37, p38);
                vop(p39, p310); vop(p313, p314); vop(p312, p313); vop(p313, p314); vop(p316, p317);
                vop(p315, p316); vop(p316, p317); vop(p312, p315); vop(p314, p317); vop(p314, p315);
                vop(p313, p316); vop(p313, p314); vop(p315, p316); vop(p319, p320); vop(p318, p319);
                vop(p319, p320); vop(p321, p322); vop(p323, p324); vop(p321, p323); vop(p322, p324);
                vop(p322, p323); vop(p318, p321); vop(p320, p323); vop(p320, p321); vop(p319, p322);
                vop(p322, p324); vop(p319, p320); vop(p321, p322); vop(p323, p324); vop(p312, p318);
                vop(p316, p322); vop(p316, p318); vop(p314, p320); vop(p320, p324); vop(p314, p316);
                vop(p318, p320); vop(p322, p324); vop(p313, p319); vop(p317, p323); vop(p317, p319);
                vop(p315, p321); vop(p315, p317); vop(p319, p321); vop(p313, p314); vop(p315, p316);
                vop(p317, p318); vop(p319, p320); vop(p321, p322); vop(p323, p324); vop(p30, p312);
                vop(p38, p320); vop(p38, p312); vop(p34, p316); vop(p316, p324); vop(p312, p316);
                vop(p32, p314); vop(p310, p322); vop(p310, p314); vop(p36, p318); vop(p36, p310);
                vop(p310, p312); vop(p31, p313); vop(p39, p321); vop(p39, p313); vop(p35, p317);
                vop(p313, p317); vop(p33, p315); vop(p311, p323); vop(p311, p315); vop(p37, p319);
                vop(p37, p311); vop(p311, p313); vop(p311, p312);
                dst = __riscv_vset_v_u8m2_u8m2x4(dst, 3, p312);
                __riscv_vsseg4e8(dst_data + i * dst_step + j * 4, dst, vl);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

} // anonymous

int medianBlur(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int depth, int cn, int ksize)
{
    const int type = CV_MAKETYPE(depth, cn);
    if (type != CV_8UC1 && type != CV_8UC4 && type != CV_16UC1 && type != CV_16SC1 && type != CV_32FC1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if ((ksize != 3 && ksize != 5) || src_data == dst_data)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    switch (ksize*100 + type)
    {
    case 300 + CV_8UC1:
        return common::invoke(height, {medianBlurC1<3, RVV_U8M4>}, src_data, src_step, dst_data, dst_step, width, height);
    case 300 + CV_16UC1:
        return common::invoke(height, {medianBlurC1<3, RVV_U16M4>}, src_data, src_step, dst_data, dst_step, width, height);
    case 300 + CV_16SC1:
        return common::invoke(height, {medianBlurC1<3, RVV_I16M4>}, src_data, src_step, dst_data, dst_step, width, height);
    case 300 + CV_32FC1:
        return common::invoke(height, {medianBlurC1<3, RVV_F32M4>}, src_data, src_step, dst_data, dst_step, width, height);
    case 500 + CV_8UC1:
        return common::invoke(height, {medianBlurC1<5, RVV_U8M1>}, src_data, src_step, dst_data, dst_step, width, height);
    case 500 + CV_16UC1:
        return common::invoke(height, {medianBlurC1<5, RVV_U16M1>}, src_data, src_step, dst_data, dst_step, width, height);
    case 500 + CV_16SC1:
        return common::invoke(height, {medianBlurC1<5, RVV_I16M1>}, src_data, src_step, dst_data, dst_step, width, height);
    case 500 + CV_32FC1:
        return common::invoke(height, {medianBlurC1<5, RVV_F32M1>}, src_data, src_step, dst_data, dst_step, width, height);

    case 300 + CV_8UC4:
        return common::invoke(height, {medianBlurC4<3>}, src_data, src_step, dst_data, dst_step, width, height);
    case 500 + CV_8UC4:
        return common::invoke(height, {medianBlurC4<5>}, src_data, src_step, dst_data, dst_step, width, height);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::imgproc
