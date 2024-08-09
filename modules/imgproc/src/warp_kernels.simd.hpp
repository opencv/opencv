// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "opencv2/core/hal/intrin.hpp"

namespace cv{
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

/* Only support bilinear interpolation on 3-channel image for now */
void warpAffineSimdInvoker(Mat &output, const Mat &input, const double M[6],
                           int interpolation, int borderType, const double borderValue[4]);
/* Only support bilinear interpolation on 3-channel image for now */
void warpPerspectiveSimdInvoker(Mat &output, const Mat &input, const double M[9],
                                int interpolation, int borderType, const double borderValue[4]);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace {
static inline int borderInterpolate_fast( int p, int len, int borderType )
{
    if( (unsigned)p < (unsigned)len )
        ;
    else if( borderType == BORDER_REPLICATE )
        p = p < 0 ? 0 : len - 1;
    else if( borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101 )
    {
        int delta = borderType == BORDER_REFLECT_101;
        do
        {
            if( p < 0 )
                p = -p - 1 + delta;
            else
                p = len - 1 - (p - len) - delta;
        }
        while( (unsigned)p >= (unsigned)len );
    }
    else if( borderType == BORDER_WRAP )
    {
        if( p < 0 )
            p -= ((p-len+1)/len)*len;
        if( p >= len )
            p %= len;
    }
    return p;
}

class WarpAffineLinearInvoker : public ParallelLoopBody {
public:
    WarpAffineLinearInvoker(Mat &output_, const Mat &input_, const double M_[6],
                            int borderType_, const double borderValue_[4])
        : output(&output_), input(&input_), borderType(borderType_) {
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(M_[i]);
        }
        for (int i = 0; i < 4; i++) {
            bval[i] = saturate_cast<uint8_t>(borderValue_[i]);
        }

        borderType_x = borderType != BORDER_CONSTANT &&
                borderType != BORDER_TRANSPARENT &&
                input->cols <= 1 ? BORDER_REPLICATE : borderType;
        borderType_y = borderType != BORDER_CONSTANT &&
                borderType != BORDER_TRANSPARENT &&
                input->rows <= 1 ? BORDER_REPLICATE : borderType;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    #if CV_SIMD_SCALABLE
        vlanes_32 = VTraits<v_float32>::vlanes();
        vlanes_16 = VTraits<v_uint16>::vlanes();
    #endif
        for (int i = 0; i < max_vlanes_32; i++) {
            start_indices[i] = static_cast<float>(i);
        }
#endif
    }

    virtual void operator() (const Range& r) const CV_OVERRIDE {
        CV_INSTRUMENT_REGION();

        // impl
        auto *src = input->ptr<const uint8_t>();
        auto *dst = output->ptr<uint8_t>();
        size_t srcstep = input->step, dststep = output->step;
        int srccols = input->cols, srcrows = input->rows;
        int dstcols = output->cols;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
    #if CV_SIMD_SCALABLE
        int unrolling_factor = vlanes_32 * 2;
    #else
        int unrolling_factor = max_unrolling_factor;
    #endif

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(unrolling_factor));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        uint8_t bvalbuf[max_unrolling_factor*3];
        for (int i = 0; i < unrolling_factor; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[unrolling_factor]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[unrolling_factor*2]);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_unrolling_factor],
                src_ix[max_unrolling_factor],
                src_iy[max_unrolling_factor];
        uint8_t pixbuf[max_unrolling_factor*4*3];
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
        uint8x8_t reds = {0, 8, 16, 24, 3, 11, 19, 27},
                  greens = {1, 9, 17, 25, 4, 12, 20, 28},
                  blues = {2, 10, 18, 26, 5, 13, 21, 29};
    #endif
#endif

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            v_float32 dst_x0 = vx_load(start_indices.data());
    #if CV_SIMD_SCALABLE
            v_float32 dst_x1 = v_add(dst_x0, vx_setall_f32(float(vlanes_32)));
    #else
            v_float32 dst_x1 = v_add(dst_x0, vx_setall_f32(float(max_vlanes_32)));
    #endif
            v_float32 M0 = vx_setall_f32(M[0]),
                      M3 = vx_setall_f32(M[3]);
            v_float32 M_x = vx_setall_f32(static_cast<float>(y * M[1] + M[2])),
                      M_y = vx_setall_f32(static_cast<float>(y * M[4] + M[5]));

            for (; x < dstcols - unrolling_factor; x += unrolling_factor) {
                // [TODO] apply halide trick

                v_float32 src_x0 = v_fma(M0, dst_x0, M_x),
                          src_y0 = v_fma(M3, dst_x0, M_y),
                          src_x1 = v_fma(M0, dst_x1, M_x),
                          src_y1 = v_fma(M3, dst_x1, M_y);
                dst_x0 = v_add(dst_x0, delta);
                dst_x1 = v_add(dst_x1, delta);
                v_int32 src_ix0 = v_floor(src_x0),
                        src_iy0 = v_floor(src_y0),
                        src_ix1 = v_floor(src_x1),
                        src_iy1 = v_floor(src_y1);
                v_uint32 mask_0 = v_lt(v_reinterpret_as_u32(src_ix0), inner_scols),
                         mask_1 = v_lt(v_reinterpret_as_u32(src_ix1), inner_scols);
                mask_0 = v_and(mask_0, v_lt(v_reinterpret_as_u32(src_iy0), inner_srows));
                mask_1 = v_and(mask_1, v_lt(v_reinterpret_as_u32(src_iy1), inner_srows));
                v_uint16 inner_mask = v_pack(mask_0, mask_1);

                v_int32 addr_0 = v_fma(v_srcstep, src_iy0, v_mul(src_ix0, three)),
                        addr_1 = v_fma(v_srcstep, src_iy1, v_mul(src_ix1, three));
                vx_store(addr, addr_0);
    #if CV_SIMD_SCALABLE
                vx_store(addr + vlanes_32, addr_1);
    #else
                vx_store(addr + max_vlanes_32, addr_1);
    #endif

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                uint8x8_t p0r, p1r, q0r, q1r, p0g, p1g, q0g, q1g, p0b, p1b, q0b, q1b;
    #endif

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    uint8x8x4_t p0 = {
                        vld1_u8(src + addr[0]),
                        vld1_u8(src + addr[1]),
                        vld1_u8(src + addr[2]),
                        vld1_u8(src + addr[3])
                    };

                    uint8x8x4_t p1 = {
                        vld1_u8(src + addr[4]),
                        vld1_u8(src + addr[5]),
                        vld1_u8(src + addr[6]),
                        vld1_u8(src + addr[7])
                    };

                    uint8x8x4_t q0 = {
                        vld1_u8(src + addr[0] + srcstep),
                        vld1_u8(src + addr[1] + srcstep),
                        vld1_u8(src + addr[2] + srcstep),
                        vld1_u8(src + addr[3] + srcstep)
                    };

                    uint8x8x4_t q1 = {
                        vld1_u8(src + addr[4] + srcstep),
                        vld1_u8(src + addr[5] + srcstep),
                        vld1_u8(src + addr[6] + srcstep),
                        vld1_u8(src + addr[7] + srcstep)
                    };

                    uint32x2_t p0_, p1_, q0_, q1_;

                    p0_ = vreinterpret_u32_u8(vtbl4_u8(p0, reds));
                    p1_ = vreinterpret_u32_u8(vtbl4_u8(p1, reds));
                    q0_ = vreinterpret_u32_u8(vtbl4_u8(q0, reds));
                    q1_ = vreinterpret_u32_u8(vtbl4_u8(q1, reds));

                    p0r = vreinterpret_u8_u32(vtrn1_u32(p0_, p1_)); // vtrn1_u32 and vtrn2_u32 are only available on A64
                    p1r = vreinterpret_u8_u32(vtrn2_u32(p0_, p1_));
                    q0r = vreinterpret_u8_u32(vtrn1_u32(q0_, q1_));
                    q1r = vreinterpret_u8_u32(vtrn2_u32(q0_, q1_));

                    p0_ = vreinterpret_u32_u8(vtbl4_u8(p0, greens));
                    p1_ = vreinterpret_u32_u8(vtbl4_u8(p1, greens));
                    q0_ = vreinterpret_u32_u8(vtbl4_u8(q0, greens));
                    q1_ = vreinterpret_u32_u8(vtbl4_u8(q1, greens));

                    p0g = vreinterpret_u8_u32(vtrn1_u32(p0_, p1_));
                    p1g = vreinterpret_u8_u32(vtrn2_u32(p0_, p1_));
                    q0g = vreinterpret_u8_u32(vtrn1_u32(q0_, q1_));
                    q1g = vreinterpret_u8_u32(vtrn2_u32(q0_, q1_));

                    p0_ = vreinterpret_u32_u8(vtbl4_u8(p0, blues));
                    p1_ = vreinterpret_u32_u8(vtbl4_u8(p1, blues));
                    q0_ = vreinterpret_u32_u8(vtbl4_u8(q0, blues));
                    q1_ = vreinterpret_u32_u8(vtbl4_u8(q1, blues));

                    p0b = vreinterpret_u8_u32(vtrn1_u32(p0_, p1_));
                    p1b = vreinterpret_u8_u32(vtrn2_u32(p0_, p1_));
                    q0b = vreinterpret_u8_u32(vtrn1_u32(q0_, q1_));
                    q1b = vreinterpret_u8_u32(vtrn2_u32(q0_, q1_));
    #else // scalar implementation when neon intrinsics are not available
                    for (int i = 0; i < unrolling_factor; i++) {
                        const uchar* srcptr = src + addr[i];
                        pixbuf[i] = srcptr[0];
                        pixbuf[i + unrolling_factor*4] = srcptr[1];
                        pixbuf[i + unrolling_factor*8] = srcptr[2];

                        pixbuf[i + unrolling_factor] = srcptr[3];
                        pixbuf[i + unrolling_factor*5] = srcptr[4];
                        pixbuf[i + unrolling_factor*9] = srcptr[5];

                        pixbuf[i + unrolling_factor*2] = srcptr[srcstep];
                        pixbuf[i + unrolling_factor*6] = srcptr[srcstep + 1];
                        pixbuf[i + unrolling_factor*10] = srcptr[srcstep + 2];

                        pixbuf[i + unrolling_factor*3] = srcptr[srcstep + 3];
                        pixbuf[i + unrolling_factor*7] = srcptr[srcstep + 4];
                        pixbuf[i + unrolling_factor*11] = srcptr[srcstep + 5];
                    }
    #endif
                } else {
                    if (borderType == BORDER_CONSTANT || borderType == BORDER_TRANSPARENT) {
                        mask_0 = v_lt(v_reinterpret_as_u32(v_add(src_ix0, one)), outer_scols);
                        mask_1 = v_lt(v_reinterpret_as_u32(v_add(src_ix1, one)), outer_scols);
                        mask_0 = v_and(mask_0, v_lt(v_reinterpret_as_u32(v_add(src_iy0, one)), outer_srows));
                        mask_1 = v_and(mask_1, v_lt(v_reinterpret_as_u32(v_add(src_iy1, one)), outer_srows));

                        v_uint16 outer_mask = v_pack(mask_0, mask_1);
                        if (v_reduce_max(outer_mask) == 0) { // all 8 pixels are completely outside of the image
                            if (borderType == BORDER_CONSTANT) {
                                v_store_low(dstptr + x*3, bval_v0);
                                v_store_low(dstptr + x*3 + unrolling_factor, bval_v1);
                                v_store_low(dstptr + x*3 + unrolling_factor*2, bval_v2);
                            }
                            continue;
                        }
                    }

                    vx_store(src_ix, src_ix0);
                    vx_store(src_iy, src_iy0);
    #if CV_SIMD_SCALABLE
                    vx_store(src_ix + vlanes_32, src_ix1);
                    vx_store(src_iy + vlanes_32, src_iy1);
    #else
                    vx_store(src_ix + max_vlanes_32, src_ix1);
                    vx_store(src_iy + max_vlanes_32, src_iy1);
    #endif

                    for (int i = 0; i < unrolling_factor; i++) {
                        int ix = src_ix[i], iy = src_iy[i];
                        #define FETCH_PIXEL_C3(dy, dx, pixbuf_ofs) \
                            if ((((unsigned)(ix + dx) < (unsigned)srccols) & ((unsigned)(iy + dy) < (unsigned)srcrows)) != 0) { \
                                size_t addr_i = addr[i] + dy*srcstep + dx*3; \
                                pixbuf[i + pixbuf_ofs] = src[addr_i]; \
                                pixbuf[i + pixbuf_ofs + unrolling_factor*4] = src[addr_i+1]; \
                                pixbuf[i + pixbuf_ofs + unrolling_factor*8] = src[addr_i+2]; \
                            } else if (borderType == BORDER_CONSTANT) { \
                                pixbuf[i + pixbuf_ofs] = bval[0]; \
                                pixbuf[i + pixbuf_ofs + unrolling_factor*4] = bval[1]; \
                                pixbuf[i + pixbuf_ofs + unrolling_factor*8] = bval[2]; \
                            } else if (borderType == BORDER_TRANSPARENT) { \
                                pixbuf[i + pixbuf_ofs] = dstptr[(x + i)*3]; \
                                pixbuf[i + pixbuf_ofs + unrolling_factor*4] = dstptr[(x + i)*3 + 1]; \
                                pixbuf[i + pixbuf_ofs + unrolling_factor*8] = dstptr[(x + i)*3 + 2]; \
                            } else { \
                                int ix_ = borderInterpolate_fast(ix + dx, srccols, borderType_x); \
                                int iy_ = borderInterpolate_fast(iy + dy, srcrows, borderType_y); \
                                size_t addr_i = iy_*srcstep + ix_*3; \
                                pixbuf[i + pixbuf_ofs] = src[addr_i]; \
                                pixbuf[i + pixbuf_ofs + unrolling_factor*4] = src[addr_i+1]; \
                                pixbuf[i + pixbuf_ofs + unrolling_factor*8] = src[addr_i+2]; \
                            }
                        FETCH_PIXEL_C3(0, 0, 0);
                        FETCH_PIXEL_C3(0, 1, unrolling_factor);
                        FETCH_PIXEL_C3(1, 0, unrolling_factor*2);
                        FETCH_PIXEL_C3(1, 1, unrolling_factor*3);
                    }

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    p0r = vld1_u8(pixbuf);
                    p1r = vld1_u8(pixbuf + 8);
                    q0r = vld1_u8(pixbuf + 16);
                    q1r = vld1_u8(pixbuf + 24);

                    p0g = vld1_u8(pixbuf + 32);
                    p1g = vld1_u8(pixbuf + 32 + 8);
                    q0g = vld1_u8(pixbuf + 32 + 16);
                    q1g = vld1_u8(pixbuf + 32 + 24);

                    p0b = vld1_u8(pixbuf + 64);
                    p1b = vld1_u8(pixbuf + 64 + 8);
                    q0b = vld1_u8(pixbuf + 64 + 16);
                    q1b = vld1_u8(pixbuf + 64 + 24);
    #endif
                }

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16 // [TODO] support risc-v fp16 intrinsics
                v_float16 f0r = v_float16(vcvtq_f16_u16(vmovl_u8(p0r)));
                v_float16 f1r = v_float16(vcvtq_f16_u16(vmovl_u8(p1r)));
                v_float16 f2r = v_float16(vcvtq_f16_u16(vmovl_u8(q0r)));
                v_float16 f3r = v_float16(vcvtq_f16_u16(vmovl_u8(q1r)));

                v_float16 f0g = v_float16(vcvtq_f16_u16(vmovl_u8(p0g)));
                v_float16 f1g = v_float16(vcvtq_f16_u16(vmovl_u8(p1g)));
                v_float16 f2g = v_float16(vcvtq_f16_u16(vmovl_u8(q0g)));
                v_float16 f3g = v_float16(vcvtq_f16_u16(vmovl_u8(q1g)));

                v_float16 f0b = v_float16(vcvtq_f16_u16(vmovl_u8(p0b)));
                v_float16 f1b = v_float16(vcvtq_f16_u16(vmovl_u8(p1b)));
                v_float16 f2b = v_float16(vcvtq_f16_u16(vmovl_u8(q0b)));
                v_float16 f3b = v_float16(vcvtq_f16_u16(vmovl_u8(q1b)));
    #else // Other platforms use fp32 intrinsics for interpolation calculation
        #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 // In case neon fp16 intrinsics are not available; still requires A64
                v_int16 f0r = v_reinterpret_as_s16(v_uint16(vmovl_u8(p0r))),
                        f1r = v_reinterpret_as_s16(v_uint16(vmovl_u8(p1r))),
                        f2r = v_reinterpret_as_s16(v_uint16(vmovl_u8(q0r))),
                        f3r = v_reinterpret_as_s16(v_uint16(vmovl_u8(q1r)));
                v_int16 f0g = v_reinterpret_as_s16(v_uint16(vmovl_u8(p0g))),
                        f1g = v_reinterpret_as_s16(v_uint16(vmovl_u8(p1g))),
                        f2g = v_reinterpret_as_s16(v_uint16(vmovl_u8(q0g))),
                        f3g = v_reinterpret_as_s16(v_uint16(vmovl_u8(q1g)));
                v_int16 f0b = v_reinterpret_as_s16(v_uint16(vmovl_u8(p0b))),
                        f1b = v_reinterpret_as_s16(v_uint16(vmovl_u8(p1b))),
                        f2b = v_reinterpret_as_s16(v_uint16(vmovl_u8(q0b))),
                        f3b = v_reinterpret_as_s16(v_uint16(vmovl_u8(q1b)));
        #else
                v_int16  f0r = v_reinterpret_as_s16(vx_load_expand(pixbuf)),
                         f1r = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor)),
                         f2r = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor*2)),
                         f3r = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor*3));
                v_int16  f0g = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor*4)),
                         f1g = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor*5)),
                         f2g = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor*6)),
                         f3g = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor*7));
                v_int16  f0b = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor*8)),
                         f1b = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor*9)),
                         f2b = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor*10)),
                         f3b = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor*11));
        #endif
                v_float32 f0rl = v_cvt_f32(v_expand_low(f0r)), f0rh = v_cvt_f32(v_expand_high(f0r)),
                          f1rl = v_cvt_f32(v_expand_low(f1r)), f1rh = v_cvt_f32(v_expand_high(f1r)),
                          f2rl = v_cvt_f32(v_expand_low(f2r)), f2rh = v_cvt_f32(v_expand_high(f2r)),
                          f3rl = v_cvt_f32(v_expand_low(f3r)), f3rh = v_cvt_f32(v_expand_high(f3r));
                v_float32 f0gl = v_cvt_f32(v_expand_low(f0g)), f0gh = v_cvt_f32(v_expand_high(f0g)),
                          f1gl = v_cvt_f32(v_expand_low(f1g)), f1gh = v_cvt_f32(v_expand_high(f1g)),
                          f2gl = v_cvt_f32(v_expand_low(f2g)), f2gh = v_cvt_f32(v_expand_high(f2g)),
                          f3gl = v_cvt_f32(v_expand_low(f3g)), f3gh = v_cvt_f32(v_expand_high(f3g));
                v_float32 f0bl = v_cvt_f32(v_expand_low(f0b)), f0bh = v_cvt_f32(v_expand_high(f0b)),
                          f1bl = v_cvt_f32(v_expand_low(f1b)), f1bh = v_cvt_f32(v_expand_high(f1b)),
                          f2bl = v_cvt_f32(v_expand_low(f2b)), f2bh = v_cvt_f32(v_expand_high(f2b)),
                          f3bl = v_cvt_f32(v_expand_low(f3b)), f3bh = v_cvt_f32(v_expand_high(f3b));
    #endif // CV_NEON_AARCH64

                src_x0 = v_sub(src_x0, v_cvt_f32(src_ix0));
                src_y0 = v_sub(src_y0, v_cvt_f32(src_iy0));
                src_x1 = v_sub(src_x1, v_cvt_f32(src_ix1));
                src_y1 = v_sub(src_y1, v_cvt_f32(src_iy1));
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
                v_float16 alpha = v_cvt_f16(src_x0, src_x1),
                          beta = v_cvt_f16(src_y0, src_y1);

                f0r = v_fma(alpha, v_sub(f1r, f0r), f0r);
                f2r = v_fma(alpha, v_sub(f3r, f2r), f2r);

                f0g = v_fma(alpha, v_sub(f1g, f0g), f0g);
                f2g = v_fma(alpha, v_sub(f3g, f2g), f2g);

                f0b = v_fma(alpha, v_sub(f1b, f0b), f0b);
                f2b = v_fma(alpha, v_sub(f3b, f2b), f2b);

                f0r = v_fma(beta,  v_sub(f2r, f0r), f0r);
                f0g = v_fma(beta,  v_sub(f2g, f0g), f0g);
                f0b = v_fma(beta,  v_sub(f2b, f0b), f0b);

                uint8x8x3_t result = {
                    vqmovun_s16(vcvtnq_s16_f16(f0r.val)),
                    vqmovun_s16(vcvtnq_s16_f16(f0g.val)),
                    vqmovun_s16(vcvtnq_s16_f16(f0b.val)),
                };
                vst3_u8(dstptr + x*3, result);
    #else
                v_float32 alphal = src_x0, alphah = src_x1,
                          betal = src_y0, betah = src_y1;

                f0rl = v_fma(alphal, v_sub(f1rl, f0rl), f0rl);
                f0rh = v_fma(alphah, v_sub(f1rh, f0rh), f0rh);
                f2rl = v_fma(alphal, v_sub(f3rl, f2rl), f2rl);
                f2rh = v_fma(alphah, v_sub(f3rh, f2rh), f2rh);

                f0gl = v_fma(alphal, v_sub(f1gl, f0gl), f0gl);
                f0gh = v_fma(alphah, v_sub(f1gh, f0gh), f0gh);
                f2gl = v_fma(alphal, v_sub(f3gl, f2gl), f2gl);
                f2gh = v_fma(alphah, v_sub(f3gh, f2gh), f2gh);

                f0bl = v_fma(alphal, v_sub(f1bl, f0bl), f0bl);
                f0bh = v_fma(alphah, v_sub(f1bh, f0bh), f0bh);
                f2bl = v_fma(alphal, v_sub(f3bl, f2bl), f2bl);
                f2bh = v_fma(alphah, v_sub(f3bh, f2bh), f2bh);

                f0rl = v_fma(betal, v_sub(f2rl, f0rl), f0rl);
                f0rh = v_fma(betah, v_sub(f2rh, f0rh), f0rh);
                f0gl = v_fma(betal, v_sub(f2gl, f0gl), f0gl);
                f0gh = v_fma(betah, v_sub(f2gh, f0gh), f0gh);
                f0bl = v_fma(betal, v_sub(f2bl, f0bl), f0bl);
                f0bh = v_fma(betah, v_sub(f2bh, f0bh), f0bh);

                v_uint16 f0r_u16 = v_pack(v_reinterpret_as_u32(v_round(f0rl)), v_reinterpret_as_u32(v_round(f0rh))),
                         f0g_u16 = v_pack(v_reinterpret_as_u32(v_round(f0gl)), v_reinterpret_as_u32(v_round(f0gh))),
                         f0b_u16 = v_pack(v_reinterpret_as_u32(v_round(f0bl)), v_reinterpret_as_u32(v_round(f0bh)));
                uint16_t tbuf[max_vlanes_16*3];
                v_store_interleave(tbuf, f0r_u16, f0g_u16, f0b_u16);
                v_pack_store(dstptr + x*3, vx_load(tbuf));
        #if CV_SIMD_SCALABLE
                v_pack_store(dstptr + x*3 + vlanes_16, vx_load(tbuf + vlanes_16));
                v_pack_store(dstptr + x*3 + vlanes_16*2, vx_load(tbuf + vlanes_16*2));
        #else
                v_pack_store(dstptr + x*3 + max_vlanes_16, vx_load(tbuf + max_vlanes_16));
                v_pack_store(dstptr + x*3 + max_vlanes_16*2, vx_load(tbuf + max_vlanes_16*2));
        #endif
    #endif // CV_NEON_AARCH64 && CV_SIMD128_FP16
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];
                int ix = (int)floorf(sx), iy = (int)floorf(sy);
                sx -= ix; sy -= iy;
                int p00r, p00g, p00b, p01r, p01g, p01b;
                int p10r, p10g, p10b, p11r, p11g, p11b;
                const uint8_t* srcptr = src + srcstep*iy + ix*3;

                if ((((unsigned)ix < (unsigned)(srccols-1)) &
                     ((unsigned)iy < (unsigned)(srcrows-1))) != 0) {
                    p00r = srcptr[0]; p00g = srcptr[1]; p00b = srcptr[2];
                    p01r = srcptr[3]; p01g = srcptr[4]; p01b = srcptr[5];
                    p10r = srcptr[srcstep + 0]; p10g = srcptr[srcstep + 1]; p10b = srcptr[srcstep + 2];
                    p11r = srcptr[srcstep + 3]; p11g = srcptr[srcstep + 4]; p11b = srcptr[srcstep + 5];
                } else {
                    if ((borderType == BORDER_CONSTANT || borderType == BORDER_TRANSPARENT) &&
                        (((unsigned)(ix+1) >= (unsigned)(srccols+1))|
                         ((unsigned)(iy+1) >= (unsigned)(srcrows+1))) != 0) {
                        if (borderType == BORDER_CONSTANT) {
                            dstptr[x*3] = bval[0];
                            dstptr[x*3+1] = bval[1];
                            dstptr[x*3+2] = bval[2];
                        }
                        continue;
                    }

                    #define FETCH_PIXEL_SCALAR_C3(dy, dx, pxy) \
                        if ((((unsigned)(ix + dx) < (unsigned)srccols) & ((unsigned)(iy + dy) < (unsigned)srcrows)) != 0) { \
                            size_t ofs = dy*srcstep + dx*3; \
                            pxy##r = srcptr[ofs]; \
                            pxy##g = srcptr[ofs+1]; \
                            pxy##b = srcptr[ofs+2]; \
                        } else if (borderType == BORDER_CONSTANT) { \
                            pxy##r = bval[0]; \
                            pxy##g = bval[1]; \
                            pxy##b = bval[2]; \
                        } else if (borderType == BORDER_TRANSPARENT) { \
                            pxy##r = dstptr[x*3]; \
                            pxy##g = dstptr[x*3+1]; \
                            pxy##b = dstptr[x*3+2]; \
                        } else { \
                            int ix_ = borderInterpolate_fast(ix + dx, srccols, borderType_x); \
                            int iy_ = borderInterpolate_fast(iy + dy, srcrows, borderType_y); \
                            size_t glob_ofs = iy_*srcstep + ix_*3; \
                            pxy##r = src[glob_ofs]; \
                            pxy##g = src[glob_ofs+1]; \
                            pxy##b = src[glob_ofs+2]; \
                        }
                    FETCH_PIXEL_SCALAR_C3(0, 0, p00);
                    FETCH_PIXEL_SCALAR_C3(0, 1, p01);
                    FETCH_PIXEL_SCALAR_C3(1, 0, p10);
                    FETCH_PIXEL_SCALAR_C3(1, 1, p11);
                }
                float v0r = p00r + sx*(p01r - p00r);
                float v0g = p00g + sx*(p01g - p00g);
                float v0b = p00b + sx*(p01b - p00b);

                float v1r = p10r + sx*(p11r - p10r);
                float v1g = p10g + sx*(p11g - p10g);
                float v1b = p10b + sx*(p11b - p10b);

                v0r += sy*(v1r - v0r);
                v0g += sy*(v1g - v0g);
                v0b += sy*(v1b - v0b);

                dstptr[x*3] = (uint8_t)(v0r + 0.5f);
                dstptr[x*3+1] = (uint8_t)(v0g + 0.5f);
                dstptr[x*3+2] = (uint8_t)(v0b + 0.5f);
            }
        }
    }

private:
    Mat *output;
    const Mat *input;
    int borderType;
    std::array<float, 6> M;
    std::array<uint8_t, 4> bval;

    int borderType_x;
    int borderType_y;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    static constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
    static constexpr int max_vlanes_16{VTraits<v_uint16>::max_nlanes};
    static constexpr int max_unrolling_factor{max_vlanes_32*2};
    #if CV_SIMD_SCALABLE
    int vlanes_32;
    int vlanes_16;
    #endif
    std::array<float, max_vlanes_32> start_indices;
#endif
};

class warpPerspectiveLinearInvoker : public ParallelLoopBody {
public:
    warpPerspectiveLinearInvoker(Mat &output_, const Mat &input_, const double M_[9],
                                 int borderType_, const double borderValue_[4])
        : output(&output_), input(&input_), borderType(borderType_) {
        double M_max = std::abs(M_[0]);
        for (int i = 1; i < 9; i++) {
            if (M_max < std::abs(M_[i])) {
                M_max = std::abs(M_[i]);
            }
        }
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(M_[i] / M_max);
        }
        // printf("res, M=[");
        // for (int i = 0; i < 9; i++) {
        //     printf("%f ", M[i]);
        // }
        // printf("], M_max=%f\n", M_max);
        for (int i = 0; i < 4; i++) {
            bval[i] = saturate_cast<uint8_t>(borderValue_[i]);
        }

        borderType_x = borderType != BORDER_CONSTANT &&
                borderType != BORDER_TRANSPARENT &&
                input->cols <= 1 ? BORDER_REPLICATE : borderType;
        borderType_y = borderType != BORDER_CONSTANT &&
                borderType != BORDER_TRANSPARENT &&
                input->rows <= 1 ? BORDER_REPLICATE : borderType;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        double longer_side = static_cast<double>(std::max(output->rows, output->cols));
        scale = static_cast<float>(1.f / longer_side);
    #if CV_SIMD_SCALABLE
        vlanes_32 = VTraits<v_float32>::vlanes();
        vlanes_16 = VTraits<v_uint16>::vlanes();
    #endif
        for (int i = 0; i < max_vlanes_32; i++) {
            start_indices[i] = static_cast<float>(i);
        }
#endif
    }

    virtual void operator() (const Range& r) const CV_OVERRIDE {
        CV_INSTRUMENT_REGION();

        // impl
        auto *src = input->ptr<const uint8_t>();
        auto *dst = output->ptr<uint8_t>();
        size_t srcstep = input->step, dststep = output->step;
        int srccols = input->cols, srcrows = input->rows;
        int dstcols = output->cols;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
    #if CV_SIMD_SCALABLE
        int unrolling_factor = vlanes_32 * 2;
    #else
        int unrolling_factor = max_unrolling_factor;
    #endif

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(unrolling_factor));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_float32 one_f32 = vx_setall_f32(1.f), zero_f32 = vx_setzero_f32();
        uint8_t bvalbuf[max_unrolling_factor*3];
        for (int i = 0; i < unrolling_factor; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[unrolling_factor]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[unrolling_factor*2]);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_unrolling_factor],
                src_ix[max_unrolling_factor],
                src_iy[max_unrolling_factor];
        uint8_t pixbuf[max_unrolling_factor*4*3];
        v_float32 v_scale = vx_setall_f32(scale);
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
        uint8x8_t reds = {0, 8, 16, 24, 3, 11, 19, 27},
                  greens = {1, 9, 17, 25, 4, 12, 20, 28},
                  blues = {2, 10, 18, 26, 5, 13, 21, 29};
    #endif
#endif

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;
            float scaled_y = y * scale;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            v_float32 dst_x0 = vx_load(start_indices.data());
    #if CV_SIMD_SCALABLE
            v_float32 dst_x1 = v_add(dst_x0, vx_setall_f32(float(vlanes_32)));
    #else
            v_float32 dst_x1 = v_add(dst_x0, vx_setall_f32(float(max_vlanes_32)));
    #endif
            v_float32 M00 = vx_setall_f32(M[0]),
                      M10 = vx_setall_f32(M[3]),
                      M20 = vx_setall_f32(M[6]);
            v_float32 M_x = vx_setall_f32(static_cast<float>(scaled_y * M[1] + M[2] * scale)),
                      M_y = vx_setall_f32(static_cast<float>(scaled_y * M[4] + M[5] * scale)),
                      M_d = vx_setall_f32(static_cast<float>(scaled_y * M[7] + M[8] * scale));

            for (; x < dstcols - unrolling_factor; x += unrolling_factor) {
                // [TODO] apply halide trick

                v_float32 scaled_dst_x0 = v_mul(v_scale, dst_x0),
                          scaled_dst_x1 = v_mul(v_scale, dst_x1);

                v_float32 src_x0 = v_fma(M00, scaled_dst_x0, M_x), // M00 * x + M01 * y + M02
                          src_y0 = v_fma(M10, scaled_dst_x0, M_y), // M10 * x + M11 * y + M12
                          src_x1 = v_fma(M00, scaled_dst_x1, M_x),
                          src_y1 = v_fma(M10, scaled_dst_x1, M_y);
                v_float32 src_d0 = v_fma(M20, scaled_dst_x0, M_d), // M20 * x + M21 * y + M22
                          src_d1 = v_fma(M20, scaled_dst_x1, M_d);

                src_d0 = v_select(v_ne(src_d0, zero_f32), v_div(one_f32, src_d0), zero_f32);
                src_d1 = v_select(v_ne(src_d1, zero_f32), v_div(one_f32, src_d1), zero_f32);
                src_x0 = v_mul(src_x0, src_d0);
                src_y0 = v_mul(src_y0, src_d0);
                src_x1 = v_mul(src_x1, src_d1);
                src_y1 = v_mul(src_y1, src_d1);

                dst_x0 = v_add(dst_x0, delta);
                dst_x1 = v_add(dst_x1, delta);

                v_int32 src_ix0 = v_floor(src_x0),
                        src_iy0 = v_floor(src_y0),
                        src_ix1 = v_floor(src_x1),
                        src_iy1 = v_floor(src_y1);
                v_uint32 mask_0 = v_lt(v_reinterpret_as_u32(src_ix0), inner_scols),
                         mask_1 = v_lt(v_reinterpret_as_u32(src_ix1), inner_scols);
                mask_0 = v_and(mask_0, v_lt(v_reinterpret_as_u32(src_iy0), inner_srows));
                mask_1 = v_and(mask_1, v_lt(v_reinterpret_as_u32(src_iy1), inner_srows));
                v_uint16 inner_mask = v_pack(mask_0, mask_1);

                v_int32 addr_0 = v_fma(v_srcstep, src_iy0, v_mul(src_ix0, three)),
                        addr_1 = v_fma(v_srcstep, src_iy1, v_mul(src_ix1, three));
                vx_store(addr, addr_0);
    #if CV_SIMD_SCALABLE
                vx_store(addr + vlanes_32, addr_1);
    #else
                vx_store(addr + max_vlanes_32, addr_1);
    #endif

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                uint8x8_t p0r, p1r, q0r, q1r, p0g, p1g, q0g, q1g, p0b, p1b, q0b, q1b;
    #endif

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    uint8x8x4_t p0 = {
                        vld1_u8(src + addr[0]),
                        vld1_u8(src + addr[1]),
                        vld1_u8(src + addr[2]),
                        vld1_u8(src + addr[3])
                    };

                    uint8x8x4_t p1 = {
                        vld1_u8(src + addr[4]),
                        vld1_u8(src + addr[5]),
                        vld1_u8(src + addr[6]),
                        vld1_u8(src + addr[7])
                    };

                    uint8x8x4_t q0 = {
                        vld1_u8(src + addr[0] + srcstep),
                        vld1_u8(src + addr[1] + srcstep),
                        vld1_u8(src + addr[2] + srcstep),
                        vld1_u8(src + addr[3] + srcstep)
                    };

                    uint8x8x4_t q1 = {
                        vld1_u8(src + addr[4] + srcstep),
                        vld1_u8(src + addr[5] + srcstep),
                        vld1_u8(src + addr[6] + srcstep),
                        vld1_u8(src + addr[7] + srcstep)
                    };

                    uint32x2_t p0_, p1_, q0_, q1_;

                    p0_ = vreinterpret_u32_u8(vtbl4_u8(p0, reds));
                    p1_ = vreinterpret_u32_u8(vtbl4_u8(p1, reds));
                    q0_ = vreinterpret_u32_u8(vtbl4_u8(q0, reds));
                    q1_ = vreinterpret_u32_u8(vtbl4_u8(q1, reds));

                    p0r = vreinterpret_u8_u32(vtrn1_u32(p0_, p1_)); // vtrn1_u32 and vtrn2_u32 are only available on A64
                    p1r = vreinterpret_u8_u32(vtrn2_u32(p0_, p1_));
                    q0r = vreinterpret_u8_u32(vtrn1_u32(q0_, q1_));
                    q1r = vreinterpret_u8_u32(vtrn2_u32(q0_, q1_));

                    p0_ = vreinterpret_u32_u8(vtbl4_u8(p0, greens));
                    p1_ = vreinterpret_u32_u8(vtbl4_u8(p1, greens));
                    q0_ = vreinterpret_u32_u8(vtbl4_u8(q0, greens));
                    q1_ = vreinterpret_u32_u8(vtbl4_u8(q1, greens));

                    p0g = vreinterpret_u8_u32(vtrn1_u32(p0_, p1_));
                    p1g = vreinterpret_u8_u32(vtrn2_u32(p0_, p1_));
                    q0g = vreinterpret_u8_u32(vtrn1_u32(q0_, q1_));
                    q1g = vreinterpret_u8_u32(vtrn2_u32(q0_, q1_));

                    p0_ = vreinterpret_u32_u8(vtbl4_u8(p0, blues));
                    p1_ = vreinterpret_u32_u8(vtbl4_u8(p1, blues));
                    q0_ = vreinterpret_u32_u8(vtbl4_u8(q0, blues));
                    q1_ = vreinterpret_u32_u8(vtbl4_u8(q1, blues));

                    p0b = vreinterpret_u8_u32(vtrn1_u32(p0_, p1_));
                    p1b = vreinterpret_u8_u32(vtrn2_u32(p0_, p1_));
                    q0b = vreinterpret_u8_u32(vtrn1_u32(q0_, q1_));
                    q1b = vreinterpret_u8_u32(vtrn2_u32(q0_, q1_));
    #else // scalar implementation when neon intrinsics are not available
                    for (int i = 0; i < unrolling_factor; i++) {
                        const uchar* srcptr = src + addr[i];
                        pixbuf[i] = srcptr[0];
                        pixbuf[i + unrolling_factor*4] = srcptr[1];
                        pixbuf[i + unrolling_factor*8] = srcptr[2];

                        pixbuf[i + unrolling_factor] = srcptr[3];
                        pixbuf[i + unrolling_factor*5] = srcptr[4];
                        pixbuf[i + unrolling_factor*9] = srcptr[5];

                        pixbuf[i + unrolling_factor*2] = srcptr[srcstep];
                        pixbuf[i + unrolling_factor*6] = srcptr[srcstep + 1];
                        pixbuf[i + unrolling_factor*10] = srcptr[srcstep + 2];

                        pixbuf[i + unrolling_factor*3] = srcptr[srcstep + 3];
                        pixbuf[i + unrolling_factor*7] = srcptr[srcstep + 4];
                        pixbuf[i + unrolling_factor*11] = srcptr[srcstep + 5];
                    }
    #endif
                } else {
                    if (borderType == BORDER_CONSTANT || borderType == BORDER_TRANSPARENT) {
                        mask_0 = v_lt(v_reinterpret_as_u32(v_add(src_ix0, one)), outer_scols);
                        mask_1 = v_lt(v_reinterpret_as_u32(v_add(src_ix1, one)), outer_scols);
                        mask_0 = v_and(mask_0, v_lt(v_reinterpret_as_u32(v_add(src_iy0, one)), outer_srows));
                        mask_1 = v_and(mask_1, v_lt(v_reinterpret_as_u32(v_add(src_iy1, one)), outer_srows));

                        v_uint16 outer_mask = v_pack(mask_0, mask_1);
                        if (v_reduce_max(outer_mask) == 0) { // all 8 pixels are completely outside of the image
                            if (borderType == BORDER_CONSTANT) {
                                v_store_low(dstptr + x*3, bval_v0);
                                v_store_low(dstptr + x*3 + unrolling_factor, bval_v1);
                                v_store_low(dstptr + x*3 + unrolling_factor*2, bval_v2);
                            }
                            continue;
                        }
                    }

                    vx_store(src_ix, src_ix0);
                    vx_store(src_iy, src_iy0);
    #if CV_SIMD_SCALABLE
                    vx_store(src_ix + vlanes_32, src_ix1);
                    vx_store(src_iy + vlanes_32, src_iy1);
    #else
                    vx_store(src_ix + max_vlanes_32, src_ix1);
                    vx_store(src_iy + max_vlanes_32, src_iy1);
    #endif

                    for (int i = 0; i < unrolling_factor; i++) {
                        int ix = src_ix[i], iy = src_iy[i];
                        #define FETCH_PIXEL_C3(dy, dx, pixbuf_ofs) \
                            if ((((unsigned)(ix + dx) < (unsigned)srccols) & ((unsigned)(iy + dy) < (unsigned)srcrows)) != 0) { \
                                size_t addr_i = addr[i] + dy*srcstep + dx*3; \
                                pixbuf[i + pixbuf_ofs] = src[addr_i]; \
                                pixbuf[i + pixbuf_ofs + unrolling_factor*4] = src[addr_i+1]; \
                                pixbuf[i + pixbuf_ofs + unrolling_factor*8] = src[addr_i+2]; \
                            } else if (borderType == BORDER_CONSTANT) { \
                                pixbuf[i + pixbuf_ofs] = bval[0]; \
                                pixbuf[i + pixbuf_ofs + unrolling_factor*4] = bval[1]; \
                                pixbuf[i + pixbuf_ofs + unrolling_factor*8] = bval[2]; \
                            } else if (borderType == BORDER_TRANSPARENT) { \
                                pixbuf[i + pixbuf_ofs] = dstptr[(x + i)*3]; \
                                pixbuf[i + pixbuf_ofs + unrolling_factor*4] = dstptr[(x + i)*3 + 1]; \
                                pixbuf[i + pixbuf_ofs + unrolling_factor*8] = dstptr[(x + i)*3 + 2]; \
                            } else { \
                                int ix_ = borderInterpolate_fast(ix + dx, srccols, borderType_x); \
                                int iy_ = borderInterpolate_fast(iy + dy, srcrows, borderType_y); \
                                size_t addr_i = iy_*srcstep + ix_*3; \
                                pixbuf[i + pixbuf_ofs] = src[addr_i]; \
                                pixbuf[i + pixbuf_ofs + unrolling_factor*4] = src[addr_i+1]; \
                                pixbuf[i + pixbuf_ofs + unrolling_factor*8] = src[addr_i+2]; \
                            }
                        FETCH_PIXEL_C3(0, 0, 0);
                        FETCH_PIXEL_C3(0, 1, unrolling_factor);
                        FETCH_PIXEL_C3(1, 0, unrolling_factor*2);
                        FETCH_PIXEL_C3(1, 1, unrolling_factor*3);
                    }

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    p0r = vld1_u8(pixbuf);
                    p1r = vld1_u8(pixbuf + 8);
                    q0r = vld1_u8(pixbuf + 16);
                    q1r = vld1_u8(pixbuf + 24);

                    p0g = vld1_u8(pixbuf + 32);
                    p1g = vld1_u8(pixbuf + 32 + 8);
                    q0g = vld1_u8(pixbuf + 32 + 16);
                    q1g = vld1_u8(pixbuf + 32 + 24);

                    p0b = vld1_u8(pixbuf + 64);
                    p1b = vld1_u8(pixbuf + 64 + 8);
                    q0b = vld1_u8(pixbuf + 64 + 16);
                    q1b = vld1_u8(pixbuf + 64 + 24);
    #endif
                }

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16 // [TODO] support risc-v fp16 intrinsics
                v_float16 f0r = v_float16(vcvtq_f16_u16(vmovl_u8(p0r)));
                v_float16 f1r = v_float16(vcvtq_f16_u16(vmovl_u8(p1r)));
                v_float16 f2r = v_float16(vcvtq_f16_u16(vmovl_u8(q0r)));
                v_float16 f3r = v_float16(vcvtq_f16_u16(vmovl_u8(q1r)));

                v_float16 f0g = v_float16(vcvtq_f16_u16(vmovl_u8(p0g)));
                v_float16 f1g = v_float16(vcvtq_f16_u16(vmovl_u8(p1g)));
                v_float16 f2g = v_float16(vcvtq_f16_u16(vmovl_u8(q0g)));
                v_float16 f3g = v_float16(vcvtq_f16_u16(vmovl_u8(q1g)));

                v_float16 f0b = v_float16(vcvtq_f16_u16(vmovl_u8(p0b)));
                v_float16 f1b = v_float16(vcvtq_f16_u16(vmovl_u8(p1b)));
                v_float16 f2b = v_float16(vcvtq_f16_u16(vmovl_u8(q0b)));
                v_float16 f3b = v_float16(vcvtq_f16_u16(vmovl_u8(q1b)));
    #else // Other platforms use fp32 intrinsics for interpolation calculation
        #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 // In case neon fp16 intrinsics are not available; still requires A64
                v_int16 f0r = v_reinterpret_as_s16(v_uint16(vmovl_u8(p0r))),
                        f1r = v_reinterpret_as_s16(v_uint16(vmovl_u8(p1r))),
                        f2r = v_reinterpret_as_s16(v_uint16(vmovl_u8(q0r))),
                        f3r = v_reinterpret_as_s16(v_uint16(vmovl_u8(q1r)));
                v_int16 f0g = v_reinterpret_as_s16(v_uint16(vmovl_u8(p0g))),
                        f1g = v_reinterpret_as_s16(v_uint16(vmovl_u8(p1g))),
                        f2g = v_reinterpret_as_s16(v_uint16(vmovl_u8(q0g))),
                        f3g = v_reinterpret_as_s16(v_uint16(vmovl_u8(q1g)));
                v_int16 f0b = v_reinterpret_as_s16(v_uint16(vmovl_u8(p0b))),
                        f1b = v_reinterpret_as_s16(v_uint16(vmovl_u8(p1b))),
                        f2b = v_reinterpret_as_s16(v_uint16(vmovl_u8(q0b))),
                        f3b = v_reinterpret_as_s16(v_uint16(vmovl_u8(q1b)));
        #else
                v_int16  f0r = v_reinterpret_as_s16(vx_load_expand(pixbuf)),
                         f1r = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor)),
                         f2r = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor*2)),
                         f3r = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor*3));
                v_int16  f0g = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor*4)),
                         f1g = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor*5)),
                         f2g = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor*6)),
                         f3g = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor*7));
                v_int16  f0b = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor*8)),
                         f1b = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor*9)),
                         f2b = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor*10)),
                         f3b = v_reinterpret_as_s16(vx_load_expand(pixbuf + unrolling_factor*11));
        #endif
                v_float32 f0rl = v_cvt_f32(v_expand_low(f0r)), f0rh = v_cvt_f32(v_expand_high(f0r)),
                          f1rl = v_cvt_f32(v_expand_low(f1r)), f1rh = v_cvt_f32(v_expand_high(f1r)),
                          f2rl = v_cvt_f32(v_expand_low(f2r)), f2rh = v_cvt_f32(v_expand_high(f2r)),
                          f3rl = v_cvt_f32(v_expand_low(f3r)), f3rh = v_cvt_f32(v_expand_high(f3r));
                v_float32 f0gl = v_cvt_f32(v_expand_low(f0g)), f0gh = v_cvt_f32(v_expand_high(f0g)),
                          f1gl = v_cvt_f32(v_expand_low(f1g)), f1gh = v_cvt_f32(v_expand_high(f1g)),
                          f2gl = v_cvt_f32(v_expand_low(f2g)), f2gh = v_cvt_f32(v_expand_high(f2g)),
                          f3gl = v_cvt_f32(v_expand_low(f3g)), f3gh = v_cvt_f32(v_expand_high(f3g));
                v_float32 f0bl = v_cvt_f32(v_expand_low(f0b)), f0bh = v_cvt_f32(v_expand_high(f0b)),
                          f1bl = v_cvt_f32(v_expand_low(f1b)), f1bh = v_cvt_f32(v_expand_high(f1b)),
                          f2bl = v_cvt_f32(v_expand_low(f2b)), f2bh = v_cvt_f32(v_expand_high(f2b)),
                          f3bl = v_cvt_f32(v_expand_low(f3b)), f3bh = v_cvt_f32(v_expand_high(f3b));
    #endif // CV_NEON_AARCH64

                src_x0 = v_sub(src_x0, v_cvt_f32(src_ix0));
                src_y0 = v_sub(src_y0, v_cvt_f32(src_iy0));
                src_x1 = v_sub(src_x1, v_cvt_f32(src_ix1));
                src_y1 = v_sub(src_y1, v_cvt_f32(src_iy1));
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
                v_float16 alpha = v_cvt_f16(src_x0, src_x1),
                          beta = v_cvt_f16(src_y0, src_y1);

                f0r = v_fma(alpha, v_sub(f1r, f0r), f0r);
                f2r = v_fma(alpha, v_sub(f3r, f2r), f2r);

                f0g = v_fma(alpha, v_sub(f1g, f0g), f0g);
                f2g = v_fma(alpha, v_sub(f3g, f2g), f2g);

                f0b = v_fma(alpha, v_sub(f1b, f0b), f0b);
                f2b = v_fma(alpha, v_sub(f3b, f2b), f2b);

                f0r = v_fma(beta,  v_sub(f2r, f0r), f0r);
                f0g = v_fma(beta,  v_sub(f2g, f0g), f0g);
                f0b = v_fma(beta,  v_sub(f2b, f0b), f0b);

                uint8x8x3_t result = {
                    vqmovun_s16(vcvtnq_s16_f16(f0r.val)),
                    vqmovun_s16(vcvtnq_s16_f16(f0g.val)),
                    vqmovun_s16(vcvtnq_s16_f16(f0b.val)),
                };
                vst3_u8(dstptr + x*3, result);
    #else
                v_float32 alphal = src_x0, alphah = src_x1,
                          betal = src_y0, betah = src_y1;

                f0rl = v_fma(alphal, v_sub(f1rl, f0rl), f0rl);
                f0rh = v_fma(alphah, v_sub(f1rh, f0rh), f0rh);
                f2rl = v_fma(alphal, v_sub(f3rl, f2rl), f2rl);
                f2rh = v_fma(alphah, v_sub(f3rh, f2rh), f2rh);

                f0gl = v_fma(alphal, v_sub(f1gl, f0gl), f0gl);
                f0gh = v_fma(alphah, v_sub(f1gh, f0gh), f0gh);
                f2gl = v_fma(alphal, v_sub(f3gl, f2gl), f2gl);
                f2gh = v_fma(alphah, v_sub(f3gh, f2gh), f2gh);

                f0bl = v_fma(alphal, v_sub(f1bl, f0bl), f0bl);
                f0bh = v_fma(alphah, v_sub(f1bh, f0bh), f0bh);
                f2bl = v_fma(alphal, v_sub(f3bl, f2bl), f2bl);
                f2bh = v_fma(alphah, v_sub(f3bh, f2bh), f2bh);

                f0rl = v_fma(betal, v_sub(f2rl, f0rl), f0rl);
                f0rh = v_fma(betah, v_sub(f2rh, f0rh), f0rh);
                f0gl = v_fma(betal, v_sub(f2gl, f0gl), f0gl);
                f0gh = v_fma(betah, v_sub(f2gh, f0gh), f0gh);
                f0bl = v_fma(betal, v_sub(f2bl, f0bl), f0bl);
                f0bh = v_fma(betah, v_sub(f2bh, f0bh), f0bh);

                v_uint16 f0r_u16 = v_pack(v_reinterpret_as_u32(v_round(f0rl)), v_reinterpret_as_u32(v_round(f0rh))),
                         f0g_u16 = v_pack(v_reinterpret_as_u32(v_round(f0gl)), v_reinterpret_as_u32(v_round(f0gh))),
                         f0b_u16 = v_pack(v_reinterpret_as_u32(v_round(f0bl)), v_reinterpret_as_u32(v_round(f0bh)));
                uint16_t tbuf[max_vlanes_16*3];
                v_store_interleave(tbuf, f0r_u16, f0g_u16, f0b_u16);
                v_pack_store(dstptr + x*3, vx_load(tbuf));
        #if CV_SIMD_SCALABLE
                v_pack_store(dstptr + x*3 + vlanes_16, vx_load(tbuf + vlanes_16));
                v_pack_store(dstptr + x*3 + vlanes_16*2, vx_load(tbuf + vlanes_16*2));
        #else
                v_pack_store(dstptr + x*3 + max_vlanes_16, vx_load(tbuf + max_vlanes_16));
                v_pack_store(dstptr + x*3 + max_vlanes_16*2, vx_load(tbuf + max_vlanes_16*2));
        #endif
    #endif // CV_NEON_AARCH64 && CV_SIMD128_FP16
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float scaled_x = x * scale;
                float sx = (scaled_x*M[0] + scaled_y*M[1] + M[2]*scale) / (scaled_x*M[6] + scaled_y*M[7] + M[8]*scale);
                float sy = (scaled_x*M[3] + scaled_y*M[4] + M[5]*scale) / (scaled_x*M[6] + scaled_y*M[7] + M[8]*scale);
                int ix = (int)floorf(sx), iy = (int)floorf(sy);
                sx -= ix; sy -= iy;
                int p00r, p00g, p00b, p01r, p01g, p01b;
                int p10r, p10g, p10b, p11r, p11g, p11b;
                const uint8_t* srcptr = src + srcstep*iy + ix*3;

                if ((((unsigned)ix < (unsigned)(srccols-1)) &
                     ((unsigned)iy < (unsigned)(srcrows-1))) != 0) {
                    p00r = srcptr[0]; p00g = srcptr[1]; p00b = srcptr[2];
                    p01r = srcptr[3]; p01g = srcptr[4]; p01b = srcptr[5];
                    p10r = srcptr[srcstep + 0]; p10g = srcptr[srcstep + 1]; p10b = srcptr[srcstep + 2];
                    p11r = srcptr[srcstep + 3]; p11g = srcptr[srcstep + 4]; p11b = srcptr[srcstep + 5];
                } else {
                    if ((borderType == BORDER_CONSTANT || borderType == BORDER_TRANSPARENT) &&
                        (((unsigned)(ix+1) >= (unsigned)(srccols+1))|
                         ((unsigned)(iy+1) >= (unsigned)(srcrows+1))) != 0) {
                        if (borderType == BORDER_CONSTANT) {
                            dstptr[x*3] = bval[0];
                            dstptr[x*3+1] = bval[1];
                            dstptr[x*3+2] = bval[2];
                        }
                        continue;
                    }

                    #define FETCH_PIXEL_SCALAR_C3(dy, dx, pxy) \
                        if ((((unsigned)(ix + dx) < (unsigned)srccols) & ((unsigned)(iy + dy) < (unsigned)srcrows)) != 0) { \
                            size_t ofs = dy*srcstep + dx*3; \
                            pxy##r = srcptr[ofs]; \
                            pxy##g = srcptr[ofs+1]; \
                            pxy##b = srcptr[ofs+2]; \
                        } else if (borderType == BORDER_CONSTANT) { \
                            pxy##r = bval[0]; \
                            pxy##g = bval[1]; \
                            pxy##b = bval[2]; \
                        } else if (borderType == BORDER_TRANSPARENT) { \
                            pxy##r = dstptr[x*3]; \
                            pxy##g = dstptr[x*3+1]; \
                            pxy##b = dstptr[x*3+2]; \
                        } else { \
                            int ix_ = borderInterpolate_fast(ix + dx, srccols, borderType_x); \
                            int iy_ = borderInterpolate_fast(iy + dy, srcrows, borderType_y); \
                            size_t glob_ofs = iy_*srcstep + ix_*3; \
                            pxy##r = src[glob_ofs]; \
                            pxy##g = src[glob_ofs+1]; \
                            pxy##b = src[glob_ofs+2]; \
                        }
                    FETCH_PIXEL_SCALAR_C3(0, 0, p00);
                    FETCH_PIXEL_SCALAR_C3(0, 1, p01);
                    FETCH_PIXEL_SCALAR_C3(1, 0, p10);
                    FETCH_PIXEL_SCALAR_C3(1, 1, p11);
                }
                float v0r = p00r + sx*(p01r - p00r);
                float v0g = p00g + sx*(p01g - p00g);
                float v0b = p00b + sx*(p01b - p00b);

                float v1r = p10r + sx*(p11r - p10r);
                float v1g = p10g + sx*(p11g - p10g);
                float v1b = p10b + sx*(p11b - p10b);

                v0r += sy*(v1r - v0r);
                v0g += sy*(v1g - v0g);
                v0b += sy*(v1b - v0b);

                dstptr[x*3] = (uint8_t)(v0r + 0.5f);
                dstptr[x*3+1] = (uint8_t)(v0g + 0.5f);
                dstptr[x*3+2] = (uint8_t)(v0b + 0.5f);
            }
        }
    }

private:
    Mat *output;
    const Mat *input;
    int borderType;
    std::array<float, 9> M;
    std::array<uint8_t, 4> bval;

    int borderType_x;
    int borderType_y;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    float scale;
    static constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
    static constexpr int max_vlanes_16{VTraits<v_uint16>::max_nlanes};
    static constexpr int max_unrolling_factor{max_vlanes_32*2};
    #if CV_SIMD_SCALABLE
    int vlanes_32;
    int vlanes_16;
    #endif
    std::array<float, max_vlanes_32> start_indices;
#endif
};

} // anonymous

void warpAffineSimdInvoker(Mat &output, const Mat &input, const double M[6],
                           int interpolation, int borderType, const double borderValue[4]) {
    CV_INSTRUMENT_REGION();

    CV_CheckEQ(interpolation, INTER_LINEAR, "");
    CV_CheckEQ(input.type(), CV_8UC3, "");
    CV_CheckEQ(output.type(), CV_8UC3, "");
    WarpAffineLinearInvoker body(output, input, M, borderType, borderValue);
    parallel_for_(Range(0, output.rows), body);
}
void warpPerspectiveSimdInvoker(Mat &output, const Mat &input, const double M[9],
                           int interpolation, int borderType, const double borderValue[4]) {
    CV_INSTRUMENT_REGION();

    CV_CheckEQ(interpolation, INTER_LINEAR, "");
    CV_CheckEQ(input.type(), CV_8UC3, "");
    CV_CheckEQ(output.type(), CV_8UC3, "");
    warpPerspectiveLinearInvoker body(output, input, M, borderType, borderValue);
    parallel_for_(Range(0, output.rows), body);
}
#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY


CV_CPU_OPTIMIZATION_NAMESPACE_END
} // cv
