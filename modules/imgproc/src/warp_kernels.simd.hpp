// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <numeric>
#include "precomp.hpp"

#include "opencv2/core/hal/intrin.hpp"

#define VECTOR_FETCH_PIXEL_C3(dy, dx, pixbuf_ofs) \
    if ((((unsigned)(ix + dx) < (unsigned)srccols) & ((unsigned)(iy + dy) < (unsigned)srcrows)) != 0) { \
        size_t addr_i = addr[i] + dy*srcstep + dx*3; \
        pixbuf[i + pixbuf_ofs] = src[addr_i]; \
        pixbuf[i + pixbuf_ofs + uf*4] = src[addr_i+1]; \
        pixbuf[i + pixbuf_ofs + uf*8] = src[addr_i+2]; \
    } else if (borderType == BORDER_CONSTANT) { \
        pixbuf[i + pixbuf_ofs] = bval[0]; \
        pixbuf[i + pixbuf_ofs + uf*4] = bval[1]; \
        pixbuf[i + pixbuf_ofs + uf*8] = bval[2]; \
    } else if (borderType == BORDER_TRANSPARENT) { \
        pixbuf[i + pixbuf_ofs] = dstptr[(x + i)*3]; \
        pixbuf[i + pixbuf_ofs + uf*4] = dstptr[(x + i)*3 + 1]; \
        pixbuf[i + pixbuf_ofs + uf*8] = dstptr[(x + i)*3 + 2]; \
    } else { \
        int ix_ = borderInterpolate_fast(ix + dx, srccols, borderType_x); \
        int iy_ = borderInterpolate_fast(iy + dy, srcrows, borderType_y); \
        size_t addr_i = iy_*srcstep + ix_*3; \
        pixbuf[i + pixbuf_ofs] = src[addr_i]; \
        pixbuf[i + pixbuf_ofs + uf*4] = src[addr_i+1]; \
        pixbuf[i + pixbuf_ofs + uf*8] = src[addr_i+2]; \
    }
#define VECTOR_FETCH_PIXEL_C1(dy, dx, pixbuf_ofs) \
    if ((((unsigned)(ix + dx) < (unsigned)srccols) & ((unsigned)(iy + dy) < (unsigned)srcrows)) != 0) { \
        size_t addr_i = addr[i] + dy*srcstep + dx; \
        pixbuf[i + pixbuf_ofs] = src[addr_i]; \
    } else if (borderType == BORDER_CONSTANT) { \
        pixbuf[i + pixbuf_ofs] = bval[0]; \
    } else if (borderType == BORDER_TRANSPARENT) { \
        pixbuf[i + pixbuf_ofs] = dstptr[x + i]; \
    } else { \
        int ix_ = borderInterpolate_fast(ix + dx, srccols, borderType_x); \
        int iy_ = borderInterpolate_fast(iy + dy, srcrows, borderType_y); \
        size_t addr_i = iy_*srcstep + ix_; \
        pixbuf[i + pixbuf_ofs] = src[addr_i]; \
    }
#define VECTOR_FETCH_PIXEL_C4(dy, dx, pixbuf_ofs) \
    if ((((unsigned)(ix + dx) < (unsigned)srccols) & ((unsigned)(iy + dy) < (unsigned)srcrows)) != 0) { \
        size_t addr_i = addr[i] + dy*srcstep + dx*4; \
        pixbuf[i + pixbuf_ofs] = src[addr_i]; \
        pixbuf[i + pixbuf_ofs + uf*4] = src[addr_i+1]; \
        pixbuf[i + pixbuf_ofs + uf*8] = src[addr_i+2]; \
        pixbuf[i + pixbuf_ofs + uf*12] = src[addr_i+3]; \
    } else if (borderType == BORDER_CONSTANT) { \
        pixbuf[i + pixbuf_ofs] = bval[0]; \
        pixbuf[i + pixbuf_ofs + uf*4] = bval[1]; \
        pixbuf[i + pixbuf_ofs + uf*8] = bval[2]; \
        pixbuf[i + pixbuf_ofs + uf*12] = bval[3]; \
    } else if (borderType == BORDER_TRANSPARENT) { \
        pixbuf[i + pixbuf_ofs] = dstptr[(x + i)*4]; \
        pixbuf[i + pixbuf_ofs + uf*4] = dstptr[(x + i)*4 + 1]; \
        pixbuf[i + pixbuf_ofs + uf*8] = dstptr[(x + i)*4 + 2]; \
        pixbuf[i + pixbuf_ofs + uf*12] = dstptr[(x + i)*4 + 3]; \
    } else { \
        int ix_ = borderInterpolate_fast(ix + dx, srccols, borderType_x); \
        int iy_ = borderInterpolate_fast(iy + dy, srcrows, borderType_y); \
        size_t addr_i = iy_*srcstep + ix_*4; \
        pixbuf[i + pixbuf_ofs] = src[addr_i]; \
        pixbuf[i + pixbuf_ofs + uf*4] = src[addr_i+1]; \
        pixbuf[i + pixbuf_ofs + uf*8] = src[addr_i+2]; \
        pixbuf[i + pixbuf_ofs + uf*12] = src[addr_i+3]; \
    }

#define SCALAR_FETCH_PIXEL_C3(dy, dx, pxy) \
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
#define SCALAR_FETCH_PIXEL_C1(dy, dx, pxy) \
    if ((((unsigned)(ix + dx) < (unsigned)srccols) & ((unsigned)(iy + dy) < (unsigned)srcrows)) != 0) { \
        size_t ofs = dy*srcstep + dx; \
        pxy = srcptr[ofs]; \
    } else if (borderType == BORDER_CONSTANT) { \
        pxy = bval[0]; \
    } else if (borderType == BORDER_TRANSPARENT) { \
        pxy = dstptr[x]; \
    } else { \
        int ix_ = borderInterpolate_fast(ix + dx, srccols, borderType_x); \
        int iy_ = borderInterpolate_fast(iy + dy, srcrows, borderType_y); \
        size_t glob_ofs = iy_*srcstep + ix_; \
        pxy = src[glob_ofs]; \
    }
#define SCALAR_FETCH_PIXEL_C4(dy, dx, pxy) \
    if ((((unsigned)(ix + dx) < (unsigned)srccols) & ((unsigned)(iy + dy) < (unsigned)srcrows)) != 0) { \
        size_t ofs = dy*srcstep + dx*4; \
        pxy##r = srcptr[ofs]; \
        pxy##g = srcptr[ofs+1]; \
        pxy##b = srcptr[ofs+2]; \
        pxy##a = srcptr[ofs+3]; \
    } else if (borderType == BORDER_CONSTANT) { \
        pxy##r = bval[0]; \
        pxy##g = bval[1]; \
        pxy##b = bval[2]; \
        pxy##a = bval[3]; \
    } else if (borderType == BORDER_TRANSPARENT) { \
        pxy##r = dstptr[x*4]; \
        pxy##g = dstptr[x*4+1]; \
        pxy##b = dstptr[x*4+2]; \
        pxy##a = dstptr[x*4+3]; \
    } else { \
        int ix_ = borderInterpolate_fast(ix + dx, srccols, borderType_x); \
        int iy_ = borderInterpolate_fast(iy + dy, srcrows, borderType_y); \
        size_t glob_ofs = iy_*srcstep + ix_*4; \
        pxy##r = src[glob_ofs]; \
        pxy##g = src[glob_ofs+1]; \
        pxy##b = src[glob_ofs+2]; \
        pxy##a = src[glob_ofs+3]; \
    }

#define VECTOR_COMPUTE_COORDINATES() \
    v_float32 src_x0 = v_fma(M0, dst_x0, M_x), \
              src_y0 = v_fma(M3, dst_x0, M_y), \
              src_x1 = v_fma(M0, dst_x1, M_x), \
              src_y1 = v_fma(M3, dst_x1, M_y); \
    dst_x0 = v_add(dst_x0, delta);             \
    dst_x1 = v_add(dst_x1, delta);             \
    v_int32 src_ix0 = v_floor(src_x0),         \
            src_iy0 = v_floor(src_y0),         \
            src_ix1 = v_floor(src_x1),         \
            src_iy1 = v_floor(src_y1);         \
    v_uint32 mask_0 = v_lt(v_reinterpret_as_u32(src_ix0), inner_scols), \
             mask_1 = v_lt(v_reinterpret_as_u32(src_ix1), inner_scols); \
    mask_0 = v_and(mask_0, v_lt(v_reinterpret_as_u32(src_iy0), inner_srows)); \
    mask_1 = v_and(mask_1, v_lt(v_reinterpret_as_u32(src_iy1), inner_srows)); \
    v_uint16 inner_mask = v_pack(mask_0, mask_1); \
    src_x0 = v_sub(src_x0, v_cvt_f32(src_ix0)); \
    src_y0 = v_sub(src_y0, v_cvt_f32(src_iy0)); \
    src_x1 = v_sub(src_x1, v_cvt_f32(src_ix1)); \
    src_y1 = v_sub(src_y1, v_cvt_f32(src_iy1));

#define VECTOR_LINEAR_C3_F32() \
    v_float32 alphal = src_x0, alphah = src_x1,\
              betal = src_y0, betah = src_y1;  \
    f00rl = v_fma(alphal, v_sub(f01rl, f00rl), f00rl); f00rh = v_fma(alphah, v_sub(f01rh, f00rh), f00rh); \
    f10rl = v_fma(alphal, v_sub(f11rl, f10rl), f10rl); f10rh = v_fma(alphah, v_sub(f11rh, f10rh), f10rh); \
    f00gl = v_fma(alphal, v_sub(f01gl, f00gl), f00gl); f00gh = v_fma(alphah, v_sub(f01gh, f00gh), f00gh); \
    f10gl = v_fma(alphal, v_sub(f11gl, f10gl), f10gl); f10gh = v_fma(alphah, v_sub(f11gh, f10gh), f10gh); \
    f00bl = v_fma(alphal, v_sub(f01bl, f00bl), f00bl); f00bh = v_fma(alphah, v_sub(f01bh, f00bh), f00bh); \
    f10bl = v_fma(alphal, v_sub(f11bl, f10bl), f10bl); f10bh = v_fma(alphah, v_sub(f11bh, f10bh), f10bh); \
    f00rl = v_fma(betal, v_sub(f10rl, f00rl), f00rl); f00rh = v_fma(betah, v_sub(f10rh, f00rh), f00rh); \
    f00gl = v_fma(betal, v_sub(f10gl, f00gl), f00gl); f00gh = v_fma(betah, v_sub(f10gh, f00gh), f00gh); \
    f00bl = v_fma(betal, v_sub(f10bl, f00bl), f00bl); f00bh = v_fma(betah, v_sub(f10bh, f00bh), f00bh);
#define VECTOR_LINEAR_C1_F32() \
    v_float32 alphal = src_x0, alphah = src_x1, \
              betal = src_y0, betah = src_y1; \
    f00l = v_fma(alphal, v_sub(f01l, f00l), f00l); f00h = v_fma(alphah, v_sub(f01h, f00h), f00h); \
    f10l = v_fma(alphal, v_sub(f11l, f10l), f10l); f10h = v_fma(alphah, v_sub(f11h, f10h), f10h); \
    f00l = v_fma(betal, v_sub(f10l, f00l), f00l);  f00h = v_fma(betah, v_sub(f10h, f00h), f00h);
#define VECTOR_LINEAR_C4_F32() \
    v_float32 alphal = src_x0, alphah = src_x1,\
              betal = src_y0, betah = src_y1;  \
    f00rl = v_fma(alphal, v_sub(f01rl, f00rl), f00rl); f00rh = v_fma(alphah, v_sub(f01rh, f00rh), f00rh); \
    f10rl = v_fma(alphal, v_sub(f11rl, f10rl), f10rl); f10rh = v_fma(alphah, v_sub(f11rh, f10rh), f10rh); \
    f00gl = v_fma(alphal, v_sub(f01gl, f00gl), f00gl); f00gh = v_fma(alphah, v_sub(f01gh, f00gh), f00gh); \
    f10gl = v_fma(alphal, v_sub(f11gl, f10gl), f10gl); f10gh = v_fma(alphah, v_sub(f11gh, f10gh), f10gh); \
    f00bl = v_fma(alphal, v_sub(f01bl, f00bl), f00bl); f00bh = v_fma(alphah, v_sub(f01bh, f00bh), f00bh); \
    f10bl = v_fma(alphal, v_sub(f11bl, f10bl), f10bl); f10bh = v_fma(alphah, v_sub(f11bh, f10bh), f10bh); \
    f00al = v_fma(alphal, v_sub(f01al, f00al), f00al); f00ah = v_fma(alphah, v_sub(f01ah, f00ah), f00ah); \
    f10al = v_fma(alphal, v_sub(f11al, f10al), f10al); f10ah = v_fma(alphah, v_sub(f11ah, f10ah), f10ah); \
    f00rl = v_fma(betal, v_sub(f10rl, f00rl), f00rl); f00rh = v_fma(betah, v_sub(f10rh, f00rh), f00rh); \
    f00gl = v_fma(betal, v_sub(f10gl, f00gl), f00gl); f00gh = v_fma(betah, v_sub(f10gh, f00gh), f00gh); \
    f00bl = v_fma(betal, v_sub(f10bl, f00bl), f00bl); f00bh = v_fma(betah, v_sub(f10bh, f00bh), f00bh); \
    f00al = v_fma(betal, v_sub(f10al, f00al), f00al); f00ah = v_fma(betah, v_sub(f10ah, f00ah), f00ah);

#define VECTOR_LINEAR_STORE_BORDER_8UC3() \
    v_store_low(dstptr + x*3,        bval_v0); \
    v_store_low(dstptr + x*3 + uf,   bval_v1); \
    v_store_low(dstptr + x*3 + uf*2, bval_v2);
#define VECTOR_LINEAR_STORE_BORDER_16UC3() \
    v_store(dstptr + x*3,        bval_v0); \
    v_store(dstptr + x*3 + uf,   bval_v1); \
    v_store(dstptr + x*3 + uf*2, bval_v2);
#define VECTOR_LINEAR_STORE_BORDER_32FC3() \
    v_store(dstptr + x*3,                    bval_v0_l); \
    v_store(dstptr + x*3 + vlanes_32,        bval_v0_h); \
    v_store(dstptr + x*3 + uf,               bval_v1_l); \
    v_store(dstptr + x*3 + uf + vlanes_32,   bval_v1_h); \
    v_store(dstptr + x*3 + uf*2,             bval_v2_l); \
    v_store(dstptr + x*3 + uf*2 + vlanes_32, bval_v2_h);
#define VECTOR_LINEAR_STORE_BORDER_8UC1() \
    v_store_low(dstptr + x, bval_v0);
#define VECTOR_LINEAR_STORE_BORDER_16UC1() \
    v_store(dstptr + x, bval_v0);
#define VECTOR_LINEAR_STORE_BORDER_32FC1() \
    v_store(dstptr + x, bval_v0_l); \
    v_store(dstptr + x + vlanes_32, bval_v0_h);
#define VECTOR_LINEAR_STORE_BORDER_8UC4() \
    v_store_low(dstptr + x*4,        bval_v0); \
    v_store_low(dstptr + x*4 + uf,   bval_v1); \
    v_store_low(dstptr + x*4 + uf*2, bval_v2); \
    v_store_low(dstptr + x*4 + uf*3, bval_v3);
#define VECTOR_LINEAR_STORE_BORDER_16UC4() \
    v_store(dstptr + x*4,        bval_v0); \
    v_store(dstptr + x*4 + uf,   bval_v1); \
    v_store(dstptr + x*4 + uf*2, bval_v2); \
    v_store(dstptr + x*4 + uf*3, bval_v3);
#define VECTOR_LINEAR_STORE_BORDER_32FC4() \
    v_store(dstptr + x*4,                    bval_v0_l); \
    v_store(dstptr + x*4 + vlanes_32,        bval_v0_h); \
    v_store(dstptr + x*4 + uf,               bval_v1_l); \
    v_store(dstptr + x*4 + uf + vlanes_32,   bval_v1_h); \
    v_store(dstptr + x*4 + uf*2,             bval_v2_l); \
    v_store(dstptr + x*4 + uf*2 + vlanes_32, bval_v2_h); \
    v_store(dstptr + x*4 + uf*3,             bval_v3_l); \
    v_store(dstptr + x*4 + uf*3 + vlanes_32, bval_v3_h);

#define VECTOR_LINEAR_SHUFFLE_NOTALLIN(depth, cn) \
    if (borderType == BORDER_CONSTANT || borderType == BORDER_TRANSPARENT) { \
        mask_0 = v_lt(v_reinterpret_as_u32(v_add(src_ix0, one)), outer_scols); \
        mask_1 = v_lt(v_reinterpret_as_u32(v_add(src_ix1, one)), outer_scols); \
        mask_0 = v_and(mask_0, v_lt(v_reinterpret_as_u32(v_add(src_iy0, one)), outer_srows)); \
        mask_1 = v_and(mask_1, v_lt(v_reinterpret_as_u32(v_add(src_iy1, one)), outer_srows)); \
        v_uint16 outer_mask = v_pack(mask_0, mask_1); \
        if (v_reduce_max(outer_mask) == 0) { \
            if (borderType == BORDER_CONSTANT) { \
                VECTOR_LINEAR_STORE_BORDER_##depth##cn(); \
            } \
            continue; \
        } \
    } \
    vx_store(src_ix, src_ix0); \
    vx_store(src_iy, src_iy0); \
    vx_store(src_ix + vlanes_32, src_ix1); \
    vx_store(src_iy + vlanes_32, src_iy1); \
    for (int i = 0; i < uf; i++) { \
        int ix = src_ix[i], iy = src_iy[i]; \
        VECTOR_FETCH_PIXEL_##cn(0, 0, 0); \
        VECTOR_FETCH_PIXEL_##cn(0, 1, uf); \
        VECTOR_FETCH_PIXEL_##cn(1, 0, uf*2); \
        VECTOR_FETCH_PIXEL_##cn(1, 1, uf*3); \
    }

#define SCALAR_LINEAR_SHUFFLE_C3() \
    if ((((unsigned)ix < (unsigned)(srccols-1)) & \
        ((unsigned)iy < (unsigned)(srcrows-1))) != 0) { \
        p00r = srcptr[0]; p00g = srcptr[1]; p00b = srcptr[2]; \
        p01r = srcptr[3]; p01g = srcptr[4]; p01b = srcptr[5]; \
        p10r = srcptr[srcstep + 0]; p10g = srcptr[srcstep + 1]; p10b = srcptr[srcstep + 2]; \
        p11r = srcptr[srcstep + 3]; p11g = srcptr[srcstep + 4]; p11b = srcptr[srcstep + 5]; \
    } else { \
        if ((borderType == BORDER_CONSTANT || borderType == BORDER_TRANSPARENT) && \
            (((unsigned)(ix+1) >= (unsigned)(srccols+1))| \
                ((unsigned)(iy+1) >= (unsigned)(srcrows+1))) != 0) { \
            if (borderType == BORDER_CONSTANT) { \
                dstptr[x*3] = bval[0]; \
                dstptr[x*3+1] = bval[1]; \
                dstptr[x*3+2] = bval[2]; \
            } \
            continue; \
        } \
        SCALAR_FETCH_PIXEL_C3(0, 0, p00); \
        SCALAR_FETCH_PIXEL_C3(0, 1, p01); \
        SCALAR_FETCH_PIXEL_C3(1, 0, p10); \
        SCALAR_FETCH_PIXEL_C3(1, 1, p11); \
    }
#define SCALAR_LINEAR_SHUFFLE_C4() \
    if ((((unsigned)ix < (unsigned)(srccols-1)) & \
        ((unsigned)iy < (unsigned)(srcrows-1))) != 0) { \
        p00r = srcptr[0]; p00g = srcptr[1]; p00b = srcptr[2]; p00a = srcptr[3]; \
        p01r = srcptr[4]; p01g = srcptr[5]; p01b = srcptr[6]; p01a = srcptr[7]; \
        p10r = srcptr[srcstep + 0]; p10g = srcptr[srcstep + 1]; p10b = srcptr[srcstep + 2]; p10a = srcptr[srcstep + 3]; \
        p11r = srcptr[srcstep + 4]; p11g = srcptr[srcstep + 5]; p11b = srcptr[srcstep + 6]; p11a = srcptr[srcstep + 7]; \
    } else { \
        if ((borderType == BORDER_CONSTANT || borderType == BORDER_TRANSPARENT) && \
            (((unsigned)(ix+1) >= (unsigned)(srccols+1))| \
                ((unsigned)(iy+1) >= (unsigned)(srcrows+1))) != 0) { \
            if (borderType == BORDER_CONSTANT) { \
                dstptr[x*4] = bval[0]; \
                dstptr[x*4+1] = bval[1]; \
                dstptr[x*4+2] = bval[2]; \
                dstptr[x*4+3] = bval[3]; \
            } \
            continue; \
        } \
        SCALAR_FETCH_PIXEL_C4(0, 0, p00); \
        SCALAR_FETCH_PIXEL_C4(0, 1, p01); \
        SCALAR_FETCH_PIXEL_C4(1, 0, p10); \
        SCALAR_FETCH_PIXEL_C4(1, 1, p11); \
    }

#define SCALAR_LINEAR_CALC_C3() \
    float v0r = p00r + sx*(p01r - p00r); \
    float v0g = p00g + sx*(p01g - p00g); \
    float v0b = p00b + sx*(p01b - p00b); \
    float v1r = p10r + sx*(p11r - p10r); \
    float v1g = p10g + sx*(p11g - p10g); \
    float v1b = p10b + sx*(p11b - p10b); \
    v0r += sy*(v1r - v0r); \
    v0g += sy*(v1g - v0g); \
    v0b += sy*(v1b - v0b);

#define SCALAR_LINEAR_CALC_C4() \
    float v0r = p00r + sx*(p01r - p00r); \
    float v0g = p00g + sx*(p01g - p00g); \
    float v0b = p00b + sx*(p01b - p00b); \
    float v0a = p00a + sx*(p01a - p00a); \
    float v1r = p10r + sx*(p11r - p10r); \
    float v1g = p10g + sx*(p11g - p10g); \
    float v1b = p10b + sx*(p11b - p10b); \
    float v1a = p10a + sx*(p11a - p10a); \
    v0r += sy*(v1r - v0r); \
    v0g += sy*(v1g - v0g); \
    v0b += sy*(v1b - v0b); \
    v0a += sy*(v1a - v0a);

namespace cv{
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

/* Only support bilinear interpolation on 3-channel image for now */
void warpAffineSimdInvoker(Mat &output, const Mat &input, const double *M,
                           int interpolation, int borderType, const double *borderValue);

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

template<typename T>
static inline void shuffle_allin_c3(const T *src, const int32_t *addr, T *pixbuf,
                                    int uf, size_t srcstep) {
    for (int i = 0; i < uf; i++) {
        const T* srcptr = src + addr[i];

        pixbuf[i] = srcptr[0];
        pixbuf[i + uf*4] = srcptr[1];
        pixbuf[i + uf*8] = srcptr[2];

        pixbuf[i + uf] = srcptr[3];
        pixbuf[i + uf*5] = srcptr[4];
        pixbuf[i + uf*9] = srcptr[5];

        pixbuf[i + uf*2] = srcptr[srcstep];
        pixbuf[i + uf*6] = srcptr[srcstep + 1];
        pixbuf[i + uf*10] = srcptr[srcstep + 2];

        pixbuf[i + uf*3] = srcptr[srcstep + 3];
        pixbuf[i + uf*7] = srcptr[srcstep + 4];
        pixbuf[i + uf*11] = srcptr[srcstep + 5];
    }
}
template<typename T>
static inline void shuffle_allin_c1(const T *src, const int32_t *addr, T *pixbuf,
                                    int uf, size_t srcstep) {
    for (int i = 0; i < uf; i++) {
        const T* srcptr = src + addr[i];

        pixbuf[i] = srcptr[0];
        pixbuf[i + uf] = srcptr[1];
        pixbuf[i + uf*2] = srcptr[srcstep];
        pixbuf[i + uf*3] = srcptr[srcstep + 1];
    }
}
template<typename T>
static inline void shuffle_allin_c4(const T *src, const int32_t *addr, T *pixbuf,
                                    int uf, size_t srcstep) {
    for (int i = 0; i < uf; i++) {
        const T* srcptr = src + addr[i];

        pixbuf[i] = srcptr[0];
        pixbuf[i + uf*4] = srcptr[1];
        pixbuf[i + uf*8] = srcptr[2];
        pixbuf[i + uf*12] = srcptr[3];

        pixbuf[i + uf] = srcptr[4];
        pixbuf[i + uf*5] = srcptr[5];
        pixbuf[i + uf*9] = srcptr[6];
        pixbuf[i + uf*13] = srcptr[7];

        pixbuf[i + uf*2] = srcptr[srcstep];
        pixbuf[i + uf*6] = srcptr[srcstep + 1];
        pixbuf[i + uf*10] = srcptr[srcstep + 2];
        pixbuf[i + uf*14] = srcptr[srcstep + 3];

        pixbuf[i + uf*3] = srcptr[srcstep + 4];
        pixbuf[i + uf*7] = srcptr[srcstep + 5];
        pixbuf[i + uf*11] = srcptr[srcstep + 6];
        pixbuf[i + uf*15] = srcptr[srcstep + 7];
    }
}

class WarpAffineLinearInvoker_8UC3 : public ParallelLoopBody {
public:
    WarpAffineLinearInvoker_8UC3(Mat *output_, const Mat *input_, const double *dM_,
                                 int borderType_, const double *borderValue_)
        : output(output_), input(input_), dM(dM_), borderType(borderType_), borderValue(borderValue_) {}

    virtual void operator() (const Range &r) const CV_OVERRIDE {
        CV_INSTRUMENT_REGION();

        auto *src = input->ptr<const uint8_t>();
        auto *dst = output->ptr<uint8_t>();
        size_t srcstep = input->step, dststep = output->step;
        int srccols = input->cols, srcrows = input->rows;
        int dstcols = output->cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(borderValue[0]),
            saturate_cast<uint8_t>(borderValue[1]),
            saturate_cast<uint8_t>(borderValue[2]),
            saturate_cast<uint8_t>(borderValue[3]),
        };
        int borderType_x = borderType != BORDER_CONSTANT &&
                           borderType != BORDER_TRANSPARENT &&
                           input->cols <= 1 ? BORDER_REPLICATE : borderType;
        int borderType_y = borderType != BORDER_CONSTANT &&
                           borderType != BORDER_TRANSPARENT &&
                           input->rows <= 1 ? BORDER_REPLICATE : borderType;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};

        int vlanes_32 = VTraits<v_float32>::vlanes();

        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        uint8_t bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4*3];
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
        uint8x8_t reds = {0, 8, 16, 24, 3, 11, 19, 27},
                  greens = {1, 9, 17, 25, 4, 12, 20, 28},
                  blues = {2, 10, 18, 26, 5, 13, 21, 29};
    #endif
    #if !CV_SIMD128_FP16
        constexpr int max_vlanes_16{VTraits<v_uint16>::max_nlanes};
        int vlanes_16 = VTraits<v_uint16>::vlanes();
    #endif
#endif

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            v_float32 dst_x0 = vx_load(start_indices.data());
            v_float32 dst_x1 = v_add(dst_x0, vx_setall_f32(float(vlanes_32)));
            v_float32 M0 = vx_setall_f32(M[0]),
                      M3 = vx_setall_f32(M[3]);
            v_float32 M_x = vx_setall_f32(static_cast<float>(y * M[1] + M[2])),
                      M_y = vx_setall_f32(static_cast<float>(y * M[4] + M[5]));

            for (; x < dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                VECTOR_COMPUTE_COORDINATES();

                v_int32 addr_0 = v_fma(v_srcstep, src_iy0, v_mul(src_ix0, three)),
                        addr_1 = v_fma(v_srcstep, src_iy1, v_mul(src_ix1, three));
                vx_store(addr, addr_0);
                vx_store(addr + vlanes_32, addr_1);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b;
    #endif

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    uint8x8x4_t p00 = {
                        vld1_u8(src + addr[0]),
                        vld1_u8(src + addr[1]),
                        vld1_u8(src + addr[2]),
                        vld1_u8(src + addr[3])
                    };

                    uint8x8x4_t p01 = {
                        vld1_u8(src + addr[4]),
                        vld1_u8(src + addr[5]),
                        vld1_u8(src + addr[6]),
                        vld1_u8(src + addr[7])
                    };

                    uint8x8x4_t p10 = {
                        vld1_u8(src + addr[0] + srcstep),
                        vld1_u8(src + addr[1] + srcstep),
                        vld1_u8(src + addr[2] + srcstep),
                        vld1_u8(src + addr[3] + srcstep)
                    };

                    uint8x8x4_t p11 = {
                        vld1_u8(src + addr[4] + srcstep),
                        vld1_u8(src + addr[5] + srcstep),
                        vld1_u8(src + addr[6] + srcstep),
                        vld1_u8(src + addr[7] + srcstep)
                    };

                    uint32x2_t p00_, p01_, p10_, p11_;

                    p00_ = vreinterpret_u32_u8(vtbl4_u8(p00, reds));
                    p01_ = vreinterpret_u32_u8(vtbl4_u8(p01, reds));
                    p10_ = vreinterpret_u32_u8(vtbl4_u8(p10, reds));
                    p11_ = vreinterpret_u32_u8(vtbl4_u8(p11, reds));

                    p00r = vreinterpret_u8_u32(vtrn1_u32(p00_, p01_));
                    p01r = vreinterpret_u8_u32(vtrn2_u32(p00_, p01_));
                    p10r = vreinterpret_u8_u32(vtrn1_u32(p10_, p11_));
                    p11r = vreinterpret_u8_u32(vtrn2_u32(p10_, p11_));

                    p00_ = vreinterpret_u32_u8(vtbl4_u8(p00, greens));
                    p01_ = vreinterpret_u32_u8(vtbl4_u8(p01, greens));
                    p10_ = vreinterpret_u32_u8(vtbl4_u8(p10, greens));
                    p11_ = vreinterpret_u32_u8(vtbl4_u8(p11, greens));

                    p00g = vreinterpret_u8_u32(vtrn1_u32(p00_, p01_));
                    p01g = vreinterpret_u8_u32(vtrn2_u32(p00_, p01_));
                    p10g = vreinterpret_u8_u32(vtrn1_u32(p10_, p11_));
                    p11g = vreinterpret_u8_u32(vtrn2_u32(p10_, p11_));

                    p00_ = vreinterpret_u32_u8(vtbl4_u8(p00, blues));
                    p01_ = vreinterpret_u32_u8(vtbl4_u8(p01, blues));
                    p10_ = vreinterpret_u32_u8(vtbl4_u8(p10, blues));
                    p11_ = vreinterpret_u32_u8(vtbl4_u8(p11, blues));

                    p00b = vreinterpret_u8_u32(vtrn1_u32(p00_, p01_));
                    p01b = vreinterpret_u8_u32(vtrn2_u32(p00_, p01_));
                    p10b = vreinterpret_u8_u32(vtrn1_u32(p10_, p11_));
                    p11b = vreinterpret_u8_u32(vtrn2_u32(p10_, p11_));
    #else // scalar implementation when neon intrinsics are not available
                    shuffle_allin_c3(src, addr, pixbuf, uf, srcstep);
    #endif
                } else {
                    VECTOR_LINEAR_SHUFFLE_NOTALLIN(8U, C3);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    p00r = vld1_u8(pixbuf);
                    p01r = vld1_u8(pixbuf + 8);
                    p10r = vld1_u8(pixbuf + 16);
                    p11r = vld1_u8(pixbuf + 24);

                    p00g = vld1_u8(pixbuf + 32);
                    p01g = vld1_u8(pixbuf + 32 + 8);
                    p10g = vld1_u8(pixbuf + 32 + 16);
                    p11g = vld1_u8(pixbuf + 32 + 24);

                    p00b = vld1_u8(pixbuf + 64);
                    p01b = vld1_u8(pixbuf + 64 + 8);
                    p10b = vld1_u8(pixbuf + 64 + 16);
                    p11b = vld1_u8(pixbuf + 64 + 24);
    #endif
                }

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16 // [TODO] support risc-v fp16 intrinsics
                v_float16 f00r = v_float16(vcvtq_f16_u16(vmovl_u8(p00r)));
                v_float16 f01r = v_float16(vcvtq_f16_u16(vmovl_u8(p01r)));
                v_float16 f10r = v_float16(vcvtq_f16_u16(vmovl_u8(p10r)));
                v_float16 f11r = v_float16(vcvtq_f16_u16(vmovl_u8(p11r)));

                v_float16 f00g = v_float16(vcvtq_f16_u16(vmovl_u8(p00g)));
                v_float16 f01g = v_float16(vcvtq_f16_u16(vmovl_u8(p01g)));
                v_float16 f10g = v_float16(vcvtq_f16_u16(vmovl_u8(p10g)));
                v_float16 f11g = v_float16(vcvtq_f16_u16(vmovl_u8(p11g)));

                v_float16 f00b = v_float16(vcvtq_f16_u16(vmovl_u8(p00b)));
                v_float16 f01b = v_float16(vcvtq_f16_u16(vmovl_u8(p01b)));
                v_float16 f10b = v_float16(vcvtq_f16_u16(vmovl_u8(p10b)));
                v_float16 f11b = v_float16(vcvtq_f16_u16(vmovl_u8(p11b)));

                v_float16 alpha = v_cvt_f16(src_x0, src_x1),
                          beta = v_cvt_f16(src_y0, src_y1);

                f00r = v_fma(alpha, v_sub(f01r, f00r), f00r);
                f10r = v_fma(alpha, v_sub(f11r, f10r), f10r);

                f00g = v_fma(alpha, v_sub(f01g, f00g), f00g);
                f10g = v_fma(alpha, v_sub(f11g, f10g), f10g);

                f00b = v_fma(alpha, v_sub(f01b, f00b), f00b);
                f10b = v_fma(alpha, v_sub(f11b, f10b), f10b);

                f00r = v_fma(beta,  v_sub(f10r, f00r), f00r);
                f00g = v_fma(beta,  v_sub(f10g, f00g), f00g);
                f00b = v_fma(beta,  v_sub(f10b, f00b), f00b);

                uint8x8x3_t result = {
                    vqmovun_s16(vcvtnq_s16_f16(f00r.val)),
                    vqmovun_s16(vcvtnq_s16_f16(f00g.val)),
                    vqmovun_s16(vcvtnq_s16_f16(f00b.val)),
                };
                vst3_u8(dstptr + x*3, result);
    #else // Other platforms use fp32 intrinsics for interpolation calculation
        #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 // In case neon fp16 intrinsics are not available; still requires A64
                v_int16 f00r = v_reinterpret_as_s16(v_uint16(vmovl_u8(p00r))),
                        f01r = v_reinterpret_as_s16(v_uint16(vmovl_u8(p01r))),
                        f10r = v_reinterpret_as_s16(v_uint16(vmovl_u8(p10r))),
                        f11r = v_reinterpret_as_s16(v_uint16(vmovl_u8(p11r)));
                v_int16 f00g = v_reinterpret_as_s16(v_uint16(vmovl_u8(p00g))),
                        f01g = v_reinterpret_as_s16(v_uint16(vmovl_u8(p01g))),
                        f10g = v_reinterpret_as_s16(v_uint16(vmovl_u8(p10g))),
                        f11g = v_reinterpret_as_s16(v_uint16(vmovl_u8(p11g)));
                v_int16 f00b = v_reinterpret_as_s16(v_uint16(vmovl_u8(p00b))),
                        f01b = v_reinterpret_as_s16(v_uint16(vmovl_u8(p01b))),
                        f10b = v_reinterpret_as_s16(v_uint16(vmovl_u8(p10b))),
                        f11b = v_reinterpret_as_s16(v_uint16(vmovl_u8(p11b)));
        #else
                v_int16  f00r = v_reinterpret_as_s16(vx_load_expand(pixbuf)),
                         f01r = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf)),
                         f10r = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*2)),
                         f11r = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*3));
                v_int16  f00g = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*4)),
                         f01g = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*5)),
                         f10g = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*6)),
                         f11g = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*7));
                v_int16  f00b = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*8)),
                         f01b = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*9)),
                         f10b = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*10)),
                         f11b = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*11));
        #endif
                v_float32 f00rl = v_cvt_f32(v_expand_low(f00r)), f00rh = v_cvt_f32(v_expand_high(f00r)),
                          f01rl = v_cvt_f32(v_expand_low(f01r)), f01rh = v_cvt_f32(v_expand_high(f01r)),
                          f10rl = v_cvt_f32(v_expand_low(f10r)), f10rh = v_cvt_f32(v_expand_high(f10r)),
                          f11rl = v_cvt_f32(v_expand_low(f11r)), f11rh = v_cvt_f32(v_expand_high(f11r));
                v_float32 f00gl = v_cvt_f32(v_expand_low(f00g)), f00gh = v_cvt_f32(v_expand_high(f00g)),
                          f01gl = v_cvt_f32(v_expand_low(f01g)), f01gh = v_cvt_f32(v_expand_high(f01g)),
                          f10gl = v_cvt_f32(v_expand_low(f10g)), f10gh = v_cvt_f32(v_expand_high(f10g)),
                          f11gl = v_cvt_f32(v_expand_low(f11g)), f11gh = v_cvt_f32(v_expand_high(f11g));
                v_float32 f00bl = v_cvt_f32(v_expand_low(f00b)), f00bh = v_cvt_f32(v_expand_high(f00b)),
                          f01bl = v_cvt_f32(v_expand_low(f01b)), f01bh = v_cvt_f32(v_expand_high(f01b)),
                          f10bl = v_cvt_f32(v_expand_low(f10b)), f10bh = v_cvt_f32(v_expand_high(f10b)),
                          f11bl = v_cvt_f32(v_expand_low(f11b)), f11bh = v_cvt_f32(v_expand_high(f11b));

                VECTOR_LINEAR_C3_F32();

                v_uint16 f00r_u16 = v_pack_u(v_round(f00rl), v_round(f00rh)),
                         f00g_u16 = v_pack_u(v_round(f00gl), v_round(f00gh)),
                         f00b_u16 = v_pack_u(v_round(f00bl), v_round(f00bh));
                uint16_t tbuf[max_vlanes_16*3];
                v_store_interleave(tbuf, f00r_u16, f00g_u16, f00b_u16);
                v_pack_store(dstptr + x*3, vx_load(tbuf));
                v_pack_store(dstptr + x*3 + vlanes_16, vx_load(tbuf + vlanes_16));
                v_pack_store(dstptr + x*3 + vlanes_16*2, vx_load(tbuf + vlanes_16*2));
    #endif // defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00r, p00g, p00b, p01r, p01g, p01b;
                int p10r, p10g, p10b, p11r, p11g, p11b;
                const uint8_t* srcptr = src + srcstep*iy + ix*3;

                SCALAR_LINEAR_SHUFFLE_C3();

                SCALAR_LINEAR_CALC_C3();

                dstptr[x*3] = saturate_cast<uint8_t>(v0r);
                dstptr[x*3+1] = saturate_cast<uint8_t>(v0g);
                dstptr[x*3+2] = saturate_cast<uint8_t>(v0b);
            }
        }
    }

private:
    Mat *output;
    const Mat *input;
    const double *dM;
    int borderType;
    const double *borderValue;
};

class WarpAffineLinearInvoker_16UC3 : public ParallelLoopBody {
public:
    WarpAffineLinearInvoker_16UC3(Mat *output_, const Mat *input_, const double *dM_,
                                 int borderType_, const double *borderValue_)
        : output(output_), input(input_), dM(dM_), borderType(borderType_), borderValue(borderValue_) {}

    virtual void operator() (const Range &r) const CV_OVERRIDE {
        CV_INSTRUMENT_REGION();

        auto *src = input->ptr<const uint16_t>();
        auto *dst = output->ptr<uint16_t>();
        size_t srcstep = input->step/sizeof(uint16_t), dststep = output->step/sizeof(uint16_t);
        int srccols = input->cols, srcrows = input->rows;
        int dstcols = output->cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint16_t bval[] = {
            saturate_cast<uint16_t>(borderValue[0]),
            saturate_cast<uint16_t>(borderValue[1]),
            saturate_cast<uint16_t>(borderValue[2]),
            saturate_cast<uint16_t>(borderValue[3]),
        };
        int borderType_x = borderType != BORDER_CONSTANT &&
                           borderType != BORDER_TRANSPARENT &&
                           input->cols <= 1 ? BORDER_REPLICATE : borderType;
        int borderType_y = borderType != BORDER_CONSTANT &&
                           borderType != BORDER_TRANSPARENT &&
                           input->rows <= 1 ? BORDER_REPLICATE : borderType;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};

        int vlanes_32 = VTraits<v_float32>::vlanes();

        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        uint16_t bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
        v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
        v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint16_t pixbuf[max_uf*4*3];
#endif

        for (int y = r.start; y < r.end; y++) {
            uint16_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            v_float32 dst_x0 = vx_load(start_indices.data());
            v_float32 dst_x1 = v_add(dst_x0, vx_setall_f32(float(vlanes_32)));
            v_float32 M0 = vx_setall_f32(M[0]),
                      M3 = vx_setall_f32(M[3]);
            v_float32 M_x = vx_setall_f32(static_cast<float>(y * M[1] + M[2])),
                      M_y = vx_setall_f32(static_cast<float>(y * M[4] + M[5]));

            for (; x < dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                VECTOR_COMPUTE_COORDINATES();

                v_int32 addr_0 = v_fma(v_srcstep, src_iy0, v_mul(src_ix0, three)),
                        addr_1 = v_fma(v_srcstep, src_iy1, v_mul(src_ix1, three));
                vx_store(addr, addr_0);
                vx_store(addr + vlanes_32, addr_1);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    shuffle_allin_c3(src, addr, pixbuf, uf, srcstep);
                } else {
                    VECTOR_LINEAR_SHUFFLE_NOTALLIN(16U, C3);
                }

                v_uint16 f00r = vx_load(pixbuf),
                         f01r = vx_load(pixbuf + uf),
                         f10r = vx_load(pixbuf + uf*2),
                         f11r = vx_load(pixbuf + uf*3);
                v_uint16 f00g = vx_load(pixbuf + uf*4),
                         f01g = vx_load(pixbuf + uf*5),
                         f10g = vx_load(pixbuf + uf*6),
                         f11g = vx_load(pixbuf + uf*7);
                v_uint16 f00b = vx_load(pixbuf + uf*8),
                         f01b = vx_load(pixbuf + uf*9),
                         f10b = vx_load(pixbuf + uf*10),
                         f11b = vx_load(pixbuf + uf*11);

                v_float32 f00rl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f00r))), f00rh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f00r))),
                          f01rl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f01r))), f01rh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f01r))),
                          f10rl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f10r))), f10rh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f10r))),
                          f11rl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f11r))), f11rh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f11r)));
                v_float32 f00gl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f00g))), f00gh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f00g))),
                          f01gl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f01g))), f01gh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f01g))),
                          f10gl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f10g))), f10gh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f10g))),
                          f11gl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f11g))), f11gh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f11g)));
                v_float32 f00bl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f00b))), f00bh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f00b))),
                          f01bl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f01b))), f01bh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f01b))),
                          f10bl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f10b))), f10bh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f10b))),
                          f11bl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f11b))), f11bh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f11b)));

                VECTOR_LINEAR_C3_F32();

                v_uint16 f00r_u16 = v_pack_u(v_round(f00rl), v_round(f00rh)),
                         f00g_u16 = v_pack_u(v_round(f00gl), v_round(f00gh)),
                         f00b_u16 = v_pack_u(v_round(f00bl), v_round(f00bh));
                v_store_interleave(dstptr + x*3, f00r_u16, f00g_u16, f00b_u16);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00r, p00g, p00b, p01r, p01g, p01b;
                int p10r, p10g, p10b, p11r, p11g, p11b;
                const uint16_t* srcptr = src + srcstep*iy + ix*3;

                SCALAR_LINEAR_SHUFFLE_C3();

                SCALAR_LINEAR_CALC_C3();

                dstptr[x*3] = saturate_cast<uint16_t>(v0r);
                dstptr[x*3+1] = saturate_cast<uint16_t>(v0g);
                dstptr[x*3+2] = saturate_cast<uint16_t>(v0b);
            }
        }
    }

private:
    Mat *output;
    const Mat *input;
    const double *dM;
    int borderType;
    const double *borderValue;
};

class WarpAffineLinearInvoker_32FC3 : public ParallelLoopBody {
public:
    WarpAffineLinearInvoker_32FC3(Mat *output_, const Mat *input_, const double *dM_,
                                  int borderType_, const double *borderValue_)
        : output(output_), input(input_), dM(dM_), borderType(borderType_), borderValue(borderValue_) {}

    virtual void operator() (const Range &r) const CV_OVERRIDE {
        CV_INSTRUMENT_REGION();

        auto *src = input->ptr<const float>();
        auto *dst = output->ptr<float>();
        size_t srcstep = input->step/sizeof(float), dststep = output->step/sizeof(float);
        int srccols = input->cols, srcrows = input->rows;
        int dstcols = output->cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        float bval[] = {
            saturate_cast<float>(borderValue[0]),
            saturate_cast<float>(borderValue[1]),
            saturate_cast<float>(borderValue[2]),
            saturate_cast<float>(borderValue[3]),
        };
        int borderType_x = borderType != BORDER_CONSTANT &&
                           borderType != BORDER_TRANSPARENT &&
                           input->cols <= 1 ? BORDER_REPLICATE : borderType;
        int borderType_y = borderType != BORDER_CONSTANT &&
                           borderType != BORDER_TRANSPARENT &&
                           input->rows <= 1 ? BORDER_REPLICATE : borderType;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};

        int vlanes_32 = VTraits<v_float32>::vlanes();

        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        float bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
        v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
        v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
        v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
        v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
        v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        float pixbuf[max_uf*4*3];
#endif

        for (int y = r.start; y < r.end; y++) {
            float* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            v_float32 dst_x0 = vx_load(start_indices.data());
            v_float32 dst_x1 = v_add(dst_x0, vx_setall_f32(float(vlanes_32)));
            v_float32 M0 = vx_setall_f32(M[0]),
                      M3 = vx_setall_f32(M[3]);
            v_float32 M_x = vx_setall_f32(static_cast<float>(y * M[1] + M[2])),
                      M_y = vx_setall_f32(static_cast<float>(y * M[4] + M[5]));

            for (; x < dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                VECTOR_COMPUTE_COORDINATES();

                v_int32 addr_0 = v_fma(v_srcstep, src_iy0, v_mul(src_ix0, three)),
                        addr_1 = v_fma(v_srcstep, src_iy1, v_mul(src_ix1, three));
                vx_store(addr, addr_0);
                vx_store(addr + vlanes_32, addr_1);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    shuffle_allin_c3(src, addr, pixbuf, uf, srcstep);
                } else {
                    VECTOR_LINEAR_SHUFFLE_NOTALLIN(32F, C3);
                }

                v_float32 f00rl = vx_load(pixbuf),         f00rh = vx_load(pixbuf + vlanes_32),
                          f01rl = vx_load(pixbuf + uf),    f01rh = vx_load(pixbuf + uf + vlanes_32),
                          f10rl = vx_load(pixbuf + uf*2),  f10rh = vx_load(pixbuf + uf*2 + vlanes_32),
                          f11rl = vx_load(pixbuf + uf*3),  f11rh = vx_load(pixbuf + uf*3 + vlanes_32);
                v_float32 f00gl = vx_load(pixbuf + uf*4),  f00gh = vx_load(pixbuf + uf*4 + vlanes_32),
                          f01gl = vx_load(pixbuf + uf*5),  f01gh = vx_load(pixbuf + uf*5 + vlanes_32),
                          f10gl = vx_load(pixbuf + uf*6),  f10gh = vx_load(pixbuf + uf*6 + vlanes_32),
                          f11gl = vx_load(pixbuf + uf*7),  f11gh = vx_load(pixbuf + uf*7 + vlanes_32);
                v_float32 f00bl = vx_load(pixbuf + uf*8),  f00bh = vx_load(pixbuf + uf*8 + vlanes_32),
                          f01bl = vx_load(pixbuf + uf*9),  f01bh = vx_load(pixbuf + uf*9 + vlanes_32),
                          f10bl = vx_load(pixbuf + uf*10), f10bh = vx_load(pixbuf + uf*10 + vlanes_32),
                          f11bl = vx_load(pixbuf + uf*11), f11bh = vx_load(pixbuf + uf*11 + vlanes_32);

                VECTOR_LINEAR_C3_F32();

                v_store_interleave(dstptr + x*3, f00rl, f00gl, f00bl);
                v_store_interleave(dstptr + x*3 + vlanes_32*3, f00rh, f00gh, f00bh);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                float p00r, p00g, p00b, p01r, p01g, p01b;
                float p10r, p10g, p10b, p11r, p11g, p11b;
                const float* srcptr = src + srcstep*iy + ix*3;

                SCALAR_LINEAR_SHUFFLE_C3();

                SCALAR_LINEAR_CALC_C3();

                dstptr[x*3] =   v0r;
                dstptr[x*3+1] = v0g;
                dstptr[x*3+2] = v0b;
            }
        }
    }

private:
    Mat *output;
    const Mat *input;
    const double *dM;
    int borderType;
    const double *borderValue;
};

class WarpAffineLinearInvoker_8UC1 : public ParallelLoopBody {
public:
    WarpAffineLinearInvoker_8UC1(Mat *output_, const Mat *input_, const double *dM_,
                                 int borderType_, const double *borderValue_)
        : output(output_), input(input_), dM(dM_), borderType(borderType_), borderValue(borderValue_) {}

    virtual void operator() (const Range &r) const CV_OVERRIDE {
        CV_INSTRUMENT_REGION();

        auto *src = input->ptr<const uint8_t>();
        auto *dst = output->ptr<uint8_t>();
        size_t srcstep = input->step, dststep = output->step;
        int srccols = input->cols, srcrows = input->rows;
        int dstcols = output->cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(borderValue[0]),
            saturate_cast<uint8_t>(borderValue[1]),
            saturate_cast<uint8_t>(borderValue[2]),
            saturate_cast<uint8_t>(borderValue[3]),
        };
        int borderType_x = borderType != BORDER_CONSTANT &&
                           borderType != BORDER_TRANSPARENT &&
                           input->cols <= 1 ? BORDER_REPLICATE : borderType;
        int borderType_y = borderType != BORDER_CONSTANT &&
                           borderType != BORDER_TRANSPARENT &&
                           input->rows <= 1 ? BORDER_REPLICATE : borderType;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};

        int vlanes_32 = VTraits<v_float32>::vlanes();

        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        uint8_t bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4];
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
        uint8x8_t gray = {0, 8, 16, 24, 1, 9, 17, 25};
    #endif
#endif

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            v_float32 dst_x0 = vx_load(start_indices.data());
            v_float32 dst_x1 = v_add(dst_x0, vx_setall_f32(float(vlanes_32)));
            v_float32 M0 = vx_setall_f32(M[0]),
                      M3 = vx_setall_f32(M[3]);
            v_float32 M_x = vx_setall_f32(static_cast<float>(y * M[1] + M[2])),
                      M_y = vx_setall_f32(static_cast<float>(y * M[4] + M[5]));

            for (; x < dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                VECTOR_COMPUTE_COORDINATES();

                v_int32 addr_0 = v_fma(v_srcstep, src_iy0, src_ix0),
                        addr_1 = v_fma(v_srcstep, src_iy1, src_ix1);
                vx_store(addr, addr_0);
                vx_store(addr + vlanes_32, addr_1);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                uint8x8_t p00, p01, p10, p11;
    #endif

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    uint8x8x4_t t00 = {
                        vld1_u8(src + addr[0]),
                        vld1_u8(src + addr[1]),
                        vld1_u8(src + addr[2]),
                        vld1_u8(src + addr[3])
                    };

                    uint8x8x4_t t01 = {
                        vld1_u8(src + addr[4]),
                        vld1_u8(src + addr[5]),
                        vld1_u8(src + addr[6]),
                        vld1_u8(src + addr[7])
                    };

                    uint8x8x4_t t10 = {
                        vld1_u8(src + addr[0] + srcstep),
                        vld1_u8(src + addr[1] + srcstep),
                        vld1_u8(src + addr[2] + srcstep),
                        vld1_u8(src + addr[3] + srcstep)
                    };

                    uint8x8x4_t t11 = {
                        vld1_u8(src + addr[4] + srcstep),
                        vld1_u8(src + addr[5] + srcstep),
                        vld1_u8(src + addr[6] + srcstep),
                        vld1_u8(src + addr[7] + srcstep)
                    };

                    uint32x2_t p00_, p01_, p10_, p11_;

                    p00_ = vreinterpret_u32_u8(vtbl4_u8(t00, gray));
                    p01_ = vreinterpret_u32_u8(vtbl4_u8(t01, gray));
                    p10_ = vreinterpret_u32_u8(vtbl4_u8(t10, gray));
                    p11_ = vreinterpret_u32_u8(vtbl4_u8(t11, gray));

                    p00 = vreinterpret_u8_u32(vtrn1_u32(p00_, p01_));
                    p01 = vreinterpret_u8_u32(vtrn2_u32(p00_, p01_));
                    p10 = vreinterpret_u8_u32(vtrn1_u32(p10_, p11_));
                    p11 = vreinterpret_u8_u32(vtrn2_u32(p10_, p11_));
    #else
                    shuffle_allin_c1(src, addr, pixbuf, uf, srcstep);
    #endif
                } else {
                    VECTOR_LINEAR_SHUFFLE_NOTALLIN(8U, C1);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    p00 = vld1_u8(pixbuf);
                    p01 = vld1_u8(pixbuf + 8);
                    p10 = vld1_u8(pixbuf + 16);
                    p11 = vld1_u8(pixbuf + 24);
    #endif
                }

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16 // [TODO] support risc-v fp16 intrinsics
                v_float16 f00 = v_float16(vcvtq_f16_u16(vmovl_u8(p00)));
                v_float16 f01 = v_float16(vcvtq_f16_u16(vmovl_u8(p01)));
                v_float16 f10 = v_float16(vcvtq_f16_u16(vmovl_u8(p10)));
                v_float16 f11 = v_float16(vcvtq_f16_u16(vmovl_u8(p11)));

                v_float16 alpha = v_cvt_f16(src_x0, src_x1),
                          beta = v_cvt_f16(src_y0, src_y1);

                f00 = v_fma(alpha, v_sub(f01, f00), f00);
                f10 = v_fma(alpha, v_sub(f11, f10), f10);
                f00 = v_fma(beta,  v_sub(f10, f00), f00);

                uint8x8_t result = {
                    vqmovun_s16(vcvtnq_s16_f16(f00.val)),
                };

                vst1_u8(dstptr + x, result);
    #else
        #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 // In case neon fp16 intrinsics are not available; still requires A64
                v_int16 f00 = v_reinterpret_as_s16(v_uint16(vmovl_u8(p00))),
                        f01 = v_reinterpret_as_s16(v_uint16(vmovl_u8(p01))),
                        f10 = v_reinterpret_as_s16(v_uint16(vmovl_u8(p10))),
                        f11 = v_reinterpret_as_s16(v_uint16(vmovl_u8(p11)));
        #else
                v_int16  f00 = v_reinterpret_as_s16(vx_load_expand(pixbuf)),
                         f01 = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf)),
                         f10 = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*2)),
                         f11 = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*3));
        #endif
                v_float32 f00l = v_cvt_f32(v_expand_low(f00)), f00h = v_cvt_f32(v_expand_high(f00)),
                          f01l = v_cvt_f32(v_expand_low(f01)), f01h = v_cvt_f32(v_expand_high(f01)),
                          f10l = v_cvt_f32(v_expand_low(f10)), f10h = v_cvt_f32(v_expand_high(f10)),
                          f11l = v_cvt_f32(v_expand_low(f11)), f11h = v_cvt_f32(v_expand_high(f11));

                VECTOR_LINEAR_C1_F32();

                v_uint16 f00_u16 = v_pack_u(v_round(f00l), v_round(f00h));
                v_uint8 f00_u8 = v_pack(f00_u16, vx_setall_u16(0));
                v_store_low(dstptr + x, f00_u8);
    #endif
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00, p01, p10, p11;
                const uint8_t *srcptr = src + srcstep * iy + ix;

                if ((((unsigned)ix < (unsigned)(srccols-1)) &
                     ((unsigned)iy < (unsigned)(srcrows-1))) != 0) {
                    p00 = srcptr[0]; p01 = srcptr[1];
                    p10 = srcptr[srcstep + 0]; p11 = srcptr[srcstep + 1];
                } else {
                    if ((borderType == BORDER_CONSTANT || borderType == BORDER_TRANSPARENT) &&
                        (((unsigned)(ix+1) >= (unsigned)(srccols+1))|
                         ((unsigned)(iy+1) >= (unsigned)(srcrows+1))) != 0) {
                        if (borderType == BORDER_CONSTANT) {
                            dstptr[x] = bval[0];
                        }
                        continue;
                    }
                    SCALAR_FETCH_PIXEL_C1(0, 0, p00);
                    SCALAR_FETCH_PIXEL_C1(0, 1, p01);
                    SCALAR_FETCH_PIXEL_C1(1, 0, p10);
                    SCALAR_FETCH_PIXEL_C1(1, 1, p11);
                }
                float v0 = p00 + sx*(p01 - p00);
                float v1 = p10 + sx*(p11 - p10);
                v0 += sy*(v1 - v0);
                dstptr[x] = saturate_cast<uint8_t>(v0);
            }
        }
    }

private:
    Mat *output;
    const Mat *input;
    const double *dM;
    int borderType;
    const double *borderValue;
};

class WarpAffineLinearInvoker_16UC1 : public ParallelLoopBody {
public:
    WarpAffineLinearInvoker_16UC1(Mat *output_, const Mat *input_, const double *dM_,
                                 int borderType_, const double *borderValue_)
        : output(output_), input(input_), dM(dM_), borderType(borderType_), borderValue(borderValue_) {}

    virtual void operator() (const Range &r) const CV_OVERRIDE {
        CV_INSTRUMENT_REGION();

        auto *src = input->ptr<const uint16_t>();
        auto *dst = output->ptr<uint16_t>();
        size_t srcstep = input->step/sizeof(uint16_t), dststep = output->step/sizeof(uint16_t);
        int srccols = input->cols, srcrows = input->rows;
        int dstcols = output->cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint16_t bval[] = {
            saturate_cast<uint16_t>(borderValue[0]),
        };
        int borderType_x = borderType != BORDER_CONSTANT &&
                           borderType != BORDER_TRANSPARENT &&
                           input->cols <= 1 ? BORDER_REPLICATE : borderType;
        int borderType_y = borderType != BORDER_CONSTANT &&
                           borderType != BORDER_TRANSPARENT &&
                           input->rows <= 1 ? BORDER_REPLICATE : borderType;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};

        int vlanes_32 = VTraits<v_float32>::vlanes();

        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        uint16_t bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint16_t pixbuf[max_uf*4];
#endif

        for (int y = r.start; y < r.end; y++) {
            uint16_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            v_float32 dst_x0 = vx_load(start_indices.data());
            v_float32 dst_x1 = v_add(dst_x0, vx_setall_f32(float(vlanes_32)));
            v_float32 M0 = vx_setall_f32(M[0]),
                      M3 = vx_setall_f32(M[3]);
            v_float32 M_x = vx_setall_f32(static_cast<float>(y * M[1] + M[2])),
                      M_y = vx_setall_f32(static_cast<float>(y * M[4] + M[5]));

            for (; x < dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                VECTOR_COMPUTE_COORDINATES();

                v_int32 addr_0 = v_fma(v_srcstep, src_iy0, src_ix0),
                        addr_1 = v_fma(v_srcstep, src_iy1, src_ix1);
                vx_store(addr, addr_0);
                vx_store(addr + vlanes_32, addr_1);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    shuffle_allin_c1(src, addr, pixbuf, uf, srcstep);
                } else {
                    VECTOR_LINEAR_SHUFFLE_NOTALLIN(16U, C1);
                }

                v_uint16 f00 = vx_load(pixbuf),
                         f01 = vx_load(pixbuf + uf),
                         f10 = vx_load(pixbuf + uf*2),
                         f11 = vx_load(pixbuf + uf*3);

                v_float32 f00l = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f00))), f00h = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f00))),
                          f01l = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f01))), f01h = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f01))),
                          f10l = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f10))), f10h = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f10))),
                          f11l = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f11))), f11h = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f11)));

                VECTOR_LINEAR_C1_F32();

                v_uint16 f00_u16 = v_pack_u(v_round(f00l), v_round(f00h));
                v_store(dstptr + x, f00_u16);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00, p01, p10, p11;
                const uint16_t *srcptr = src + srcstep * iy + ix;

                if ((((unsigned)ix < (unsigned)(srccols-1)) &
                     ((unsigned)iy < (unsigned)(srcrows-1))) != 0) {
                    p00 = srcptr[0]; p01 = srcptr[1];
                    p10 = srcptr[srcstep + 0]; p11 = srcptr[srcstep + 1];
                } else {
                    if ((borderType == BORDER_CONSTANT || borderType == BORDER_TRANSPARENT) &&
                        (((unsigned)(ix+1) >= (unsigned)(srccols+1))|
                         ((unsigned)(iy+1) >= (unsigned)(srcrows+1))) != 0) {
                        if (borderType == BORDER_CONSTANT) {
                            dstptr[x] = bval[0];
                        }
                        continue;
                    }
                    SCALAR_FETCH_PIXEL_C1(0, 0, p00);
                    SCALAR_FETCH_PIXEL_C1(0, 1, p01);
                    SCALAR_FETCH_PIXEL_C1(1, 0, p10);
                    SCALAR_FETCH_PIXEL_C1(1, 1, p11);
                }
                float v0 = p00 + sx*(p01 - p00);
                float v1 = p10 + sx*(p11 - p10);
                v0 += sy*(v1 - v0);
                dstptr[x] = saturate_cast<uint16_t>(v0);
            }
        }
    }

private:
    Mat *output;
    const Mat *input;
    const double *dM;
    int borderType;
    const double *borderValue;
};

class WarpAffineLinearInvoker_32FC1 : public ParallelLoopBody {
public:
    WarpAffineLinearInvoker_32FC1(Mat *output_, const Mat *input_, const double *dM_,
                                 int borderType_, const double *borderValue_)
        : output(output_), input(input_), dM(dM_), borderType(borderType_), borderValue(borderValue_) {}

    virtual void operator() (const Range &r) const CV_OVERRIDE {
        CV_INSTRUMENT_REGION();

        auto *src = input->ptr<const float>();
        auto *dst = output->ptr<float>();
        size_t srcstep = input->step/sizeof(float), dststep = output->step/sizeof(float);
        int srccols = input->cols, srcrows = input->rows;
        int dstcols = output->cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        float bval[] = {
            saturate_cast<float>(borderValue[0]),
        };
        int borderType_x = borderType != BORDER_CONSTANT &&
                           borderType != BORDER_TRANSPARENT &&
                           input->cols <= 1 ? BORDER_REPLICATE : borderType;
        int borderType_y = borderType != BORDER_CONSTANT &&
                           borderType != BORDER_TRANSPARENT &&
                           input->rows <= 1 ? BORDER_REPLICATE : borderType;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};

        int vlanes_32 = VTraits<v_float32>::vlanes();

        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        float bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
        v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        float pixbuf[max_uf*4];
#endif

        for (int y = r.start; y < r.end; y++) {
            float* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            v_float32 dst_x0 = vx_load(start_indices.data());
            v_float32 dst_x1 = v_add(dst_x0, vx_setall_f32(float(vlanes_32)));
            v_float32 M0 = vx_setall_f32(M[0]),
                      M3 = vx_setall_f32(M[3]);
            v_float32 M_x = vx_setall_f32(static_cast<float>(y * M[1] + M[2])),
                      M_y = vx_setall_f32(static_cast<float>(y * M[4] + M[5]));

            for (; x < dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                VECTOR_COMPUTE_COORDINATES();

                v_int32 addr_0 = v_fma(v_srcstep, src_iy0, src_ix0),
                        addr_1 = v_fma(v_srcstep, src_iy1, src_ix1);
                vx_store(addr, addr_0);
                vx_store(addr + vlanes_32, addr_1);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    shuffle_allin_c1(src, addr, pixbuf, uf, srcstep);
                } else {
                    VECTOR_LINEAR_SHUFFLE_NOTALLIN(32F, C1);
                }

                v_float32 f00l = vx_load(pixbuf),         f00h = vx_load(pixbuf + vlanes_32),
                          f01l = vx_load(pixbuf + uf),    f01h = vx_load(pixbuf + uf + vlanes_32),
                          f10l = vx_load(pixbuf + uf*2),  f10h = vx_load(pixbuf + uf*2 + vlanes_32),
                          f11l = vx_load(pixbuf + uf*3),  f11h = vx_load(pixbuf + uf*3 + vlanes_32);

                VECTOR_LINEAR_C1_F32();

                vx_store(dstptr + x, f00l);
                vx_store(dstptr + x + vlanes_32, f00h);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                float p00, p01, p10, p11;
                const float *srcptr = src + srcstep * iy + ix;

                if ((((unsigned)ix < (unsigned)(srccols-1)) &
                     ((unsigned)iy < (unsigned)(srcrows-1))) != 0) {
                    p00 = srcptr[0]; p01 = srcptr[1];
                    p10 = srcptr[srcstep + 0]; p11 = srcptr[srcstep + 1];
                } else {
                    if ((borderType == BORDER_CONSTANT || borderType == BORDER_TRANSPARENT) &&
                        (((unsigned)(ix+1) >= (unsigned)(srccols+1))|
                         ((unsigned)(iy+1) >= (unsigned)(srcrows+1))) != 0) {
                        if (borderType == BORDER_CONSTANT) {
                            dstptr[x] = bval[0];
                        }
                        continue;
                    }
                    SCALAR_FETCH_PIXEL_C1(0, 0, p00);
                    SCALAR_FETCH_PIXEL_C1(0, 1, p01);
                    SCALAR_FETCH_PIXEL_C1(1, 0, p10);
                    SCALAR_FETCH_PIXEL_C1(1, 1, p11);
                }
                float v0 = p00 + sx*(p01 - p00);
                float v1 = p10 + sx*(p11 - p10);
                v0 += sy*(v1 - v0);
                dstptr[x] = v0;
            }
        }
    }

private:
    Mat *output;
    const Mat *input;
    const double *dM;
    int borderType;
    const double *borderValue;
};

class WarpAffineLinearInvoker_8UC4 : public ParallelLoopBody {
public:
    WarpAffineLinearInvoker_8UC4(Mat *output_, const Mat *input_, const double *dM_,
                                 int borderType_, const double *borderValue_)
        : output(output_), input(input_), dM(dM_), borderType(borderType_), borderValue(borderValue_) {}

    virtual void operator() (const Range &r) const CV_OVERRIDE {
        CV_INSTRUMENT_REGION();

        auto *src = input->ptr<const uint8_t>();
        auto *dst = output->ptr<uint8_t>();
        size_t srcstep = input->step, dststep = output->step;
        int srccols = input->cols, srcrows = input->rows;
        int dstcols = output->cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(borderValue[0]),
            saturate_cast<uint8_t>(borderValue[1]),
            saturate_cast<uint8_t>(borderValue[2]),
            saturate_cast<uint8_t>(borderValue[3]),
        };
        int borderType_x = borderType != BORDER_CONSTANT &&
                           borderType != BORDER_TRANSPARENT &&
                           input->cols <= 1 ? BORDER_REPLICATE : borderType;
        int borderType_y = borderType != BORDER_CONSTANT &&
                           borderType != BORDER_TRANSPARENT &&
                           input->rows <= 1 ? BORDER_REPLICATE : borderType;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};

        int vlanes_32 = VTraits<v_float32>::vlanes();

        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        uint8_t bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
        v_uint8 bval_v3 = vx_load_low(&bvalbuf[uf*3]);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4*4];
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
        uint8x8_t reds = {0, 8, 16, 24, 4, 12, 20, 28},
                  greens = {1, 9, 17, 25, 5, 13, 21, 29},
                  blues = {2, 10, 18, 26, 6, 14, 22, 30},
                  alphas = {3, 11, 19, 27, 7, 15, 23, 31};
    #endif
    #if !CV_SIMD128_FP16
        constexpr int max_vlanes_16{VTraits<v_uint16>::max_nlanes};
        int vlanes_16 = VTraits<v_uint16>::vlanes();
    #endif
#endif

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            v_float32 dst_x0 = vx_load(start_indices.data());
            v_float32 dst_x1 = v_add(dst_x0, vx_setall_f32(float(vlanes_32)));
            v_float32 M0 = vx_setall_f32(M[0]),
                      M3 = vx_setall_f32(M[3]);
            v_float32 M_x = vx_setall_f32(static_cast<float>(y * M[1] + M[2])),
                      M_y = vx_setall_f32(static_cast<float>(y * M[4] + M[5]));

            for (; x < dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                VECTOR_COMPUTE_COORDINATES();

                v_int32 addr_0 = v_fma(v_srcstep, src_iy0, v_mul(src_ix0, four)),
                        addr_1 = v_fma(v_srcstep, src_iy1, v_mul(src_ix1, four));
                vx_store(addr, addr_0);
                vx_store(addr + vlanes_32, addr_1);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b,
                          p00a, p01a, p10a, p11a;
    #endif

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    uint8x8x4_t p00 = {
                        vld1_u8(src + addr[0]),
                        vld1_u8(src + addr[1]),
                        vld1_u8(src + addr[2]),
                        vld1_u8(src + addr[3])
                    };

                    uint8x8x4_t p01 = {
                        vld1_u8(src + addr[4]),
                        vld1_u8(src + addr[5]),
                        vld1_u8(src + addr[6]),
                        vld1_u8(src + addr[7])
                    };

                    uint8x8x4_t p10 = {
                        vld1_u8(src + addr[0] + srcstep),
                        vld1_u8(src + addr[1] + srcstep),
                        vld1_u8(src + addr[2] + srcstep),
                        vld1_u8(src + addr[3] + srcstep)
                    };

                    uint8x8x4_t p11 = {
                        vld1_u8(src + addr[4] + srcstep),
                        vld1_u8(src + addr[5] + srcstep),
                        vld1_u8(src + addr[6] + srcstep),
                        vld1_u8(src + addr[7] + srcstep)
                    };

                    uint32x2_t p00_, p01_, p10_, p11_;

                    p00_ = vreinterpret_u32_u8(vtbl4_u8(p00, reds));
                    p01_ = vreinterpret_u32_u8(vtbl4_u8(p01, reds));
                    p10_ = vreinterpret_u32_u8(vtbl4_u8(p10, reds));
                    p11_ = vreinterpret_u32_u8(vtbl4_u8(p11, reds));

                    p00r = vreinterpret_u8_u32(vtrn1_u32(p00_, p01_));
                    p01r = vreinterpret_u8_u32(vtrn2_u32(p00_, p01_));
                    p10r = vreinterpret_u8_u32(vtrn1_u32(p10_, p11_));
                    p11r = vreinterpret_u8_u32(vtrn2_u32(p10_, p11_));

                    p00_ = vreinterpret_u32_u8(vtbl4_u8(p00, greens));
                    p01_ = vreinterpret_u32_u8(vtbl4_u8(p01, greens));
                    p10_ = vreinterpret_u32_u8(vtbl4_u8(p10, greens));
                    p11_ = vreinterpret_u32_u8(vtbl4_u8(p11, greens));

                    p00g = vreinterpret_u8_u32(vtrn1_u32(p00_, p01_));
                    p01g = vreinterpret_u8_u32(vtrn2_u32(p00_, p01_));
                    p10g = vreinterpret_u8_u32(vtrn1_u32(p10_, p11_));
                    p11g = vreinterpret_u8_u32(vtrn2_u32(p10_, p11_));

                    p00_ = vreinterpret_u32_u8(vtbl4_u8(p00, blues));
                    p01_ = vreinterpret_u32_u8(vtbl4_u8(p01, blues));
                    p10_ = vreinterpret_u32_u8(vtbl4_u8(p10, blues));
                    p11_ = vreinterpret_u32_u8(vtbl4_u8(p11, blues));

                    p00b = vreinterpret_u8_u32(vtrn1_u32(p00_, p01_));
                    p01b = vreinterpret_u8_u32(vtrn2_u32(p00_, p01_));
                    p10b = vreinterpret_u8_u32(vtrn1_u32(p10_, p11_));
                    p11b = vreinterpret_u8_u32(vtrn2_u32(p10_, p11_));

                    p00_ = vreinterpret_u32_u8(vtbl4_u8(p00, alphas));
                    p01_ = vreinterpret_u32_u8(vtbl4_u8(p01, alphas));
                    p10_ = vreinterpret_u32_u8(vtbl4_u8(p10, alphas));
                    p11_ = vreinterpret_u32_u8(vtbl4_u8(p11, alphas));

                    p00a = vreinterpret_u8_u32(vtrn1_u32(p00_, p01_));
                    p01a = vreinterpret_u8_u32(vtrn2_u32(p00_, p01_));
                    p10a = vreinterpret_u8_u32(vtrn1_u32(p10_, p11_));
                    p11a = vreinterpret_u8_u32(vtrn2_u32(p10_, p11_));
    #else
                    shuffle_allin_c4(src, addr, pixbuf, uf, srcstep);
    #endif
                } else {
                    VECTOR_LINEAR_SHUFFLE_NOTALLIN(8U, C4);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    p00r = vld1_u8(pixbuf);
                    p01r = vld1_u8(pixbuf + 8);
                    p10r = vld1_u8(pixbuf + 16);
                    p11r = vld1_u8(pixbuf + 24);

                    p00g = vld1_u8(pixbuf + 32);
                    p01g = vld1_u8(pixbuf + 32 + 8);
                    p10g = vld1_u8(pixbuf + 32 + 16);
                    p11g = vld1_u8(pixbuf + 32 + 24);

                    p00b = vld1_u8(pixbuf + 64);
                    p01b = vld1_u8(pixbuf + 64 + 8);
                    p10b = vld1_u8(pixbuf + 64 + 16);
                    p11b = vld1_u8(pixbuf + 64 + 24);

                    p00a = vld1_u8(pixbuf + 96);
                    p01a = vld1_u8(pixbuf + 96 + 8);
                    p10a = vld1_u8(pixbuf + 96 + 16);
                    p11a = vld1_u8(pixbuf + 96 + 24);
    #endif
                }

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16 // [TODO] support risc-v fp16 intrinsics
                v_float16 f00r = v_float16(vcvtq_f16_u16(vmovl_u8(p00r)));
                v_float16 f01r = v_float16(vcvtq_f16_u16(vmovl_u8(p01r)));
                v_float16 f10r = v_float16(vcvtq_f16_u16(vmovl_u8(p10r)));
                v_float16 f11r = v_float16(vcvtq_f16_u16(vmovl_u8(p11r)));

                v_float16 f00g = v_float16(vcvtq_f16_u16(vmovl_u8(p00g)));
                v_float16 f01g = v_float16(vcvtq_f16_u16(vmovl_u8(p01g)));
                v_float16 f10g = v_float16(vcvtq_f16_u16(vmovl_u8(p10g)));
                v_float16 f11g = v_float16(vcvtq_f16_u16(vmovl_u8(p11g)));

                v_float16 f00b = v_float16(vcvtq_f16_u16(vmovl_u8(p00b)));
                v_float16 f01b = v_float16(vcvtq_f16_u16(vmovl_u8(p01b)));
                v_float16 f10b = v_float16(vcvtq_f16_u16(vmovl_u8(p10b)));
                v_float16 f11b = v_float16(vcvtq_f16_u16(vmovl_u8(p11b)));

                v_float16 f00a = v_float16(vcvtq_f16_u16(vmovl_u8(p00a)));
                v_float16 f01a = v_float16(vcvtq_f16_u16(vmovl_u8(p01a)));
                v_float16 f10a = v_float16(vcvtq_f16_u16(vmovl_u8(p10a)));
                v_float16 f11a = v_float16(vcvtq_f16_u16(vmovl_u8(p11a)));

                v_float16 alpha = v_cvt_f16(src_x0, src_x1),
                          beta = v_cvt_f16(src_y0, src_y1);

                f00r = v_fma(alpha, v_sub(f01r, f00r), f00r);
                f10r = v_fma(alpha, v_sub(f11r, f10r), f10r);

                f00g = v_fma(alpha, v_sub(f01g, f00g), f00g);
                f10g = v_fma(alpha, v_sub(f11g, f10g), f10g);

                f00b = v_fma(alpha, v_sub(f01b, f00b), f00b);
                f10b = v_fma(alpha, v_sub(f11b, f10b), f10b);

                f00a = v_fma(alpha, v_sub(f01a, f00a), f00a);
                f10a = v_fma(alpha, v_sub(f11a, f10a), f10a);

                f00r = v_fma(beta,  v_sub(f10r, f00r), f00r);
                f00g = v_fma(beta,  v_sub(f10g, f00g), f00g);
                f00b = v_fma(beta,  v_sub(f10b, f00b), f00b);
                f00a = v_fma(beta,  v_sub(f10a, f00a), f00a);

                uint8x8x4_t result = {
                    vqmovun_s16(vcvtnq_s16_f16(f00r.val)),
                    vqmovun_s16(vcvtnq_s16_f16(f00g.val)),
                    vqmovun_s16(vcvtnq_s16_f16(f00b.val)),
                    vqmovun_s16(vcvtnq_s16_f16(f00a.val)),
                };
                vst4_u8(dstptr + x*4, result);
    #else
        #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 // In case neon fp16 intrinsics are not available; still requires A64
                v_int16 f00r = v_reinterpret_as_s16(v_uint16(vmovl_u8(p00r))),
                        f01r = v_reinterpret_as_s16(v_uint16(vmovl_u8(p01r))),
                        f10r = v_reinterpret_as_s16(v_uint16(vmovl_u8(p10r))),
                        f11r = v_reinterpret_as_s16(v_uint16(vmovl_u8(p11r)));
                v_int16 f00g = v_reinterpret_as_s16(v_uint16(vmovl_u8(p00g))),
                        f01g = v_reinterpret_as_s16(v_uint16(vmovl_u8(p01g))),
                        f10g = v_reinterpret_as_s16(v_uint16(vmovl_u8(p10g))),
                        f11g = v_reinterpret_as_s16(v_uint16(vmovl_u8(p11g)));
                v_int16 f00b = v_reinterpret_as_s16(v_uint16(vmovl_u8(p00b))),
                        f01b = v_reinterpret_as_s16(v_uint16(vmovl_u8(p01b))),
                        f10b = v_reinterpret_as_s16(v_uint16(vmovl_u8(p10b))),
                        f11b = v_reinterpret_as_s16(v_uint16(vmovl_u8(p11b)));
                v_int16 f00a = v_reinterpret_as_s16(v_uint16(vmovl_u8(p00a))),
                        f01a = v_reinterpret_as_s16(v_uint16(vmovl_u8(p01a))),
                        f10a = v_reinterpret_as_s16(v_uint16(vmovl_u8(p10a))),
                        f11a = v_reinterpret_as_s16(v_uint16(vmovl_u8(p11a)));
        #else
                v_int16  f00r = v_reinterpret_as_s16(vx_load_expand(pixbuf)),
                         f01r = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf)),
                         f10r = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*2)),
                         f11r = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*3));
                v_int16  f00g = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*4)),
                         f01g = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*5)),
                         f10g = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*6)),
                         f11g = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*7));
                v_int16  f00b = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*8)),
                         f01b = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*9)),
                         f10b = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*10)),
                         f11b = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*11));
                v_int16  f00a = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*12)),
                         f01a = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*13)),
                         f10a = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*14)),
                         f11a = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf*15));
        #endif
                v_float32 f00rl = v_cvt_f32(v_expand_low(f00r)), f00rh = v_cvt_f32(v_expand_high(f00r)),
                          f01rl = v_cvt_f32(v_expand_low(f01r)), f01rh = v_cvt_f32(v_expand_high(f01r)),
                          f10rl = v_cvt_f32(v_expand_low(f10r)), f10rh = v_cvt_f32(v_expand_high(f10r)),
                          f11rl = v_cvt_f32(v_expand_low(f11r)), f11rh = v_cvt_f32(v_expand_high(f11r));
                v_float32 f00gl = v_cvt_f32(v_expand_low(f00g)), f00gh = v_cvt_f32(v_expand_high(f00g)),
                          f01gl = v_cvt_f32(v_expand_low(f01g)), f01gh = v_cvt_f32(v_expand_high(f01g)),
                          f10gl = v_cvt_f32(v_expand_low(f10g)), f10gh = v_cvt_f32(v_expand_high(f10g)),
                          f11gl = v_cvt_f32(v_expand_low(f11g)), f11gh = v_cvt_f32(v_expand_high(f11g));
                v_float32 f00bl = v_cvt_f32(v_expand_low(f00b)), f00bh = v_cvt_f32(v_expand_high(f00b)),
                          f01bl = v_cvt_f32(v_expand_low(f01b)), f01bh = v_cvt_f32(v_expand_high(f01b)),
                          f10bl = v_cvt_f32(v_expand_low(f10b)), f10bh = v_cvt_f32(v_expand_high(f10b)),
                          f11bl = v_cvt_f32(v_expand_low(f11b)), f11bh = v_cvt_f32(v_expand_high(f11b));
                v_float32 f00al = v_cvt_f32(v_expand_low(f00a)), f00ah = v_cvt_f32(v_expand_high(f00a)),
                          f01al = v_cvt_f32(v_expand_low(f01a)), f01ah = v_cvt_f32(v_expand_high(f01a)),
                          f10al = v_cvt_f32(v_expand_low(f10a)), f10ah = v_cvt_f32(v_expand_high(f10a)),
                          f11al = v_cvt_f32(v_expand_low(f11a)), f11ah = v_cvt_f32(v_expand_high(f11a));

                VECTOR_LINEAR_C4_F32();

                v_uint16 f00r_u16 = v_pack_u(v_round(f00rl), v_round(f00rh)),
                         f00g_u16 = v_pack_u(v_round(f00gl), v_round(f00gh)),
                         f00b_u16 = v_pack_u(v_round(f00bl), v_round(f00bh)),
                         f00a_u16 = v_pack_u(v_round(f00al), v_round(f00ah));
                uint16_t tbuf[max_vlanes_16*4];
                v_store_interleave(tbuf, f00r_u16, f00g_u16, f00b_u16, f00a_u16);
                v_pack_store(dstptr + x*4, vx_load(tbuf));
                v_pack_store(dstptr + x*4 + vlanes_16, vx_load(tbuf + vlanes_16));
                v_pack_store(dstptr + x*4 + vlanes_16*2, vx_load(tbuf + vlanes_16*2));
                v_pack_store(dstptr + x*4 + vlanes_16*3, vx_load(tbuf + vlanes_16*3));
    #endif // defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00r, p00g, p00b, p00a, p01r, p01g, p01b, p01a;
                int p10r, p10g, p10b, p10a, p11r, p11g, p11b, p11a;
                const uint8_t* srcptr = src + srcstep*iy + ix*4;

                SCALAR_LINEAR_SHUFFLE_C4();

                SCALAR_LINEAR_CALC_C4();

                dstptr[x*4] = saturate_cast<uint8_t>(v0r);
                dstptr[x*4+1] = saturate_cast<uint8_t>(v0g);
                dstptr[x*4+2] = saturate_cast<uint8_t>(v0b);
                dstptr[x*4+3] = saturate_cast<uint8_t>(v0a);
            }
        }
    }

private:
    Mat *output;
    const Mat *input;
    const double *dM;
    int borderType;
    const double *borderValue;
};

class WarpAffineLinearInvoker_16UC4 : public ParallelLoopBody {
public:
    WarpAffineLinearInvoker_16UC4(Mat *output_, const Mat *input_, const double *dM_,
                                 int borderType_, const double *borderValue_)
        : output(output_), input(input_), dM(dM_), borderType(borderType_), borderValue(borderValue_) {}

    virtual void operator() (const Range &r) const CV_OVERRIDE {
        CV_INSTRUMENT_REGION();

        auto *src = input->ptr<const uint16_t>();
        auto *dst = output->ptr<uint16_t>();
        size_t srcstep = input->step/sizeof(uint16_t), dststep = output->step/sizeof(uint16_t);
        int srccols = input->cols, srcrows = input->rows;
        int dstcols = output->cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint16_t bval[] = {
            saturate_cast<uint16_t>(borderValue[0]),
            saturate_cast<uint16_t>(borderValue[1]),
            saturate_cast<uint16_t>(borderValue[2]),
            saturate_cast<uint16_t>(borderValue[3]),
        };
        int borderType_x = borderType != BORDER_CONSTANT &&
                           borderType != BORDER_TRANSPARENT &&
                           input->cols <= 1 ? BORDER_REPLICATE : borderType;
        int borderType_y = borderType != BORDER_CONSTANT &&
                           borderType != BORDER_TRANSPARENT &&
                           input->rows <= 1 ? BORDER_REPLICATE : borderType;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};

        int vlanes_32 = VTraits<v_float32>::vlanes();

        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        uint16_t bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
        v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
        v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
        v_uint16 bval_v3 = vx_load(&bvalbuf[uf*3]);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint16_t pixbuf[max_uf*4*4];
#endif

        for (int y = r.start; y < r.end; y++) {
            uint16_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            v_float32 dst_x0 = vx_load(start_indices.data());
            v_float32 dst_x1 = v_add(dst_x0, vx_setall_f32(float(vlanes_32)));
            v_float32 M0 = vx_setall_f32(M[0]),
                      M3 = vx_setall_f32(M[3]);
            v_float32 M_x = vx_setall_f32(static_cast<float>(y * M[1] + M[2])),
                      M_y = vx_setall_f32(static_cast<float>(y * M[4] + M[5]));

            for (; x < dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                VECTOR_COMPUTE_COORDINATES();

                v_int32 addr_0 = v_fma(v_srcstep, src_iy0, v_mul(src_ix0, four)),
                        addr_1 = v_fma(v_srcstep, src_iy1, v_mul(src_ix1, four));
                vx_store(addr, addr_0);
                vx_store(addr + vlanes_32, addr_1);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    shuffle_allin_c4(src, addr, pixbuf, uf, srcstep);
                } else {
                    VECTOR_LINEAR_SHUFFLE_NOTALLIN(16U, C4);
                }

                v_uint16 f00r = vx_load(pixbuf),
                         f01r = vx_load(pixbuf + uf),
                         f10r = vx_load(pixbuf + uf*2),
                         f11r = vx_load(pixbuf + uf*3);
                v_uint16 f00g = vx_load(pixbuf + uf*4),
                         f01g = vx_load(pixbuf + uf*5),
                         f10g = vx_load(pixbuf + uf*6),
                         f11g = vx_load(pixbuf + uf*7);
                v_uint16 f00b = vx_load(pixbuf + uf*8),
                         f01b = vx_load(pixbuf + uf*9),
                         f10b = vx_load(pixbuf + uf*10),
                         f11b = vx_load(pixbuf + uf*11);
                v_uint16 f00a = vx_load(pixbuf + uf*12),
                         f01a = vx_load(pixbuf + uf*13),
                         f10a = vx_load(pixbuf + uf*14),
                         f11a = vx_load(pixbuf + uf*15);

                v_float32 f00rl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f00r))), f00rh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f00r))),
                          f01rl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f01r))), f01rh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f01r))),
                          f10rl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f10r))), f10rh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f10r))),
                          f11rl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f11r))), f11rh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f11r)));
                v_float32 f00gl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f00g))), f00gh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f00g))),
                          f01gl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f01g))), f01gh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f01g))),
                          f10gl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f10g))), f10gh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f10g))),
                          f11gl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f11g))), f11gh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f11g)));
                v_float32 f00bl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f00b))), f00bh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f00b))),
                          f01bl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f01b))), f01bh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f01b))),
                          f10bl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f10b))), f10bh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f10b))),
                          f11bl = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f11b))), f11bh = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f11b)));
                v_float32 f00al = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f00a))), f00ah = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f00a))),
                          f01al = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f01a))), f01ah = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f01a))),
                          f10al = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f10a))), f10ah = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f10a))),
                          f11al = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f11a))), f11ah = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f11a)));

                VECTOR_LINEAR_C4_F32();

                v_uint16 f00r_u16 = v_pack_u(v_round(f00rl), v_round(f00rh)),
                         f00g_u16 = v_pack_u(v_round(f00gl), v_round(f00gh)),
                         f00b_u16 = v_pack_u(v_round(f00bl), v_round(f00bh)),
                         f00a_u16 = v_pack_u(v_round(f00al), v_round(f00ah));
                v_store_interleave(dstptr + x*4, f00r_u16, f00g_u16, f00b_u16, f00a_u16);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00r, p00g, p00b, p00a, p01r, p01g, p01b, p01a;
                int p10r, p10g, p10b, p10a, p11r, p11g, p11b, p11a;
                const uint16_t* srcptr = src + srcstep*iy + ix*4;

                SCALAR_LINEAR_SHUFFLE_C4();

                SCALAR_LINEAR_CALC_C4();

                dstptr[x*4] =   saturate_cast<uint16_t>(v0r);
                dstptr[x*4+1] = saturate_cast<uint16_t>(v0g);
                dstptr[x*4+2] = saturate_cast<uint16_t>(v0b);
                dstptr[x*4+3] = saturate_cast<uint16_t>(v0a);
            }
        }
    }

private:
    Mat *output;
    const Mat *input;
    const double *dM;
    int borderType;
    const double *borderValue;
};

class WarpAffineLinearInvoker_32FC4 : public ParallelLoopBody {
public:
    WarpAffineLinearInvoker_32FC4(Mat *output_, const Mat *input_, const double *dM_,
                                 int borderType_, const double *borderValue_)
        : output(output_), input(input_), dM(dM_), borderType(borderType_), borderValue(borderValue_) {}

    virtual void operator() (const Range &r) const CV_OVERRIDE {
        CV_INSTRUMENT_REGION();

        auto *src = input->ptr<const float>();
        auto *dst = output->ptr<float>();
        size_t srcstep = input->step/sizeof(float), dststep = output->step/sizeof(float);
        int srccols = input->cols, srcrows = input->rows;
        int dstcols = output->cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        float bval[] = {
            saturate_cast<float>(borderValue[0]),
            saturate_cast<float>(borderValue[1]),
            saturate_cast<float>(borderValue[2]),
            saturate_cast<float>(borderValue[3]),
        };
        int borderType_x = borderType != BORDER_CONSTANT &&
                           borderType != BORDER_TRANSPARENT &&
                           input->cols <= 1 ? BORDER_REPLICATE : borderType;
        int borderType_y = borderType != BORDER_CONSTANT &&
                           borderType != BORDER_TRANSPARENT &&
                           input->rows <= 1 ? BORDER_REPLICATE : borderType;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};

        int vlanes_32 = VTraits<v_float32>::vlanes();

        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        float bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
        v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
        v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
        v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
        v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
        v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
        v_float32 bval_v3_l = vx_load(&bvalbuf[uf*3]);
        v_float32 bval_v3_h = vx_load(&bvalbuf[uf*3+vlanes_32]);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        float pixbuf[max_uf*4*4];
#endif

        for (int y = r.start; y < r.end; y++) {
            float* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            v_float32 dst_x0 = vx_load(start_indices.data());
            v_float32 dst_x1 = v_add(dst_x0, vx_setall_f32(float(vlanes_32)));
            v_float32 M0 = vx_setall_f32(M[0]),
                      M3 = vx_setall_f32(M[3]);
            v_float32 M_x = vx_setall_f32(static_cast<float>(y * M[1] + M[2])),
                      M_y = vx_setall_f32(static_cast<float>(y * M[4] + M[5]));

            for (; x < dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                VECTOR_COMPUTE_COORDINATES();

                v_int32 addr_0 = v_fma(v_srcstep, src_iy0, v_mul(src_ix0, four)),
                        addr_1 = v_fma(v_srcstep, src_iy1, v_mul(src_ix1, four));
                vx_store(addr, addr_0);
                vx_store(addr + vlanes_32, addr_1);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    shuffle_allin_c4(src, addr, pixbuf, uf, srcstep);
                } else {
                    VECTOR_LINEAR_SHUFFLE_NOTALLIN(32F, C4);
                }

                v_float32 f00rl = vx_load(pixbuf),         f00rh = vx_load(pixbuf + vlanes_32),
                          f01rl = vx_load(pixbuf + uf),    f01rh = vx_load(pixbuf + uf + vlanes_32),
                          f10rl = vx_load(pixbuf + uf*2),  f10rh = vx_load(pixbuf + uf*2 + vlanes_32),
                          f11rl = vx_load(pixbuf + uf*3),  f11rh = vx_load(pixbuf + uf*3 + vlanes_32);
                v_float32 f00gl = vx_load(pixbuf + uf*4),  f00gh = vx_load(pixbuf + uf*4 + vlanes_32),
                          f01gl = vx_load(pixbuf + uf*5),  f01gh = vx_load(pixbuf + uf*5 + vlanes_32),
                          f10gl = vx_load(pixbuf + uf*6),  f10gh = vx_load(pixbuf + uf*6 + vlanes_32),
                          f11gl = vx_load(pixbuf + uf*7),  f11gh = vx_load(pixbuf + uf*7 + vlanes_32);
                v_float32 f00bl = vx_load(pixbuf + uf*8),  f00bh = vx_load(pixbuf + uf*8 + vlanes_32),
                          f01bl = vx_load(pixbuf + uf*9),  f01bh = vx_load(pixbuf + uf*9 + vlanes_32),
                          f10bl = vx_load(pixbuf + uf*10), f10bh = vx_load(pixbuf + uf*10 + vlanes_32),
                          f11bl = vx_load(pixbuf + uf*11), f11bh = vx_load(pixbuf + uf*11 + vlanes_32);
                v_float32 f00al = vx_load(pixbuf + uf*12), f00ah = vx_load(pixbuf + uf*12 + vlanes_32),
                          f01al = vx_load(pixbuf + uf*13), f01ah = vx_load(pixbuf + uf*13 + vlanes_32),
                          f10al = vx_load(pixbuf + uf*14), f10ah = vx_load(pixbuf + uf*14 + vlanes_32),
                          f11al = vx_load(pixbuf + uf*15), f11ah = vx_load(pixbuf + uf*15 + vlanes_32);

                VECTOR_LINEAR_C4_F32();

                v_store_interleave(dstptr + x*4, f00rl, f00gl, f00bl, f00al);
                v_store_interleave(dstptr + x*4 + vlanes_32*4, f00rh, f00gh, f00bh, f00ah);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                float p00r, p00g, p00b, p00a, p01r, p01g, p01b, p01a;
                float p10r, p10g, p10b, p10a, p11r, p11g, p11b, p11a;
                const float* srcptr = src + srcstep*iy + ix*4;

                SCALAR_LINEAR_SHUFFLE_C4();

                SCALAR_LINEAR_CALC_C4();

                dstptr[x*4] =   v0r;
                dstptr[x*4+1] = v0g;
                dstptr[x*4+2] = v0b;
                dstptr[x*4+3] = v0a;
            }
        }
    }

private:
    Mat *output;
    const Mat *input;
    const double *dM;
    int borderType;
    const double *borderValue;
};

} // anonymous

void warpAffineSimdInvoker(Mat &output, const Mat &input, const double *M,
                           int interpolation, int borderType, const double *borderValue) {
    CV_INSTRUMENT_REGION();

    CV_CheckEQ(interpolation, INTER_LINEAR, "");
    switch (output.type()) {
        case CV_8UC3: {
            WarpAffineLinearInvoker_8UC3 body(&output, &input, M, borderType, borderValue);
            parallel_for_(Range(0, output.rows), body);
            break;
        }
        case CV_16UC3: {
            WarpAffineLinearInvoker_16UC3 body(&output, &input, M, borderType, borderValue);
            parallel_for_(Range(0, output.rows), body);
            break;
        }
        case CV_32FC3: {
            WarpAffineLinearInvoker_32FC3 body(&output, &input, M, borderType, borderValue);
            parallel_for_(Range(0, output.rows), body);
            break;
        }
        case CV_8UC1: {
            WarpAffineLinearInvoker_8UC1 body(&output, &input, M, borderType, borderValue);
            parallel_for_(Range(0, output.rows), body);
            break;
        }
        case CV_16UC1: {
            WarpAffineLinearInvoker_16UC1 body(&output, &input, M, borderType, borderValue);
            parallel_for_(Range(0, output.rows), body);
            break;
        }
        case CV_32FC1: {
            WarpAffineLinearInvoker_32FC1 body(&output, &input, M, borderType, borderValue);
            parallel_for_(Range(0, output.rows), body);
            break;
        }
        case CV_8UC4: {
            WarpAffineLinearInvoker_8UC4 body(&output, &input, M, borderType, borderValue);
            parallel_for_(Range(0, output.rows), body);
            break;
        }
        case CV_16UC4: {
            WarpAffineLinearInvoker_16UC4 body(&output, &input, M, borderType, borderValue);
            parallel_for_(Range(0, output.rows), body);
            break;
        }
        case CV_32FC4: {
            WarpAffineLinearInvoker_32FC4 body(&output, &input, M, borderType, borderValue);
            parallel_for_(Range(0, output.rows), body);
            break;
        }
        default: CV_Error(Error::StsBadArg, "Unsupported type");
    }
}
#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY


CV_CPU_OPTIMIZATION_NAMESPACE_END
} // cv
