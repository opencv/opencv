// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Shuffle (all pixels within image)
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_C1(dtype) \
    for (int i = 0; i < uf; i++) { \
        const dtype* srcptr = src + addr[i]; \
        pixbuf[i] = srcptr[0]; \
        pixbuf[i + uf] = srcptr[1]; \
        pixbuf[i + uf*2] = srcptr[srcstep]; \
        pixbuf[i + uf*3] = srcptr[srcstep + 1]; \
    }
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_C3(dtype) \
    for (int i = 0; i < uf; i++) { \
        const dtype* srcptr = src + addr[i]; \
        pixbuf[i] = srcptr[0]; \
        pixbuf[i + uf*4] = srcptr[1]; \
        pixbuf[i + uf*8] = srcptr[2]; \
        pixbuf[i + uf] = srcptr[3]; \
        pixbuf[i + uf*5] = srcptr[4]; \
        pixbuf[i + uf*9] = srcptr[5]; \
        pixbuf[i + uf*2] = srcptr[srcstep]; \
        pixbuf[i + uf*6] = srcptr[srcstep + 1]; \
        pixbuf[i + uf*10] = srcptr[srcstep + 2]; \
        pixbuf[i + uf*3] = srcptr[srcstep + 3]; \
        pixbuf[i + uf*7] = srcptr[srcstep + 4]; \
        pixbuf[i + uf*11] = srcptr[srcstep + 5]; \
    }
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_C4(dtype) \
    for (int i = 0; i < uf; i++) { \
        const dtype* srcptr = src + addr[i]; \
        pixbuf[i] = srcptr[0]; \
        pixbuf[i + uf*4] = srcptr[1]; \
        pixbuf[i + uf*8] = srcptr[2]; \
        pixbuf[i + uf*12] = srcptr[3]; \
        pixbuf[i + uf] = srcptr[4]; \
        pixbuf[i + uf*5] = srcptr[5]; \
        pixbuf[i + uf*9] = srcptr[6]; \
        pixbuf[i + uf*13] = srcptr[7]; \
        pixbuf[i + uf*2] = srcptr[srcstep]; \
        pixbuf[i + uf*6] = srcptr[srcstep + 1]; \
        pixbuf[i + uf*10] = srcptr[srcstep + 2]; \
        pixbuf[i + uf*14] = srcptr[srcstep + 3]; \
        pixbuf[i + uf*3] = srcptr[srcstep + 4]; \
        pixbuf[i + uf*7] = srcptr[srcstep + 5]; \
        pixbuf[i + uf*11] = srcptr[srcstep + 6]; \
        pixbuf[i + uf*15] = srcptr[srcstep + 7]; \
    }
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_8U(CN) \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_##CN(uint8_t)
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_16U(CN) \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_##CN(uint16_t)
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_32F(CN) \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_##CN(float)

#define CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(CN, DEPTH) \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_##DEPTH(CN)

// Shuffle (ARM NEON)
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8_LOAD() \
    uint8x8x4_t t00 = { \
        vld1_u8(src + addr[0]), \
        vld1_u8(src + addr[1]), \
        vld1_u8(src + addr[2]), \
        vld1_u8(src + addr[3]) \
    }; \
    uint8x8x4_t t01 = { \
        vld1_u8(src + addr[4]), \
        vld1_u8(src + addr[5]), \
        vld1_u8(src + addr[6]), \
        vld1_u8(src + addr[7]) \
    }; \
    uint8x8x4_t t10 = { \
        vld1_u8(src + addr[0] + srcstep), \
        vld1_u8(src + addr[1] + srcstep), \
        vld1_u8(src + addr[2] + srcstep), \
        vld1_u8(src + addr[3] + srcstep) \
    }; \
    uint8x8x4_t t11 = { \
        vld1_u8(src + addr[4] + srcstep), \
        vld1_u8(src + addr[5] + srcstep), \
        vld1_u8(src + addr[6] + srcstep), \
        vld1_u8(src + addr[7] + srcstep) \
    }; \
    uint32x2_t p00_, p01_, p10_, p11_;
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8_TRN(coords, suffix) \
    p00_ = vreinterpret_u32_u8(vtbl4_u8(t00, coords)); \
    p01_ = vreinterpret_u32_u8(vtbl4_u8(t01, coords)); \
    p10_ = vreinterpret_u32_u8(vtbl4_u8(t10, coords)); \
    p11_ = vreinterpret_u32_u8(vtbl4_u8(t11, coords)); \
    p00##suffix = vreinterpret_u8_u32(vtrn1_u32(p00_, p01_)); \
    p01##suffix = vreinterpret_u8_u32(vtrn2_u32(p00_, p01_)); \
    p10##suffix = vreinterpret_u8_u32(vtrn1_u32(p10_, p11_)); \
    p11##suffix = vreinterpret_u8_u32(vtrn2_u32(p10_, p11_));
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8_C1() \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8_LOAD() \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8_TRN(grays, g)
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8_C3() \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8_LOAD() \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8_TRN(reds, r) \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8_TRN(greens, g) \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8_TRN(blues, b)
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8_C4() \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8_LOAD() \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8_TRN(reds, r) \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8_TRN(greens, g) \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8_TRN(blues, b) \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8_TRN(alphas, a)
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8(CN) \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8_##CN()


// Shuffle (not all pixels within image)
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_STORE_CONSTANT_BORDER_8UC1() \
    v_store_low(dstptr + x, bval_v0);
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_STORE_CONSTANT_BORDER_8UC3() \
    v_store_low(dstptr + x*3,        bval_v0); \
    v_store_low(dstptr + x*3 + uf,   bval_v1); \
    v_store_low(dstptr + x*3 + uf*2, bval_v2);
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_STORE_CONSTANT_BORDER_8UC4() \
    v_store_low(dstptr + x*4,        bval_v0); \
    v_store_low(dstptr + x*4 + uf,   bval_v1); \
    v_store_low(dstptr + x*4 + uf*2, bval_v2); \
    v_store_low(dstptr + x*4 + uf*3, bval_v3);
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_STORE_CONSTANT_BORDER_16UC1() \
    v_store(dstptr + x, bval_v0);
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_STORE_CONSTANT_BORDER_16UC3() \
    v_store(dstptr + x*3,        bval_v0); \
    v_store(dstptr + x*3 + uf,   bval_v1); \
    v_store(dstptr + x*3 + uf*2, bval_v2);
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_STORE_CONSTANT_BORDER_16UC4() \
    v_store(dstptr + x*4,        bval_v0); \
    v_store(dstptr + x*4 + uf,   bval_v1); \
    v_store(dstptr + x*4 + uf*2, bval_v2); \
    v_store(dstptr + x*4 + uf*3, bval_v3);
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_STORE_CONSTANT_BORDER_32FC1() \
    v_store(dstptr + x,             bval_v0_l); \
    v_store(dstptr + x + vlanes_32, bval_v0_h);
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_STORE_CONSTANT_BORDER_32FC3() \
    v_store(dstptr + x*3,                    bval_v0_l); \
    v_store(dstptr + x*3 + vlanes_32,        bval_v0_h); \
    v_store(dstptr + x*3 + uf,               bval_v1_l); \
    v_store(dstptr + x*3 + uf + vlanes_32,   bval_v1_h); \
    v_store(dstptr + x*3 + uf*2,             bval_v2_l); \
    v_store(dstptr + x*3 + uf*2 + vlanes_32, bval_v2_h);
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_STORE_CONSTANT_BORDER_32FC4() \
    v_store(dstptr + x*4,                    bval_v0_l); \
    v_store(dstptr + x*4 + vlanes_32,        bval_v0_h); \
    v_store(dstptr + x*4 + uf,               bval_v1_l); \
    v_store(dstptr + x*4 + uf + vlanes_32,   bval_v1_h); \
    v_store(dstptr + x*4 + uf*2,             bval_v2_l); \
    v_store(dstptr + x*4 + uf*2 + vlanes_32, bval_v2_h); \
    v_store(dstptr + x*4 + uf*3,             bval_v3_l); \
    v_store(dstptr + x*4 + uf*3 + vlanes_32, bval_v3_h);

#define CV_WARP_LINEAR_VECTOR_FETCH_PIXEL_C1(dy, dx, pixbuf_ofs) \
    if ((((unsigned)(ix + dx) < (unsigned)srccols) & ((unsigned)(iy + dy) < (unsigned)srcrows)) != 0) { \
        size_t addr_i = addr[i] + dy*srcstep + dx; \
        pixbuf[i + pixbuf_ofs] = src[addr_i]; \
    } else if (border_type == BORDER_CONSTANT) { \
        pixbuf[i + pixbuf_ofs] = bval[0]; \
    } else if (border_type == BORDER_TRANSPARENT) { \
        pixbuf[i + pixbuf_ofs] = dstptr[x + i]; \
    } else { \
        int ix_ = borderInterpolate_fast(ix + dx, srccols, border_type_x); \
        int iy_ = borderInterpolate_fast(iy + dy, srcrows, border_type_y); \
        size_t addr_i = iy_*srcstep + ix_; \
        pixbuf[i + pixbuf_ofs] = src[addr_i]; \
    }
#define CV_WARP_LINEAR_VECTOR_FETCH_PIXEL_C3(dy, dx, pixbuf_ofs) \
    if ((((unsigned)(ix + dx) < (unsigned)srccols) & ((unsigned)(iy + dy) < (unsigned)srcrows)) != 0) { \
        size_t addr_i = addr[i] + dy*srcstep + dx*3; \
        pixbuf[i + pixbuf_ofs] = src[addr_i]; \
        pixbuf[i + pixbuf_ofs + uf*4] = src[addr_i+1]; \
        pixbuf[i + pixbuf_ofs + uf*8] = src[addr_i+2]; \
    } else if (border_type == BORDER_CONSTANT) { \
        pixbuf[i + pixbuf_ofs] = bval[0]; \
        pixbuf[i + pixbuf_ofs + uf*4] = bval[1]; \
        pixbuf[i + pixbuf_ofs + uf*8] = bval[2]; \
    } else if (border_type == BORDER_TRANSPARENT) { \
        pixbuf[i + pixbuf_ofs] = dstptr[(x + i)*3]; \
        pixbuf[i + pixbuf_ofs + uf*4] = dstptr[(x + i)*3 + 1]; \
        pixbuf[i + pixbuf_ofs + uf*8] = dstptr[(x + i)*3 + 2]; \
    } else { \
        int ix_ = borderInterpolate_fast(ix + dx, srccols, border_type_x); \
        int iy_ = borderInterpolate_fast(iy + dy, srcrows, border_type_y); \
        size_t addr_i = iy_*srcstep + ix_*3; \
        pixbuf[i + pixbuf_ofs] = src[addr_i]; \
        pixbuf[i + pixbuf_ofs + uf*4] = src[addr_i+1]; \
        pixbuf[i + pixbuf_ofs + uf*8] = src[addr_i+2]; \
    }
#define CV_WARP_LINEAR_VECTOR_FETCH_PIXEL_C4(dy, dx, pixbuf_ofs) \
    if ((((unsigned)(ix + dx) < (unsigned)srccols) & ((unsigned)(iy + dy) < (unsigned)srcrows)) != 0) { \
        size_t addr_i = addr[i] + dy*srcstep + dx*4; \
        pixbuf[i + pixbuf_ofs] = src[addr_i]; \
        pixbuf[i + pixbuf_ofs + uf*4] = src[addr_i+1]; \
        pixbuf[i + pixbuf_ofs + uf*8] = src[addr_i+2]; \
        pixbuf[i + pixbuf_ofs + uf*12] = src[addr_i+3]; \
    } else if (border_type == BORDER_CONSTANT) { \
        pixbuf[i + pixbuf_ofs] = bval[0]; \
        pixbuf[i + pixbuf_ofs + uf*4] = bval[1]; \
        pixbuf[i + pixbuf_ofs + uf*8] = bval[2]; \
        pixbuf[i + pixbuf_ofs + uf*12] = bval[3]; \
    } else if (border_type == BORDER_TRANSPARENT) { \
        pixbuf[i + pixbuf_ofs] = dstptr[(x + i)*4]; \
        pixbuf[i + pixbuf_ofs + uf*4] = dstptr[(x + i)*4 + 1]; \
        pixbuf[i + pixbuf_ofs + uf*8] = dstptr[(x + i)*4 + 2]; \
        pixbuf[i + pixbuf_ofs + uf*12] = dstptr[(x + i)*4 + 3]; \
    } else { \
        int ix_ = borderInterpolate_fast(ix + dx, srccols, border_type_x); \
        int iy_ = borderInterpolate_fast(iy + dy, srcrows, border_type_y); \
        size_t addr_i = iy_*srcstep + ix_*4; \
        pixbuf[i + pixbuf_ofs] = src[addr_i]; \
        pixbuf[i + pixbuf_ofs + uf*4] = src[addr_i+1]; \
        pixbuf[i + pixbuf_ofs + uf*8] = src[addr_i+2]; \
        pixbuf[i + pixbuf_ofs + uf*12] = src[addr_i+3]; \
    }

#define CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(CN, DEPTH) \
    if (border_type == BORDER_CONSTANT || border_type == BORDER_TRANSPARENT) { \
        mask_0 = v_lt(v_reinterpret_as_u32(v_add(src_ix0, one)), outer_scols); \
        mask_1 = v_lt(v_reinterpret_as_u32(v_add(src_ix1, one)), outer_scols); \
        mask_0 = v_and(mask_0, v_lt(v_reinterpret_as_u32(v_add(src_iy0, one)), outer_srows)); \
        mask_1 = v_and(mask_1, v_lt(v_reinterpret_as_u32(v_add(src_iy1, one)), outer_srows)); \
        v_uint16 outer_mask = v_pack(mask_0, mask_1); \
        if (v_reduce_max(outer_mask) == 0) { \
            if (border_type == BORDER_CONSTANT) { \
                CV_WARP_LINEAR_VECTOR_SHUFFLE_STORE_CONSTANT_BORDER_##DEPTH##CN() \
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
        CV_WARP_LINEAR_VECTOR_FETCH_PIXEL_##CN(0, 0, 0); \
        CV_WARP_LINEAR_VECTOR_FETCH_PIXEL_##CN(0, 1, uf); \
        CV_WARP_LINEAR_VECTOR_FETCH_PIXEL_##CN(1, 0, uf*2); \
        CV_WARP_LINEAR_VECTOR_FETCH_PIXEL_##CN(1, 1, uf*3); \
    }

// Shuffle (not all pixels within image) (ARM NEON)
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8_LOAD(suffix, offset)\
    p00##suffix = vld1_u8(pixbuf + offset);      \
    p01##suffix = vld1_u8(pixbuf + offset + 8);  \
    p10##suffix = vld1_u8(pixbuf + offset + 16); \
    p11##suffix = vld1_u8(pixbuf + offset + 24);
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8_C1() \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8_LOAD(g, 0)
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8_C3() \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8_LOAD(r, 0) \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8_LOAD(g, 32) \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8_LOAD(b, 64)
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8_C4() \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8_LOAD(r, 0) \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8_LOAD(g, 32) \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8_LOAD(b, 64) \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8_LOAD(a, 96)
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8(CN) \
    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8_##CN()

// Load pixels for linear interpolation (uint8_t -> int16_t)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8S16(cn, i) \
    v_int16  f00##cn = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf * i)), \
             f01##cn = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf * (i+1))), \
             f10##cn = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf * (i+2))), \
             f11##cn = v_reinterpret_as_s16(vx_load_expand(pixbuf + uf * (i+3)));
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16_C1() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8S16(g, 0)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16_C3() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8S16(r, 0) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8S16(g, 4) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8S16(b, 8)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16_C4() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8S16(r, 0) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8S16(g, 4) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8S16(b, 8) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8S16(a, 12)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16(CN) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16_##CN();

// Load pixels for linear interpolation (uint8_t -> int16_t) (ARM NEON)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8S16_NEON(suffix) \
    v_int16 f00##suffix = v_reinterpret_as_s16(v_uint16(vmovl_u8(p00##suffix))), \
            f01##suffix = v_reinterpret_as_s16(v_uint16(vmovl_u8(p01##suffix))), \
            f10##suffix = v_reinterpret_as_s16(v_uint16(vmovl_u8(p10##suffix))), \
            f11##suffix = v_reinterpret_as_s16(v_uint16(vmovl_u8(p11##suffix)));
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16_NEON_C1() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8S16_NEON(g)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16_NEON_C3() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8S16_NEON(r) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8S16_NEON(g) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8S16_NEON(b)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16_NEON_C4() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8S16_NEON(r) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8S16_NEON(g) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8S16_NEON(b) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8S16_NEON(a)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16_NEON(CN) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16_NEON_##CN();

// Load pixels for linear interpolation (uint16_t -> uint16_t)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U16(cn, i) \
    v_uint16 f00##cn = vx_load(pixbuf + uf * i), \
             f01##cn = vx_load(pixbuf + uf * (i+1)), \
             f10##cn = vx_load(pixbuf + uf * (i+2)), \
             f11##cn = vx_load(pixbuf + uf * (i+3));
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U16_C1() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U16(g, 0)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U16_C3() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U16(r, 0) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U16(g, 4) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U16(b, 8)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U16_C4() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U16(r, 0) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U16(g, 4) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U16(b, 8) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U16(a, 12)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U16(CN) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_U16_##CN();

// Load pixels for linear interpolation (int16_t -> float)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_S16F32(cn) \
    v_float32 f00##cn##l = v_cvt_f32(v_expand_low(f00##cn)), f00##cn##h = v_cvt_f32(v_expand_high(f00##cn)), \
              f01##cn##l = v_cvt_f32(v_expand_low(f01##cn)), f01##cn##h = v_cvt_f32(v_expand_high(f01##cn)), \
              f10##cn##l = v_cvt_f32(v_expand_low(f10##cn)), f10##cn##h = v_cvt_f32(v_expand_high(f10##cn)), \
              f11##cn##l = v_cvt_f32(v_expand_low(f11##cn)), f11##cn##h = v_cvt_f32(v_expand_high(f11##cn));
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_S16F32_C1() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_S16F32(g)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_S16F32_C3() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_S16F32(r) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_S16F32(g) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_S16F32(b)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_S16F32_C4() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_S16F32(r) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_S16F32(g) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_S16F32(b) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_S16F32(a)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_S16F32(CN) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_S16F32_##CN()

// Load pixels for linear interpolation (uint16_t -> float)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U16F32(cn) \
    v_float32 f00##cn##l = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f00##cn))), f00##cn##h = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f00##cn))), \
              f01##cn##l = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f01##cn))), f01##cn##h = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f01##cn))), \
              f10##cn##l = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f10##cn))), f10##cn##h = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f10##cn))), \
              f11##cn##l = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f11##cn))), f11##cn##h = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f11##cn)));
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U16F32_C1() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U16F32(g)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U16F32_C3() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U16F32(r) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U16F32(g) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U16F32(b)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U16F32_C4() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U16F32(r) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U16F32(g) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U16F32(b) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U16F32(a)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U16F32(CN) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_U16F32_##CN()

// Load pixels for linear interpolation (float -> float)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_F32(cn, i) \
    v_float32 f00##cn##l = vx_load(pixbuf + uf * i),      f00##cn##h = vx_load(pixbuf + uf * i     + vlanes_32), \
              f01##cn##l = vx_load(pixbuf + uf * (i+1)),  f01##cn##h = vx_load(pixbuf + uf * (i+1) + vlanes_32), \
              f10##cn##l = vx_load(pixbuf + uf * (i+2)),  f10##cn##h = vx_load(pixbuf + uf * (i+2) + vlanes_32), \
              f11##cn##l = vx_load(pixbuf + uf * (i+3)),  f11##cn##h = vx_load(pixbuf + uf * (i+3) + vlanes_32);
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_F32_C1() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_F32(g, 0)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_F32_C3() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_F32(r, 0) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_F32(g, 4) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_F32(b, 8)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_F32_C4() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_F32(r, 0) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_F32(g, 4) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_F32(b, 8) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_F32(a, 12)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_F32(CN) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_F32_##CN()


// Linear interpolation calculation
#define CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32(cn) \
    f00##cn##l = v_fma(alphal, v_sub(f01##cn##l, f00##cn##l), f00##cn##l); f00##cn##h = v_fma(alphah, v_sub(f01##cn##h, f00##cn##h), f00##cn##h); \
    f10##cn##l = v_fma(alphal, v_sub(f11##cn##l, f10##cn##l), f10##cn##l); f10##cn##h = v_fma(alphah, v_sub(f11##cn##h, f10##cn##h), f10##cn##h);
#define CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32_C1() \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32(g)
#define CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32_C3() \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32(r) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32(g) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32(b)
#define CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32_C4() \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32(r) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32(g) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32(b) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32(a)

#define CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32(cn) \
    f00##cn##l = v_fma(betal,  v_sub(f10##cn##l, f00##cn##l), f00##cn##l); f00##cn##h = v_fma(betah,  v_sub(f10##cn##h, f00##cn##h), f00##cn##h);
#define CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32_C1() \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32(g)
#define CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32_C3() \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32(r) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32(g) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32(b)
#define CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32_C4() \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32(r) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32(g) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32(b) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32(a)

#define CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(CN) \
    v_float32 alphal = src_x0, alphah = src_x1, \
              betal = src_y0, betah = src_y1; \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32_##CN() \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32_##CN()


// Store
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8_C1() \
    v_uint16 f00_u16 = v_pack_u(v_round(f00gl), v_round(f00gh)); \
    v_uint8 f00_u8 = v_pack(f00_u16, vx_setall_u16(0)); \
    v_store_low(dstptr + x, f00_u8);
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8_C3() \
    v_uint16 f00r_u16 = v_pack_u(v_round(f00rl), v_round(f00rh)), \
             f00g_u16 = v_pack_u(v_round(f00gl), v_round(f00gh)), \
             f00b_u16 = v_pack_u(v_round(f00bl), v_round(f00bh)); \
    uint16_t tbuf[max_vlanes_16*3]; \
    v_store_interleave(tbuf, f00r_u16, f00g_u16, f00b_u16); \
    v_pack_store(dstptr + x*3, vx_load(tbuf)); \
    v_pack_store(dstptr + x*3 + vlanes_16, vx_load(tbuf + vlanes_16)); \
    v_pack_store(dstptr + x*3 + vlanes_16*2, vx_load(tbuf + vlanes_16*2));
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8_C4() \
    v_uint16 f00r_u16 = v_pack_u(v_round(f00rl), v_round(f00rh)), \
             f00g_u16 = v_pack_u(v_round(f00gl), v_round(f00gh)), \
             f00b_u16 = v_pack_u(v_round(f00bl), v_round(f00bh)), \
             f00a_u16 = v_pack_u(v_round(f00al), v_round(f00ah)); \
    uint16_t tbuf[max_vlanes_16*4]; \
    v_store_interleave(tbuf, f00r_u16, f00g_u16, f00b_u16, f00a_u16); \
    v_pack_store(dstptr + x*4, vx_load(tbuf)); \
    v_pack_store(dstptr + x*4 + vlanes_16, vx_load(tbuf + vlanes_16)); \
    v_pack_store(dstptr + x*4 + vlanes_16*2, vx_load(tbuf + vlanes_16*2)); \
    v_pack_store(dstptr + x*4 + vlanes_16*3, vx_load(tbuf + vlanes_16*3));
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(CN) \
    CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8_##CN()

#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16_C1() \
    v_uint16 f00_u16 = v_pack_u(v_round(f00gl), v_round(f00gh)); \
    v_store(dstptr + x, f00_u16);
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16_C3() \
    v_uint16 f00r_u16 = v_pack_u(v_round(f00rl), v_round(f00rh)), \
             f00g_u16 = v_pack_u(v_round(f00gl), v_round(f00gh)), \
             f00b_u16 = v_pack_u(v_round(f00bl), v_round(f00bh)); \
    v_store_interleave(dstptr + x*3, f00r_u16, f00g_u16, f00b_u16);
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16_C4() \
    v_uint16 f00r_u16 = v_pack_u(v_round(f00rl), v_round(f00rh)), \
             f00g_u16 = v_pack_u(v_round(f00gl), v_round(f00gh)), \
             f00b_u16 = v_pack_u(v_round(f00bl), v_round(f00bh)), \
             f00a_u16 = v_pack_u(v_round(f00al), v_round(f00ah)); \
    v_store_interleave(dstptr + x*4, f00r_u16, f00g_u16, f00b_u16, f00a_u16);
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(CN) \
    CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16_##CN()

#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32_C1() \
    vx_store(dstptr + x, f00gl); \
    vx_store(dstptr + x + vlanes_32, f00gh);
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32_C3() \
    v_store_interleave(dstptr + x*3, f00rl, f00gl, f00bl); \
    v_store_interleave(dstptr + x*3 + vlanes_32*3, f00rh, f00gh, f00bh);
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32_C4() \
    v_store_interleave(dstptr + x*4, f00rl, f00gl, f00bl, f00al); \
    v_store_interleave(dstptr + x*4 + vlanes_32*4, f00rh, f00gh, f00bh, f00ah);
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(CN) \
    CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32_##CN()
