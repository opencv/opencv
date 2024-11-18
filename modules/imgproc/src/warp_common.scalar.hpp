// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Shuffle
#define CV_WARP_LINEAR_SCALAR_SHUFFLE_DEF(cn, dtype_reg) \
    dtype_reg p00##cn, p01##cn, p10##cn, p11##cn;
#define CV_WARP_LINEAR_SCALAR_SHUFFLE_DEF_C1(dtype_reg, dtype_ptr) \
    CV_WARP_LINEAR_SCALAR_SHUFFLE_DEF(g, dtype_reg) \
    const dtype_ptr *srcptr = src + srcstep * iy + ix;
#define CV_WARP_LINEAR_SCALAR_SHUFFLE_DEF_C3(dtype_reg, dtype_ptr) \
    CV_WARP_LINEAR_SCALAR_SHUFFLE_DEF(r, dtype_reg) \
    CV_WARP_LINEAR_SCALAR_SHUFFLE_DEF(g, dtype_reg) \
    CV_WARP_LINEAR_SCALAR_SHUFFLE_DEF(b, dtype_reg) \
    const dtype_ptr *srcptr = src + srcstep * iy + ix*3;
#define CV_WARP_LINEAR_SCALAR_SHUFFLE_DEF_C4(dtype_reg, dtype_ptr) \
    CV_WARP_LINEAR_SCALAR_SHUFFLE_DEF(r, dtype_reg) \
    CV_WARP_LINEAR_SCALAR_SHUFFLE_DEF(g, dtype_reg) \
    CV_WARP_LINEAR_SCALAR_SHUFFLE_DEF(b, dtype_reg) \
    CV_WARP_LINEAR_SCALAR_SHUFFLE_DEF(a, dtype_reg) \
    const dtype_ptr *srcptr = src + srcstep * iy + ix*4;
#define CV_WARP_LINEAR_SCALAR_SHUFFLE_DEF_8U(CN) \
    CV_WARP_LINEAR_SCALAR_SHUFFLE_DEF_##CN(int, uint8_t)
#define CV_WARP_LINEAR_SCALAR_SHUFFLE_DEF_16U(CN) \
    CV_WARP_LINEAR_SCALAR_SHUFFLE_DEF_##CN(int, uint16_t)
#define CV_WARP_LINEAR_SCALAR_SHUFFLE_DEF_32F(CN) \
    CV_WARP_LINEAR_SCALAR_SHUFFLE_DEF_##CN(float, float)

#define CV_WARP_LINEAR_SCALAR_SHUFFLE_LOAD(CN, cn, i) \
    p00##CN = srcptr[i]; p01##CN = srcptr[i + cn]; \
    p10##CN = srcptr[srcstep + i]; p11##CN = srcptr[srcstep + cn + i];
#define CV_WARP_LINEAR_SCALAR_SHUFFLE_LOAD_C1() \
    CV_WARP_LINEAR_SCALAR_SHUFFLE_LOAD(g, 1, 0)
#define CV_WARP_LINEAR_SCALAR_SHUFFLE_LOAD_C3() \
    CV_WARP_LINEAR_SCALAR_SHUFFLE_LOAD(r, 3, 0) \
    CV_WARP_LINEAR_SCALAR_SHUFFLE_LOAD(g, 3, 1) \
    CV_WARP_LINEAR_SCALAR_SHUFFLE_LOAD(b, 3, 2)
#define CV_WARP_LINEAR_SCALAR_SHUFFLE_LOAD_C4() \
    CV_WARP_LINEAR_SCALAR_SHUFFLE_LOAD(r, 4, 0) \
    CV_WARP_LINEAR_SCALAR_SHUFFLE_LOAD(g, 4, 1) \
    CV_WARP_LINEAR_SCALAR_SHUFFLE_LOAD(b, 4, 2) \
    CV_WARP_LINEAR_SCALAR_SHUFFLE_LOAD(a, 4, 3)

#define CV_WARP_LINEAR_SCALAR_SHUFFLE_STORE_CONSTANT_BORDER_C1() \
    dstptr[x] = bval[0];
#define CV_WARP_LINEAR_SCALAR_SHUFFLE_STORE_CONSTANT_BORDER_C3() \
    dstptr[x*3] = bval[0]; \
    dstptr[x*3+1] = bval[1]; \
    dstptr[x*3+2] = bval[2];
#define CV_WARP_LINEAR_SCALAR_SHUFFLE_STORE_CONSTANT_BORDER_C4() \
    dstptr[x*4] = bval[0]; \
    dstptr[x*4+1] = bval[1]; \
    dstptr[x*4+2] = bval[2]; \
    dstptr[x*4+3] = bval[3];

#define CV_WARP_LINEAR_SCALAR_FETCH_PIXEL_C1(dy, dx, pxy) \
    if ((((unsigned)(ix + dx) < (unsigned)srccols) & ((unsigned)(iy + dy) < (unsigned)srcrows)) != 0) { \
        size_t ofs = dy*srcstep + dx; \
        pxy##g = srcptr[ofs]; \
    } else if (border_type == BORDER_CONSTANT) { \
        pxy##g = bval[0]; \
    } else if (border_type == BORDER_TRANSPARENT) { \
        pxy##g = dstptr[x]; \
    } else { \
        int ix_ = borderInterpolate_fast(ix + dx, srccols, border_type_x); \
        int iy_ = borderInterpolate_fast(iy + dy, srcrows, border_type_y); \
        size_t glob_ofs = iy_*srcstep + ix_; \
        pxy##g = src[glob_ofs]; \
    }
#define CV_WARP_LINEAR_SCALAR_FETCH_PIXEL_C3(dy, dx, pxy) \
    if ((((unsigned)(ix + dx) < (unsigned)srccols) & ((unsigned)(iy + dy) < (unsigned)srcrows)) != 0) { \
        size_t ofs = dy*srcstep + dx*3; \
        pxy##r = srcptr[ofs]; \
        pxy##g = srcptr[ofs+1]; \
        pxy##b = srcptr[ofs+2]; \
    } else if (border_type == BORDER_CONSTANT) { \
        pxy##r = bval[0]; \
        pxy##g = bval[1]; \
        pxy##b = bval[2]; \
    } else if (border_type == BORDER_TRANSPARENT) { \
        pxy##r = dstptr[x*3]; \
        pxy##g = dstptr[x*3+1]; \
        pxy##b = dstptr[x*3+2]; \
    } else { \
        int ix_ = borderInterpolate_fast(ix + dx, srccols, border_type_x); \
        int iy_ = borderInterpolate_fast(iy + dy, srcrows, border_type_y); \
        size_t glob_ofs = iy_*srcstep + ix_*3; \
        pxy##r = src[glob_ofs]; \
        pxy##g = src[glob_ofs+1]; \
        pxy##b = src[glob_ofs+2]; \
    }
#define CV_WARP_LINEAR_SCALAR_FETCH_PIXEL_C4(dy, dx, pxy) \
    if ((((unsigned)(ix + dx) < (unsigned)srccols) & ((unsigned)(iy + dy) < (unsigned)srcrows)) != 0) { \
        size_t ofs = dy*srcstep + dx*4; \
        pxy##r = srcptr[ofs]; \
        pxy##g = srcptr[ofs+1]; \
        pxy##b = srcptr[ofs+2]; \
        pxy##a = srcptr[ofs+3]; \
    } else if (border_type == BORDER_CONSTANT) { \
        pxy##r = bval[0]; \
        pxy##g = bval[1]; \
        pxy##b = bval[2]; \
        pxy##a = bval[3]; \
    } else if (border_type == BORDER_TRANSPARENT) { \
        pxy##r = dstptr[x*4]; \
        pxy##g = dstptr[x*4+1]; \
        pxy##b = dstptr[x*4+2]; \
        pxy##a = dstptr[x*4+3]; \
    } else { \
        int ix_ = borderInterpolate_fast(ix + dx, srccols, border_type_x); \
        int iy_ = borderInterpolate_fast(iy + dy, srcrows, border_type_y); \
        size_t glob_ofs = iy_*srcstep + ix_*4; \
        pxy##r = src[glob_ofs]; \
        pxy##g = src[glob_ofs+1]; \
        pxy##b = src[glob_ofs+2]; \
        pxy##a = src[glob_ofs+3]; \
    }

#define CV_WARP_LINEAR_SCALAR_SHUFFLE(CN, DEPTH) \
    int ix = cvFloor(sx), iy = cvFloor(sy); \
    sx -= ix; sy -= iy; \
    CV_WARP_LINEAR_SCALAR_SHUFFLE_DEF_##DEPTH(CN); \
    if ((((unsigned)ix < (unsigned)(srccols-1)) & \
        ((unsigned)iy < (unsigned)(srcrows-1))) != 0) { \
        CV_WARP_LINEAR_SCALAR_SHUFFLE_LOAD_##CN() \
    } else { \
        if ((border_type == BORDER_CONSTANT || border_type == BORDER_TRANSPARENT) && \
            (((unsigned)(ix+1) >= (unsigned)(srccols+1))| \
                ((unsigned)(iy+1) >= (unsigned)(srcrows+1))) != 0) { \
            if (border_type == BORDER_CONSTANT) { \
                CV_WARP_LINEAR_SCALAR_SHUFFLE_STORE_CONSTANT_BORDER_##CN() \
            } \
            continue; \
        } \
        CV_WARP_LINEAR_SCALAR_FETCH_PIXEL_##CN(0, 0, p00); \
        CV_WARP_LINEAR_SCALAR_FETCH_PIXEL_##CN(0, 1, p01); \
        CV_WARP_LINEAR_SCALAR_FETCH_PIXEL_##CN(1, 0, p10); \
        CV_WARP_LINEAR_SCALAR_FETCH_PIXEL_##CN(1, 1, p11); \
    }


// Linear interpolation calculation
#define CV_WARP_LINEAR_SCALAR_INTER_CALC_ALPHA_F32(cn) \
    float v0##cn = p00##cn + sx*(p01##cn - p00##cn); \
    float v1##cn = p10##cn + sx*(p11##cn - p10##cn);
#define CV_WARP_LINEAR_SCALAR_INTER_CALC_ALPHA_F32_C1() \
    CV_WARP_LINEAR_SCALAR_INTER_CALC_ALPHA_F32(g)
#define CV_WARP_LINEAR_SCALAR_INTER_CALC_ALPHA_F32_C3() \
    CV_WARP_LINEAR_SCALAR_INTER_CALC_ALPHA_F32(r) \
    CV_WARP_LINEAR_SCALAR_INTER_CALC_ALPHA_F32(g) \
    CV_WARP_LINEAR_SCALAR_INTER_CALC_ALPHA_F32(b)
#define CV_WARP_LINEAR_SCALAR_INTER_CALC_ALPHA_F32_C4() \
    CV_WARP_LINEAR_SCALAR_INTER_CALC_ALPHA_F32(r) \
    CV_WARP_LINEAR_SCALAR_INTER_CALC_ALPHA_F32(g) \
    CV_WARP_LINEAR_SCALAR_INTER_CALC_ALPHA_F32(b) \
    CV_WARP_LINEAR_SCALAR_INTER_CALC_ALPHA_F32(a)

#define CV_WARP_LINEAR_SCALAR_INTER_CALC_BETA_F32(cn) \
    v0##cn += sy*(v1##cn - v0##cn);
#define CV_WARP_LINEAR_SCALAR_INTER_CALC_BETA_F32_C1() \
    CV_WARP_LINEAR_SCALAR_INTER_CALC_BETA_F32(g)
#define CV_WARP_LINEAR_SCALAR_INTER_CALC_BETA_F32_C3() \
    CV_WARP_LINEAR_SCALAR_INTER_CALC_BETA_F32(r) \
    CV_WARP_LINEAR_SCALAR_INTER_CALC_BETA_F32(g) \
    CV_WARP_LINEAR_SCALAR_INTER_CALC_BETA_F32(b)
#define CV_WARP_LINEAR_SCALAR_INTER_CALC_BETA_F32_C4() \
    CV_WARP_LINEAR_SCALAR_INTER_CALC_BETA_F32(r) \
    CV_WARP_LINEAR_SCALAR_INTER_CALC_BETA_F32(g) \
    CV_WARP_LINEAR_SCALAR_INTER_CALC_BETA_F32(b) \
    CV_WARP_LINEAR_SCALAR_INTER_CALC_BETA_F32(a)

#define CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(CN) \
    CV_WARP_LINEAR_SCALAR_INTER_CALC_ALPHA_F32_##CN() \
    CV_WARP_LINEAR_SCALAR_INTER_CALC_BETA_F32_##CN()


// Store
#define CV_WARP_LINEAR_SCALAR_STORE_C1(dtype) \
    dstptr[x] = saturate_cast<dtype>(v0g);
#define CV_WARP_LINEAR_SCALAR_STORE_C3(dtype) \
    dstptr[x*3] = saturate_cast<dtype>(v0r); \
    dstptr[x*3+1] = saturate_cast<dtype>(v0g); \
    dstptr[x*3+2] = saturate_cast<dtype>(v0b);
#define CV_WARP_LINEAR_SCALAR_STORE_C4(dtype) \
    dstptr[x*4] = saturate_cast<dtype>(v0r); \
    dstptr[x*4+1] = saturate_cast<dtype>(v0g); \
    dstptr[x*4+2] = saturate_cast<dtype>(v0b); \
    dstptr[x*4+3] = saturate_cast<dtype>(v0a);
#define CV_WARP_LINEAR_SCALAR_STORE_8U(CN) \
    CV_WARP_LINEAR_SCALAR_STORE_##CN(uint8_t)
#define CV_WARP_LINEAR_SCALAR_STORE_16U(CN) \
    CV_WARP_LINEAR_SCALAR_STORE_##CN(uint16_t)
#define CV_WARP_LINEAR_SCALAR_STORE_32F(CN) \
    CV_WARP_LINEAR_SCALAR_STORE_##CN(float)

#define CV_WARP_LINEAR_SCALAR_STORE(CN, DEPTH) \
    CV_WARP_LINEAR_SCALAR_STORE_##DEPTH(CN)
