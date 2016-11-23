// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

__constant short4 vec_offset = (short4)(0, 1, 2, 3);

#define GET_VAL(x, y) ((x) < 0 || (x) >= src_cols || (y) < 0 || (y) >= src_rows) ? scalar : src[src_offset + y * src_step + x]

__kernel void warpAffine_nearest_8u(__global const uchar * src, int src_step, int src_offset, int src_rows, int src_cols,
                                    __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                    __constant float * M, ST scalar_)
{
    int x = get_global_id(0) * 4;
    int y = get_global_id(1);
    uchar scalar = convert_uchar_sat_rte(scalar_);

    if (x >= dst_cols || y >= dst_rows) return;

    /* { M0, M1, M2 }
     * { M3, M4, M5 }
     */

    short4 new_x, new_y;
    new_x = convert_short4_sat_rte(M[0] * convert_float4(vec_offset + (short4)(x)) +
                                   M[1] * convert_float4((short4)y) + M[2]);

    new_y = convert_short4_sat_rte(M[3] * convert_float4(vec_offset + (short4)(x)) +
                                   M[4] * convert_float4((short4)y) + M[5]);

    uchar4 pix = (uchar4)scalar;

    pix.s0 = GET_VAL(new_x.s0, new_y.s0);
    pix.s1 = GET_VAL(new_x.s1, new_y.s1);
    pix.s2 = GET_VAL(new_x.s2, new_y.s2);
    pix.s3 = GET_VAL(new_x.s3, new_y.s3);

    int dst_index = x + y * dst_step + dst_offset;

    vstore4(pix, 0,  dst + dst_index);
}

uchar4 read_pixels(__global const uchar * src, short tx, short ty,
                   int src_offset, int src_step, int src_cols, int
                   src_rows, uchar scalar)
{
    uchar2 pt, pb;
    short bx, by;

    bx = tx + 1;
    by = ty + 1;

    if (tx >= 0 && (tx + 1) < src_cols && ty >= 0 && ty < src_rows)
    {
        pt = vload2(0, src + src_offset + ty * src_step + tx);
    }
    else
    {
        pt.s0 = GET_VAL(tx, ty);
        pt.s1 = GET_VAL(bx, ty);
    }

    if (tx >= 0 && (tx + 1) < src_cols && by >= 0 && by < src_rows)
    {
        pb = vload2(0, src + src_offset + by * src_step + tx);
    }
    else
    {
        pb.s0 = GET_VAL(tx, by);
        pb.s1 = GET_VAL(bx, by);
    }

    return (uchar4)(pt, pb);
}

__kernel void warpAffine_linear_8u(__global const uchar * src, int src_step, int src_offset, int src_rows, int src_cols,
                                   __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                   __constant float * M, ST scalar_)
{
    int x = get_global_id(0) * 4;
    int y = get_global_id(1);
    uchar scalar = convert_uchar_sat_rte(scalar_);

    if (x >= dst_cols || y >= dst_rows) return;

    /* { M0, M1, M2 }
     * { M3, M4, M5 }
     */

    float4 nx, ny;
    nx = M[0] * convert_float4((vec_offset + (short4)x)) + M[1] * convert_float4((short4)y) + M[2];
    ny = M[3] * convert_float4((vec_offset + (short4)x)) + M[4] * convert_float4((short4)y) + M[5];

    float4 s, t;
    s = round((nx - floor(nx)) * 32.0f) / 32.0f;
    t = round((ny - floor(ny)) * 32.0f) / 32.0f;

    short4 tx, ty;
    tx = convert_short4_sat_rtn(nx);
    ty = convert_short4_sat_rtn(ny);

    uchar4 pix[4];
    pix[0] = read_pixels(src, tx.s0, ty.s0, src_offset, src_step, src_cols, src_rows, scalar);
    pix[1] = read_pixels(src, tx.s1, ty.s1, src_offset, src_step, src_cols, src_rows, scalar);
    pix[2] = read_pixels(src, tx.s2, ty.s2, src_offset, src_step, src_cols, src_rows, scalar);
    pix[3] = read_pixels(src, tx.s3, ty.s3, src_offset, src_step, src_cols, src_rows, scalar);

    float4 tl, tr, bl, br;
    tl = convert_float4((uchar4)(pix[0].s0, pix[1].s0, pix[2].s0, pix[3].s0));
    tr = convert_float4((uchar4)(pix[0].s1, pix[1].s1, pix[2].s1, pix[3].s1));
    bl = convert_float4((uchar4)(pix[0].s2, pix[1].s2, pix[2].s2, pix[3].s2));
    br = convert_float4((uchar4)(pix[0].s3, pix[1].s3, pix[2].s3, pix[3].s3));

    float4 pixel;
    pixel = tl * (1 - s) * (1 - t) + tr * s * (1 - t) + bl * (1 - s) * t + br * s * t;

    int dst_index = x + y * dst_step + dst_offset;
    vstore4(convert_uchar4_sat_rte(pixel), 0, dst + dst_index);
}

__kernel void warpPerspective_nearest_8u(__global const uchar * src, int src_step, int src_offset, int src_rows, int src_cols,
                                         __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                         __constant float * M, ST scalar_)
{
    int x = get_global_id(0) * 4;
    int y = get_global_id(1);
    uchar scalar = convert_uchar_sat_rte(scalar_);

    if (x >= dst_cols || y >= dst_rows) return;

    /* { M0, M1, M2 }
     * { M3, M4, M5 }
     * { M6, M7, M8 }
     */

    float4 nx, ny, nz;
    nx = M[0] * convert_float4(vec_offset + (short4)(x)) +
         M[1] * convert_float4((short4)y) + M[2];

    ny = M[3] * convert_float4(vec_offset + (short4)(x)) +
         M[4] * convert_float4((short4)y) + M[5];

    nz = M[6] * convert_float4(vec_offset + (short4)(x)) +
         M[7] * convert_float4((short4)y) + M[8];

    short4 new_x, new_y;
    float4 fz = select((float4)(0.0f), (float4)(1.0f / nz), nz != 0.0f);
    new_x = convert_short4_sat_rte(nx * fz);
    new_y = convert_short4_sat_rte(ny * fz);

    uchar4 pix = (uchar4)scalar;

    pix.s0 = GET_VAL(new_x.s0, new_y.s0);
    pix.s1 = GET_VAL(new_x.s1, new_y.s1);
    pix.s2 = GET_VAL(new_x.s2, new_y.s2);
    pix.s3 = GET_VAL(new_x.s3, new_y.s3);

    int dst_index = x + y * dst_step + dst_offset;

    vstore4(pix, 0,  dst + dst_index);
}

__kernel void warpPerspective_linear_8u(__global const uchar * src, int src_step, int src_offset, int src_rows, int src_cols,
                                        __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                        __constant float * M, ST scalar_)
{
    int x = get_global_id(0) * 4;
    int y = get_global_id(1);
    uchar scalar = convert_uchar_sat_rte(scalar_);

    if (x >= dst_cols || y >= dst_rows) return;

    /* { M0, M1, M2 }
     * { M3, M4, M5 }
     * { M6, M7, M8 }
     */

    float4 nx, ny, nz;
    nx = M[0] * convert_float4(vec_offset + (short4)(x)) + M[1] * convert_float4((short4)y) + M[2];

    ny = M[3] * convert_float4(vec_offset + (short4)(x)) + M[4] * convert_float4((short4)y) + M[5];

    nz = M[6] * convert_float4(vec_offset + (short4)(x)) + M[7] * convert_float4((short4)y) + M[8];

    float4 fz = select((float4)(0.0f), (float4)(1.0f / nz), nz != 0.0f);

    nx = nx * fz;
    ny = ny * fz;

    float4 s, t;
    s = round((nx - floor(nx)) * 32.0f) / (float4)32.0f;
    t = round((ny - floor(ny)) * 32.0f) / (float4)32.0f;

    short4 tx, ty;
    tx = convert_short4_sat_rtn(nx);
    ty = convert_short4_sat_rtn(ny);

    uchar4 pix[4];
    pix[0] = read_pixels(src, tx.s0, ty.s0, src_offset, src_step, src_cols, src_rows, scalar);
    pix[1] = read_pixels(src, tx.s1, ty.s1, src_offset, src_step, src_cols, src_rows, scalar);
    pix[2] = read_pixels(src, tx.s2, ty.s2, src_offset, src_step, src_cols, src_rows, scalar);
    pix[3] = read_pixels(src, tx.s3, ty.s3, src_offset, src_step, src_cols, src_rows, scalar);

    float4 tl, tr, bl, br;
    tl = convert_float4((uchar4)(pix[0].s0, pix[1].s0, pix[2].s0, pix[3].s0));
    tr = convert_float4((uchar4)(pix[0].s1, pix[1].s1, pix[2].s1, pix[3].s1));
    bl = convert_float4((uchar4)(pix[0].s2, pix[1].s2, pix[2].s2, pix[3].s2));
    br = convert_float4((uchar4)(pix[0].s3, pix[1].s3, pix[2].s3, pix[3].s3));

    float4 pixel;
    pixel = tl * (1 - s) * (1 - t) + tr * s * (1 - t) + bl * (1 - s) * t + br * s * t;

    int dst_index = x + y * dst_step + dst_offset;
    vstore4(convert_uchar4_sat_rte(pixel), 0,  dst + dst_index);
}
