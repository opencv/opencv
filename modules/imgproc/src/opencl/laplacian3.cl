// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#define DIG(a) a,
__constant float kx[] = { KERNEL_MATRIX };

#define OP(delta, x) (convert_float16(arr[delta + x]) * kx[x])

__kernel void laplacian3_8UC1_cols16_rows2(__global const uint* src, int src_step,
                                           __global uint* dst, int dst_step,
                                           int rows, int cols, float delta)
{
    int block_x = get_global_id(0);
    int y = get_global_id(1) * 2;
    int ssx, dsx;

    if ((block_x * 16) >= cols || y >= rows) return;

    uint4 line[4];
    uint4 line_out[2];
    uchar a; uchar16 b; uchar c;
    uchar d; uchar16 e; uchar f;
    uchar g; uchar16 h; uchar i;
    uchar j; uchar16 k; uchar l;

    ssx = dsx = 1;
    int src_index = block_x * 4 * ssx + (y - 1) * (src_step / 4);
    line[1] = vload4(0, src + src_index + (src_step / 4));
    line[2] = vload4(0, src + src_index + 2 * (src_step / 4));

#ifdef BORDER_CONSTANT
    line[0] = (y == 0) ? (uint4)0 : vload4(0, src + src_index);
    line[3] = (y == (rows - 2)) ? (uint4)0 : vload4(0, src + src_index + 3 * (src_step / 4));
#elif defined BORDER_REFLECT_101
    line[0] = (y == 0) ? line[2] : vload4(0, src + src_index);
    line[3] = (y == (rows - 2)) ? line[1] : vload4(0, src + src_index + 3 * (src_step / 4));
#elif defined (BORDER_REPLICATE) || defined(BORDER_REFLECT)
    line[0] = (y == 0) ? line[1] : vload4(0, src + src_index);
    line[3] = (y == (rows - 2)) ? line[2] : vload4(0, src + src_index + 3 * (src_step / 4));
#endif

    __global uchar *src_p = (__global uchar *)src;

    src_index = block_x * 16 * ssx + (y - 1) * src_step;
    bool line_end = ((block_x + 1) * 16 == cols);

    b = as_uchar16(line[0]);
    e = as_uchar16(line[1]);
    h = as_uchar16(line[2]);
    k = as_uchar16(line[3]);

#ifdef BORDER_CONSTANT
    a = (block_x == 0 || y == 0) ? 0 : src_p[src_index - 1];
    c = (line_end || y == 0) ? 0 : src_p[src_index + 16];

    d = (block_x == 0) ? 0 : src_p[src_index + src_step - 1];
    f = line_end ? 0 : src_p[src_index + src_step + 16];

    g = (block_x == 0) ? 0 : src_p[src_index + 2 * src_step - 1];
    i = line_end ? 0 : src_p[src_index + 2 * src_step + 16];

    j = (block_x == 0 || y == (rows - 2)) ? 0 : src_p[src_index + 3 * src_step - 1];
    l = (line_end || y == (rows - 2))? 0 : src_p[src_index + 3 * src_step + 16];

#elif defined BORDER_REFLECT_101
    int offset;
    offset = (y == 0) ? (2 * src_step) : 0;

    a = (block_x == 0) ? src_p[src_index + offset + 1] : src_p[src_index + offset - 1];
    c = line_end ? src_p[src_index + offset + 14] : src_p[src_index + offset + 16];

    d = (block_x == 0) ? src_p[src_index + src_step + 1] : src_p[src_index + src_step - 1];
    f = line_end ? src_p[src_index + src_step + 14] : src_p[src_index + src_step + 16];

    g = (block_x == 0) ? src_p[src_index + 2 * src_step + 1] : src_p[src_index + 2 * src_step - 1];
    i = line_end ? src_p[src_index + 2 * src_step + 14] : src_p[src_index + 2 * src_step + 16];

    offset = (y == (rows - 2)) ? (1 * src_step) : (3 * src_step);

    j = (block_x == 0) ? src_p[src_index + offset + 1] : src_p[src_index + offset - 1];
    l = line_end ? src_p[src_index + offset + 14] : src_p[src_index + offset + 16];

#elif defined (BORDER_REPLICATE) || defined(BORDER_REFLECT)
    int offset;
    offset = (y == 0) ? (1 * src_step) : 0;

    a = (block_x == 0) ? src_p[src_index + offset] : src_p[src_index + offset - 1];
    c = line_end ? src_p[src_index + offset + 15] : src_p[src_index + offset + 16];

    d = (block_x == 0) ? src_p[src_index + src_step] : src_p[src_index + src_step - 1];
    f = line_end ? src_p[src_index + src_step + 15] : src_p[src_index + src_step + 16];

    g = (block_x == 0) ? src_p[src_index + 2 * src_step] : src_p[src_index + 2 * src_step - 1];
    i = line_end ? src_p[src_index + 2 * src_step + 15] : src_p[src_index + 2 * src_step + 16];

    offset = (y == (rows - 2)) ? (2 * src_step) : (3 * src_step);

    j = (block_x == 0) ? src_p[src_index + offset] : src_p[src_index + offset - 1];
    l = line_end ? src_p[src_index + offset + 15] : src_p[src_index + offset + 16];

#endif

    uchar16 arr[12];
    float16 sum[2];

    arr[0] = (uchar16)(a, b.s0123, b.s456789ab, b.scde);
    arr[1] = b;
    arr[2] = (uchar16)(b.s123, b.s4567, b.s89abcdef, c);
    arr[3] = (uchar16)(d, e.s0123, e.s456789ab, e.scde);
    arr[4] = e;
    arr[5] = (uchar16)(e.s123, e.s4567, e.s89abcdef, f);
    arr[6] = (uchar16)(g, h.s0123, h.s456789ab, h.scde);
    arr[7] = h;
    arr[8] = (uchar16)(h.s123, h.s4567, h.s89abcdef, i);
    arr[9] = (uchar16)(j, k.s0123, k.s456789ab, k.scde);
    arr[10] = k;
    arr[11] = (uchar16)(k.s123, k.s4567, k.s89abcdef, l);

    sum[0] = OP(0, 0) + OP(0, 1) + OP(0, 2) +
             OP(0, 3) + OP(0, 4) + OP(0, 5) +
             OP(0, 6) + OP(0, 7) + OP(0, 8);

    sum[1] = OP(3, 0) + OP(3, 1) + OP(3, 2) +
             OP(3, 3) + OP(3, 4) + OP(3, 5) +
             OP(3, 6) + OP(3, 7) + OP(3, 8);

    line_out[0] = as_uint4(convert_uchar16_sat_rte(sum[0] + delta));
    line_out[1] = as_uint4(convert_uchar16_sat_rte(sum[1] + delta));

    int dst_index = block_x * 4 * dsx + y * (dst_step / 4);
    vstore4(line_out[0], 0, dst + dst_index);
    vstore4(line_out[1], 0, dst + dst_index + (dst_step / 4));
}
