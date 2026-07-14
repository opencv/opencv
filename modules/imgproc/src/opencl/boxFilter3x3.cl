// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

__kernel void boxFilter3x3_8UC1_cols16_rows2(__global const uint* src, int src_step,
                                             __global uint* dst, int dst_step, int rows, int cols
#ifdef NORMALIZE
                                             , float alpha
#endif
                         )
{
    int block_x = get_global_id(0);
    int y = get_global_id(1) * 2;
    int ssx, dsx;

    if ((block_x * 16) >= cols || y >= rows) return;

    uint4 line[4];
    uint4 line_out[2];
    ushort a; ushort16 b; ushort c;
    ushort d; ushort16 e; ushort f;
    ushort g; ushort16 h; ushort i;
    ushort j; ushort16 k; ushort l;

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

    ushort16 sum, mid;
    __global uchar *src_p = (__global uchar *)src;

    src_index = block_x * 16 * ssx + (y - 1) * src_step;
    bool line_end = ((block_x + 1) * 16 == cols);

    b = convert_ushort16(as_uchar16(line[0]));
    e = convert_ushort16(as_uchar16(line[1]));
    h = convert_ushort16(as_uchar16(line[2]));
    k = convert_ushort16(as_uchar16(line[3]));

#ifdef BORDER_CONSTANT
    a = (block_x == 0 || y == 0) ? 0 : convert_ushort(src_p[src_index - 1]);
    c = (line_end || y == 0) ? 0 : convert_ushort(src_p[src_index + 16]);

    d = (block_x == 0) ? 0 : convert_ushort(src_p[src_index + src_step - 1]);
    f = line_end ? 0 : convert_ushort(src_p[src_index + src_step + 16]);

    g = (block_x == 0) ? 0 : convert_ushort(src_p[src_index + 2 * src_step - 1]);
    i = line_end ? 0 : convert_ushort(src_p[src_index + 2 * src_step + 16]);

    j = (block_x == 0 || y == (rows - 2)) ? 0 : convert_ushort(src_p[src_index + 3 * src_step - 1]);
    l = (line_end || y == (rows - 2))? 0 : convert_ushort(src_p[src_index + 3 * src_step + 16]);

#elif defined BORDER_REFLECT_101
    int offset;
    offset = (y == 0) ? (2 * src_step) : 0;

    a = (block_x == 0) ? convert_ushort(src_p[src_index + offset + 1]) : convert_ushort(src_p[src_index + offset - 1]);
    c = line_end ? convert_ushort(src_p[src_index + offset + 14]) : convert_ushort(src_p[src_index + offset + 16]);

    d = (block_x == 0) ? convert_ushort(src_p[src_index + src_step + 1]) : convert_ushort(src_p[src_index + src_step - 1]);
    f = line_end ? convert_ushort(src_p[src_index + src_step + 14]) : convert_ushort(src_p[src_index + src_step + 16]);

    g = (block_x == 0) ? convert_ushort(src_p[src_index + 2 * src_step + 1]) : convert_ushort(src_p[src_index + 2 * src_step - 1]);
    i = line_end ? convert_ushort(src_p[src_index + 2 * src_step + 14]) : convert_ushort(src_p[src_index + 2 * src_step + 16]);

    offset = (y == (rows - 2)) ? (1 * src_step) : (3 * src_step);

    j = (block_x == 0) ? convert_ushort(src_p[src_index + offset + 1]) : convert_ushort(src_p[src_index + offset - 1]);
    l = line_end ? convert_ushort(src_p[src_index + offset + 14]) : convert_ushort(src_p[src_index + offset + 16]);

#elif defined (BORDER_REPLICATE) || defined(BORDER_REFLECT)
    int offset;
    offset = (y == 0) ? (1 * src_step) : 0;

    a = (block_x == 0) ? convert_ushort(src_p[src_index + offset]) : convert_ushort(src_p[src_index + offset - 1]);
    c = line_end ? convert_ushort(src_p[src_index + offset + 15]) : convert_ushort(src_p[src_index + offset + 16]);

    d = (block_x == 0) ? convert_ushort(src_p[src_index + src_step]) : convert_ushort(src_p[src_index + src_step - 1]);
    f = line_end ? convert_ushort(src_p[src_index + src_step + 15]) : convert_ushort(src_p[src_index + src_step + 16]);

    g = (block_x == 0) ? convert_ushort(src_p[src_index + 2 * src_step]) : convert_ushort(src_p[src_index + 2 * src_step - 1]);
    i = line_end ? convert_ushort(src_p[src_index + 2 * src_step + 15]) : convert_ushort(src_p[src_index + 2 * src_step + 16]);

    offset = (y == (rows - 2)) ? (2 * src_step) : (3 * src_step);

    j = (block_x == 0) ? convert_ushort(src_p[src_index + offset]) : convert_ushort(src_p[src_index + offset - 1]);
    l = line_end ? convert_ushort(src_p[src_index + offset + 15]) : convert_ushort(src_p[src_index + offset + 16]);

#endif

    mid = (ushort16)(d, e.s0123, e.s456789ab, e.scde) + e + (ushort16)(e.s123, e.s4567, e.s89abcdef, f) +
          (ushort16)(g, h.s0123, h.s456789ab, h.scde) + h + (ushort16)(h.s123, h.s4567, h.s89abcdef, i);

    sum = (ushort16)(a, b.s0123, b.s456789ab, b.scde) + b + (ushort16)(b.s123, b.s4567, b.s89abcdef, c) +
          mid;

#ifdef NORMALIZE
    line_out[0] = as_uint4(convert_uchar16_sat_rte((convert_float16(sum) * alpha)));
#else
    line_out[0] = as_uint4(convert_uchar16_sat_rte(sum));
#endif

    sum = mid +
          (ushort16)(j, k.s0123, k.s456789ab, k.scde) + k + (ushort16)(k.s123, k.s4567, k.s89abcdef, l);

#ifdef NORMALIZE
    line_out[1] = as_uint4(convert_uchar16_sat_rte((convert_float16(sum) * alpha)));
#else
    line_out[1] = as_uint4(convert_uchar16_sat_rte(sum));
#endif

    int dst_index = block_x * 4 * dsx + y * (dst_step / 4);
    vstore4(line_out[0], 0, dst + dst_index);
    vstore4(line_out[1], 0, dst + dst_index + (dst_step / 4));
}
