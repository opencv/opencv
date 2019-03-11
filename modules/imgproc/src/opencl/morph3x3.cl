// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifdef OP_ERODE
#define OP(m1, m2) min(m1, m2)
#define VAL UCHAR_MAX
#endif

#ifdef OP_DILATE
#define OP(m1, m2) max(m1, m2)
#define VAL 0
#endif

#if defined OP_GRADIENT || defined OP_TOPHAT || defined OP_BLACKHAT
#define EXTRA_PARAMS , __global const uchar * matptr, int mat_step, int mat_offset
#else
#define EXTRA_PARAMS
#endif

#define PROCESS(_y, _x) \
    line_out[0] = OP(line_out[0], arr[_x + 3 * _y]); \
    line_out[1] = OP(line_out[1], arr[_x + 3 * (_y + 1)]);

#define PROCESS_ELEM \
    line_out[0] = (uchar16)VAL; \
    line_out[1] = (uchar16)VAL; \
    PROCESS_ELEM_

__kernel void morph3x3_8UC1_cols16_rows2(__global const uint* src, int src_step,
                                         __global uint* dst, int dst_step,
                                         int rows, int cols
                                         EXTRA_PARAMS)
{
    int block_x = get_global_id(0);
    int y = get_global_id(1) * 2;
    int ssx = 1, dsx = 1;

    if ((block_x * 16) >= cols || y >= rows) return;

    uchar a; uchar16 b; uchar c;
    uchar d; uchar16 e; uchar f;
    uchar g; uchar16 h; uchar i;
    uchar j; uchar16 k; uchar l;

    uchar16 line[4];
    uchar16 line_out[2];

    int src_index = block_x * 4 * ssx + (y - 1) * (src_step / 4);
    line[0] = (y == 0) ? (uchar16)VAL: as_uchar16(vload4(0, src + src_index));
    line[1] = as_uchar16(vload4(0, src + src_index + (src_step / 4)));
    line[2] = as_uchar16(vload4(0, src + src_index + 2 * (src_step / 4)));
    line[3] = (y == (rows - 2)) ? (uchar16)VAL: as_uchar16(vload4(0, src + src_index + 3 * (src_step / 4)));

    __global uchar *src_p = (__global uchar *)src;
    bool line_end = ((block_x + 1) * 16 == cols);

    src_index = block_x * 16 * ssx + (y - 1) * src_step;

    a = (block_x == 0 || y == 0) ? VAL : src_p[src_index - 1];
    b = line[0];
    c = (line_end || y == 0) ? VAL : src_p[src_index + 16];

    d = (block_x == 0) ? VAL : src_p[src_index + src_step - 1];
    e = line[1];
    f = line_end ? VAL : src_p[src_index + src_step + 16];

    g = (block_x == 0) ? VAL : src_p[src_index + 2 * src_step - 1];
    h = line[2];
    i = line_end ? VAL : src_p[src_index + 2 * src_step + 16];

    j = (block_x == 0 || y == (rows - 2)) ? VAL : src_p[src_index + 3 * src_step - 1];
    k = line[3];
    l = (line_end || y == (rows - 2)) ? VAL : src_p[src_index + 3 * src_step + 16];

    uchar16 arr[12];
    arr[0] = (uchar16)(a, b.s01234567, b.s89ab, b.scde);
    arr[1] = b;
    arr[2] = (uchar16)(b.s12345678, b.s9abc, b.sdef, c);
    arr[3] = (uchar16)(d, e.s01234567, e.s89ab, e.scde);
    arr[4] = e;
    arr[5] = (uchar16)(e.s12345678, e.s9abc, e.sdef, f);
    arr[6] = (uchar16)(g, h.s01234567, h.s89ab, h.scde);
    arr[7] = h;
    arr[8] = (uchar16)(h.s12345678, h.s9abc, h.sdef, i);
    arr[9] = (uchar16)(j, k.s01234567, k.s89ab, k.scde);
    arr[10] = k;
    arr[11] = (uchar16)(k.s12345678, k.s9abc, k.sdef, l);

    PROCESS_ELEM;

    int dst_index = block_x * 4 * dsx + y * (dst_step / 4);

#if defined OP_GRADIENT || defined OP_TOPHAT || defined OP_BLACKHAT
    int mat_index = y * mat_step + block_x * 16 * ssx + mat_offset;
    uchar16 val0 = vload16(0, matptr + mat_index);
    uchar16 val1 = vload16(0, matptr + mat_index + mat_step);

#ifdef OP_GRADIENT
    line_out[0] = convert_uchar16_sat(convert_int16(line_out[0]) - convert_int16(val0));
    line_out[1] = convert_uchar16_sat(convert_int16(line_out[1]) - convert_int16(val1));
    vstore4(as_uint4(line_out[0]), 0, dst + dst_index);
    vstore4(as_uint4(line_out[1]), 0, dst + dst_index + (dst_step / 4));
#elif defined OP_TOPHAT
    line_out[0] = convert_uchar16_sat(convert_int16(val0) - convert_int16(line_out[0]));
    line_out[1] = convert_uchar16_sat(convert_int16(val1) - convert_int16(line_out[1]));
    vstore4(as_uint4(line_out[0]), 0, dst + dst_index);
    vstore4(as_uint4(line_out[1]), 0, dst + dst_index + (dst_step / 4));
#elif defined OP_BLACKHAT
    line_out[0] = convert_uchar16_sat(convert_int16(line_out[0]) - convert_int16(val0));
    line_out[1] = convert_uchar16_sat(convert_int16(line_out[1]) - convert_int16(val1));
    vstore4(as_uint4(line_out[0]), 0, dst + dst_index);
    vstore4(as_uint4(line_out[1]), 0, dst + dst_index + (dst_step / 4));
#endif
#else
    vstore4(as_uint4(line_out[0]), 0, dst + dst_index);
    vstore4(as_uint4(line_out[1]), 0, dst + dst_index + (dst_step / 4));
#endif
}
