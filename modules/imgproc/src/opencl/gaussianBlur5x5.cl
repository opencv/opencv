// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#define DIG(a) a,
__constant float kx[] = { KERNEL_MATRIX_X };
__constant float ky[] = { KERNEL_MATRIX_Y };

#define OP(y, x) (convert_float4(arr[y * 5 + x]) * ky[y] * kx[x])

#define FILL_ARR(s1, s2, n, e1, e2)                                                   \
    arr[5 * n + 0] = row_s ? (uchar4)(s1, s2, line[n].s23) : (uchar4)(line[n].s0123); \
    arr[5 * n + 1] = row_s ? (uchar4)(s2, line[n].s234) : (uchar4)(line[n].s1234);    \
    arr[5 * n + 2] = (uchar4)(line[n].s2345);                                         \
    arr[5 * n + 3] = row_e ? (uchar4)(line[n].s345, e1) : (uchar4)(line[n].s3456);    \
    arr[5 * n + 4] = row_e ? (uchar4)(line[n].s45, e1, e2) : (uchar4)(line[n].s4567);

__kernel void gaussianBlur5x5_8UC1_cols4(__global const uchar* src, int src_step,
                                         __global uint* dst, int dst_step, int rows, int cols)
{
    int x = get_global_id(0) * 4;
    int y = get_global_id(1);

    if (x >= cols || y >= rows) return;

    uchar8 line[5];
    int offset, src_index;

    src_index = x + (y - 2) * src_step - 2;
    offset = max(0, src_index + 2 * src_step);
    line[2] = vload8(0, src + offset);
    if (offset == 0) line[2] = (uchar8)(0, 0, line[2].s0123, line[2].s45);

#if defined BORDER_CONSTANT || defined BORDER_REPLICATE
    uchar8 tmp;
#ifdef BORDER_CONSTANT
    tmp = (uchar8)0;
#elif defined BORDER_REPLICATE
    tmp = line[2];
#endif
    line[0] = line[1] = tmp;
    if (y > 1)
    {
        offset = max(0, src_index);
        line[0] = vload8(0, src + offset);
        if (offset == 0) line[0] = (uchar8)(0, 0, line[0].s0123, line[0].s45);
    }

    if (y > 0)
    {
        offset = max(0, src_index + src_step);
        line[1] = vload8(0, src + offset);
        if (offset == 0) line[1] = (uchar8)(0, 0, line[1].s0123, line[1].s45);
    }

    line[3] = (y == (rows - 1)) ? tmp : vload8(0, src + src_index + 3 * src_step);
    line[4] = (y >= (rows - 2)) ? tmp : vload8(0, src + src_index + 4 * src_step);
#elif BORDER_REFLECT
    int t;
    t = (y <= 1) ? (abs(y - 1) - y + 2) : 0;
    offset = max(0, src_index + t * src_step);
    line[0] = vload8(0, src + offset);
    if (offset == 0) line[0] = (uchar8)(0, 0, line[0].s0123, line[0].s45);

    if (y == 0)
        line[1] = line[2];
    else
    {
        offset = max(0, src_index + 1 * src_step);
        line[1] = vload8(0, src + offset);
        if (offset == 0) line[1] = (uchar8)(0, 0, line[1].s0123, line[0].s45);
    }

    line[3] = (y == (rows - 1)) ? line[2] : vload8(0, src + src_index + 3 * src_step);

    t = (y >= (rows - 2)) ? (abs(y - (rows - 1)) - (y - (rows - 2)) + 2) : 4;
    line[4] = vload8(0, src + src_index + t * src_step);
#elif BORDER_REFLECT_101
    if (y == 1)
        line[0] = line[2];
    else
    {
        offset = (y == 0) ? (src_index + 4 * src_step) : max(0, src_index);
        line[0] = vload8(0, src + offset);
        if (offset == 0) line[0] = (uchar8)(0, 0, line[0].s0123, line[0].s45);
    }

    offset = (y == 0) ? (src_index + 3 * src_step) : max(0, src_index + 1 * src_step);
    line[1] = vload8(0, src + offset);
    if (offset == 0) line[1] = (uchar8)(0, 0, line[1].s0123, line[1].s45);

    line[3] = vload8(0, src + src_index + ((y == (rows - 1)) ? 1 : 3) * src_step);
    if (y == (rows - 2))
        line[4] = line[2];
    else
    {
        line[4] = vload8(0, src + src_index + ((y == (rows - 1)) ? 1 : 4) * src_step);
    }
#endif

    bool row_s = (x == 0);
    bool row_e = ((x + 4) == cols);
    uchar4 arr[25];
    uchar s, e;

#ifdef BORDER_CONSTANT
    s = e = 0;

    FILL_ARR(s, s, 0, e, e);
    FILL_ARR(s, s, 1, e, e);
    FILL_ARR(s, s, 2, e, e);
    FILL_ARR(s, s, 3, e, e);
    FILL_ARR(s, s, 4, e, e);
#elif defined BORDER_REPLICATE
    s = line[0].s2;
    e = line[0].s5;
    FILL_ARR(s, s, 0, e, e);

    s = line[1].s2;
    e = line[1].s5;
    FILL_ARR(s, s, 1, e, e);

    s = line[2].s2;
    e = line[2].s5;
    FILL_ARR(s, s, 2, e, e);

    s = line[3].s2;
    e = line[3].s5;
    FILL_ARR(s, s, 3, e, e);

    s = line[4].s2;
    e = line[4].s5;
    FILL_ARR(s, s, 4, e, e);
#elif BORDER_REFLECT
    uchar s1, s2;
    uchar e1, e2;

    s1 = line[0].s3;
    s2 = line[0].s2;
    e1 = line[0].s5;
    e2 = line[0].s4;
    FILL_ARR(s1, s2, 0, e1, e2);

    s1 = line[1].s3;
    s2 = line[1].s2;
    e1 = line[1].s5;
    e2 = line[1].s4;
    FILL_ARR(s1, s2, 1, e1, e2);

    s1 = line[2].s3;
    s2 = line[2].s2;
    e1 = line[2].s5;
    e2 = line[2].s4;
    FILL_ARR(s1, s2, 2, e1, e2);

    s1 = line[3].s3;
    s2 = line[3].s2;
    e1 = line[3].s5;
    e2 = line[3].s4;
    FILL_ARR(s1, s2, 3, e1, e2);

    s1 = line[4].s3;
    s2 = line[4].s2;
    e1 = line[4].s5;
    e2 = line[4].s4;
    FILL_ARR(s1, s2, 4, e1, e2);
#elif BORDER_REFLECT_101
    s = line[0].s4;
    e = line[0].s3;
    FILL_ARR(s, e, 0, s, e);

    s = line[1].s4;
    e = line[1].s3;
    FILL_ARR(s, e, 1, s, e);

    s = line[2].s4;
    e = line[2].s3;
    FILL_ARR(s, e, 2, s, e);

    s = line[3].s4;
    e = line[3].s3;
    FILL_ARR(s, e, 3, s, e);

    s = line[4].s4;
    e = line[4].s3;
    FILL_ARR(s, e, 4, s, e);
#endif

    float4 sum;
    sum = OP(0, 0) + OP(0, 1) + OP(0, 2) + OP(0, 3) + OP(0, 4) +
          OP(1, 0) + OP(1, 1) + OP(1, 2) + OP(1, 3) + OP(1, 4) +
          OP(2, 0) + OP(2, 1) + OP(2, 2) + OP(2, 3) + OP(2, 4) +
          OP(3, 0) + OP(3, 1) + OP(3, 2) + OP(3, 3) + OP(3, 4) +
          OP(4, 0) + OP(4, 1) + OP(4, 2) + OP(4, 3) + OP(4, 4);

    int dst_index = (x / 4) + y * (dst_step / 4);
    dst[dst_index] = as_uint(convert_uchar4_sat_rte(sum));
}
