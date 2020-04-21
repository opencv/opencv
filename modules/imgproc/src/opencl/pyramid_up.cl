// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

__constant float kx[] = { 0.125, 0.5, 0.75, 0.5, 0.125 };
__constant float ky[] = { 0.125, 0.5, 0.75, 0.5, 0.125 };

#define OP(delta, y, x) (convert_float4(arr[(y + delta) * 5 + x]) * ky[y] * kx[x])

__kernel void pyrUp_cols2(__global const uchar * src, int src_step, int src_offset, int src_rows, int src_cols,
                          __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols)
{
    int block_x = get_global_id(0);
    int y = get_global_id(1) * 2;

    if ((block_x * 4) >= dst_cols || y >= dst_rows) return;

    uchar8 line[6];
    uchar4 line_out;

    int offset, src_index;
    src_index = block_x * 2 + (y / 2 - 1) * src_step - 1 + src_offset;

    uchar4 tmp;

    line[0] = line[2] = line[4] = (uchar8)0;
    line[1] = line[3] = line[5] = (uchar8)0;

    offset = max(0, src_index + 1 * src_step);
    tmp = vload4(0, src + offset);
    if (offset == 0) tmp = (uchar4)(0, tmp.s012);
    line[2].even = tmp;

    offset = max(0, src_index + ((y == 0) ? 2 : 0) * src_step);
    tmp = vload4(0, src + offset);
    if (offset == 0) tmp = (uchar4)(0, tmp.s012);
    line[0].even = tmp;

    if (y == (dst_rows - 2))
        line[4] = line[2];
    else
        line[4].even = vload4(0, src + src_index + 2 * src_step);

    bool row_s = (block_x == 0);
    bool row_e = ((block_x + 1) * 4 == dst_cols);
    uchar4 arr[30];
    uchar s, e;

    s = line[0].s4;
    e = line[0].s3;
    arr[0] = row_s ? (uchar4)(s, e, line[0].s23) : (uchar4)(line[0].s0123);
    arr[1] = row_s ? (uchar4)(e, line[0].s234) : (uchar4)(line[0].s1234);
    arr[2] = (uchar4)(line[0].s2345);
    arr[3] = row_e ? (uchar4)(line[0].s345, s) : (uchar4)(line[0].s3456);
    arr[4] = row_e ? (uchar4)(line[0].s45, s, e) : (uchar4)(line[0].s4567);

    s = line[1].s4;
    e = line[1].s3;
    arr[5] = row_s ? (uchar4)(s, e, line[1].s23) : (uchar4)(line[1].s0123);
    arr[6] = row_s ? (uchar4)(e, line[1].s234) : (uchar4)(line[1].s1234);
    arr[7] = (uchar4)(line[1].s2345);
    arr[8] = row_e ? (uchar4)(line[1].s345, s) : (uchar4)(line[1].s3456);
    arr[9] = row_e ? (uchar4)(line[1].s45, s, e) : (uchar4)(line[1].s4567);

    s = line[2].s4;
    e = line[2].s3;
    arr[10] = row_s ? (uchar4)(s, e, line[2].s23) : (uchar4)(line[2].s0123);
    arr[11] = row_s ? (uchar4)(e, line[2].s234) : (uchar4)(line[2].s1234);
    arr[12] = (uchar4)(line[2].s2345);
    arr[13] = row_e ? (uchar4)(line[2].s345, s) : (uchar4)(line[2].s3456);
    arr[14] = row_e ? (uchar4)(line[2].s45, s, e) : (uchar4)(line[2].s4567);

    s = line[3].s4;
    e = line[3].s3;
    arr[15] = row_s ? (uchar4)(s, e, line[3].s23) : (uchar4)(line[3].s0123);
    arr[16] = row_s ? (uchar4)(e, line[3].s234) : (uchar4)(line[3].s1234);
    arr[17] = (uchar4)(line[3].s2345);
    arr[18] = row_e ? (uchar4)(line[3].s345, s) : (uchar4)(line[3].s3456);
    arr[19] = row_e ? (uchar4)(line[3].s45, s, e) : (uchar4)(line[3].s4567);

    s = line[4].s4;
    e = line[4].s3;
    arr[20] = row_s ? (uchar4)(s, e, line[4].s23) : (uchar4)(line[4].s0123);
    arr[21] = row_s ? (uchar4)(e, line[4].s234) : (uchar4)(line[4].s1234);
    arr[22] = (uchar4)(line[4].s2345);
    arr[23] = row_e ? (uchar4)(line[4].s345, s) : (uchar4)(line[4].s3456);
    arr[24] = row_e ? (uchar4)(line[4].s45, s, e) : (uchar4)(line[4].s4567);

    s = line[5].s4;
    e = line[5].s3;
    arr[25] = row_s ? (uchar4)(s, e, line[5].s23) : (uchar4)(line[5].s0123);
    arr[26] = row_s ? (uchar4)(e, line[5].s234) : (uchar4)(line[5].s1234);
    arr[27] = (uchar4)(line[5].s2345);
    arr[28] = row_e ? (uchar4)(line[5].s345, s) : (uchar4)(line[5].s3456);
    arr[29] = row_e ? (uchar4)(line[5].s45, s, e) : (uchar4)(line[5].s4567);

    float4 sum[2];

    sum[0] = OP(0, 0, 0) + OP(0, 0, 1) + OP(0, 0, 2) + OP(0, 0, 3) + OP(0, 0, 4) +
             OP(0, 1, 0) + OP(0, 1, 1) + OP(0, 1, 2) + OP(0, 1, 3) + OP(0, 1, 4) +
             OP(0, 2, 0) + OP(0, 2, 1) + OP(0, 2, 2) + OP(0, 2, 3) + OP(0, 2, 4) +
             OP(0, 3, 0) + OP(0, 3, 1) + OP(0, 3, 2) + OP(0, 3, 3) + OP(0, 3, 4) +
             OP(0, 4, 0) + OP(0, 4, 1) + OP(0, 4, 2) + OP(0, 4, 3) + OP(0, 4, 4);

    sum[1] = OP(1, 0, 0) + OP(1, 0, 1) + OP(1, 0, 2) + OP(1, 0, 3) + OP(1, 0, 4) +
             OP(1, 1, 0) + OP(1, 1, 1) + OP(1, 1, 2) + OP(1, 1, 3) + OP(1, 1, 4) +
             OP(1, 2, 0) + OP(1, 2, 1) + OP(1, 2, 2) + OP(1, 2, 3) + OP(1, 2, 4) +
             OP(1, 3, 0) + OP(1, 3, 1) + OP(1, 3, 2) + OP(1, 3, 3) + OP(1, 3, 4) +
             OP(1, 4, 0) + OP(1, 4, 1) + OP(1, 4, 2) + OP(1, 4, 3) + OP(1, 4, 4);

    int dst_index = block_x * 4 + y * dst_step + dst_offset;
    vstore4(convert_uchar4_sat_rte(sum[0]), 0, dst + dst_index);
    vstore4(convert_uchar4_sat_rte(sum[1]), 0, dst + dst_index + dst_step);
}
