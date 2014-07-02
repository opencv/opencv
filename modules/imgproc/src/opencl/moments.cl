/* See LICENSE file in the root OpenCV directory */

#if TILE_SIZE != 32
#error "TILE SIZE should be 32"
#endif

__kernel void moments(__global const uchar* src, int src_step, int src_offset,
    int src_rows, int src_cols, __global int* mom0, int xtiles)
{
    int x0 = get_global_id(0);
    int y0 = get_group_id(1);
    int x, y = get_local_id(1);
    int x_min = x0*TILE_SIZE;
    int ypix = y0*TILE_SIZE + y;
    __local int mom[TILE_SIZE][10];

    if (x_min < src_cols && y0*TILE_SIZE < src_rows)
    {
        if (ypix < src_rows)
        {
            int x_max = min(src_cols - x_min, TILE_SIZE);
            __global const uchar* ptr = src + src_offset + ypix*src_step + x_min;
            int4 S = (int4)(0, 0, 0, 0);
            x = x_max & -4;

#define SUM_ELEM(elem, ofs) \
    (int4)(1, (ofs), (ofs)*(ofs), (ofs)*(ofs)*(ofs))*elem

            //get bit value
            int bitval0 = (int)((bool)(x_max / 32));// int bv0 = (int)bitval0;
            int bitval1 = (int)((bool)(x_max / 28));// int bv1 = (int)bitval1;
            int bitval2 = (int)((bool)(x_max / 24));// int bv2 = (int)bitval2;
            int bitval3 = (int)((bool)(x_max / 20));// int bv3 = (int)bitval3;
            int bitval4 = (int)((bool)(x_max / 16));// int bv4 = (int)bitval4;
            int bitval5 = (int)((bool)(x_max / 12));// int bv5 = (int)bitval5;
            int bitval6 = (int)((bool)(x_max / 8));// int bv6 = (int)bitval6;
            int bitval7 = (int)((bool)(x_max / 4));// int bv7 = (int)bitval7;

            int16 p;
            p = convert_int16(vload16(0, ptr));
#ifdef OP_MOMENTS_BINARY
            p = min(p, 1);
#endif
            //ptr + 0
            S += bitval7 * ((int4)(p.s0, 0, 0, 0) + (int4)(p.s1, p.s1, p.s1, p.s1) +
                (int4)(p.s2, p.s2 * 2, p.s2 * 4, p.s2 * 8) + (int4)(p.s3, p.s3 * 3, p.s3 * 9, p.s3 * 27)) +//ptr + 4
                bitval6 * ((int4)(p.s4, p.s4 * 4, p.s4 * 16, p.s4 * 64) + (int4)(p.s5, p.s5 * 5, p.s5 * 25, p.s5 * 125) +
                (int4)(p.s6, p.s6 * 6, p.s6 * 36, p.s6 * 216) + (int4)(p.s7, p.s7 * 7, p.s7 * 49, p.s7 * 343)) +//ptr + 8
                bitval5 * ((int4)(p.s8, p.s8 * 8, p.s8 * 64, p.s8 * 512) + (int4)(p.s9, p.s9 * 9, p.s9 * 81, p.s9 * 729) +
                (int4)(p.sa, p.sa * 10, p.sa * 100, p.sa * 1000) + (int4)(p.sb, p.sb * 11, p.sb * 121, p.sb * 1331)) +//ptr + 12
                bitval4 * ((int4)(p.sc, p.sc * 12, p.sc * 144, p.sc * 1728) + (int4)(p.sd, p.sd * 13, p.sd * 169, p.sd * 2197) +
                (int4)(p.se, p.se * 14, p.se * 196, p.se * 2744) + (int4)(p.sf, p.sf * 15, p.sf * 225, p.sf * 3375));

            //read next half of tile
            p = convert_int16(vload16(0, ptr + 16));
#ifdef OP_MOMENTS_BINARY
            p = min(p, 1);
#endif
            //ptr + 16
            S += bitval3 * ((int4)(p.s0, p.s0 * 16, p.s0 * 256, p.s0 * 4096) + (int4)(p.s1, p.s1 * 17, p.s1 * 289, p.s1 * 4913) +
                (int4)(p.s2, p.s2 * 18, p.s2 * 324, p.s2 * 5832) + (int4)(p.s3, p.s3 * 19, p.s3 * 361, p.s3 * 6859)) +//ptr + 20
                bitval2 * ((int4)(p.s4, p.s4 * 20, p.s4 * 400, p.s4 * 8000) + (int4)(p.s5, p.s5 * 21, p.s5 * 441, p.s5 * 9261) +
                (int4)(p.s6, p.s6 * 22, p.s6 * 484, p.s6 * 10648) + (int4)(p.s7, p.s7 * 23, p.s7 * 529, p.s7 * 12167)) +//ptr  + 24
                bitval1 * ((int4)(p.s8, p.s8 * 24, p.s8 * 576, p.s8 * 13824) + (int4)(p.s9, p.s9 * 25, p.s9 * 625, p.s9 * 15625) +
                (int4)(p.sa, p.sa * 26, p.sa * 676, p.sa * 17576) + (int4)(p.sb, p.sb * 27, p.sb * 729, p.sb * 19683)) + //ptr + 28
                bitval0 * ((int4)(p.sc, p.sc * 28, p.sc * 784, p.sc * 21952) + (int4)(p.sd, p.sd * 29, p.sd * 841, p.sd * 24389) +
                (int4)(p.se, p.se * 30, p.se * 900, p.se * 27000) + (int4)(p.sf, p.sf * 31, p.sf * 961, p.sf * 29791));

            if (x < x_max)
            {
                int ps = ptr[x];
#ifdef OP_MOMENTS_BINARY
                ps = min(ps, 1);
#endif
                S += SUM_ELEM(ps, x);
                if (x + 1 < x_max)
                {
                    ps = ptr[x + 1];
#ifdef OP_MOMENTS_BINARY
                    ps = min(ps, 1);
#endif
                    S += SUM_ELEM(ps, x + 1);
                    if (x + 2 < x_max)
                    {
                        ps = ptr[x + 2];
#ifdef OP_MOMENTS_BINARY
                        ps = min(ps, 1);
#endif
                        S += SUM_ELEM(ps, x + 2);
                    }
                }
            }

            int sy = y*y;

            mom[y][0] = S.s0;
            mom[y][1] = S.s1;
            mom[y][2] = y*S.s0;
            mom[y][3] = S.s2;
            mom[y][4] = y*S.s1;
            mom[y][5] = sy*S.s0;
            mom[y][6] = S.s3;
            mom[y][7] = y*S.s2;
            mom[y][8] = sy*S.s1;
            mom[y][9] = y*sy*S.s0;
        }
        else
            mom[y][0] = mom[y][1] = mom[y][2] = mom[y][3] = mom[y][4] =
            mom[y][5] = mom[y][6] = mom[y][7] = mom[y][8] = mom[y][9] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);

#define REDUCE(d) \
        if (y < d) \
        { \
        mom[y][0] += mom[y + d][0]; \
        mom[y][1] += mom[y + d][1]; \
        mom[y][2] += mom[y + d][2]; \
        mom[y][3] += mom[y + d][3]; \
        mom[y][4] += mom[y + d][4]; \
        mom[y][5] += mom[y + d][5]; \
        mom[y][6] += mom[y + d][6]; \
        mom[y][7] += mom[y + d][7]; \
        mom[y][8] += mom[y + d][8]; \
        mom[y][9] += mom[y + d][9]; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE)

        REDUCE(16);
        REDUCE(8);
        REDUCE(4);
        REDUCE(2);

        if (y < 10)
        {
            __global int* momout = mom0 + (y0*xtiles + x0) * 10;
            momout[y] = (int)(mom[0][y] + mom[1][y]);
        }
    }
}
