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
            float4 S = (float4)(0, 0, 0, 0);
            x = x_max & -4;

#define SUM_ELEM(elem, ofs) \
    (float4)(1, (ofs), (ofs)*(ofs), (ofs)*(ofs)*(ofs))*elem
            /*
            //in most of cases we work with all x from tile, process it separately
            //load all values together, and use for S computation
            if (x_max >= 32)
            {
                int16 p;
                p = convert_int16(vload16(0, ptr));
#ifdef OP_MOMENTS_BINARY
                p = min(p, 1);
#endif
                //ptr + 0
                S += (int4)(p.s0, 0, 0, 0) + (int4)(p.s1, p.s1, p.s1, p.s1) +
                    (int4)(p.s2, p.s2 * 2, p.s2 * 4, p.s2 * 8) + (int4)(p.s3, p.s3 * 3, p.s3 * 9, p.s3 * 27) +//ptr + 4
                    (int4)(p.s4, p.s4 * 4, p.s4 * 16, p.s4 * 64) + (int4)(p.s5, p.s5 * 5, p.s5 * 25, p.s5 * 125) +
                    (int4)(p.s6, p.s6 * 6, p.s6 * 36, p.s6 * 216) + (int4)(p.s7, p.s7 * 7, p.s7 * 49, p.s7 * 343) +//ptr + 8
                    (int4)(p.s8, p.s8 * 8, p.s8 * 64, p.s8 * 512)  + (int4)(p.s9, p.s9 * 9,  p.s9 * 81,  p.s9 * 729) +
                    (int4)(p.sa, p.sa * 10, p.sa * 100, p.sa * 1000) + (int4)(p.sb, p.sb * 11, p.sb * 121, p.sb * 1331) +//ptr + 12
                    (int4)(p.sc, p.sc * 12, p.sc * 144, p.sc * 1728) + (int4)(p.sd, p.sd * 13, p.sd * 169, p.sd * 2197) +
                    (int4)(p.se, p.se * 14, p.se * 196, p.se * 2744) + (int4)(p.sf, p.sf * 15, p.sf * 225, p.sf * 3375);
                //read next half of tile
                p = convert_int16(vload16(0, ptr + 16));
#ifdef OP_MOMENTS_BINARY
                p = min(p, 1);
#endif
                //ptr + 16
                S += (int4)(p.s0, p.s0 * 16, p.s0 * 256, p.s0 * 4096) + (int4)(p.s1, p.s1 * 17, p.s1 * 289, p.s1 * 4913) +
                    (int4)(p.s2, p.s2 * 18, p.s2 * 324, p.s2 * 5832) + (int4)(p.s3, p.s3 * 19, p.s3 * 361, p.s3 * 6859) +//ptr + 20
                    (int4)(p.s4, p.s4 * 20, p.s4 * 400, p.s4 * 8000) + (int4)(p.s5, p.s5 * 21, p.s5 * 441, p.s5 * 9261) +
                    (int4)(p.s6, p.s6 * 22, p.s6 * 484, p.s6 * 10648) + (int4)(p.s7, p.s7 * 23, p.s7 * 529, p.s7 * 12167) +//ptr  + 24
                    (int4)(p.s8, p.s8 * 24, p.s8 * 576, p.s8 * 13824) + (int4)(p.s9, p.s9 * 25, p.s9 * 625, p.s9 * 15625) +
                    (int4)(p.sa, p.sa * 26, p.sa * 676, p.sa * 17576) + (int4)(p.sb, p.sb * 27, p.sb * 729, p.sb * 19683)+ //ptr + 28
                    (int4)(p.sc, p.sc * 28, p.sc * 784, p.sc * 21952) + (int4)(p.sd, p.sd * 29, p.sd * 841, p.sd * 24389) +
                    (int4)(p.se, p.se * 30, p.se * 900, p.se * 27000) + (int4)(p.sf, p.sf * 31, p.sf * 961, p.sf * 29791);
            }
            else
            {*/
                float4 p;
                if (x_max >= 4)
                {
                    //__global const uchar* ptr = src + src_offset + ypix*src_step + x_min;
                    p = convert_float4(vload4(0, ptr));
#ifdef OP_MOMENTS_BINARY
                    p = min(p, 1);
#endif
                    S += (float4)(p.s0, 0, 0, 0) + (float4)(p.s1, p.s1, p.s1, p.s1) +
                        (float4)(p.s2, p.s2 * 2, p.s2 * 4, p.s2 * 8) + (float4)(p.s3, p.s3 * 3, p.s3 * 9, p.s3 * 27);
                    //SUM_ELEM(p.s0, 0) + SUM_ELEM(p.s1, 1) + SUM_ELEM(p.s2, 2) + SUM_ELEM(p.s3, 3);

                    if (x_max >= 8)
                    {
                        p = convert_float4(vload4(0, ptr + 4));
#ifdef OP_MOMENTS_BINARY
                        p = min(p, 1);
#endif
                        S += (float4)(p.s0, p.s0 * 4, p.s0 * 16, p.s0 * 64) + (float4)(p.s1, p.s1 * 5, p.s1 * 25, p.s1 * 125) +
                            (float4)(p.s2, p.s2 * 6, p.s2 * 36, p.s2 * 216) + (float4)(p.s3, p.s3 * 7, p.s3 * 49, p.s3 * 343);
                        //SUM_ELEM(p.s0, 4) + SUM_ELEM(p.s1, 5) + SUM_ELEM(p.s2, 6) + SUM_ELEM(p.s3, 7);

                        if (x_max >= 12)
                        {
                            p = convert_float4(vload4(0, ptr + 8));
#ifdef OP_MOMENTS_BINARY
                            p = min(p, 1);
#endif
                            S += (float4)(p.s0, p.s0 * 8, p.s0 * 64, p.s0 * 512) + (float4)(p.s1, p.s1 * 9, p.s1 * 81, p.s1 * 729) +
                                (float4)(p.s2, p.s2 * 10, p.s2 * 100, p.s2 * 1000) + (float4)(p.s3, p.s3 * 11, p.s3 * 121, p.s3 * 1331);
                            //SUM_ELEM(p.s0, 8) + SUM_ELEM(p.s1, 9) + SUM_ELEM(p.s2, 10) + SUM_ELEM(p.s3, 11);

                            if (x_max >= 16)
                            {
                                p = convert_float4(vload4(0, ptr + 12));
#ifdef OP_MOMENTS_BINARY
                                p = min(p, 1);
#endif
                                S += (float4)(p.s0, p.s0 * 12, p.s0 * 144, p.s0 * 1728) + (float4)(p.s1, p.s1 * 13, p.s1 * 169, p.s1 * 2197) +
                                    (float4)(p.s2, p.s2 * 14, p.s2 * 196, p.s2 * 2744) + (float4)(p.s3, p.s3 * 15, p.s3 * 225, p.s3 * 3375);
                                //SUM_ELEM(p.s0, 12) + SUM_ELEM(p.s1, 13) + SUM_ELEM(p.s2, 14) + SUM_ELEM(p.s3, 15);
                            }
                        }
                    }
                }

                if (x_max >= 20)
                {
                    p = convert_float4(vload4(0, ptr + 16));
#ifdef OP_MOMENTS_BINARY
                    p = min(p, 1);
#endif
                    S += (float4)(p.s0, p.s0 * 16, p.s0 * 256, p.s0 * 4096) + (float4)(p.s1, p.s1 * 17, p.s1 * 289, p.s1 * 4913) +
                        (float4)(p.s2, p.s2 * 18, p.s2 * 324, p.s2 * 5832) + (float4)(p.s3, p.s3 * 19, p.s3 * 361, p.s3 * 6859);
                    //SUM_ELEM(p.s0, 16) + SUM_ELEM(p.s1, 17) + SUM_ELEM(p.s2, 18) + SUM_ELEM(p.s3, 19);

                    if (x_max >= 24)
                    {
                        p = convert_float4(vload4(0, ptr + 20));
#ifdef OP_MOMENTS_BINARY
                        p = min(p, 1);
#endif
                        S += (float4)(p.s0, p.s0 * 20, p.s0 * 400, p.s0 * 8000) + (float4)(p.s1, p.s1 * 21, p.s1 * 441, p.s1 * 9261) +
                            (float4)(p.s2, p.s2 * 22, p.s2 * 484, p.s2 * 10648) + (float4)(p.s3, p.s3 * 23, p.s3 * 529, p.s3 * 12167);
                        //SUM_ELEM(p.s0, 20) + SUM_ELEM(p.s1, 21) + SUM_ELEM(p.s2, 22) + SUM_ELEM(p.s3, 23);

                        if (x_max >= 28)
                        {
                            p = convert_float4(vload4(0, ptr + 24));
#ifdef OP_MOMENTS_BINARY
                            p = min(p, 1);
#endif
                            S += (float4)(p.s0, p.s0 * 24, p.s0 * 576, p.s0 * 13824) + (float4)(p.s1, p.s1 * 25, p.s1 * 625, p.s1 * 15625) +
                                (float4)(p.s2, p.s2 * 26, p.s2 * 676, p.s2 * 17576) + (float4)(p.s3, p.s3 * 27, p.s3 * 729, p.s3 * 19683);
                            //SUM_ELEM(p.s0, 24) + SUM_ELEM(p.s1, 25) + SUM_ELEM(p.s2, 26) + SUM_ELEM(p.s3, 27);

                            if (x_max >= 32)
                            {
                                p = convert_float4(vload4(0, ptr + 28));
#ifdef OP_MOMENTS_BINARY
                                p = min(p, 1);
#endif
                                S += (float4)(p.s0, p.s0 * 28, p.s0 * 784, p.s0 * 21952) + (float4)(p.s1, p.s1 * 29, p.s1 * 841, p.s1 * 24389) +
                                    (float4)(p.s2, p.s2 * 30, p.s2 * 900, p.s2 * 27000) + (float4)(p.s3, p.s3 * 31, p.s3 * 961, p.s3 * 29791);
                                //SUM_ELEM(p.s0, 28) + SUM_ELEM(p.s1, 29) + SUM_ELEM(p.s2, 30) + SUM_ELEM(p.s3, 31);
                            }
                        }
                    }
//                }
            }
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
