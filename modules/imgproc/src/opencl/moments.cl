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
    __local float mom[TILE_SIZE][10];

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
            float4 p;
            if (x_max >= 4)
            {
                //__global const uchar* ptr = src + src_offset + ypix*src_step + x_min;
                p = convert_float4(vload4(0, ptr));
#ifdef OP_MOMENTS_BINARY
                p = min(p, 1);
#endif
                S += (float4)(p.s0, 0.0, 0.0, 0.0) + (float4)(p.s1, p.s1, p.s1, p.s1) +
                    (float4)(p.s2, p.s2 * 2.0, p.s2 * 4.0, p.s2 * 8.0) + (float4)(p.s3, p.s3 * 3.0, p.s3 * 9.0, p.s3 * 27.0);
                //SUM_ELEM(p.s0, 0) + SUM_ELEM(p.s1, 1) + SUM_ELEM(p.s2, 2) + SUM_ELEM(p.s3, 3);

                if (x_max >= 8)
                {
                    p = convert_float4(vload4(0, ptr + 4));
#ifdef OP_MOMENTS_BINARY
                    p = min(p, 1);
#endif
                    S += (float4)(p.s0, p.s0 * 4.0, p.s0 * 16.0, p.s0 * 64.0) + (float4)(p.s1, p.s1 * 5.0, p.s1 * 25.0, p.s1 * 125.0) +
                        (float4)(p.s2, p.s2 * 6.0, p.s2 * 36.0, p.s2 * 216.0) + (float4)(p.s3, p.s3 * 7.0, p.s3 * 49.0, p.s3 * 343.0);
                    //SUM_ELEM(p.s0, 4) + SUM_ELEM(p.s1, 5) + SUM_ELEM(p.s2, 6) + SUM_ELEM(p.s3, 7);

                    if (x_max >= 12)
                    {
                        p = convert_float4(vload4(0, ptr + 8));
#ifdef OP_MOMENTS_BINARY
                        p = min(p, 1);
#endif
                        S += (float4)(p.s0, p.s0 * 8.0, p.s0 * 64.0, p.s0 * 512.0) + (float4)(p.s1, p.s1 * 9.0, p.s1 * 81.0, p.s1 * 729.0) +
                            (float4)(p.s2, p.s2 * 10.0, p.s2 * 100.0, p.s2 * 1000.0) + (float4)(p.s3, p.s3 * 11.0, p.s3 * 121.0, p.s3 * 1331.0);
                        //SUM_ELEM(p.s0, 8) + SUM_ELEM(p.s1, 9) + SUM_ELEM(p.s2, 10) + SUM_ELEM(p.s3, 11);

                        if (x_max >= 16)
                        {
                            p = convert_float4(vload4(0, ptr + 12));
#ifdef OP_MOMENTS_BINARY
                            p = min(p, 1);
#endif
                            S += (float4)(p.s0, p.s0 * 12.0, p.s0 * 144.0, p.s0 * 1728.0) + (float4)(p.s1, p.s1 * 13.0, p.s1 * 169.0, p.s1 * 2197.0) +
                                (float4)(p.s2, p.s2 * 14.0, p.s2 * 196.0, p.s2 * 2744.0) + (float4)(p.s3, p.s3 * 15.0, p.s3 * 225.0, p.s3 * 3375.0);
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
                S += (float4)(p.s0, p.s0 * 16.0, p.s0 * 256.0, p.s0 * 4096.0) + (float4)(p.s1, p.s1 * 17.0, p.s1 * 289.0, p.s1 * 4913.0) +
                    (float4)(p.s2, p.s2 * 18.0, p.s2 * 324.0, p.s2 * 5832.0) + (float4)(p.s3, p.s3 * 19.0, p.s3 * 361.0, p.s3 * 6859.0);
                //SUM_ELEM(p.s0, 16) + SUM_ELEM(p.s1, 17) + SUM_ELEM(p.s2, 18) + SUM_ELEM(p.s3, 19);

                if (x_max >= 24)
                {
                    p = convert_float4(vload4(0, ptr + 20));
#ifdef OP_MOMENTS_BINARY
                    p = min(p, 1);
#endif
                    S += (float4)(p.s0, p.s0 * 20.0, p.s0 * 400.0, p.s0 * 8000.0) + (float4)(p.s1, p.s1 * 21.0, p.s1 * 441.0, p.s1 * 9261.0) +
                        (float4)(p.s2, p.s2 * 22.0, p.s2 * 484.0, p.s2 * 10648.0) + (float4)(p.s3, p.s3 * 23.0, p.s3 * 529.0, p.s3 * 12167.0);
                    //SUM_ELEM(p.s0, 20) + SUM_ELEM(p.s1, 21) + SUM_ELEM(p.s2, 22) + SUM_ELEM(p.s3, 23);

                    if (x_max >= 28)
                    {
                        p = convert_float4(vload4(0, ptr + 24));
#ifdef OP_MOMENTS_BINARY
                        p = min(p, 1);
#endif
                        S += (float4)(p.s0, p.s0 * 24.0, p.s0 * 576.0, p.s0 * 13824.0) + (float4)(p.s1, p.s1 * 25.0, p.s1 * 625.0, p.s1 * 15625.0) +
                            (float4)(p.s2, p.s2 * 26.0, p.s2 * 676.0, p.s2 * 17576.0) + (float4)(p.s3, p.s3 * 27.0, p.s3 * 729.0, p.s3 * 19683.0);
                        //SUM_ELEM(p.s0, 24) + SUM_ELEM(p.s1, 25) + SUM_ELEM(p.s2, 26) + SUM_ELEM(p.s3, 27);

                        if (x_max >= 32)
                        {
                            p = convert_float4(vload4(0, ptr + 28));
#ifdef OP_MOMENTS_BINARY
                            p = min(p, 1);
#endif
                            S += (float4)(p.s0, p.s0 * 28.0, p.s0 * 784.0, p.s0 * 21952.0) + (float4)(p.s1, p.s1 * 29.0, p.s1 * 841.0, p.s1 * 24389.0) +
                                (float4)(p.s2, p.s2 * 30.0, p.s2 * 900.0, p.s2 * 27000.0) + (float4)(p.s3, p.s3 * 31.0, p.s3 * 961.0, p.s3 * 29791.0);
                            //SUM_ELEM(p.s0, 28) + SUM_ELEM(p.s1, 29) + SUM_ELEM(p.s2, 30) + SUM_ELEM(p.s3, 31);
                        }
                    }
                }
            }
            if (x < x_max)
            {
                int ps = (ptr[x]);
#ifdef OP_MOMENTS_BINARY
                ps = min((ps), 1);
#endif
                S += SUM_ELEM(ps,(float)(x));
                if (x + 1 < x_max)
                {
                    ps = (ptr[x + 1]);
#ifdef OP_MOMENTS_BINARY
                    ps = min((ps), 1);
#endif
                    S += SUM_ELEM(ps, (float)(x + 1));
                    if (x + 2 < x_max)
                    {
                        ps = (ptr[x + 2]);
#ifdef OP_MOMENTS_BINARY
                        ps = min((ps), 1);
#endif
                        S += SUM_ELEM(ps, (float)(x + 2));
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
            mom[y][5] = mom[y][6] = mom[y][7] = mom[y][8] = mom[y][9] = 0.0;
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
