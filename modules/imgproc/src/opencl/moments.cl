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

    if( x_min < src_cols && y0*TILE_SIZE < src_rows )
    {
        if( ypix < src_rows )
        {
            int x_max = min(src_cols - x_min, TILE_SIZE);
            __global const uchar* ptr = src + src_offset + ypix*src_step + x_min;
            int4 S = (int4)(0,0,0,0), p;

            #define SUM_ELEM(elem, ofs) \
                (int4)(1, (ofs), (ofs)*(ofs), (ofs)*(ofs)*(ofs))*elem

            x = x_max & -4;
            if( x_max >= 4 )
            {
                p = convert_int4(vload4(0, ptr));
                S += SUM_ELEM(p.s0, 0) + SUM_ELEM(p.s1, 1) + SUM_ELEM(p.s2, 2) + SUM_ELEM(p.s3, 3);

                if( x_max >= 8 )
                {
                    p = convert_int4(vload4(0, ptr+4));
                    S += SUM_ELEM(p.s0, 4) + SUM_ELEM(p.s1, 5) + SUM_ELEM(p.s2, 6) + SUM_ELEM(p.s3, 7);

                    if( x_max >= 12 )
                    {
                        p = convert_int4(vload4(0, ptr+8));
                        S += SUM_ELEM(p.s0, 8) + SUM_ELEM(p.s1, 9) + SUM_ELEM(p.s2, 10) + SUM_ELEM(p.s3, 11);

                        if( x_max >= 16 )
                        {
                            p = convert_int4(vload4(0, ptr+12));
                            S += SUM_ELEM(p.s0, 12) + SUM_ELEM(p.s1, 13) + SUM_ELEM(p.s2, 14) + SUM_ELEM(p.s3, 15);
                        }
                    }
                }
            }

            if( x_max >= 20 )
            {
                p = convert_int4(vload4(0, ptr+16));
                S += SUM_ELEM(p.s0, 16) + SUM_ELEM(p.s1, 17) + SUM_ELEM(p.s2, 18) + SUM_ELEM(p.s3, 19);

                if( x_max >= 24 )
                {
                    p = convert_int4(vload4(0, ptr+20));
                    S += SUM_ELEM(p.s0, 20) + SUM_ELEM(p.s1, 21) + SUM_ELEM(p.s2, 22) + SUM_ELEM(p.s3, 23);

                    if( x_max >= 28 )
                    {
                        p = convert_int4(vload4(0, ptr+24));
                        S += SUM_ELEM(p.s0, 24) + SUM_ELEM(p.s1, 25) + SUM_ELEM(p.s2, 26) + SUM_ELEM(p.s3, 27);

                        if( x_max >= 32 )
                        {
                            p = convert_int4(vload4(0, ptr+28));
                            S += SUM_ELEM(p.s0, 28) + SUM_ELEM(p.s1, 29) + SUM_ELEM(p.s2, 30) + SUM_ELEM(p.s3, 31);
                        }
                    }
                }
            }

            if( x < x_max )
            {
                int ps = ptr[x];
                S += SUM_ELEM(ps, x);
                if( x+1 < x_max )
                {
                    ps = ptr[x+1];
                    S += SUM_ELEM(ps, x+1);
                    if( x+2 < x_max )
                    {
                        ps = ptr[x+2];
                        S += SUM_ELEM(ps, x+2);
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
        if( y < d ) \
        { \
            mom[y][0] += mom[y+d][0]; \
            mom[y][1] += mom[y+d][1]; \
            mom[y][2] += mom[y+d][2]; \
            mom[y][3] += mom[y+d][3]; \
            mom[y][4] += mom[y+d][4]; \
            mom[y][5] += mom[y+d][5]; \
            mom[y][6] += mom[y+d][6]; \
            mom[y][7] += mom[y+d][7]; \
            mom[y][8] += mom[y+d][8]; \
            mom[y][9] += mom[y+d][9]; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE)

        REDUCE(16);
        REDUCE(8);
        REDUCE(4);
        REDUCE(2);

        if( y == 0 )
        {
            __global int* momout = mom0 + (y0*xtiles + x0)*10;
            momout[0] = mom[0][0] + mom[1][0];
            momout[1] = mom[0][1] + mom[1][1];
            momout[2] = mom[0][2] + mom[1][2];
            momout[3] = mom[0][3] + mom[1][3];
            momout[4] = mom[0][4] + mom[1][4];
            momout[5] = mom[0][5] + mom[1][5];
            momout[6] = mom[0][6] + mom[1][6];
            momout[7] = mom[0][7] + mom[1][7];
            momout[8] = mom[0][8] + mom[1][8];
            momout[9] = mom[0][9] + mom[1][9];
        }
    }
}
