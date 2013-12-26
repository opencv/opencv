/* See LICENSE file in the root OpenCV directory */

#if TILE_SIZE > 16
#error "TILE SIZE should be <= 16"
#endif

__kernel void moments(__global const uchar* src, int src_step, int src_offset,
                      int src_rows, int src_cols, __global int* mom0, int xtiles)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int x_min = x*TILE_SIZE;
    int y_min = y*TILE_SIZE;

    if( x_min < src_cols && y_min < src_rows )
    {
        int x_max = min(src_cols - x_min, TILE_SIZE);
        int y_max = min(src_rows - y_min, TILE_SIZE);
        int m00=0, m10=0, m01=0, m20=0, m11=0, m02=0, m30=0, m21=0, m12=0, m03=0;
        __global const uchar* ptr = src + src_offset + y_min*src_step + x_min;
        __global int* mom = mom0 + (xtiles*y + x)*10;
        x = x_max & -4;

        for( y = 0; y < y_max; y++, ptr += src_step )
        {
            int4 S = (int4)(0,0,0,0), p;

            #define SUM_ELEM(elem, ofs) \
                (int4)(1, (ofs), ((ofs)*(ofs)), ((ofs)*(ofs)*(ofs)))*elem
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
            m00 += S.s0;
            m10 += S.s1;
            m01 += y*S.s0;
            m20 += S.s2;
            m11 += y*S.s1;
            m02 += sy*S.s0;
            m30 += S.s3;
            m21 += y*S.s2;
            m12 += sy*S.s1;
            m03 += y*sy*S.s0;
        }

        mom[0] = m00;
        mom[1] = m10;
        mom[2] = m01;
        mom[3] = m20;
        mom[4] = m11;
        mom[5] = m02;
        mom[6] = m30;
        mom[7] = m21;
        mom[8] = m12;
        mom[9] = m03;
    }
}
