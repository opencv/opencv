/* See LICENSE file in the root OpenCV directory */

#ifdef BINARY_MOMENTS
#define READ_PIX(ref) (ref != 0)
#else
#define READ_PIX(ref) ref
#endif

__kernel void moments(__global const uchar* src, int src_step, int src_offset,
                      int src_rows, int src_cols, __global int* mom0,
                      int tile_size, int xtiles, int ytiles)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int x_min = x*tile_size;
    int y_min = y*tile_size;

    if( x_min < src_cols && y_min < src_rows )
    {
        int x_max = src_cols - x_min;
        int y_max = src_rows - y_min;
        int m[10]={0,0,0,0,0,0,0,0,0,0};
        __global const uchar* ptr = (src + src_offset);// + y_min*src_step + x_min;
        __global int* mom = mom0 + (xtiles*y + x)*10;
        
        x_max = x_max < tile_size ? x_max : tile_size;
        y_max = y_max < tile_size ? y_max : tile_size;

        for( y = 0; y < y_max; y++ )
        {
            int x00, x10, x20, x30;
            int sx, sy, p;
            x00 = x10 = x20 = x30 = 0;
            sy = y*y;

            for( x = 0; x < x_max; x++ )
            {
                p = ptr[0];//READ_PIX(ptr[x]);
                sx = x*x;
                x00 += p;
                x10 += x*p;
                x20 += sx*p;
                x30 += x*sx*p;
            }

            m[0] += x00;
            m[1] += x10;
            m[2] += y*x00;
            m[3] += x20;
            m[4] += y*x10;
            m[5] += sy*x00;
            m[6] += x30;
            m[7] += y*x20;
            m[8] += sy*x10;
            m[9] += y*sy*x00;
            //ptr += src_step;
        }

        mom[0] = m[0];

        mom[1] = m[1];
        mom[2] = m[2];

        mom[3] = m[3];
        mom[4] = m[4];
        mom[5] = m[5];

        mom[6] = m[6];
        mom[7] = m[7];
        mom[8] = m[8];
        mom[9] = m[9];
    }
}

/*__kernel void moments(__global const uchar* src, int src_step, int src_offset,
                     int src_rows, int src_cols, __global float* mom0,
                     int tile_size, int xtiles, int ytiles)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if( x < xtiles && y < ytiles )
    {
        //int x_min = x*tile_size;
        //int y_min = y*tile_size;
        //int x_max = src_cols - x_min;
        //int y_max = src_rows - y_min;
        __global const uchar* ptr = src + src_offset;// + src_step*y_min + x_min;
        __global float* mom = mom0;// + (y*xtiles + x)*16;
        //int x00, x10, x20, x30, m00=0;
        //x_max = min(x_max, tile_size);
        //y_max = min(y_max, tile_size);
        //int m00 = 0;
        
        //for( y = 0; y < y_max; y++, ptr += src_step )
        //{
            //int x00 = 0, x10 = 0, x20 = 0, x30 = 0;
            //for( x = 0; x < x_max; x++ )
            //{
                int p = ptr[x];
                //m00 = p;
                //x10 += x*p;
                /*x20 += x*x*p;
                x30 += x*x*x*p;
            //}
            //m00 = m00 + x00;
        //}
        mom[0] = p;
    }
}*/

