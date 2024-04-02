#include <climits>

#include "rvv_hal.hpp"

void cvt_vector(const uchar* src, uchar * dst, uchar * index, int n, int scn, int dcn, int bi, int vsize_pixels, int vsize)
{
    vuint8m2_t vec_index = vle8_v_u8m2(index, vsize);

    int i = 0;

    for( ; i <= n-vsize; i += vsize_pixels, src += vsize, dst += vsize)
    {
        vuint8m2_t vec_src = vle8_v_u8m2(src, vsize);
        vuint8m2_t vec_dst = vrgather_vv_u8m2(vec_src, vec_index, vsize);
        vse8_v_u8m2(dst, vec_dst, vsize);
    }

    for ( ; i < n; i++, src += scn, dst += dcn )
    {
        uchar t0 = src[0], t1 = src[1], t2 = src[2];
        dst[bi  ] = t0;
        dst[1]    = t1;
        dst[bi^2] = t2;
        if(dcn == 4)
        {
            uchar d = scn == 4 ? src[3] : UCHAR_MAX;
            dst[3] = d;
        }
    }
}


void cvt_scalar(const uchar* src, uchar * dst, int n, int scn, int dcn, int bi)
{
    for (int i = 0; i < n; i++, src += scn, dst += dcn)
    {
        uchar t0 = src[0], t1 = src[1], t2 = src[2];
        dst[bi  ] = t0;
        dst[1]    = t1;
        dst[bi^2] = t2;
        if(dcn == 4)
        {
            uchar d = scn == 4 ? src[3] : UCHAR_MAX;
            dst[3] = d;
        }
    }
}

int cvt_hal_BGRtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int scn, int dcn, bool swapBlue)
{
    int blueIdx = swapBlue ? 2 : 0;
    if (scn == dcn)
    {   
        const int vsize_pixels = 8;
        const int vsize = vsize_pixels*scn;
    
        unsigned char* index;
        if (scn == 4)
        {
            index = new unsigned char [vsize] 
                        { 2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15, 18, 17, 16, 19, 22, 21, 20, 23, 26, 25, 24, 27, 30, 29, 28, 31  };
        }
        else
        {
            index = new unsigned char [vsize]
                        { 2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 14, 13, 12, 17, 16, 15, 20, 19, 18, 23, 22, 21  };
        }

        size_t vl = vsetvl_e8m2(vsize);

        for(int i = 0; i < height; i++, src_data += src_step, dst_data += dst_step)
            cvt_vector(src_data, dst_data, index, width, scn, dcn, blueIdx, vsize_pixels, vsize);
    }
    else
    {
        //return CV_HAL_ERROR_NOT_IMPLEMENTED;
        for(int i = 0; i < height; i++, src_data += src_step, dst_data += dst_step)
        {
            cvt_scalar(src_data, dst_data, width, scn, dcn, blueIdx);
        }
    }

    return CV_HAL_ERROR_OK;
};