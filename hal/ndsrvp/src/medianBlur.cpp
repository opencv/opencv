// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "ndsrvp_hal.hpp"
#include "opencv2/imgproc/hal/interface.h"
#include "cvutils.hpp"

namespace cv {

namespace ndsrvp {

struct operators_minmax_t {
    inline void vector(uint8x8_t & a, uint8x8_t & b) const {
        uint8x8_t t = a;
        a = __nds__v_umin8(a, b);
        b = __nds__v_umax8(t, b);
    }
    inline void scalar(uchar & a, uchar & b) const {
        uchar t = a;
        a = __nds__umin8(a, b);
        b = __nds__umax8(t, b);
    }
    inline void vector(int8x8_t & a, int8x8_t & b) const {
        int8x8_t t = a;
        a = __nds__v_smin8(a, b);
        b = __nds__v_smax8(t, b);
    }
    inline void scalar(schar & a, schar & b) const {
        schar t = a;
        a = __nds__smin8(a, b);
        b = __nds__smax8(t, b);
    }
    inline void vector(uint16x4_t & a, uint16x4_t & b) const {
        uint16x4_t t = a;
        a = __nds__v_umin16(a, b);
        b = __nds__v_umax16(t, b);
    }
    inline void scalar(ushort & a, ushort & b) const {
        ushort t = a;
        a = __nds__umin16(a, b);
        b = __nds__umax16(t, b);
    }
    inline void vector(int16x4_t & a, int16x4_t & b) const {
        int16x4_t t = a;
        a = __nds__v_smin16(a, b);
        b = __nds__v_smax16(t, b);
    }
    inline void scalar(short & a, short & b) const {
        short t = a;
        a = __nds__smin16(a, b);
        b = __nds__smax16(t, b);
    }
};

template<typename T, typename WT, typename VT> // type, widen type, vector type
static void
medianBlur_SortNet( const uchar* src_data, size_t src_step,
                    uchar* dst_data, size_t dst_step,
                    int width, int height, int cn, int ksize )
{
    const T* src = (T*)src_data;
    T* dst = (T*)dst_data;
    int sstep = (int)(src_step / sizeof(T));
    int dstep = (int)(dst_step / sizeof(T));
    int i, j, k;
    operators_minmax_t op;

    if( ksize == 3 )
    {
        if( width == 1 || height == 1 )
        {
            int len = width + height - 1;
            int sdelta = height == 1 ? cn : sstep;
            int sdelta0 = height == 1 ? 0 : sstep - cn;
            int ddelta = height == 1 ? cn : dstep;

            for( i = 0; i < len; i++, src += sdelta0, dst += ddelta )
                for( j = 0; j < cn; j++, src++ )
                {
                    T p0 = src[i > 0 ? -sdelta : 0];
                    T p1 = src[0];
                    T p2 = src[i < len - 1 ? sdelta : 0];

                    op.scalar(p0, p1); op.scalar(p1, p2); op.scalar(p0, p1);
                    dst[j] = (T)p1;
                }
            return;
        }

        width *= cn;
        for( i = 0; i < height; i++, dst += dstep )
        {
            const T* row0 = src + std::max(i - 1, 0)*sstep;
            const T* row1 = src + i*sstep;
            const T* row2 = src + std::min(i + 1, height-1)*sstep;
            int limit = cn;

            for(j = 0;; )
            {
                for( ; j < limit; j++ )
                {
                    int j0 = j >= cn ? j - cn : j;
                    int j2 = j < width - cn ? j + cn : j;
                    T p0 = row0[j0], p1 = row0[j], p2 = row0[j2];
                    T p3 = row1[j0], p4 = row1[j], p5 = row1[j2];
                    T p6 = row2[j0], p7 = row2[j], p8 = row2[j2];

                    op.scalar(p1, p2); op.scalar(p4, p5); op.scalar(p7, p8); op.scalar(p0, p1);
                    op.scalar(p3, p4); op.scalar(p6, p7); op.scalar(p1, p2); op.scalar(p4, p5);
                    op.scalar(p7, p8); op.scalar(p0, p3); op.scalar(p5, p8); op.scalar(p4, p7);
                    op.scalar(p3, p6); op.scalar(p1, p4); op.scalar(p2, p5); op.scalar(p4, p7);
                    op.scalar(p4, p2); op.scalar(p6, p4); op.scalar(p4, p2);
                    dst[j] = (T)p4;
                }

                if( limit == width )
                    break;

                int nlanes = 8 / sizeof(T);

                for( ; (cn % nlanes == 0) && (j <= width - nlanes - cn); j += nlanes ) // alignment
                {
                    VT p0 = *(VT*)(row0+j-cn), p1 = *(VT*)(row0+j), p2 = *(VT*)(row0+j+cn);
                    VT p3 = *(VT*)(row1+j-cn), p4 = *(VT*)(row1+j), p5 = *(VT*)(row1+j+cn);
                    VT p6 = *(VT*)(row2+j-cn), p7 = *(VT*)(row2+j), p8 = *(VT*)(row2+j+cn);

                    op.vector(p1, p2); op.vector(p4, p5); op.vector(p7, p8); op.vector(p0, p1);
                    op.vector(p3, p4); op.vector(p6, p7); op.vector(p1, p2); op.vector(p4, p5);
                    op.vector(p7, p8); op.vector(p0, p3); op.vector(p5, p8); op.vector(p4, p7);
                    op.vector(p3, p6); op.vector(p1, p4); op.vector(p2, p5); op.vector(p4, p7);
                    op.vector(p4, p2); op.vector(p6, p4); op.vector(p4, p2);
                    *(VT*)(dst+j) = p4;
                }

                limit = width;
            }
        }
    }
    else if( ksize == 5 )
    {
        if( width == 1 || height == 1 )
        {
            int len = width + height - 1;
            int sdelta = height == 1 ? cn : sstep;
            int sdelta0 = height == 1 ? 0 : sstep - cn;
            int ddelta = height == 1 ? cn : dstep;

            for( i = 0; i < len; i++, src += sdelta0, dst += ddelta )
                for( j = 0; j < cn; j++, src++ )
                {
                    int i1 = i > 0 ? -sdelta : 0;
                    int i0 = i > 1 ? -sdelta*2 : i1;
                    int i3 = i < len-1 ? sdelta : 0;
                    int i4 = i < len-2 ? sdelta*2 : i3;
                    T p0 = src[i0], p1 = src[i1], p2 = src[0], p3 = src[i3], p4 = src[i4];

                    op.scalar(p0, p1); op.scalar(p3, p4); op.scalar(p2, p3); op.scalar(p3, p4); op.scalar(p0, p2);
                    op.scalar(p2, p4); op.scalar(p1, p3); op.scalar(p1, p2);
                    dst[j] = (T)p2;
                }
            return;
        }

        width *= cn;
        for( i = 0; i < height; i++, dst += dstep )
        {
            const T* row[5];
            row[0] = src + std::max(i - 2, 0)*sstep;
            row[1] = src + std::max(i - 1, 0)*sstep;
            row[2] = src + i*sstep;
            row[3] = src + std::min(i + 1, height-1)*sstep;
            row[4] = src + std::min(i + 2, height-1)*sstep;
            int limit = cn*2;

            for(j = 0;; )
            {
                for( ; j < limit; j++ )
                {
                    T p[25];
                    int j1 = j >= cn ? j - cn : j;
                    int j0 = j >= cn*2 ? j - cn*2 : j1;
                    int j3 = j < width - cn ? j + cn : j;
                    int j4 = j < width - cn*2 ? j + cn*2 : j3;
                    for( k = 0; k < 5; k++ )
                    {
                        const T* rowk = row[k];
                        p[k*5] = rowk[j0]; p[k*5+1] = rowk[j1];
                        p[k*5+2] = rowk[j]; p[k*5+3] = rowk[j3];
                        p[k*5+4] = rowk[j4];
                    }

                    op.scalar(p[1], p[2]); op.scalar(p[0], p[1]); op.scalar(p[1], p[2]); op.scalar(p[4], p[5]); op.scalar(p[3], p[4]);
                    op.scalar(p[4], p[5]); op.scalar(p[0], p[3]); op.scalar(p[2], p[5]); op.scalar(p[2], p[3]); op.scalar(p[1], p[4]);
                    op.scalar(p[1], p[2]); op.scalar(p[3], p[4]); op.scalar(p[7], p[8]); op.scalar(p[6], p[7]); op.scalar(p[7], p[8]);
                    op.scalar(p[10], p[11]); op.scalar(p[9], p[10]); op.scalar(p[10], p[11]); op.scalar(p[6], p[9]); op.scalar(p[8], p[11]);
                    op.scalar(p[8], p[9]); op.scalar(p[7], p[10]); op.scalar(p[7], p[8]); op.scalar(p[9], p[10]); op.scalar(p[0], p[6]);
                    op.scalar(p[4], p[10]); op.scalar(p[4], p[6]); op.scalar(p[2], p[8]); op.scalar(p[2], p[4]); op.scalar(p[6], p[8]);
                    op.scalar(p[1], p[7]); op.scalar(p[5], p[11]); op.scalar(p[5], p[7]); op.scalar(p[3], p[9]); op.scalar(p[3], p[5]);
                    op.scalar(p[7], p[9]); op.scalar(p[1], p[2]); op.scalar(p[3], p[4]); op.scalar(p[5], p[6]); op.scalar(p[7], p[8]);
                    op.scalar(p[9], p[10]); op.scalar(p[13], p[14]); op.scalar(p[12], p[13]); op.scalar(p[13], p[14]); op.scalar(p[16], p[17]);
                    op.scalar(p[15], p[16]); op.scalar(p[16], p[17]); op.scalar(p[12], p[15]); op.scalar(p[14], p[17]); op.scalar(p[14], p[15]);
                    op.scalar(p[13], p[16]); op.scalar(p[13], p[14]); op.scalar(p[15], p[16]); op.scalar(p[19], p[20]); op.scalar(p[18], p[19]);
                    op.scalar(p[19], p[20]); op.scalar(p[21], p[22]); op.scalar(p[23], p[24]); op.scalar(p[21], p[23]); op.scalar(p[22], p[24]);
                    op.scalar(p[22], p[23]); op.scalar(p[18], p[21]); op.scalar(p[20], p[23]); op.scalar(p[20], p[21]); op.scalar(p[19], p[22]);
                    op.scalar(p[22], p[24]); op.scalar(p[19], p[20]); op.scalar(p[21], p[22]); op.scalar(p[23], p[24]); op.scalar(p[12], p[18]);
                    op.scalar(p[16], p[22]); op.scalar(p[16], p[18]); op.scalar(p[14], p[20]); op.scalar(p[20], p[24]); op.scalar(p[14], p[16]);
                    op.scalar(p[18], p[20]); op.scalar(p[22], p[24]); op.scalar(p[13], p[19]); op.scalar(p[17], p[23]); op.scalar(p[17], p[19]);
                    op.scalar(p[15], p[21]); op.scalar(p[15], p[17]); op.scalar(p[19], p[21]); op.scalar(p[13], p[14]); op.scalar(p[15], p[16]);
                    op.scalar(p[17], p[18]); op.scalar(p[19], p[20]); op.scalar(p[21], p[22]); op.scalar(p[23], p[24]); op.scalar(p[0], p[12]);
                    op.scalar(p[8], p[20]); op.scalar(p[8], p[12]); op.scalar(p[4], p[16]); op.scalar(p[16], p[24]); op.scalar(p[12], p[16]);
                    op.scalar(p[2], p[14]); op.scalar(p[10], p[22]); op.scalar(p[10], p[14]); op.scalar(p[6], p[18]); op.scalar(p[6], p[10]);
                    op.scalar(p[10], p[12]); op.scalar(p[1], p[13]); op.scalar(p[9], p[21]); op.scalar(p[9], p[13]); op.scalar(p[5], p[17]);
                    op.scalar(p[13], p[17]); op.scalar(p[3], p[15]); op.scalar(p[11], p[23]); op.scalar(p[11], p[15]); op.scalar(p[7], p[19]);
                    op.scalar(p[7], p[11]); op.scalar(p[11], p[13]); op.scalar(p[11], p[12]);
                    dst[j] = (T)p[12];
                }

                if( limit == width )
                    break;

                int nlanes = 8 / sizeof(T);

                for( ; (cn % nlanes == 0) && (j <= width - nlanes - cn*2); j += nlanes )
                {
                    VT p0 = *(VT*)(row[0]+j-cn*2), p5 = *(VT*)(row[1]+j-cn*2), p10 = *(VT*)(row[2]+j-cn*2), p15 = *(VT*)(row[3]+j-cn*2), p20 = *(VT*)(row[4]+j-cn*2);
                    VT p1 = *(VT*)(row[0]+j-cn*1), p6 = *(VT*)(row[1]+j-cn*1), p11 = *(VT*)(row[2]+j-cn*1), p16 = *(VT*)(row[3]+j-cn*1), p21 = *(VT*)(row[4]+j-cn*1);
                    VT p2 = *(VT*)(row[0]+j-cn*0), p7 = *(VT*)(row[1]+j-cn*0), p12 = *(VT*)(row[2]+j-cn*0), p17 = *(VT*)(row[3]+j-cn*0), p22 = *(VT*)(row[4]+j-cn*0);
                    VT p3 = *(VT*)(row[0]+j+cn*1), p8 = *(VT*)(row[1]+j+cn*1), p13 = *(VT*)(row[2]+j+cn*1), p18 = *(VT*)(row[3]+j+cn*1), p23 = *(VT*)(row[4]+j+cn*1);
                    VT p4 = *(VT*)(row[0]+j+cn*2), p9 = *(VT*)(row[1]+j+cn*2), p14 = *(VT*)(row[2]+j+cn*2), p19 = *(VT*)(row[3]+j+cn*2), p24 = *(VT*)(row[4]+j+cn*2);

                    op.vector(p1, p2); op.vector(p0, p1); op.vector(p1, p2); op.vector(p4, p5); op.vector(p3, p4);
                    op.vector(p4, p5); op.vector(p0, p3); op.vector(p2, p5); op.vector(p2, p3); op.vector(p1, p4);
                    op.vector(p1, p2); op.vector(p3, p4); op.vector(p7, p8); op.vector(p6, p7); op.vector(p7, p8);
                    op.vector(p10, p11); op.vector(p9, p10); op.vector(p10, p11); op.vector(p6, p9); op.vector(p8, p11);
                    op.vector(p8, p9); op.vector(p7, p10); op.vector(p7, p8); op.vector(p9, p10); op.vector(p0, p6);
                    op.vector(p4, p10); op.vector(p4, p6); op.vector(p2, p8); op.vector(p2, p4); op.vector(p6, p8);
                    op.vector(p1, p7); op.vector(p5, p11); op.vector(p5, p7); op.vector(p3, p9); op.vector(p3, p5);
                    op.vector(p7, p9); op.vector(p1, p2); op.vector(p3, p4); op.vector(p5, p6); op.vector(p7, p8);
                    op.vector(p9, p10); op.vector(p13, p14); op.vector(p12, p13); op.vector(p13, p14); op.vector(p16, p17);
                    op.vector(p15, p16); op.vector(p16, p17); op.vector(p12, p15); op.vector(p14, p17); op.vector(p14, p15);
                    op.vector(p13, p16); op.vector(p13, p14); op.vector(p15, p16); op.vector(p19, p20); op.vector(p18, p19);
                    op.vector(p19, p20); op.vector(p21, p22); op.vector(p23, p24); op.vector(p21, p23); op.vector(p22, p24);
                    op.vector(p22, p23); op.vector(p18, p21); op.vector(p20, p23); op.vector(p20, p21); op.vector(p19, p22);
                    op.vector(p22, p24); op.vector(p19, p20); op.vector(p21, p22); op.vector(p23, p24); op.vector(p12, p18);
                    op.vector(p16, p22); op.vector(p16, p18); op.vector(p14, p20); op.vector(p20, p24); op.vector(p14, p16);
                    op.vector(p18, p20); op.vector(p22, p24); op.vector(p13, p19); op.vector(p17, p23); op.vector(p17, p19);
                    op.vector(p15, p21); op.vector(p15, p17); op.vector(p19, p21); op.vector(p13, p14); op.vector(p15, p16);
                    op.vector(p17, p18); op.vector(p19, p20); op.vector(p21, p22); op.vector(p23, p24); op.vector(p0, p12);
                    op.vector(p8, p20); op.vector(p8, p12); op.vector(p4, p16); op.vector(p16, p24); op.vector(p12, p16);
                    op.vector(p2, p14); op.vector(p10, p22); op.vector(p10, p14); op.vector(p6, p18); op.vector(p6, p10);
                    op.vector(p10, p12); op.vector(p1, p13); op.vector(p9, p21); op.vector(p9, p13); op.vector(p5, p17);
                    op.vector(p13, p17); op.vector(p3, p15); op.vector(p11, p23); op.vector(p11, p15); op.vector(p7, p19);
                    op.vector(p7, p11); op.vector(p11, p13); op.vector(p11, p12);
                    *(VT*)(dst+j) = p12;
                }

                limit = width;
            }
        }
    }
}

int medianBlur(const uchar* src_data, size_t src_step,
    uchar* dst_data, size_t dst_step,
    int width, int height, int depth, int cn, int ksize)
{
    bool useSortNet = ((ksize == 3) || (ksize == 5 && ( depth > CV_8U || cn == 2 || cn > 4 )));

    if( useSortNet )
    {
        uchar* src_data_rep;
        if( dst_data == src_data ) {
            std::vector<uchar> src_data_copy(src_step * height);
            memcpy(src_data_copy.data(), src_data, src_step * height);
            src_data_rep = &src_data_copy[0];
        }
        else {
            src_data_rep = (uchar*)src_data;
        }

        if( depth == CV_8U )
            medianBlur_SortNet<uchar, int, uint8x8_t>( src_data_rep, src_step, dst_data, dst_step, width, height, cn, ksize );
        else if( depth == CV_8S )
            medianBlur_SortNet<schar, int, int8x8_t>( src_data_rep, src_step, dst_data, dst_step, width, height, cn, ksize );
        else if( depth == CV_16U )
            medianBlur_SortNet<ushort, int, uint16x4_t>( src_data_rep, src_step, dst_data, dst_step, width, height, cn, ksize );
        else if( depth == CV_16S )
            medianBlur_SortNet<short, int, int16x4_t>( src_data_rep, src_step, dst_data, dst_step, width, height, cn, ksize );
        else
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        return CV_HAL_ERROR_OK;
    }
    else return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

} // namespace ndsrvp

} // namespace cv
