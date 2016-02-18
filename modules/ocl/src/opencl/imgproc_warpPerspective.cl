/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010,2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Zhang Ying, zhangying913@gmail.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/


//wrapPerspective kernel
//support data types: CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4, and three interpolation methods: NN, Linear, Cubic.

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
typedef double F;
typedef double4 F4;
#define convert_F4 convert_double4
#else
typedef float F;
typedef float4 F4;
#define convert_F4 convert_float4
#endif


#define INTER_BITS 5
#define INTER_TAB_SIZE (1 << INTER_BITS)
#define INTER_SCALE 1.f/INTER_TAB_SIZE
#define AB_BITS max(10, (int)INTER_BITS)
#define AB_SCALE (1 << AB_BITS)
#define INTER_REMAP_COEF_BITS 15
#define INTER_REMAP_COEF_SCALE (1 << INTER_REMAP_COEF_BITS)

void interpolateCubic( float x, float* coeffs )
{
    const float A = -0.75f;

    coeffs[0] = ((A*(x + 1.f) - 5.0f*A)*(x + 1.f) + 8.0f*A)*(x + 1.f) - 4.0f*A;
    coeffs[1] = ((A + 2.f)*x - (A + 3.f))*x*x + 1.f;
    coeffs[2] = ((A + 2.f)*(1.f - x) - (A + 3.f))*(1.f - x)*(1.f - x) + 1.f;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}


/**********************************************8UC1*********************************************
***********************************************************************************************/
__kernel void warpPerspectiveNN_C1_D0(__global uchar const * restrict src, __global uchar * dst, int src_cols, int src_rows,
                                      int dst_cols, int dst_rows, int srcStep, int dstStep,
                                      int src_offset, int dst_offset,  __constant F * M, int threadCols )
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if( dx < threadCols && dy < dst_rows)
    {
        dx = (dx<<2) - (dst_offset&3);

        F4 DX = (F4)(dx, dx+1, dx+2, dx+3);
        F4 X0 = M[0]*DX + M[1]*dy + M[2];
        F4 Y0 = M[3]*DX + M[4]*dy + M[5];
        F4 W = M[6]*DX + M[7]*dy + M[8],one=1,zero=0;
        W = (W!=zero) ? one/W : zero;
        short4 X = convert_short4_sat_rte(X0*W);
        short4 Y = convert_short4_sat_rte(Y0*W);
        int4 sx = convert_int4(X);
        int4 sy = convert_int4(Y);

        int4 DXD = (int4)(dx, dx+1, dx+2, dx+3);
        __global uchar4 * d = (__global uchar4 *)(dst+dst_offset+dy*dstStep+dx);
        uchar4 dval = *d;
        int4 dcon = DXD >= 0 && DXD < dst_cols && dy >= 0 && dy < dst_rows;
        int4 scon = sx >= 0 && sx < src_cols && sy >= 0 && sy < src_rows;
        int4 spos = src_offset + sy * srcStep + sx;
        uchar4 sval;
        sval.s0 = scon.s0 ? src[spos.s0] : 0;
        sval.s1 = scon.s1 ? src[spos.s1] : 0;
        sval.s2 = scon.s2 ? src[spos.s2] : 0;
        sval.s3 = scon.s3 ? src[spos.s3] : 0;
        dval = convert_uchar4(dcon) != (uchar4)(0,0,0,0) ? sval : dval;
        *d = dval;
    }
}

__kernel void warpPerspectiveLinear_C1_D0(__global const uchar * restrict src, __global uchar * dst,
        int src_cols, int src_rows, int dst_cols, int dst_rows, int srcStep,
        int dstStep, int src_offset, int dst_offset,  __constant F * M, int threadCols )
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if( dx < threadCols && dy < dst_rows)
    {
        F X0 = M[0]*dx + M[1]*dy + M[2];
        F Y0 = M[3]*dx + M[4]*dy + M[5];
        F W = M[6]*dx + M[7]*dy + M[8];
        W = (W != 0.0f) ? INTER_TAB_SIZE/W : 0.0f;
        int X = rint(X0*W);
        int Y = rint(Y0*W);

        int sx = convert_short_sat(X >> INTER_BITS);
        int sy = convert_short_sat(Y >> INTER_BITS);
        int ay = (short)(Y & (INTER_TAB_SIZE-1));
        int ax = (short)(X & (INTER_TAB_SIZE-1));

        uchar v[4];
        int i;
#pragma unroll 4
        for(i=0; i<4;  i++)
            v[i] = (sx+(i&1) >= 0 && sx+(i&1) < src_cols && sy+(i>>1) >= 0 && sy+(i>>1) < src_rows) ? src[src_offset + (sy+(i>>1)) * srcStep + (sx+(i&1))] : (uchar)0;

        short itab[4];
        float tab1y[2], tab1x[2];
        tab1y[0] = 1.0f - 1.f/INTER_TAB_SIZE*ay;
        tab1y[1] = 1.f/INTER_TAB_SIZE*ay;
        tab1x[0] = 1.0f - 1.f/INTER_TAB_SIZE*ax;
        tab1x[1] = 1.f/INTER_TAB_SIZE*ax;

#pragma unroll 4
        for(i=0; i<4;  i++)
        {
            float v = tab1y[(i>>1)] * tab1x[(i&1)];
            itab[i] = convert_short_sat_rte( v * INTER_REMAP_COEF_SCALE );
        }
        if(dx >=0 && dx < dst_cols && dy >= 0 && dy < dst_rows)
        {
            int sum = 0;
            for ( i =0; i<4; i++ )
            {
                sum += v[i] * itab[i] ;
            }
            dst[dst_offset+dy*dstStep+dx] = convert_uchar_sat ( (sum + (1 << (INTER_REMAP_COEF_BITS-1))) >> INTER_REMAP_COEF_BITS ) ;
        }
    }
}

__kernel void warpPerspectiveCubic_C1_D0(__global uchar * src, __global uchar * dst, int src_cols, int src_rows,
        int dst_cols, int dst_rows, int srcStep, int dstStep,
        int src_offset, int dst_offset,  __constant F * M, int threadCols )
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if( dx < threadCols && dy < dst_rows)
    {
        F X0 = M[0]*dx + M[1]*dy + M[2];
        F Y0 = M[3]*dx + M[4]*dy + M[5];
        F W = M[6]*dx + M[7]*dy + M[8];
        W = (W != 0.0f) ? INTER_TAB_SIZE/W : 0.0f;
        int X = rint(X0*W);
        int Y = rint(Y0*W);

        short sx = convert_short_sat(X >> INTER_BITS) - 1;
        short sy = convert_short_sat(Y >> INTER_BITS) - 1;
        short ay = (short)(Y & (INTER_TAB_SIZE-1));
        short ax = (short)(X & (INTER_TAB_SIZE-1));

        uchar v[16];
        int i, j;

#pragma unroll 4
        for(i=0; i<4;  i++)
            for(j=0; j<4;  j++)
            {
                v[i*4+j] = (sx+j >= 0 && sx+j < src_cols && sy+i >= 0 && sy+i < src_rows) ? src[src_offset+(sy+i) * srcStep + (sx+j)] : (uchar)0;
            }

        short itab[16];
        float tab1y[4], tab1x[4];
        float axx, ayy;

        ayy = 1.f/INTER_TAB_SIZE * ay;
        axx = 1.f/INTER_TAB_SIZE * ax;
        interpolateCubic(ayy, tab1y);
        interpolateCubic(axx, tab1x);

        int isum = 0;
#pragma unroll 16
        for( i=0; i<16; i++ )
        {
            F v = tab1y[(i>>2)] * tab1x[(i&3)];
            isum += itab[i] = convert_short_sat( rint( v * INTER_REMAP_COEF_SCALE ) );
        }
        if( isum != INTER_REMAP_COEF_SCALE )
        {
            int k1, k2;
            int diff = isum - INTER_REMAP_COEF_SCALE;
            int Mk1=2, Mk2=2, mk1=2, mk2=2;
            for( k1 = 2; k1 < 4; k1++ )
                for( k2 = 2; k2 < 4; k2++ )
                {
                    if( itab[(k1<<2)+k2] < itab[(mk1<<2)+mk2] )
                        mk1 = k1, mk2 = k2;
                    else if( itab[(k1<<2)+k2] > itab[(Mk1<<2)+Mk2] )
                        Mk1 = k1, Mk2 = k2;
                }
            diff<0 ? (itab[(Mk1<<2)+Mk2]=(short)(itab[(Mk1<<2)+Mk2]-diff)) : (itab[(mk1<<2)+mk2]=(short)(itab[(mk1<<2)+mk2]-diff));
        }


        if( dx >= 0 && dx < dst_cols && dy >= 0 && dy < dst_rows)
        {
            int sum=0;
            for ( i =0; i<16; i++ )
            {
                sum += v[i] * itab[i] ;
            }
            dst[dst_offset+dy*dstStep+dx] = convert_uchar_sat( (sum + (1 << (INTER_REMAP_COEF_BITS-1))) >> INTER_REMAP_COEF_BITS ) ;
        }
    }
}

/**********************************************8UC4*********************************************
***********************************************************************************************/

__kernel void warpPerspectiveNN_C4_D0(__global uchar4 const * restrict src, __global uchar4 * dst,
                                      int src_cols, int src_rows, int dst_cols, int dst_rows, int srcStep,
                                      int dstStep, int src_offset, int dst_offset,  __constant F * M, int threadCols )
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if( dx < threadCols && dy < dst_rows)
    {

        F X0 = M[0]*dx + M[1]*dy + M[2];
        F Y0 = M[3]*dx + M[4]*dy + M[5];
        F W = M[6]*dx + M[7]*dy + M[8];
        W = (W != 0.0f) ? 1.f/W : 0.0f;
        short sx = convert_short_sat_rte(X0*W);
        short sy = convert_short_sat_rte(Y0*W);

        if(dx >= 0 && dx < dst_cols && dy >= 0 && dy < dst_rows)
            dst[(dst_offset>>2)+dy*(dstStep>>2)+dx]= (sx>=0 && sx<src_cols && sy>=0 && sy<src_rows) ? src[(src_offset>>2)+sy*(srcStep>>2)+sx] : (uchar4)0;
    }
}

__kernel void warpPerspectiveLinear_C4_D0(__global uchar4 const * restrict src, __global uchar4 * dst,
        int src_cols, int src_rows, int dst_cols, int dst_rows, int srcStep,
        int dstStep, int src_offset, int dst_offset,  __constant F * M, int threadCols )
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if( dx < threadCols && dy < dst_rows)
    {
        src_offset = (src_offset>>2);
        srcStep = (srcStep>>2);

        F X0 = M[0]*dx + M[1]*dy + M[2];
        F Y0 = M[3]*dx + M[4]*dy + M[5];
        F W = M[6]*dx + M[7]*dy + M[8];
        W = (W != 0.0f) ? INTER_TAB_SIZE/W : 0.0f;
        int X = rint(X0*W);
        int Y = rint(Y0*W);

        short sx = convert_short_sat(X >> INTER_BITS);
        short sy = convert_short_sat(Y >> INTER_BITS);
        short ay = (short)(Y & (INTER_TAB_SIZE-1));
        short ax = (short)(X & (INTER_TAB_SIZE-1));


        int4 v0, v1, v2, v3;

        v0 = (sx >= 0 && sx < src_cols && sy >= 0 && sy < src_rows) ? convert_int4(src[src_offset+sy * srcStep + sx]) : (int4)0;
        v1 = (sx+1 >= 0 && sx+1 < src_cols && sy >= 0 && sy < src_rows) ? convert_int4(src[src_offset+sy * srcStep + sx+1]) : (int4)0;
        v2 = (sx >= 0 && sx < src_cols && sy+1 >= 0 && sy+1 < src_rows) ? convert_int4(src[src_offset+(sy+1) * srcStep + sx]) : (int4)0;
        v3 = (sx+1 >= 0 && sx+1 < src_cols && sy+1 >= 0 && sy+1 < src_rows) ? convert_int4(src[src_offset+(sy+1) * srcStep + sx+1]) : (int4)0;

        int itab0, itab1, itab2, itab3;
        float taby, tabx;
        taby = 1.f/INTER_TAB_SIZE*ay;
        tabx = 1.f/INTER_TAB_SIZE*ax;

        itab0 = convert_short_sat(rint( (1.0f-taby)*(1.0f-tabx) * INTER_REMAP_COEF_SCALE ));
        itab1 = convert_short_sat(rint( (1.0f-taby)*tabx * INTER_REMAP_COEF_SCALE ));
        itab2 = convert_short_sat(rint( taby*(1.0f-tabx) * INTER_REMAP_COEF_SCALE ));
        itab3 = convert_short_sat(rint( taby*tabx * INTER_REMAP_COEF_SCALE ));

        int4 val;
        val = v0 * itab0 +  v1 * itab1 + v2 * itab2 + v3 * itab3;

        if(dx >= 0 && dx < dst_cols && dy >= 0 && dy < dst_rows)
            dst[(dst_offset>>2)+dy*(dstStep>>2)+dx] =  convert_uchar4_sat ( (val + (1 << (INTER_REMAP_COEF_BITS-1))) >> INTER_REMAP_COEF_BITS ) ;
    }
}

__kernel void warpPerspectiveCubic_C4_D0(__global uchar4 const * restrict src, __global uchar4 * dst,
        int src_cols, int src_rows, int dst_cols, int dst_rows, int srcStep,
        int dstStep, int src_offset, int dst_offset,  __constant F * M, int threadCols )
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if( dx < threadCols && dy < dst_rows)
    {
        src_offset = (src_offset>>2);
        srcStep = (srcStep>>2);
        dst_offset = (dst_offset>>2);
        dstStep = (dstStep>>2);

        F X0 = M[0]*dx + M[1]*dy + M[2];
        F Y0 = M[3]*dx + M[4]*dy + M[5];
        F W = M[6]*dx + M[7]*dy + M[8];
        W = (W != 0.0f) ? INTER_TAB_SIZE/W : 0.0f;
        int X = rint(X0*W);
        int Y = rint(Y0*W);

        short sx = convert_short_sat(X >> INTER_BITS) - 1;
        short sy = convert_short_sat(Y >> INTER_BITS) - 1;
        short ay = (short)(Y & (INTER_TAB_SIZE-1));
        short ax = (short)(X & (INTER_TAB_SIZE-1));

        uchar4 v[16];
        int i,j;
#pragma unroll 4
        for(i=0; i<4; i++)
            for(j=0; j<4; j++)
            {
                v[i*4+j] = (sx+j >= 0 && sx+j < src_cols && sy+i >= 0 && sy+i < src_rows) ? (src[src_offset+(sy+i) * srcStep + (sx+j)])  : (uchar4)0;
            }
        int itab[16];
        float tab1y[4], tab1x[4];
        float axx, ayy;

        ayy = INTER_SCALE * ay;
        axx = INTER_SCALE * ax;
        interpolateCubic(ayy, tab1y);
        interpolateCubic(axx, tab1x);
        int isum = 0;

#pragma unroll 16
        for( i=0; i<16; i++ )
        {
            float tmp;
            tmp = tab1y[(i>>2)] * tab1x[(i&3)] * INTER_REMAP_COEF_SCALE;
            itab[i] = rint(tmp);
            isum += itab[i];
        }

        if( isum != INTER_REMAP_COEF_SCALE )
        {
            int k1, k2;
            int diff = isum - INTER_REMAP_COEF_SCALE;
            int Mk1=2, Mk2=2, mk1=2, mk2=2;

            for( k1 = 2; k1 < 4; k1++ )
                for( k2 = 2; k2 < 4; k2++ )
                {

                    if( itab[(k1<<2)+k2] < itab[(mk1<<2)+mk2] )
                        mk1 = k1, mk2 = k2;
                    else if( itab[(k1<<2)+k2] > itab[(Mk1<<2)+Mk2] )
                        Mk1 = k1, Mk2 = k2;
                }

            diff<0 ? (itab[(Mk1<<2)+Mk2]=(short)(itab[(Mk1<<2)+Mk2]-diff)) : (itab[(mk1<<2)+mk2]=(short)(itab[(mk1<<2)+mk2]-diff));
        }

        if( dx >= 0 && dx < dst_cols && dy >= 0 && dy < dst_rows)
        {
            int4 sum=0;
            for ( i =0; i<16; i++ )
            {
                sum += convert_int4(v[i]) * itab[i];
            }
            dst[dst_offset+dy*dstStep+dx] = convert_uchar4_sat( (sum + (1 << (INTER_REMAP_COEF_BITS-1))) >> INTER_REMAP_COEF_BITS ) ;
        }
    }
}


/**********************************************32FC1********************************************
***********************************************************************************************/

__kernel void warpPerspectiveNN_C1_D5(__global float * src, __global float * dst, int src_cols, int src_rows,
                                      int dst_cols, int dst_rows, int srcStep, int dstStep,
                                      int src_offset, int dst_offset,  __constant F * M, int threadCols )
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if( dx < threadCols && dy < dst_rows)
    {
        F X0 = M[0]*dx + M[1]*dy + M[2];
        F Y0 = M[3]*dx + M[4]*dy + M[5];
        F W = M[6]*dx + M[7]*dy + M[8];
        W = (W != 0.0f) ? 1.f/W : 0.0f;
        short sx = convert_short_sat_rte(X0*W);
        short sy = convert_short_sat_rte(Y0*W);

        if(dx >= 0 && dx < dst_cols && dy >= 0 && dy < dst_rows)
            dst[(dst_offset>>2)+dy*dstStep+dx]= (sx>=0 && sx<src_cols && sy>=0 && sy<src_rows) ? src[(src_offset>>2)+sy*srcStep+sx] : 0;
    }
}

__kernel void warpPerspectiveLinear_C1_D5(__global float * src, __global float * dst, int src_cols, int src_rows,
        int dst_cols, int dst_rows, int srcStep, int dstStep,
        int src_offset, int dst_offset,  __constant F * M, int threadCols )
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if( dx < threadCols && dy < dst_rows)
    {
        src_offset = (src_offset>>2);

        F X0 = M[0]*dx + M[1]*dy + M[2];
        F Y0 = M[3]*dx + M[4]*dy + M[5];
        F W = M[6]*dx + M[7]*dy + M[8];
        W = (W != 0.0f) ? INTER_TAB_SIZE/W : 0.0f;
        int X = rint(X0*W);
        int Y = rint(Y0*W);

        short sx = convert_short_sat(X >> INTER_BITS);
        short sy = convert_short_sat(Y >> INTER_BITS);
        short ay = (short)(Y & (INTER_TAB_SIZE-1));
        short ax = (short)(X & (INTER_TAB_SIZE-1));

        float v0, v1, v2, v3;

        v0 = (sx >= 0 && sx < src_cols && sy >= 0 && sy < src_rows) ? src[src_offset+sy * srcStep + sx] : (float)0;
        v1 = (sx+1 >= 0 && sx+1 < src_cols && sy >= 0 && sy < src_rows) ? src[src_offset+sy * srcStep + sx+1] : (float)0;
        v2 = (sx >= 0 && sx < src_cols && sy+1 >= 0 && sy+1 < src_rows) ? src[src_offset+(sy+1) * srcStep + sx] : (float)0;
        v3 = (sx+1 >= 0 && sx+1 < src_cols && sy+1 >= 0 && sy+1 < src_rows) ? src[src_offset+(sy+1) * srcStep + sx+1] : (float)0;

        float tab[4];
        float taby[2], tabx[2];
        taby[0] = 1.0f - 1.f/INTER_TAB_SIZE*ay;
        taby[1] = 1.f/INTER_TAB_SIZE*ay;
        tabx[0] = 1.0f - 1.f/INTER_TAB_SIZE*ax;
        tabx[1] = 1.f/INTER_TAB_SIZE*ax;

        tab[0] = taby[0] * tabx[0];
        tab[1] = taby[0] * tabx[1];
        tab[2] = taby[1] * tabx[0];
        tab[3] = taby[1] * tabx[1];

        float sum = 0;
        sum += v0 * tab[0] +  v1 * tab[1] +  v2 * tab[2] +  v3 * tab[3];
        if(dx >= 0 && dx < dst_cols && dy >= 0 && dy < dst_rows)
            dst[(dst_offset>>2)+dy*dstStep+dx] = sum;
    }
}

__kernel void warpPerspectiveCubic_C1_D5(__global float * src, __global float * dst, int src_cols, int src_rows,
        int dst_cols, int dst_rows, int srcStep, int dstStep,
        int src_offset, int dst_offset,  __constant F * M, int threadCols )
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if( dx < threadCols && dy < dst_rows)
    {
        src_offset = (src_offset>>2);
        dst_offset = (dst_offset>>2);

        F X0 = M[0]*dx + M[1]*dy + M[2];
        F Y0 = M[3]*dx + M[4]*dy + M[5];
        F W = M[6]*dx + M[7]*dy + M[8];
        W = (W != 0.0f) ? INTER_TAB_SIZE/W : 0.0f;
        int X = rint(X0*W);
        int Y = rint(Y0*W);

        short sx = convert_short_sat(X >> INTER_BITS) - 1;
        short sy = convert_short_sat(Y >> INTER_BITS) - 1;
        short ay = (short)(Y & (INTER_TAB_SIZE-1));
        short ax = (short)(X & (INTER_TAB_SIZE-1));

        float v[16];
        int i;

        for(i=0; i<16;  i++)
            v[i] = (sx+(i&3) >= 0 && sx+(i&3) < src_cols && sy+(i>>2) >= 0 && sy+(i>>2) < src_rows) ? src[src_offset+(sy+(i>>2)) * srcStep + (sx+(i&3))] : (float)0;

        float tab[16];
        float tab1y[4], tab1x[4];
        float axx, ayy;

        ayy = 1.f/INTER_TAB_SIZE * ay;
        axx = 1.f/INTER_TAB_SIZE * ax;
        interpolateCubic(ayy, tab1y);
        interpolateCubic(axx, tab1x);

#pragma unroll 4
        for( i=0; i<16; i++ )
        {
            tab[i] = tab1y[(i>>2)] * tab1x[(i&3)];
        }

        if( dx >= 0 && dx < dst_cols && dy >= 0 && dy < dst_rows)
        {
            float sum = 0;
#pragma unroll 4
            for ( i =0; i<16; i++ )
            {
                sum += v[i] * tab[i];
            }
            dst[dst_offset+dy*dstStep+dx] = sum;

        }
    }
}


/**********************************************32FC4********************************************
***********************************************************************************************/

__kernel void warpPerspectiveNN_C4_D5(__global float4 * src, __global float4 * dst, int src_cols, int src_rows,
                                      int dst_cols, int dst_rows, int srcStep, int dstStep,
                                      int src_offset, int dst_offset,  __constant F * M, int threadCols )
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if( dx < threadCols && dy < dst_rows)
    {
        F X0 = M[0]*dx + M[1]*dy + M[2];
        F Y0 = M[3]*dx + M[4]*dy + M[5];
        F W = M[6]*dx + M[7]*dy + M[8];
        W =(W != 0.0f)? 1.f/W : 0.0f;
        short sx = convert_short_sat_rte(X0*W);
        short sy = convert_short_sat_rte(Y0*W);

        if(dx >= 0 && dx < dst_cols && dy >= 0 && dy < dst_rows)
            dst[(dst_offset>>4)+dy*(dstStep>>2)+dx]= (sx>=0 && sx<src_cols && sy>=0 && sy<src_rows) ? src[(src_offset>>4)+sy*(srcStep>>2)+sx] : (float)0;
    }
}

__kernel void warpPerspectiveLinear_C4_D5(__global float4 * src, __global float4 * dst, int src_cols, int src_rows,
        int dst_cols, int dst_rows, int srcStep, int dstStep,
        int src_offset, int dst_offset,  __constant F * M, int threadCols )
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if( dx < threadCols && dy < dst_rows)
    {
        src_offset = (src_offset>>4);
        dst_offset = (dst_offset>>4);
        srcStep = (srcStep>>2);
        dstStep = (dstStep>>2);

        F X0 = M[0]*dx + M[1]*dy + M[2];
        F Y0 = M[3]*dx + M[4]*dy + M[5];
        F W = M[6]*dx + M[7]*dy + M[8];
        W = (W != 0.0f) ? INTER_TAB_SIZE/W : 0.0f;
        int X = rint(X0*W);
        int Y = rint(Y0*W);

        short sx0 = convert_short_sat(X >> INTER_BITS);
        short sy0 = convert_short_sat(Y >> INTER_BITS);
        short ay0 = (short)(Y & (INTER_TAB_SIZE-1));
        short ax0 = (short)(X & (INTER_TAB_SIZE-1));


        float4 v0, v1, v2, v3;

        v0 = (sx0 >= 0 && sx0 < src_cols && sy0 >= 0 && sy0 < src_rows) ? src[src_offset+sy0 * srcStep + sx0] : (float4)0;
        v1 = (sx0+1 >= 0 && sx0+1 < src_cols && sy0 >= 0 && sy0 < src_rows) ? src[src_offset+sy0 * srcStep + sx0+1] : (float4)0;
        v2 = (sx0 >= 0 && sx0 < src_cols && sy0+1 >= 0 && sy0+1 < src_rows) ? src[src_offset+(sy0+1) * srcStep + sx0] : (float4)0;
        v3 = (sx0+1 >= 0 && sx0+1 < src_cols && sy0+1 >= 0 && sy0+1 < src_rows) ? src[src_offset+(sy0+1) * srcStep + sx0+1] : (float4)0;

        float tab[4];
        float taby[2], tabx[2];
        taby[0] = 1.0f - 1.f/INTER_TAB_SIZE*ay0;
        taby[1] = 1.f/INTER_TAB_SIZE*ay0;
        tabx[0] = 1.0f - 1.f/INTER_TAB_SIZE*ax0;
        tabx[1] = 1.f/INTER_TAB_SIZE*ax0;

        tab[0] = taby[0] * tabx[0];
        tab[1] = taby[0] * tabx[1];
        tab[2] = taby[1] * tabx[0];
        tab[3] = taby[1] * tabx[1];

        float4 sum = 0;
        sum += v0 * tab[0] +  v1 * tab[1] +  v2 * tab[2] +  v3 * tab[3];
        if(dx >= 0 && dx < dst_cols && dy >= 0 && dy < dst_rows)
            dst[dst_offset+dy*dstStep+dx] = sum;
    }
}

__kernel void warpPerspectiveCubic_C4_D5(__global float4 * src, __global float4 * dst,
        int src_cols, int src_rows, int dst_cols, int dst_rows, int srcStep,
        int dstStep, int src_offset, int dst_offset,  __constant F * M, int threadCols )
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if( dx < threadCols && dy < dst_rows )
    {
        src_offset = (src_offset>>4);
        dst_offset = (dst_offset>>4);
        srcStep = (srcStep>>2);
        dstStep = (dstStep>>2);

        F X0 = M[0]*dx + M[1]*dy + M[2];
        F Y0 = M[3]*dx + M[4]*dy + M[5];
        F W = M[6]*dx + M[7]*dy + M[8];
        W = (W != 0.0f) ? INTER_TAB_SIZE/W : 0.0f;
        int X = rint(X0*W);
        int Y = rint(Y0*W);

        short sx = convert_short_sat(X >> INTER_BITS)-1;
        short sy = convert_short_sat(Y >> INTER_BITS)-1;
        short ay = (short)(Y & (INTER_TAB_SIZE-1));
        short ax = (short)(X & (INTER_TAB_SIZE-1));


        float4 v[16];
        int i;

        for(i=0; i<16;  i++)
            v[i] = (sx+(i&3) >= 0 && sx+(i&3) < src_cols && sy+(i>>2) >= 0 && sy+(i>>2) < src_rows) ? src[src_offset+(sy+(i>>2)) * srcStep + (sx+(i&3))] : (float4)0;

        float tab[16];
        float tab1y[4], tab1x[4];
        float axx, ayy;

        ayy = 1.f/INTER_TAB_SIZE * ay;
        axx = 1.f/INTER_TAB_SIZE * ax;
        interpolateCubic(ayy, tab1y);
        interpolateCubic(axx, tab1x);

#pragma unroll 4
        for( i=0; i<16; i++ )
        {
            tab[i] = tab1y[(i>>2)] * tab1x[(i&3)];
        }

        if( dx >= 0 && dx < dst_cols && dy >= 0 && dy < dst_rows)
        {
            float4 sum = 0;
#pragma unroll 4
            for ( i =0; i<16; i++ )
            {
                sum += v[i] * tab[i];
            }
            dst[dst_offset+dy*dstStep+dx] = sum;

        }
    }
}
