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
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Shengen Yan,yanshengen@gmail.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
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

#if defined (DOUBLE_SUPPORT)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
//wrapAffine kernel
//support four data types: CV_8U, CV_16U, CV_32S, CV_32F, and three interpolation methods: NN, Linear, Cubic.

#define INTER_BITS 5
#define INTER_TAB_SIZE (1 << INTER_BITS)
#define AB_BITS max(10, (int)INTER_BITS) 
#define AB_SCALE (1 << AB_BITS) 
#define INTER_REMAP_COEF_BITS 15
#define INTER_REMAP_COEF_SCALE (1 << INTER_REMAP_COEF_BITS)

//this round operation is to approximate CPU's saturate_cast<int>
int round2_int(double v)
{
    int v1=(int)v;
    if(((v-v1)==0.5 || (v1-v)==0.5) && (v1%2)==0)
        return v1;
    else
        return convert_int_sat(v+(v>=0 ? 0.5 : -0.5));
}

inline void interpolateCubic( float x, float* coeffs )
{
    const float A = -0.75f;

    coeffs[0] = ((A*(x + 1) - 5*A)*(x + 1) + 8*A)*(x + 1) - 4*A;
    coeffs[1] = ((A + 2)*x - (A + 3))*x*x + 1;
    coeffs[2] = ((A + 2)*(1 - x) - (A + 3))*(1 - x)*(1 - x) + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

__kernel void warpAffine_8u_NN(__global uchar * src, __global uchar * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    int round_delta = AB_SCALE/2;
    
    int X0 = round2_int(M[0] * dx * AB_SCALE);
    int Y0 = round2_int(M[3] * dx * AB_SCALE);
    X0 += round2_int((M[1]*dy + M[2]) * AB_SCALE) + round_delta;
    Y0 += round2_int((M[4]*dy + M[5]) * AB_SCALE) + round_delta;

    short sx = (short)(X0 >> AB_BITS);
    short sy = (short)(Y0 >> AB_BITS);
    for(int c = 0; c < cn; c++)
        dst[dy*dstStep+dx*cn+c] = (sx >= 0 && sx < cols && sy >= 0 && sy < rows) ? src[sy*srcStep+sx*cn+c] : 0; 
}

__kernel void warpAffine_8u_Linear(__global uchar * src, __global uchar * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    int round_delta = AB_SCALE/INTER_TAB_SIZE/2;
    
    int X0 = round2_int(M[0] * dx * AB_SCALE);
    int Y0 = round2_int(M[3] * dx * AB_SCALE);
    X0 += round2_int((M[1]*dy + M[2]) * AB_SCALE) + round_delta;
    Y0 += round2_int((M[4]*dy + M[5]) * AB_SCALE) + round_delta;
    int X = X0 >> (AB_BITS - INTER_BITS);
    int Y = Y0 >> (AB_BITS - INTER_BITS);

    short sx = (short)(X >> INTER_BITS);
    short sy = (short)(Y >> INTER_BITS);
    short ay = (short)(Y & (INTER_TAB_SIZE-1));
    short ax = (short)(X & (INTER_TAB_SIZE-1));
    
    int v[16];
    int i, j, c;

    for(i=0; i<2;  i++)
        for(j=0; j<2; j++)
            for(c=0; c<cn; c++)
                v[i*2*cn + j*cn + c] = (sx+j >= 0 && sx+j < cols && sy+i >= 0 && sy+i < rows) ? src[(sy+i) * srcStep + (sx+j)*cn + c] : 0;
   
    short itab[4];
    float tab1y[2], tab1x[2];
    tab1y[0] = 1.0 - 1.f/INTER_TAB_SIZE*ay;
    tab1y[1] = 1.f/INTER_TAB_SIZE*ay;
    tab1x[0] = 1.0 - 1.f/INTER_TAB_SIZE*ax;
    tab1x[1] = 1.f/INTER_TAB_SIZE*ax;
    
    for( i=0; i<2; i++ )
    {
        for( j=0; j<2; j++)
        {
            float v = tab1y[i] * tab1x[j];
            itab[i*2+j] = convert_short_sat(round2_int( v * INTER_REMAP_COEF_SCALE ));
        }
    }
    if( sx+1 < 0 || sx >= cols || sy+1 < 0 || sy >= rows)
    {
        for(c = 0; c < cn; c++)
            dst[dy*dstStep+dx*cn+c] = 0;
    }
    else
    {
        int sum;
        for(c = 0; c < cn; c++)
        {
            sum = 0;
            for ( i =0; i<4; i++ )
            {
                sum += v[i*cn+c] * itab[i] ;
            }
            dst[dy*dstStep+dx*cn+c] = convert_uchar_sat ( ((int)sum + (1 << (INTER_REMAP_COEF_BITS-1))) >> INTER_REMAP_COEF_BITS ) ;
        }
    }
}

__kernel void warpAffine_8u_Cubic(__global uchar * src, __global uchar * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    int round_delta = AB_SCALE/INTER_TAB_SIZE/2;
    
    int X0 = round2_int(M[0] * dx * AB_SCALE);
    int Y0 = round2_int(M[3] * dx * AB_SCALE);
    X0 += round2_int((M[1]*dy + M[2]) * AB_SCALE) + round_delta;
    Y0 += round2_int((M[4]*dy + M[5]) * AB_SCALE) + round_delta;
    int X = X0 >> (AB_BITS - INTER_BITS);
    int Y = Y0 >> (AB_BITS - INTER_BITS);

    short sx = (short)(X >> INTER_BITS) - 1;
    short sy = (short)(Y >> INTER_BITS) - 1;
    short ay = (short)(Y & (INTER_TAB_SIZE-1));
    short ax = (short)(X & (INTER_TAB_SIZE-1));
    
    uchar v[64];
    int i, j, c;

    for(i=0; i<4;  i++)
        for(j=0; j<4; j++)
            for(c=0; c<cn; c++)
                v[i*4*cn + j*cn + c] = (sx+j >= 0 && sx+j < cols && sy+i >= 0 && sy+i < rows) ? src[(sy+i) * srcStep + (sx+j)*cn + c] : 0;
   
    short itab[16];
    float tab1y[4], tab1x[4];
    float axx, ayy;

    ayy = 1.f/INTER_TAB_SIZE * ay;
    axx = 1.f/INTER_TAB_SIZE * ax;
    interpolateCubic(ayy, tab1y);
    interpolateCubic(axx, tab1x);
    int isum = 0;
    for( i=0; i<4; i++ )
    {
        for( j=0; j<4; j++)
        {
            double v = tab1y[i] * tab1x[j];
            isum += itab[i*4+j] = convert_short_sat( round2_int( v * INTER_REMAP_COEF_SCALE ) );
        }
    }
    if( isum != INTER_REMAP_COEF_SCALE )
    {
        int k1, k2, ksize = 4;
        int diff = isum - INTER_REMAP_COEF_SCALE;
        int ksize2 = ksize/2, Mk1=ksize2, Mk2=ksize2, mk1=ksize2, mk2=ksize2;
        for( k1 = ksize2; k1 < ksize2+2; k1++ )
            for( k2 = ksize2; k2 < ksize2+2; k2++ )
            {
                if( itab[k1*ksize+k2] < itab[mk1*ksize+mk2] )
                    mk1 = k1, mk2 = k2;
                else if( itab[k1*ksize+k2] > itab[Mk1*ksize+Mk2] )
                     Mk1 = k1, Mk2 = k2;
            }
            if( diff < 0 )
                itab[Mk1*ksize + Mk2] = (short)(itab[Mk1*ksize + Mk2] - diff);
            else
                itab[mk1*ksize + mk2] = (short)(itab[mk1*ksize + mk2] - diff);
    }

    if( sx+4 < 0 || sx >= cols || sy+4 < 0 || sy >= rows)
    {
        for(c = 0; c < cn; c++)
            dst[dy*dstStep+dx*cn+c] = 0;
    }
    else
    {
        int sum;
        for(c = 0; c < cn; c++)
        {
            sum = 0;
            for ( i =0; i<16; i++ )
            {
                sum += v[i*cn+c] * itab[i] ;
            }
            dst[dy*dstStep+dx*cn+c] = convert_uchar_sat( (int)(sum + (1 << (INTER_REMAP_COEF_BITS-1))) >> INTER_REMAP_COEF_BITS ) ;
        }
    }
}

__kernel void warpAffine_16u_NN(__global ushort * src, __global ushort * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    int round_delta = AB_SCALE/2;
    
    int X0 = round2_int(M[0] * dx * AB_SCALE);
    int Y0 = round2_int(M[3] * dx * AB_SCALE);
    X0 += round2_int((M[1]*dy + M[2]) * AB_SCALE) + round_delta;
    Y0 += round2_int((M[4]*dy + M[5]) * AB_SCALE) + round_delta;

    short sx = (short)(X0 >> AB_BITS);
    short sy = (short)(Y0 >> AB_BITS);
    for(int c = 0; c < cn; c++)
        dst[dy*dstStep+dx*cn+c] = (sx >= 0 && sx < cols && sy >= 0 && sy < rows) ? src[sy*srcStep+sx*cn+c] : 0; 
}

__kernel void warpAffine_16u_Linear(__global ushort * src, __global ushort * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    int round_delta = AB_SCALE/INTER_TAB_SIZE/2;
    
    int X0 = round2_int(M[0] * dx * AB_SCALE);
    int Y0 = round2_int(M[3] * dx * AB_SCALE);
    X0 += round2_int((M[1]*dy + M[2]) * AB_SCALE) + round_delta;
    Y0 += round2_int((M[4]*dy + M[5]) * AB_SCALE) + round_delta;
    int X = X0 >> (AB_BITS - INTER_BITS);
    int Y = Y0 >> (AB_BITS - INTER_BITS);

    short sx = (short)(X >> INTER_BITS);
    short sy = (short)(Y >> INTER_BITS);
    short ay = (short)(Y & (INTER_TAB_SIZE-1));
    short ax = (short)(X & (INTER_TAB_SIZE-1));
    
    ushort v[16];
    int i, j, c;

    for(i=0; i<2;  i++)
        for(j=0; j<2; j++)
            for(c=0; c<cn; c++)
                v[i*2*cn + j*cn + c] = (sx+j >= 0 && sx+j < cols && sy+i >= 0 && sy+i < rows) ? src[(sy+i) * srcStep + (sx+j)*cn + c] : 0;
   
    float tab[4];
    float tab1y[2], tab1x[2];
    tab1y[0] = 1.0 - 1.f/INTER_TAB_SIZE*ay;
    tab1y[1] = 1.f/INTER_TAB_SIZE*ay;
    tab1x[0] = 1.0 - 1.f/INTER_TAB_SIZE*ax;
    tab1x[1] = 1.f/INTER_TAB_SIZE*ax;
    
    for( i=0; i<2; i++ )
    {
        for( j=0; j<2; j++)
        {
            tab[i*2+j] = tab1y[i] * tab1x[j];
        }
    }
    if( sx+1 < 0 || sx >= cols || sy+1 < 0 || sy >= rows)
    {
        for(c = 0; c < cn; c++)
            dst[dy*dstStep+dx*cn+c] = 0;
    }
    else
    {
        float sum;
        for(c = 0; c < cn; c++)
        {
            sum = 0;
            for ( i =0; i<4; i++ )
            {
                sum += v[i*cn+c] * tab[i] ;
            }
            dst[dy*dstStep+dx*cn+c] = convert_ushort_sat( round2_int(sum) ) ;
        }
    }
}

__kernel void warpAffine_16u_Cubic(__global ushort * src, __global ushort * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    int round_delta = AB_SCALE/INTER_TAB_SIZE/2;
    
    int X0 = round2_int(M[0] * dx * AB_SCALE);
    int Y0 = round2_int(M[3] * dx * AB_SCALE);
    X0 += round2_int((M[1]*dy + M[2]) * AB_SCALE) + round_delta;
    Y0 += round2_int((M[4]*dy + M[5]) * AB_SCALE) + round_delta;
    int X = X0 >> (AB_BITS - INTER_BITS);
    int Y = Y0 >> (AB_BITS - INTER_BITS);

    short sx = (short)(X >> INTER_BITS) - 1;
    short sy = (short)(Y >> INTER_BITS) - 1;
    short ay = (short)(Y & (INTER_TAB_SIZE-1));
    short ax = (short)(X & (INTER_TAB_SIZE-1));
    
    ushort v[64];
    int i, j, c;

    for(i=0; i<4;  i++)
        for(j=0; j<4; j++)
            for(c=0; c<cn; c++)
                v[i*4*cn + j*cn + c] = (sx+j >= 0 && sx+j < cols && sy+i >= 0 && sy+i < rows) ? src[(sy+i) * srcStep + (sx+j)*cn + c] : 0;
   
    float tab[16];
    float tab1y[4], tab1x[4];
    float axx, ayy;

    ayy = 1.f/INTER_TAB_SIZE * ay;
    axx = 1.f/INTER_TAB_SIZE * ax;
    interpolateCubic(ayy, tab1y);
    interpolateCubic(axx, tab1x);
    for( i=0; i<4; i++ )
    {
        for( j=0; j<4; j++)
        {
            tab[i*4+j] = tab1y[i] * tab1x[j];
        }
    }

    int width = cols-3>0 ? cols-3 : 0;
    int height = rows-3>0 ? rows-3 : 0;
    if((unsigned)sx < width && (unsigned)sy < height )
    {
        float sum;
        for(c = 0; c < cn; c++)
        {
            sum = 0;
            for ( i =0; i<4; i++ )
            {
                    sum += v[i*4*cn+c] * tab[i*4] + v[i*4*cn+c+1]*tab[i*4+1]
                          +v[i*4*cn+c+2] * tab[i*4+2] + v[i*4*cn+c+3]*tab[i*4+3];
            }
            dst[dy*dstStep+dx*cn+c] = convert_ushort_sat( round2_int(sum ));
        }
    }
    else if( sx+4 < 0 || sx >= cols || sy+4 < 0 || sy >= rows)
    {
        for(c = 0; c < cn; c++)
            dst[dy*dstStep+dx*cn+c] = 0;
    }
    else
    {
        float sum;
        for(c = 0; c < cn; c++)
        {
            sum = 0;
            for ( i =0; i<16; i++ )
            {
                    sum += v[i*cn+c] * tab[i];
            }
            dst[dy*dstStep+dx*cn+c] = convert_ushort_sat( round2_int(sum ));
        }
    }
}


__kernel void warpAffine_32s_NN(__global int * src, __global int * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    int round_delta = AB_SCALE/2;
    
    int X0 = round2_int(M[0] * dx * AB_SCALE);
    int Y0 = round2_int(M[3] * dx * AB_SCALE);
    X0 += round2_int((M[1]*dy + M[2]) * AB_SCALE) + round_delta;
    Y0 += round2_int((M[4]*dy + M[5]) * AB_SCALE) + round_delta;

    short sx = (short)(X0 >> AB_BITS);
    short sy = (short)(Y0 >> AB_BITS);
    for(int c = 0; c < cn; c++)
        dst[dy*dstStep+dx*cn+c] = (sx >= 0 && sx < cols && sy >= 0 && sy < rows) ? src[sy*srcStep+sx*cn+c] : 0; 
}

__kernel void warpAffine_32s_Linear(__global int * src, __global int * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    int round_delta = AB_SCALE/INTER_TAB_SIZE/2;
    
    int X0 = round2_int(M[0] * dx * AB_SCALE);
    int Y0 = round2_int(M[3] * dx * AB_SCALE);
    X0 += round2_int((M[1]*dy + M[2]) * AB_SCALE) + round_delta;
    Y0 += round2_int((M[4]*dy + M[5]) * AB_SCALE) + round_delta;
    int X = X0 >> (AB_BITS - INTER_BITS);
    int Y = Y0 >> (AB_BITS - INTER_BITS);

    short sx = (short)(X >> INTER_BITS);
    short sy = (short)(Y >> INTER_BITS);
    short ay = (short)(Y & (INTER_TAB_SIZE-1));
    short ax = (short)(X & (INTER_TAB_SIZE-1));
    
    int v[16];
    int i, j, c;

    for(i=0; i<2;  i++)
        for(j=0; j<2; j++)
            for(c=0; c<cn; c++)
                v[i*2*cn + j*cn + c] = (sx+j >= 0 && sx+j < cols && sy+i >= 0 && sy+i < rows) ? src[(sy+i) * srcStep + (sx+j)*cn + c] : 0;
   
    float tab[4];
    float tab1y[2], tab1x[2];
    tab1y[0] = 1.0 - 1.f/INTER_TAB_SIZE*ay;
    tab1y[1] = 1.f/INTER_TAB_SIZE*ay;
    tab1x[0] = 1.0 - 1.f/INTER_TAB_SIZE*ax;
    tab1x[1] = 1.f/INTER_TAB_SIZE*ax;
    
    for( i=0; i<2; i++ )
    {
        for( j=0; j<2; j++)
        {
            tab[i*2+j] = tab1y[i] * tab1x[j];
        }
    }
    if( sx+1 < 0 || sx >= cols || sy+1 < 0 || sy >= rows)
    {
        for(c = 0; c < cn; c++)
            dst[dy*dstStep+dx*cn+c] = 0;
    }
    else
    {
        float sum;
        for(c = 0; c < cn; c++)
        {
            sum = 0;
            for ( i =0; i<4; i++ )
            {
                sum += v[i*cn+c] * tab[i] ;
            }
            dst[dy*dstStep+dx*cn+c] = convert_int_sat( round2_int(sum) ) ;
        }
    }
}

__kernel void warpAffine_32s_Cubic(__global int * src, __global int * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    int round_delta = AB_SCALE/INTER_TAB_SIZE/2;
    
    int X0 = round2_int(M[0] * dx * AB_SCALE);
    int Y0 = round2_int(M[3] * dx * AB_SCALE);
    X0 += round2_int((M[1]*dy + M[2]) * AB_SCALE) + round_delta;
    Y0 += round2_int((M[4]*dy + M[5]) * AB_SCALE) + round_delta;
    int X = X0 >> (AB_BITS - INTER_BITS);
    int Y = Y0 >> (AB_BITS - INTER_BITS);

    short sx = (short)(X >> INTER_BITS) - 1;
    short sy = (short)(Y >> INTER_BITS) - 1;
    short ay = (short)(Y & (INTER_TAB_SIZE-1));
    short ax = (short)(X & (INTER_TAB_SIZE-1));
    
    int v[64];
    int i, j, c;

    for(i=0; i<4;  i++)
        for(j=0; j<4; j++)
            for(c=0; c<cn; c++)
                v[i*4*cn + j*cn + c] = (sx+j >= 0 && sx+j < cols && sy+i >= 0 && sy+i < rows) ? src[(sy+i) * srcStep + (sx+j)*cn + c] : 0;
   
    float tab[16];
    float tab1y[4], tab1x[4];
    float axx, ayy;

    ayy = 1.f/INTER_TAB_SIZE * ay;
    axx = 1.f/INTER_TAB_SIZE * ax;
    interpolateCubic(ayy, tab1y);
    interpolateCubic(axx, tab1x);
    for( i=0; i<4; i++ )
    {
        for( j=0; j<4; j++)
        {
            tab[i*4+j] = tab1y[i] * tab1x[j];
        }
    }

    if( sx+4 < 0 || sx >= cols || sy+4 < 0 || sy >= rows)
    {
        for(c = 0; c < cn; c++)
            dst[dy*dstStep+dx*cn+c] = 0;
    }
    else
    {
        float sum;
        for(c = 0; c < cn; c++)
        {
            sum = 0;
            for ( i =0; i<16; i++ )
            {
                sum += v[i*cn+c] * tab[i] ;
            }
            dst[dy*dstStep+dx*cn+c] = convert_int_sat( round2_int(sum ));
        }
    }
}


__kernel void warpAffine_32f_NN(__global float * src, __global float * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    int round_delta = AB_SCALE/2;
    
    int X0 = round2_int(M[0] * dx * AB_SCALE);
    int Y0 = round2_int(M[3] * dx * AB_SCALE);
    X0 += round2_int((M[1]*dy + M[2]) * AB_SCALE) + round_delta;
    Y0 += round2_int((M[4]*dy + M[5]) * AB_SCALE) + round_delta;

    short sx = (short)(X0 >> AB_BITS);
    short sy = (short)(Y0 >> AB_BITS);
    for(int c = 0; c < cn; c++)
        dst[dy*dstStep+dx*cn+c] = (sx >= 0 && sx < cols && sy >= 0 && sy < rows) ? src[sy*srcStep+sx*cn+c] : 0; 
}

__kernel void warpAffine_32f_Linear(__global float * src, __global float * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    int round_delta = AB_SCALE/INTER_TAB_SIZE/2;
    
    int X0 = round2_int(M[0] * dx * AB_SCALE);
    int Y0 = round2_int(M[3] * dx * AB_SCALE);
    X0 += round2_int((M[1]*dy + M[2]) * AB_SCALE) + round_delta;
    Y0 += round2_int((M[4]*dy + M[5]) * AB_SCALE) + round_delta;
    int X = X0 >> (AB_BITS - INTER_BITS);
    int Y = Y0 >> (AB_BITS - INTER_BITS);

    short sx = (short)(X >> INTER_BITS);
    short sy = (short)(Y >> INTER_BITS);
    short ay = (short)(Y & (INTER_TAB_SIZE-1));
    short ax = (short)(X & (INTER_TAB_SIZE-1));
    
    float v[16];
    int i, j, c;

    for(i=0; i<2;  i++)
        for(j=0; j<2; j++)
            for(c=0; c<cn; c++)
                v[i*2*cn + j*cn + c] = (sx+j >= 0 && sx+j < cols && sy+i >= 0 && sy+i < rows) ? src[(sy+i) * srcStep + (sx+j)*cn + c] : 0;
   
    float tab[4];
    float tab1y[2], tab1x[2];
    tab1y[0] = 1.0 - 1.f/INTER_TAB_SIZE*ay;
    tab1y[1] = 1.f/INTER_TAB_SIZE*ay;
    tab1x[0] = 1.0 - 1.f/INTER_TAB_SIZE*ax;
    tab1x[1] = 1.f/INTER_TAB_SIZE*ax;
    
    for( i=0; i<2; i++ )
    {
        for( j=0; j<2; j++)
        {
            tab[i*2+j] = tab1y[i] * tab1x[j];
        }
    }
    if( sx+1 < 0 || sx >= cols || sy+1 < 0 || sy >= rows)
    {
        for(c = 0; c < cn; c++)
            dst[dy*dstStep+dx*cn+c] = 0;
    }
    else
    {
        float sum;
        for(c = 0; c < cn; c++)
        {
            sum = 0;
            for ( i =0; i<4; i++ )
            {
                sum += v[i*cn+c] * tab[i] ;
            }
            dst[dy*dstStep+dx*cn+c] = sum ;
        }
    }
}

__kernel void warpAffine_32f_Cubic(__global float * src, __global float * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    int round_delta = AB_SCALE/INTER_TAB_SIZE/2;
    
    int X0 = round2_int(M[0] * dx * AB_SCALE);
    int Y0 = round2_int(M[3] * dx * AB_SCALE);
    X0 += round2_int((M[1]*dy + M[2]) * AB_SCALE) + round_delta;
    Y0 += round2_int((M[4]*dy + M[5]) * AB_SCALE) + round_delta;
    int X = X0 >> (AB_BITS - INTER_BITS);
    int Y = Y0 >> (AB_BITS - INTER_BITS);

    short sx = (short)(X >> INTER_BITS) - 1;
    short sy = (short)(Y >> INTER_BITS) - 1;
    short ay = (short)(Y & (INTER_TAB_SIZE-1));
    short ax = (short)(X & (INTER_TAB_SIZE-1));
    
    float v[64];
    int i, j, c;

    for(i=0; i<4;  i++)
        for(j=0; j<4; j++)
            for(c=0; c<cn; c++)
                v[i*4*cn + j*cn + c] = (sx+j >= 0 && sx+j < cols && sy+i >= 0 && sy+i < rows) ? src[(sy+i) * srcStep + (sx+j)*cn + c] : 0;
   
    float tab[16];
    float tab1y[4], tab1x[4];
    float axx, ayy;

    ayy = 1.f/INTER_TAB_SIZE * ay;
    axx = 1.f/INTER_TAB_SIZE * ax;
    interpolateCubic(ayy, tab1y);
    interpolateCubic(axx, tab1x);
    for( i=0; i<4; i++ )
    {
        for( j=0; j<4; j++)
        {
            tab[i*4+j] = tab1y[i] * tab1x[j];
        }
    }
    int width = cols-3>0 ? cols-3 : 0;
    int height = rows-3>0 ? rows-3 : 0;
    if((unsigned)sx < width && (unsigned)sy < height )
    {
        float sum;
        for(c = 0; c < cn; c++)
        {
            sum = 0;
            for ( i =0; i<4; i++ )
            {
                    sum += v[i*4*cn+c] * tab[i*4] + v[i*4*cn+c+1]*tab[i*4+1]
                          +v[i*4*cn+c+2] * tab[i*4+2] + v[i*4*cn+c+3]*tab[i*4+3];
            }
            dst[dy*dstStep+dx*cn+c] = sum;
        }
    }
    else if( sx+4 < 0 || sx >= cols || sy+4 < 0 || sy >= rows)
    {
        for(c = 0; c < cn; c++)
            dst[dy*dstStep+dx*cn+c] = 0;
    }
    else
    {
        float sum;
        for(c = 0; c < cn; c++)
        {
            sum = 0;
            for ( i =0; i<16; i++ )
            {
                    sum += v[i*cn+c] * tab[i];
            }
            dst[dy*dstStep+dx*cn+c] = sum;
        }
    }
}

__kernel void warpPerspective_8u_NN(__global uchar * src, __global uchar * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    double X0 = M[0]*dx + M[1]*dy + M[2];
    double Y0 = M[3]*dx + M[4]*dy + M[5];
    double W = M[6]*dx + M[7]*dy + M[8];
    W = W ? 1./W : 0;
    int X = round2_int(X0*W);
    int Y = round2_int(Y0*W);
    short sx = (short)X;
    short sy = (short)Y;

    for(int c = 0; c < cn; c++)
        dst[dy*dstStep+dx*cn+c] = (sx >= 0 && sx < cols && sy >= 0 && sy < rows) ? src[sy*srcStep+sx*cn+c] : 0; 
}

__kernel void warpPerspective_8u_Linear(__global uchar * src, __global uchar * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    double X0 = M[0]*dx + M[1]*dy + M[2];
    double Y0 = M[3]*dx + M[4]*dy + M[5];
    double W = M[6]*dx + M[7]*dy + M[8];
    W = W ? INTER_TAB_SIZE/W : 0;
    int X = round2_int(X0*W);
    int Y = round2_int(Y0*W);
    
    short sx = (short)(X >> INTER_BITS);
    short sy = (short)(Y >> INTER_BITS);
    short ay = (short)(Y & (INTER_TAB_SIZE-1));
    short ax = (short)(X & (INTER_TAB_SIZE-1));
    
    uchar v[16];
    int i, j, c;

    for(i=0; i<2;  i++)
        for(j=0; j<2; j++)
            for(c=0; c<cn; c++)
                v[i*2*cn + j*cn + c] = (sx+j >= 0 && sx+j < cols && sy+i >= 0 && sy+i < rows) ? src[(sy+i) * srcStep + (sx+j)*cn + c] : 0;
   
    short itab[4];
    float tab1y[2], tab1x[2];
    tab1y[0] = 1.0 - 1.f/INTER_TAB_SIZE*ay;
    tab1y[1] = 1.f/INTER_TAB_SIZE*ay;
    tab1x[0] = 1.0 - 1.f/INTER_TAB_SIZE*ax;
    tab1x[1] = 1.f/INTER_TAB_SIZE*ax;
    
    for( i=0; i<2; i++ )
    {
        for( j=0; j<2; j++)
        {
            float v = tab1y[i] * tab1x[j];
            itab[i*2+j] = convert_short_sat(round2_int( v * INTER_REMAP_COEF_SCALE ));
        }
    }
    if( sx+1 < 0 || sx >= cols || sy+1 < 0 || sy >= rows)
    {
        for(c = 0; c < cn; c++)
            dst[dy*dstStep+dx*cn+c] = 0;
    }
    else
    {
        int sum;
        for(c = 0; c < cn; c++)
        {
            sum = 0;
            for ( i =0; i<4; i++ )
            {
                sum += v[i*cn+c] * itab[i] ;
            }
            dst[dy*dstStep+dx*cn+c] = convert_uchar_sat ( round2_int(sum + (1 << (INTER_REMAP_COEF_BITS-1))) >> INTER_REMAP_COEF_BITS ) ;
        }
    }
}

__kernel void warpPerspective_8u_Cubic(__global uchar * src, __global uchar * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    double X0 = M[0]*dx + M[1]*dy + M[2];
    double Y0 = M[3]*dx + M[4]*dy + M[5];
    double W = M[6]*dx + M[7]*dy + M[8];
    W = W ? INTER_TAB_SIZE/W : 0;
    int X = round2_int(X0*W);
    int Y = round2_int(Y0*W);
    
    short sx = (short)(X >> INTER_BITS) - 1;
    short sy = (short)(Y >> INTER_BITS) - 1;
    short ay = (short)(Y & (INTER_TAB_SIZE-1));
    short ax = (short)(X & (INTER_TAB_SIZE-1));
    
    uchar v[64];
    int i, j, c;

    for(i=0; i<4;  i++)
        for(j=0; j<4; j++)
            for(c=0; c<cn; c++)
                v[i*4*cn + j*cn + c] = (sx+j >= 0 && sx+j < cols && sy+i >= 0 && sy+i < rows) ? src[(sy+i) * srcStep + (sx+j)*cn + c] : 0;
   
    short itab[16];
    float tab1y[4], tab1x[4];
    float axx, ayy;

    ayy = 1.f/INTER_TAB_SIZE * ay;
    axx = 1.f/INTER_TAB_SIZE * ax;
    interpolateCubic(ayy, tab1y);
    interpolateCubic(axx, tab1x);
    int isum = 0;
    for( i=0; i<4; i++ )
    {
        for( j=0; j<4; j++)
        {
            double v = tab1y[i] * tab1x[j];
            isum += itab[i*4+j] = convert_short_sat( round2_int( v * INTER_REMAP_COEF_SCALE ) );
        }
    }
    if( isum != INTER_REMAP_COEF_SCALE )
    {
        int k1, k2, ksize = 4;
        int diff = isum - INTER_REMAP_COEF_SCALE;
        int ksize2 = ksize/2, Mk1=ksize2, Mk2=ksize2, mk1=ksize2, mk2=ksize2;
        for( k1 = ksize2; k1 < ksize2+2; k1++ )
            for( k2 = ksize2; k2 < ksize2+2; k2++ )
            {
                if( itab[k1*ksize+k2] < itab[mk1*ksize+mk2] )
                    mk1 = k1, mk2 = k2;
                else if( itab[k1*ksize+k2] > itab[Mk1*ksize+Mk2] )
                     Mk1 = k1, Mk2 = k2;
            }
            if( diff < 0 )
                itab[Mk1*ksize + Mk2] = (short)(itab[Mk1*ksize + Mk2] - diff);
            else
                itab[mk1*ksize + mk2] = (short)(itab[mk1*ksize + mk2] - diff);
    }

    if( sx+4 < 0 || sx >= cols || sy+4 < 0 || sy >= rows)
    {
        for(c = 0; c < cn; c++)
            dst[dy*dstStep+dx*cn+c] = 0;
    }
    else
    {
        int sum;
        for(c = 0; c < cn; c++)
        {
            sum = 0;
            for ( i =0; i<16; i++ )
            {
                sum += v[i*cn+c] * itab[i] ;
            }
            dst[dy*dstStep+dx*cn+c] = convert_uchar_sat( round2_int(sum + (1 << (INTER_REMAP_COEF_BITS-1))) >> INTER_REMAP_COEF_BITS ) ;
        }
    }
}

__kernel void warpPerspective_16u_NN(__global ushort * src, __global ushort * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    double X0 = M[0]*dx + M[1]*dy + M[2];
    double Y0 = M[3]*dx + M[4]*dy + M[5];
    double W = M[6]*dx + M[7]*dy + M[8];
    W = W ? 1./W : 0;
    int X = round2_int(X0*W);
    int Y = round2_int(Y0*W);
    short sx = (short)X;
    short sy = (short)Y;

    for(int c = 0; c < cn; c++)
        dst[dy*dstStep+dx*cn+c] = (sx >= 0 && sx < cols && sy >= 0 && sy < rows) ? src[sy*srcStep+sx*cn+c] : 0; 
}

__kernel void warpPerspective_16u_Linear(__global ushort * src, __global ushort * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    double X0 = M[0]*dx + M[1]*dy + M[2];
    double Y0 = M[3]*dx + M[4]*dy + M[5];
    double W = M[6]*dx + M[7]*dy + M[8];
    W = W ? INTER_TAB_SIZE/W : 0;
    int X = round2_int(X0*W);
    int Y = round2_int(Y0*W);
    
    short sx = (short)(X >> INTER_BITS);
    short sy = (short)(Y >> INTER_BITS);
    short ay = (short)(Y & (INTER_TAB_SIZE-1));
    short ax = (short)(X & (INTER_TAB_SIZE-1));
    
    ushort v[16];
    int i, j, c;

    for(i=0; i<2;  i++)
        for(j=0; j<2; j++)
            for(c=0; c<cn; c++)
                v[i*2*cn + j*cn + c] = (sx+j >= 0 && sx+j < cols && sy+i >= 0 && sy+i < rows) ? src[(sy+i) * srcStep + (sx+j)*cn + c] : 0;
   
    float tab[4];
    float tab1y[2], tab1x[2];
    tab1y[0] = 1.0 - 1.f/INTER_TAB_SIZE*ay;
    tab1y[1] = 1.f/INTER_TAB_SIZE*ay;
    tab1x[0] = 1.0 - 1.f/INTER_TAB_SIZE*ax;
    tab1x[1] = 1.f/INTER_TAB_SIZE*ax;
    
    for( i=0; i<2; i++ )
    {
        for( j=0; j<2; j++)
        {
            tab[i*2+j] = tab1y[i] * tab1x[j];
        }
    }
    if( sx+1 < 0 || sx >= cols || sy+1 < 0 || sy >= rows)
    {
        for(c = 0; c < cn; c++)
            dst[dy*dstStep+dx*cn+c] = 0;
    }
    else
    {
        float sum;
        for(c = 0; c < cn; c++)
        {
            sum = 0;
            for ( i =0; i<4; i++ )
            {
                sum += v[i*cn+c] * tab[i] ;
            }
            dst[dy*dstStep+dx*cn+c] = convert_ushort_sat( round2_int(sum) ) ;
        }
    }
}

__kernel void warpPerspective_16u_Cubic(__global ushort * src, __global ushort * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    double X0 = M[0]*dx + M[1]*dy + M[2];
    double Y0 = M[3]*dx + M[4]*dy + M[5];
    double W = M[6]*dx + M[7]*dy + M[8];
    W = W ? INTER_TAB_SIZE/W : 0;
    int X = round2_int(X0*W);
    int Y = round2_int(Y0*W);
    
    short sx = (short)(X >> INTER_BITS) - 1;
    short sy = (short)(Y >> INTER_BITS) - 1;
    short ay = (short)(Y & (INTER_TAB_SIZE-1));
    short ax = (short)(X & (INTER_TAB_SIZE-1));
    
    ushort v[64];
    int i, j, c;

    for(i=0; i<4;  i++)
        for(j=0; j<4; j++)
            for(c=0; c<cn; c++)
                v[i*4*cn + j*cn + c] = (sx+j >= 0 && sx+j < cols && sy+i >= 0 && sy+i < rows) ? src[(sy+i) * srcStep + (sx+j)*cn + c] : 0;
   
    float tab[16];
    float tab1y[4], tab1x[4];
    float axx, ayy;

    ayy = 1.f/INTER_TAB_SIZE * ay;
    axx = 1.f/INTER_TAB_SIZE * ax;
    interpolateCubic(ayy, tab1y);
    interpolateCubic(axx, tab1x);
    for( i=0; i<4; i++ )
    {
        for( j=0; j<4; j++)
        {
            tab[i*4+j] = tab1y[i] * tab1x[j];
        }
    }

    int width = cols-3>0 ? cols-3 : 0;
    int height = rows-3>0 ? rows-3 : 0;
    if((unsigned)sx < width && (unsigned)sy < height )
    {
        float sum;
        for(c = 0; c < cn; c++)
        {
            sum = 0;
            for ( i =0; i<4; i++ )
            {
                    sum += v[i*4*cn+c] * tab[i*4] + v[i*4*cn+c+1]*tab[i*4+1]
                          +v[i*4*cn+c+2] * tab[i*4+2] + v[i*4*cn+c+3]*tab[i*4+3];
            }
            dst[dy*dstStep+dx*cn+c] = convert_ushort_sat( round2_int(sum ));
        }
    }
    else if( sx+4 < 0 || sx >= cols || sy+4 < 0 || sy >= rows)
    {
        for(c = 0; c < cn; c++)
            dst[dy*dstStep+dx*cn+c] = 0;
    }
    else
    {
        float sum;
        for(c = 0; c < cn; c++)
        {
            sum = 0;
            for ( i =0; i<16; i++ )
            {
                    sum += v[i*cn+c] * tab[i];
            }
            dst[dy*dstStep+dx*cn+c] = convert_ushort_sat( round2_int(sum ));
        }
    }
}


__kernel void warpPerspective_32s_NN(__global int * src, __global int * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    double X0 = M[0]*dx + M[1]*dy + M[2];
    double Y0 = M[3]*dx + M[4]*dy + M[5];
    double W = M[6]*dx + M[7]*dy + M[8];
    W = W ? 1./W : 0;
    int X = round2_int(X0*W);
    int Y = round2_int(Y0*W);
    short sx = (short)X;
    short sy = (short)Y;

    for(int c = 0; c < cn; c++)
        dst[dy*dstStep+dx*cn+c] = (sx >= 0 && sx < cols && sy >= 0 && sy < rows) ? src[sy*srcStep+sx*cn+c] : 0; 
}

__kernel void warpPerspective_32s_Linear(__global int * src, __global int * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    double X0 = M[0]*dx + M[1]*dy + M[2];
    double Y0 = M[3]*dx + M[4]*dy + M[5];
    double W = M[6]*dx + M[7]*dy + M[8];
    W = W ? INTER_TAB_SIZE/W : 0;
    int X = round2_int(X0*W);
    int Y = round2_int(Y0*W);
    
    short sx = (short)(X >> INTER_BITS);
    short sy = (short)(Y >> INTER_BITS);
    short ay = (short)(Y & (INTER_TAB_SIZE-1));
    short ax = (short)(X & (INTER_TAB_SIZE-1));
    
    int v[16];
    int i, j, c;

    for(i=0; i<2;  i++)
        for(j=0; j<2; j++)
            for(c=0; c<cn; c++)
                v[i*2*cn + j*cn + c] = (sx+j >= 0 && sx+j < cols && sy+i >= 0 && sy+i < rows) ? src[(sy+i) * srcStep + (sx+j)*cn + c] : 0;
   
    float tab[4];
    float tab1y[2], tab1x[2];
    tab1y[0] = 1.0 - 1.f/INTER_TAB_SIZE*ay;
    tab1y[1] = 1.f/INTER_TAB_SIZE*ay;
    tab1x[0] = 1.0 - 1.f/INTER_TAB_SIZE*ax;
    tab1x[1] = 1.f/INTER_TAB_SIZE*ax;
    
    for( i=0; i<2; i++ )
    {
        for( j=0; j<2; j++)
        {
            tab[i*2+j] = tab1y[i] * tab1x[j];
        }
    }
    if( sx+1 < 0 || sx >= cols || sy+1 < 0 || sy >= rows)
    {
        for(c = 0; c < cn; c++)
            dst[dy*dstStep+dx*cn+c] = 0;
    }
    else
    {
        float sum;
        for(c = 0; c < cn; c++)
        {
            sum = 0;
            for ( i =0; i<4; i++ )
            {
                sum += v[i*cn+c] * tab[i] ;
            }
            dst[dy*dstStep+dx*cn+c] = convert_int_sat( round2_int(sum) ) ;
        }
    }
}

__kernel void warpPerspective_32s_Cubic(__global int * src, __global int * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    double X0 = M[0]*dx + M[1]*dy + M[2];
    double Y0 = M[3]*dx + M[4]*dy + M[5];
    double W = M[6]*dx + M[7]*dy + M[8];
    W = W ? INTER_TAB_SIZE/W : 0;
    int X = round2_int(X0*W);
    int Y = round2_int(Y0*W);
    
    short sx = (short)(X >> INTER_BITS) - 1;
    short sy = (short)(Y >> INTER_BITS) - 1;
    short ay = (short)(Y & (INTER_TAB_SIZE-1));
    short ax = (short)(X & (INTER_TAB_SIZE-1));

    int v[64];
    int i, j, c;

    for(i=0; i<4;  i++)
        for(j=0; j<4; j++)
            for(c=0; c<cn; c++)
                v[i*4*cn + j*cn + c] = (sx+j >= 0 && sx+j < cols && sy+i >= 0 && sy+i < rows) ? src[(sy+i) * srcStep + (sx+j)*cn + c] : 0;
   
    float tab[16];
    float tab1y[4], tab1x[4];
    float axx, ayy;

    ayy = 1.f/INTER_TAB_SIZE * ay;
    axx = 1.f/INTER_TAB_SIZE * ax;
    interpolateCubic(ayy, tab1y);
    interpolateCubic(axx, tab1x);
    for( i=0; i<4; i++ )
    {
        for( j=0; j<4; j++)
        {
            tab[i*4+j] = tab1y[i] * tab1x[j];
        }
    }

    if( sx+4 < 0 || sx >= cols || sy+4 < 0 || sy >= rows)
    {
        for(c = 0; c < cn; c++)
            dst[dy*dstStep+dx*cn+c] = 0;
    }
    else
    {
        float sum;
        for(c = 0; c < cn; c++)
        {
            sum = 0;
            for ( i =0; i<16; i++ )
            {
                sum += v[i*cn+c] * tab[i] ;
            }
            dst[dy*dstStep+dx*cn+c] = convert_int_sat( round2_int(sum ));
        }
    }
}


__kernel void warpPerspective_32f_NN(__global float * src, __global float * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    double X0 = M[0]*dx + M[1]*dy + M[2];
    double Y0 = M[3]*dx + M[4]*dy + M[5];
    double W = M[6]*dx + M[7]*dy + M[8];
    W = W ? 1./W : 0;
    int X = round2_int(X0*W);
    int Y = round2_int(Y0*W);
    short sx = (short)X;
    short sy = (short)Y;

    for(int c = 0; c < cn; c++)
        dst[dy*dstStep+dx*cn+c] = (sx >= 0 && sx < cols && sy >= 0 && sy < rows) ? src[sy*srcStep+sx*cn+c] : 0; 
}

__kernel void warpPerspective_32f_Linear(__global float * src, __global float * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    double X0 = M[0]*dx + M[1]*dy + M[2];
    double Y0 = M[3]*dx + M[4]*dy + M[5];
    double W = M[6]*dx + M[7]*dy + M[8];
    W = W ? INTER_TAB_SIZE/W : 0;
    int X = round2_int(X0*W);
    int Y = round2_int(Y0*W);
    
    short sx = (short)(X >> INTER_BITS);
    short sy = (short)(Y >> INTER_BITS);
    short ay = (short)(Y & (INTER_TAB_SIZE-1));
    short ax = (short)(X & (INTER_TAB_SIZE-1));
    
    float v[16];
    int i, j, c;

    for(i=0; i<2;  i++)
        for(j=0; j<2; j++)
            for(c=0; c<cn; c++)
                v[i*2*cn + j*cn + c] = (sx+j >= 0 && sx+j < cols && sy+i >= 0 && sy+i < rows) ? src[(sy+i) * srcStep + (sx+j)*cn + c] : 0;
   
    float tab[4];
    float tab1y[2], tab1x[2];
    tab1y[0] = 1.0 - 1.f/INTER_TAB_SIZE*ay;
    tab1y[1] = 1.f/INTER_TAB_SIZE*ay;
    tab1x[0] = 1.0 - 1.f/INTER_TAB_SIZE*ax;
    tab1x[1] = 1.f/INTER_TAB_SIZE*ax;
    
    for( i=0; i<2; i++ )
    {
        for( j=0; j<2; j++)
        {
            tab[i*2+j] = tab1y[i] * tab1x[j];
        }
    }
    if( sx+1 < 0 || sx >= cols || sy+1 < 0 || sy >= rows)
    {
        for(c = 0; c < cn; c++)
            dst[dy*dstStep+dx*cn+c] = 0;
    }
    else
    {
        float sum;
        for(c = 0; c < cn; c++)
        {
            sum = 0;
            for ( i =0; i<4; i++ )
            {
                sum += v[i*cn+c] * tab[i] ;
            }
            dst[dy*dstStep+dx*cn+c] = sum ;
        }
    }
}

__kernel void warpPerspective_32f_Cubic(__global float * src, __global float * dst, int cols, int rows,  int cn,
                            int srcStep, int dstStep, __global double * M, int interpolation)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    
    double X0 = M[0]*dx + M[1]*dy + M[2];
    double Y0 = M[3]*dx + M[4]*dy + M[5];
    double W = M[6]*dx + M[7]*dy + M[8];
    W = W ? INTER_TAB_SIZE/W : 0;
    int X = round2_int(X0*W);
    int Y = round2_int(Y0*W);
    
    short sx = (short)(X >> INTER_BITS) - 1;
    short sy = (short)(Y >> INTER_BITS) - 1;
    short ay = (short)(Y & (INTER_TAB_SIZE-1));
    short ax = (short)(X & (INTER_TAB_SIZE-1));

    float v[64];
    int i, j, c;

    for(i=0; i<4;  i++)
        for(j=0; j<4; j++)
            for(c=0; c<cn; c++)
                v[i*4*cn + j*cn + c] = (sx+j >= 0 && sx+j < cols && sy+i >= 0 && sy+i < rows) ? src[(sy+i) * srcStep + (sx+j)*cn + c] : 0;
   
    float tab[16];
    float tab1y[4], tab1x[4];
    float axx, ayy;

    ayy = 1.f/INTER_TAB_SIZE * ay;
    axx = 1.f/INTER_TAB_SIZE * ax;
    interpolateCubic(ayy, tab1y);
    interpolateCubic(axx, tab1x);
    for( i=0; i<4; i++ )
    {
        for( j=0; j<4; j++)
        {
            tab[i*4+j] = tab1y[i] * tab1x[j];
        }
    }

    int width = cols-3>0 ? cols-3 : 0;
    int height = rows-3>0 ? rows-3 : 0;
    if((unsigned)sx < width && (unsigned)sy < height )
    {
        float sum;
        for(c = 0; c < cn; c++)
        {
            sum = 0;
            for ( i =0; i<4; i++ )
            {
                    sum += v[i*4*cn+c] * tab[i*4] + v[i*4*cn+c+1]*tab[i*4+1]
                          +v[i*4*cn+c+2] * tab[i*4+2] + v[i*4*cn+c+3]*tab[i*4+3];
            }
            dst[dy*dstStep+dx*cn+c] = sum;
        }
    }
    else if( sx+4 < 0 || sx >= cols || sy+4 < 0 || sy >= rows)
    {
        for(c = 0; c < cn; c++)
            dst[dy*dstStep+dx*cn+c] = 0;
    }
    else
    {
        float sum;
        for(c = 0; c < cn; c++)
        {
            sum = 0;
            for ( i =0; i<16; i++ )
            {
                    sum += v[i*cn+c] * tab[i];
            }
            dst[dy*dstStep+dx*cn+c] = sum;
        }
    }
}
#endif
