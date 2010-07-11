/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
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

//////////////////////////////////////////////////////////////////////////////////////////
////////////////// tests for discrete linear transforms (FFT, DCT ...) ///////////////////
//////////////////////////////////////////////////////////////////////////////////////////

#include "cxcoretest.h"
#include <float.h>

typedef struct CvTsComplex32f
{
    float re, im;
}
CvTsComplex32f;

typedef struct CvTsComplex64f
{
    double re, im;
}
CvTsComplex64f;

static CvMat* cvTsInitDFTWave( int n, int inv )
{
    int i;
    double angle = (inv ? 1 : -1)*CV_PI*2/n;
    CvTsComplex64f wi, w1;
    CvMat* wave = cvCreateMat( 1, n, CV_64FC2 );
    CvTsComplex64f* w = (CvTsComplex64f*)wave->data.db;

    w1.re = cos(angle);
    w1.im = sin(angle);
    w[0].re = wi.re = 1.;
    w[0].im = wi.im = 0.;

    for( i = 1; i < n; i++ )
    {
        double t = wi.re*w1.re - wi.im*w1.im;
        wi.im = wi.re*w1.im + wi.im*w1.re;
        wi.re = t;
        w[i] = wi;
    }

    return wave;
}


static void cvTsDFT_1D( const CvMat* _src, CvMat* _dst, int flags, CvMat* wave=0 )
{
    int i, j, k, n = _dst->cols + _dst->rows - 1;
    const CvMat* wave0 = wave;
    double scale = (flags & CV_DXT_SCALE) ? 1./n : 1.;
    assert( _src->cols + _src->rows - 1 == n );
    int src_step = 1, dst_step = 1;
    CvTsComplex64f* w;

    assert( CV_ARE_TYPES_EQ(_src,_dst) && _src->rows*_src->cols == _dst->rows*_dst->cols );

    if( !wave )
        wave = cvTsInitDFTWave( n, flags & CV_DXT_INVERSE );

    w = (CvTsComplex64f*)wave->data.db;
    if( !CV_IS_MAT_CONT(_src->type) )
        src_step = _src->step/CV_ELEM_SIZE(_src->type);
    if( !CV_IS_MAT_CONT(_dst->type) )
        dst_step = _dst->step/CV_ELEM_SIZE(_dst->type);

    if( CV_MAT_TYPE(_src->type) == CV_32FC2 )
    {
        CvTsComplex32f* dst = (CvTsComplex32f*)_dst->data.fl;
        for( i = 0; i < n; i++, dst += dst_step )
        {
            CvTsComplex32f* src = (CvTsComplex32f*)_src->data.fl;
            CvTsComplex64f sum = {0,0};
            int delta = i;
            k = 0;

            for( j = 0; j < n; j++, src += src_step )
            {
                sum.re += src->re*w[k].re - src->im*w[k].im;
                sum.im += src->re*w[k].im + src->im*w[k].re;
                k += delta;
                k -= (k >= n ? n : 0);
            }

            dst->re = (float)(sum.re*scale);
            dst->im = (float)(sum.im*scale);
        }
    }
    else if( CV_MAT_TYPE(_src->type) == CV_64FC2 )
    {
        CvTsComplex64f* dst = (CvTsComplex64f*)_dst->data.db;
        for( i = 0; i < n; i++, dst += dst_step )
        {
            CvTsComplex64f* src = (CvTsComplex64f*)_src->data.db;
            CvTsComplex64f sum = {0,0};
            int delta = i;
            k = 0;

            for( j = 0; j < n; j++, src += src_step )
            {
                sum.re += src->re*w[k].re - src->im*w[k].im;
                sum.im += src->re*w[k].im + src->im*w[k].re;
                k += delta;
                k -= (k >= n ? n : 0);
            }

            dst->re = sum.re*scale;
            dst->im = sum.im*scale;
        }
    }
    else
        assert(0);

    if( !wave0 )
        cvReleaseMat( &wave );
}


static void cvTsDFT_2D( const CvMat* src, CvMat* dst, int flags )
{
    int i;
    CvMat* tmp = cvCreateMat( dst->cols, dst->rows, dst->type );
    CvMat* wave = cvTsInitDFTWave( dst->cols, flags & CV_DXT_INVERSE );

    // 1. row-wise transform
    for( i = 0; i < dst->rows; i++ )
    {
        CvMat src_row, dst_row;
        cvGetRow( src, &src_row, i );
        cvGetCol( tmp, &dst_row, i );
        cvTsDFT_1D( &src_row, &dst_row, flags, wave );
    }

    if( !(flags & CV_DXT_ROWS) )
    {
        if( dst->cols != dst->rows )
        {
            cvReleaseMat( &wave );
            wave = cvTsInitDFTWave( dst->rows, flags & CV_DXT_INVERSE );
        }

        // 2. column-wise transform
        for( i = 0; i < dst->cols; i++ )
        {
            CvMat src_row, dst_row;
            cvGetRow( tmp, &src_row, i );
            cvGetCol( dst, &dst_row, i );
            cvTsDFT_1D( &src_row, &dst_row, flags, wave );
        }
    }
    else
        cvTsTranspose( tmp, dst );

    cvReleaseMat( &wave );
    cvReleaseMat( &tmp );
}


static CvMat* cvTsInitDCTWave( int n, int inv )
{
    int i, k;
    double angle = CV_PI*0.5/n;
    CvMat* wave = cvCreateMat( n, n, CV_64FC1 );

    double scale = sqrt(1./n);
    for( k = 0; k < n; k++ )
        wave->data.db[k] = scale;
    scale *= sqrt(2.);
    for( i = 1; i < n; i++ )
        for( k = 0; k < n; k++ )
            wave->data.db[i*n + k] = scale*cos( angle*i*(2*k + 1) );

    if( inv )
        cvTsTranspose( wave, wave );

    return wave;
}


static void cvTsDCT_1D( const CvMat* _src, CvMat* _dst, int flags, CvMat* wave=0 )
{
    int i, j, n = _dst->cols + _dst->rows - 1;
    const CvMat* wave0 = wave;
    assert( _src->cols + _src->rows - 1 == n);
    int src_step = 1, dst_step = 1;
    double* w;

    assert( CV_ARE_TYPES_EQ(_src,_dst) && _src->rows*_src->cols == _dst->rows*_dst->cols );

    if( !wave )
        wave = cvTsInitDCTWave( n, flags & CV_DXT_INVERSE );
    w = wave->data.db;

    if( !CV_IS_MAT_CONT(_src->type) )
        src_step = _src->step/CV_ELEM_SIZE(_src->type);
    if( !CV_IS_MAT_CONT(_dst->type) )
        dst_step = _dst->step/CV_ELEM_SIZE(_dst->type);

    if( CV_MAT_TYPE(_src->type) == CV_32FC1 )
    {
        float *dst = _dst->data.fl;

        for( i = 0; i < n; i++, dst += dst_step )
        {
            const float* src = _src->data.fl;
            double sum = 0;

            for( j = 0; j < n; j++, src += src_step )
                sum += src[0]*w[j];
            w += n;
            dst[0] = (float)sum;
        }
    }
    else if( CV_MAT_TYPE(_src->type) == CV_64FC1 )
    {
        double *dst = _dst->data.db;

        for( i = 0; i < n; i++, dst += dst_step )
        {
            const double* src = _src->data.db;
            double sum = 0;

            for( j = 0; j < n; j++, src += src_step )
                sum += src[0]*w[j];
            w += n;
            dst[0] = sum;
        }
    }
    else
        assert(0);

    if( !wave0 )
        cvReleaseMat( &wave );
}


static void cvTsDCT_2D( const CvMat* src, CvMat* dst, int flags )
{
    int i;
    CvMat* tmp = cvCreateMat( dst->cols, dst->rows, dst->type );
    CvMat* wave = cvTsInitDCTWave( dst->cols, flags & CV_DXT_INVERSE );

    // 1. row-wise transform
    for( i = 0; i < dst->rows; i++ )
    {
        CvMat src_row, dst_row;
        cvGetRow( src, &src_row, i );
        cvGetCol( tmp, &dst_row, i );
        cvTsDCT_1D( &src_row, &dst_row, flags, wave );
    }

    if( !(flags & CV_DXT_ROWS) )
    {
        if( dst->cols != dst->rows )
        {
            cvReleaseMat( &wave );
            wave = cvTsInitDCTWave( dst->rows, flags & CV_DXT_INVERSE );
        }

        // 2. column-wise transform
        for( i = 0; i < dst->cols; i++ )
        {
            CvMat src_row, dst_row;
            cvGetRow( tmp, &src_row, i );
            cvGetCol( dst, &dst_row, i );
            cvTsDCT_1D( &src_row, &dst_row, flags, wave );
        }
    }
    else
    {
        cvTranspose( tmp, dst );
    }

    cvReleaseMat( &wave );
    cvReleaseMat( &tmp );
}


static void cvTsConvertFromCCS( const CvMat* _src0, const CvMat* _src1,
                                CvMat* _dst, int flags )
{
    if( _dst->rows > 1 && (_dst->cols > 1 || (flags & CV_DXT_ROWS)) )
    {
        int i, count = _dst->rows, len = _dst->cols;
        int is_2d = (flags & CV_DXT_ROWS) == 0;
        CvMat src0_row, src1_row, dst_row;
        for( i = 0; i < count; i++ )
        {
            int j = !is_2d || i == 0 ? i : count - i;
            cvGetRow( _src0, &src0_row, i );
            cvGetRow( _src1, &src1_row, j );
            cvGetRow( _dst, &dst_row, i );
            cvTsConvertFromCCS( &src0_row, &src1_row, &dst_row, 0 );
        }

        if( is_2d )
        {
            cvGetCol( _src0, &src0_row, 0 );
            cvGetCol( _dst, &dst_row, 0 );
            cvTsConvertFromCCS( &src0_row, &src0_row, &dst_row, 0 );
            if( (len & 1) == 0 )
            {
                cvGetCol( _src0, &src0_row, _src0->cols - 1 );
                cvGetCol( _dst, &dst_row, len/2 );
                cvTsConvertFromCCS( &src0_row, &src0_row, &dst_row, 0 );
            }
        }
    }
    else
    {
        int i, n = _dst->cols + _dst->rows - 1, n2 = (n+1) >> 1;
        int cn = CV_MAT_CN(_src0->type);
        int src_step = cn, dst_step = 1;

        if( !CV_IS_MAT_CONT(_dst->type) )
            dst_step = _dst->step/CV_ELEM_SIZE(_dst->type);

        if( !CV_IS_MAT_CONT(_src0->type) )
            src_step = _src0->step/CV_ELEM_SIZE(_src0->type & CV_MAT_DEPTH_MASK);

        if( CV_MAT_DEPTH(_dst->type) == CV_32F )
        {
            CvTsComplex32f* dst = (CvTsComplex32f*)_dst->data.fl;
            const float* src0 = _src0->data.fl;
            const float* src1 = _src1->data.fl;
            int delta0, delta1;

            dst->re = src0[0];
            dst->im = 0;

            if( (n & 1) == 0 )
            {
                dst[n2*dst_step].re = src0[(cn == 1 ? n-1 : n2)*src_step];
                dst[n2*dst_step].im = 0;
            }

            delta0 = src_step;
            delta1 = delta0 + (cn == 1 ? src_step : 1);
            if( cn == 1 )
                src_step *= 2;

            for( i = 1; i < n2; i++, delta0 += src_step, delta1 += src_step )
            {
                float t0 = src0[delta0];
                float t1 = src0[delta1];

                dst[i*dst_step].re = t0;
                dst[i*dst_step].im = t1;

                t0 = src1[delta0];
                t1 = -src1[delta1];

                dst[(n-i)*dst_step].re = t0;
                dst[(n-i)*dst_step].im = t1;
            }
        }
        else
        {
            CvTsComplex64f* dst = (CvTsComplex64f*)_dst->data.db;
            const double* src0 = _src0->data.db;
            const double* src1 = _src1->data.db;
            int delta0, delta1;

            dst->re = src0[0];
            dst->im = 0;

            if( (n & 1) == 0 )
            {
                dst[n2*dst_step].re = src0[(cn == 1 ? n-1 : n2)*src_step];
                dst[n2*dst_step].im = 0;
            }

            delta0 = src_step;
            delta1 = delta0 + (cn == 1 ? src_step : 1);
            if( cn == 1 )
                src_step *= 2;

            for( i = 1; i < n2; i++, delta0 += src_step, delta1 += src_step )
            {
                double t0 = src0[delta0];
                double t1 = src0[delta1];

                dst[i*dst_step].re = t0;
                dst[i*dst_step].im = t1;

                t0 = src1[delta0];
                t1 = -src1[delta1];

                dst[(n-i)*dst_step].re = t0;
                dst[(n-i)*dst_step].im = t1;
            }
        }
    }
}


static void cvTsFixCCS( CvMat* mat, int cols, int flags )
{
    int i, rows = mat->rows;
    int rows2 = flags & CV_DXT_ROWS ? rows : rows/2 + 1, cols2 = cols/2 + 1;

    assert( cols2 == mat->cols );

    if( CV_MAT_TYPE(mat->type) == CV_32FC2 )
    {
        for( i = 0; i < rows2; i++ )
        {
            CvTsComplex32f* row = (CvTsComplex32f*)(mat->data.ptr + mat->step*i);
            if( (flags & CV_DXT_ROWS) || i == 0 || (i == rows2 - 1 && rows % 2 == 0) )
            {
                row[0].im = 0;
                if( cols % 2 == 0 )
                    row[cols2-1].im = 0;
            }
            else
            {
                CvTsComplex32f* row2 = (CvTsComplex32f*)(mat->data.ptr + mat->step*(rows-i));
                row2[0].re = row[0].re;
                row2[0].im = -row[0].im;

                if( cols % 2 == 0 )
                {
                    row2[cols2-1].re = row[cols2-1].re;
                    row2[cols2-1].im = -row[cols2-1].im;
                }
            }
        }
    }
    else if( CV_MAT_TYPE(mat->type) == CV_64FC2 )
    {
        for( i = 0; i < rows2; i++ )
        {
            CvTsComplex64f* row = (CvTsComplex64f*)(mat->data.ptr + mat->step*i);
            if( (flags & CV_DXT_ROWS) || i == 0 || (i == rows2 - 1 && rows % 2 == 0) )
            {
                row[0].im = 0;
                if( cols % 2 == 0 )
                    row[cols2-1].im = 0;
            }
            else
            {
                CvTsComplex64f* row2 = (CvTsComplex64f*)(mat->data.ptr + mat->step*(rows-i));
                row2[0].re = row[0].re;
                row2[0].im = -row[0].im;

                if( cols % 2 == 0 )
                {
                    row2[cols2-1].re = row[cols2-1].re;
                    row2[cols2-1].im = -row[cols2-1].im;
                }
            }
        }
    }
}


static const CvSize dxt_sizes[] = {{16,1}, {256,1}, {1024,1}, {65536,1},
    {10,1}, {100,1}, {1000,1}, {100000,1}, {256, 256}, {1024,1024}, {-1,-1}};
static const int dxt_depths[] = { CV_32F, CV_64F, -1 };
static const char* dxt_param_names[] = { "size", "depth", "transform_type", 0 };
static const char* dft_transforms[] = { "Fwd_CToC", "Inv_CToC", "Fwd_RToPack", "Inv_PackToR", 0 };
static const char* mulsp_transforms[] = { "Fwd_CToC", "Fwd_RToPack", 0 };
static const char* dct_transforms[] = { "Fwd", "Inv", 0 };

class CxCore_DXTBaseTestImpl : public CvArrTest
{
public:
    CxCore_DXTBaseTestImpl( const char* test_name, const char* test_funcs,
                        bool _allow_complex=false, bool _allow_odd=false,
                        bool _spectrum_mode=false );
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    int prepare_test_case( int test_case_idx );
    double get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ );
    void get_timing_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool* are_images );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
    int write_default_params( CvFileStorage* fs );
    int flags; // transformation flags
    bool allow_complex, // whether input/output may be complex or not:
                        // true for DFT and MulSpectrums, false for DCT
         allow_odd,     // whether input/output may be have odd (!=1) dimensions:
                        // true for DFT and MulSpectrums, false for DCT
         spectrum_mode, // (2 complex/ccs inputs, 1 complex/ccs output):
                        // true for MulSpectrums, false for DFT and DCT
         inplace,       // inplace operation (set for each individual test case)
         temp_dst;      // use temporary destination (for real->ccs DFT and ccs MulSpectrums)
    const char** transform_type_list;
};


CxCore_DXTBaseTestImpl::CxCore_DXTBaseTestImpl( const char* test_name, const char* test_funcs,
                                        bool _allow_complex, bool _allow_odd, bool _spectrum_mode )
    : CvArrTest( test_name, test_funcs, "" ),
    flags(0), allow_complex(_allow_complex), allow_odd(_allow_odd),
    spectrum_mode(_spectrum_mode), inplace(false), temp_dst(false)
{
    test_array[INPUT].push(NULL);
    if( spectrum_mode )
        test_array[INPUT].push(NULL);
    test_array[OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);
    test_array[TEMP].push(NULL);
    test_array[TEMP].push(NULL);

    max_log_array_size = 9;
    element_wise_relative_error = spectrum_mode;

    size_list = (CvSize*)dxt_sizes;
    whole_size_list = 0;
    depth_list = (int*)dxt_depths;
    cn_list = 0;
    transform_type_list = 0;
}


void CxCore_DXTBaseTestImpl::get_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int bits = cvTsRandInt(rng);
    int depth = cvTsRandInt(rng)%2 + CV_32F;
    int cn = !allow_complex || !(bits & 256) ? 1 : 2;
    CvSize size;
    CvArrTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    flags = bits & (CV_DXT_INVERSE | CV_DXT_SCALE | CV_DXT_ROWS | CV_DXT_MUL_CONJ);
    if( spectrum_mode )
        flags &= ~CV_DXT_INVERSE;
    types[TEMP][0] = types[TEMP][1] = types[INPUT][0] =
        types[OUTPUT][0] = CV_MAKETYPE(depth, cn);
    size = sizes[INPUT][0];

    //size.width = size.width % 10 + 1;
    //size.height = size.width % 10 + 1;
    //size.width = 1;
    //flags &= ~CV_DXT_ROWS;
    temp_dst = false;

    if( flags & CV_DXT_ROWS && (bits&1024) )
    {
        if( bits&16 )
            size.width = 1;
        else
            size.height = 1;
        flags &= ~CV_DXT_ROWS;
    }

    const int P2_MIN_SIZE = 32;
    if( ((bits >> 10) & 1) == 0 )
    {
        size.width = (size.width / P2_MIN_SIZE)*P2_MIN_SIZE;
        size.width = MAX(size.width, 1);
        size.height = (size.height / P2_MIN_SIZE)*P2_MIN_SIZE;
        size.height = MAX(size.height, 1);
    }

    if( !allow_odd )
    {
        if( size.width > 1 && (size.width&1) != 0 )
            size.width = (size.width + 1) & -2;

        if( size.height > 1 && (size.height&1) != 0 && !(flags & CV_DXT_ROWS) )
            size.height = (size.height + 1) & -2;
    }

    sizes[INPUT][0] = sizes[OUTPUT][0] = size;
    sizes[TEMP][0] = sizes[TEMP][1] = cvSize(0,0);

    if( spectrum_mode )
    {
        if( cn == 1 )
        {
            types[OUTPUT][0] = depth + 8;
            sizes[TEMP][0] = size;
        }
        sizes[INPUT][0] = sizes[INPUT][1] = size;
        types[INPUT][1] = types[INPUT][0];
    }
    else if( /*(cn == 2 && (bits&32)) ||*/ (cn == 1 && allow_complex) )
    {
        types[TEMP][0] = depth + 8; // CV_??FC2
        sizes[TEMP][0] = size;
        size = cvSize(size.width/2+1, size.height);

        if( flags & CV_DXT_INVERSE )
        {
            if( cn == 2 )
            {
                types[OUTPUT][0] = depth;
                sizes[INPUT][0] = size;
            }
            types[TEMP][1] = types[TEMP][0];
            sizes[TEMP][1] = sizes[TEMP][0];
        }
        else
        {
            if( allow_complex )
                types[OUTPUT][0] = depth + 8;

            if( cn == 2 )
            {
                types[INPUT][0] = depth;
                types[TEMP][1] = types[TEMP][0];
                sizes[TEMP][1] = size;
            }
            else
            {
                types[TEMP][1] = depth;
                sizes[TEMP][1] = sizes[TEMP][0];
            }
            temp_dst = true;
        }
    }

    inplace = false;
    if( spectrum_mode ||
        (!temp_dst && types[INPUT][0] == types[OUTPUT][0]) ||
        (temp_dst && types[INPUT][0] == types[TEMP][1]) )
        inplace = (bits & 64) != 0;

    types[REF_OUTPUT][0] = types[OUTPUT][0];
    sizes[REF_OUTPUT][0] = sizes[OUTPUT][0];
}


void CxCore_DXTBaseTestImpl::get_timing_test_array_types_and_sizes( int test_case_idx,
                                                    CvSize** sizes, int** types,
                                                    CvSize** whole_sizes, bool* are_images )
{
    CvArrTest::get_timing_test_array_types_and_sizes( test_case_idx,
                            sizes, types, whole_sizes, are_images );
    const char* transform_type = cvReadString( find_timing_param( "transform_type" ), "" );
    int depth = CV_MAT_DEPTH(types[INPUT][0]);
    int in_type = depth, out_type = depth;

    if( strcmp( transform_type, "Fwd_CToC" ) == 0 || strcmp( transform_type, "Inv_CToC" ) == 0 )
        in_type = out_type = CV_MAKETYPE(depth,2);

    if( strncmp( transform_type, "Fwd", 3 ) == 0 )
        flags = CV_DXT_FORWARD;
    else
        flags = CV_DXT_INV_SCALE;

    types[INPUT][0] = in_type;
    if( spectrum_mode )
        types[INPUT][1] = in_type;
    types[OUTPUT][0] = types[REF_OUTPUT][0] = out_type;
    sizes[TEMP][0] = cvSize(0,0);

    inplace = false;
}


int CxCore_DXTBaseTestImpl::write_default_params( CvFileStorage* fs )
{
    int code = CvArrTest::write_default_params(fs);
    if( code < 0 || ts->get_testing_mode() != CvTS::TIMING_MODE )
        return code;
    write_string_list( fs, "transform_type", transform_type_list );
    return code;
}


void CxCore_DXTBaseTestImpl::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    sprintf( ptr, "%s,", cvReadString( find_timing_param("transform_type"), "" ) );
    ptr += strlen(ptr);
    params_left--;
    CvArrTest::print_timing_params( test_case_idx, ptr, params_left );
}


double CxCore_DXTBaseTestImpl::get_success_error_level( int test_case_idx, int i, int j )
{
    return CvArrTest::get_success_error_level( test_case_idx, i, j );
}


int CxCore_DXTBaseTestImpl::prepare_test_case( int test_case_idx )
{
    int code = CvArrTest::prepare_test_case( test_case_idx );
    if( code > 0 && ts->get_testing_mode() == CvTS::CORRECTNESS_CHECK_MODE )
    {
        int in_type = CV_MAT_TYPE(test_mat[INPUT][0].type);
        int out_type = CV_MAT_TYPE(test_mat[OUTPUT][0].type);

        if( CV_MAT_CN(in_type) == 2 && CV_MAT_CN(out_type) == 1 )
            cvTsFixCCS( &test_mat[INPUT][0], test_mat[OUTPUT][0].cols, flags );

        if( inplace )
            cvTsCopy( &test_mat[INPUT][test_case_idx & (int)spectrum_mode],
                temp_dst ? &test_mat[TEMP][1] :
                in_type == out_type ? &test_mat[OUTPUT][0] :
                &test_mat[TEMP][0] );
    }

    return code;
}

CxCore_DXTBaseTestImpl dxt_test( "dxt", "" );


class CxCore_DXTBaseTest : public CxCore_DXTBaseTestImpl
{
public:
    CxCore_DXTBaseTest( const char* test_name, const char* test_funcs,
                        bool _allow_complex=false, bool _allow_odd=false,
                        bool _spectrum_mode=false );
};

CxCore_DXTBaseTest::CxCore_DXTBaseTest( const char* test_name, const char* test_funcs,
                                        bool _allow_complex, bool _allow_odd, bool _spectrum_mode )
    : CxCore_DXTBaseTestImpl( test_name, test_funcs, _allow_complex, _allow_odd, _spectrum_mode )
{
    size_list = 0;
    depth_list = 0;
    default_timing_param_names = dxt_param_names;
    transform_type_list = dft_transforms;
}


////////////////////// FFT ////////////////////////
class CxCore_DFTTest : public CxCore_DXTBaseTest
{
public:
    CxCore_DFTTest();
protected:
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


CxCore_DFTTest::CxCore_DFTTest() : CxCore_DXTBaseTest( "dxt-dft", "cvDFT", true, true, false )
{
}


void CxCore_DFTTest::run_func()
{
    CvArr* dst = temp_dst ? test_array[TEMP][1] : test_array[OUTPUT][0];
    CvArr* src = inplace ? dst : test_array[INPUT][0];

    if(!(flags & CV_DXT_INVERSE))
        cvDFT( src, dst, flags );
    else
    {
        cv::Mat _src = cv::cvarrToMat(src), _dst = cv::cvarrToMat(dst);
        idft(_src, _dst, flags & ~CV_DXT_INVERSE);
    }
}


void CxCore_DFTTest::prepare_to_validation( int /*test_case_idx*/ )
{
    CvMat* src = &test_mat[INPUT][0];
    CvMat* dst = &test_mat[REF_OUTPUT][0];
    CvMat* tmp_src = src;
    CvMat* tmp_dst = dst;
    int src_cn = CV_MAT_CN( src->type );
    int dst_cn = CV_MAT_CN( dst->type );

    if( src_cn != 2 || dst_cn != 2 )
    {
        tmp_src = &test_mat[TEMP][0];

        if( !(flags & CV_DXT_INVERSE ) )
        {
            CvMat* cvdft_dst = &test_mat[TEMP][1];
            cvTsConvertFromCCS( cvdft_dst, cvdft_dst,
                                &test_mat[OUTPUT][0], flags );
            cvTsZero( tmp_src );
            cvTsInsert( src, tmp_src, 0 );
        }
        else
        {
            cvTsConvertFromCCS( src, src, tmp_src, flags );
            tmp_dst = &test_mat[TEMP][1];
        }
    }

    if( src->rows == 1 || (src->cols == 1 && !(flags & CV_DXT_ROWS)) )
        cvTsDFT_1D( tmp_src, tmp_dst, flags );
    else
        cvTsDFT_2D( tmp_src, tmp_dst, flags );

    if( tmp_dst != dst )
        cvTsExtract( tmp_dst, dst, 0 );
}


//CxCore_DFTTest dft_test;


////////////////////// DCT ////////////////////////
class CxCore_DCTTest : public CxCore_DXTBaseTest
{
public:
    CxCore_DCTTest();
protected:
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


CxCore_DCTTest::CxCore_DCTTest() : CxCore_DXTBaseTest( "dxt-dct", "cvDCT", false, false, false )
{
    transform_type_list = dct_transforms;
}


void CxCore_DCTTest::run_func()
{
    CvArr* dst = test_array[OUTPUT][0];
    CvArr* src = inplace ? dst : test_array[INPUT][0];

    if(!(flags & CV_DXT_INVERSE))
        cvDCT( src, dst, flags );
    else
    {
        cv::Mat _src = cv::cvarrToMat(src), _dst = cv::cvarrToMat(dst);
        idct(_src, _dst, flags & ~CV_DXT_INVERSE);
    }
}


void CxCore_DCTTest::prepare_to_validation( int /*test_case_idx*/ )
{
    CvMat* src = &test_mat[INPUT][0];
    CvMat* dst = &test_mat[REF_OUTPUT][0];

    if( src->rows == 1 || (src->cols == 1 && !(flags & CV_DXT_ROWS)) )
        cvTsDCT_1D( src, dst, flags );
    else
        cvTsDCT_2D( src, dst, flags );
}


//CxCore_DCTTest dct_test;


////////////////////// MulSpectrums ////////////////////////
class CxCore_MulSpectrumsTest : public CxCore_DXTBaseTest
{
public:
    CxCore_MulSpectrumsTest();
protected:
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


CxCore_MulSpectrumsTest::CxCore_MulSpectrumsTest() :
    CxCore_DXTBaseTest( "dxt-mulspectrums", "cvMulSpectrums", true, true, true )
{
    transform_type_list = mulsp_transforms;
}


void CxCore_MulSpectrumsTest::run_func()
{
    CvArr* dst = test_array[TEMP].size() > 0 && test_array[TEMP][0] ?
                 test_array[TEMP][0] : test_array[OUTPUT][0];
    CvArr *src1 = test_array[INPUT][0], *src2 = test_array[INPUT][1];

    if( inplace )
    {
        if( ts->get_current_test_info()->test_case_idx & 1 )
            src2 = dst;
        else
            src1 = dst;
    }

    cvMulSpectrums( src1, src2, dst, flags );
}


static void cvTsMulComplex( const CvMat* A, const CvMat* B, CvMat* C, int flags )
{
    int i, j, depth = CV_MAT_DEPTH(A->type), cols = A->cols*2;

    assert( CV_ARE_SIZES_EQ(A,B) && CV_ARE_SIZES_EQ(B,C) &&
            CV_ARE_TYPES_EQ(A,B) && CV_ARE_TYPES_EQ(B,C) &&
            CV_MAT_CN(A->type) == 2 && CV_MAT_DEPTH(A->type) >= CV_32F );

    for( i = 0; i < C->rows; i++ )
    {
        if( depth == CV_32F )
        {
            const float* a = (float*)(A->data.ptr + A->step*i);
            const float* b = (float*)(B->data.ptr + B->step*i);
            float* c = (float*)(C->data.ptr + C->step*i);

            if( !(flags & CV_DXT_MUL_CONJ) )
                for( j = 0; j < cols; j += 2 )
                {
                    double re = (double)a[j]*b[j] - (double)a[j+1]*b[j+1];
                    double im = (double)a[j+1]*b[j] + (double)a[j]*b[j+1];

                    c[j] = (float)re;
                    c[j+1] = (float)im;
                }
            else
                for( j = 0; j < cols; j += 2 )
                {
                    double re = (double)a[j]*b[j] + (double)a[j+1]*b[j+1];
                    double im = (double)a[j+1]*b[j] - (double)a[j]*b[j+1];

                    c[j] = (float)re;
                    c[j+1] = (float)im;
                }
        }
        else
        {
            const double* a = (double*)(A->data.ptr + A->step*i);
            const double* b = (double*)(B->data.ptr + B->step*i);
            double* c = (double*)(C->data.ptr + C->step*i);

            if( !(flags & CV_DXT_MUL_CONJ) )
                for( j = 0; j < cols; j += 2 )
                {
                    double re = a[j]*b[j] - a[j+1]*b[j+1];
                    double im = a[j+1]*b[j] + a[j]*b[j+1];

                    c[j] = re;
                    c[j+1] = im;
                }
            else
                for( j = 0; j < cols; j += 2 )
                {
                    double re = a[j]*b[j] + a[j+1]*b[j+1];
                    double im = a[j+1]*b[j] - a[j]*b[j+1];

                    c[j] = re;
                    c[j+1] = im;
                }
        }
    }
}


void CxCore_MulSpectrumsTest::prepare_to_validation( int /*test_case_idx*/ )
{
    CvMat* src1 = &test_mat[INPUT][0];
    CvMat* src2 = &test_mat[INPUT][1];
    CvMat* dst = &test_mat[OUTPUT][0];
    CvMat* dst0 = &test_mat[REF_OUTPUT][0];
    CvMat* temp = test_array[TEMP].size() > 0 && test_array[TEMP][0] ? &test_mat[TEMP][0] : 0;
    int cn = CV_MAT_CN(src1->type);

    if( cn == 1 )
    {
        cvTsConvertFromCCS( src1, src1, dst, flags );
        cvTsConvertFromCCS( src2, src2, dst0, flags );
        src1 = dst;
        src2 = dst0;
    }

    cvTsMulComplex( src1, src2, dst0, flags );
    if( cn == 1 )
    {
        assert( temp != 0 );
        cvTsConvertFromCCS( temp, temp, dst, flags );
    }
}


CxCore_MulSpectrumsTest mulspectrums_test;


/* End of file. */
