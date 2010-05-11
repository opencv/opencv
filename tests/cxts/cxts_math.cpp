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

#include "_cxts.h"

/****************************************************************************************\
*                                   Utility Functions                                    *
\****************************************************************************************/

const char* cvTsGetTypeName( int type )
{
    static const char* type_names[] = { "8u", "8s", "16u", "16s", "32s", "32f", "64f", "ptr" };
    return type_names[CV_MAT_DEPTH(type)];
}


int cvTsTypeByName( const char* name )
{
    int i;
    for( i = 0; i < CV_DEPTH_MAX; i++ )
        if( strcmp(name, cvTsGetTypeName(i)) == 0 )
            return i;
    return -1;
}


void cvTsRandUni( CvRNG* rng, CvMat* a, CvScalar param0, CvScalar param1 )
{
    int i, j, k, cn, ncols;
    CvScalar scale = param0;
    CvScalar delta = param1;
    double C = 1./(65536.*65536.);

    cn = CV_MAT_CN(a->type);
    ncols = a->cols*cn;

    for( k = 0; k < 4; k++ )
    {
        double s = scale.val[k] - delta.val[k];
        if( s >= 0 )
            scale.val[k] = s;
        else
        {
            delta.val[k] = scale.val[k];
            scale.val[k] = -s;
        }
        scale.val[k] *= C;
    }

    for( i = 0; i < a->rows; i++ )
    {
        uchar* data = a->data.ptr + i*a->step;

        switch( CV_MAT_DEPTH(a->type) )
        {
        case CV_8U:
            for( j = 0; j < ncols; j += cn )
                for( k = 0; k < cn; k++ )
                {
                    int val = cvFloor( cvTsRandInt(rng)*scale.val[k] + delta.val[k] );
                    ((uchar*)data)[j + k] = CV_CAST_8U(val);
                }
            break;
        case CV_8S:
            for( j = 0; j < ncols; j += cn )
                for( k = 0; k < cn; k++ )
                {
                    int val = cvFloor( cvTsRandInt(rng)*scale.val[k] + delta.val[k] );
                    ((schar*)data)[j + k] = CV_CAST_8S(val);
                }
            break;
        case CV_16U:
            for( j = 0; j < ncols; j += cn )
                for( k = 0; k < cn; k++ )
                {
                    int val = cvFloor( cvTsRandInt(rng)*scale.val[k] + delta.val[k] );
                    ((ushort*)data)[j + k] = CV_CAST_16U(val);
                }
            break;
        case CV_16S:
            for( j = 0; j < ncols; j += cn )
                for( k = 0; k < cn; k++ )
                {
                    int val = cvFloor( cvTsRandInt(rng)*scale.val[k] + delta.val[k] );
                    ((short*)data)[j + k] = CV_CAST_16S(val);
                }
            break;
        case CV_32S:
            for( j = 0; j < ncols; j += cn )
                for( k = 0; k < cn; k++ )
                {
                    int val = cvFloor( cvTsRandInt(rng)*scale.val[k] + delta.val[k] );
                    ((int*)data)[j + k] = val;
                }
            break;
        case CV_32F:
            for( j = 0; j < ncols; j += cn )
                for( k = 0; k < cn; k++ )
                {
                    double val = cvTsRandInt(rng)*scale.val[k] + delta.val[k];
                    ((float*)data)[j + k] = (float)val;
                }
            break;
        case CV_64F:
            for( j = 0; j < ncols; j += cn )
                for( k = 0; k < cn; k++ )
                {
                    double val = cvTsRandInt(rng);
                    val = (val + cvTsRandInt(rng)*C)*scale.val[k] + delta.val[k];
                    ((double*)data)[j + k] = val;
                }
            break;
        default:
            assert(0);
            return;
        }
    }
}


void cvTsZero( CvMat* c, const CvMat* mask )
{
    int i, j, elem_size = CV_ELEM_SIZE(c->type), width = c->cols;
    
    for( i = 0; i < c->rows; i++ )
    {
        if( !mask )
            memset( c->data.ptr + i*c->step, 0, width*elem_size );
        else
        {
            const uchar* mrow = mask->data.ptr + mask->step*i;
            uchar* cptr = c->data.ptr + c->step*i;
            for( j = 0; j < width; j++, cptr += elem_size )
                if( mrow[j] )
                    memset( cptr, 0, elem_size );
        }
    }
}


// initializes scaled identity matrix
void cvTsSetIdentity( CvMat* c, CvScalar diag_value )
{
    int i, width;
    cvTsZero( c );
    width = MIN(c->rows, c->cols);
    for( i = 0; i < width; i++ )
        cvSet2D( c, i, i, diag_value );
}


// copies selected region of one array to another array
void cvTsCopy( const CvMat* a, CvMat* b, const CvMat* mask )
{
    int i = 0, j = 0, k;
    int el_size, ncols;

    el_size = CV_ELEM_SIZE(a->type);
    ncols = a->cols;
    if( mask )
    {
        assert( CV_ARE_SIZES_EQ(a,mask) &&
            (CV_MAT_TYPE(mask->type) == CV_8UC1 ||
             CV_MAT_TYPE(mask->type) == CV_8SC1 ));
    }

    assert( CV_ARE_TYPES_EQ(a,b) && CV_ARE_SIZES_EQ(a,b) );

    if( !mask )
        ncols *= el_size;

    for( i = 0; i < a->rows; i++ )
    {
        uchar* a_data = a->data.ptr + a->step*i;
        uchar* b_data = b->data.ptr + b->step*i;

        if( !mask )
            memcpy( b_data, a_data, ncols );
        else
        {
            uchar* m_data = mask->data.ptr + mask->step*i;

            for( j = 0; j < ncols; j++, b_data += el_size, a_data += el_size )
            {
                if( m_data[j] )
                {
                    for( k = 0; k < el_size; k++ )
                        b_data[k] = a_data[k];
                }
            }
        }
    }
}


void cvTsConvert( const CvMat* a, CvMat* b )
{
    int i, j, ncols = b->cols*CV_MAT_CN(b->type);
    double* buf = 0;

    assert( CV_ARE_SIZES_EQ(a,b) && CV_ARE_CNS_EQ(a,b) );
    buf = (double*)cvStackAlloc(ncols*sizeof(buf[0]));

    for( i = 0; i < b->rows; i++ )
    {
        uchar* a_data = a->data.ptr + i*a->step;
        uchar* b_data = b->data.ptr + i*b->step;

        switch( CV_MAT_DEPTH(a->type) )
        {
        case CV_8U:
            for( j = 0; j < ncols; j++ )
                buf[j] = ((uchar*)a_data)[j];
            break;
        case CV_8S:
            for( j = 0; j < ncols; j++ )
                buf[j] = ((schar*)a_data)[j];
            break;
        case CV_16U:
            for( j = 0; j < ncols; j++ )
                buf[j] = ((ushort*)a_data)[j];
            break;
        case CV_16S:
            for( j = 0; j < ncols; j++ )
                buf[j] = ((short*)a_data)[j];
            break;
        case CV_32S:
            for( j = 0; j < ncols; j++ )
                buf[j] = ((int*)a_data)[j];
            break;
        case CV_32F:
            for( j = 0; j < ncols; j++ )
                buf[j] = ((float*)a_data)[j];
            break;
        case CV_64F:
            for( j = 0; j < ncols; j++ )
                buf[j] = ((double*)a_data)[j];
            break;
        default:
            assert(0);
            return;
        }

        switch( CV_MAT_DEPTH(b->type) )
        {
        case CV_8U:
            for( j = 0; j < ncols; j++ )
            {
                int val = cvRound(buf[j]);
                ((uchar*)b_data)[j] = CV_CAST_8U(val);
            }
            break;
        case CV_8S:
            for( j = 0; j < ncols; j++ )
            {
                int val = cvRound(buf[j]);
                ((schar*)b_data)[j] = CV_CAST_8S(val);
            }
            break;
        case CV_16U:
            for( j = 0; j < ncols; j++ )
            {
                int val = cvRound(buf[j]);
                ((ushort*)b_data)[j] = CV_CAST_16U(val);
            }
            break;
        case CV_16S:
            for( j = 0; j < ncols; j++ )
            {
                int val = cvRound(buf[j]);
                ((short*)b_data)[j] = CV_CAST_16S(val);
            }
            break;
        case CV_32S:
            for( j = 0; j < ncols; j++ )
            {
                int val = cvRound(buf[j]);
                ((int*)b_data)[j] = CV_CAST_32S(val);
            }
            break;
        case CV_32F:
            for( j = 0; j < ncols; j++ )
                ((float*)b_data)[j] = CV_CAST_32F(buf[j]);
            break;
        case CV_64F:
            for( j = 0; j < ncols; j++ )
                ((double*)b_data)[j] = CV_CAST_64F(buf[j]);
            break;
        default:
            assert(0);
        }
    }
}


// extracts a single channel from a multi-channel array
void cvTsExtract( const CvMat* a, CvMat* b, int coi )
{
    int i = 0, j = 0, k;
    int el_size, el_size1, ncols;

    el_size = CV_ELEM_SIZE(a->type);
    el_size1 = CV_ELEM_SIZE(b->type);
    ncols = a->cols;

    assert( CV_ARE_DEPTHS_EQ(a,b) && CV_ARE_SIZES_EQ(a,b) &&
            (unsigned)coi < (unsigned)CV_MAT_CN(a->type) &&
            CV_MAT_CN(b->type) == 1 );

    for( i = 0; i < a->rows; i++ )
    {
        uchar* a_data = a->data.ptr + a->step*i;
        uchar* b_data = b->data.ptr + b->step*i;
        a_data += el_size1*coi;
        for( j = 0; j < ncols; j++, b_data += el_size1, a_data += el_size )
        {
            for( k = 0; k < el_size1; k++ )
                b_data[k] = a_data[k];
        }
    }
}

// replaces a single channel in a multi-channel array
void cvTsInsert( const CvMat* a, CvMat* b, int coi )
{
    int i = 0, j = 0, k;
    int el_size, el_size1, ncols;

    el_size = CV_ELEM_SIZE(b->type);
    el_size1 = CV_ELEM_SIZE(a->type);
    ncols = a->cols;

    assert( CV_ARE_DEPTHS_EQ(a,b) && CV_ARE_SIZES_EQ(a,b) &&
            (unsigned)coi < (unsigned)CV_MAT_CN(b->type) &&
            CV_MAT_CN(a->type) == 1 );

    for( i = 0; i < a->rows; i++ )
    {
        uchar* a_data = a->data.ptr + a->step*i;
        uchar* b_data = b->data.ptr + b->step*i;
        b_data += el_size1*coi;
        for( j = 0; j < ncols; j++, b_data += el_size, a_data += el_size1 )
        {
            for( k = 0; k < el_size1; k++ )
                b_data[k] = a_data[k];
        }
    }
}


// c = alpha*a + beta*b + gamma
void cvTsAdd( const CvMat* a, CvScalar alpha, const CvMat* b, CvScalar beta,
              CvScalar gamma, CvMat* c, int calc_abs )
{
    int i, j, k, cn, ncols;
    double* buf = 0;
    double* alpha_buf = 0;
    double* beta_buf = 0;
    double* gamma_buf = 0;

    if( !c )
    {
        assert(0);
        return;
    }

    cn = CV_MAT_CN(c->type);
    ncols = c->cols;

    if( !a )
    {
        a = b;
        alpha = beta;
        b = 0;
        beta = cvScalar(0);
    }

    if( a )
    {
        assert( CV_ARE_SIZES_EQ(a,c) && CV_MAT_CN(a->type) == cn );
        buf = (double*)malloc( a->cols * cn * sizeof(buf[0]) );
        alpha_buf = (double*)malloc( a->cols * cn * sizeof(alpha_buf[0]) );
    }

    if( b )
    {
        assert( CV_ARE_SIZES_EQ(b,c) && CV_MAT_CN(b->type) == cn );
        beta_buf = (double*)malloc( b->cols * cn * sizeof(beta_buf[0]) );
    }

    ncols *= cn;
    gamma_buf = (double*)malloc( ncols * sizeof(gamma_buf[0]) );
    if( !buf )
        buf = gamma_buf;

    if( !a && !b && calc_abs )
    {
        for( k = 0; k < cn; k++ )
            gamma.val[k] = fabs(gamma.val[k]);
    }

    for( i = 0; i < 1 + (a != 0) + (b != 0); i++ )
    {
        double* scalar_buf = i == 0 ? gamma_buf : i == 1 ? alpha_buf : beta_buf;
        CvScalar scalar = i == 0 ? gamma : i == 1 ? alpha : beta;
        for( j = 0; j < ncols; j += cn )
            for( k = 0; k < cn; k++ )
                scalar_buf[j + k] = scalar.val[k];
    }

    for( i = 0; i < c->rows; i++ )
    {
        uchar* c_data = c->data.ptr + i*c->step;

        if( a )
        {
            uchar* a_data = a->data.ptr + i*a->step;

            switch( CV_MAT_DEPTH(a->type) )
            {
            case CV_8U:
                for( j = 0; j < ncols; j++ )
                    buf[j] = ((uchar*)a_data)[j]*alpha_buf[j] + gamma_buf[j];
                break;
            case CV_8S:
                for( j = 0; j < ncols; j++ )
                    buf[j] = ((schar*)a_data)[j]*alpha_buf[j] + gamma_buf[j];
                break;
            case CV_16U:
                for( j = 0; j < ncols; j++ )
                    buf[j] = ((ushort*)a_data)[j]*alpha_buf[j] + gamma_buf[j];
                break;
            case CV_16S:
                for( j = 0; j < ncols; j++ )
                    buf[j] = ((short*)a_data)[j]*alpha_buf[j] + gamma_buf[j];
                break;
            case CV_32S:
                for( j = 0; j < ncols; j++ )
                    buf[j] = ((int*)a_data)[j]*alpha_buf[j] + gamma_buf[j];
                break;
            case CV_32F:
                for( j = 0; j < ncols; j++ )
                    buf[j] = ((float*)a_data)[j]*alpha_buf[j] + gamma_buf[j];
                break;
            case CV_64F:
                for( j = 0; j < ncols; j++ )
                    buf[j] = ((double*)a_data)[j]*alpha_buf[j] + gamma_buf[j];
                break;
            default:
                assert(0);
                return;
            }
        }

        if( b )
        {
            uchar* b_data = b->data.ptr + i*b->step;

            switch( CV_MAT_DEPTH(b->type) )
            {
            case CV_8U:
                for( j = 0; j < ncols; j++ )
                    buf[j] += ((uchar*)b_data)[j]*beta_buf[j];
                break;
            case CV_8S:
                for( j = 0; j < ncols; j++ )
                    buf[j] += ((schar*)b_data)[j]*beta_buf[j];
                break;
            case CV_16U:
                for( j = 0; j < ncols; j++ )
                    buf[j] += ((ushort*)b_data)[j]*beta_buf[j];
                break;
            case CV_16S:
                for( j = 0; j < ncols; j++ )
                    buf[j] += ((short*)b_data)[j]*beta_buf[j];
                break;
            case CV_32S:
                for( j = 0; j < ncols; j++ )
                    buf[j] += ((int*)b_data)[j]*beta_buf[j];
                break;
            case CV_32F:
                for( j = 0; j < ncols; j++ )
                    buf[j] += ((float*)b_data)[j]*beta_buf[j];
                break;
            case CV_64F:
                for( j = 0; j < ncols; j++ )
                    buf[j] += ((double*)b_data)[j]*beta_buf[j];
                break;
            default:
                assert(0);
                return;
            }
        }

        if( a || b )
        {
            if( calc_abs )
            {
                for( j = 0; j < ncols; j++ )
                    buf[j] = fabs(buf[j]);
            }
        }
        else if( i > 0 )
        {
            memcpy( c_data, c_data - c->step, c->cols*CV_ELEM_SIZE(c->type) );
            continue;
        }

        switch( CV_MAT_DEPTH(c->type) )
        {
        case CV_8U:
            for( j = 0; j < ncols; j++ )
            {
                int val = cvRound(buf[j]);
                ((uchar*)c_data)[j] = CV_CAST_8U(val);
            }
            break;
        case CV_8S:
            for( j = 0; j < ncols; j++ )
            {
                int val = cvRound(buf[j]);
                ((schar*)c_data)[j] = CV_CAST_8S(val);
            }
            break;
        case CV_16U:
            for( j = 0; j < ncols; j++ )
            {
                int val = cvRound(buf[j]);
                ((ushort*)c_data)[j] = CV_CAST_16U(val);
            }
            break;
        case CV_16S:
            for( j = 0; j < ncols; j++ )
            {
                int val = cvRound(buf[j]);
                ((short*)c_data)[j] = CV_CAST_16S(val);
            }
            break;
        case CV_32S:
            for( j = 0; j < ncols; j++ )
            {
                int val = cvRound(buf[j]);
                ((int*)c_data)[j] = CV_CAST_32S(val);
            }
            break;
        case CV_32F:
            for( j = 0; j < ncols; j++ )
                ((float*)c_data)[j] = CV_CAST_32F(buf[j]);
            break;
        case CV_64F:
            for( j = 0; j < ncols; j++ )
                ((double*)c_data)[j] = CV_CAST_64F(buf[j]);
            break;
        default:
            assert(0);
        }
    }

    if( buf && buf != gamma_buf )
        free( buf );
    if( gamma_buf )
        free( gamma_buf );
    if( alpha_buf )
        free( alpha_buf );
    if( beta_buf )
        free( beta_buf );
}


// c = a*b*alpha
void cvTsMul( const CvMat* a, const CvMat* b, CvScalar alpha, CvMat* c )
{
    int i, j, k, cn, ncols;
    double* buf = 0;
    double* alpha_buf = 0;

    if( !a || !b || !c )
    {
        assert(0);
        return;
    }

    assert( CV_ARE_SIZES_EQ(a,c) && CV_ARE_SIZES_EQ(b,c) &&
            CV_ARE_TYPES_EQ(a,b) && CV_ARE_CNS_EQ(a,c) );

    cn = CV_MAT_CN(c->type);
    ncols = c->cols * cn;
    alpha_buf = (double*)malloc( ncols * sizeof(alpha_buf[0]) );
    buf = (double*)malloc( ncols * sizeof(buf[0]) );

    for( j = 0; j < ncols; j += cn )
        for( k = 0; k < cn; k++ )
            alpha_buf[j + k] = alpha.val[k];

    for( i = 0; i < c->rows; i++ )
    {
        uchar* c_data = c->data.ptr + i*c->step;
        uchar* a_data = a->data.ptr + i*a->step;
        uchar* b_data = b->data.ptr + i*b->step;

        switch( CV_MAT_DEPTH(a->type) )
        {
        case CV_8U:
            for( j = 0; j < ncols; j++ )
                buf[j] = (alpha_buf[j]*((uchar*)a_data)[j])*((uchar*)b_data)[j];
            break;
        case CV_8S:
            for( j = 0; j < ncols; j++ )
                buf[j] = (alpha_buf[j]*((schar*)a_data)[j])*((schar*)b_data)[j];
            break;
        case CV_16U:
            for( j = 0; j < ncols; j++ )
                buf[j] = (alpha_buf[j]*((ushort*)a_data)[j])*((ushort*)b_data)[j];
            break;
        case CV_16S:
            for( j = 0; j < ncols; j++ )
                buf[j] = (alpha_buf[j]*((short*)a_data)[j])*((short*)b_data)[j];
            break;
        case CV_32S:
            for( j = 0; j < ncols; j++ )
                buf[j] = (alpha_buf[j]*((int*)a_data)[j])*((int*)b_data)[j];
            break;
        case CV_32F:
            for( j = 0; j < ncols; j++ )
                buf[j] = (alpha_buf[j]*((float*)a_data)[j])*((float*)b_data)[j];
            break;
        case CV_64F:
            for( j = 0; j < ncols; j++ )
                buf[j] = (alpha_buf[j]*((double*)a_data)[j])*((double*)b_data)[j];
            break;
        default:
            assert(0);
            return;
        }

        switch( CV_MAT_DEPTH(c->type) )
        {
        case CV_8U:
            for( j = 0; j < ncols; j++ )
            {
                int val = cvRound(buf[j]);
                ((uchar*)c_data)[j] = CV_CAST_8U(val);
            }
            break;
        case CV_8S:
            for( j = 0; j < ncols; j++ )
            {
                int val = cvRound(buf[j]);
                ((schar*)c_data)[j] = CV_CAST_8S(val);
            }
            break;
        case CV_16U:
            for( j = 0; j < ncols; j++ )
            {
                int val = cvRound(buf[j]);
                ((ushort*)c_data)[j] = CV_CAST_16U(val);
            }
            break;
        case CV_16S:
            for( j = 0; j < ncols; j++ )
            {
                int val = cvRound(buf[j]);
                ((short*)c_data)[j] = CV_CAST_16S(val);
            }
            break;
        case CV_32S:
            for( j = 0; j < ncols; j++ )
            {
                int val = cvRound(buf[j]);
                ((int*)c_data)[j] = CV_CAST_32S(val);
            }
            break;
        case CV_32F:
            for( j = 0; j < ncols; j++ )
                ((float*)c_data)[j] = CV_CAST_32F(buf[j]);
            break;
        case CV_64F:
            for( j = 0; j < ncols; j++ )
                ((double*)c_data)[j] = CV_CAST_64F(buf[j]);
            break;
        default:
            assert(0);
        }
    }

    if( alpha_buf )
        free( alpha_buf );

    if( buf )
        free( buf );
}


// c = a*alpha/b
void cvTsDiv( const CvMat* a, const CvMat* b, CvScalar alpha, CvMat* c )
{
    int i, j, k, cn, ncols;
    double* buf = 0;
    double* alpha_buf = 0;

    if( !b || !c )
    {
        assert(0);
        return;
    }

    if( a )
    {
        assert( CV_ARE_SIZES_EQ(a,c) &&
                CV_ARE_TYPES_EQ(a,b) && CV_ARE_CNS_EQ(a,c) );
    }

    assert( CV_ARE_SIZES_EQ(b,c) && CV_ARE_CNS_EQ(b,c) );

    cn = CV_MAT_CN(c->type);
    ncols = c->cols * cn;
    alpha_buf = (double*)malloc( ncols * sizeof(alpha_buf[0]) );
    buf = (double*)malloc( ncols * sizeof(buf[0]) );

    for( j = 0; j < ncols; j += cn )
        for( k = 0; k < cn; k++ )
            alpha_buf[j + k] = alpha.val[k];

    for( i = 0; i < c->rows; i++ )
    {
        uchar* c_data = c->data.ptr + i*c->step;
        uchar* a_data = a ? a->data.ptr + i*a->step : 0;
        uchar* b_data = b->data.ptr + i*b->step;

        switch( CV_MAT_DEPTH(b->type) )
        {
        case CV_8U:
            for( j = 0; j < ncols; j++ )
            {
                int denom = ((uchar*)b_data)[j];
                int num = a_data ? ((uchar*)a_data)[j] : 1;
                buf[j] = !denom ? 0 : (alpha_buf[j]*num/denom);
            }
            break;
        case CV_8S:
            for( j = 0; j < ncols; j++ )
            {
                int denom = ((schar*)b_data)[j];
                int num = a_data ? ((schar*)a_data)[j] : 1;
                buf[j] = !denom ? 0 : (alpha_buf[j]*num/denom);
            }
            break;
        case CV_16U:
            for( j = 0; j < ncols; j++ )
            {
                int denom = ((ushort*)b_data)[j];
                int num = a_data ? ((ushort*)a_data)[j] : 1;
                buf[j] = !denom ? 0 : (alpha_buf[j]*num/denom);
            }
            break;
        case CV_16S:
            for( j = 0; j < ncols; j++ )
            {
                int denom = ((short*)b_data)[j];
                int num = a_data ? ((short*)a_data)[j] : 1;
                buf[j] = !denom ? 0 : (alpha_buf[j]*num/denom);
            }
            break;
        case CV_32S:
            for( j = 0; j < ncols; j++ )
            {
                int denom = ((int*)b_data)[j];
                int num = a_data ? ((int*)a_data)[j] : 1;
                buf[j] = !denom ? 0 : (alpha_buf[j]*num/denom);
            }
            break;
        case CV_32F:
            for( j = 0; j < ncols; j++ )
            {
                double denom = ((float*)b_data)[j];
                double num = a_data ? ((float*)a_data)[j] : 1;
                buf[j] = !denom ? 0 : (alpha_buf[j]*num/denom);
            }
            break;
        case CV_64F:
            for( j = 0; j < ncols; j++ )
            {
                double denom = ((double*)b_data)[j];
                double num = a_data ? ((double*)a_data)[j] : 1;
                buf[j] = !denom ? 0 : (alpha_buf[j]*num/denom);
            }
            break;
        default:
            assert(0);
            return;
        }

        switch( CV_MAT_DEPTH(c->type) )
        {
        case CV_8U:
            for( j = 0; j < ncols; j++ )
            {
                int val = cvRound(buf[j]);
                ((uchar*)c_data)[j] = CV_CAST_8U(val);
            }
            break;
        case CV_8S:
            for( j = 0; j < ncols; j++ )
            {
                int val = cvRound(buf[j]);
                ((schar*)c_data)[j] = CV_CAST_8S(val);
            }
            break;
        case CV_16U:
            for( j = 0; j < ncols; j++ )
            {
                int val = cvRound(buf[j]);
                ((ushort*)c_data)[j] = CV_CAST_16U(val);
            }
            break;
        case CV_16S:
            for( j = 0; j < ncols; j++ )
            {
                int val = cvRound(buf[j]);
                ((short*)c_data)[j] = CV_CAST_16S(val);
            }
            break;
        case CV_32S:
            for( j = 0; j < ncols; j++ )
            {
                int val = cvRound(buf[j]);
                ((int*)c_data)[j] = CV_CAST_32S(val);
            }
            break;
        case CV_32F:
            for( j = 0; j < ncols; j++ )
                ((float*)c_data)[j] = CV_CAST_32F(buf[j]);
            break;
        case CV_64F:
            for( j = 0; j < ncols; j++ )
                ((double*)c_data)[j] = CV_CAST_64F(buf[j]);
            break;
        default:
            assert(0);
        }
    }

    if( alpha_buf )
        free( alpha_buf );

    if( buf )
        free( buf );
}


// c = min(a,b) or c = max(a,b)
void cvTsMinMax( const CvMat* a, const CvMat* b, CvMat* c, int op_type )
{
    int i, j, ncols;
    int calc_max = op_type == CV_TS_MAX;

    if( !a || !b || !c )
    {
        assert(0);
        return;
    }

    assert( CV_ARE_SIZES_EQ(a,c) && CV_ARE_TYPES_EQ(a,c) &&
            CV_ARE_SIZES_EQ(b,c) && CV_ARE_TYPES_EQ(b,c) &&
            CV_MAT_CN(a->type) == 1 );
    ncols = c->cols;

    for( i = 0; i < c->rows; i++ )
    {
        uchar* c_data = c->data.ptr + i*c->step;
        uchar* a_data = a->data.ptr + i*a->step;
        uchar* b_data = b->data.ptr + i*b->step;

        switch( CV_MAT_DEPTH(a->type) )
        {
        case CV_8U:
            for( j = 0; j < ncols; j++ )
            {
                int aj = ((uchar*)a_data)[j];
                int bj = ((uchar*)b_data)[j];
                ((uchar*)c_data)[j] = (uchar)(calc_max ? MAX(aj, bj) : MIN(aj,bj));
            }
            break;
        case CV_8S:
            for( j = 0; j < ncols; j++ )
            {
                int aj = ((schar*)a_data)[j];
                int bj = ((schar*)b_data)[j];
                ((schar*)c_data)[j] = (schar)(calc_max ? MAX(aj, bj) : MIN(aj,bj));
            }
            break;
        case CV_16U:
            for( j = 0; j < ncols; j++ )
            {
                int aj = ((ushort*)a_data)[j];
                int bj = ((ushort*)b_data)[j];
                ((ushort*)c_data)[j] = (ushort)(calc_max ? MAX(aj, bj) : MIN(aj,bj));
            }
            break;
        case CV_16S:
            for( j = 0; j < ncols; j++ )
            {
                int aj = ((short*)a_data)[j];
                int bj = ((short*)b_data)[j];
                ((short*)c_data)[j] = (short)(calc_max ? MAX(aj, bj) : MIN(aj,bj));
            }
            break;
        case CV_32S:
            for( j = 0; j < ncols; j++ )
            {
                int aj = ((int*)a_data)[j];
                int bj = ((int*)b_data)[j];
                ((int*)c_data)[j] = calc_max ? MAX(aj, bj) : MIN(aj,bj);
            }
            break;
        case CV_32F:
            for( j = 0; j < ncols; j++ )
            {
                float aj = ((float*)a_data)[j];
                float bj = ((float*)b_data)[j];
                ((float*)c_data)[j] = calc_max ? MAX(aj, bj) : MIN(aj,bj);
            }
            break;
        case CV_64F:
            for( j = 0; j < ncols; j++ )
            {
                double aj = ((double*)a_data)[j];
                double bj = ((double*)b_data)[j];
                ((double*)c_data)[j] = calc_max ? MAX(aj, bj) : MIN(aj,bj);
            }
            break;
        default:
            assert(0);
            return;
        }
    }
}

// c = min(a,b) or c = max(a,b)
void cvTsMinMaxS( const CvMat* a, double s, CvMat* c, int op_type )
{
    int i, j, ncols;
    int calc_max = op_type == CV_TS_MAX;
    float fs = (float)s;
    int is = cvRound(s);

    if( !a || !c )
    {
        assert(0);
        return;
    }

    assert( CV_ARE_SIZES_EQ(a,c) && CV_ARE_TYPES_EQ(a,c) &&
            CV_MAT_CN(a->type) == 1 );
    ncols = c->cols;

    switch( CV_MAT_DEPTH(a->type) )
    {
    case CV_8U:
        is = CV_CAST_8U(is);
        break;
    case CV_8S:
        is = CV_CAST_8S(is);
        break;
    case CV_16U:
        is = CV_CAST_16U(is);
        break;
    case CV_16S:
        is = CV_CAST_16S(is);
        break;
    default:
        ;
    }

    for( i = 0; i < c->rows; i++ )
    {
        uchar* c_data = c->data.ptr + i*c->step;
        uchar* a_data = a->data.ptr + i*a->step;

        switch( CV_MAT_DEPTH(a->type) )
        {
        case CV_8U:
            for( j = 0; j < ncols; j++ )
            {
                int aj = ((uchar*)a_data)[j];
                ((uchar*)c_data)[j] = (uchar)(calc_max ? MAX(aj, is) : MIN(aj, is));
            }
            break;
        case CV_8S:
            for( j = 0; j < ncols; j++ )
            {
                int aj = ((schar*)a_data)[j];
                ((schar*)c_data)[j] = (schar)(calc_max ? MAX(aj, is) : MIN(aj, is));
            }
            break;
        case CV_16U:
            for( j = 0; j < ncols; j++ )
            {
                int aj = ((ushort*)a_data)[j];
                ((ushort*)c_data)[j] = (ushort)(calc_max ? MAX(aj, is) : MIN(aj, is));
            }
            break;
        case CV_16S:
            for( j = 0; j < ncols; j++ )
            {
                int aj = ((short*)a_data)[j];
                ((short*)c_data)[j] = (short)(calc_max ? MAX(aj, is) : MIN(aj, is));
            }
            break;
        case CV_32S:
            for( j = 0; j < ncols; j++ )
            {
                int aj = ((int*)a_data)[j];
                ((int*)c_data)[j] = calc_max ? MAX(aj, is) : MIN(aj, is);
            }
            break;
        case CV_32F:
            for( j = 0; j < ncols; j++ )
            {
                float aj = ((float*)a_data)[j];
                ((float*)c_data)[j] = calc_max ? MAX(aj, fs) : MIN(aj, fs);
            }
            break;
        case CV_64F:
            for( j = 0; j < ncols; j++ )
            {
                double aj = ((double*)a_data)[j];
                ((double*)c_data)[j] = calc_max ? MAX(aj, s) : MIN(aj, s);
            }
            break;
        default:
            assert(0);
            return;
        }
    }
}

// checks that the array does not have NaNs and/or Infs and all the elements are
// within [min_val,max_val). idx is the index of the first "bad" element.
int cvTsCheck( const CvMat* a, double min_val, double max_val, CvPoint* idx )
{
    int i = 0, j = 0;
    int cn, ncols;
    int imin = 0, imax = 0;
    cn = CV_MAT_CN(a->type);
    ncols = a->cols*cn;

    if( CV_MAT_DEPTH(a->type) <= CV_32S )
    {
        imin = cvCeil(min_val);
        imax = cvFloor(max_val);
    }

    for( i = 0; i < a->rows; i++ )
    {
        uchar* data = a->data.ptr + a->step*i;

        switch( CV_MAT_DEPTH(a->type) )
        {
        case CV_8U:
            for( j = 0; j < ncols; j++ )
            {
                int val = ((uchar*)data)[j];
                if( val < imin || imax < val )
                    goto _exit_;
            }
            break;
        case CV_8S:
            for( j = 0; j < ncols; j++ )
            {
                int val = ((schar*)data)[j];
                if( val < imin || imax < val )
                    goto _exit_;
            }
            break;
        case CV_16U:
            for( j = 0; j < ncols; j++ )
            {
                int val = ((ushort*)data)[j];
                if( val < imin || imax < val )
                    goto _exit_;
            }
            break;
        case CV_16S:
            for( j = 0; j < ncols; j++ )
            {
                int val = ((short*)data)[j];
                if( val < imin || imax < val )
                    goto _exit_;
            }
            break;
        case CV_32S:
            for( j = 0; j < ncols; j++ )
            {
                int val = ((int*)data)[j];
                if( val < imin || imax < val )
                    goto _exit_;
            }
            break;
        case CV_32F:
            for( j = 0; j < ncols; j++ )
            {
                double val = ((float*)data)[j];
                if( cvIsNaN(val) || cvIsInf(val) || val < min_val || max_val < val )
                    goto _exit_;
            }
            break;
        case CV_64F:
            for( j = 0; j < ncols; j++ )
            {
                double val = ((double*)data)[j];
                if( cvIsNaN(val) || cvIsInf(val) || val < min_val || max_val < val )
                    goto _exit_;
            }
            break;
        default:
            assert(0);
            return -1;
        }
    }

    return 0;
_exit_:

    idx->x = j;
    idx->y = i;
    return -1;
}

// compares two arrays. max_diff is the maximum actual difference,
// success_err_level is maximum allowed difference, idx is the index of the first
// element for which difference is >success_err_level
// (or index of element with the maximum difference)
int cvTsCmpEps( const CvMat* check_arr, const CvMat* etalon, double* _max_diff,
                double success_err_level, CvPoint* _idx, bool element_wise_relative_error )
{
    int i = 0, j = 0;
    int cn, ncols;
    double maxdiff = 0;
    double maxval = 0;
    int imaxdiff = 0;
    int ilevel = 0;
    int result = -1;
    CvPoint stub, *idx = _idx ? _idx : &stub;

    cn = CV_MAT_CN(check_arr->type);
    ncols = check_arr->cols*cn;

    *idx = cvPoint(0,0);

    assert( CV_ARE_TYPES_EQ(check_arr,etalon) && CV_ARE_SIZES_EQ(check_arr,etalon) );

    if( CV_MAT_DEPTH(check_arr->type) < CV_32S )
        ilevel = cvFloor(success_err_level);

    if( CV_MAT_DEPTH(check_arr->type) >= CV_32F && !element_wise_relative_error )
    {
        double maxval0 = 1.;
        maxval = cvTsNorm( etalon, 0, CV_C, 0 );
        maxval = MAX(maxval, maxval0);
    }

    for( i = 0; i < check_arr->rows; i++ )
    {
        uchar* a_data = check_arr->data.ptr + check_arr->step*i;
        uchar* b_data = etalon->data.ptr + etalon->step*i;

        switch( CV_MAT_DEPTH(check_arr->type) )
        {
        case CV_8U:
            for( j = 0; j < ncols; j++ )
            {
                int val = abs(((uchar*)a_data)[j] - ((uchar*)b_data)[j]);
                if( val > imaxdiff )
                {
                    imaxdiff = val;
                    *idx = cvPoint(j,i);
                    if( val > ilevel )
                        goto _exit_;
                }
            }
            break;
        case CV_8S:
            for( j = 0; j < ncols; j++ )
            {
                int val = abs(((schar*)a_data)[j] - ((schar*)b_data)[j]);
                if( val > imaxdiff )
                {
                    imaxdiff = val;
                    *idx = cvPoint(j,i);
                    if( val > ilevel )
                        goto _exit_;
                }
            }
            break;
        case CV_16U:
            for( j = 0; j < ncols; j++ )
            {
                int val = abs(((ushort*)a_data)[j] - ((ushort*)b_data)[j]);
                if( val > imaxdiff )
                {
                    imaxdiff = val;
                    *idx = cvPoint(j,i);
                    if( val > ilevel )
                        goto _exit_;
                }
            }
            break;
        case CV_16S:
            for( j = 0; j < ncols; j++ )
            {
                int val = abs(((short*)a_data)[j] - ((short*)b_data)[j]);
                if( val > imaxdiff )
                {
                    imaxdiff = val;
                    *idx = cvPoint(j,i);
                    if( val > ilevel )
                        goto _exit_;
                }
            }
            break;
        case CV_32S:
            for( j = 0; j < ncols; j++ )
            {
                double val = fabs((double)((int*)a_data)[j] - (double)((int*)b_data)[j]);
                if( val > maxdiff )
                {
                    maxdiff = val;
                    *idx = cvPoint(j,i);
                    if( val > success_err_level )
                        goto _exit_;
                }
            }
            break;
        case CV_32F:
            for( j = 0; j < ncols; j++ )
            {
                double a_val = ((float*)a_data)[j];
                double b_val = ((float*)b_data)[j];
                double threshold;
                if( ((int*)a_data)[j] == ((int*)b_data)[j] )
                    continue;
                if( cvIsNaN(a_val) || cvIsInf(a_val) )
                {
                    result = -2;
                    *idx = cvPoint(j,i);
                    goto _exit_;
                }
                if( cvIsNaN(b_val) || cvIsInf(b_val) )
                {
                    result = -3;
                    *idx = cvPoint(j,i);
                    goto _exit_;
                }
                a_val = fabs(a_val - b_val);
                threshold = element_wise_relative_error ? fabs(b_val) + 1 : maxval;
                if( a_val > threshold*success_err_level )
                {
                    maxdiff = a_val/threshold;
                    *idx = cvPoint(j,i);
                    goto _exit_;
                }
            }
            break;
        case CV_64F:
            for( j = 0; j < ncols; j++ )
            {
                double a_val = ((double*)a_data)[j];
                double b_val = ((double*)b_data)[j];
                double threshold;
                if( ((int64*)a_data)[j] == ((int64*)b_data)[j] )
                    continue;
                if( cvIsNaN(a_val) || cvIsInf(a_val) )
                {
                    result = -2;
                    *idx = cvPoint(j,i);
                    goto _exit_;
                }
                if( cvIsNaN(b_val) || cvIsInf(b_val) )
                {
                    result = -3;
                    *idx = cvPoint(j,i);
                    goto _exit_;
                }
                a_val = fabs(a_val - b_val);
                threshold = element_wise_relative_error ? fabs(b_val) + 1 : maxval;
                if( a_val > threshold*success_err_level )
                {
                    maxdiff = a_val/threshold;
                    *idx = cvPoint(j,i);
                    goto _exit_;
                }
            }
            break;
        default:
            assert(0);
            return -1;
        }
    }

    result = 0;
_exit_:

    if( CV_MAT_DEPTH(check_arr->type) < CV_32S )
        maxdiff = imaxdiff;

    if( result < -1 )
        maxdiff = exp(1000.);
    *_max_diff = maxdiff;
    return result;
}


int cvTsCmpEps2( CvTS* ts, const CvArr* _a, const CvArr* _b, double success_err_level,
                 bool element_wise_relative_error, const char* desc )
{
    char msg[100];
    double diff = 0;
    CvMat astub, bstub, *a, *b;
    CvPoint idx = {0,0};
    int code;

    a = cvGetMat( _a, &astub );
    b = cvGetMat( _b, &bstub );
    code = cvTsCmpEps( a, b, &diff, success_err_level, &idx,
        element_wise_relative_error );

    switch( code )
    {
    case -1:
        sprintf( msg, "%s: Too big difference (=%g)", desc, diff );
        code = CvTS::FAIL_BAD_ACCURACY;
        break;
    case -2:
        sprintf( msg, "%s: Invalid output", desc );
        code = CvTS::FAIL_INVALID_OUTPUT;
        break;
    case -3:
        sprintf( msg, "%s: Invalid reference output", desc );
        code = CvTS::FAIL_INVALID_OUTPUT;
        break;
    default:
        ;
    }

    if( code < 0 )
    {
        if( a->rows == 1 && a->cols == 1 )
        {
            assert( idx.x == 0 && idx.y == 0 );
            ts->printf( CvTS::LOG, "%s\n", msg );
        }
        else if( a->rows == 1 || a->cols == 1 )
        {
            assert( idx.x == 0 || idx.y == 0 );
            ts->printf( CvTS::LOG, "%s at element %d\n", msg, idx.x + idx.y );
        }
        else
            ts->printf( CvTS::LOG, "%s at (%d,%d)\n", msg, idx.x, idx.y );
    }

    return code;
}


int cvTsCmpEps2_64f( CvTS* ts, const double* val, const double* ref_val, int len,
                     double eps, const char* param_name )
{
    CvMat _val = cvMat( 1, len, CV_64F, (void*)val );
    CvMat _ref_val = cvMat( 1, len, CV_64F, (void*)ref_val );

    return cvTsCmpEps2( ts, &_val, &_ref_val, eps, true, param_name );
}

// compares two arrays. the result is 8s image that takes values -1, 0, 1
void cvTsCmp( const CvMat* a, const CvMat* b, CvMat* result, int cmp_op )
{
    int i = 0, j = 0, ncols;
    ncols = a->cols;

    assert( CV_ARE_TYPES_EQ(a,b) && CV_ARE_SIZES_EQ(a,b) && CV_MAT_CN(a->type) == 1 );
    assert( CV_ARE_SIZES_EQ(a,result) &&
            (CV_MAT_TYPE(result->type) == CV_8UC1 ||
             CV_MAT_TYPE(result->type) == CV_8SC1 ));

    for( i = 0; i < a->rows; i++ )
    {
        uchar* a_data = a->data.ptr + a->step*i;
        uchar* b_data = b->data.ptr + b->step*i;
        schar* r_data = (schar*)(result->data.ptr + result->step*i);

        switch( CV_MAT_DEPTH(a->type) )
        {
        case CV_8U:
            for( j = 0; j < ncols; j++ )
            {
                int a_val = ((uchar*)a_data)[j];
                int b_val = ((uchar*)b_data)[j];
                r_data[j] = (schar)CV_CMP(a_val,b_val);
            }
            break;
        case CV_8S:
            for( j = 0; j < ncols; j++ )
            {
                int a_val = ((schar*)a_data)[j];
                int b_val = ((schar*)b_data)[j];
                r_data[j] = (schar)CV_CMP(a_val,b_val);
            }
            break;
        case CV_16U:
            for( j = 0; j < ncols; j++ )
            {
                int a_val = ((ushort*)a_data)[j];
                int b_val = ((ushort*)b_data)[j];
                r_data[j] = (schar)CV_CMP(a_val,b_val);
            }
            break;
        case CV_16S:
            for( j = 0; j < ncols; j++ )
            {
                int a_val = ((short*)a_data)[j];
                int b_val = ((short*)b_data)[j];
                r_data[j] = (schar)CV_CMP(a_val,b_val);
            }
            break;
        case CV_32S:
            for( j = 0; j < ncols; j++ )
            {
                int a_val = ((int*)a_data)[j];
                int b_val = ((int*)b_data)[j];
                r_data[j] = (schar)CV_CMP(a_val,b_val);
            }
            break;
        case CV_32F:
            for( j = 0; j < ncols; j++ )
            {
                float a_val = ((float*)a_data)[j];
                float b_val = ((float*)b_data)[j];
                r_data[j] = (schar)CV_CMP(a_val,b_val);
            }
            break;
        case CV_64F:
            for( j = 0; j < ncols; j++ )
            {
                double a_val = ((double*)a_data)[j];
                double b_val = ((double*)b_data)[j];
                r_data[j] = (schar)CV_CMP(a_val,b_val);
            }
            break;
        default:
            assert(0);
        }

        switch( cmp_op )
        {
        case CV_CMP_EQ:
            for( j = 0; j < ncols; j++ )
                r_data[j] = (schar)(r_data[j] == 0 ? -1 : 0);
            break;
        case CV_CMP_NE:
            for( j = 0; j < ncols; j++ )
                r_data[j] = (schar)(r_data[j] != 0 ? -1 : 0);
            break;
        case CV_CMP_LT:
            for( j = 0; j < ncols; j++ )
                r_data[j] = (schar)(r_data[j] < 0 ? -1 : 0);
            break;
        case CV_CMP_LE:
            for( j = 0; j < ncols; j++ )
                r_data[j] = (schar)(r_data[j] <= 0 ? -1 : 0);
            break;
        case CV_CMP_GE:
            for( j = 0; j < ncols; j++ )
                r_data[j] = (schar)(r_data[j] >= 0 ? -1 : 0);
            break;
        case CV_CMP_GT:
            for( j = 0; j < ncols; j++ )
                r_data[j] = (schar)(r_data[j] > 0 ? -1 : 0);
            break;
        default:
            ;
        }
    }
}

// compares two arrays. the result is 8s image that takes values -1, 0, 1
void cvTsCmpS( const CvMat* a, double fval, CvMat* result, int cmp_op )
{
    int i = 0, j = 0;
    int ncols, ival = 0;
    ncols = a->cols;

    if( CV_MAT_DEPTH(a->type) <= CV_32S )
        ival = cvRound(fval);

    assert( CV_MAT_CN(a->type) == 1 );
    assert( CV_ARE_SIZES_EQ(a,result) &&
            (CV_MAT_TYPE(result->type) == CV_8UC1 ||
             CV_MAT_TYPE(result->type) == CV_8SC1 ));

    for( i = 0; i < a->rows; i++ )
    {
        uchar* a_data = a->data.ptr + a->step*i;
        schar* r_data = (schar*)(result->data.ptr + result->step*i);

        switch( CV_MAT_DEPTH(a->type) )
        {
        case CV_8U:
            for( j = 0; j < ncols; j++ )
            {
                int a_val = ((uchar*)a_data)[j];
                r_data[j] = (schar)CV_CMP(a_val,ival);
            }
            break;
        case CV_8S:
            for( j = 0; j < ncols; j++ )
            {
                int a_val = ((schar*)a_data)[j];
                r_data[j] = (schar)CV_CMP(a_val,ival);
            }
            break;
        case CV_16U:
            for( j = 0; j < ncols; j++ )
            {
                int a_val = ((ushort*)a_data)[j];
                r_data[j] = (schar)CV_CMP(a_val,ival);
            }
            break;
        case CV_16S:
            for( j = 0; j < ncols; j++ )
            {
                int a_val = ((short*)a_data)[j];
                r_data[j] = (schar)CV_CMP(a_val,ival);
            }
            break;
        case CV_32S:
            for( j = 0; j < ncols; j++ )
            {
                int a_val = ((int*)a_data)[j];
                r_data[j] = (schar)CV_CMP(a_val,ival);
            }
            break;
        case CV_32F:
            for( j = 0; j < ncols; j++ )
            {
                float a_val = ((float*)a_data)[j];
                r_data[j] = (schar)CV_CMP(a_val,fval);
            }
            break;
        case CV_64F:
            for( j = 0; j < ncols; j++ )
            {
                double a_val = ((double*)a_data)[j];
                r_data[j] = (schar)CV_CMP(a_val,fval);
            }
            break;
        default:
            assert(0);
        }

        switch( cmp_op )
        {
        case CV_CMP_EQ:
            for( j = 0; j < ncols; j++ )
                r_data[j] = (schar)(r_data[j] == 0 ? -1 : 0);
            break;
        case CV_CMP_NE:
            for( j = 0; j < ncols; j++ )
                r_data[j] = (schar)(r_data[j] != 0 ? -1 : 0);
            break;
        case CV_CMP_LT:
            for( j = 0; j < ncols; j++ )
                r_data[j] = (schar)(r_data[j] < 0 ? -1 : 0);
            break;
        case CV_CMP_LE:
            for( j = 0; j < ncols; j++ )
                r_data[j] = (schar)(r_data[j] <= 0 ? -1 : 0);
            break;
        case CV_CMP_GE:
            for( j = 0; j < ncols; j++ )
                r_data[j] = (schar)(r_data[j] >= 0 ? -1 : 0);
            break;
        case CV_CMP_GT:
            for( j = 0; j < ncols; j++ )
                r_data[j] = (schar)(r_data[j] > 0 ? -1 : 0);
            break;
        default:
            ;
        }
    }
}


// calculates norm of a matrix
double cvTsNorm( const CvMat* arr, const CvMat* mask, int norm_type, int coi )
{
    int i = 0, j = 0, k;
    int depth, cn0, cn, ncols, el_size1;
    int inorm = 0;
    double fnorm = 0;
    void* buffer = 0;
    uchar* zerobuf = 0;

    cn0 = cn = CV_MAT_CN(arr->type);
    ncols = arr->cols*cn;
    depth = CV_MAT_DEPTH(arr->type);
    el_size1 = CV_ELEM_SIZE(depth);
    zerobuf = (uchar*)cvStackAlloc(el_size1*cn);
    memset( zerobuf, 0, el_size1*cn);

    if( mask )
    {
        assert( CV_ARE_SIZES_EQ( arr, mask ) && CV_IS_MASK_ARR(mask) );
        buffer = cvStackAlloc( el_size1*ncols );
    }

    if( coi == 0 )
        cn = 1;

    for( i = 0; i < arr->rows; i++ )
    {
        const uchar* data = arr->data.ptr + arr->step*i + (coi - (coi != 0))*el_size1;

        if( mask )
        {
            const uchar* mdata = mask->data.ptr + mask->step*i;
            switch( depth )
            {
            case CV_8U:
            case CV_8S:
                for( j = 0; j < ncols; j += cn0 )
                {
                    const uchar* src = *mdata++ ? (uchar*)data + j : zerobuf;
                    for( k = 0; k < cn0; k++ )
                        ((uchar*)buffer)[j+k] = src[k];
                }
                break;
            case CV_16U:
            case CV_16S:
                for( j = 0; j < ncols; j += cn0 )
                {
                    const short* src = *mdata++ ? (short*)data + j : (short*)zerobuf;
                    for( k = 0; k < cn0; k++ )
                        ((short*)buffer)[j+k] = src[k];
                }
                break;
            case CV_32S:
            case CV_32F:
                for( j = 0; j < ncols; j += cn0 )
                {
                    const int* src = *mdata++ ? (int*)data + j : (int*)zerobuf;
                    for( k = 0; k < cn0; k++ )
                        ((int*)buffer)[j+k] = src[k];
                }
                break;
            case CV_64F:
                for( j = 0; j < ncols; j += cn0 )
                {
                    const double* src = *mdata++ ? (double*)data + j : (double*)zerobuf;
                    for( k = 0; k < cn0; k++ )
                        ((double*)buffer)[j+k] = src[k];
                }
                break;
            default:
                assert(0);
                return -1;
            }
            data = (const uchar*)buffer;
        }

        switch( depth )
        {
        case CV_8U:
            if( norm_type == CV_C )
            {
                for( j = 0; j < ncols; j += cn )
                {
                    int val = ((const uchar*)data)[j];
                    inorm = MAX( inorm, val );
                }
            }
            else if( norm_type == CV_L1 )
            {
                inorm = 0;
                for( j = 0; j < ncols; j += cn )
                {
                    int val = ((const uchar*)data)[j];
                    inorm += val;
                }
                fnorm += inorm;
            }
            else
            {
                inorm = 0;
                for( j = 0; j < ncols; j += cn )
                {
                    int val = ((const uchar*)data)[j];
                    inorm += val*val;
                }
                fnorm += inorm;
            }
            break;
        case CV_8S:
            if( norm_type == CV_C )
            {
                for( j = 0; j < ncols; j += cn )
                {
                    int val = abs(((const schar*)data)[j]);
                    inorm = MAX( inorm, val );
                }
            }
            else if( norm_type == CV_L1 )
            {
                inorm = 0;
                for( j = 0; j < ncols; j += cn )
                {
                    int val = abs(((const schar*)data)[j]);
                    inorm += val;
                }
                fnorm += inorm;
            }
            else
            {
                inorm = 0;
                for( j = 0; j < ncols; j += cn )
                {
                    int val = ((const schar*)data)[j];
                    inorm += val*val;
                }
                fnorm += inorm;
            }
            break;
        case CV_16U:
            if( norm_type == CV_C )
            {
                for( j = 0; j < ncols; j += cn )
                {
                    int val = ((const ushort*)data)[j];
                    inorm = MAX( inorm, val );
                }
            }
            else if( norm_type == CV_L1 )
            {
                inorm = 0;
                for( j = 0; j < ncols; j += cn )
                {
                    int val = ((const ushort*)data)[j];
                    inorm += val;
                }
                fnorm += inorm;
            }
            else
            {
                for( j = 0; j < ncols; j += cn )
                {
                    double val = ((const ushort*)data)[j];
                    fnorm += val*val;
                }
            }
            break;
        case CV_16S:
            if( norm_type == CV_C )
            {
                for( j = 0; j < ncols; j += cn )
                {
                    int val = abs(((const short*)data)[j]);
                    inorm = MAX( inorm, val );
                }
            }
            else if( norm_type == CV_L1 )
            {
                inorm = 0;
                for( j = 0; j < ncols; j += cn )
                {
                    int val = abs(((const short*)data)[j]);
                    inorm += val;
                }
                fnorm += inorm;
            }
            else
            {
                for( j = 0; j < ncols; j += cn )
                {
                    double val = ((const short*)data)[j];
                    fnorm += val*val;
                }
            }
            break;
        case CV_32S:
            if( norm_type == CV_C )
            {
                for( j = 0; j < ncols; j += cn )
                {
                    int val = abs(((const int*)data)[j]);
                    inorm = MAX( inorm, val );
                }
            }
            else if( norm_type == CV_L1 )
            {
                for( j = 0; j < ncols; j += cn )
                {
                    double val = fabs((double)((const int*)data)[j]);
                    fnorm += val;
                }
            }
            else
            {
                for( j = 0; j < ncols; j += cn )
                {
                    double val = ((const int*)data)[j];
                    fnorm += val*val;
                }
            }
            break;
        case CV_32F:
            if( norm_type == CV_C )
            {
                for( j = 0; j < ncols; j += cn )
                {
                    double val = fabs((double)((const float*)data)[j]);
                    fnorm = MAX( fnorm, val );
                }
            }
            else if( norm_type == CV_L1 )
            {
                for( j = 0; j < ncols; j += cn )
                {
                    double val = fabs((double)((const float*)data)[j]);
                    fnorm += val;
                }
            }
            else
            {
                for( j = 0; j < ncols; j += cn )
                {
                    double val = ((const float*)data)[j];
                    fnorm += val*val;
                }
            }
            break;
        case CV_64F:
            if( norm_type == CV_C )
            {
                for( j = 0; j < ncols; j += cn )
                {
                    double val = fabs(((const double*)data)[j]);
                    fnorm = MAX( fnorm, val );
                }
            }
            else if( norm_type == CV_L1 )
            {
                for( j = 0; j < ncols; j += cn )
                {
                    double val = fabs(((const double*)data)[j]);
                    fnorm += val;
                }
            }
            else
            {
                for( j = 0; j < ncols; j += cn )
                {
                    double val = ((const double*)data)[j];
                    fnorm += val*val;
                }
            }
            break;
        default:
            assert(0);
            return -1;
        }
    }

    if( norm_type == CV_L2 )
        fnorm = sqrt( fnorm );
    else if( depth < CV_32F && norm_type == CV_C )
        fnorm = inorm;

    return fnorm;
}


// retrieves mean, standard deviation and the number of nonzero mask pixels
int cvTsMeanStdDevNonZero( const CvMat* arr, const CvMat* mask,
                           CvScalar* _mean, CvScalar* _stddev, int coi )
{
    int i = 0, j = 0, k;
    int depth, cn0, cn, cols, ncols, el_size1;
    CvScalar sum = cvScalar(0), sqsum = cvScalar(0);
    double inv_area;
    int isum[4], isqsum[4];
    int nonzero = 0;
    uchar* maskbuf = 0;

    cn0 = cn = CV_MAT_CN(arr->type);
    cols = arr->cols;
    ncols = arr->cols*cn;
    depth = CV_MAT_DEPTH(arr->type);
    el_size1 = CV_ELEM_SIZE(depth);
    if( mask )
    {
        assert( CV_ARE_SIZES_EQ( arr, mask ) && CV_IS_MASK_ARR(mask) );
    }
    else
    {
        maskbuf = (uchar*)cvStackAlloc( cols );
        memset( maskbuf, 1, cols );
        nonzero = cols*arr->rows;
    }

    if( coi != 0 )
        cn = 1;

    for( i = 0; i < arr->rows; i++ )
    {
        const uchar* data = arr->data.ptr + arr->step*i + (coi - (coi != 0))*el_size1;
        const uchar* mdata;

        if( mask )
        {
            mdata = mask->data.ptr + mask->step*i;
            for( j = 0; j < cols; j++ )
                nonzero += mdata[j] != 0;
        }
        else
        {
            mdata = maskbuf;
        }

        // if only a number of pixels in the mask is needed, skip the rest of the loop body
        if( !_mean && !_stddev )
            continue;

        switch( depth )
        {
        case CV_8U:
            for( k = 0; k < cn; k++ )
                isum[k] = isqsum[k] = 0;
            for( j = 0; j < ncols; j += cn0 )
                if( *mdata++ )
                {
                    for( k = 0; k < cn; k++ )
                    {
                        int val = ((const uchar*)data)[j+k];
                        isum[k] += val;
                        isqsum[k] += val*val;
                    }
                }
            for( k = 0; k < cn; k++ )
            {
                sum.val[k] += isum[k];
                sqsum.val[k] += isqsum[k];
            }
            break;
        case CV_8S:
            for( k = 0; k < cn; k++ )
                isum[k] = isqsum[k] = 0;
            for( j = 0; j < ncols; j += cn0 )
                if( *mdata++ )
                {
                    for( k = 0; k < cn; k++ )
                    {
                        int val = ((const schar*)data)[j+k];
                        isum[k] += val;
                        isqsum[k] += val*val;
                    }
                }
            for( k = 0; k < cn; k++ )
            {
                sum.val[k] += isum[k];
                sqsum.val[k] += isqsum[k];
            }
            break;
        case CV_16U:
            for( k = 0; k < cn; k++ )
                isum[k] = 0;
            for( j = 0; j < ncols; j += cn0 )
                if( *mdata++ )
                {
                    for( k = 0; k < cn; k++ )
                    {
                        int val = ((const ushort*)data)[j+k];
                        isum[k] += val;
                        sqsum.val[k] += ((double)val)*val;
                    }
                }
            for( k = 0; k < cn; k++ )
                sum.val[k] += isum[k];
            break;
        case CV_16S:
            for( k = 0; k < cn; k++ )
                isum[k] = 0;
            for( j = 0; j < ncols; j += cn0 )
                if( *mdata++ )
                {
                    for( k = 0; k < cn; k++ )
                    {
                        int val = ((const short*)data)[j+k];
                        isum[k] += val;
                        sqsum.val[k] += ((double)val)*val;
                    }
                }
            for( k = 0; k < cn; k++ )
                sum.val[k] += isum[k];
            break;
        case CV_32S:
            for( j = 0; j < ncols; j += cn0 )
                if( *mdata++ )
                {
                    for( k = 0; k < cn; k++ )
                    {
                        double val = ((const int*)data)[j+k];
                        sum.val[k] += val;
                        sqsum.val[k] += val*val;
                    }
                }
            break;
        case CV_32F:
            for( j = 0; j < ncols; j += cn0 )
                if( *mdata++ )
                {
                    for( k = 0; k < cn; k++ )
                    {
                        double val = ((const float*)data)[j+k];
                        sum.val[k] += val;
                        sqsum.val[k] += val*val;
                    }
                }
            break;
        case CV_64F:
            for( j = 0; j < ncols; j += cn0 )
                if( *mdata++ )
                {
                    for( k = 0; k < cn; k++ )
                    {
                        double val = ((const double*)data)[j+k];
                        sum.val[k] += val;
                        sqsum.val[k] += val*val;
                    }
                }
            break;
        default:
            assert(0);
            return -1;
        }
    }

    inv_area = nonzero ? 1./nonzero : 0.;
    for( k = 0; k < cn; k++ )
    {
        sum.val[k] *= inv_area;
        double t = sqsum.val[k]*inv_area - sum.val[k]*sum.val[k];
        sqsum.val[k] = sqrt(MAX(t, 0));
    }
    if( _mean )
        *_mean = sum;
    if( _stddev )
        *_stddev = sqsum;
    return nonzero;
}

// retrieves global extremums and their positions
void cvTsMinMaxLoc( const CvMat* arr, const CvMat* mask,
                    double* _minval, double* _maxval,
                    CvPoint* _minidx, CvPoint* _maxidx, int coi )
{
    int i = 0, j = 0;
    int depth, cn, cols, ncols, el_size1;
    CvPoint minidx = {-1,-1}, maxidx = {-1,-1};
    uchar* maskbuf = 0;
    int iminval = INT_MAX, imaxval = INT_MIN;
    double minval = DBL_MAX, maxval = -minval;

    cn = CV_MAT_CN(arr->type);
    cols = arr->cols;
    ncols = arr->cols*cn;
    depth = CV_MAT_DEPTH(arr->type);
    el_size1 = CV_ELEM_SIZE(depth);

    if( mask )
    {
        assert( CV_ARE_SIZES_EQ( arr, mask ) && CV_IS_MASK_ARR(mask) );
    }
    else
    {
        maskbuf = (uchar*)cvStackAlloc( cols );
        memset( maskbuf, 1, cols );
    }

    if( coi == 0 && cn > 1 )
    {
        assert(0);
        return;
    }

    for( i = 0; i < arr->rows; i++ )
    {
        const uchar* data = arr->data.ptr + arr->step*i + (coi - (coi != 0))*el_size1;
        const uchar* mdata = mask ? mask->data.ptr + mask->step*i : maskbuf;

        switch( depth )
        {
        case CV_8U:
            for( j = 0; j < ncols; j += cn, mdata++ )
            {
                int val = ((const uchar*)data)[j];
                if( val < iminval && *mdata )
                {
                    iminval = val;
                    minidx = cvPoint(j,i);
                }
                if( val > imaxval && *mdata )
                {
                    imaxval = val;
                    maxidx = cvPoint(j,i);
                }
            }
            break;
        case CV_8S:
            for( j = 0; j < ncols; j += cn, mdata++ )
            {
                int val = ((const schar*)data)[j];
                if( val < iminval && *mdata )
                {
                    iminval = val;
                    minidx = cvPoint(j,i);
                }
                if( val > imaxval && *mdata )
                {
                    imaxval = val;
                    maxidx = cvPoint(j,i);
                }
            }
            break;
        case CV_16U:
            for( j = 0; j < ncols; j += cn, mdata++ )
            {
                int val = ((const ushort*)data)[j];
                if( val < iminval && *mdata )
                {
                    iminval = val;
                    minidx = cvPoint(j,i);
                }
                if( val > imaxval && *mdata )
                {
                    imaxval = val;
                    maxidx = cvPoint(j,i);
                }
            }
            break;
        case CV_16S:
            for( j = 0; j < ncols; j += cn, mdata++ )
            {
                int val = ((const short*)data)[j];
                if( val < iminval && *mdata )
                {
                    iminval = val;
                    minidx = cvPoint(j,i);
                }
                if( val > imaxval && *mdata )
                {
                    imaxval = val;
                    maxidx = cvPoint(j,i);
                }
            }
            break;
        case CV_32S:
            for( j = 0; j < ncols; j += cn, mdata++ )
            {
                int val = ((const int*)data)[j];
                if( val < iminval && *mdata )
                {
                    iminval = val;
                    minidx = cvPoint(j,i);
                }
                if( val > imaxval && *mdata )
                {
                    imaxval = val;
                    maxidx = cvPoint(j,i);
                }
            }
            break;
        case CV_32F:
            for( j = 0; j < ncols; j += cn, mdata++ )
            {
                float val = ((const float*)data)[j];
                if( val < minval && *mdata )
                {
                    minval = val;
                    minidx = cvPoint(j,i);
                }
                if( val > maxval && *mdata )
                {
                    maxval = val;
                    maxidx = cvPoint(j,i);
                }
            }
            break;
        case CV_64F:
            for( j = 0; j < ncols; j += cn, mdata++ )
            {
                double val = ((const double*)data)[j];
                if( val < minval && *mdata )
                {
                    minval = val;
                    minidx = cvPoint(j,i);
                }
                if( val > maxval && *mdata )
                {
                    maxval = val;
                    maxidx = cvPoint(j,i);
                }
            }
            break;
        default:
            assert(0);
            return;
        }
    }

    if( minidx.x < 0 )
        minval = maxval = 0;
    else
    {
        if( depth < CV_32F )
            minval = iminval, maxval = imaxval;
        minidx.x /= cn;
        maxidx.x /= cn;
    }

    if( _minval )
        *_minval = minval;

    if( _maxval )
        *_maxval = maxval;

    if( _minidx )
        *_minidx = minidx;

    if( _maxidx )
        *_maxidx = maxidx;
}


void cvTsLogic( const CvMat* a, const CvMat* b, CvMat* c, int logic_op )
{
    int i = 0, j = 0, ncols;
    ncols = a->cols*CV_ELEM_SIZE(a->type);

    assert( CV_ARE_TYPES_EQ(a,b) && CV_ARE_SIZES_EQ(a,b) );
    assert( CV_ARE_TYPES_EQ(a,c) && CV_ARE_SIZES_EQ(a,c) );

    for( i = 0; i < a->rows; i++ )
    {
        uchar* a_data = a->data.ptr + a->step*i;
        uchar* b_data = b->data.ptr + b->step*i;
        uchar* c_data = c->data.ptr + c->step*i;

        switch( logic_op )
        {
        case CV_TS_LOGIC_AND:
            for( j = 0; j < ncols; j++ )
                c_data[j] = (uchar)(a_data[j] & b_data[j]);
            break;
        case CV_TS_LOGIC_OR:
            for( j = 0; j < ncols; j++ )
                c_data[j] = (uchar)(a_data[j] | b_data[j]);
            break;
        case CV_TS_LOGIC_XOR:
            for( j = 0; j < ncols; j++ )
                c_data[j] = (uchar)(a_data[j] ^ b_data[j]);
            break;
        default:
            assert(0);
            return;
        }
    }
}

void cvTsLogicS( const CvMat* a, CvScalar s, CvMat* c, int logic_op )
{
    int i = 0, j = 0, k;
    int cn, ncols, elem_size;
    uchar* b_data;
    union
    {
        uchar ptr[4];
        schar c[4];
        short s[4];
        ushort w[4];
        int i[4];
        float f[4];
        double d[4];
    } buf;
    cn = CV_MAT_CN(a->type);
    elem_size = CV_ELEM_SIZE(a->type);
    ncols = a->cols * elem_size;
    b_data = (uchar*)malloc( ncols );

    assert( CV_ARE_TYPES_EQ(a,c) && CV_ARE_SIZES_EQ(a,c) );

    if( logic_op == CV_TS_LOGIC_NOT )
    {
        memset( b_data, -1, ncols );
        logic_op = CV_TS_LOGIC_XOR;
    }
    else
    {
        switch( CV_MAT_DEPTH(a->type) )
        {
        case CV_8U:
            for( k = 0; k < cn; k++ )
            {
                int val = cvRound(s.val[k]);
                buf.ptr[k] = CV_CAST_8U(val);
            }
            break;
        case CV_8S:
            for( k = 0; k < cn; k++ )
            {
                int val = cvRound(s.val[k]);
                buf.c[k] = CV_CAST_8S(val);
            }
            break;
        case CV_16U:
            for( k = 0; k < cn; k++ )
            {
                int val = cvRound(s.val[k]);
                buf.w[k] = CV_CAST_16U(val);
            }
            break;
        case CV_16S:
            for( k = 0; k < cn; k++ )
            {
                int val = cvRound(s.val[k]);
                buf.s[k] = CV_CAST_16S(val);
            }
            break;
        case CV_32S:
            for( k = 0; k < cn; k++ )
            {
                int val = cvRound(s.val[k]);
                buf.i[k] = CV_CAST_32S(val);
            }
            break;
        case CV_32F:
            for( k = 0; k < cn; k++ )
            {
                double val = s.val[k];
                buf.f[k] = CV_CAST_32F(val);
            }
            break;
        case CV_64F:
            for( k = 0; k < cn; k++ )
            {
                double val = s.val[k];
                buf.d[k] = CV_CAST_64F(val);
            }
            break;
        default:
            assert(0);
            return;
        }

        for( j = 0; j < ncols; j += elem_size )
            memcpy( b_data + j, buf.ptr, elem_size );
    }

    for( i = 0; i < a->rows; i++ )
    {
        uchar* a_data = a->data.ptr + a->step*i;
        uchar* c_data = c->data.ptr + c->step*i;

        switch( logic_op )
        {
        case CV_TS_LOGIC_AND:
            for( j = 0; j < ncols; j++ )
                c_data[j] = (uchar)(a_data[j] & b_data[j]);
            break;
        case CV_TS_LOGIC_OR:
            for( j = 0; j < ncols; j++ )
                c_data[j] = (uchar)(a_data[j] | b_data[j]);
            break;
        case CV_TS_LOGIC_XOR:
            for( j = 0; j < ncols; j++ )
                c_data[j] = (uchar)(a_data[j] ^ b_data[j]);
            break;
        default:
            assert(0);
            return;
        }
    }

    if( b_data )
        free( b_data );
}


void cvTsGEMM( const CvMat* a, const CvMat* b, double alpha,
               const CvMat* c, double beta, CvMat* d, int flags )
{
    int i, j, k;
    int a_rows, a_cols, b_rows, b_cols;
    int c_rows, c_cols, d_rows, d_cols;
    int cn, el_size;
    int a_step, a_delta, b_step, b_delta;
    int c_step, c_delta, d_step;

    a_rows = a->rows; a_cols = a->cols;
    cn = CV_MAT_CN(a->type);
    el_size = CV_ELEM_SIZE(a->type & ~CV_MAT_CN_MASK);
    a_step = a->step / el_size; a_delta = cn;
    d_rows = d->rows; d_cols = d->cols;
    b_rows = b->rows; b_cols = b->cols;
    b_step = b->step / el_size; b_delta = cn;
    c_rows = c ? c->rows : 0; c_cols = c ? c->cols : 0;
    c_step = c ? c->step / el_size : 0; c_delta = c ? cn : 0;
    d_step = d->step / el_size;

    assert( CV_ARE_TYPES_EQ(a,b) && CV_ARE_TYPES_EQ(a,d) );
    assert( CV_MAT_CN(a->type) <= 2 );

    if( flags & CV_TS_GEMM_A_T )
    {
        CV_SWAP( a_rows, a_cols, i );
        CV_SWAP( a_step, a_delta, i );
    }

    if( flags & CV_TS_GEMM_B_T )
    {
        CV_SWAP( b_rows, b_cols, i );
        CV_SWAP( b_step, b_delta, i );
    }

    if( flags & CV_TS_GEMM_C_T )
    {
        CV_SWAP( c_rows, c_cols, i );
        CV_SWAP( c_step, c_delta, i );
    }

    assert( a_rows == d_rows && a_cols == b_rows && b_cols == d_cols );
    assert( a->data.ptr != d->data.ptr && b->data.ptr != d->data.ptr );

    if( c )
    {
        assert( CV_ARE_TYPES_EQ(a,c) && c_rows == d_rows && c_cols == d_cols );
        assert( c->data.ptr != d->data.ptr || (flags & CV_TS_GEMM_C_T) == 0 );
    }

    if( CV_MAT_DEPTH(a->type) == CV_32F )
    {
        float* a_data0 = a->data.fl;
        float* b_data0 = b->data.fl;
        float* c_data0 = c ? c->data.fl : 0;
        float* d_data = d->data.fl;

        for( i = 0; i < d_rows; i++, d_data += d_step, c_data0 += c_step, a_data0 += a_step )
        {
            for( j = 0; j < d_cols; j++ )
            {
                float* a_data = a_data0;
                float* b_data = b_data0 + j*b_delta;
                float* c_data = c_data0 + j*c_delta;

                if( cn == 1 )
                {
                    double s = 0;
                    for( k = 0; k < a_cols; k++ )
                    {
                        s += ((double)a_data[0])*b_data[0];
                        a_data += a_delta;
                        b_data += b_step;
                    }
                    d_data[j] = (float)(s*alpha + (c_data ? c_data[0]*beta : 0));
                }
                else
                {
                    double s_re = 0, s_im = 0;

                    for( k = 0; k < a_cols; k++ )
                    {
                        s_re += ((double)a_data[0])*b_data[0] - ((double)a_data[1])*b_data[1];
                        s_im += ((double)a_data[0])*b_data[1] + ((double)a_data[1])*b_data[0];
                        a_data += a_delta;
                        b_data += b_step;
                    }

                    s_re *= alpha;
                    s_im *= alpha;

                    if( c_data )
                    {
                        s_re += c_data[0]*beta;
                        s_im += c_data[1]*beta;
                    }

                    d_data[j*2] = (float)s_re;
                    d_data[j*2+1] = (float)s_im;
                }
            }
        }
    }
    else if( CV_MAT_DEPTH(a->type) == CV_64F )
    {
        double* a_data0 = a->data.db;
        double* b_data0 = b->data.db;
        double* c_data0 = c ? c->data.db : 0;
        double* d_data = d->data.db;

        for( i = 0; i < d_rows; i++, d_data += d_step, c_data0 += c_step, a_data0 += a_step )
        {
            for( j = 0; j < d_cols; j++ )
            {
                double* a_data = a_data0;
                double* b_data = b_data0 + j*b_delta;
                double* c_data = c_data0 + j*c_delta;

                if( cn == 1 )
                {
                    double s = 0;
                    for( k = 0; k < a_cols; k++ )
                    {
                        s += a_data[0]*b_data[0];
                        a_data += a_delta;
                        b_data += b_step;
                    }
                    d_data[j] = s*alpha + (c_data ? c_data[0]*beta : 0);
                }
                else
                {
                    double s_re = 0, s_im = 0;

                    for( k = 0; k < a_cols; k++ )
                    {
                        s_re += a_data[0]*b_data[0] - a_data[1]*b_data[1];
                        s_im += a_data[0]*b_data[1] + a_data[1]*b_data[0];
                        a_data += a_delta;
                        b_data += b_step;
                    }
                    s_re *= alpha;
                    s_im *= alpha;

                    if( c_data )
                    {
                        s_re += c_data[0]*beta;
                        s_im += c_data[1]*beta;
                    }

                    d_data[j*2] = s_re;
                    d_data[j*2+1] = s_im;
                }
            }
        }
    }
    else
    {
        assert(0);
    }
}


CvMat* cvTsSelect( const CvMat* a, CvMat* header, CvRect rect )
{
    CvMat h;
    int el_size;

    h = cvMat( rect.height, rect.width, a->type );
    el_size = CV_ELEM_SIZE(a->type);

    h.data.ptr = a->data.ptr + rect.y*a->step + rect.x*el_size;
    h.step = rect.height > 1 ? a->step : 0;
    h.type &= ~CV_MAT_CONT_FLAG;
    if( rect.height == 1 || h.step == h.cols*el_size )
        h.type |= CV_MAT_CONT_FLAG;
    *header = h;
    return header;
}


double cvTsMinVal( int type )
{
    switch( CV_MAT_DEPTH(type) )
    {
    case CV_8U:
        return 0;
    case CV_8S:
        return -128;
    case CV_16U:
        return 0;
    case CV_16S:
        return -32768;
    case CV_32S:
        return -1000000;
    case CV_32F:
        return -1000.;
    case CV_64F:
        return -1000.;
    }
    return log(-1.);
}


double cvTsMaxVal( int type )
{
    switch( CV_MAT_DEPTH(type) )
    {
    case CV_8U:
        return 256;
    case CV_8S:
        return 128;
    case CV_16U:
        return 65536;
    case CV_16S:
        return 32768;
    case CV_32S:
        return 1000000;
    case CV_32F:
        return 1000.;
    case CV_64F:
        return 1000.;
    }
    return log(-1.);
}


void cvTsPrepareToFilter( const CvMat* a, CvMat* b, CvPoint ofs,
                          int border_mode, CvScalar fill_val )
{
    int i, j, dir;
    CvMat temp, temp2;

    assert( 0 <= ofs.x && ofs.x <= b->cols - a->cols &&
            0 <= ofs.y && ofs.y <= b->rows - a->rows );

    cvTsSelect( b, &temp, cvRect( ofs.x, ofs.y, a->cols, a->rows ));
    cvTsCopy( a, &temp, 0 );

    assert( border_mode == CV_TS_BORDER_FILL ||
            border_mode == CV_TS_BORDER_REPLICATE ||
            border_mode == CV_TS_BORDER_REFLECT );

    if( ofs.y > 0 )
    {
        if( border_mode == CV_TS_BORDER_FILL )
        {
            cvTsSelect( b, &temp, cvRect( ofs.x, 0, a->cols, ofs.y ));
            cvTsAdd( 0, cvScalar(0), 0, cvScalar(0), fill_val, &temp, 0 );
        }
        else if( border_mode == CV_TS_BORDER_REPLICATE || a->rows == 1 )
        {
            cvTsSelect( b, &temp, cvRect( ofs.x, ofs.y, a->cols, 1 ));
            for( i = ofs.y-1; i >= 0; i-- )
            {
                cvTsSelect( b, &temp2, cvRect( ofs.x, i, a->cols, 1 ));
                cvTsCopy( &temp, &temp2, 0 );
            }
        }
        else if( border_mode == CV_TS_BORDER_REFLECT )
        {
            j = 1; dir = 1;
            for( i = ofs.y-1; i >= 0; i-- )
            {
                cvTsSelect( b, &temp, cvRect( ofs.x, ofs.y+j, a->cols, 1 ));
                cvTsSelect( b, &temp2, cvRect( ofs.x, i, a->cols, 1 ));
                cvTsCopy( &temp, &temp2, 0 );
                if( (unsigned)(j + dir) >= (unsigned)a->rows )
                    dir = -dir;
                j += dir;
            }
        }
    }

    ofs.y += a->rows;
    if( ofs.y < b->rows )
    {
        if( border_mode == CV_TS_BORDER_FILL )
        {
            cvTsSelect( b, &temp, cvRect( ofs.x, ofs.y, a->cols, b->rows - ofs.y ));
            cvTsAdd( 0, cvScalar(0), 0, cvScalar(0), fill_val, &temp, 0 );
        }
        else if( border_mode == CV_TS_BORDER_REPLICATE || a->rows == 1 )
        {
            cvTsSelect( b, &temp, cvRect( ofs.x, ofs.y - 1, a->cols, 1 ));
            for( i = ofs.y; i < b->rows; i++ )
            {
                cvTsSelect( b, &temp2, cvRect( ofs.x, i, a->cols, 1 ));
                cvTsCopy( &temp, &temp2, 0 );
            }
        }
        else
        {
            j = a->rows - 2; dir = -1;
            for( i = ofs.y; i < b->rows; i++ )
            {
                cvTsSelect( b, &temp, cvRect( ofs.x, ofs.y-a->rows+j, a->cols, 1 ));
                cvTsSelect( b, &temp2, cvRect( ofs.x, i, a->cols, 1 ));
                cvTsCopy( &temp, &temp2, 0 );
                if( (unsigned)(j + dir) >= (unsigned)a->rows )
                    dir = -dir;
                j += dir;
            }
        }
    }

    if( ofs.x > 0 )
    {
        if( border_mode == CV_TS_BORDER_FILL )
        {
            cvTsSelect( b, &temp, cvRect( 0, 0, ofs.x, b->rows ));
            cvTsAdd( 0, cvScalar(0), 0, cvScalar(0), fill_val, &temp, 0 );
        }
        else if( border_mode == CV_TS_BORDER_REPLICATE || a->cols == 1 )
        {
            cvTsSelect( b, &temp, cvRect( ofs.x, 0, 1, b->rows ));
            for( i = ofs.x-1; i >= 0; i-- )
            {
                cvTsSelect( b, &temp2, cvRect( i, 0, 1, b->rows ));
                cvTsCopy( &temp, &temp2, 0 );
            }
        }
        else if( border_mode == CV_TS_BORDER_REFLECT )
        {
            j = 1; dir = 1;
            for( i = ofs.x-1; i >= 0; i-- )
            {
                cvTsSelect( b, &temp, cvRect( ofs.x+j, 0, 1, b->rows ));
                cvTsSelect( b, &temp2, cvRect( i, 0, 1, b->rows ));
                cvTsCopy( &temp, &temp2, 0 );
                if( (unsigned)(j + dir) >= (unsigned)a->cols )
                    dir = -dir;
                j += dir;
            }
        }
    }

    ofs.x += a->cols;
    if( ofs.x < b->cols )
    {
        if( border_mode == CV_TS_BORDER_FILL )
        {
            cvTsSelect( b, &temp, cvRect( ofs.x, 0, b->cols - ofs.x, b->rows ));
            cvTsAdd( 0, cvScalar(0), 0, cvScalar(0), fill_val, &temp, 0 );
        }
        else if( border_mode == CV_TS_BORDER_REPLICATE || a->cols == 1 )
        {
            cvTsSelect( b, &temp, cvRect( ofs.x-1, 0, 1, b->rows ));
            for( i = ofs.x; i < b->cols; i++ )
            {
                cvTsSelect( b, &temp2, cvRect( i, 0, 1, b->rows ));
                cvTsCopy( &temp, &temp2, 0 );
            }
        }
        else if( border_mode == CV_TS_BORDER_REFLECT )
        {
            j = a->cols - 2; dir = -1;
            for( i = ofs.x; i < b->cols; i++ )
            {
                cvTsSelect( b, &temp, cvRect( ofs.x-a->cols+j, 0, 1, b->rows ));
                cvTsSelect( b, &temp2, cvRect( i, 0, 1, b->rows ));
                cvTsCopy( &temp, &temp2, 0 );
                if( (unsigned)(j + dir) >= (unsigned)a->cols )
                    dir = -dir;
                j += dir;
            }
        }
    }
}


void cvTsConvolve2D( const CvMat* a, CvMat* b, const CvMat* kernel, CvPoint anchor )
{
    int i, j, k;
    int cn, ncols, a_step;
    int ker_size = kernel->rows*kernel->cols;
    int* offset = (int*)malloc( ker_size*sizeof(offset[0]));
    float* k_data = (float*)malloc( ker_size*sizeof(k_data[0]));
    int all_same = 1;
    float first = kernel->data.fl[0];
    uchar *a_data, *b_data;

    cn = CV_MAT_CN(a->type);
    ncols = b->cols*cn;
    a_step = a->step / CV_ELEM_SIZE(a->type & ~CV_MAT_CN_MASK);

    assert( a->cols == b->cols + kernel->cols - 1 &&
            a->rows == b->rows + kernel->rows - 1 && CV_ARE_TYPES_EQ( a, b ) );
    assert( CV_MAT_TYPE(kernel->type) == CV_32FC1 );
    assert( 0 <= anchor.x && anchor.x < kernel->cols &&
            0 <= anchor.y && anchor.y < kernel->rows );

    for( i = 0, k = 0; i < kernel->rows; i++ )
        for( j = 0; j < kernel->cols; j++ )
        {
            float f = ((float*)(kernel->data.ptr + kernel->step*i))[j];
            if( f )
            {
                k_data[k] = f;
                offset[k++] = (i - anchor.y)*a_step + (j - anchor.x)*cn;
            }
            if( f != first )
                all_same = 0;
        }

    ker_size = k;
    a_data = a->data.ptr + a->step*anchor.y + CV_ELEM_SIZE(a->type)*anchor.x;
    b_data = b->data.ptr;

    for( i = 0; i < b->rows; i++, a_data += a->step, b_data += b->step )
    {
        switch( CV_MAT_DEPTH(a->type) )
        {
        case CV_8U:
            for( j = 0; j < ncols; j++ )
            {
                double s = 0;
                int val;
                for( k = 0; k < ker_size; k++ )
                    s += ((uchar*)a_data)[j+offset[k]]*k_data[k];
                val = cvRound(s);
                ((uchar*)b_data)[j] = CV_CAST_8U(val);
            }
            break;
        case CV_8S:
            for( j = 0; j < ncols; j++ )
            {
                double s = 0;
                int val;
                for( k = 0; k < ker_size; k++ )
                    s += ((schar*)a_data)[j+offset[k]]*k_data[k];
                val = cvRound(s);
                ((schar*)b_data)[j] = CV_CAST_8S(val);
            }
            break;
        case CV_16U:
            for( j = 0; j < ncols; j++ )
            {
                double s = 0;
                int val;
                for( k = 0; k < ker_size; k++ )
                    s += ((ushort*)a_data)[j+offset[k]]*k_data[k];
                val = cvRound(s);
                ((ushort*)b_data)[j] = CV_CAST_16U(val);
            }
            break;
        case CV_16S:
            for( j = 0; j < ncols; j++ )
            {
                double s = 0;
                int val;
                for( k = 0; k < ker_size; k++ )
                    s += ((short*)a_data)[j+offset[k]]*k_data[k];
                val = cvRound(s);
                ((short*)b_data)[j] = CV_CAST_16S(val);
            }
            break;
        case CV_32S:
            for( j = 0; j < ncols; j++ )
            {
                double s = 0;
                for( k = 0; k < ker_size; k++ )
                    s += ((int*)a_data)[j+offset[k]]*k_data[k];
                ((int*)b_data)[j] = cvRound(s);
            }
            break;
        case CV_32F:
            if( !all_same )
            {
                for( j = 0; j < ncols; j++ )
                {
                    double s = 0;
                    for( k = 0; k < ker_size; k++ )
                        s += (double)((float*)a_data)[j+offset[k]]*k_data[k];
                    ((float*)b_data)[j] = (float)s;
                }
            }
            else
            {
                // special branch to speedup feature selection and blur tests
                for( j = 0; j < ncols; j++ )
                {
                    double s = 0;
                    for( k = 0; k < ker_size; k++ )
                        s += (double)((float*)a_data)[j+offset[k]];
                    ((float*)b_data)[j] = (float)(s*first);
                }
            }
            break;
        case CV_64F:
            for( j = 0; j < ncols; j++ )
            {
                double s = 0;
                for( k = 0; k < ker_size; k++ )
                    s += ((double*)a_data)[j+offset[k]]*k_data[k];
                ((double*)b_data)[j] = (double)s;
            }
            break;
        default:
            assert(0);
        }
    }

    free( offset );
    free( k_data );
}


void cvTsMinMaxFilter( const CvMat* a, CvMat* b, const IplConvKernel* kernel, int op_type )
{
    int i, j, k;
    int cn, ncols, a_step;
    int ker_size = kernel->nRows*kernel->nCols;
    int* offset = (int*)malloc( ker_size*sizeof(offset[0]));
    int calc_max = op_type != 0;
    uchar *a_data, *b_data;

    cn = CV_MAT_CN(a->type);
    ncols = b->cols*cn;
    a_step = a->step / CV_ELEM_SIZE(a->type & ~CV_MAT_CN_MASK);

    assert( a->cols == b->cols + kernel->nCols - 1 &&
            a->rows == b->rows + kernel->nRows - 1 && CV_ARE_TYPES_EQ( a, b ) );
    assert( 0 <= kernel->anchorX && kernel->anchorX < kernel->nCols &&
            0 <= kernel->anchorY && kernel->anchorY < kernel->nRows );

    for( i = 0, k = 0; i < kernel->nRows; i++ )
        for( j = 0; j < kernel->nCols; j++ )
        {
            if( !kernel->values || kernel->values[i*kernel->nCols + j] )
                offset[k++] = (i - kernel->anchorY)*a_step + (j - kernel->anchorX)*cn;
        }

    if( k == 0 )
        offset[k++] = 0;

    ker_size = k;

    a_data = a->data.ptr + kernel->anchorY*a->step + kernel->anchorX*CV_ELEM_SIZE(a->type);
    b_data = b->data.ptr;

    for( i = 0; i < b->rows; i++, a_data += a->step, b_data += b->step )
    {
        switch( CV_MAT_DEPTH(a->type) )
        {
        case CV_8U:
            for( j = 0; j < ncols; j++ )
            {
                int m = ((uchar*)a_data)[j+offset[0]];
                for( k = 1; k < ker_size; k++ )
                {
                    int v = ((uchar*)a_data)[j+offset[k]];
                    if( calc_max )
                    {
                        if( m < v )
                            m = v;
                    }
                    else if( m > v )
                        m = v;
                }
                ((uchar*)b_data)[j] = (uchar)m;
            }
            break;
        case CV_16U:
            for( j = 0; j < ncols; j++ )
            {
                int m = ((ushort*)a_data)[j+offset[0]];
                for( k = 1; k < ker_size; k++ )
                {
                    int v = ((ushort*)a_data)[j+offset[k]];
                    if( calc_max )
                    {
                        if( m < v )
                            m = v;
                    }
                    else if( m > v )
                        m = v;
                }
                ((ushort*)b_data)[j] = (ushort)m;
            }
            break;
        case CV_16S:
            for( j = 0; j < ncols; j++ )
            {
                int m = ((short*)a_data)[j+offset[0]];
                for( k = 1; k < ker_size; k++ )
                {
                    int v = ((short*)a_data)[j+offset[k]];
                    if( calc_max )
                    {
                        if( m < v )
                            m = v;
                    }
                    else if( m > v )
                        m = v;
                }
                ((short*)b_data)[j] = (short)m;
            }
            break;
        case CV_32S:
            for( j = 0; j < ncols; j++ )
            {
                int m = ((int*)a_data)[j+offset[0]];
                for( k = 1; k < ker_size; k++ )
                {
                    int v = ((int*)a_data)[j+offset[k]];
                    if( calc_max )
                    {
                        if( m < v )
                            m = v;
                    }
                    else if( m > v )
                        m = v;
                }
                ((int*)b_data)[j] = m;
            }
            break;
        case CV_32F:
            for( j = 0; j < ncols; j++ )
            {
                float m = ((float*)a_data)[j+offset[0]];
                for( k = 1; k < ker_size; k++ )
                {
                    float v = ((float*)a_data)[j+offset[k]];
                    if( calc_max )
                    {
                        if( m < v )
                            m = v;
                    }
                    else if( m > v )
                        m = v;
                }
                ((float*)b_data)[j] = (float)m;
            }
            break;
        case CV_64F:
            for( j = 0; j < ncols; j++ )
            {
                double m = ((double*)a_data)[j+offset[0]];
                for( k = 1; k < ker_size; k++ )
                {
                    double v = ((double*)a_data)[j+offset[k]];
                    if( calc_max )
                    {
                        if( m < v )
                            m = v;
                    }
                    else if( m > v )
                        m = v;
                }
                ((double*)b_data)[j] = (double)m;
            }
            break;
        default:
            assert(0);
        }
    }

    free( offset );
}


double cvTsCrossCorr( const CvMat* a, const CvMat* b )
{
    int i, j;
    int cn, ncols;
    double s = 0;

    cn = CV_MAT_CN(a->type);
    ncols = a->cols*cn;

    assert( CV_ARE_SIZES_EQ( a, b ) && CV_ARE_TYPES_EQ( a, b ) );
    for( i = 0; i < a->rows; i++ )
    {
        uchar* a_data = a->data.ptr + a->step*i;
        uchar* b_data = b->data.ptr + b->step*i;

        switch( CV_MAT_DEPTH(a->type) )
        {
        case CV_8U:
            for( j = 0; j < ncols; j++ )
                s += ((uchar*)a_data)[j]*((uchar*)b_data)[j];
            break;
        case CV_8S:
            for( j = 0; j < ncols; j++ )
                s += ((schar*)a_data)[j]*((schar*)b_data)[j];
            break;
        case CV_16U:
            for( j = 0; j < ncols; j++ )
                s += (double)((ushort*)a_data)[j]*((ushort*)b_data)[j];
            break;
        case CV_16S:
            for( j = 0; j < ncols; j++ )
                s += ((short*)a_data)[j]*((short*)b_data)[j];
            break;
        case CV_32S:
            for( j = 0; j < ncols; j++ )
                s += ((double)((int*)a_data)[j])*((int*)b_data)[j];
            break;
        case CV_32F:
            for( j = 0; j < ncols; j++ )
                s += ((double)((float*)a_data)[j])*((float*)b_data)[j];
            break;
        case CV_64F:
            for( j = 0; j < ncols; j++ )
                s += ((double*)a_data)[j]*((double*)b_data)[j];
            break;
        default:
            assert(0);
            return log(-1.);
        }
    }

    return s;
}


void cvTsTransform( const CvMat* a, CvMat* b, const CvMat* transmat, const CvMat* shift )
{
    int i, j, k, cols, dst_cols;
    int cn, dst_cn, depth, mat_depth, shiftstep;
    double mat[20], *buf, *dst_buf;

    cn = CV_MAT_CN(a->type);
    dst_cn = CV_MAT_CN(b->type);
    depth = CV_MAT_DEPTH(a->type);
    mat_depth = CV_MAT_DEPTH(transmat->type);
    cols = transmat->cols;

    // prepare cn x (cn + 1) transform matrix
    if( mat_depth == CV_32F )
    {
        shiftstep = shift && shift->rows > 1 ? shift->step/sizeof(float) : 1;
        for( i = 0; i < transmat->rows; i++ )
        {
            mat[i*(cn+1) + cn] = 0.;
            for( j = 0; j < cols; j++ )
                mat[i*(cn+1) + j] = ((float*)(transmat->data.ptr + transmat->step*i))[j];
            if( shift )
                mat[i*(cn+1) + cn] = shift->data.fl[i*shiftstep];
        }
    }
    else
    {
        assert( mat_depth == CV_64F );

        shiftstep = shift && shift->rows > 1 ? shift->step/sizeof(double) : 1;
        for( i = 0; i < transmat->rows; i++ )
        {
            mat[i*(cn+1) + cn] = 0.;
            for( j = 0; j < cols; j++ )
                mat[i*(cn+1) + j] = ((double*)(transmat->data.ptr + transmat->step*i))[j];
            if( shift )
                mat[i*(cn+1) + cn] = shift->data.db[i*shiftstep];
        }
    }

    // transform data
    cols = a->cols * cn;
    dst_cols = a->cols * dst_cn;
    buf = (double*)cvStackAlloc( cols * sizeof(double) );
    dst_buf = (double*)cvStackAlloc( dst_cols * sizeof(double) );

    for( i = 0; i < a->rows; i++ )
    {
        uchar* src = a->data.ptr + i*a->step;
        uchar* dst = b->data.ptr + i*b->step;
        double* _dst = dst_buf;

        switch( depth )
        {
        case CV_8U:
            for( j = 0; j < cols; j++ )
                buf[j] = ((uchar*)src)[j];
            break;
        case CV_16U:
            for( j = 0; j < cols; j++ )
                buf[j] = ((ushort*)src)[j];
            break;
        case CV_16S:
            for( j = 0; j < cols; j++ )
                buf[j] = ((short*)src)[j];
            break;
        case CV_32S:
            for( j = 0; j < cols; j++ )
                buf[j] = ((int*)src)[j];
            break;
        case CV_32F:
            for( j = 0; j < cols; j++ )
                buf[j] = ((float*)src)[j];
            break;
        case CV_64F:
            for( j = 0; j < cols; j++ )
                buf[j] = ((double*)src)[j];
            break;
        default:
            assert(0);
        }

        switch( cn )
        {
        case 1:
            for( j = 0; j < cols; j++, _dst += dst_cn )
                for( k = 0; k < dst_cn; k++ )
                    _dst[k] = buf[j]*mat[2*k] + mat[2*k+1];
            break;
        case 2:
            for( j = 0; j < cols; j += 2, _dst += dst_cn )
                for( k = 0; k < dst_cn; k++ )
                    _dst[k] = buf[j]*mat[3*k] + buf[j+1]*mat[3*k+1] + mat[3*k+2];
            break;
        case 3:
            for( j = 0; j < cols; j += 3, _dst += dst_cn )
                for( k = 0; k < dst_cn; k++ )
                    _dst[k] = buf[j]*mat[4*k] + buf[j+1]*mat[4*k+1] +
                              buf[j+2]*mat[4*k+2] + mat[4*k+3];
            break;
        case 4:
            for( j = 0; j < cols; j += 4, _dst += dst_cn )
                for( k = 0; k < dst_cn; k++ )
                    _dst[k] = buf[j]*mat[5*k] + buf[j+1]*mat[5*k+1] +
                        buf[j+2]*mat[5*k+2] + buf[j+3]*mat[5*k+3] + mat[5*k+4];
            break;
        default:
            assert(0);
        }

        switch( depth )
        {
        case CV_8U:
            for( j = 0; j < dst_cols; j++ )
            {
                int val = cvRound(dst_buf[j]);
                ((uchar*)dst)[j] = CV_CAST_8U(val);
            }
            break;
        case CV_16U:
            for( j = 0; j < dst_cols; j++ )
            {
                int val = cvRound(dst_buf[j]);
                ((ushort*)dst)[j] = CV_CAST_16U(val);
            }
            break;
        case CV_16S:
            for( j = 0; j < dst_cols; j++ )
            {
                int val = cvRound(dst_buf[j]);
                ((short*)dst)[j] = CV_CAST_16S(val);
            }
            break;
        case CV_32S:
            for( j = 0; j < dst_cols; j++ )
                ((int*)dst)[j] = cvRound(dst_buf[j]);
            break;
        case CV_32F:
            for( j = 0; j < dst_cols; j++ )
                ((float*)dst)[j] = (float)dst_buf[j];
            break;
        case CV_64F:
            for( j = 0; j < dst_cols; j++ )
                ((double*)dst)[j] = dst_buf[j];
            break;
        default:
            assert(0);
        }
    }
}


CvMat* cvTsTranspose( const CvMat* a, CvMat* b )
{
    int i, j, k, rows, cols, elem_size;
    uchar *a_data, *b_data;
    int a_step, b_step;

    elem_size = CV_ELEM_SIZE(a->type);
    rows = a->rows;
    cols = a->cols;

    assert( a->rows == b->cols && a->cols == b->rows && CV_ARE_TYPES_EQ(a,b) );
    a_data = a->data.ptr;
    a_step = a->step;
    b_data = b->data.ptr;
    b_step = b->step;

    if( rows == cols )
    {
        for( i = 0; i < rows; i++ )
        {
            for( j = 0; j <= i; j++ )
            {
                uchar* a_ij = a_data + a_step*i + elem_size*j;
                uchar* a_ji = a_data + a_step*j + elem_size*i;
                uchar* b_ij = b_data + b_step*i + elem_size*j;
                uchar* b_ji = b_data + b_step*j + elem_size*i;
                for( k = 0; k < elem_size; k++ )
                {
                    uchar t0 = a_ij[k];
                    uchar t1 = a_ji[k];
                    b_ji[k] = t0;
                    b_ij[k] = t1;
                }
            }
        }
    }
    else
    {
        for( i = 0; i < cols; i++ )
        {
            for( j = 0; j < rows; j++ )
            {
                uchar* a_ji = a_data + a_step*j + elem_size*i;
                uchar* b_ij = b_data + b_step*i + elem_size*j;
                for( k = 0; k < elem_size; k++ )
                    b_ij[k] = a_ji[k];
            }
        }
    }

    return b;
}


void cvTsFlip( const CvMat* a, CvMat* b, int flip_type )
{
    int i, j, k, rows, cols, elem_size;
    uchar *a_data, *b_data;
    int a_step, b_step;

    elem_size = CV_ELEM_SIZE(a->type);
    rows = a->rows;
    cols = a->cols*elem_size;

    assert( CV_ARE_SIZES_EQ(a,b) && CV_ARE_TYPES_EQ(a,b) && a->data.ptr != b->data.ptr );
    a_data = a->data.ptr;
    a_step = a->step;
    b_data = b->data.ptr;
    b_step = b->step;

    if( flip_type <= 0 )
    {
        a_data += a_step*(rows-1);
        a_step = -a_step;
    }

    for( i = 0; i < rows; i++ )
    {
        if( flip_type == 0 )
            memcpy( b_data, a_data, cols );
        else
        {
            for( j = 0; j < cols; j += elem_size )
                for( k = 0; k < elem_size; k++ )
                    b_data[j+k] = a_data[cols - elem_size - j + k];
        }
        a_data += a_step;
        b_data += b_step;
    }
}


void  cvTsPatchZeros( CvMat* mat, double level )
{
    int i, j, ncols = mat->cols * CV_MAT_CN(mat->type);

    for( i = 0; i < mat->rows; i++ )
    {
        switch( CV_MAT_DEPTH(mat->type) )
        {
        case CV_32F:
            {
            float* data = (float*)(mat->data.ptr + i*mat->step);
            for( j = 0; j < ncols; j++ )
                if( fabs(data[j]) < level )
                    data[j] += 1;
            }
            break;
        case CV_64F:
            {
            double* data = (double*)(mat->data.ptr + i*mat->step);
            for( j = 0; j < ncols; j++ )
                if( fabs(data[j]) < level )
                    data[j] += 1;
            }
            break;
        default:
            assert(0);
            return;
        }
    }
}


/* End of file. */
