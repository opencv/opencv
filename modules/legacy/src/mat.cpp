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

#include "precomp.hpp"

// temporarily remove it from build
#if 0 && ((_MSC_VER>=1200) || defined __BORLANDC__)

double CvMAT::get( const uchar* ptr, int type, int coi )
{
    double t = 0;
    assert( (unsigned)coi < (unsigned)CV_MAT_CN(type) );

    switch( CV_MAT_DEPTH(type) )
    {
    case CV_8U:
        t = ((uchar*)ptr)[coi];
        break;
    case CV_8S:
        t = ((char*)ptr)[coi];
        break;
    case CV_16S:
        t = ((short*)ptr)[coi];
        break;
    case CV_32S:
        t = ((int*)ptr)[coi];
        break;
    case CV_32F:
        t = ((float*)ptr)[coi];
        break;
    case CV_64F:
        t = ((double*)ptr)[coi];
        break;
    }

    return t;
}

void CvMAT::set( uchar* ptr, int type, int coi, double d )
{
    int i;
    assert( (unsigned)coi < (unsigned)CV_MAT_CN(type) );

    switch( CV_MAT_DEPTH(type))
    {
    case CV_8U:
        i = cvRound(d);
        ((uchar*)ptr)[coi] = CV_CAST_8U(i);
        break;
    case CV_8S:
        i = cvRound(d);
        ((char*)ptr)[coi] = CV_CAST_8S(i);
        break;
    case CV_16S:
        i = cvRound(d);
        ((short*)ptr)[coi] = CV_CAST_16S(i);
        break;
    case CV_32S:
        i = cvRound(d);
        ((int*)ptr)[coi] = CV_CAST_32S(i);
        break;
    case CV_32F:
        ((float*)ptr)[coi] = (float)d;
        break;
    case CV_64F:
        ((double*)ptr)[coi] = d;
        break;
    }
}


void CvMAT::set( uchar* ptr, int type, int coi, int i )
{
    assert( (unsigned)coi < (unsigned)CV_MAT_CN(type) );

    switch( CV_MAT_DEPTH(type))
    {
    case CV_8U:
        ((uchar*)ptr)[coi] = CV_CAST_8U(i);
        break;
    case CV_8S:
        ((char*)ptr)[coi] = CV_CAST_8S(i);
        break;
    case CV_16S:
        ((short*)ptr)[coi] = CV_CAST_16S(i);
        break;
    case CV_32S:
        ((int*)ptr)[coi] = i;
        break;
    case CV_32F:
        ((float*)ptr)[coi] = (float)i;
        break;
    case CV_64F:
        ((double*)ptr)[coi] = (double)i;
        break;
    }
}


void CvMAT::set( uchar* ptr, int type, double d )
{
    int i, cn = CV_MAT_CN(type);

    switch( CV_MAT_DEPTH(type))
    {
    case CV_8U:
        i = cvRound(d);
        ((uchar*)ptr)[0] = CV_CAST_8U(i);
        i = cn;
        while( --i ) ((uchar*)ptr)[i] = 0;
        break;
    case CV_8S:
        i = cvRound(d);
        ((char*)ptr)[0] = CV_CAST_8S(i);
        i = cn;
        while( --i ) ((char*)ptr)[i] = 0;
        break;
    case CV_16S:
        i = cvRound(d);
        ((short*)ptr)[0] = CV_CAST_16S(i);
        i = cn;
        while( --i ) ((short*)ptr)[i] = 0;
        break;
    case CV_32S:
        i = cvRound(d);
        ((int*)ptr)[0] = i;
        i = cn;
        while( --i ) ((int*)ptr)[i] = 0;
        break;
    case CV_32F:
        ((float*)ptr)[0] = (float)d;
        i = cn;
        while( --i ) ((float*)ptr)[i] = 0;
        break;
    case CV_64F:
        ((double*)ptr)[0] = d;
        i = cn;
        while( --i ) ((double*)ptr)[i] = 0;
        break;
    }
}


void CvMAT::set( uchar* ptr, int type, int i )
{
    int cn = CV_MAT_CN(type);

    switch( CV_MAT_DEPTH(type))
    {
    case CV_8U:
        ((uchar*)ptr)[0] = CV_CAST_8U(i);
        i = cn;
        while( --i ) ((uchar*)ptr)[i] = 0;
        break;
    case CV_8S:
        ((char*)ptr)[0] = CV_CAST_8S(i);
        i = cn;
        while( --i ) ((char*)ptr)[i] = 0;
        break;
    case CV_16S:
        ((short*)ptr)[0] = CV_CAST_16S(i);
        i = cn;
        while( --i ) ((short*)ptr)[i] = 0;
        break;
    case CV_32S:
        ((int*)ptr)[0] = i;
        i = cn;
        while( --i ) ((int*)ptr)[i] = 0;
        break;
    case CV_32F:
        ((float*)ptr)[0] = (float)i;
        i = cn;
        while( --i ) ((float*)ptr)[i] = 0;
        break;
    case CV_64F:
        ((double*)ptr)[0] = (double)i;
        i = cn;
        while( --i ) ((double*)ptr)[i] = 0;
        break;
    }
}


CvMAT::CvMAT( const _CvMAT_T_& mat_t )
{
    data.ptr = 0;
    type = 0;
    refcount = 0;
    *this = mat_t;
}


CvMAT::CvMAT( const _CvMAT_ADD_& mat_add )
{
    data.ptr = 0;
    type = 0;
    refcount = 0;
    *this = mat_add;
}


CvMAT::CvMAT( const _CvMAT_ADD_EX_& mat_add )
{
    data.ptr = 0;
    type = 0;
    refcount = 0;
    *this = mat_add;
}


CvMAT::CvMAT( const _CvMAT_SCALE_& scale_mat )
{
    data.ptr = 0;
    type = 0;
    refcount = 0;
    *this = scale_mat;
}


CvMAT::CvMAT( const _CvMAT_SCALE_SHIFT_& scale_shift_mat )
{
    data.ptr = 0;
    type = 0;
    refcount = 0;
    *this = scale_shift_mat;
}


CvMAT::CvMAT( const _CvMAT_MUL_& mmul )
{
    data.ptr = 0;
    type = 0;
    refcount = 0;
    *this = mmul;
}


CvMAT::CvMAT( const _CvMAT_MUL_ADD_& mmuladd )
{
    data.ptr = 0;
    type = 0;
    refcount = 0;
    *this = mmuladd;
}


CvMAT::CvMAT( const _CvMAT_INV_& inv_mat )
{
    data.ptr = 0;
    type = 0;
    refcount = 0;
    *this = inv_mat;
}


CvMAT::CvMAT( const _CvMAT_NOT_& not_mat )
{
    type = 0;
    data.ptr = 0;
    refcount = 0;
    *this = not_mat;
}


CvMAT::CvMAT( const _CvMAT_UN_LOGIC_& mat_logic )
{
    type = 0;
    data.ptr = 0;
    refcount = 0;
    *this = mat_logic;
}


CvMAT::CvMAT( const _CvMAT_LOGIC_& mat_logic )
{
    type = 0;
    data.ptr = 0;
    refcount = 0;
    *this = mat_logic;
}


CvMAT::CvMAT( const _CvMAT_COPY_& mat_copy )
{
    CvMAT* src = (CvMAT*)mat_copy.a;
    create( src->height, src->width, src->type );
    cvCopy( src, this );
}


CvMAT::CvMAT( const _CvMAT_CVT_& mat_cvt )
{
    type = 0;
    data.ptr = 0;
    refcount = 0;
    *this = mat_cvt;
}


CvMAT::CvMAT( const _CvMAT_DOT_OP_& dot_op )
{
    data.ptr = 0;
    type = 0;
    refcount = 0;
    *this = dot_op;
}


CvMAT::CvMAT( const _CvMAT_SOLVE_& solve_mat )
{
    type = 0;
    data.ptr = 0;
    refcount = 0;
    *this = solve_mat;
}


CvMAT::CvMAT( const _CvMAT_CMP_& cmp_mat )
{
    type = 0;
    data.ptr = 0;
    refcount = 0;
    *this = cmp_mat;
}


/****************************************************************************************\
*                                  CvMAT::operator =                                     *
\****************************************************************************************/

CvMAT& CvMAT::operator = ( const _CvMAT_T_& mat_t )
{
    CvMAT* src = (CvMAT*)&mat_t.a;
    if( !data.ptr )
    {
        create( src->width, src->height, src->type );
    }

    cvTranspose( src, this );
    return *this;
}


CvMAT& CvMAT::operator = ( const _CvMAT_ADD_& mat_add )
{
    CvMAT* a = mat_add.a;
    CvMAT* b = mat_add.b;

    if( !data.ptr )
    {
        create( a->height, a->width, a->type );
    }

    if( mat_add.beta == 1 )
    {
        cvAdd( a, b, this );
        return *this;
    }

    if( mat_add.beta == -1 )
    {
        cvSub( a, b, this );
        return *this;
    }
    
    if( CV_MAT_DEPTH(a->type) >= CV_32F && CV_MAT_CN(a->type) <= 2 )
        cvScaleAdd( b, cvScalar(mat_add.beta), a, this );
    else
        cvAddWeighted( a, 1, b, mat_add.beta, 0, this );
    return *this;
}


CvMAT& CvMAT::operator = ( const _CvMAT_ADD_EX_& mat_add )
{
    CvMAT* a = mat_add.a;
    CvMAT* b = mat_add.b;

    if( !data.ptr )
    {
        create( a->height, a->width, a->type );
    }

    cvAddWeighted( a, mat_add.alpha, b, mat_add.beta, mat_add.gamma, this );
    return *this;
}


CvMAT& CvMAT::operator = ( const _CvMAT_SCALE_& scale_mat )
{
    CvMAT* src = scale_mat.a;

    if( !data.ptr )
    {
        create( src->height, src->width, src->type );
    }

    cvConvertScale( src, this, scale_mat.alpha, 0 );
    return *this;
}


CvMAT& CvMAT::operator = ( const _CvMAT_SCALE_SHIFT_& scale_shift_mat )
{
    CvMAT* src = scale_shift_mat.a;
    
    if( !data.ptr )
    {
        create( src->height, src->width, src->type );
    }

    cvConvertScale( src, this, scale_shift_mat.alpha, scale_shift_mat.beta );
    return *this;
}


CvMAT& CvMAT::operator = ( const _CvMAT_MUL_& mmul )
{
    CvMAT* a = mmul.a;
    CvMAT* b = mmul.b;
    int t_a = mmul.t_ab & 1;
    int t_b = (mmul.t_ab & 2) != 0;
    int m = (&(a->rows))[t_a];
    int n = (&(b->rows))[t_b ^ 1];
    /* this(m x n) = (a^o1(t))(m x l) * (b^o2(t))(l x n) */

    if( !data.ptr )
    {
        create( m, n, a->type );
    }

    if( mmul.alpha == 1 )
    {
        if( mmul.t_ab == 0 )
        {
            cvMatMulAdd( a, b, 0, this );
            return *this;
        }
        
        if( a->data.ptr == b->data.ptr && mmul.t_ab < 3 &&
            a->rows == b->rows && a->cols == b->cols &&
            a->data.ptr != data.ptr )
        {
            cvMulTransposed( a, this, mmul.t_ab & 1 );
            return *this;
        }
    }

    cvGEMM( a, b, mmul.alpha, 0, 0, this, mmul.t_ab );
    return *this;
}


CvMAT& CvMAT::operator = ( const _CvMAT_MUL_ADD_& mmuladd )
{
    CvMAT* a = mmuladd.a;
    CvMAT* b = mmuladd.b;
    CvMAT* c = mmuladd.c;
    int t_a = mmuladd.t_abc & 1;
    int t_b = (mmuladd.t_abc & 2) != 0;
    int m = (&(a->rows))[t_a];
    int n = (&(b->rows))[t_b ^ 1];
    /* this(m x n) = (a^o1(t))(m x l) * (b^o2(t))(l x n) */

    if( !data.ptr )
    {
        create( m, n, a->type );
    }

    if( mmuladd.t_abc == 0 && mmuladd.alpha == 1 && mmuladd.beta == 1 )
        cvMatMulAdd( a, b, c, this );
    else
        cvGEMM( a, b, mmuladd.alpha, c, mmuladd.beta, this, mmuladd.t_abc );
    return *this;
}


CvMAT& CvMAT::operator = ( const _CvMAT_INV_& inv_mat )
{
    CvMAT* src = (CvMAT*)&inv_mat.a;
    
    if( !data.ptr )
    {
        create( src->height, src->width, src->type );
    }

    if( inv_mat.method == 0 )
        cvInvert( src, this );
    else
        cvPseudoInv( src, this );
    return *this;
}


CvMAT& CvMAT::operator = ( const _CvMAT_NOT_& not_mat )
{
    CvMAT* src = not_mat.a;
    
    if( !data.ptr )
    {
        create( src->height, src->width, src->type );
    }

    cvNot( src, this );
    return *this;
}


CvMAT& CvMAT::operator = ( const _CvMAT_LOGIC_& mat_logic )
{
    CvMAT* a = mat_logic.a;
    CvMAT* b = mat_logic.b;
    int flags = mat_logic.flags;
    _CvMAT_LOGIC_::Op op = mat_logic.op;

    if( !data.ptr )
    {
        create( a->height, a->width, a->type );
    }

    switch( op )
    {
    case _CvMAT_LOGIC_::AND:

        if( flags == 0 )
            cvAnd( a, b, this );
        else if( flags == 3 )
        {
            cvOr( a, b, this );
            cvNot( this, this );
        }
        else if( flags == 1 )
        {
            if( data.ptr == b->data.ptr )
            {
                cvNot( b, this );
                cvOr( this, a, this );
                cvNot( this, this );
            }
            else
            {
                cvNot( a, this );
                cvAnd( this, b, this );
            }
        }
        else
        {
            if( data.ptr == a->data.ptr )
            {
                cvNot( a, this );
                cvOr( this, b, this );
                cvNot( this, this );
            }
            else
            {
                cvNot( b, this );
                cvAnd( this, a, this );
            }
        }
        break;

    case _CvMAT_LOGIC_::OR:

        if( flags == 0 )
            cvOr( a, b, this );
        else if( flags == 3 )
        {
            cvAnd( a, b, this );
            cvNot( this, this );
        }
        else if( flags == 1 )
        {
            if( data.ptr == b->data.ptr )
            {
                cvNot( b, this );
                cvAnd( this, a, this );
                cvNot( this, this );
            }
            else
            {
                cvNot( a, this );
                cvOr( this, b, this );
            }
        }
        else
        {
            if( data.ptr == a->data.ptr )
            {
                cvNot( a, this );
                cvAnd( this, b, this );
                cvNot( this, this );
            }
            else
            {
                cvNot( b, this );
                cvOr( this, a, this );
            }
        }
        break;

    case _CvMAT_LOGIC_::XOR:

        cvXor( a, b, this );
        if( flags == 1 || flags == 2 )
            cvNot( this, this );
        break;
    }

    return *this;
}


CvMAT& CvMAT::operator = ( const _CvMAT_UN_LOGIC_& mat_logic )
{
    CvMAT* a = mat_logic.a;
    CvScalar scalar = cvScalarAll( mat_logic.alpha );
    int flags = mat_logic.flags;
    _CvMAT_LOGIC_::Op op = mat_logic.op;

    if( !data.ptr )
    {
        create( a->height, a->width, a->type );
    }

    switch( op )
    {
    case _CvMAT_LOGIC_::AND:

        if( flags == 0 )
            cvAndS( a, scalar, this );
        else
        {
            cvNot( a, this );
            cvAndS( this, scalar, this );
        }
        break;

    case _CvMAT_LOGIC_::OR:

        if( flags == 0 )
            cvOrS( a, scalar, this );
        else
        {
            cvNot( a, this );
            cvOrS( this, scalar, this );
        }
        break;

    case _CvMAT_LOGIC_::XOR:

        if( flags == 0 )
            cvXorS( a, scalar, this );
        else
            cvXorS( a, ~scalar, this );
        break;
    }

    return *this;
}


CvMAT& CvMAT::operator = ( const _CvMAT_COPY_& mat_copy )
{
    CvMAT* src = (CvMAT*)mat_copy.a;

    if( !data.ptr )
    {
        create( src->height, src->width, src->type );
    }

    if( src != this )
        cvCopy( src, this );

    return *this;
}


CvMAT& CvMAT::operator = ( const _CvMAT_CVT_& mat_cvt )
{
    CvMAT* src = (CvMAT*)&mat_cvt.a;
    
    if( !data.ptr )
    {
        int depth = mat_cvt.newdepth;
        create( src->height, src->width, depth < 0 ? src->type :
                CV_MAT_CN(src->type)|CV_MAT_DEPTH(depth));
    }

    cvCvtScale( src, this, mat_cvt.scale, mat_cvt.shift );
    return *this;
}


CvMAT& CvMAT::operator = ( const _CvMAT_DOT_OP_& dot_op )
{
    CvMAT* a = (CvMAT*)&(dot_op.a);
    CvMAT* b = dot_op.b;
    
    if( !data.ptr )
    {
        create( a->height, a->width, a->type );
    }

    switch( dot_op.op )
    {
    case '*':
        cvMul( a, b, this, dot_op.alpha );
        break;
    case '/':
        if( b != 0 )
            cvDiv( a, b, this, dot_op.alpha );
        else
            cvDiv( 0, a, this, dot_op.alpha );
        break;
    case 'm':
        if( b != 0 )
            cvMin( a, b, this );
        else
            cvMinS( a, dot_op.alpha, this );
        break;
    case 'M':
        if( b != 0 )
            cvMax( a, b, this );
        else
            cvMaxS( a, dot_op.alpha, this );
        break;
    case 'a':
        if( b != 0 )
            cvAbsDiff( a, b, this );
        else
            cvAbsDiffS( a, this, cvScalar(dot_op.alpha) );
        break;
    default:
        assert(0);
    }

    return *this;
}


CvMAT& CvMAT::operator = ( const _CvMAT_SOLVE_& solve_mat )
{
    CvMAT* a = (CvMAT*)(solve_mat.a);
    CvMAT* b = (CvMAT*)(solve_mat.b);

    if( !data.ptr )
    {
        create( a->height, b->width, a->type );
    }

    if( solve_mat.method == 0 )
        cvSolve( a, b, this );
    else
    {
        CvMAT temp;
        cvInitMatHeader( &temp, a->cols, a->rows, a->type );
        cvCreateData( &temp );

        cvPseudoInv( a, &temp );
        cvMatMul( &temp, b, this );
    }

    return *this;
}


CvMAT& CvMAT::operator = ( const _CvMAT_CMP_& mat_cmp )
{
    CvMAT* a = mat_cmp.a;
    CvMAT* b = mat_cmp.b;

    if( !data.ptr )
    {
        create( a->height, a->width, CV_8UC1 );
    }

    if( b )
        cvCmp( a, b, this, mat_cmp.cmp_op );
    else
        cvCmpS( a, mat_cmp.alpha, this, mat_cmp.cmp_op );
    return *this;
}


/****************************************************************************************\
*                                  CvMAT I/O operations                                  *
\****************************************************************************************/

void  CvMAT::write( const char* name, FILE* f, const char* fmt )
{
    int i, j, w = width * CV_MAT_CN(type);
    FILE* out = f ? f : stdout;

    if( name )
        fprintf( stdout, "%s(%d x %d) =\n\t", name, rows, cols );
    
    for( i = 0; i < rows; i++ )
    {
        switch( CV_MAT_DEPTH(type))
        {
        case CV_8U: if( !fmt )
                        fmt = "%4d";
                    for( j = 0; j < w; j++ )
                        fprintf( out, fmt, ((uchar*)(data.ptr + i*step))[j] );
                    break;
        case CV_8S: if( !fmt )
                        fmt = "%5d";
                    for( j = 0; j < w; j++ )
                        fprintf( out, fmt, ((char*)(data.ptr + i*step))[j] );
                    break;
        case CV_16S: if( !fmt )
                        fmt = "%7d";
                    for( j = 0; j < w; j++ )
                        fprintf( out, fmt, ((short*)(data.ptr + i*step))[j] );
                    break;
        case CV_32S: if( !fmt )
                        fmt = " %08x";
                    for( j = 0; j < w; j++ )
                        fprintf( out, fmt, ((int*)(data.ptr + i*step))[j] );
                    break;
        case CV_32F: if( !fmt )
                        fmt = "%15g";
                    for( j = 0; j < w; j++ )
                        fprintf( out, fmt, ((float*)(data.ptr + i*step))[j] );
                    break;
        case CV_64F: if( !fmt )
                        fmt = "%15g";
                    for( j = 0; j < w; j++ )
                        fprintf( out, fmt, ((double*)(data.ptr + i*step))[j] );
                    break;
        }
        fprintf( out, "\n%s", i < rows - 1 ? "\t" : "" );
    }
    fprintf( out, "\n" );
}

#endif /* _MSC_VER || __BORLANDC__ */

/* End of file. */


