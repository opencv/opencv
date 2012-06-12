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

static CvStatus
icvJacobiEigens_32f(float *A, float *V, float *E, int n, float eps)
{
    int i, j, k, ind;
    float *AA = A, *VV = V;
    double Amax, anorm = 0, ax;

    if( A == NULL || V == NULL || E == NULL )
        return CV_NULLPTR_ERR;
    if( n <= 0 )
        return CV_BADSIZE_ERR;
    if( eps < 1.0e-7f )
        eps = 1.0e-7f;

    /*-------- Prepare --------*/
    for( i = 0; i < n; i++, VV += n, AA += n )
    {
        for( j = 0; j < i; j++ )
        {
            double Am = AA[j];

            anorm += Am * Am;
        }
        for( j = 0; j < n; j++ )
            VV[j] = 0.f;
        VV[i] = 1.f;
    }

    anorm = sqrt( anorm + anorm );
    ax = anorm * eps / n;
    Amax = anorm;

    while( Amax > ax )
    {
        Amax /= n;
        do                      /* while (ind) */
        {
            int p, q;
            float *V1 = V, *A1 = A;

            ind = 0;
            for( p = 0; p < n - 1; p++, A1 += n, V1 += n )
            {
                float *A2 = A + n * (p + 1), *V2 = V + n * (p + 1);

                for( q = p + 1; q < n; q++, A2 += n, V2 += n )
                {
                    double x, y, c, s, c2, s2, a;
                    float *A3, Apq = A1[q], App, Aqq, Aip, Aiq, Vpi, Vqi;

                    if( fabs( Apq ) < Amax )
                        continue;

                    ind = 1;

                    /*---- Calculation of rotation angle's sine & cosine ----*/
                    App = A1[p];
                    Aqq = A2[q];
                    y = 5.0e-1 * (App - Aqq);
                    x = -Apq / sqrt( (double)Apq * Apq + (double)y * y );
                    if( y < 0.0 )
                        x = -x;
                    s = x / sqrt( 2.0 * (1.0 + sqrt( 1.0 - (double)x * x )));
                    s2 = s * s;
                    c = sqrt( 1.0 - s2 );
                    c2 = c * c;
                    a = 2.0 * Apq * c * s;

                    /*---- Apq annulation ----*/
                    A3 = A;
                    for( i = 0; i < p; i++, A3 += n )
                    {
                        Aip = A3[p];
                        Aiq = A3[q];
                        Vpi = V1[i];
                        Vqi = V2[i];
                        A3[p] = (float) (Aip * c - Aiq * s);
                        A3[q] = (float) (Aiq * c + Aip * s);
                        V1[i] = (float) (Vpi * c - Vqi * s);
                        V2[i] = (float) (Vqi * c + Vpi * s);
                    }
                    for( ; i < q; i++, A3 += n )
                    {
                        Aip = A1[i];
                        Aiq = A3[q];
                        Vpi = V1[i];
                        Vqi = V2[i];
                        A1[i] = (float) (Aip * c - Aiq * s);
                        A3[q] = (float) (Aiq * c + Aip * s);
                        V1[i] = (float) (Vpi * c - Vqi * s);
                        V2[i] = (float) (Vqi * c + Vpi * s);
                    }
                    for( ; i < n; i++ )
                    {
                        Aip = A1[i];
                        Aiq = A2[i];
                        Vpi = V1[i];
                        Vqi = V2[i];
                        A1[i] = (float) (Aip * c - Aiq * s);
                        A2[i] = (float) (Aiq * c + Aip * s);
                        V1[i] = (float) (Vpi * c - Vqi * s);
                        V2[i] = (float) (Vqi * c + Vpi * s);
                    }
                    A1[p] = (float) (App * c2 + Aqq * s2 - a);
                    A2[q] = (float) (App * s2 + Aqq * c2 + a);
                    A1[q] = A2[p] = 0.0f;
                }               /*q */
            }                   /*p */
        }
        while( ind );
        Amax /= n;
    }                           /* while ( Amax > ax ) */

    for( i = 0, k = 0; i < n; i++, k += n + 1 )
        E[i] = A[k];
    /*printf(" M = %d\n", M); */

    /* -------- ordering -------- */
    for( i = 0; i < n; i++ )
    {
        int m = i;
        float Em = (float) fabs( E[i] );

        for( j = i + 1; j < n; j++ )
        {
            float Ej = (float) fabs( E[j] );

            m = (Em < Ej) ? j : m;
            Em = (Em < Ej) ? Ej : Em;
        }
        if( m != i )
        {
            int l;
            float b = E[i];

            E[i] = E[m];
            E[m] = b;
            for( j = 0, k = i * n, l = m * n; j < n; j++, k++, l++ )
            {
                b = V[k];
                V[k] = V[l];
                V[l] = b;
            }
        }
    }

    return CV_NO_ERR;
}

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: icvCalcCovarMatrixEx_8u32fR
//    Purpose: The function calculates a covariance matrix for a group of input objects
//             (images, vectors, etc.). ROI supported.
//    Context:
//    Parameters:  nObjects    - number of source objects
//                 objects     - array of pointers to ROIs of the source objects
//                 imgStep     - full width of each source object row in bytes
//                 avg         - pointer to averaged object
//                 avgStep     - full width of averaged object row in bytes
//                 size        - ROI size of each source and averaged objects
//                 covarMatrix - covariance matrix (output parameter; must be allocated
//                               before call)
//
//    Returns: CV_NO_ERR or error code
//
//    Notes:
//F*/
static CvStatus  CV_STDCALL
icvCalcCovarMatrixEx_8u32fR( int nObjects, void *input, int objStep1,
                             int ioFlags, int ioBufSize, uchar* buffer,
                             void *userData, float *avg, int avgStep,
                             CvSize size, float *covarMatrix )
{
    int objStep = objStep1;

    /* ---- TEST OF PARAMETERS ---- */

    if( nObjects < 2 )
        return CV_BADFACTOR_ERR;
    if( ioFlags < 0 || ioFlags > 3 )
        return CV_BADFACTOR_ERR;
    if( ioFlags && ioBufSize < 1024 )
        return CV_BADFACTOR_ERR;
    if( ioFlags && buffer == NULL )
        return CV_NULLPTR_ERR;
    if( input == NULL || avg == NULL || covarMatrix == NULL )
        return CV_NULLPTR_ERR;
    if( size.width > objStep || 4 * size.width > avgStep || size.height < 1 )
        return CV_BADSIZE_ERR;

    avgStep /= 4;

    if( ioFlags & CV_EIGOBJ_INPUT_CALLBACK )    /* ==== USE INPUT CALLBACK ==== */
    {
        int nio, ngr, igr, n = size.width * size.height, mm = 0;
        CvCallback read_callback = ((CvInput *) & input)->callback;
        uchar *buffer2;

        objStep = n;
        nio = ioBufSize / n;    /* number of objects in buffer */
        ngr = nObjects / nio;   /* number of io groups */
        if( nObjects % nio )
            mm = 1;
        ngr += mm;

        buffer2 = (uchar *)cvAlloc( sizeof( uchar ) * n );
        if( buffer2 == NULL )
            return CV_OUTOFMEM_ERR;

        for( igr = 0; igr < ngr; igr++ )
        {
            int k, l;
            int io, jo, imin = igr * nio, imax = imin + nio;
            uchar *bu1 = buffer, *bu2;

            if( imax > nObjects )
                imax = nObjects;

            /* read igr group */
            for( io = imin; io < imax; io++, bu1 += n )
            {
                CvStatus r;

                r = (CvStatus)read_callback( io, (void *) bu1, userData );
                if( r )
                    return r;
            }

            /* diagonal square calc */
            bu1 = buffer;
            for( io = imin; io < imax; io++, bu1 += n )
            {
                bu2 = bu1;
                for( jo = io; jo < imax; jo++, bu2 += n )
                {
                    float w = 0.f;
                    float *fu = avg;
                    int ij = 0;

                    for( k = 0; k < size.height; k++, fu += avgStep )
                        for( l = 0; l < size.width; l++, ij++ )
                        {
                            float f = fu[l], u1 = bu1[ij], u2 = bu2[ij];

                            w += (u1 - f) * (u2 - f);
                        }
                    covarMatrix[io * nObjects + jo] = covarMatrix[jo * nObjects + io] = w;
                }
            }

            /* non-diagonal elements calc */
            for( jo = imax; jo < nObjects; jo++ )
            {
                CvStatus r;

                bu1 = buffer;
                bu2 = buffer2;

                /* read jo object */
                r = (CvStatus)read_callback( jo, (void *) bu2, userData );
                if( r )
                    return r;

                for( io = imin; io < imax; io++, bu1 += n )
                {
                    float w = 0.f;
                    float *fu = avg;
                    int ij = 0;

                    for( k = 0; k < size.height; k++, fu += avgStep )
                    {
                        for( l = 0; l < size.width - 3; l += 4, ij += 4 )
                        {
                            float f = fu[l];
                            uchar u1 = bu1[ij];
                            uchar u2 = bu2[ij];

                            w += (u1 - f) * (u2 - f);
                            f = fu[l + 1];
                            u1 = bu1[ij + 1];
                            u2 = bu2[ij + 1];
                            w += (u1 - f) * (u2 - f);
                            f = fu[l + 2];
                            u1 = bu1[ij + 2];
                            u2 = bu2[ij + 2];
                            w += (u1 - f) * (u2 - f);
                            f = fu[l + 3];
                            u1 = bu1[ij + 3];
                            u2 = bu2[ij + 3];
                            w += (u1 - f) * (u2 - f);
                        }
                        for( ; l < size.width; l++, ij++ )
                        {
                            float f = fu[l], u1 = bu1[ij], u2 = bu2[ij];

                            w += (u1 - f) * (u2 - f);
                        }
                    }
                    covarMatrix[io * nObjects + jo] = covarMatrix[jo * nObjects + io] = w;
                }
            }
        }                       /* igr */

        cvFree( &buffer2 );
    }                           /* if() */

    else
        /* ==== NOT USE INPUT CALLBACK ==== */
    {
        int i, j;
        uchar **objects = (uchar **) (((CvInput *) & input)->data);

        for( i = 0; i < nObjects; i++ )
        {
            uchar *bu = objects[i];

            for( j = i; j < nObjects; j++ )
            {
                int k, l;
                float w = 0.f;
                float *a = avg;
                uchar *bu1 = bu;
                uchar *bu2 = objects[j];

                for( k = 0; k < size.height;
                     k++, bu1 += objStep, bu2 += objStep, a += avgStep )
                {
                    for( l = 0; l < size.width - 3; l += 4 )
                    {
                        float f = a[l];
                        uchar u1 = bu1[l];
                        uchar u2 = bu2[l];

                        w += (u1 - f) * (u2 - f);
                        f = a[l + 1];
                        u1 = bu1[l + 1];
                        u2 = bu2[l + 1];
                        w += (u1 - f) * (u2 - f);
                        f = a[l + 2];
                        u1 = bu1[l + 2];
                        u2 = bu2[l + 2];
                        w += (u1 - f) * (u2 - f);
                        f = a[l + 3];
                        u1 = bu1[l + 3];
                        u2 = bu2[l + 3];
                        w += (u1 - f) * (u2 - f);
                    }
                    for( ; l < size.width; l++ )
                    {
                        float f = a[l];
                        uchar u1 = bu1[l];
                        uchar u2 = bu2[l];

                        w += (u1 - f) * (u2 - f);
                    }
                }

                covarMatrix[i * nObjects + j] = covarMatrix[j * nObjects + i] = w;
            }
        }
    }                           /* else */

    return CV_NO_ERR;
}

/*======================== end of icvCalcCovarMatrixEx_8u32fR ===========================*/


static int
icvDefaultBufferSize( void )
{
    return 10 * 1024 * 1024;
}

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: icvCalcEigenObjects_8u32fR
//    Purpose: The function calculates an orthonormal eigen basis and a mean (averaged)
//             object for a group of input objects (images, vectors, etc.). ROI supported.
//    Context:
//    Parameters: nObjects  - number of source objects
//                input     - pointer either to array of pointers to input objects
//                            or to read callback function (depending on ioFlags)
//                imgStep   - full width of each source object row in bytes
//                output    - pointer either to array of pointers to output eigen objects
//                            or to write callback function (depending on ioFlags)
//                eigStep   - full width of each eigenobject row in bytes
//                size      - ROI size of each source object
//                ioFlags   - input/output flags (see Notes)
//                ioBufSize - input/output buffer size
//                userData  - pointer to the structure which contains all necessary
//                            data for the callback functions
//                calcLimit - determines the calculation finish conditions
//                avg       - pointer to averaged object (has the same size as ROI)
//                avgStep   - full width of averaged object row in bytes
//                eigVals   - pointer to corresponding eigenvalues (array of <nObjects>
//                            elements in descending order)
//
//    Returns: CV_NO_ERR or error code
//
//    Notes: 1. input/output data (that is, input objects and eigen ones) may either
//              be allocated in the RAM or be read from/written to the HDD (or any
//              other device) by read/write callback functions. It depends on the
//              value of ioFlags paramater, which may be the following:
//                  CV_EIGOBJ_NO_CALLBACK, or 0;
//                  CV_EIGOBJ_INPUT_CALLBACK;
//                  CV_EIGOBJ_OUTPUT_CALLBACK;
//                  CV_EIGOBJ_BOTH_CALLBACK, or
//                            CV_EIGOBJ_INPUT_CALLBACK | CV_EIGOBJ_OUTPUT_CALLBACK.
//              The callback functions as well as the user data structure must be
//              developed by the user.
//
//           2. If ioBufSize = 0, or it's too large, the function dermines buffer size
//              itself.
//
//           3. Depending on calcLimit parameter, calculations are finished either if
//              eigenfaces number comes up to certain value or the relation of the
//              current eigenvalue and the largest one comes down to certain value
//              (or any of the above conditions takes place). The calcLimit->type value
//              must be CV_TERMCRIT_NUMB, CV_TERMCRIT_EPS or
//              CV_TERMCRIT_NUMB | CV_TERMCRIT_EPS. The function returns the real
//              values calcLimit->max_iter and calcLimit->epsilon.
//
//           4. eigVals may be equal to NULL (if you don't need eigen values in further).
//
//F*/
static CvStatus CV_STDCALL
icvCalcEigenObjects_8u32fR( int nObjects, void* input, int objStep,
                            void* output, int eigStep, CvSize size,
                            int  ioFlags, int ioBufSize, void* userData,
                            CvTermCriteria* calcLimit, float* avg,
                            int    avgStep, float  *eigVals )
{
    int i, j, n, iev = 0, m1 = nObjects - 1, objStep1 = objStep, eigStep1 = eigStep / 4;
    CvSize objSize, eigSize, avgSize;
    float *c = 0;
    float *ev = 0;
    float *bf = 0;
    uchar *buf = 0;
    void *buffer = 0;
    float m = 1.0f / (float) nObjects;
    CvStatus r;

    if( m1 > calcLimit->max_iter && calcLimit->type != CV_TERMCRIT_EPS )
        m1 = calcLimit->max_iter;

    /* ---- TEST OF PARAMETERS ---- */

    if( nObjects < 2 )
        return CV_BADFACTOR_ERR;
    if( ioFlags < 0 || ioFlags > 3 )
        return CV_BADFACTOR_ERR;
    if( input == NULL || output == NULL || avg == NULL )
        return CV_NULLPTR_ERR;
    if( size.width > objStep || 4 * size.width > eigStep ||
        4 * size.width > avgStep || size.height < 1 )
        return CV_BADSIZE_ERR;
    if( !(ioFlags & CV_EIGOBJ_INPUT_CALLBACK) )
        for( i = 0; i < nObjects; i++ )
            if( ((uchar **) input)[i] == NULL )
                return CV_NULLPTR_ERR;
    if( !(ioFlags & CV_EIGOBJ_OUTPUT_CALLBACK) )
        for( i = 0; i < m1; i++ )
            if( ((float **) output)[i] == NULL )
                return CV_NULLPTR_ERR;

    avgStep /= 4;
    eigStep /= 4;

    if( objStep == size.width && eigStep == size.width && avgStep == size.width )
    {
        size.width *= size.height;
        size.height = 1;
        objStep = objStep1 = eigStep = eigStep1 = avgStep = size.width;
    }
    objSize = eigSize = avgSize = size;

    if( ioFlags & CV_EIGOBJ_INPUT_CALLBACK )
    {
        objSize.width *= objSize.height;
        objSize.height = 1;
        objStep = objSize.width;
        objStep1 = size.width;
    }

    if( ioFlags & CV_EIGOBJ_OUTPUT_CALLBACK )
    {
        eigSize.width *= eigSize.height;
        eigSize.height = 1;
        eigStep = eigSize.width;
        eigStep1 = size.width;
    }

    n = objSize.height * objSize.width * (ioFlags & CV_EIGOBJ_INPUT_CALLBACK) +
        2 * eigSize.height * eigSize.width * (ioFlags & CV_EIGOBJ_OUTPUT_CALLBACK);

    /* Buffer size determination */
    if( ioFlags )
    {
        ioBufSize = MIN( icvDefaultBufferSize(), n );
    }

    /* memory allocation (if necesseay) */

    if( ioFlags & CV_EIGOBJ_INPUT_CALLBACK )
    {
        buf = (uchar *) cvAlloc( sizeof( uchar ) * objSize.width );
        if( buf == NULL )
            return CV_OUTOFMEM_ERR;
    }

    if( ioFlags )
    {
        buffer = (void *) cvAlloc( ioBufSize );
        if( buffer == NULL )
        {
            if( buf )
                cvFree( &buf );
            return CV_OUTOFMEM_ERR;
        }
    }

    /* Calculation of averaged object */
    bf = avg;
    for( i = 0; i < avgSize.height; i++, bf += avgStep )
        for( j = 0; j < avgSize.width; j++ )
            bf[j] = 0.f;

    for( i = 0; i < nObjects; i++ )
    {
        int k, l;
        uchar *bu = (ioFlags & CV_EIGOBJ_INPUT_CALLBACK) ? buf : ((uchar **) input)[i];

        if( ioFlags & CV_EIGOBJ_INPUT_CALLBACK )
        {
            CvCallback read_callback = ((CvInput *) & input)->callback;

            r = (CvStatus)read_callback( i, (void *) buf, userData );
            if( r )
            {
                if( buffer )
                    cvFree( &buffer );
                if( buf )
                    cvFree( &buf );
                return r;
            }
        }

        bf = avg;
        for( k = 0; k < avgSize.height; k++, bf += avgStep, bu += objStep1 )
            for( l = 0; l < avgSize.width; l++ )
                bf[l] += bu[l];
    }

    bf = avg;
    for( i = 0; i < avgSize.height; i++, bf += avgStep )
        for( j = 0; j < avgSize.width; j++ )
            bf[j] *= m;

    /* Calculation of covariance matrix */
    c = (float *) cvAlloc( sizeof( float ) * nObjects * nObjects );

    if( c == NULL )
    {
        if( buffer )
            cvFree( &buffer );
        if( buf )
            cvFree( &buf );
        return CV_OUTOFMEM_ERR;
    }

    r = icvCalcCovarMatrixEx_8u32fR( nObjects, input, objStep1, ioFlags, ioBufSize,
                                      (uchar *) buffer, userData, avg, 4 * avgStep, size, c );
    if( r )
    {
        cvFree( &c );
        if( buffer )
            cvFree( &buffer );
        if( buf )
            cvFree( &buf );
        return r;
    }

    /* Calculation of eigenvalues & eigenvectors */
    ev = (float *) cvAlloc( sizeof( float ) * nObjects * nObjects );

    if( ev == NULL )
    {
        cvFree( &c );
        if( buffer )
            cvFree( &buffer );
        if( buf )
            cvFree( &buf );
        return CV_OUTOFMEM_ERR;
    }

    if( eigVals == NULL )
    {
        eigVals = (float *) cvAlloc( sizeof( float ) * nObjects );

        if( eigVals == NULL )
        {
            cvFree( &c );
            cvFree( &ev );
            if( buffer )
                cvFree( &buffer );
            if( buf )
                cvFree( &buf );
            return CV_OUTOFMEM_ERR;
        }
        iev = 1;
    }

    r = icvJacobiEigens_32f( c, ev, eigVals, nObjects, 0.0f );
    cvFree( &c );
    if( r )
    {
        cvFree( &ev );
        if( buffer )
            cvFree( &buffer );
        if( buf )
            cvFree( &buf );
        if( iev )
            cvFree( &eigVals );
        return r;
    }

    /* Eigen objects number determination */
    if( calcLimit->type != CV_TERMCRIT_NUMBER )
    {
        for( i = 0; i < m1; i++ )
            if( fabs( eigVals[i] / eigVals[0] ) < calcLimit->epsilon )
                break;
        m1 = calcLimit->max_iter = i;
    }
    else
        m1 = calcLimit->max_iter;
    calcLimit->epsilon = (float) fabs( eigVals[m1 - 1] / eigVals[0] );

    for( i = 0; i < m1; i++ )
        eigVals[i] = (float) (1.0 / sqrt( (double)eigVals[i] ));

    /* ----------------- Calculation of eigenobjects ----------------------- */
    if( ioFlags & CV_EIGOBJ_OUTPUT_CALLBACK )
    {
        int nio, ngr, igr;

        nio = ioBufSize / (4 * eigSize.width);  /* number of eigen objects in buffer */
        ngr = m1 / nio;         /* number of io groups */
        if( nObjects % nio )
            ngr += 1;

        for( igr = 0; igr < ngr; igr++ )
        {
            int io, ie, imin = igr * nio, imax = imin + nio;

            if( imax > m1 )
                imax = m1;

            for(int k = 0; k < eigSize.width * (imax - imin); k++ )
                ((float *) buffer)[k] = 0.f;

            for( io = 0; io < nObjects; io++ )
            {
                uchar *bu = ioFlags & CV_EIGOBJ_INPUT_CALLBACK ? buf : ((uchar **) input)[io];

                if( ioFlags & CV_EIGOBJ_INPUT_CALLBACK )
                {
                    CvCallback read_callback = ((CvInput *) & input)->callback;

                    r = (CvStatus)read_callback( io, (void *) buf, userData );
                    if( r )
                    {
                        cvFree( &ev );
                        if( iev )
                            cvFree( &eigVals );
                        if( buffer )
                            cvFree( &buffer );
                        if( buf )
                            cvFree( &buf );
                        return r;
                    }
                }

                for( ie = imin; ie < imax; ie++ )
                {
                    int k, l;
                    uchar *bv = bu;
                    float e = ev[ie * nObjects + io] * eigVals[ie];
                    float *be = ((float *) buffer) + ((ie - imin) * eigStep);

                    bf = avg;
                    for( k = 0; k < size.height; k++, bv += objStep1,
                         bf += avgStep, be += eigStep1 )
                    {
                        for( l = 0; l < size.width - 3; l += 4 )
                        {
                            float f = bf[l];
                            uchar v = bv[l];

                            be[l] += e * (v - f);
                            f = bf[l + 1];
                            v = bv[l + 1];
                            be[l + 1] += e * (v - f);
                            f = bf[l + 2];
                            v = bv[l + 2];
                            be[l + 2] += e * (v - f);
                            f = bf[l + 3];
                            v = bv[l + 3];
                            be[l + 3] += e * (v - f);
                        }
                        for( ; l < size.width; l++ )
                            be[l] += e * (bv[l] - bf[l]);
                    }
                }
            }                   /* io */

            for( ie = imin; ie < imax; ie++ )   /* calculated eigen objects writting */
            {
                CvCallback write_callback = ((CvInput *) & output)->callback;
                float *be = ((float *) buffer) + ((ie - imin) * eigStep);

                r = (CvStatus)write_callback( ie, (void *) be, userData );
                if( r )
                {
                    cvFree( &ev );
                    if( iev )
                        cvFree( &eigVals );
                    if( buffer )
                        cvFree( &buffer );
                    if( buf )
                        cvFree( &buf );
                    return r;
                }
            }
        }                       /* igr */
    }

    else
    {
        int k, p, l;

        for( i = 0; i < m1; i++ )       /* e.o. annulation */
        {
            float *be = ((float **) output)[i];

            for( p = 0; p < eigSize.height; p++, be += eigStep )
                for( l = 0; l < eigSize.width; l++ )
                    be[l] = 0.0f;
        }

        for( k = 0; k < nObjects; k++ )
        {
            uchar *bv = (ioFlags & CV_EIGOBJ_INPUT_CALLBACK) ? buf : ((uchar **) input)[k];

            if( ioFlags & CV_EIGOBJ_INPUT_CALLBACK )
            {
                CvCallback read_callback = ((CvInput *) & input)->callback;

                r = (CvStatus)read_callback( k, (void *) buf, userData );
                if( r )
                {
                    cvFree( &ev );
                    if( iev )
                        cvFree( &eigVals );
                    if( buffer )
                        cvFree( &buffer );
                    if( buf )
                        cvFree( &buf );
                    return r;
                }
            }

            for( i = 0; i < m1; i++ )
            {
                float v = eigVals[i] * ev[i * nObjects + k];
                float *be = ((float **) output)[i];
                uchar *bu = bv;

                bf = avg;

                for( p = 0; p < size.height; p++, bu += objStep1,
                     bf += avgStep, be += eigStep1 )
                {
                    for( l = 0; l < size.width - 3; l += 4 )
                    {
                        float f = bf[l];
                        uchar u = bu[l];

                        be[l] += v * (u - f);
                        f = bf[l + 1];
                        u = bu[l + 1];
                        be[l + 1] += v * (u - f);
                        f = bf[l + 2];
                        u = bu[l + 2];
                        be[l + 2] += v * (u - f);
                        f = bf[l + 3];
                        u = bu[l + 3];
                        be[l + 3] += v * (u - f);
                    }
                    for( ; l < size.width; l++ )
                        be[l] += v * (bu[l] - bf[l]);
                }
            }                   /* i */
        }                       /* k */
    }                           /* else */

    cvFree( &ev );
    if( iev )
        cvFree( &eigVals );
    else
        for( i = 0; i < m1; i++ )
            eigVals[i] = 1.f / (eigVals[i] * eigVals[i]);
    if( buffer )
        cvFree( &buffer );
    if( buf )
        cvFree( &buf );
    return CV_NO_ERR;
}

/* --- End of icvCalcEigenObjects_8u32fR --- */

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: icvCalcDecompCoeff_8u32fR
//    Purpose: The function calculates one decomposition coefficient of input object
//             using previously calculated eigen object and the mean (averaged) object
//    Context:
//    Parameters:  obj     - input object
//                 objStep - its step (in bytes)
//                 eigObj  - pointer to eigen object
//                 eigStep - its step (in bytes)
//                 avg     - pointer to averaged object
//                 avgStep - its step (in bytes)
//                 size    - ROI size of each source object
//
//    Returns: decomposition coefficient value or large negative value (if error)
//
//    Notes:
//F*/
static float CV_STDCALL
icvCalcDecompCoeff_8u32fR( uchar* obj, int objStep,
                           float *eigObj, int eigStep,
                           float *avg, int avgStep, CvSize size )
{
    int i, k;
    float w = 0.0f;

    if( size.width > objStep || 4 * size.width > eigStep
        || 4 * size.width > avgStep || size.height < 1 )
        return -1.0e30f;
    if( obj == NULL || eigObj == NULL || avg == NULL )
        return -1.0e30f;

    eigStep /= 4;
    avgStep /= 4;

    if( size.width == objStep && size.width == eigStep && size.width == avgStep )
    {
        size.width *= size.height;
        size.height = 1;
        objStep = eigStep = avgStep = size.width;
    }

    for( i = 0; i < size.height; i++, obj += objStep, eigObj += eigStep, avg += avgStep )
    {
        for( k = 0; k < size.width - 4; k += 4 )
        {
            float o = (float) obj[k];
            float e = eigObj[k];
            float a = avg[k];

            w += e * (o - a);
            o = (float) obj[k + 1];
            e = eigObj[k + 1];
            a = avg[k + 1];
            w += e * (o - a);
            o = (float) obj[k + 2];
            e = eigObj[k + 2];
            a = avg[k + 2];
            w += e * (o - a);
            o = (float) obj[k + 3];
            e = eigObj[k + 3];
            a = avg[k + 3];
            w += e * (o - a);
        }
        for( ; k < size.width; k++ )
            w += eigObj[k] * ((float) obj[k] - avg[k]);
    }

    return w;
}

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Names: icvEigenDecomposite_8u32fR
//    Purpose: The function calculates all decomposition coefficients for input object
//             using previously calculated eigen objects basis and the mean (averaged)
//             object
//    Context:
//    Parameters:  obj         - input object
//                 objStep     - its step (in bytes)
//                 nEigObjs    - number of eigen objects
//                 eigInput    - pointer either to array of pointers to eigen objects
//                               or to read callback function (depending on ioFlags)
//                 eigStep     - eigen objects step (in bytes)
//                 ioFlags     - input/output flags
//                 iserData    - pointer to the structure which contains all necessary
//                               data for the callback function
//                 avg         - pointer to averaged object
//                 avgStep     - its step (in bytes)
//                 size        - ROI size of each source object
//                 coeffs      - calculated coefficients (output data)
//
//    Returns: icv status
//
//    Notes:   see notes for icvCalcEigenObjects_8u32fR function
//F*/
static CvStatus CV_STDCALL
icvEigenDecomposite_8u32fR( uchar * obj, int objStep, int nEigObjs,
                            void *eigInput, int eigStep, int ioFlags,
                            void *userData, float *avg, int avgStep,
                            CvSize size, float *coeffs )
{
    int i;

    if( nEigObjs < 2 )
        return CV_BADFACTOR_ERR;
    if( ioFlags < 0 || ioFlags > 1 )
        return CV_BADFACTOR_ERR;
    if( size.width > objStep || 4 * size.width > eigStep ||
        4 * size.width > avgStep || size.height < 1 )
        return CV_BADSIZE_ERR;
    if( obj == NULL || eigInput == NULL || coeffs == NULL || avg == NULL )
        return CV_NULLPTR_ERR;
    if( !ioFlags )
        for( i = 0; i < nEigObjs; i++ )
            if( ((uchar **) eigInput)[i] == NULL )
                return CV_NULLPTR_ERR;

    if( ioFlags )               /* callback */

    {
        float *buffer;
        CvCallback read_callback = ((CvInput *) & eigInput)->callback;

        eigStep = 4 * size.width;

        /* memory allocation */
        buffer = (float *) cvAlloc( sizeof( float ) * size.width * size.height );

        if( buffer == NULL )
            return CV_OUTOFMEM_ERR;

        for( i = 0; i < nEigObjs; i++ )
        {
            float w;
            CvStatus r = (CvStatus)read_callback( i, (void *) buffer, userData );

            if( r )
            {
                cvFree( &buffer );
                return r;
            }
            w = icvCalcDecompCoeff_8u32fR( obj, objStep, buffer,
                                            eigStep, avg, avgStep, size );
            if( w < -1.0e29f )
            {
                cvFree( &buffer );
                return CV_NOTDEFINED_ERR;
            }
            coeffs[i] = w;
        }
        cvFree( &buffer );
    }

    else
        /* no callback */
        for( i = 0; i < nEigObjs; i++ )
        {
            float w = icvCalcDecompCoeff_8u32fR( obj, objStep, ((float **) eigInput)[i],
                                                  eigStep, avg, avgStep, size );

            if( w < -1.0e29f )
                return CV_NOTDEFINED_ERR;
            coeffs[i] = w;
        }

    return CV_NO_ERR;
}


/*F///////////////////////////////////////////////////////////////////////////////////////
//    Names: icvEigenProjection_8u32fR
//    Purpose: The function calculates object projection to the eigen sub-space (restores
//             an object) using previously calculated eigen objects basis, mean (averaged)
//             object and decomposition coefficients of the restored object
//    Context:
//    Parameters:  nEigObjs - Number of eigen objects
//                 eigens   - Array of pointers to eigen objects
//                 eigStep  - Eigen objects step (in bytes)
//                 coeffs   - Previously calculated decomposition coefficients
//                 avg      - Pointer to averaged object
//                 avgStep  - Its step (in bytes)
//                 rest     - Pointer to restored object
//                 restStep - Its step (in bytes)
//                 size     - ROI size of each object
//
//    Returns: CV status
//
//    Notes:
//F*/
static CvStatus CV_STDCALL
icvEigenProjection_8u32fR( int nEigObjs, void *eigInput, int eigStep,
                           int ioFlags, void *userData, float *coeffs,
                           float *avg, int avgStep, uchar * rest,
                           int restStep, CvSize size )
{
    int i, j, k;
    float *buf;
    float *buffer = NULL;
    float *b;
    CvCallback read_callback = ((CvInput *) & eigInput)->callback;

    if( size.width > avgStep || 4 * size.width > eigStep || size.height < 1 )
        return CV_BADSIZE_ERR;
    if( rest == NULL || eigInput == NULL || avg == NULL || coeffs == NULL )
        return CV_NULLPTR_ERR;
    if( ioFlags < 0 || ioFlags > 1 )
        return CV_BADFACTOR_ERR;
    if( !ioFlags )
        for( i = 0; i < nEigObjs; i++ )
            if( ((uchar **) eigInput)[i] == NULL )
                return CV_NULLPTR_ERR;
    eigStep /= 4;
    avgStep /= 4;

    if( size.width == restStep && size.width == eigStep && size.width == avgStep )
    {
        size.width *= size.height;
        size.height = 1;
        restStep = eigStep = avgStep = size.width;
    }

    buf = (float *) cvAlloc( sizeof( float ) * size.width * size.height );

    if( buf == NULL )
        return CV_OUTOFMEM_ERR;
    b = buf;
    for( i = 0; i < size.height; i++, avg += avgStep, b += size.width )
        for( j = 0; j < size.width; j++ )
            b[j] = avg[j];

    if( ioFlags )
    {
        buffer = (float *) cvAlloc( sizeof( float ) * size.width * size.height );

        if( buffer == NULL )
        {
            cvFree( &buf );
            return CV_OUTOFMEM_ERR;
        }
        eigStep = size.width;
    }

    for( k = 0; k < nEigObjs; k++ )
    {
        float *e = ioFlags ? buffer : ((float **) eigInput)[k];
        float c = coeffs[k];

        if( ioFlags )           /* read eigen object */
        {
            CvStatus r = (CvStatus)read_callback( k, (void *) buffer, userData );

            if( r )
            {
                cvFree( &buf );
                cvFree( &buffer );
                return r;
            }
        }

        b = buf;
        for( i = 0; i < size.height; i++, e += eigStep, b += size.width )
        {
            for( j = 0; j < size.width - 3; j += 4 )
            {
                float b0 = c * e[j];
                float b1 = c * e[j + 1];
                float b2 = c * e[j + 2];
                float b3 = c * e[j + 3];

                b[j] += b0;
                b[j + 1] += b1;
                b[j + 2] += b2;
                b[j + 3] += b3;
            }
            for( ; j < size.width; j++ )
                b[j] += c * e[j];
        }
    }

    b = buf;
    for( i = 0; i < size.height; i++, avg += avgStep, b += size.width, rest += restStep )
        for( j = 0; j < size.width; j++ )
        {
            int w = cvRound( b[j] );

            w = !(w & ~255) ? w : w < 0 ? 0 : 255;
            rest[j] = (uchar) w;
        }

    cvFree( &buf );
    if( ioFlags )
        cvFree( &buffer );
    return CV_NO_ERR;
}

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: cvCalcCovarMatrixEx
//    Purpose: The function calculates a covariance matrix for a group of input objects
//             (images, vectors, etc.).
//    Context:
//    Parameters:  nObjects    - number of source objects
//                 input       - pointer either to array of input objects
//                               or to read callback function (depending on ioFlags)
//                 ioFlags     - input/output flags (see Notes to
//                               cvCalcEigenObjects function)
//                 ioBufSize   - input/output buffer size
//                 userData    - pointer to the structure which contains all necessary
//                               data for the callback functions
//                 avg         - averaged object
//                 covarMatrix - covariance matrix (output parameter; must be allocated
//                               before call)
//
//    Notes:  See Notes to cvCalcEigenObjects function
//F*/

CV_IMPL void
cvCalcCovarMatrixEx( int  nObjects, void*  input, int  ioFlags,
                     int  ioBufSize, uchar*  buffer, void*  userData,
                     IplImage* avg, float*  covarMatrix )
{
    float *avg_data;
    int avg_step = 0;
    CvSize avg_size;
    int i;

    CV_FUNCNAME( "cvCalcCovarMatrixEx" );

    __BEGIN__;

    cvGetImageRawData( avg, (uchar **) & avg_data, &avg_step, &avg_size );
    if( avg->depth != IPL_DEPTH_32F )
        CV_ERROR( CV_BadDepth, cvUnsupportedFormat );
    if( avg->nChannels != 1 )
        CV_ERROR( CV_BadNumChannels, cvUnsupportedFormat );

    if( ioFlags == CV_EIGOBJ_NO_CALLBACK )
    {
        IplImage **images = (IplImage **) (((CvInput *) & input)->data);
        uchar **objects = (uchar **) cvAlloc( sizeof( uchar * ) * nObjects );
        int img_step = 0, old_step = 0;
        CvSize img_size = avg_size, old_size = avg_size;

        if( objects == NULL )
            CV_ERROR( CV_StsBadArg, "Insufficient memory" );

        for( i = 0; i < nObjects; i++ )
        {
            IplImage *img = images[i];
            uchar *img_data;

            cvGetImageRawData( img, &img_data, &img_step, &img_size );
            if( img->depth != IPL_DEPTH_8U )
                CV_ERROR( CV_BadDepth, cvUnsupportedFormat );
            if( img_size != avg_size || img_size != old_size )
                CV_ERROR( CV_StsBadArg, "Different sizes of objects" );
            if( img->nChannels != 1 )
                CV_ERROR( CV_BadNumChannels, cvUnsupportedFormat );
            if( i > 0 && img_step != old_step )
                CV_ERROR( CV_StsBadArg, "Different steps of objects" );

            old_step = img_step;
            old_size = img_size;
            objects[i] = img_data;
        }

        CV_CALL( icvCalcCovarMatrixEx_8u32fR( nObjects,
                                              (void*) objects,
                                              img_step,
                                              CV_EIGOBJ_NO_CALLBACK,
                                              0,
                                              NULL,
                                              NULL,
                                              avg_data,
                                              avg_step,
                                              avg_size,
                                              covarMatrix ));
        cvFree( &objects );
    }

    else

    {
        CV_CALL( icvCalcCovarMatrixEx_8u32fR( nObjects,
                                              input,
                                              avg_step / 4,
                                              ioFlags,
                                              ioBufSize,
                                              buffer,
                                              userData,
                                              avg_data,
                                              avg_step,
                                              avg_size,
                                              covarMatrix ));
    }

    __END__;
}

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: cvCalcEigenObjects
//    Purpose: The function calculates an orthonormal eigen basis and a mean (averaged)
//             object for a group of input objects (images, vectors, etc.).
//    Context:
//    Parameters: nObjects  - number of source objects
//                input     - pointer either to array of input objects
//                            or to read callback function (depending on ioFlags)
//                output    - pointer either to output eigen objects
//                            or to write callback function (depending on ioFlags)
//                ioFlags   - input/output flags (see Notes)
//                ioBufSize - input/output buffer size
//                userData  - pointer to the structure which contains all necessary
//                            data for the callback functions
//                calcLimit - determines the calculation finish conditions
//                avg       - averaged object (has the same size as ROI)
//                eigVals   - pointer to corresponding eigen values (array of <nObjects>
//                            elements in descending order)
//
//    Notes: 1. input/output data (that is, input objects and eigen ones) may either
//              be allocated in the RAM or be read from/written to the HDD (or any
//              other device) by read/write callback functions. It depends on the
//              value of ioFlags paramater, which may be the following:
//                  CV_EIGOBJ_NO_CALLBACK, or 0;
//                  CV_EIGOBJ_INPUT_CALLBACK;
//                  CV_EIGOBJ_OUTPUT_CALLBACK;
//                  CV_EIGOBJ_BOTH_CALLBACK, or
//                            CV_EIGOBJ_INPUT_CALLBACK | CV_EIGOBJ_OUTPUT_CALLBACK.
//              The callback functions as well as the user data structure must be
//              developed by the user.
//
//           2. If ioBufSize = 0, or it's too large, the function dermines buffer size
//              itself.
//
//           3. Depending on calcLimit parameter, calculations are finished either if
//              eigenfaces number comes up to certain value or the relation of the
//              current eigenvalue and the largest one comes down to certain value
//              (or any of the above conditions takes place). The calcLimit->type value
//              must be CV_TERMCRIT_NUMB, CV_TERMCRIT_EPS or
//              CV_TERMCRIT_NUMB | CV_TERMCRIT_EPS. The function returns the real
//              values calcLimit->max_iter and calcLimit->epsilon.
//
//           4. eigVals may be equal to NULL (if you don't need eigen values in further).
//
//F*/
CV_IMPL void
cvCalcEigenObjects( int       nObjects,
                    void*     input,
                    void*     output,
                    int       ioFlags,
                    int       ioBufSize,
                    void*     userData,
                    CvTermCriteria* calcLimit,
                    IplImage* avg,
                    float*    eigVals )
{
    float *avg_data;
    int avg_step = 0;
    CvSize avg_size;
    int i;
    int nEigens = nObjects - 1;

    CV_FUNCNAME( "cvCalcEigenObjects" );

    __BEGIN__;

    cvGetImageRawData( avg, (uchar **) & avg_data, &avg_step, &avg_size );
    if( avg->depth != IPL_DEPTH_32F )
        CV_ERROR( CV_BadDepth, cvUnsupportedFormat );
    if( avg->nChannels != 1 )
        CV_ERROR( CV_BadNumChannels, cvUnsupportedFormat );

    if( nEigens > calcLimit->max_iter && calcLimit->type != CV_TERMCRIT_EPS )
        nEigens = calcLimit->max_iter;

    switch (ioFlags)
    {
    case CV_EIGOBJ_NO_CALLBACK:
        {
            IplImage **objects = (IplImage **) (((CvInput *) & input)->data);
            IplImage **eigens = (IplImage **) (((CvInput *) & output)->data);
            uchar **objs = (uchar **) cvAlloc( sizeof( uchar * ) * nObjects );
            float **eigs = (float **) cvAlloc( sizeof( float * ) * nEigens );
            int obj_step = 0, old_step = 0;
            int eig_step = 0, oldeig_step = 0;
            CvSize obj_size = avg_size, old_size = avg_size,

                eig_size = avg_size, oldeig_size = avg_size;

            if( objects == NULL || eigens == NULL )
                CV_ERROR( CV_StsBadArg, "Insufficient memory" );

            for( i = 0; i < nObjects; i++ )
            {
                IplImage *img = objects[i];
                uchar *obj_data;

                cvGetImageRawData( img, &obj_data, &obj_step, &obj_size );
                if( img->depth != IPL_DEPTH_8U )
                    CV_ERROR( CV_BadDepth, cvUnsupportedFormat );
                if( obj_size != avg_size || obj_size != old_size )
                    CV_ERROR( CV_StsBadArg, "Different sizes of objects" );
                if( img->nChannels != 1 )
                    CV_ERROR( CV_BadNumChannels, cvUnsupportedFormat );
                if( i > 0 && obj_step != old_step )
                    CV_ERROR( CV_StsBadArg, "Different steps of objects" );

                old_step = obj_step;
                old_size = obj_size;
                objs[i] = obj_data;
            }
            for( i = 0; i < nEigens; i++ )
            {
                IplImage *eig = eigens[i];
                float *eig_data;

                cvGetImageRawData( eig, (uchar **) & eig_data, &eig_step, &eig_size );
                if( eig->depth != IPL_DEPTH_32F )
                    CV_ERROR( CV_BadDepth, cvUnsupportedFormat );
                if( eig_size != avg_size || eig_size != oldeig_size )
                    CV_ERROR( CV_StsBadArg, "Different sizes of objects" );
                if( eig->nChannels != 1 )
                    CV_ERROR( CV_BadNumChannels, cvUnsupportedFormat );
                if( i > 0 && eig_step != oldeig_step )
                    CV_ERROR( CV_StsBadArg, "Different steps of objects" );

                oldeig_step = eig_step;
                oldeig_size = eig_size;
                eigs[i] = eig_data;
            }
            CV_CALL( icvCalcEigenObjects_8u32fR( nObjects, (void*) objs, obj_step,
                                                 (void*) eigs, eig_step, obj_size,
                                                 ioFlags, ioBufSize, userData,
                                                 calcLimit, avg_data, avg_step, eigVals ));
            cvFree( &objs );
            cvFree( &eigs );
            break;
        }

    case CV_EIGOBJ_OUTPUT_CALLBACK:
        {
            IplImage **objects = (IplImage **) (((CvInput *) & input)->data);
            uchar **objs = (uchar **) cvAlloc( sizeof( uchar * ) * nObjects );
            int obj_step = 0, old_step = 0;
            CvSize obj_size = avg_size, old_size = avg_size;

            if( objects == NULL )
                CV_ERROR( CV_StsBadArg, "Insufficient memory" );

            for( i = 0; i < nObjects; i++ )
            {
                IplImage *img = objects[i];
                uchar *obj_data;

                cvGetImageRawData( img, &obj_data, &obj_step, &obj_size );
                if( img->depth != IPL_DEPTH_8U )
                    CV_ERROR( CV_BadDepth, cvUnsupportedFormat );
                if( obj_size != avg_size || obj_size != old_size )
                    CV_ERROR( CV_StsBadArg, "Different sizes of objects" );
                if( img->nChannels != 1 )
                    CV_ERROR( CV_BadNumChannels, cvUnsupportedFormat );
                if( i > 0 && obj_step != old_step )
                    CV_ERROR( CV_StsBadArg, "Different steps of objects" );

                old_step = obj_step;
                old_size = obj_size;
                objs[i] = obj_data;
            }
            CV_CALL( icvCalcEigenObjects_8u32fR( nObjects,
                                                 (void*) objs,
                                                 obj_step,
                                                 output,
                                                 avg_step,
                                                 obj_size,
                                                 ioFlags,
                                                 ioBufSize,
                                                 userData,
                                                 calcLimit,
                                                 avg_data,
                                                 avg_step,
                                                 eigVals   ));
            cvFree( &objs );
            break;
        }

    case CV_EIGOBJ_INPUT_CALLBACK:
        {
            IplImage **eigens = (IplImage **) (((CvInput *) & output)->data);
            float **eigs = (float**) cvAlloc( sizeof( float* ) * nEigens );
            int eig_step = 0, oldeig_step = 0;
            CvSize eig_size = avg_size, oldeig_size = avg_size;

            if( eigens == NULL )
                CV_ERROR( CV_StsBadArg, "Insufficient memory" );

            for( i = 0; i < nEigens; i++ )
            {
                IplImage *eig = eigens[i];
                float *eig_data;

                cvGetImageRawData( eig, (uchar **) & eig_data, &eig_step, &eig_size );
                if( eig->depth != IPL_DEPTH_32F )
                    CV_ERROR( CV_BadDepth, cvUnsupportedFormat );
                if( eig_size != avg_size || eig_size != oldeig_size )
                    CV_ERROR( CV_StsBadArg, "Different sizes of objects" );
                if( eig->nChannels != 1 )
                    CV_ERROR( CV_BadNumChannels, cvUnsupportedFormat );
                if( i > 0 && eig_step != oldeig_step )
                    CV_ERROR( CV_StsBadArg, "Different steps of objects" );

                oldeig_step = eig_step;
                oldeig_size = eig_size;
                eigs[i] = eig_data;
            }
            CV_CALL( icvCalcEigenObjects_8u32fR( nObjects,
                                                 input,
                                                 avg_step / 4,
                                                 (void*) eigs,
                                                 eig_step,
                                                 eig_size,
                                                 ioFlags,
                                                 ioBufSize,
                                                 userData,
                                                 calcLimit,
                                                 avg_data,
                                                 avg_step,
                                                 eigVals   ));
            cvFree( &eigs );
            break;
        }
    case CV_EIGOBJ_INPUT_CALLBACK | CV_EIGOBJ_OUTPUT_CALLBACK:

        CV_CALL( icvCalcEigenObjects_8u32fR( nObjects,
                                             input,
                                             avg_step / 4,
                                             output,
                                             avg_step,
                                             avg_size,
                                             ioFlags,
                                             ioBufSize,
                                             userData,
                                             calcLimit,
                                             avg_data,
                                             avg_step,
                                             eigVals   ));
        break;

    default:
        CV_ERROR( CV_StsBadArg, "Unsupported i/o flag" );
    }

    __END__;
}

/*--------------------------------------------------------------------------------------*/
/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: cvCalcDecompCoeff
//    Purpose: The function calculates one decomposition coefficient of input object
//             using previously calculated eigen object and the mean (averaged) object
//    Context:
//    Parameters:  obj     - input object
//                 eigObj  - eigen object
//                 avg     - averaged object
//
//    Returns: decomposition coefficient value or large negative value (if error)
//
//    Notes:
//F*/

CV_IMPL double
cvCalcDecompCoeff( IplImage * obj, IplImage * eigObj, IplImage * avg )
{
    double coeff = DBL_MAX;

    uchar *obj_data;
    float *eig_data;
    float *avg_data;
    int obj_step = 0, eig_step = 0, avg_step = 0;
    CvSize obj_size, eig_size, avg_size;

    CV_FUNCNAME( "cvCalcDecompCoeff" );

    __BEGIN__;

    cvGetImageRawData( obj, &obj_data, &obj_step, &obj_size );
    if( obj->depth != IPL_DEPTH_8U )
        CV_ERROR( CV_BadDepth, cvUnsupportedFormat );
    if( obj->nChannels != 1 )
        CV_ERROR( CV_BadNumChannels, cvUnsupportedFormat );

    cvGetImageRawData( eigObj, (uchar **) & eig_data, &eig_step, &eig_size );
    if( eigObj->depth != IPL_DEPTH_32F )
        CV_ERROR( CV_BadDepth, cvUnsupportedFormat );
    if( eigObj->nChannels != 1 )
        CV_ERROR( CV_BadNumChannels, cvUnsupportedFormat );

    cvGetImageRawData( avg, (uchar **) & avg_data, &avg_step, &avg_size );
    if( avg->depth != IPL_DEPTH_32F )
        CV_ERROR( CV_BadDepth, cvUnsupportedFormat );
    if( avg->nChannels != 1 )
        CV_ERROR( CV_BadNumChannels, cvUnsupportedFormat );

    if( obj_size != eig_size || obj_size != avg_size )
        CV_ERROR( CV_StsBadArg, "different sizes of images" );

    coeff = icvCalcDecompCoeff_8u32fR( obj_data, obj_step,
                                       eig_data, eig_step,
                                       avg_data, avg_step, obj_size );

    __END__;

    return coeff;
}

/*--------------------------------------------------------------------------------------*/
/*F///////////////////////////////////////////////////////////////////////////////////////
//    Names: cvEigenDecomposite
//    Purpose: The function calculates all decomposition coefficients for input object
//             using previously calculated eigen objects basis and the mean (averaged)
//             object
//
//    Parameters:  obj         - input object
//                 nEigObjs    - number of eigen objects
//                 eigInput    - pointer either to array of pointers to eigen objects
//                               or to read callback function (depending on ioFlags)
//                 ioFlags     - input/output flags
//                 userData    - pointer to the structure which contains all necessary
//                               data for the callback function
//                 avg         - averaged object
//                 coeffs      - calculated coefficients (output data)
//
//    Notes:   see notes for cvCalcEigenObjects function
//F*/

CV_IMPL void
cvEigenDecomposite( IplImage* obj,
                    int       nEigObjs,
                    void*     eigInput,
                    int       ioFlags,
                    void*     userData,
                    IplImage* avg,
                    float*    coeffs )
{
    float *avg_data;
    uchar *obj_data;
    int avg_step = 0, obj_step = 0;
    CvSize avg_size, obj_size;
    int i;

    CV_FUNCNAME( "cvEigenDecomposite" );

    __BEGIN__;

    cvGetImageRawData( avg, (uchar **) & avg_data, &avg_step, &avg_size );
    if( avg->depth != IPL_DEPTH_32F )
        CV_ERROR( CV_BadDepth, cvUnsupportedFormat );
    if( avg->nChannels != 1 )
        CV_ERROR( CV_BadNumChannels, cvUnsupportedFormat );

    cvGetImageRawData( obj, &obj_data, &obj_step, &obj_size );
    if( obj->depth != IPL_DEPTH_8U )
        CV_ERROR( CV_BadDepth, cvUnsupportedFormat );
    if( obj->nChannels != 1 )
        CV_ERROR( CV_BadNumChannels, cvUnsupportedFormat );

    if( obj_size != avg_size )
        CV_ERROR( CV_StsBadArg, "Different sizes of objects" );

    if( ioFlags == CV_EIGOBJ_NO_CALLBACK )
    {
        IplImage **eigens = (IplImage **) (((CvInput *) & eigInput)->data);
        float **eigs = (float **) cvAlloc( sizeof( float * ) * nEigObjs );
        int eig_step = 0, old_step = 0;
        CvSize eig_size = avg_size, old_size = avg_size;

        if( eigs == NULL )
            CV_ERROR( CV_StsBadArg, "Insufficient memory" );

        for( i = 0; i < nEigObjs; i++ )
        {
            IplImage *eig = eigens[i];
            float *eig_data;

            cvGetImageRawData( eig, (uchar **) & eig_data, &eig_step, &eig_size );
            if( eig->depth != IPL_DEPTH_32F )
                CV_ERROR( CV_BadDepth, cvUnsupportedFormat );
            if( eig_size != avg_size || eig_size != old_size )
                CV_ERROR( CV_StsBadArg, "Different sizes of objects" );
            if( eig->nChannels != 1 )
                CV_ERROR( CV_BadNumChannels, cvUnsupportedFormat );
            if( i > 0 && eig_step != old_step )
                CV_ERROR( CV_StsBadArg, "Different steps of objects" );

            old_step = eig_step;
            old_size = eig_size;
            eigs[i] = eig_data;
        }

        CV_CALL( icvEigenDecomposite_8u32fR( obj_data,
                                             obj_step,
                                             nEigObjs,
                                             (void*) eigs,
                                             eig_step,
                                             ioFlags,
                                             userData,
                                             avg_data,
                                             avg_step,
                                             obj_size,
                                             coeffs   ));
        cvFree( &eigs );
    }

    else

    {
        CV_CALL( icvEigenDecomposite_8u32fR( obj_data,
                                             obj_step,
                                             nEigObjs,
                                             eigInput,
                                             avg_step,
                                             ioFlags,
                                             userData,
                                             avg_data,
                                             avg_step,
                                             obj_size,
                                             coeffs   ));
    }

    __END__;
}

/*--------------------------------------------------------------------------------------*/
/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: cvEigenProjection
//    Purpose: The function calculates object projection to the eigen sub-space (restores
//             an object) using previously calculated eigen objects basis, mean (averaged)
//             object and decomposition coefficients of the restored object
//    Context:
//    Parameters:  nEigObjs    - number of eigen objects
//                 eigInput    - pointer either to array of pointers to eigen objects
//                               or to read callback function (depending on ioFlags)
//                 ioFlags     - input/output flags
//                 userData    - pointer to the structure which contains all necessary
//                               data for the callback function
//                 coeffs      - array of decomposition coefficients
//                 avg         - averaged object
//                 proj        - object projection (output data)
//
//    Notes:   see notes for cvCalcEigenObjects function
//F*/

CV_IMPL void
cvEigenProjection( void*     eigInput,
                   int       nEigObjs,
                   int       ioFlags,
                   void*     userData,
                   float*    coeffs,
                   IplImage* avg,
                   IplImage* proj )
{
    float *avg_data;
    uchar *proj_data;
    int avg_step = 0, proj_step = 0;
    CvSize avg_size, proj_size;
    int i;

    CV_FUNCNAME( "cvEigenProjection" );

    __BEGIN__;

    cvGetImageRawData( avg, (uchar **) & avg_data, &avg_step, &avg_size );
    if( avg->depth != IPL_DEPTH_32F )
        CV_ERROR( CV_BadDepth, cvUnsupportedFormat );
    if( avg->nChannels != 1 )
        CV_ERROR( CV_BadNumChannels, cvUnsupportedFormat );

    cvGetImageRawData( proj, &proj_data, &proj_step, &proj_size );
    if( proj->depth != IPL_DEPTH_8U )
        CV_ERROR( CV_BadDepth, cvUnsupportedFormat );
    if( proj->nChannels != 1 )
        CV_ERROR( CV_BadNumChannels, cvUnsupportedFormat );

    if( proj_size != avg_size )
        CV_ERROR( CV_StsBadArg, "Different sizes of projects" );

    if( ioFlags == CV_EIGOBJ_NO_CALLBACK )
    {
        IplImage **eigens = (IplImage**) (((CvInput *) & eigInput)->data);
        float **eigs = (float**) cvAlloc( sizeof( float * ) * nEigObjs );
        int eig_step = 0, old_step = 0;
        CvSize eig_size = avg_size, old_size = avg_size;

        if( eigs == NULL )
            CV_ERROR( CV_StsBadArg, "Insufficient memory" );

        for( i = 0; i < nEigObjs; i++ )
        {
            IplImage *eig = eigens[i];
            float *eig_data;

            cvGetImageRawData( eig, (uchar **) & eig_data, &eig_step, &eig_size );
            if( eig->depth != IPL_DEPTH_32F )
                CV_ERROR( CV_BadDepth, cvUnsupportedFormat );
            if( eig_size != avg_size || eig_size != old_size )
                CV_ERROR( CV_StsBadArg, "Different sizes of objects" );
            if( eig->nChannels != 1 )
                CV_ERROR( CV_BadNumChannels, cvUnsupportedFormat );
            if( i > 0 && eig_step != old_step )
                CV_ERROR( CV_StsBadArg, "Different steps of objects" );

            old_step = eig_step;
            old_size = eig_size;
            eigs[i] = eig_data;
        }

        CV_CALL( icvEigenProjection_8u32fR( nEigObjs,
                                            (void*) eigs,
                                            eig_step,
                                            ioFlags,
                                            userData,
                                            coeffs,
                                            avg_data,
                                            avg_step,
                                            proj_data,
                                            proj_step,
                                            avg_size   ));
        cvFree( &eigs );
    }

    else

    {
        CV_CALL( icvEigenProjection_8u32fR( nEigObjs,
                                            eigInput,
                                            avg_step,
                                            ioFlags,
                                            userData,
                                            coeffs,
                                            avg_data,
                                            avg_step,
                                            proj_data,
                                            proj_step,
                                            avg_size   ));
    }

    __END__;
}

/* End of file. */
