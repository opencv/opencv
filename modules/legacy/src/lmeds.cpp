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
#include "_vm.h"
#include <stdlib.h>

#define Sgn(x)              ( (x)<0 ? -1:1 )    /* Sgn(0) = 1 ! */
/*===========================================================================*/
CvStatus
icvLMedS( int *points1, int *points2, int numPoints, CvMatrix3 * fundamentalMatrix )
{
    int sample, j, amount_samples, done;
    int amount_solutions;
    int ml7[21], mr7[21];

    double F_try[9 * 3];
    double F[9];
    double Mj, Mj_new;

    int i, num;

    int *ml;
    int *mr;
    int *new_ml;
    int *new_mr;
    int new_num;
    CvStatus error;

    error = CV_NO_ERR;

    if( fundamentalMatrix == 0 )
        return CV_BADFACTOR_ERR;

    num = numPoints;

    if( num < 6 )
    {
        return CV_BADFACTOR_ERR;
    }                           /* if */

    ml = (int *) cvAlloc( sizeof( int ) * num * 3 );
    mr = (int *) cvAlloc( sizeof( int ) * num * 3 );

    for( i = 0; i < num; i++ )
    {

        ml[i * 3] = points1[i * 2];
        ml[i * 3 + 1] = points1[i * 2 + 1];

        ml[i * 3 + 2] = 1;

        mr[i * 3] = points2[i * 2];
        mr[i * 3 + 1] = points2[i * 2 + 1];

        mr[i * 3 + 2] = 1;
    }                           /* for */

    if( num > 7 )
    {

        Mj = -1;
        amount_samples = 1000;  /*  -------  Must be changed !  --------- */

        for( sample = 0; sample < amount_samples; sample++ )
        {

            icvChoose7( ml, mr, num, ml7, mr7 );
            icvPoint7( ml7, mr7, F_try, &amount_solutions );

            for( i = 0; i < amount_solutions / 9; i++ )
            {

                Mj_new = icvMedian( ml, mr, num, F_try + i * 9 );

                if( Mj_new >= 0 && (Mj == -1 || Mj_new < Mj) )
                {

                    for( j = 0; j < 9; j++ )
                    {

                        F[j] = F_try[i * 9 + j];
                    }           /* for */

                    Mj = Mj_new;
                }               /* if */
            }                   /* for */
        }                       /* for */

        if( Mj == -1 )
            return CV_BADFACTOR_ERR;

        done = icvBoltingPoints( ml, mr, num, F, Mj, &new_ml, &new_mr, &new_num );

        if( done == -1 )
        {

            cvFree( &mr );
            cvFree( &ml );
            return CV_OUTOFMEM_ERR;
        }                       /* if */

        if( done > 7 )
            error = icvPoints8( new_ml, new_mr, new_num, F );

        cvFree( &new_mr );
        cvFree( &new_ml );

    }
    else
    {
        error = icvPoint7( ml, mr, F, &i );
    }                           /* if */

    if( error == CV_NO_ERR )
        error = icvRank2Constraint( F );

    for( i = 0; i < 3; i++ )
        for( j = 0; j < 3; j++ )
            fundamentalMatrix->m[i][j] = (float) F[i * 3 + j];

    return error;

}                               /* icvLMedS */

/*===========================================================================*/
/*===========================================================================*/
void
icvChoose7( int *ml, int *mr, int num, int *ml7, int *mr7 )
{
    int indexes[7], i, j;

    if( !ml || !mr || num < 7 || !ml7 || !mr7 )
        return;

    for( i = 0; i < 7; i++ )
    {

        indexes[i] = (int) ((double) rand() / RAND_MAX * num);

        for( j = 0; j < i; j++ )
        {

            if( indexes[i] == indexes[j] )
                i--;
        }                       /* for */
    }                           /* for */

    for( i = 0; i < 21; i++ )
    {

        ml7[i] = ml[3 * indexes[i / 3] + i % 3];
        mr7[i] = mr[3 * indexes[i / 3] + i % 3];
    }                           /* for */
}                               /* cs_Choose7 */

/*===========================================================================*/
/*===========================================================================*/
CvStatus
icvCubic( double a2, double a1, double a0, double *squares )
{
    double p, q, D, c1, c2, b1, b2, ro1, ro2, fi1, fi2, tt;
    double x[6][3];
    int i, j, t;

    if( !squares )
        return CV_BADFACTOR_ERR;

    p = a1 - a2 * a2 / 3;
    q = (9 * a1 * a2 - 27 * a0 - 2 * a2 * a2 * a2) / 27;
    D = q * q / 4 + p * p * p / 27;

    if( D < 0 )
    {

        c1 = q / 2;
        c2 = c1;
        b1 = sqrt( -D );
        b2 = -b1;

        ro1 = sqrt( c1 * c1 - D );
        ro2 = ro1;

        fi1 = atan2( b1, c1 );
        fi2 = -fi1;
    }
    else
    {

        c1 = q / 2 + sqrt( D );
        c2 = q / 2 - sqrt( D );
        b1 = 0;
        b2 = 0;

        ro1 = fabs( c1 );
        ro2 = fabs( c2 );
        fi1 = CV_PI * (1 - SIGN( c1 )) / 2;
        fi2 = CV_PI * (1 - SIGN( c2 )) / 2;
    }                           /* if */

    for( i = 0; i < 6; i++ )
    {

        x[i][0] = -a2 / 3;
        x[i][1] = 0;
        x[i][2] = 0;

        squares[i] = x[i][i % 2];
    }                           /* for */

    if( !REAL_ZERO( ro1 ))
    {
        tt = SIGN( ro1 ) * pow( fabs( ro1 ), 0.333333333333 );
        c1 = tt - p / (3. * tt);
        c2 = tt + p / (3. * tt);
    }                           /* if */

    if( !REAL_ZERO( ro2 ))
    {
        tt = SIGN( ro2 ) * pow( fabs( ro2 ), 0.333333333333 );
        b1 = tt - p / (3. * tt);
        b2 = tt + p / (3. * tt);
    }                           /* if */

    for( i = 0; i < 6; i++ )
    {

        if( i < 3 )
        {

            if( !REAL_ZERO( ro1 ))
            {

                x[i][0] = cos( fi1 / 3. + 2 * CV_PI * (i % 3) / 3. ) * c1 - a2 / 3;
                x[i][1] = sin( fi1 / 3. + 2 * CV_PI * (i % 3) / 3. ) * c2;
            }
            else
            {

                x[i][2] = 1;
            }                   /* if */
        }
        else
        {

            if( !REAL_ZERO( ro2 ))
            {

                x[i][0] = cos( fi2 / 3. + 2 * CV_PI * (i % 3) / 3. ) * b1 - a2 / 3;
                x[i][1] = sin( fi2 / 3. + 2 * CV_PI * (i % 3) / 3. ) * b2;
            }
            else
            {

                x[i][2] = 1;
            }                   /* if */
        }                       /* if */
    }                           /* for */

    t = 0;

    for( i = 0; i < 6; i++ )
    {

        if( !x[i][2] )
        {

            squares[t++] = x[i][0];
            squares[t++] = x[i][1];
            x[i][2] = 1;

            for( j = i + 1; j < 6; j++ )
            {

                if( !x[j][2] && REAL_ZERO( x[i][0] - x[j][0] )
                    && REAL_ZERO( x[i][1] - x[j][1] ))
                {

                    x[j][2] = 1;
                    break;
                }               /* if */
            }                   /* for */
        }                       /* if */
    }                           /* for */
    return CV_NO_ERR;
}                               /* icvCubic */

/*======================================================================================*/
double
icvDet( double *M )
{
    double value;

    if( !M )
        return 0;

    value = M[0] * M[4] * M[8] + M[2] * M[3] * M[7] + M[1] * M[5] * M[6] -
        M[2] * M[4] * M[6] - M[0] * M[5] * M[7] - M[1] * M[3] * M[8];

    return value;

}                               /* icvDet */

/*===============================================================================*/
double
icvMinor( double *M, int x, int y )
{
    int row1, row2, col1, col2;
    double value;

    if( !M || x < 0 || x > 2 || y < 0 || y > 2 )
        return 0;

    row1 = (y == 0 ? 1 : 0);
    row2 = (y == 2 ? 1 : 2);
    col1 = (x == 0 ? 1 : 0);
    col2 = (x == 2 ? 1 : 2);

    value = M[row1 * 3 + col1] * M[row2 * 3 + col2] - M[row2 * 3 + col1] * M[row1 * 3 + col2];

    value *= 1 - (x + y) % 2 * 2;

    return value;

}                               /* icvMinor */

/*======================================================================================*/
CvStatus
icvGetCoef( double *f1, double *f2, double *a2, double *a1, double *a0 )
{
    double G[9], a3;
    int i;

    if( !f1 || !f2 || !a0 || !a1 || !a2 )
        return CV_BADFACTOR_ERR;

    for( i = 0; i < 9; i++ )
    {

        G[i] = f1[i] - f2[i];
    }                           /* for */

    a3 = icvDet( G );

    if( REAL_ZERO( a3 ))
        return CV_BADFACTOR_ERR;

    *a2 = 0;
    *a1 = 0;
    *a0 = icvDet( f2 );

    for( i = 0; i < 9; i++ )
    {

        *a2 += f2[i] * icvMinor( G, (int) (i % 3), (int) (i / 3) );
        *a1 += G[i] * icvMinor( f2, (int) (i % 3), (int) (i / 3) );
    }                           /* for */

    *a0 /= a3;
    *a1 /= a3;
    *a2 /= a3;

    return CV_NO_ERR;

}                               /* icvGetCoef */

/*===========================================================================*/
double
icvMedian( int *ml, int *mr, int num, double *F )
{
    double l1, l2, l3, d1, d2, value;
    double *deviation;
    int i, i3;

    if( !ml || !mr || !F )
        return -1;

    deviation = (double *) cvAlloc( (num) * sizeof( double ));

    if( !deviation )
        return -1;

    for( i = 0, i3 = 0; i < num; i++, i3 += 3 )
    {

        l1 = F[0] * mr[i3] + F[1] * mr[i3 + 1] + F[2];
        l2 = F[3] * mr[i3] + F[4] * mr[i3 + 1] + F[5];
        l3 = F[6] * mr[i3] + F[7] * mr[i3 + 1] + F[8];

        d1 = (l1 * ml[i3] + l2 * ml[i3 + 1] + l3) / sqrt( l1 * l1 + l2 * l2 );

        l1 = F[0] * ml[i3] + F[3] * ml[i3 + 1] + F[6];
        l2 = F[1] * ml[i3] + F[4] * ml[i3 + 1] + F[7];
        l3 = F[2] * ml[i3] + F[5] * ml[i3 + 1] + F[8];

        d2 = (l1 * mr[i3] + l2 * mr[i3 + 1] + l3) / sqrt( l1 * l1 + l2 * l2 );

        deviation[i] = (double) (d1 * d1 + d2 * d2);
    }                           /* for */

    if( icvSort( deviation, num ) != CV_NO_ERR )
    {

        cvFree( &deviation );
        return -1;
    }                           /* if */

    value = deviation[num / 2];
    cvFree( &deviation );
    return value;

}                               /* cs_Median */

/*===========================================================================*/
CvStatus
icvSort( double *array, int length )
{
    int i, j, index;
    double swapd;

    if( !array || length < 1 )
        return CV_BADFACTOR_ERR;

    for( i = 0; i < length - 1; i++ )
    {

        index = i;

        for( j = i + 1; j < length; j++ )
        {

            if( array[j] < array[index] )
                index = j;
        }                       /* for */

        if( index - i )
        {

            swapd = array[i];
            array[i] = array[index];
            array[index] = swapd;
        }                       /* if */
    }                           /* for */

    return CV_NO_ERR;

}                               /* cs_Sort */

/*===========================================================================*/
int
icvBoltingPoints( int *ml, int *mr,
                  int num, double *F, double Mj, int **new_ml, int **new_mr, int *new_num )
{
    double l1, l2, l3, d1, d2, sigma;
    int i, j, length;
    int *index;

    if( !ml || !mr || num < 1 || !F || Mj < 0 )
        return -1;

    index = (int *) cvAlloc( (num) * sizeof( int ));

    if( !index )
        return -1;

    length = 0;
    sigma = (double) (2.5 * 1.4826 * (1 + 5. / (num - 7)) * sqrt( Mj ));

    for( i = 0; i < num * 3; i += 3 )
    {

        l1 = F[0] * mr[i] + F[1] * mr[i + 1] + F[2];
        l2 = F[3] * mr[i] + F[4] * mr[i + 1] + F[5];
        l3 = F[6] * mr[i] + F[7] * mr[i + 1] + F[8];

        d1 = (l1 * ml[i] + l2 * ml[i + 1] + l3) / sqrt( l1 * l1 + l2 * l2 );

        l1 = F[0] * ml[i] + F[3] * ml[i + 1] + F[6];
        l2 = F[1] * ml[i] + F[4] * ml[i + 1] + F[7];
        l3 = F[2] * ml[i] + F[5] * ml[i + 1] + F[8];

        d2 = (l1 * mr[i] + l2 * mr[i + 1] + l3) / sqrt( l1 * l1 + l2 * l2 );

        if( d1 * d1 + d2 * d2 <= sigma * sigma )
        {

            index[i / 3] = 1;
            length++;
        }
        else
        {

            index[i / 3] = 0;
        }                       /* if */
    }                           /* for */

    *new_num = length;

    *new_ml = (int *) cvAlloc( (length * 3) * sizeof( int ));

    if( !new_ml )
    {

        cvFree( &index );
        return -1;
    }                           /* if */

    *new_mr = (int *) cvAlloc( (length * 3) * sizeof( int ));

    if( !new_mr )
    {

        cvFree( &new_ml );
        cvFree( &index );
        return -1;
    }                           /* if */

    j = 0;

    for( i = 0; i < num * 3; )
    {

        if( index[i / 3] )
        {

            (*new_ml)[j] = ml[i];
            (*new_mr)[j++] = mr[i++];
            (*new_ml)[j] = ml[i];
            (*new_mr)[j++] = mr[i++];
            (*new_ml)[j] = ml[i];
            (*new_mr)[j++] = mr[i++];
        }
        else
            i += 3;
    }                           /* for */

    cvFree( &index );
    return length;

}                               /* cs_BoltingPoints */

/*===========================================================================*/
CvStatus
icvPoints8( int *ml, int *mr, int num, double *F )
{
    double *U;
    double l1, l2, w, old_norm = -1, new_norm = -2, summ;
    int i3, i9, j, num3, its = 0, a, t;

    if( !ml || !mr || num < 8 || !F )
        return CV_BADFACTOR_ERR;

    U = (double *) cvAlloc( (num * 9) * sizeof( double ));

    if( !U )
        return CV_OUTOFMEM_ERR;

    num3 = num * 3;

    while( !REAL_ZERO( new_norm - old_norm ))
    {

        if( its++ > 1e+2 )
        {

            cvFree( &U );
            return CV_BADFACTOR_ERR;
        }                       /* if */

        old_norm = new_norm;

        for( i3 = 0, i9 = 0; i3 < num3; i3 += 3, i9 += 9 )
        {

            l1 = F[0] * mr[i3] + F[1] * mr[i3 + 1] + F[2];
            l2 = F[3] * mr[i3] + F[4] * mr[i3 + 1] + F[5];

            if( REAL_ZERO( l1 ) && REAL_ZERO( l2 ))
            {

                cvFree( &U );
                return CV_BADFACTOR_ERR;
            }                   /* if */

            w = 1 / (l1 * l1 + l2 * l2);

            l1 = F[0] * ml[i3] + F[3] * ml[i3 + 1] + F[6];
            l2 = F[1] * ml[i3] + F[4] * ml[i3 + 1] + F[7];

            if( REAL_ZERO( l1 ) && REAL_ZERO( l2 ))
            {

                cvFree( &U );
                return CV_BADFACTOR_ERR;
            }                   /* if */

            w += 1 / (l1 * l1 + l2 * l2);
            w = sqrt( w );

            for( j = 0; j < 9; j++ )
            {

                U[i9 + j] = w * (double) ml[i3 + j / 3] * (double) mr[i3 + j % 3];
            }                   /* for */
        }                       /* for */

        new_norm = 0;

        for( a = 0; a < num; a++ )
        {                       /* row */

            summ = 0;

            for( t = 0; t < 9; t++ )
            {

                summ += U[a * 9 + t] * F[t];
            }                   /* for */

            new_norm += summ * summ;
        }                       /* for */

        new_norm = sqrt( new_norm );

        icvAnalyticPoints8( U, num, F );
    }                           /* while */

    cvFree( &U );
    return CV_NO_ERR;

}                               /* cs_Points8 */

/*===========================================================================*/
double
icvAnalyticPoints8( double *A, int num, double *F )
{
    double *U;
    double V[8 * 8];
    double W[8];
    double *f;
    double solution[9];
    double temp1[8 * 8];
    double *temp2;
    double *A_short;
    double norm, summ, best_norm;
    int num8 = num * 8, num9 = num * 9;
    int i, j, j8, j9, value, a, a8, a9, a_num, b, b8, t;

    /* --------- Initialization data ------------------ */

    if( !A || num < 8 || !F )
        return -1;

    best_norm = -1;
    U = (double *) cvAlloc( (num8) * sizeof( double ));

    if( !U )
        return -1;

    f = (double *) cvAlloc( (num) * sizeof( double ));

    if( !f )
    {
        cvFree( &U );
        return -1;
    }                           /* if */

    temp2 = (double *) cvAlloc( (num8) * sizeof( double ));

    if( !temp2 )
    {
        cvFree( &f );
        cvFree( &U );
        return -1;
    }                           /* if */

    A_short = (double *) cvAlloc( (num8) * sizeof( double ));

    if( !A_short )
    {
        cvFree( &temp2 );
        cvFree( &f );
        cvFree( &U );
        return -1;
    }                           /* if */

    for( i = 0; i < 8; i++ )
    {
        for( j8 = 0, j9 = 0; j9 < num9; j8 += 8, j9 += 9 )
        {
            A_short[j8 + i] = A[j9 + i + 1];
        }                       /* for */
    }                           /* for */

    for( i = 0; i < 9; i++ )
    {

        for( j = 0, j8 = 0, j9 = 0; j < num; j++, j8 += 8, j9 += 9 )
        {

            f[j] = -A[j9 + i];

            if( i )
                A_short[j8 + i - 1] = A[j9 + i - 1];
        }                       /* for */

        value = icvSingularValueDecomposition( num, 8, A_short, W, 1, U, 1, V );

        if( !value )
        {                       /* -----------  computing the solution  ----------- */

            /*  -----------  W = W(-1)  ----------- */
            for( j = 0; j < 8; j++ )
            {
                if( !REAL_ZERO( W[j] ))
                    W[j] = 1 / W[j];
            }                   /* for */

            /* -----------  temp1 = V * W(-1)  ----------- */
            for( a8 = 0; a8 < 64; a8 += 8 )
            {                   /* row */
                for( b = 0; b < 8; b++ )
                {               /* column */
                    temp1[a8 + b] = V[a8 + b] * W[b];
                }               /* for */
            }                   /* for */

            /*  -----------  temp2 = V * W(-1) * U(T)  ----------- */
            for( a8 = 0, a_num = 0; a8 < 64; a8 += 8, a_num += num )
            {                   /* row */
                for( b = 0, b8 = 0; b < num; b++, b8 += 8 )
                {               /* column */

                    temp2[a_num + b] = 0;

                    for( t = 0; t < 8; t++ )
                    {

                        temp2[a_num + b] += temp1[a8 + t] * U[b8 + t];
                    }           /* for */
                }               /* for */
            }                   /* for */

            /* -----------  solution = V * W(-1) * U(T) * f  ----------- */
            for( a = 0, a_num = 0; a < 8; a++, a_num += num )
            {                   /* row */
                for( b = 0; b < num; b++ )
                {               /* column */

                    solution[a] = 0;

                    for( t = 0; t < num && W[a]; t++ )
                    {
                        solution[a] += temp2[a_num + t] * f[t];
                    }           /* for */
                }               /* for */
            }                   /* for */

            for( a = 8; a > 0; a-- )
            {

                if( a == i )
                    break;
                solution[a] = solution[a - 1];
            }                   /* for */

            solution[a] = 1;

            norm = 0;

            for( a9 = 0; a9 < num9; a9 += 9 )
            {                   /* row */

                summ = 0;

                for( t = 0; t < 9; t++ )
                {

                    summ += A[a9 + t] * solution[t];
                }               /* for */

                norm += summ * summ;
            }                   /* for */

            norm = sqrt( norm );

            if( best_norm == -1 || norm < best_norm )
            {

                for( j = 0; j < 9; j++ )
                    F[j] = solution[j];

                best_norm = norm;
            }                   /* if */
        }                       /* if */
    }                           /* for */

    cvFree( &A_short );
    cvFree( &temp2 );
    cvFree( &f );
    cvFree( &U );

    return best_norm;

}                               /* cs_AnalyticPoints8 */

/*===========================================================================*/
CvStatus
icvRank2Constraint( double *F )
{
    double U[9], V[9], W[3];
    double aW[3]; 
    int i, i3, j, j3, t;

    if( F == 0 )
        return CV_BADFACTOR_ERR;

    if( icvSingularValueDecomposition( 3, 3, F, W, 1, U, 1, V ))
        return CV_BADFACTOR_ERR;

    aW[0] = fabs(W[0]);
    aW[1] = fabs(W[1]);
    aW[2] = fabs(W[2]);

    if( aW[0] < aW[1] )
    {
        if( aW[0] < aW[2] )
        {

            if( REAL_ZERO( W[0] ))
                return CV_NO_ERR;
            else
                W[0] = 0;
        }
        else
        {

            if( REAL_ZERO( W[2] ))
                return CV_NO_ERR;
            else
                W[2] = 0;
        }                       /* if */
    }
    else
    {

        if( aW[1] < aW[2] )
        {

            if( REAL_ZERO( W[1] ))
                return CV_NO_ERR;
            else
                W[1] = 0;
        }
        else
        {
            if( REAL_ZERO( W[2] ))
                return CV_NO_ERR;
            else
                W[2] = 0;
        }                       /* if */
    }                           /* if */

    for( i = 0; i < 3; i++ )
    {
        for( j3 = 0; j3 < 9; j3 += 3 )
        {
            U[j3 + i] *= W[i];
        }                       /* for */
    }                           /* for */

    for( i = 0, i3 = 0; i < 3; i++, i3 += 3 )
    {
        for( j = 0, j3 = 0; j < 3; j++, j3 += 3 )
        {

            F[i3 + j] = 0;

            for( t = 0; t < 3; t++ )
            {
                F[i3 + j] += U[i3 + t] * V[j3 + t];
            }                   /* for */
        }                       /* for */
    }                           /* for */

    return CV_NO_ERR;
}                               /* cs_Rank2Constraint */


/*===========================================================================*/

int
icvSingularValueDecomposition( int M,
                               int N,
                               double *A,
                               double *W, int get_U, double *U, int get_V, double *V )
{
    int i = 0, j, k, l = 0, i1, k1, l1 = 0;
    int iterations, error = 0, jN, iN, kN, lN = 0;
    double *rv1;
    double c, f, g, h, s, x, y, z, scale, anorm;
    double af, ag, ah, t;
    int MN = M * N;
    int NN = N * N;

    /*  max_iterations - maximum number QR-iterations
       cc - reduces requirements to number stitch (cc>1)
     */

    int max_iterations = 100;
    double cc = 100;

    if( M < N )
        return N;

    rv1 = (double *) cvAlloc( N * sizeof( double ));

    if( rv1 == 0 )
        return N;

    for( iN = 0; iN < MN; iN += N )
    {
        for( j = 0; j < N; j++ )
            U[iN + j] = A[iN + j];
    }                           /* for */

    /*  Adduction to bidiagonal type (transformations of reflection).
       Bidiagonal matrix is located in W (diagonal elements)
       and in rv1 (upperdiagonal elements)
     */

    g = 0;
    scale = 0;
    anorm = 0;

    for( i = 0, iN = 0; i < N; i++, iN += N )
    {

        l = i + 1;
        lN = iN + N;
        rv1[i] = scale * g;

        /*  Multiplyings on the left  */

        g = 0;
        s = 0;
        scale = 0;

        for( kN = iN; kN < MN; kN += N )
            scale += fabs( U[kN + i] );

        if( !REAL_ZERO( scale ))
        {

            for( kN = iN; kN < MN; kN += N )
            {

                U[kN + i] /= scale;
                s += U[kN + i] * U[kN + i];
            }                   /* for */

            f = U[iN + i];
            g = -sqrt( s ) * Sgn( f );
            h = f * g - s;
            U[iN + i] = f - g;

            for( j = l; j < N; j++ )
            {

                s = 0;

                for( kN = iN; kN < MN; kN += N )
                {

                    s += U[kN + i] * U[kN + j];
                }               /* for */

                f = s / h;

                for( kN = iN; kN < MN; kN += N )
                {

                    U[kN + j] += f * U[kN + i];
                }               /* for */
            }                   /* for */

            for( kN = iN; kN < MN; kN += N )
                U[kN + i] *= scale;
        }                       /* if */

        W[i] = scale * g;

        /*  Multiplyings on the right  */

        g = 0;
        s = 0;
        scale = 0;

        for( k = l; k < N; k++ )
            scale += fabs( U[iN + k] );

        if( !REAL_ZERO( scale ))
        {

            for( k = l; k < N; k++ )
            {

                U[iN + k] /= scale;
                s += (U[iN + k]) * (U[iN + k]);
            }                   /* for */

            f = U[iN + l];
            g = -sqrt( s ) * Sgn( f );
            h = f * g - s;
            U[i * N + l] = f - g;

            for( k = l; k < N; k++ )
                rv1[k] = U[iN + k] / h;

            for( jN = lN; jN < MN; jN += N )
            {

                s = 0;

                for( k = l; k < N; k++ )
                    s += U[jN + k] * U[iN + k];

                for( k = l; k < N; k++ )
                    U[jN + k] += s * rv1[k];

            }                   /* for */

            for( k = l; k < N; k++ )
                U[iN + k] *= scale;
        }                       /* if */

        t = fabs( W[i] );
        t += fabs( rv1[i] );
        anorm = MAX( anorm, t );
    }                           /* for */

    anorm *= cc;

    /*  accumulation of right transformations, if needed  */

    if( get_V )
    {

        for( i = N - 1, iN = NN - N; i >= 0; i--, iN -= N )
        {

            if( i < N - 1 )
            {

                /*  pass-by small g  */
                if( !REAL_ZERO( g ))
                {

                    for( j = l, jN = lN; j < N; j++, jN += N )
                        V[jN + i] = U[iN + j] / U[iN + l] / g;

                    for( j = l; j < N; j++ )
                    {

                        s = 0;

                        for( k = l, kN = lN; k < N; k++, kN += N )
                            s += U[iN + k] * V[kN + j];

                        for( kN = lN; kN < NN; kN += N )
                            V[kN + j] += s * V[kN + i];
                    }           /* for */
                }               /* if */

                for( j = l, jN = lN; j < N; j++, jN += N )
                {
                    V[iN + j] = 0;
                    V[jN + i] = 0;
                }               /* for */
            }                   /* if */

            V[iN + i] = 1;
            g = rv1[i];
            l = i;
            lN = iN;
        }                       /* for */
    }                           /* if */

    /*  accumulation of left transformations, if needed  */

    if( get_U )
    {

        for( i = N - 1, iN = NN - N; i >= 0; i--, iN -= N )
        {

            l = i + 1;
            lN = iN + N;
            g = W[i];

            for( j = l; j < N; j++ )
                U[iN + j] = 0;

            /*  pass-by small g  */
            if( !REAL_ZERO( g ))
            {

                for( j = l; j < N; j++ )
                {

                    s = 0;

                    for( kN = lN; kN < MN; kN += N )
                        s += U[kN + i] * U[kN + j];

                    f = s / U[iN + i] / g;

                    for( kN = iN; kN < MN; kN += N )
                        U[kN + j] += f * U[kN + i];
                }               /* for */

                for( jN = iN; jN < MN; jN += N )
                    U[jN + i] /= g;
            }
            else
            {

                for( jN = iN; jN < MN; jN += N )
                    U[jN + i] = 0;
            }                   /* if */

            U[iN + i] += 1;
        }                       /* for */
    }                           /* if */

    /*  Iterations QR-algorithm for bidiagonal matrixes
       W[i] - is the main diagonal
       rv1[i] - is the top diagonal, rv1[0]=0.
     */

    for( k = N - 1; k >= 0; k-- )
    {

        k1 = k - 1;
        iterations = 0;

        for( ;; )
        {

            /*  Cycle: checking a possibility of fission matrix  */
            for( l = k; l >= 0; l-- )
            {

                l1 = l - 1;

                if( REAL_ZERO( rv1[l] ) || REAL_ZERO( W[l1] ))
                    break;
            }                   /* for */

            if( !REAL_ZERO( rv1[l] ))
            {

                /*  W[l1] = 0,  matrix possible to fission
                   by clearing out rv1[l]  */

                c = 0;
                s = 1;

                for( i = l; i <= k; i++ )
                {

                    f = s * rv1[i];
                    rv1[i] = c * rv1[i];

                    /*  Rotations are done before the end of the block,
                       or when element in the line is finagle.
                     */

                    if( REAL_ZERO( f ))
                        break;

                    g = W[i];

                    /*  Scaling prevents finagling H ( F!=0!) */

                    af = fabs( f );
                    ag = fabs( g );

                    if( af < ag )
                        h = ag * sqrt( 1 + (f / g) * (f / g) );
                    else
                        h = af * sqrt( 1 + (f / g) * (f / g) );

                    W[i] = h;
                    c = g / h;
                    s = -f / h;

                    if( get_U )
                    {

                        for( jN = 0; jN < MN; jN += N )
                        {

                            y = U[jN + l1];
                            z = U[jN + i];
                            U[jN + l1] = y * c + z * s;
                            U[jN + i] = -y * s + z * c;
                        }       /* for */
                    }           /* if */
                }               /* for */
            }                   /* if */


            /*  Output in this place of program means,
               that rv1[L] = 0, matrix fissioned
               Iterations of the process of the persecution
               will be executed always for
               the bottom block ( from l before k ),
               with increase l possible.
             */

            z = W[k];

            if( l == k )
                break;

            /*  Completion iterations: lower block
               became trivial ( rv1[K]=0)  */

            if( iterations++ == max_iterations )
                return k;

            /*  Shift is computed on the lowest order 2 minor.  */

            x = W[l];
            y = W[k1];
            g = rv1[k1];
            h = rv1[k];

            /*  consequent fission prevents forming a machine zero  */
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h) / y;

            /*  prevented overflow  */
            if( fabs( f ) > 1 )
            {
                g = fabs( f );
                g *= sqrt( 1 + (1 / f) * (1 / f) );
            }
            else
                g = sqrt( f * f + 1 );

            f = ((x - z) * (x + z) + h * (y / (f + fabs( g ) * Sgn( f )) - h)) / x;
            c = 1;
            s = 1;

            for( i1 = l; i1 <= k1; i1++ )
            {

                i = i1 + 1;
                g = rv1[i];
                y = W[i];
                h = s * g;
                g *= c;

                /*  Scaling at calculation Z prevents its clearing,
                   however if F and H both are zero - pass-by of fission on Z.
                 */

                af = fabs( f );
                ah = fabs( h );

                if( af < ah )
                    z = ah * sqrt( 1 + (f / h) * (f / h) );

                else
                {

                    z = 0;
                    if( !REAL_ZERO( af ))
                        z = af * sqrt( 1 + (h / f) * (h / f) );
                }               /* if */

                rv1[i1] = z;

                /*  if Z=0, the rotation is free.  */
                if( !REAL_ZERO( z ))
                {

                    c = f / z;
                    s = h / z;
                }               /* if */

                f = x * c + g * s;
                g = -x * s + g * c;
                h = y * s;
                y *= c;

                if( get_V )
                {

                    for( jN = 0; jN < NN; jN += N )
                    {

                        x = V[jN + i1];
                        z = V[jN + i];
                        V[jN + i1] = x * c + z * s;
                        V[jN + i] = -x * s + z * c;
                    }           /* for */
                }               /* if */

                af = fabs( f );
                ah = fabs( h );

                if( af < ah )
                    z = ah * sqrt( 1 + (f / h) * (f / h) );
                else
                {

                    z = 0;
                    if( !REAL_ZERO( af ))
                        z = af * sqrt( 1 + (h / f) * (h / f) );
                }               /* if */

                W[i1] = z;

                if( !REAL_ZERO( z ))
                {

                    c = f / z;
                    s = h / z;
                }               /* if */

                f = c * g + s * y;
                x = -s * g + c * y;

                if( get_U )
                {

                    for( jN = 0; jN < MN; jN += N )
                    {

                        y = U[jN + i1];
                        z = U[jN + i];
                        U[jN + i1] = y * c + z * s;
                        U[jN + i] = -y * s + z * c;
                    }           /* for */
                }               /* if */
            }                   /* for */

            rv1[l] = 0;
            rv1[k] = f;
            W[k] = x;
        }                       /* for */

        if( z < 0 )
        {

            W[k] = -z;

            if( get_V )
            {

                for( jN = 0; jN < NN; jN += N )
                    V[jN + k] *= -1;
            }                   /* if */
        }                       /* if */
    }                           /* for */

    cvFree( &rv1 );

    return error;

}                               /* vm_SingularValueDecomposition */

/*========================================================================*/

/* Obsolete functions. Just for ViewMorping */
/*=====================================================================================*/

int
icvGaussMxN( double *A, double *B, int M, int N, double **solutions )
{
    int *variables;
    int row, swapi, i, i_best = 0, j, j_best = 0, t;
    double swapd, ratio, bigest;

    if( !A || !B || !M || !N )
        return -1;

    variables = (int *) cvAlloc( (size_t) N * sizeof( int ));

    if( variables == 0 )
        return -1;

    for( i = 0; i < N; i++ )
    {
        variables[i] = i;
    }                           /* for */

    /* -----  Direct way  ----- */

    for( row = 0; row < M; row++ )
    {

        bigest = 0;

        for( j = row; j < M; j++ )
        {                       /* search non null element */
            for( i = row; i < N; i++ )
            {
                double a = fabs( A[j * N + i] ), b = fabs( bigest );
                if( a > b )
                {
                    bigest = A[j * N + i];
                    i_best = i;
                    j_best = j;
                }               /* if */
            }                   /* for */
        }                       /* for */

        if( REAL_ZERO( bigest ))
            break;              /* if all shank elements are null */

        if( j_best - row )
        {

            for( t = 0; t < N; t++ )
            {                   /* swap a rows */

                swapd = A[row * N + t];
                A[row * N + t] = A[j_best * N + t];
                A[j_best * N + t] = swapd;
            }                   /* for */

            swapd = B[row];
            B[row] = B[j_best];
            B[j_best] = swapd;
        }                       /* if */

        if( i_best - row )
        {

            for( t = 0; t < M; t++ )
            {                   /* swap a columns  */

                swapd = A[t * N + i_best];
                A[t * N + i_best] = A[t * N + row];
                A[t * N + row] = swapd;
            }                   /* for */

            swapi = variables[row];
            variables[row] = variables[i_best];
            variables[i_best] = swapi;
        }                       /* if */

        for( i = row + 1; i < M; i++ )
        {                       /* recounting A and B */

            ratio = -A[i * N + row] / A[row * N + row];
            B[i] += B[row] * ratio;

            for( j = N - 1; j >= row; j-- )
            {

                A[i * N + j] += A[row * N + j] * ratio;
            }                   /* for */
        }                       /* for */
    }                           /* for */

    if( row < M )
    {                           /* if rank(A)<M */

        for( j = row; j < M; j++ )
        {
            if( !REAL_ZERO( B[j] ))
            {

                cvFree( &variables );
                return -1;      /* if system is antithetic */
            }                   /* if */
        }                       /* for */

        M = row;                /* decreasing size of the task */
    }                           /* if */

    /* ----- Reverse way ----- */

    if( M < N )
    {                           /* if solution are not exclusive */

        *solutions = (double *) cvAlloc( ((N - M + 1) * N) * sizeof( double ));

        if( *solutions == 0 )
        {
            cvFree( &variables );
            return -1;
        }


        for( t = M; t <= N; t++ )
        {
            for( j = M; j < N; j++ )
            {

                (*solutions)[(t - M) * N + variables[j]] = (double) (t == j);
            }                   /* for */

            for( i = M - 1; i >= 0; i-- )
            {                   /* finding component of solution */

                if( t < N )
                {
                    (*solutions)[(t - M) * N + variables[i]] = 0;
                }
                else
                {
                    (*solutions)[(t - M) * N + variables[i]] = B[i] / A[i * N + i];
                }               /* if */

                for( j = i + 1; j < N; j++ )
                {

                    (*solutions)[(t - M) * N + variables[i]] -=
                        (*solutions)[(t - M) * N + variables[j]] * A[i * N + j] / A[i * N + i];
                }               /* for */
            }                   /* for */
        }                       /* for */

        cvFree( &variables );
        return N - M;
    }                           /* if */

    *solutions = (double *) cvAlloc( (N) * sizeof( double ));

    if( solutions == 0 )
        return -1;

    for( i = N - 1; i >= 0; i-- )
    {                           /* finding exclusive solution */

        (*solutions)[variables[i]] = B[i] / A[i * N + i];

        for( j = i + 1; j < N; j++ )
        {

            (*solutions)[variables[i]] -=
                (*solutions)[variables[j]] * A[i * N + j] / A[i * N + i];
        }                       /* for */
    }                           /* for */

    cvFree( &variables );
    return 0;

}                               /* icvGaussMxN */


/*======================================================================================*/
/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name:    icvPoint7
//    Purpose:
//      
//      
//    Context:
//    Parameters:
//     
//      
//      
//     
//      
//    
//     
//    Returns:
//      CV_NO_ERR if all Ok or error code
//    Notes:
//F*/

CvStatus
icvPoint7( int *ml, int *mr, double *F, int *amount )
{
    double A[63], B[7];
    double *solutions;
    double a2, a1, a0;
    double squares[6];
    int i, j;

/*    int         amount; */
/*    float*     F; */

    CvStatus error = CV_BADFACTOR_ERR;

/*    F = (float*)matrix->m; */

    if( !ml || !mr || !F )
        return CV_BADFACTOR_ERR;

    for( i = 0; i < 7; i++ )
    {
        for( j = 0; j < 9; j++ )
        {

            A[i * 9 + j] = (double) ml[i * 3 + j / 3] * (double) mr[i * 3 + j % 3];
        }                       /* for */
        B[i] = 0;
    }                           /* for */

    *amount = 0;

    if( icvGaussMxN( A, B, 7, 9, &solutions ) == 2 )
    {
        if( icvGetCoef( solutions, solutions + 9, &a2, &a1, &a0 ) == CV_NO_ERR )
        {
            icvCubic( a2, a1, a0, squares );

            for( i = 0; i < 1; i++ )
            {

                if( REAL_ZERO( squares[i * 2 + 1] ))
                {

                    for( j = 0; j < 9; j++ )
                    {

                        F[*amount + j] = (float) (squares[i] * solutions[j] +
                                                  (1 - squares[i]) * solutions[j + 9]);
                    }           /* for */

                    *amount += 9;

                    error = CV_NO_ERR;
                }               /* if */
            }                   /* for */

            cvFree( &solutions );
            return error;
        }
        else
        {
            cvFree( &solutions );
        }                       /* if */

    }
    else
    {
        cvFree( &solutions );
    }                           /* if */

    return error;
}                               /* icvPoint7 */

