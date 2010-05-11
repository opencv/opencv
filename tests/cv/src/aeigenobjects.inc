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

/*______________________________________________________________________________________*/
/*                                                                                      */
/*              Test functions for the Eigen Objects functions group                    */
/*______________________________________________________________________________________*/

#include "cvtest.h"

static int _cvCalcCovarMatrix_8u32fR_q( int      nObjects,
                                     uchar**  objects,
                                     int      objStep,
                                     float*   avg,
                                     int      avgStep,
                                     CvSize size,
                                     float*   covarMatrix )
{
    int i, j;

    if ( nObjects < 2 )                                          return CV_BADFACTOR_ERR;
    if ( size.width > objStep || 4*size.width > avgStep || size.height < 1)
                                                                 return CV_BADSIZE_ERR;
    if ( objects == NULL || avg == NULL || covarMatrix == NULL ) return CV_NULLPTR_ERR;

    avgStep /= 4;

    for(i=0; i<nObjects; i++)
    {
        uchar* bu = objects[i];
        for(j=i; j<nObjects; j++)
        {
            int     ij = i*nObjects + j, k, l;
            float    w = 0.f;
            float*   a = avg;
            uchar* bu1 = bu;
            uchar* bu2 = objects[j];

            for(k=0; k<size.height; k++, bu1 += objStep, bu2 += objStep, a += avgStep)
            {
                for(l = 0; l < size.width-3; l+=4)
                {
                    float  f = a  [l];
                    uchar u1 = bu1[l];
                    uchar u2 = bu2[l];
                    w += ( u1 - f ) * ( u2 - f );
                    f  = a  [l+1];
                    u1 = bu1[l+1];
                    u2 = bu2[l+1];
                    w += ( u1 - f ) * ( u2 - f );
                    f  = a  [l+2];
                    u1 = bu1[l+2];
                    u2 = bu2[l+2];
                    w += ( u1 - f ) * ( u2 - f );
                    f  = a  [l+3];
                    u1 = bu1[l+3];
                    u2 = bu2[l+3];
                    w += ( u1 - f ) * ( u2 - f );
                }
                for(; l < size.width; l++)
                {
                    float  f = a  [l];
                    uchar u1 = bu1[l];
                    uchar u2 = bu2[l];
                    w += ( u1 - f ) * ( u2 - f );
                }
            }

            covarMatrix[ij] = w;
            ij = j*nObjects + i;
            covarMatrix[ij] = w;
        }
    }

    return CV_NO_ERR;
}   /* end of _cvCalcCovarMatrix_8u32fR */


/* copy of _cvJacobiEigen_32f */
int _cvJacobiEigens_32f ( float* A,
                               float* V,
                               float* E,
                               int    n,
                               float  eps )
{
    int i, j, k, ind;
    float *AA = A, *VV = V;
    double Amax, anorm=0, ax;

    if ( A == NULL || V == NULL || E == NULL ) return CV_NULLPTR_ERR;
    if ( n <= 0 )                              return CV_BADSIZE_ERR;
    if (eps < 1.0e-7f )  eps = 1.0e-7f;

    /*-------- Prepare --------*/
    for(i=0; i<n; i++, VV+=n, AA+=n)
    {
        for(j=0; j<i; j++)
        {
            double Am = AA[j];
            anorm += Am*Am;
        }
        for(j=0; j<n; j++) VV[j] = 0.f;
        VV[i] = 1.f;
    }

    anorm = sqrt( anorm + anorm );
    ax    = anorm*eps/n;
    Amax  = anorm;

    while ( Amax > ax )
    {
        Amax /= n;
        do  /* while (ind) */
        {
            int p, q;
            float *V1  = V, *A1  = A;
            ind = 0;
            for(p=0; p<n-1; p++, A1+=n, V1+=n)
            {
                float *A2 = A + n*(p+1), *V2 = V + n*(p+1);
                for(q=p+1; q<n; q++, A2+=n, V2+=n)
                {
                    double x, y, c, s, c2, s2, a;
                    float *A3, Apq=A1[q], App, Aqq, Aip, Aiq, Vpi, Vqi;
                    if( fabs( Apq ) < Amax ) continue;

                    ind=1;

                    /*---- Calculation of rotation angle's sine & cosine ----*/
                    App = A1[p];
                    Aqq = A2[q];
                    y   = 5.0e-1*(App - Aqq);
                    x = -Apq / sqrt(Apq*Apq + y*y);
                    if(y<0.0) x = -x;
                    s = x / sqrt(2.0*(1.0 + sqrt(1.0 - x*x)));
                    s2 = s*s;
                    c  = sqrt(1.0 - s2);
                    c2 = c*c;
                    a  = 2.0*Apq*c*s;

                    /*---- Apq annulation ----*/
                    A3 = A;
                    for(i=0; i<p; i++, A3+=n)
                    {
                        Aip = A3[p];
                        Aiq = A3[q];
                        Vpi = V1[i];
                        Vqi = V2[i];
                        A3[p] = (float)(Aip*c - Aiq*s);
                        A3[q] = (float)(Aiq*c + Aip*s);
                        V1[i] = (float)(Vpi*c - Vqi*s);
                        V2[i] = (float)(Vqi*c + Vpi*s);
                    }
                    for(; i<q; i++, A3+=n)
                    {
                        Aip = A1[i];
                        Aiq = A3[q];
                        Vpi = V1[i];
                        Vqi = V2[i];
                        A1[i] = (float)(Aip*c - Aiq*s);
                        A3[q] = (float)(Aiq*c + Aip*s);
                        V1[i] = (float)(Vpi*c - Vqi*s);
                        V2[i] = (float)(Vqi*c + Vpi*s);
                    }
                    for(; i<n; i++)
                    {
                        Aip = A1[i];
                        Aiq = A2[i];
                        Vpi = V1[i];
                        Vqi = V2[i];
                        A1[i] = (float)(Aip*c - Aiq*s);
                        A2[i] = (float)(Aiq*c + Aip*s);
                        V1[i] = (float)(Vpi*c - Vqi*s);
                        V2[i] = (float)(Vqi*c + Vpi*s);
                    }
                    A1[p] = (float)(App*c2 + Aqq*s2 - a);
                    A2[q] = (float)(App*s2 + Aqq*c2 + a);
                    A1[q] = A2[p] = 0.0f;
                } /*q*/
            }     /*p*/
        } while (ind);
        Amax /= n;
    }   /* while ( Amax > ax ) */

    for(i=0, k=0; i<n; i++, k+=n+1) E[i] = A[k];
    /*printf(" M = %d\n", M);*/

    /* -------- ordering --------*/
    for(i=0; i<n; i++)
    {
        int m = i;
        float Em = (float)fabs(E[i]);
        for(j=i+1; j<n; j++)
        {
            float Ej = (float)fabs(E[j]);
            m  = ( Em < Ej ) ?  j :  m;
            Em = ( Em < Ej ) ? Ej : Em;
        }
        if( m != i )
        {
            int l;
            float b = E[i];
            E[i] = E[m];
            E[m] = b;
            for(j=0, k=i*n, l=m*n; j<n; j++, k++, l++)
            {
                b    = V[k];
                V[k] = V[l];
                V[l] = b;
            }
        }
    }

    return CV_NO_ERR;
}



/*______________________________________________________________________________________*/

int  _cvCalcEigenObjects_8u32fR_q( int      nObjects,
                                      uchar**  objects,
                                      int      objStep,
                                      float**  eigObjs,
                                      int      eigStep,
                                      CvSize size,
                                      float*   eigVals,
                                      float*   avg,
                                      int      avgStep,
                                      int*     nEigObjs,
                                      double*  eps      )
{
    int i, j, k, l;
    uchar *bu;
    float *c=0, *ev=0, *bf, *bf1, *bf2, m;
    int  r;


    if ( nObjects < 2 )                                         return CV_BADFACTOR_ERR;
    if ( size.width > objStep || 4*size.width > eigStep ||
         4*size.width > avgStep || size.height < 1)             return CV_BADSIZE_ERR;
    if ( objects == NULL || eigObjs == NULL || eigVals == NULL ||
         avg == NULL || nEigObjs == NULL || eps == NULL )       return CV_NULLPTR_ERR;
    for( i=0; i<nObjects;  i++ ) if( objects[i] == NULL ) return CV_NULLPTR_ERR;
    for( i=0; i<*nEigObjs; i++ ) if( eigObjs[i] == NULL ) return CV_NULLPTR_ERR;

    eigStep /= 4;
    avgStep /= 4;

/* Calculation of averaged object */
    bf = avg;
    for(i = 0; i < size.height; i++, bf += avgStep)
        for(j = 0; j < size.width; j++)
            bf[j] = 0.f;
    for(k = 0; k < nObjects; k++)
    {
        bu = objects[k];
        bf = avg;
        for(i = 0; i < size.height; i++, bu +=objStep, bf += avgStep)
            for(j = 0; j < size.width; j++)
                bf[j] += bu[j];
    }
    m = 1.0f/(float)nObjects;
    bf = avg;
    for(i = 0; i < size.height; i++, bf += avgStep)
        for(j = 0; j < size.width; j++)
            bf[j] *= m;

/* Calculation of covariance matrix */
    c  = (float*)cvAlloc ( sizeof(float)*nObjects*nObjects );
    if(c==NULL) return CV_OUTOFMEM_ERR;

    r = _cvCalcCovarMatrix_8u32fR_q ( nObjects, objects, objStep,
                                     avg, 4*avgStep, size, c );
    if(r) { cvFree( &c );  return r; }

/* Calculation of eigenvalues & eigenvectors */
    ev = (float*)cvAlloc ( sizeof(float)*nObjects*nObjects );
    if(ev==NULL) { cvFree( &c );  return CV_OUTOFMEM_ERR; }

    _cvJacobiEigens_32f( c, ev, eigVals, nObjects, 0.0f );
    cvFree( &c );

    for(i=0; i<*nEigObjs; i++) if( fabs(eigVals[i]/eigVals[0]) < *eps ) break;
    *nEigObjs = i;
    *eps = fabs(eigVals[*nEigObjs-1]/eigVals[0]);

/* Calculation of eigenobjects */
    bf2 = ev;
    for(i=0; i<*nEigObjs; i++, bf2+=nObjects)
    {
        float  e = (float)(1.0/sqrt(eigVals[i]));
        float* u = eigObjs[i];

        bf  = u;
        for(l=0; l<size.height; l++, bf+=eigStep)
            for(j=0; j<size.width; j++) bf[j] = 0.0f;

        for(k=0; k<nObjects; k++)
        {
            float v = e*bf2[k];
            bf  = u;
            bu  = objects[k];
            bf1 = avg;
            for(l=0; l<size.height; l++, bf+=eigStep, bf1+=avgStep, bu+=objStep)
                for(j=0; j<size.width; j++) bf[j] += v * ((float)bu[j] - bf1[j]);
        }
    }

    cvFree( &ev );
    return CV_NO_ERR;
} /* --- End of _cvCalcEigenObjects_8u32fR --- */
/*______________________________________________________________________________________*/

float _cvCalcDecompCoeff_8u32fR_q( uchar*  obj,
                                   int     objStep,
                                   float*  eigObj,
                                   int     eigStep,
                                   float*  avg,
                                   int     avgStep,
                                   CvSize size )
{
    int i, k;
    float w = 0.0f;

    if ( size.width > objStep || 4*size.width > eigStep
         || 4*size.width > avgStep || size.height < 1)  return -1.0e30f;
    if ( obj == NULL || eigObj == NULL || avg == NULL ) return -1.0e30f;

    eigStep /= 4;
    avgStep /= 4;

    for(i = 0; i < size.height; i++, obj += objStep, eigObj += eigStep, avg += avgStep)
        for(k = 0; k < size.width; k++)
            w += eigObj[k]*( (float)obj[k] - avg[k] );

    return w;
}
/*______________________________________________________________________________________*/

int  _cvEigenDecomposite_8u32fR_q( uchar*  obj,
                                          int     objStep,
                                          int     nEigObjs,
                                          float** eigObjs,
                                          int     eigStep,
                                          float*  avg,
                                          int     avgStep,
                                          CvSize size,
                                          float*  coeffs )
{
    int i;

    if ( nEigObjs < 2 )                                    return CV_BADFACTOR_ERR;
    if ( size.width > objStep || 4*size.width > eigStep ||
         4*size.width > avgStep || size.height < 1)        return CV_BADSIZE_ERR;
    if ( obj == NULL || eigObjs == NULL || coeffs == NULL || avg == NULL)
        return CV_NULLPTR_ERR;

    for(i=0; i<nEigObjs; i++)
    {
        float w = _cvCalcDecompCoeff_8u32fR_q( obj, objStep, eigObjs[i], eigStep,
                                              avg, avgStep, size );
        if( w < -1.0e29f ) return CV_NOTDEFINED_ERR;
        coeffs[i] = w;
    }
    return CV_NO_ERR;
}
/*______________________________________________________________________________________*/

int  _cvEigenProjection_8u32fR_q( int     nEigObjs,
                                         float** eigens,
                                         int     eigStep,
                                         float*  coeffs,
                                         float*  avg,
                                         int     avgStep,
                                         uchar*  rest,
                                         int     restStep,
                                         CvSize size )
{
    int i, j, k;

    if ( size.width > avgStep || 4*size.width > eigStep || size.height < 1)
                                                        return CV_BADSIZE_ERR;
    if ( rest == NULL || eigens == NULL || avg == NULL || coeffs == NULL )
                                                        return CV_NULLPTR_ERR;
    eigStep /= 4;
    avgStep /= 4;

    for(i = 0; i < size.height; i++, rest+=restStep, avg+=avgStep)
    {
        int ij = i*eigStep;
        for(j = 0; j < size.width; j++, ij++)
        {
            float w = avg[j];
            for(k=0; k<nEigObjs-3; k+=4)
            {
                float* b = eigens[k];
                w += coeffs[k  ] * b[ij];
                b = eigens [k+1];
                w += coeffs[k+1] * b[ij];
                b = eigens [k+2];
                w += coeffs[k+2] * b[ij];
                b = eigens [k+3];
                w += coeffs[k+3] * b[ij];
            }
            for(; k<nEigObjs; k++)
            {
                float* b = eigens[k];
                w += coeffs[k] * b[ij];
            }
            w = w<-0.499999f ? -0.499999f : w>255.499f ? 255.499f : w;
            rest[j] = (uchar)cvRound( w );
        }
    }
    return CV_NO_ERR;
}
/*______________________________________________________________________________________*/

/*   << End  of  file >>  */
