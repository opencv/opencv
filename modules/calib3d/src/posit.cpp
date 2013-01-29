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

/* POSIT structure */
struct CvPOSITObject
{
    int N;
    float* inv_matr;
    float* obj_vecs;
    float* img_vecs;
};

static void icvPseudoInverse3D( float *a, float *b, int n, int method );

static  CvStatus  icvCreatePOSITObject( CvPoint3D32f *points,
                                        int numPoints,
                                        CvPOSITObject **ppObject )
{
    int i;

    /* Compute size of required memory */
    /* buffer for inverse matrix = N*3*float */
    /* buffer for storing weakImagePoints = numPoints * 2 * float */
    /* buffer for storing object vectors = N*3*float */
    /* buffer for storing image vectors = N*2*float */

    int N = numPoints - 1;
    int inv_matr_size = N * 3 * sizeof( float );
    int obj_vec_size = inv_matr_size;
    int img_vec_size = N * 2 * sizeof( float );
    CvPOSITObject *pObject;

    /* check bad arguments */
    if( points == NULL )
        return CV_NULLPTR_ERR;
    if( numPoints < 4 )
        return CV_BADSIZE_ERR;
    if( ppObject == NULL )
        return CV_NULLPTR_ERR;

    /* memory allocation */
    pObject = (CvPOSITObject *) cvAlloc( sizeof( CvPOSITObject ) +
                                         inv_matr_size + obj_vec_size + img_vec_size );

    if( !pObject )
        return CV_OUTOFMEM_ERR;

    /* part the memory between all structures */
    pObject->N = N;
    pObject->inv_matr = (float *) ((char *) pObject + sizeof( CvPOSITObject ));
    pObject->obj_vecs = (float *) ((char *) (pObject->inv_matr) + inv_matr_size);
    pObject->img_vecs = (float *) ((char *) (pObject->obj_vecs) + obj_vec_size);

/****************************************************************************************\
*          Construct object vectors from object points                                   *
\****************************************************************************************/
    for( i = 0; i < numPoints - 1; i++ )
    {
        pObject->obj_vecs[i] = points[i + 1].x - points[0].x;
        pObject->obj_vecs[N + i] = points[i + 1].y - points[0].y;
        pObject->obj_vecs[2 * N + i] = points[i + 1].z - points[0].z;
    }
/****************************************************************************************\
*   Compute pseudoinverse matrix                                                         *
\****************************************************************************************/
    icvPseudoInverse3D( pObject->obj_vecs, pObject->inv_matr, N, 0 );

    *ppObject = pObject;
    return CV_NO_ERR;
}


static  CvStatus  icvPOSIT( CvPOSITObject *pObject, CvPoint2D32f *imagePoints,
                            float focalLength, CvTermCriteria criteria,
                            float* rotation, float* translation )
{
    int i, j, k;
    int count = 0, converged = 0;
    float inorm, jnorm, invInorm, invJnorm, invScale, scale = 0, inv_Z = 0;
    float diff = (float)criteria.epsilon;
    float inv_focalLength = 1 / focalLength;

    /* Check bad arguments */
    if( imagePoints == NULL )
        return CV_NULLPTR_ERR;
    if( pObject == NULL )
        return CV_NULLPTR_ERR;
    if( focalLength <= 0 )
        return CV_BADFACTOR_ERR;
    if( !rotation )
        return CV_NULLPTR_ERR;
    if( !translation )
        return CV_NULLPTR_ERR;
    if( (criteria.type == 0) || (criteria.type > (CV_TERMCRIT_ITER | CV_TERMCRIT_EPS)))
        return CV_BADFLAG_ERR;
    if( (criteria.type & CV_TERMCRIT_EPS) && criteria.epsilon < 0 )
        return CV_BADFACTOR_ERR;
    if( (criteria.type & CV_TERMCRIT_ITER) && criteria.max_iter <= 0 )
        return CV_BADFACTOR_ERR;

    /* init variables */
    int N = pObject->N;
    float *objectVectors = pObject->obj_vecs;
    float *invMatrix = pObject->inv_matr;
    float *imgVectors = pObject->img_vecs;

    while( !converged )
    {
        if( count == 0 )
        {
            /* subtract out origin to get image vectors */
            for( i = 0; i < N; i++ )
            {
                imgVectors[i] = imagePoints[i + 1].x - imagePoints[0].x;
                imgVectors[N + i] = imagePoints[i + 1].y - imagePoints[0].y;
            }
        }
        else
        {
            diff = 0;
            /* Compute new SOP (scaled orthograthic projection) image from pose */
            for( i = 0; i < N; i++ )
            {
                /* objectVector * k */
                float old;
                float tmp = objectVectors[i] * rotation[6] /*[2][0]*/ +
                    objectVectors[N + i] * rotation[7]     /*[2][1]*/ +
                    objectVectors[2 * N + i] * rotation[8] /*[2][2]*/;

                tmp *= inv_Z;
                tmp += 1;

                old = imgVectors[i];
                imgVectors[i] = imagePoints[i + 1].x * tmp - imagePoints[0].x;

                diff = MAX( diff, (float) fabs( imgVectors[i] - old ));

                old = imgVectors[N + i];
                imgVectors[N + i] = imagePoints[i + 1].y * tmp - imagePoints[0].y;

                diff = MAX( diff, (float) fabs( imgVectors[N + i] - old ));
            }
        }

        /* calculate I and J vectors */
        for( i = 0; i < 2; i++ )
        {
            for( j = 0; j < 3; j++ )
            {
                rotation[3*i+j] /*[i][j]*/ = 0;
                for( k = 0; k < N; k++ )
                {
                    rotation[3*i+j] /*[i][j]*/ += invMatrix[j * N + k] * imgVectors[i * N + k];
                }
            }
        }

        inorm = rotation[0] /*[0][0]*/ * rotation[0] /*[0][0]*/ +
                rotation[1] /*[0][1]*/ * rotation[1] /*[0][1]*/ +
                rotation[2] /*[0][2]*/ * rotation[2] /*[0][2]*/;

        jnorm = rotation[3] /*[1][0]*/ * rotation[3] /*[1][0]*/ +
                rotation[4] /*[1][1]*/ * rotation[4] /*[1][1]*/ +
                rotation[5] /*[1][2]*/ * rotation[5] /*[1][2]*/;

        invInorm = cvInvSqrt( inorm );
        invJnorm = cvInvSqrt( jnorm );

        inorm *= invInorm;
        jnorm *= invJnorm;

        rotation[0] /*[0][0]*/ *= invInorm;
        rotation[1] /*[0][1]*/ *= invInorm;
        rotation[2] /*[0][2]*/ *= invInorm;

        rotation[3] /*[1][0]*/ *= invJnorm;
        rotation[4] /*[1][1]*/ *= invJnorm;
        rotation[5] /*[1][2]*/ *= invJnorm;

        /* row2 = row0 x row1 (cross product) */
        rotation[6] /*->m[2][0]*/ = rotation[1] /*->m[0][1]*/ * rotation[5] /*->m[1][2]*/ -
                                    rotation[2] /*->m[0][2]*/ * rotation[4] /*->m[1][1]*/;

        rotation[7] /*->m[2][1]*/ = rotation[2] /*->m[0][2]*/ * rotation[3] /*->m[1][0]*/ -
                                    rotation[0] /*->m[0][0]*/ * rotation[5] /*->m[1][2]*/;

        rotation[8] /*->m[2][2]*/ = rotation[0] /*->m[0][0]*/ * rotation[4] /*->m[1][1]*/ -
                                    rotation[1] /*->m[0][1]*/ * rotation[3] /*->m[1][0]*/;

        scale = (inorm + jnorm) / 2.0f;
        inv_Z = scale * inv_focalLength;

        count++;
        converged = ((criteria.type & CV_TERMCRIT_EPS) && (diff < criteria.epsilon));
        converged |= ((criteria.type & CV_TERMCRIT_ITER) && (count == criteria.max_iter));
    }
    invScale = 1 / scale;
    translation[0] = imagePoints[0].x * invScale;
    translation[1] = imagePoints[0].y * invScale;
    translation[2] = 1 / inv_Z;

    return CV_NO_ERR;
}


static  CvStatus  icvReleasePOSITObject( CvPOSITObject ** ppObject )
{
    cvFree( ppObject );
    return CV_NO_ERR;
}

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name:       icvPseudoInverse3D
//    Purpose:    Pseudoinverse N x 3 matrix     N >= 3
//    Context:
//    Parameters:
//                a - input matrix
//                b - pseudoinversed a
//                n - number of rows in a
//                method - if 0, then b = inv(transpose(a)*a) * transpose(a)
//                         if 1, then SVD used.
//    Returns:
//    Notes:      Both matrix are stored by n-dimensional vectors.
//                Now only method == 0 supported.
//F*/
void
icvPseudoInverse3D( float *a, float *b, int n, int method )
{
    int k;

    if( method == 0 )
    {
        float ata00 = 0;
        float ata11 = 0;
        float ata22 = 0;
        float ata01 = 0;
        float ata02 = 0;
        float ata12 = 0;
        float det = 0;

        /* compute matrix ata = transpose(a) * a  */
        for( k = 0; k < n; k++ )
        {
            float a0 = a[k];
            float a1 = a[n + k];
            float a2 = a[2 * n + k];

            ata00 += a0 * a0;
            ata11 += a1 * a1;
            ata22 += a2 * a2;

            ata01 += a0 * a1;
            ata02 += a0 * a2;
            ata12 += a1 * a2;
        }
        /* inverse matrix ata */
        {
            float inv_det;
            float p00 = ata11 * ata22 - ata12 * ata12;
            float p01 = -(ata01 * ata22 - ata12 * ata02);
            float p02 = ata12 * ata01 - ata11 * ata02;

            float p11 = ata00 * ata22 - ata02 * ata02;
            float p12 = -(ata00 * ata12 - ata01 * ata02);
            float p22 = ata00 * ata11 - ata01 * ata01;

            det += ata00 * p00;
            det += ata01 * p01;
            det += ata02 * p02;

            inv_det = 1 / det;

            /* compute resultant matrix */
            for( k = 0; k < n; k++ )
            {
                float a0 = a[k];
                float a1 = a[n + k];
                float a2 = a[2 * n + k];

                b[k] = (p00 * a0 + p01 * a1 + p02 * a2) * inv_det;
                b[n + k] = (p01 * a0 + p11 * a1 + p12 * a2) * inv_det;
                b[2 * n + k] = (p02 * a0 + p12 * a1 + p22 * a2) * inv_det;
            }
        }
    }

    /*if ( method == 1 )
       {
       }
     */

    return;
}

CV_IMPL CvPOSITObject *
cvCreatePOSITObject( CvPoint3D32f * points, int numPoints )
{
    CvPOSITObject *pObject = 0;
    IPPI_CALL( icvCreatePOSITObject( points, numPoints, &pObject ));
    return pObject;
}


CV_IMPL void
cvPOSIT( CvPOSITObject * pObject, CvPoint2D32f * imagePoints,
         double focalLength, CvTermCriteria criteria,
         float* rotation, float* translation )
{
    IPPI_CALL( icvPOSIT( pObject, imagePoints,(float) focalLength, criteria,
                         rotation, translation ));
}

CV_IMPL void
cvReleasePOSITObject( CvPOSITObject ** ppObject )
{
    IPPI_CALL( icvReleasePOSITObject( ppObject ));
}

/* End of file. */
