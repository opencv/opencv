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
#include <float.h>
#include <limits.h>

/* Valery Mosyagin */

#undef quad

#define EPS64D 1e-9

int cvComputeEssentialMatrix(  CvMatr32f rotMatr,
                                    CvMatr32f transVect,
                                    CvMatr32f essMatr);

int cvConvertEssential2Fundamental( CvMatr32f essMatr,
                                         CvMatr32f fundMatr,
                                         CvMatr32f cameraMatr1,
                                         CvMatr32f cameraMatr2);

int cvComputeEpipolesFromFundMatrix(CvMatr32f fundMatr,
                                         CvPoint3D32f* epipole1,
                                         CvPoint3D32f* epipole2);

void icvTestPoint( CvPoint2D64d testPoint,
                CvVect64d line1,CvVect64d line2,
                CvPoint2D64d basePoint,
                int* result);



int icvGetSymPoint3D(  CvPoint3D64d pointCorner,
                            CvPoint3D64d point1,
                            CvPoint3D64d point2,
                            CvPoint3D64d *pointSym2)
{
    double len1,len2;
    double alpha;
    icvGetPieceLength3D(pointCorner,point1,&len1);
    if( len1 < EPS64D )
    {
        return CV_BADARG_ERR;
    }
    icvGetPieceLength3D(pointCorner,point2,&len2);
    alpha = len2 / len1;

    pointSym2->x = pointCorner.x + alpha*(point1.x - pointCorner.x);
    pointSym2->y = pointCorner.y + alpha*(point1.y - pointCorner.y);
    pointSym2->z = pointCorner.z + alpha*(point1.z - pointCorner.z);
    return CV_NO_ERR;
}

/*  author Valery Mosyagin */

/* Compute 3D point for scanline and alpha betta */
int icvCompute3DPoint( double alpha,double betta,
                            CvStereoLineCoeff* coeffs,
                            CvPoint3D64d* point)
{

    double partX;
    double partY;
    double partZ;
    double partAll;
    double invPartAll;

    double alphabetta = alpha*betta;

    partAll = alpha - betta;
    if( fabs(partAll) > 0.00001  ) /* alpha must be > betta */
    {

        partX   = coeffs->Xcoef        + coeffs->XcoefA *alpha +
                  coeffs->XcoefB*betta + coeffs->XcoefAB*alphabetta;

        partY   = coeffs->Ycoef        + coeffs->YcoefA *alpha +
                  coeffs->YcoefB*betta + coeffs->YcoefAB*alphabetta;

        partZ   = coeffs->Zcoef        + coeffs->ZcoefA *alpha +
                  coeffs->ZcoefB*betta + coeffs->ZcoefAB*alphabetta;

        invPartAll = 1.0 / partAll;

        point->x = partX * invPartAll;
        point->y = partY * invPartAll;
        point->z = partZ * invPartAll;
        return CV_NO_ERR;
    }
    else
    {
        return CV_BADFACTOR_ERR;
    }
}

/*--------------------------------------------------------------------------------------*/

/* Compute rotate matrix and trans vector for change system */
int icvCreateConvertMatrVect( CvMatr64d     rotMatr1,
                                CvMatr64d     transVect1,
                                CvMatr64d     rotMatr2,
                                CvMatr64d     transVect2,
                                CvMatr64d     convRotMatr,
                                CvMatr64d     convTransVect)
{
    double invRotMatr2[9];
    double tmpVect[3];


    icvInvertMatrix_64d(rotMatr2,3,invRotMatr2);
    /* Test for error */

    icvMulMatrix_64d(   rotMatr1,
                        3,3,
                        invRotMatr2,
                        3,3,
                        convRotMatr);

    icvMulMatrix_64d(   convRotMatr,
                        3,3,
                        transVect2,
                        1,3,
                        tmpVect);

    icvSubVector_64d(transVect1,tmpVect,convTransVect,3);


    return CV_NO_ERR;
}

/*--------------------------------------------------------------------------------------*/

/* Compute point coordinates in other system */
int icvConvertPointSystem(CvPoint3D64d  M2,
                            CvPoint3D64d* M1,
                            CvMatr64d     rotMatr,
                            CvMatr64d     transVect
                            )
{
    double tmpVect[3];

    icvMulMatrix_64d(   rotMatr,
                        3,3,
                        (double*)&M2,
                        1,3,
                        tmpVect);

    icvAddVector_64d(tmpVect,transVect,(double*)M1,3);

    return CV_NO_ERR;
}
/*--------------------------------------------------------------------------------------*/
static int icvComputeCoeffForStereoV3( double quad1[4][2],
                                double quad2[4][2],
                                int    numScanlines,
                                CvMatr64d    camMatr1,
                                CvMatr64d    rotMatr1,
                                CvMatr64d    transVect1,
                                CvMatr64d    camMatr2,
                                CvMatr64d    rotMatr2,
                                CvMatr64d    transVect2,
                                CvStereoLineCoeff*    startCoeffs,
                                int* needSwapCamera)
{
    /* For each pair */
    /* In this function we must define position of cameras */

    CvPoint2D64d point1;
    CvPoint2D64d point2;
    CvPoint2D64d point3;
    CvPoint2D64d point4;

    int currLine;
    *needSwapCamera = 0;
    for( currLine = 0; currLine < numScanlines; currLine++ )
    {
        /* Compute points */
        double alpha = ((double)currLine)/((double)(numScanlines)); /* maybe - 1 */

        point1.x = (1.0 - alpha) * quad1[0][0] + alpha * quad1[3][0];
        point1.y = (1.0 - alpha) * quad1[0][1] + alpha * quad1[3][1];

        point2.x = (1.0 - alpha) * quad1[1][0] + alpha * quad1[2][0];
        point2.y = (1.0 - alpha) * quad1[1][1] + alpha * quad1[2][1];

        point3.x = (1.0 - alpha) * quad2[0][0] + alpha * quad2[3][0];
        point3.y = (1.0 - alpha) * quad2[0][1] + alpha * quad2[3][1];

        point4.x = (1.0 - alpha) * quad2[1][0] + alpha * quad2[2][0];
        point4.y = (1.0 - alpha) * quad2[1][1] + alpha * quad2[2][1];

        /* We can compute coeffs for this line */
        icvComCoeffForLine(    point1,
                            point2,
                            point3,
                            point4,
                            camMatr1,
                            rotMatr1,
                            transVect1,
                            camMatr2,
                            rotMatr2,
                            transVect2,
                            &startCoeffs[currLine],
                            needSwapCamera);
    }
    return CV_NO_ERR;
}
/*--------------------------------------------------------------------------------------*/
static int icvComputeCoeffForStereoNew(   double quad1[4][2],
                                        double quad2[4][2],
                                        int    numScanlines,
                                        CvMatr32f    camMatr1,
                                        CvMatr32f    rotMatr1,
                                        CvMatr32f    transVect1,
                                        CvMatr32f    camMatr2,
                                        CvStereoLineCoeff*    startCoeffs,
                                        int* needSwapCamera)
{
    /* Convert data */

    double camMatr1_64d[9];
    double camMatr2_64d[9];

    double rotMatr1_64d[9];
    double transVect1_64d[3];

    double rotMatr2_64d[9];
    double transVect2_64d[3];

    icvCvt_32f_64d(camMatr1,camMatr1_64d,9);
    icvCvt_32f_64d(camMatr2,camMatr2_64d,9);

    icvCvt_32f_64d(rotMatr1,rotMatr1_64d,9);
    icvCvt_32f_64d(transVect1,transVect1_64d,3);

    rotMatr2_64d[0] = 1;
    rotMatr2_64d[1] = 0;
    rotMatr2_64d[2] = 0;
    rotMatr2_64d[3] = 0;
    rotMatr2_64d[4] = 1;
    rotMatr2_64d[5] = 0;
    rotMatr2_64d[6] = 0;
    rotMatr2_64d[7] = 0;
    rotMatr2_64d[8] = 1;

    transVect2_64d[0] = 0;
    transVect2_64d[1] = 0;
    transVect2_64d[2] = 0;

    int status = icvComputeCoeffForStereoV3( quad1,
                                                quad2,
                                                numScanlines,
                                                camMatr1_64d,
                                                rotMatr1_64d,
                                                transVect1_64d,
                                                camMatr2_64d,
                                                rotMatr2_64d,
                                                transVect2_64d,
                                                startCoeffs,
                                                needSwapCamera);


    return status;

}
/*--------------------------------------------------------------------------------------*/
int icvComputeCoeffForStereo(  CvStereoCamera* stereoCamera)
{
    double quad1[4][2];
    double quad2[4][2];

    int i;
    for( i = 0; i < 4; i++ )
    {
        quad1[i][0] = stereoCamera->quad[0][i].x;
        quad1[i][1] = stereoCamera->quad[0][i].y;

        quad2[i][0] = stereoCamera->quad[1][i].x;
        quad2[i][1] = stereoCamera->quad[1][i].y;
    }

    icvComputeCoeffForStereoNew(        quad1,
                                        quad2,
                                        stereoCamera->warpSize.height,
                                        stereoCamera->camera[0]->matrix,
                                        stereoCamera->rotMatrix,
                                        stereoCamera->transVector,
                                        stereoCamera->camera[1]->matrix,
                                        stereoCamera->lineCoeffs,
                                        &(stereoCamera->needSwapCameras));
    return CV_OK;
}


/*--------------------------------------------------------------------------------------*/
int icvComCoeffForLine(   CvPoint2D64d point1,
                            CvPoint2D64d point2,
                            CvPoint2D64d point3,
                            CvPoint2D64d point4,
                            CvMatr64d    camMatr1,
                            CvMatr64d    rotMatr1,
                            CvMatr64d    transVect1,
                            CvMatr64d    camMatr2,
                            CvMatr64d    rotMatr2,
                            CvMatr64d    transVect2,
                            CvStereoLineCoeff* coeffs,
                            int* needSwapCamera)
{
    /* Get direction for all points */
    /* Direction for camera 1 */

    CvPoint3D64f direct1;
    CvPoint3D64f direct2;
    CvPoint3D64f camPoint1;

    CvPoint3D64f directS3;
    CvPoint3D64f directS4;
    CvPoint3D64f direct3;
    CvPoint3D64f direct4;
    CvPoint3D64f camPoint2;

    icvGetDirectionForPoint(   point1,
                            camMatr1,
                            &direct1);

    icvGetDirectionForPoint(   point2,
                            camMatr1,
                            &direct2);

    /* Direction for camera 2 */

    icvGetDirectionForPoint(   point3,
                            camMatr2,
                            &directS3);

    icvGetDirectionForPoint(   point4,
                            camMatr2,
                            &directS4);

    /* Create convertion for camera 2: two direction and camera point */

    double convRotMatr[9];
    double convTransVect[3];

    icvCreateConvertMatrVect(  rotMatr1,
                            transVect1,
                            rotMatr2,
                            transVect2,
                            convRotMatr,
                            convTransVect);

    CvPoint3D64f zeroVect;
    zeroVect.x = zeroVect.y = zeroVect.z = 0.0;
    camPoint1.x = camPoint1.y = camPoint1.z = 0.0;

    icvConvertPointSystem(directS3,&direct3,convRotMatr,convTransVect);
    icvConvertPointSystem(directS4,&direct4,convRotMatr,convTransVect);
    icvConvertPointSystem(zeroVect,&camPoint2,convRotMatr,convTransVect);

    CvPoint3D64f pointB;

    int postype = 0;

    /* Changed order */
    /* Compute point B: xB,yB,zB */
    icvGetCrossLines(camPoint1,direct2,
                  camPoint2,direct3,
                  &pointB);

    if( pointB.z < 0 )/* If negative use other lines for cross */
    {
        postype = 1;
        icvGetCrossLines(camPoint1,direct1,
                      camPoint2,direct4,
                      &pointB);
    }

    CvPoint3D64d pointNewA;
    CvPoint3D64d pointNewC;

    pointNewA.x = pointNewA.y = pointNewA.z = 0;
    pointNewC.x = pointNewC.y = pointNewC.z = 0;

    if( postype == 0 )
    {
        icvGetSymPoint3D(   camPoint1,
                            direct1,
                            pointB,
                            &pointNewA);

        icvGetSymPoint3D(   camPoint2,
                            direct4,
                            pointB,
                            &pointNewC);
    }
    else
    {/* In this case we must change cameras */
        *needSwapCamera = 1;
        icvGetSymPoint3D(   camPoint2,
                            direct3,
                            pointB,
                            &pointNewA);

        icvGetSymPoint3D(   camPoint1,
                            direct2,
                            pointB,
                            &pointNewC);
    }


    double gamma;

    double xA,yA,zA;
    double xB,yB,zB;
    double xC,yC,zC;

    xA = pointNewA.x;
    yA = pointNewA.y;
    zA = pointNewA.z;

    xB = pointB.x;
    yB = pointB.y;
    zB = pointB.z;

    xC = pointNewC.x;
    yC = pointNewC.y;
    zC = pointNewC.z;

    double len1,len2;
    len1 = sqrt( (xA-xB)*(xA-xB) + (yA-yB)*(yA-yB) + (zA-zB)*(zA-zB) );
    len2 = sqrt( (xB-xC)*(xB-xC) + (yB-yC)*(yB-yC) + (zB-zC)*(zB-zC) );
    gamma = len2 / len1;

    icvComputeStereoLineCoeffs( pointNewA,
                                pointB,
                                camPoint1,
                                gamma,
                                coeffs);

    return CV_NO_ERR;
}


/*--------------------------------------------------------------------------------------*/

int icvGetDirectionForPoint(  CvPoint2D64d point,
                                CvMatr64d camMatr,
                                CvPoint3D64d* direct)
{
    /*  */
    double invMatr[9];

    /* Invert matrix */

    icvInvertMatrix_64d(camMatr,3,invMatr);
    /* TEST FOR ERRORS */

    double vect[3];
    vect[0] = point.x;
    vect[1] = point.y;
    vect[2] = 1;

    /* Mul matr */
    icvMulMatrix_64d(   invMatr,
                        3,3,
                        vect,
                        1,3,
                        (double*)direct);

    return CV_NO_ERR;
}

/*--------------------------------------------------------------------------------------*/

int icvGetCrossLines(CvPoint3D64d point11,CvPoint3D64d point12,
                       CvPoint3D64d point21,CvPoint3D64d point22,
                       CvPoint3D64d* midPoint)
{
    double xM,yM,zM;
    double xN,yN,zN;

    double xA,yA,zA;
    double xB,yB,zB;

    double xC,yC,zC;
    double xD,yD,zD;

    xA = point11.x;
    yA = point11.y;
    zA = point11.z;

    xB = point12.x;
    yB = point12.y;
    zB = point12.z;

    xC = point21.x;
    yC = point21.y;
    zC = point21.z;

    xD = point22.x;
    yD = point22.y;
    zD = point22.z;

    double a11,a12,a21,a22;
    double b1,b2;

    a11 =  (xB-xA)*(xB-xA)+(yB-yA)*(yB-yA)+(zB-zA)*(zB-zA);
    a12 = -(xD-xC)*(xB-xA)-(yD-yC)*(yB-yA)-(zD-zC)*(zB-zA);
    a21 =  (xB-xA)*(xD-xC)+(yB-yA)*(yD-yC)+(zB-zA)*(zD-zC);
    a22 = -(xD-xC)*(xD-xC)-(yD-yC)*(yD-yC)-(zD-zC)*(zD-zC);
    b1  = -( (xA-xC)*(xB-xA)+(yA-yC)*(yB-yA)+(zA-zC)*(zB-zA) );
    b2  = -( (xA-xC)*(xD-xC)+(yA-yC)*(yD-yC)+(zA-zC)*(zD-zC) );

    double delta;
    double deltaA,deltaB;
    double alpha,betta;

    delta  = a11*a22-a12*a21;

    if( fabs(delta) < EPS64D )
    {
        /*return ERROR;*/
    }

    deltaA = b1*a22-b2*a12;
    deltaB = a11*b2-b1*a21;

    alpha = deltaA / delta;
    betta = deltaB / delta;

    xM = xA+alpha*(xB-xA);
    yM = yA+alpha*(yB-yA);
    zM = zA+alpha*(zB-zA);

    xN = xC+betta*(xD-xC);
    yN = yC+betta*(yD-yC);
    zN = zC+betta*(zD-zC);

    /* Compute middle point */
    midPoint->x = (xM + xN) * 0.5;
    midPoint->y = (yM + yN) * 0.5;
    midPoint->z = (zM + zN) * 0.5;

    return CV_NO_ERR;
}

/*--------------------------------------------------------------------------------------*/

int icvComputeStereoLineCoeffs(   CvPoint3D64d pointA,
                                    CvPoint3D64d pointB,
                                    CvPoint3D64d pointCam1,
                                    double gamma,
                                    CvStereoLineCoeff*    coeffs)
{
    double x1,y1,z1;

    x1 = pointCam1.x;
    y1 = pointCam1.y;
    z1 = pointCam1.z;

    double xA,yA,zA;
    double xB,yB,zB;

    xA = pointA.x;
    yA = pointA.y;
    zA = pointA.z;

    xB = pointB.x;
    yB = pointB.y;
    zB = pointB.z;

    if( gamma > 0 )
    {
        coeffs->Xcoef   = -x1 + xA;
        coeffs->XcoefA  =  xB + x1 - xA;
        coeffs->XcoefB  = -xA - gamma * x1 + gamma * xA;
        coeffs->XcoefAB = -xB + xA + gamma * xB - gamma * xA;

        coeffs->Ycoef   = -y1 + yA;
        coeffs->YcoefA  =  yB + y1 - yA;
        coeffs->YcoefB  = -yA - gamma * y1 + gamma * yA;
        coeffs->YcoefAB = -yB + yA + gamma * yB - gamma * yA;

        coeffs->Zcoef   = -z1 + zA;
        coeffs->ZcoefA  =  zB + z1 - zA;
        coeffs->ZcoefB  = -zA - gamma * z1 + gamma * zA;
        coeffs->ZcoefAB = -zB + zA + gamma * zB - gamma * zA;
    }
    else
    {
        gamma = - gamma;
        coeffs->Xcoef   = -( -x1 + xA);
        coeffs->XcoefB  = -(  xB + x1 - xA);
        coeffs->XcoefA  = -( -xA - gamma * x1 + gamma * xA);
        coeffs->XcoefAB = -( -xB + xA + gamma * xB - gamma * xA);

        coeffs->Ycoef   = -( -y1 + yA);
        coeffs->YcoefB  = -(  yB + y1 - yA);
        coeffs->YcoefA  = -( -yA - gamma * y1 + gamma * yA);
        coeffs->YcoefAB = -( -yB + yA + gamma * yB - gamma * yA);

        coeffs->Zcoef   = -( -z1 + zA);
        coeffs->ZcoefB  = -(  zB + z1 - zA);
        coeffs->ZcoefA  = -( -zA - gamma * z1 + gamma * zA);
        coeffs->ZcoefAB = -( -zB + zA + gamma * zB - gamma * zA);
    }



    return CV_NO_ERR;
}
/*--------------------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------------------*/

/* This function get minimum angle started at point which contains rect */
int icvGetAngleLine( CvPoint2D64d startPoint, CvSize imageSize,CvPoint2D64d *point1,CvPoint2D64d *point2)
{
    /* Get crosslines with image corners */

    /* Find four lines */

    CvPoint2D64d pa,pb,pc,pd;

    pa.x = 0;
    pa.y = 0;

    pb.x = imageSize.width-1;
    pb.y = 0;

    pd.x = imageSize.width-1;
    pd.y = imageSize.height-1;

    pc.x = 0;
    pc.y = imageSize.height-1;

    /* We can compute points for angle */
    /* Test for place section */

    if( startPoint.x < 0 )
    {/* 1,4,7 */
        if( startPoint.y < 0)
        {/* 1 */
            *point1 = pb;
            *point2 = pc;
        }
        else if( startPoint.y > imageSize.height-1 )
        {/* 7 */
            *point1 = pa;
            *point2 = pd;
        }
        else
        {/* 4 */
            *point1 = pa;
            *point2 = pc;
        }
    }
    else if ( startPoint.x > imageSize.width-1 )
    {/* 3,6,9 */
        if( startPoint.y < 0 )
        {/* 3 */
            *point1 = pa;
            *point2 = pd;
        }
        else if ( startPoint.y > imageSize.height-1 )
        {/* 9 */
            *point1 = pb;
            *point2 = pc;
        }
        else
        {/* 6 */
            *point1 = pb;
            *point2 = pd;
        }
    }
    else
    {/* 2,5,8 */
        if( startPoint.y < 0 )
        {/* 2 */
            if( startPoint.x < imageSize.width/2 )
            {
                *point1 = pb;
                *point2 = pa;
            }
            else
            {
                *point1 = pa;
                *point2 = pb;
            }
        }
        else if( startPoint.y > imageSize.height-1 )
        {/* 8 */
            if( startPoint.x < imageSize.width/2 )
            {
                *point1 = pc;
                *point2 = pd;
            }
            else
            {
                *point1 = pd;
                *point2 = pc;
            }
        }
        else
        {/* 5 - point in the image */
            return 2;
        }
    }
    return 0;
}/* GetAngleLine */

/*---------------------------------------------------------------------------------------*/

void icvGetCoefForPiece(   CvPoint2D64d p_start,CvPoint2D64d p_end,
                        double *a,double *b,double *c,
                        int* result)
{
    double det;
    double detA,detB,detC;

    det = p_start.x*p_end.y+p_end.x+p_start.y-p_end.y-p_start.y*p_end.x-p_start.x;
    if( fabs(det) < EPS64D)/* Error */
    {
        *result = 0;
        return;
    }

    detA = p_start.y - p_end.y;
    detB = p_end.x - p_start.x;
    detC = p_start.x*p_end.y - p_end.x*p_start.y;

    double invDet = 1.0 / det;
    *a = detA * invDet;
    *b = detB * invDet;
    *c = detC * invDet;

    *result = 1;
    return;
}

/*---------------------------------------------------------------------------------------*/

/* Get common area of rectifying */
static void icvGetCommonArea( CvSize imageSize,
                    CvPoint3D64d epipole1,CvPoint3D64d epipole2,
                    CvMatr64d fundMatr,
                    CvVect64d coeff11,CvVect64d coeff12,
                    CvVect64d coeff21,CvVect64d coeff22,
                    int* result)
{
    int res = 0;
    CvPoint2D64d point11;
    CvPoint2D64d point12;
    CvPoint2D64d point21;
    CvPoint2D64d point22;

    double corr11[3];
    double corr12[3];
    double corr21[3];
    double corr22[3];

    double pointW11[3];
    double pointW12[3];
    double pointW21[3];
    double pointW22[3];

    double transFundMatr[3*3];
    /* Compute transpose of fundamental matrix */
    icvTransposeMatrix_64d( fundMatr, 3, 3, transFundMatr );

    CvPoint2D64d epipole1_2d;
    CvPoint2D64d epipole2_2d;

    if( fabs(epipole1.z) < 1e-8 )
    {/* epipole1 in infinity */
        *result = 0;
        return;
    }
    epipole1_2d.x = epipole1.x / epipole1.z;
    epipole1_2d.y = epipole1.y / epipole1.z;

    if( fabs(epipole2.z) < 1e-8 )
    {/* epipole2 in infinity */
        *result = 0;
        return;
    }
    epipole2_2d.x = epipole2.x / epipole2.z;
    epipole2_2d.y = epipole2.y / epipole2.z;

    int stat = icvGetAngleLine( epipole1_2d, imageSize,&point11,&point12);
    if( stat == 2 )
    {
        /* No angle */
        *result = 0;
        return;
    }

    stat = icvGetAngleLine( epipole2_2d, imageSize,&point21,&point22);
    if( stat == 2 )
    {
        /* No angle */
        *result = 0;
        return;
    }

    /* ============= Computation for line 1 ================ */
    /* Find correspondence line for angle points11 */
    /* corr21 = Fund'*p1 */

    pointW11[0] = point11.x;
    pointW11[1] = point11.y;
    pointW11[2] = 1.0;

    icvTransformVector_64d( transFundMatr, /* !!! Modified from not transposed */
                            pointW11,
                            corr21,
                            3,3);

    /* Find crossing of line with image 2 */
    CvPoint2D64d start;
    CvPoint2D64d end;
    icvGetCrossRectDirect( imageSize,
                        corr21[0],corr21[1],corr21[2],
                        &start,&end,
                        &res);

    if( res == 0 )
    {/* We have not cross */
        /* We must define new angle */

        pointW21[0] = point21.x;
        pointW21[1] = point21.y;
        pointW21[2] = 1.0;

        /* Find correspondence line for this angle points */
        /* We know point and try to get corr line */
        /* For point21 */
        /* corr11 = Fund * p21 */

        icvTransformVector_64d( fundMatr, /* !!! Modified */
                                pointW21,
                                corr11,
                                3,3);

        /* We have cross. And it's result cross for up line. Set result coefs */

        /* Set coefs for line 1 image 1 */
        coeff11[0] = corr11[0];
        coeff11[1] = corr11[1];
        coeff11[2] = corr11[2];

        /* Set coefs for line 1 image 2 */
        icvGetCoefForPiece(    epipole2_2d,point21,
                            &coeff21[0],&coeff21[1],&coeff21[2],
                            &res);
        if( res == 0 )
        {
            *result = 0;
            return;/* Error */
        }
    }
    else
    {/* Line 1 cross image 2 */
        /* Set coefs for line 1 image 1 */
        icvGetCoefForPiece(    epipole1_2d,point11,
                            &coeff11[0],&coeff11[1],&coeff11[2],
                            &res);
        if( res == 0 )
        {
            *result = 0;
            return;/* Error */
        }

        /* Set coefs for line 1 image 2 */
        coeff21[0] = corr21[0];
        coeff21[1] = corr21[1];
        coeff21[2] = corr21[2];

    }

    /* ============= Computation for line 2 ================ */
    /* Find correspondence line for angle points11 */
    /* corr22 = Fund*p2 */

    pointW12[0] = point12.x;
    pointW12[1] = point12.y;
    pointW12[2] = 1.0;

    icvTransformVector_64d( transFundMatr,
                            pointW12,
                            corr22,
                            3,3);

    /* Find crossing of line with image 2 */
    icvGetCrossRectDirect( imageSize,
                        corr22[0],corr22[1],corr22[2],
                        &start,&end,
                        &res);

    if( res == 0 )
    {/* We have not cross */
        /* We must define new angle */

        pointW22[0] = point22.x;
        pointW22[1] = point22.y;
        pointW22[2] = 1.0;

        /* Find correspondence line for this angle points */
        /* We know point and try to get corr line */
        /* For point21 */
        /* corr2 = Fund' * p1 */

        icvTransformVector_64d( fundMatr,
                                pointW22,
                                corr12,
                                3,3);


        /* We have cross. And it's result cross for down line. Set result coefs */

        /* Set coefs for line 2 image 1 */
        coeff12[0] = corr12[0];
        coeff12[1] = corr12[1];
        coeff12[2] = corr12[2];

        /* Set coefs for line 1 image 2 */
        icvGetCoefForPiece(    epipole2_2d,point22,
                            &coeff22[0],&coeff22[1],&coeff22[2],
                            &res);
        if( res == 0 )
        {
            *result = 0;
            return;/* Error */
        }
    }
    else
    {/* Line 2 cross image 2 */
        /* Set coefs for line 2 image 1 */
        icvGetCoefForPiece(    epipole1_2d,point12,
                            &coeff12[0],&coeff12[1],&coeff12[2],
                            &res);
        if( res == 0 )
        {
            *result = 0;
            return;/* Error */
        }

        /* Set coefs for line 1 image 2 */
        coeff22[0] = corr22[0];
        coeff22[1] = corr22[1];
        coeff22[2] = corr22[2];

    }

    /* Now we know common area */

    return;

}/* GetCommonArea */

/*---------------------------------------------------------------------------------------*/

/* Get cross for direction1 and direction2 */
/*  Result = 1 - cross */
/*  Result = 2 - parallel and not equal */
/*  Result = 3 - parallel and equal */

void icvGetCrossDirectDirect(  CvVect64d direct1,CvVect64d direct2,
                            CvPoint2D64d *cross,int* result)
{
    double det  = direct1[0]*direct2[1] - direct2[0]*direct1[1];
    double detx = -direct1[2]*direct2[1] + direct1[1]*direct2[2];

    if( fabs(det) > EPS64D )
    {/* Have cross */
        cross->x = detx/det;
        cross->y = (-direct1[0]*direct2[2] + direct2[0]*direct1[2])/det;
        *result = 1;
    }
    else
    {/* may be parallel */
        if( fabs(detx) > EPS64D )
        {/* parallel and not equal */
            *result = 2;
        }
        else
        {/* equals */
            *result = 3;
        }
    }

    return;
}

/*---------------------------------------------------------------------------------------*/

/* Get cross for piece p1,p2 and direction a,b,c */
/*  Result = 0 - no cross */
/*  Result = 1 - cross */
/*  Result = 2 - parallel and not equal */
/*  Result = 3 - parallel and equal */

void icvGetCrossPieceDirect(   CvPoint2D64d p_start,CvPoint2D64d p_end,
                            double a,double b,double c,
                            CvPoint2D64d *cross,int* result)
{

    if( (a*p_start.x + b*p_start.y + c) * (a*p_end.x + b*p_end.y + c) <= 0 )
    {/* Have cross */
        double det;
        double detxc,detyc;

        det = a * (p_end.x - p_start.x) + b * (p_end.y - p_start.y);

        if( fabs(det) < EPS64D )
        {/* lines are parallel and may be equal or line is point */
            if(  fabs(a*p_start.x + b*p_start.y + c) < EPS64D )
            {/* line is point or not diff */
                *result = 3;
                return;
            }
            else
            {
                *result = 2;
            }
            return;
        }

        detxc = b*(p_end.y*p_start.x - p_start.y*p_end.x) + c*(p_start.x - p_end.x);
        detyc = a*(p_end.x*p_start.y - p_start.x*p_end.y) + c*(p_start.y - p_end.y);

        cross->x = detxc / det;
        cross->y = detyc / det;
        *result = 1;

    }
    else
    {
        *result = 0;
    }
    return;
}
/*--------------------------------------------------------------------------------------*/

void icvGetCrossPiecePiece( CvPoint2D64d p1_start,CvPoint2D64d p1_end,
                            CvPoint2D64d p2_start,CvPoint2D64d p2_end,
                            CvPoint2D64d* cross,
                            int* result)
{
    double ex1,ey1,ex2,ey2;
    double px1,py1,px2,py2;
    double del;
    double delA,delB,delX,delY;
    double alpha,betta;

    ex1 = p1_start.x;
    ey1 = p1_start.y;
    ex2 = p1_end.x;
    ey2 = p1_end.y;

    px1 = p2_start.x;
    py1 = p2_start.y;
    px2 = p2_end.x;
    py2 = p2_end.y;

    del = (py1-py2)*(ex1-ex2)-(px1-px2)*(ey1-ey2);
    if( fabs(del) <= EPS64D )
    {/* May be they are parallel !!! */
        *result = 0;
        return;
    }

    delA =  (ey1-ey2)*(ex1-px1) + (ex1-ex2)*(py1-ey1);
    delB =  (py1-py2)*(ex1-px1) + (px1-px2)*(py1-ey1);

    alpha = delA / del;
    betta = delB / del;

    if( alpha < 0 || alpha > 1.0 || betta < 0 || betta > 1.0)
    {
        *result = 0;
        return;
    }

    delX =  (px1-px2)*(ey1*(ex1-ex2)-ex1*(ey1-ey2))+
            (ex1-ex2)*(px1*(py1-py2)-py1*(px1-px2));

    delY =  (py1-py2)*(ey1*(ex1-ex2)-ex1*(ey1-ey2))+
            (ey1-ey2)*(px1*(py1-py2)-py1*(px1-px2));

    cross->x = delX / del;
    cross->y = delY / del;

    *result = 1;
    return;
}


/*---------------------------------------------------------------------------------------*/

void icvGetPieceLength(CvPoint2D64d point1,CvPoint2D64d point2,double* dist)
{
    double dx = point2.x - point1.x;
    double dy = point2.y - point1.y;
    *dist = sqrt( dx*dx + dy*dy );
    return;
}

/*---------------------------------------------------------------------------------------*/

void icvGetPieceLength3D(CvPoint3D64d point1,CvPoint3D64d point2,double* dist)
{
    double dx = point2.x - point1.x;
    double dy = point2.y - point1.y;
    double dz = point2.z - point1.z;
    *dist = sqrt( dx*dx + dy*dy + dz*dz );
    return;
}

/*---------------------------------------------------------------------------------------*/

/* Find line from epipole which cross image rect */
/* Find points of cross 0 or 1 or 2. Return number of points in cross */
void icvGetCrossRectDirect(    CvSize imageSize,
                            double a,double b,double c,
                            CvPoint2D64d *start,CvPoint2D64d *end,
                            int* result)
{
    CvPoint2D64d frameBeg;
    CvPoint2D64d frameEnd;
    CvPoint2D64d cross[4];
    int     haveCross[4];

    haveCross[0] = 0;
    haveCross[1] = 0;
    haveCross[2] = 0;
    haveCross[3] = 0;

    frameBeg.x = 0;
    frameBeg.y = 0;
    frameEnd.x = imageSize.width;
    frameEnd.y = 0;

    icvGetCrossPieceDirect(frameBeg,frameEnd,a,b,c,&cross[0],&haveCross[0]);

    frameBeg.x = imageSize.width;
    frameBeg.y = 0;
    frameEnd.x = imageSize.width;
    frameEnd.y = imageSize.height;
    icvGetCrossPieceDirect(frameBeg,frameEnd,a,b,c,&cross[1],&haveCross[1]);

    frameBeg.x = imageSize.width;
    frameBeg.y = imageSize.height;
    frameEnd.x = 0;
    frameEnd.y = imageSize.height;
    icvGetCrossPieceDirect(frameBeg,frameEnd,a,b,c,&cross[2],&haveCross[2]);

    frameBeg.x = 0;
    frameBeg.y = imageSize.height;
    frameEnd.x = 0;
    frameEnd.y = 0;
    icvGetCrossPieceDirect(frameBeg,frameEnd,a,b,c,&cross[3],&haveCross[3]);

    double maxDist;

    int maxI=0,maxJ=0;


    int i,j;

    maxDist = -1.0;

    double distance;

    for( i = 0; i < 3; i++ )
    {
        if( haveCross[i] == 1 )
        {
            for( j = i + 1; j < 4; j++ )
            {
                if( haveCross[j] == 1)
                {/* Compute dist */
                    icvGetPieceLength(cross[i],cross[j],&distance);
                    if( distance > maxDist )
                    {
                        maxI = i;
                        maxJ = j;
                        maxDist = distance;
                    }
                }
            }
        }
    }

    if( maxDist >= 0 )
    {/* We have cross */
        *start = cross[maxI];
        *result = 1;
        if( maxDist > 0 )
        {
            *end   = cross[maxJ];
            *result = 2;
        }
    }
    else
    {
        *result = 0;
    }

    return;
}/* GetCrossRectDirect */

/*---------------------------------------------------------------------------------------*/
void icvProjectPointToImage(   CvPoint3D64d point,
                            CvMatr64d camMatr,CvMatr64d rotMatr,CvVect64d transVect,
                            CvPoint2D64d* projPoint)
{

    double tmpVect1[3];
    double tmpVect2[3];

    icvMulMatrix_64d (  rotMatr,
                        3,3,
                        (double*)&point,
                        1,3,
                        tmpVect1);

    icvAddVector_64d ( tmpVect1, transVect,tmpVect2, 3);

    icvMulMatrix_64d (  camMatr,
                        3,3,
                        tmpVect2,
                        1,3,
                        tmpVect1);

    projPoint->x = tmpVect1[0] / tmpVect1[2];
    projPoint->y = tmpVect1[1] / tmpVect1[2];

    return;
}

/*---------------------------------------------------------------------------------------*/
/* Get quads for transform images */
void icvGetQuadsTransform(
                          CvSize        imageSize,
                        CvMatr64d     camMatr1,
                        CvMatr64d     rotMatr1,
                        CvVect64d     transVect1,
                        CvMatr64d     camMatr2,
                        CvMatr64d     rotMatr2,
                        CvVect64d     transVect2,
                        CvSize*       warpSize,
                        double quad1[4][2],
                        double quad2[4][2],
                        CvMatr64d     fundMatr,
                        CvPoint3D64d* epipole1,
                        CvPoint3D64d* epipole2
                        )
{
    /* First compute fundamental matrix and epipoles */
    int res;


    /* Compute epipoles and fundamental matrix using new functions */
    {
        double convRotMatr[9];
        double convTransVect[3];

        icvCreateConvertMatrVect( rotMatr1,
                                  transVect1,
                                  rotMatr2,
                                  transVect2,
                                  convRotMatr,
                                  convTransVect);
        float convRotMatr_32f[9];
        float convTransVect_32f[3];

        icvCvt_64d_32f(convRotMatr,convRotMatr_32f,9);
        icvCvt_64d_32f(convTransVect,convTransVect_32f,3);

        /* We know R and t */
        /* Compute essential matrix */
        float essMatr[9];
        float fundMatr_32f[9];

        float camMatr1_32f[9];
        float camMatr2_32f[9];

        icvCvt_64d_32f(camMatr1,camMatr1_32f,9);
        icvCvt_64d_32f(camMatr2,camMatr2_32f,9);

        cvComputeEssentialMatrix(   convRotMatr_32f,
                                    convTransVect_32f,
                                    essMatr);

        cvConvertEssential2Fundamental( essMatr,
                                        fundMatr_32f,
                                        camMatr1_32f,
                                        camMatr2_32f);

        CvPoint3D32f epipole1_32f;
        CvPoint3D32f epipole2_32f;

        cvComputeEpipolesFromFundMatrix( fundMatr_32f,
                                         &epipole1_32f,
                                         &epipole2_32f);
        /* copy to 64d epipoles */
        epipole1->x = epipole1_32f.x;
        epipole1->y = epipole1_32f.y;
        epipole1->z = epipole1_32f.z;

        epipole2->x = epipole2_32f.x;
        epipole2->y = epipole2_32f.y;
        epipole2->z = epipole2_32f.z;

        /* Convert fundamental matrix */
        icvCvt_32f_64d(fundMatr_32f,fundMatr,9);
    }

    double coeff11[3];
    double coeff12[3];
    double coeff21[3];
    double coeff22[3];

    icvGetCommonArea(   imageSize,
                        *epipole1,*epipole2,
                        fundMatr,
                        coeff11,coeff12,
                        coeff21,coeff22,
                        &res);

    CvPoint2D64d point11, point12,point21, point22;
    double width1,width2;
    double height1,height2;
    double tmpHeight1,tmpHeight2;

    CvPoint2D64d epipole1_2d;
    CvPoint2D64d epipole2_2d;

    /* ----- Image 1 ----- */
    if( fabs(epipole1->z) < 1e-8 )
    {
        return;
    }
    epipole1_2d.x = epipole1->x / epipole1->z;
    epipole1_2d.y = epipole1->y / epipole1->z;

    icvGetCutPiece( coeff11,coeff12,
                epipole1_2d,
                imageSize,
                &point11,&point12,
                &point21,&point22,
                &res);

    /* Compute distance */
    icvGetPieceLength(point11,point21,&width1);
    icvGetPieceLength(point11,point12,&tmpHeight1);
    icvGetPieceLength(point21,point22,&tmpHeight2);
    height1 = MAX(tmpHeight1,tmpHeight2);

    quad1[0][0] = point11.x;
    quad1[0][1] = point11.y;

    quad1[1][0] = point21.x;
    quad1[1][1] = point21.y;

    quad1[2][0] = point22.x;
    quad1[2][1] = point22.y;

    quad1[3][0] = point12.x;
    quad1[3][1] = point12.y;

    /* ----- Image 2 ----- */
    if( fabs(epipole2->z) < 1e-8 )
    {
        return;
    }
    epipole2_2d.x = epipole2->x / epipole2->z;
    epipole2_2d.y = epipole2->y / epipole2->z;

    icvGetCutPiece( coeff21,coeff22,
                epipole2_2d,
                imageSize,
                &point11,&point12,
                &point21,&point22,
                &res);

    /* Compute distance */
    icvGetPieceLength(point11,point21,&width2);
    icvGetPieceLength(point11,point12,&tmpHeight1);
    icvGetPieceLength(point21,point22,&tmpHeight2);
    height2 = MAX(tmpHeight1,tmpHeight2);

    quad2[0][0] = point11.x;
    quad2[0][1] = point11.y;

    quad2[1][0] = point21.x;
    quad2[1][1] = point21.y;

    quad2[2][0] = point22.x;
    quad2[2][1] = point22.y;

    quad2[3][0] = point12.x;
    quad2[3][1] = point12.y;


    /*=======================================================*/
    /* This is a new additional way to compute quads. */
    /* We must correct quads */
    {
        double convRotMatr[9];
        double convTransVect[3];

        double newQuad1[4][2];
        double newQuad2[4][2];


        icvCreateConvertMatrVect( rotMatr1,
                                  transVect1,
                                  rotMatr2,
                                  transVect2,
                                  convRotMatr,
                                  convTransVect);

        /* -------------Compute for first image-------------- */
        CvPoint2D32f pointb1;
        CvPoint2D32f pointe1;

        CvPoint2D32f pointb2;
        CvPoint2D32f pointe2;

        pointb1.x = (float)quad1[0][0];
        pointb1.y = (float)quad1[0][1];

        pointe1.x = (float)quad1[3][0];
        pointe1.y = (float)quad1[3][1];

        icvComputeeInfiniteProject1(convRotMatr,
                                    camMatr1,
                                    camMatr2,
                                    pointb1,
                                    &pointb2);

        icvComputeeInfiniteProject1(convRotMatr,
                                    camMatr1,
                                    camMatr2,
                                    pointe1,
                                    &pointe2);

        /*  JUST TEST FOR POINT */

        /* Compute distances */
        double dxOld,dyOld;
        double dxNew,dyNew;
        double distOld,distNew;

        dxOld = quad2[1][0] - quad2[0][0];
        dyOld = quad2[1][1] - quad2[0][1];
        distOld = dxOld*dxOld + dyOld*dyOld;

        dxNew = quad2[1][0] - pointb2.x;
        dyNew = quad2[1][1] - pointb2.y;
        distNew = dxNew*dxNew + dyNew*dyNew;

        if( distNew > distOld )
        {/* Get new points for second quad */
            newQuad2[0][0] = pointb2.x;
            newQuad2[0][1] = pointb2.y;
            newQuad2[3][0] = pointe2.x;
            newQuad2[3][1] = pointe2.y;
            newQuad1[0][0] = quad1[0][0];
            newQuad1[0][1] = quad1[0][1];
            newQuad1[3][0] = quad1[3][0];
            newQuad1[3][1] = quad1[3][1];
        }
        else
        {/* Get new points for first quad */

            pointb2.x = (float)quad2[0][0];
            pointb2.y = (float)quad2[0][1];

            pointe2.x = (float)quad2[3][0];
            pointe2.y = (float)quad2[3][1];

            icvComputeeInfiniteProject2(convRotMatr,
                                        camMatr1,
                                        camMatr2,
                                        &pointb1,
                                        pointb2);

            icvComputeeInfiniteProject2(convRotMatr,
                                        camMatr1,
                                        camMatr2,
                                        &pointe1,
                                        pointe2);


            /*  JUST TEST FOR POINT */

            newQuad2[0][0] = quad2[0][0];
            newQuad2[0][1] = quad2[0][1];
            newQuad2[3][0] = quad2[3][0];
            newQuad2[3][1] = quad2[3][1];

            newQuad1[0][0] = pointb1.x;
            newQuad1[0][1] = pointb1.y;
            newQuad1[3][0] = pointe1.x;
            newQuad1[3][1] = pointe1.y;
        }

        /* -------------Compute for second image-------------- */
        pointb1.x = (float)quad1[1][0];
        pointb1.y = (float)quad1[1][1];

        pointe1.x = (float)quad1[2][0];
        pointe1.y = (float)quad1[2][1];

        icvComputeeInfiniteProject1(convRotMatr,
                                    camMatr1,
                                    camMatr2,
                                    pointb1,
                                    &pointb2);

        icvComputeeInfiniteProject1(convRotMatr,
                                    camMatr1,
                                    camMatr2,
                                    pointe1,
                                    &pointe2);

        /* Compute distances */

        dxOld = quad2[0][0] - quad2[1][0];
        dyOld = quad2[0][1] - quad2[1][1];
        distOld = dxOld*dxOld + dyOld*dyOld;

        dxNew = quad2[0][0] - pointb2.x;
        dyNew = quad2[0][1] - pointb2.y;
        distNew = dxNew*dxNew + dyNew*dyNew;

        if( distNew > distOld )
        {/* Get new points for second quad */
            newQuad2[1][0] = pointb2.x;
            newQuad2[1][1] = pointb2.y;
            newQuad2[2][0] = pointe2.x;
            newQuad2[2][1] = pointe2.y;
            newQuad1[1][0] = quad1[1][0];
            newQuad1[1][1] = quad1[1][1];
            newQuad1[2][0] = quad1[2][0];
            newQuad1[2][1] = quad1[2][1];
        }
        else
        {/* Get new points for first quad */

            pointb2.x = (float)quad2[1][0];
            pointb2.y = (float)quad2[1][1];

            pointe2.x = (float)quad2[2][0];
            pointe2.y = (float)quad2[2][1];

            icvComputeeInfiniteProject2(convRotMatr,
                                        camMatr1,
                                        camMatr2,
                                        &pointb1,
                                        pointb2);

            icvComputeeInfiniteProject2(convRotMatr,
                                        camMatr1,
                                        camMatr2,
                                        &pointe1,
                                        pointe2);

            newQuad2[1][0] = quad2[1][0];
            newQuad2[1][1] = quad2[1][1];
            newQuad2[2][0] = quad2[2][0];
            newQuad2[2][1] = quad2[2][1];

            newQuad1[1][0] = pointb1.x;
            newQuad1[1][1] = pointb1.y;
            newQuad1[2][0] = pointe1.x;
            newQuad1[2][1] = pointe1.y;
        }



/*-------------------------------------------------------------------------------*/

        /* Copy new quads to old quad */
        int i;
        for( i = 0; i < 4; i++ )
        {
            {
                quad1[i][0] = newQuad1[i][0];
                quad1[i][1] = newQuad1[i][1];
                quad2[i][0] = newQuad2[i][0];
                quad2[i][1] = newQuad2[i][1];
            }
        }
    }
    /*=======================================================*/

    double warpWidth,warpHeight;

    warpWidth  = MAX(width1,width2);
    warpHeight = MAX(height1,height2);

    warpSize->width  = (int)warpWidth;
    warpSize->height = (int)warpHeight;

    warpSize->width  = cvRound(warpWidth-1);
    warpSize->height = cvRound(warpHeight-1);

/* !!! by Valery Mosyagin. this lines added just for test no warp */
    warpSize->width  = imageSize.width;
    warpSize->height = imageSize.height;

    return;
}


/*---------------------------------------------------------------------------------------*/

static void icvGetQuadsTransformNew(  CvSize        imageSize,
                            CvMatr32f     camMatr1,
                            CvMatr32f     camMatr2,
                            CvMatr32f     rotMatr1,
                            CvVect32f     transVect1,
                            CvSize*       warpSize,
                            double        quad1[4][2],
                            double        quad2[4][2],
                            CvMatr32f     fundMatr,
                            CvPoint3D32f* epipole1,
                            CvPoint3D32f* epipole2
                        )
{
    /* Convert data */
    /* Convert camera matrix */
    double camMatr1_64d[9];
    double camMatr2_64d[9];
    double rotMatr1_64d[9];
    double transVect1_64d[3];
    double rotMatr2_64d[9];
    double transVect2_64d[3];
    double fundMatr_64d[9];
    CvPoint3D64d epipole1_64d;
    CvPoint3D64d epipole2_64d;

    icvCvt_32f_64d(camMatr1,camMatr1_64d,9);
    icvCvt_32f_64d(camMatr2,camMatr2_64d,9);
    icvCvt_32f_64d(rotMatr1,rotMatr1_64d,9);
    icvCvt_32f_64d(transVect1,transVect1_64d,3);

    /* Create vector and matrix */

    rotMatr2_64d[0] = 1;
    rotMatr2_64d[1] = 0;
    rotMatr2_64d[2] = 0;
    rotMatr2_64d[3] = 0;
    rotMatr2_64d[4] = 1;
    rotMatr2_64d[5] = 0;
    rotMatr2_64d[6] = 0;
    rotMatr2_64d[7] = 0;
    rotMatr2_64d[8] = 1;

    transVect2_64d[0] = 0;
    transVect2_64d[1] = 0;
    transVect2_64d[2] = 0;

    icvGetQuadsTransform(   imageSize,
                            camMatr1_64d,
                            rotMatr1_64d,
                            transVect1_64d,
                            camMatr2_64d,
                            rotMatr2_64d,
                            transVect2_64d,
                            warpSize,
                            quad1,
                            quad2,
                            fundMatr_64d,
                            &epipole1_64d,
                            &epipole2_64d
                        );

    /* Convert epipoles */
    epipole1->x = (float)(epipole1_64d.x);
    epipole1->y = (float)(epipole1_64d.y);
    epipole1->z = (float)(epipole1_64d.z);

    epipole2->x = (float)(epipole2_64d.x);
    epipole2->y = (float)(epipole2_64d.y);
    epipole2->z = (float)(epipole2_64d.z);

    /* Convert fundamental matrix */
    icvCvt_64d_32f(fundMatr_64d,fundMatr,9);

    return;
}

/*---------------------------------------------------------------------------------------*/
void icvGetQuadsTransformStruct(  CvStereoCamera* stereoCamera)
{
    /* Wrapper for icvGetQuadsTransformNew */


    double  quad1[4][2];
    double  quad2[4][2];

    icvGetQuadsTransformNew(     cvSize(cvRound(stereoCamera->camera[0]->imgSize[0]),cvRound(stereoCamera->camera[0]->imgSize[1])),
                            stereoCamera->camera[0]->matrix,
                            stereoCamera->camera[1]->matrix,
                            stereoCamera->rotMatrix,
                            stereoCamera->transVector,
                            &(stereoCamera->warpSize),
                            quad1,
                            quad2,
                            stereoCamera->fundMatr,
                            &(stereoCamera->epipole[0]),
                            &(stereoCamera->epipole[1])
                        );

    int i;
    for( i = 0; i < 4; i++ )
    {
        stereoCamera->quad[0][i] = cvPoint2D32f(quad1[i][0],quad1[i][1]);
        stereoCamera->quad[1][i] = cvPoint2D32f(quad2[i][0],quad2[i][1]);
    }

    return;
}

/*---------------------------------------------------------------------------------------*/
void icvComputeStereoParamsForCameras(CvStereoCamera* stereoCamera)
{
    /* For given intrinsic and extrinsic parameters computes rest parameters
    **   such as fundamental matrix. warping coeffs, epipoles, ...
    */


    /* compute rotate matrix and translate vector */
    double rotMatr1[9];
    double rotMatr2[9];

    double transVect1[3];
    double transVect2[3];

    double convRotMatr[9];
    double convTransVect[3];

    /* fill matrices */
    icvCvt_32f_64d(stereoCamera->camera[0]->rotMatr,rotMatr1,9);
    icvCvt_32f_64d(stereoCamera->camera[1]->rotMatr,rotMatr2,9);

    icvCvt_32f_64d(stereoCamera->camera[0]->transVect,transVect1,3);
    icvCvt_32f_64d(stereoCamera->camera[1]->transVect,transVect2,3);

    icvCreateConvertMatrVect(   rotMatr1,
                                transVect1,
                                rotMatr2,
                                transVect2,
                                convRotMatr,
                                convTransVect);

    /* copy to stereo camera params */
    icvCvt_64d_32f(convRotMatr,stereoCamera->rotMatrix,9);
    icvCvt_64d_32f(convTransVect,stereoCamera->transVector,3);


    icvGetQuadsTransformStruct(stereoCamera);
    icvComputeRestStereoParams(stereoCamera);
}



/*---------------------------------------------------------------------------------------*/

/* Get cut line for one image */
void icvGetCutPiece(   CvVect64d areaLineCoef1,CvVect64d areaLineCoef2,
                    CvPoint2D64d epipole,
                    CvSize imageSize,
                    CvPoint2D64d* point11,CvPoint2D64d* point12,
                    CvPoint2D64d* point21,CvPoint2D64d* point22,
                    int* result)
{
    /* Compute nearest cut line to epipole */
    /* Get corners inside sector */
    /* Collect all candidate point */

    CvPoint2D64d candPoints[8];
    CvPoint2D64d midPoint = {0, 0};
    int numPoints = 0;
    int res;
    int i;

    double cutLine1[3];
    double cutLine2[3];

    /* Find middle line of sector */
    double midLine[3]={0,0,0};


    /* Different way  */
    CvPoint2D64d pointOnLine1;  pointOnLine1.x = pointOnLine1.y = 0;
    CvPoint2D64d pointOnLine2;  pointOnLine2.x = pointOnLine2.y = 0;

    CvPoint2D64d start1,end1;

    icvGetCrossRectDirect( imageSize,
                        areaLineCoef1[0],areaLineCoef1[1],areaLineCoef1[2],
                        &start1,&end1,&res);
    if( res > 0 )
    {
        pointOnLine1 = start1;
    }

    icvGetCrossRectDirect( imageSize,
                        areaLineCoef2[0],areaLineCoef2[1],areaLineCoef2[2],
                        &start1,&end1,&res);
    if( res > 0 )
    {
        pointOnLine2 = start1;
    }

    icvGetMiddleAnglePoint(epipole,pointOnLine1,pointOnLine2,&midPoint);

    icvGetCoefForPiece(epipole,midPoint,&midLine[0],&midLine[1],&midLine[2],&res);

    /* Test corner points */
    CvPoint2D64d cornerPoint;
    CvPoint2D64d tmpPoints[2];

    cornerPoint.x = 0;
    cornerPoint.y = 0;
    icvTestPoint( cornerPoint, areaLineCoef1, areaLineCoef2, epipole, &res);
    if( res == 1 )
    {/* Add point */
        candPoints[numPoints] = cornerPoint;
        numPoints++;
    }

    cornerPoint.x = imageSize.width;
    cornerPoint.y = 0;
    icvTestPoint( cornerPoint, areaLineCoef1, areaLineCoef2, epipole, &res);
    if( res == 1 )
    {/* Add point */
        candPoints[numPoints] = cornerPoint;
        numPoints++;
    }

    cornerPoint.x = imageSize.width;
    cornerPoint.y = imageSize.height;
    icvTestPoint( cornerPoint, areaLineCoef1, areaLineCoef2, epipole, &res);
    if( res == 1 )
    {/* Add point */
        candPoints[numPoints] = cornerPoint;
        numPoints++;
    }

    cornerPoint.x = 0;
    cornerPoint.y = imageSize.height;
    icvTestPoint( cornerPoint, areaLineCoef1, areaLineCoef2, epipole, &res);
    if( res == 1 )
    {/* Add point */
        candPoints[numPoints] = cornerPoint;
        numPoints++;
    }

    /* Find cross line 1 with image border */
    icvGetCrossRectDirect( imageSize,
                        areaLineCoef1[0],areaLineCoef1[1],areaLineCoef1[2],
                        &tmpPoints[0], &tmpPoints[1],
                        &res);
    for( i = 0; i < res; i++ )
    {
        candPoints[numPoints++] = tmpPoints[i];
    }

    /* Find cross line 2 with image border */
    icvGetCrossRectDirect( imageSize,
                        areaLineCoef2[0],areaLineCoef2[1],areaLineCoef2[2],
                        &tmpPoints[0], &tmpPoints[1],
                        &res);

    for( i = 0; i < res; i++ )
    {
        candPoints[numPoints++] = tmpPoints[i];
    }

    if( numPoints < 2 )
    {
        *result = 0;
        return;/* Error. Not enought points */
    }
    /* Project all points to middle line and get max and min */

    CvPoint2D64d projPoint;
    CvPoint2D64d minPoint; minPoint.x = minPoint.y = FLT_MAX;
    CvPoint2D64d maxPoint; maxPoint.x = maxPoint.y = -FLT_MAX;


    double dist;
    double maxDist = 0;
    double minDist = 10000000;


    for( i = 0; i < numPoints; i++ )
    {
        icvProjectPointToDirect(candPoints[i], midLine, &projPoint);
        icvGetPieceLength(epipole,projPoint,&dist);
        if( dist < minDist)
        {
            minDist = dist;
            minPoint = projPoint;
        }

        if( dist > maxDist)
        {
            maxDist = dist;
            maxPoint = projPoint;
        }
    }

    /* We know maximum and minimum points. Now we can compute cut lines */

    icvGetNormalDirect(midLine,minPoint,cutLine1);
    icvGetNormalDirect(midLine,maxPoint,cutLine2);

    /* Test for begin of line. */
    CvPoint2D64d tmpPoint2;

    /* Get cross with */
    icvGetCrossDirectDirect(areaLineCoef1,cutLine1,point11,&res);
    icvGetCrossDirectDirect(areaLineCoef2,cutLine1,point12,&res);

    icvGetCrossDirectDirect(areaLineCoef1,cutLine2,point21,&res);
    icvGetCrossDirectDirect(areaLineCoef2,cutLine2,point22,&res);

    if( epipole.x > imageSize.width * 0.5 )
    {/* Need to change points */
        tmpPoint2 = *point11;
        *point11 = *point21;
        *point21 = tmpPoint2;

        tmpPoint2 = *point12;
        *point12 = *point22;
        *point22 = tmpPoint2;
    }

    return;
}
/*---------------------------------------------------------------------------------------*/
/* Get middle angle */
void icvGetMiddleAnglePoint(   CvPoint2D64d basePoint,
                            CvPoint2D64d point1,CvPoint2D64d point2,
                            CvPoint2D64d* midPoint)
{/* !!! May be need to return error */

    double dist1;
    double dist2;
    icvGetPieceLength(basePoint,point1,&dist1);
    icvGetPieceLength(basePoint,point2,&dist2);
    CvPoint2D64d pointNew1;
    CvPoint2D64d pointNew2;
    double alpha = dist2/dist1;

    pointNew1.x = basePoint.x + (1.0/alpha) * ( point2.x - basePoint.x );
    pointNew1.y = basePoint.y + (1.0/alpha) * ( point2.y - basePoint.y );

    pointNew2.x = basePoint.x + alpha * ( point1.x - basePoint.x );
    pointNew2.y = basePoint.y + alpha * ( point1.y - basePoint.y );

    int res;
    icvGetCrossPiecePiece(point1,point2,pointNew1,pointNew2,midPoint,&res);

    return;
}

/*---------------------------------------------------------------------------------------*/
/* Get normal direct to direct in line */
void icvGetNormalDirect(CvVect64d direct,CvPoint2D64d point,CvVect64d normDirect)
{
    normDirect[0] =   direct[1];
    normDirect[1] = - direct[0];
    normDirect[2] = -(normDirect[0]*point.x + normDirect[1]*point.y);
    return;
}

/*---------------------------------------------------------------------------------------*/
CV_IMPL double icvGetVect(CvPoint2D64d basePoint,CvPoint2D64d point1,CvPoint2D64d point2)
{
    return  (point1.x - basePoint.x)*(point2.y - basePoint.y) -
            (point2.x - basePoint.x)*(point1.y - basePoint.y);
}
/*---------------------------------------------------------------------------------------*/
/* Test for point in sector           */
/* Return 0 - point not inside sector */
/* Return 1 - point inside sector     */
void icvTestPoint( CvPoint2D64d testPoint,
                CvVect64d line1,CvVect64d line2,
                CvPoint2D64d basePoint,
                int* result)
{
    CvPoint2D64d point1,point2;

    icvProjectPointToDirect(testPoint,line1,&point1);
    icvProjectPointToDirect(testPoint,line2,&point2);

    double sign1 = icvGetVect(basePoint,point1,point2);
    double sign2 = icvGetVect(basePoint,point1,testPoint);
    if( sign1 * sign2 > 0 )
    {/* Correct for first line */
        sign1 = - sign1;
        sign2 = icvGetVect(basePoint,point2,testPoint);
        if( sign1 * sign2 > 0 )
        {/* Correct for both lines */
            *result = 1;
        }
        else
        {
            *result = 0;
        }
    }
    else
    {
        *result = 0;
    }

    return;
}

/*---------------------------------------------------------------------------------------*/
/* Project point to line */
void icvProjectPointToDirect(  CvPoint2D64d point,CvVect64d lineCoeff,
                            CvPoint2D64d* projectPoint)
{
    double a = lineCoeff[0];
    double b = lineCoeff[1];

    double det =  1.0 / ( a*a + b*b );
    double delta =  a*point.y - b*point.x;

    projectPoint->x = ( -a*lineCoeff[2] - b * delta ) * det;
    projectPoint->y = ( -b*lineCoeff[2] + a * delta ) * det ;

    return;
}

/*---------------------------------------------------------------------------------------*/
/* Get distance from point to direction */
void icvGetDistanceFromPointToDirect( CvPoint2D64d point,CvVect64d lineCoef,double*dist)
{
    CvPoint2D64d tmpPoint;
    icvProjectPointToDirect(point,lineCoef,&tmpPoint);
    double dx = point.x - tmpPoint.x;
    double dy = point.y - tmpPoint.y;
    *dist = sqrt(dx*dx+dy*dy);
    return;
}
/*---------------------------------------------------------------------------------------*/

CV_IMPL IplImage* icvCreateIsometricImage( IplImage* src, IplImage* dst,
                                       int desired_depth, int desired_num_channels )
{
    CvSize src_size ;
    src_size.width = src->width;
    src_size.height = src->height;

    CvSize dst_size = src_size;

    if( dst )
    {
        dst_size.width = dst->width;
        dst_size.height = dst->height;
    }

    if( !dst || dst->depth != desired_depth ||
        dst->nChannels != desired_num_channels ||
        dst_size.width != src_size.width ||
        dst_size.height != src_size.height )
    {
        cvReleaseImage( &dst );
        dst = cvCreateImage( src_size, desired_depth, desired_num_channels );
        CvRect rect = cvRect(0,0,src_size.width,src_size.height);
        cvSetImageROI( dst, rect );

    }

    return dst;
}

static int
icvCvt_32f_64d( float *src, double *dst, int size )
{
    int t;

    if( !src || !dst )
        return CV_NULLPTR_ERR;
    if( size <= 0 )
        return CV_BADRANGE_ERR;

    for( t = 0; t < size; t++ )
    {
        dst[t] = (double) (src[t]);
    }

    return CV_OK;
}

/*======================================================================================*/
/* Type conversion double -> float */
static int
icvCvt_64d_32f( double *src, float *dst, int size )
{
    int t;

    if( !src || !dst )
        return CV_NULLPTR_ERR;
    if( size <= 0 )
        return CV_BADRANGE_ERR;

    for( t = 0; t < size; t++ )
    {
        dst[t] = (float) (src[t]);
    }

    return CV_OK;
}

/*----------------------------------------------------------------------------------*/

#if 0
/* Find line which cross frame by line(a,b,c) */
static void FindLineForEpiline(    CvSize imageSize,
                            float a,float b,float c,
                            CvPoint2D32f *start,CvPoint2D32f *end,
                            int*)
{
    CvPoint2D32f frameBeg;

    CvPoint2D32f frameEnd;
    CvPoint2D32f cross[4];
    int     haveCross[4];
    float   dist;

    haveCross[0] = 0;
    haveCross[1] = 0;
    haveCross[2] = 0;
    haveCross[3] = 0;

    frameBeg.x = 0;
    frameBeg.y = 0;
    frameEnd.x = (float)(imageSize.width);
    frameEnd.y = 0;
    haveCross[0] = icvGetCrossLineDirect(frameBeg,frameEnd,a,b,c,&cross[0]);

    frameBeg.x = (float)(imageSize.width);
    frameBeg.y = 0;
    frameEnd.x = (float)(imageSize.width);
    frameEnd.y = (float)(imageSize.height);
    haveCross[1] = icvGetCrossLineDirect(frameBeg,frameEnd,a,b,c,&cross[1]);

    frameBeg.x = (float)(imageSize.width);
    frameBeg.y = (float)(imageSize.height);
    frameEnd.x = 0;
    frameEnd.y = (float)(imageSize.height);
    haveCross[2] = icvGetCrossLineDirect(frameBeg,frameEnd,a,b,c,&cross[2]);

    frameBeg.x = 0;
    frameBeg.y = (float)(imageSize.height);
    frameEnd.x = 0;
    frameEnd.y = 0;
    haveCross[3] = icvGetCrossLineDirect(frameBeg,frameEnd,a,b,c,&cross[3]);

    int n;
    float minDist = (float)(INT_MAX);
    float maxDist = (float)(INT_MIN);

    int maxN = -1;
    int minN = -1;

    double midPointX = imageSize.width  / 2.0;
    double midPointY = imageSize.height / 2.0;

    for( n = 0; n < 4; n++ )
    {
        if( haveCross[n] > 0 )
        {
            dist =  (float)((midPointX - cross[n].x)*(midPointX - cross[n].x) +
                            (midPointY - cross[n].y)*(midPointY - cross[n].y));

            if( dist < minDist )
            {
                minDist = dist;
                minN = n;
            }

            if( dist > maxDist )
            {
                maxDist = dist;
                maxN = n;
            }
        }
    }

    if( minN >= 0 && maxN >= 0 && (minN != maxN) )
    {
        *start = cross[minN];
        *end   = cross[maxN];
    }
    else
    {
        start->x = 0;
        start->y = 0;
        end->x = 0;
        end->y = 0;
    }

    return;

}


/*----------------------------------------------------------------------------------*/
static int GetAngleLinee( CvPoint2D32f epipole, CvSize imageSize,CvPoint2D32f point1,CvPoint2D32f point2)
{
    float width  = (float)(imageSize.width);
    float height = (float)(imageSize.height);

    /* Get crosslines with image corners */

    /* Find four lines */

    CvPoint2D32f pa,pb,pc,pd;

    pa.x = 0;
    pa.y = 0;

    pb.x = width;
    pb.y = 0;

    pd.x = width;
    pd.y = height;

    pc.x = 0;
    pc.y = height;

    /* We can compute points for angle */
    /* Test for place section */

    float x,y;
    x = epipole.x;
    y = epipole.y;

    if( x < 0 )
    {/* 1,4,7 */
        if( y < 0)
        {/* 1 */
            point1 = pb;
            point2 = pc;
        }
        else if( y > height )
        {/* 7 */
            point1 = pa;
            point2 = pd;
        }
        else
        {/* 4 */
            point1 = pa;
            point2 = pc;
        }
    }
    else if ( x > width )
    {/* 3,6,9 */
        if( y < 0 )
        {/* 3 */
            point1 = pa;
            point2 = pd;
        }
        else if ( y > height )
        {/* 9 */
            point1 = pc;
            point2 = pb;
        }
        else
        {/* 6 */
            point1 = pb;
            point2 = pd;
        }
    }
    else
    {/* 2,5,8 */
        if( y < 0 )
        {/* 2 */
            point1 = pa;
            point2 = pb;
        }
        else if( y > height )
        {/* 8 */
            point1 = pc;
            point2 = pd;
        }
        else
        {/* 5 - point in the image */
            return 2;
        }


    }


    return 0;
}

/*--------------------------------------------------------------------------------------*/
static void icvComputePerspectiveCoeffs(const CvPoint2D32f srcQuad[4],const CvPoint2D32f dstQuad[4],double coeffs[3][3])
{/* Computes perspective coeffs for transformation from src to dst quad */


    CV_FUNCNAME( "icvComputePerspectiveCoeffs" );

    __BEGIN__;

    double A[64];
    double b[8];
    double c[8];
    CvPoint2D32f pt[4];
    int i;

    pt[0] = srcQuad[0];
    pt[1] = srcQuad[1];
    pt[2] = srcQuad[2];
    pt[3] = srcQuad[3];

    for( i = 0; i < 4; i++ )
    {
#if 0
        double x = dstQuad[i].x;
        double y = dstQuad[i].y;
        double X = pt[i].x;
        double Y = pt[i].y;
#else
        double x = pt[i].x;
        double y = pt[i].y;
        double X = dstQuad[i].x;
        double Y = dstQuad[i].y;
#endif
        double* a = A + i*16;

        a[0] = x;
        a[1] = y;
        a[2] = 1;
        a[3] = 0;
        a[4] = 0;
        a[5] = 0;
        a[6] = -X*x;
        a[7] = -X*y;

        a += 8;

        a[0] = 0;
        a[1] = 0;
        a[2] = 0;
        a[3] = x;
        a[4] = y;
        a[5] = 1;
        a[6] = -Y*x;
        a[7] = -Y*y;

        b[i*2] = X;
        b[i*2 + 1] = Y;
    }

    {
    double invA[64];
    CvMat matA = cvMat( 8, 8, CV_64F, A );
    CvMat matInvA = cvMat( 8, 8, CV_64F, invA );
    CvMat matB = cvMat( 8, 1, CV_64F, b );
    CvMat matX = cvMat( 8, 1, CV_64F, c );

    CV_CALL( cvPseudoInverse( &matA, &matInvA ));
    CV_CALL( cvMatMulAdd( &matInvA, &matB, 0, &matX ));
    }

    coeffs[0][0] = c[0];
    coeffs[0][1] = c[1];
    coeffs[0][2] = c[2];
    coeffs[1][0] = c[3];
    coeffs[1][1] = c[4];
    coeffs[1][2] = c[5];
    coeffs[2][0] = c[6];
    coeffs[2][1] = c[7];
    coeffs[2][2] = 1.0;

    __END__;

    return;
}
#endif

/*--------------------------------------------------------------------------------------*/

CV_IMPL void cvComputePerspectiveMap(const double c[3][3], CvArr* rectMapX, CvArr* rectMapY )
{
    CV_FUNCNAME( "cvComputePerspectiveMap" );

    __BEGIN__;

    CvSize size;
    CvMat  stubx, *mapx = (CvMat*)rectMapX;
    CvMat  stuby, *mapy = (CvMat*)rectMapY;
    int i, j;

    CV_CALL( mapx = cvGetMat( mapx, &stubx ));
    CV_CALL( mapy = cvGetMat( mapy, &stuby ));

    if( CV_MAT_TYPE( mapx->type ) != CV_32FC1 || CV_MAT_TYPE( mapy->type ) != CV_32FC1 )
        CV_ERROR( CV_StsUnsupportedFormat, "" );

    size = cvGetMatSize(mapx);
    assert( fabs(c[2][2] - 1.) < FLT_EPSILON );

    for( i = 0; i < size.height; i++ )
    {
        float* mx = (float*)(mapx->data.ptr + mapx->step*i);
        float* my = (float*)(mapy->data.ptr + mapy->step*i);

        for( j = 0; j < size.width; j++ )
        {
            double w = 1./(c[2][0]*j + c[2][1]*i + 1.);
            double x = (c[0][0]*j + c[0][1]*i + c[0][2])*w;
            double y = (c[1][0]*j + c[1][1]*i + c[1][2])*w;

            mx[j] = (float)x;
            my[j] = (float)y;
        }
    }

    __END__;
}

/*--------------------------------------------------------------------------------------*/

CV_IMPL void cvInitPerspectiveTransform( CvSize size, const CvPoint2D32f quad[4], double matrix[3][3],
                                              CvArr* rectMap )
{
    /* Computes Perspective Transform coeffs and map if need
        for given image size and given result quad */
    CV_FUNCNAME( "cvInitPerspectiveTransform" );

    __BEGIN__;

    double A[64];
    double b[8];
    double c[8];
    CvPoint2D32f pt[4];
    CvMat  mapstub, *map = (CvMat*)rectMap;
    int i, j;

    if( map )
    {
        CV_CALL( map = cvGetMat( map, &mapstub ));

        if( CV_MAT_TYPE( map->type ) != CV_32FC2 )
            CV_ERROR( CV_StsUnsupportedFormat, "" );

        if( map->width != size.width || map->height != size.height )
            CV_ERROR( CV_StsUnmatchedSizes, "" );
    }

    pt[0] = cvPoint2D32f( 0, 0 );
    pt[1] = cvPoint2D32f( size.width, 0 );
    pt[2] = cvPoint2D32f( size.width, size.height );
    pt[3] = cvPoint2D32f( 0, size.height );

    for( i = 0; i < 4; i++ )
    {
#if 0
        double x = quad[i].x;
        double y = quad[i].y;
        double X = pt[i].x;
        double Y = pt[i].y;
#else
        double x = pt[i].x;
        double y = pt[i].y;
        double X = quad[i].x;
        double Y = quad[i].y;
#endif
        double* a = A + i*16;

        a[0] = x;
        a[1] = y;
        a[2] = 1;
        a[3] = 0;
        a[4] = 0;
        a[5] = 0;
        a[6] = -X*x;
        a[7] = -X*y;

        a += 8;

        a[0] = 0;
        a[1] = 0;
        a[2] = 0;
        a[3] = x;
        a[4] = y;
        a[5] = 1;
        a[6] = -Y*x;
        a[7] = -Y*y;

        b[i*2] = X;
        b[i*2 + 1] = Y;
    }

    {
    double invA[64];
    CvMat matA = cvMat( 8, 8, CV_64F, A );
    CvMat matInvA = cvMat( 8, 8, CV_64F, invA );
    CvMat matB = cvMat( 8, 1, CV_64F, b );
    CvMat matX = cvMat( 8, 1, CV_64F, c );

    CV_CALL( cvPseudoInverse( &matA, &matInvA ));
    CV_CALL( cvMatMulAdd( &matInvA, &matB, 0, &matX ));
    }

    matrix[0][0] = c[0];
    matrix[0][1] = c[1];
    matrix[0][2] = c[2];
    matrix[1][0] = c[3];
    matrix[1][1] = c[4];
    matrix[1][2] = c[5];
    matrix[2][0] = c[6];
    matrix[2][1] = c[7];
    matrix[2][2] = 1.0;

    if( map )
    {
        for( i = 0; i < size.height; i++ )
        {
            CvPoint2D32f* maprow = (CvPoint2D32f*)(map->data.ptr + map->step*i);
            for( j = 0; j < size.width; j++ )
            {
                double w = 1./(c[6]*j + c[7]*i + 1.);
                double x = (c[0]*j + c[1]*i + c[2])*w;
                double y = (c[3]*j + c[4]*i + c[5])*w;

                maprow[j].x = (float)x;
                maprow[j].y = (float)y;
            }
        }
    }

    __END__;

    return;
}


/*-----------------------------------------------------------------------*/
/* Compute projected infinite point for second image if first image point is known */
void icvComputeeInfiniteProject1(   CvMatr64d     rotMatr,
                                    CvMatr64d     camMatr1,
                                    CvMatr64d     camMatr2,
                                    CvPoint2D32f  point1,
                                    CvPoint2D32f* point2)
{
    double invMatr1[9];
    icvInvertMatrix_64d(camMatr1,3,invMatr1);
    double P1[3];
    double p1[3];
    p1[0] = (double)(point1.x);
    p1[1] = (double)(point1.y);
    p1[2] = 1;

    icvMulMatrix_64d(   invMatr1,
                        3,3,
                        p1,
                        1,3,
                        P1);

    double invR[9];
    icvTransposeMatrix_64d( rotMatr, 3, 3, invR );

    /* Change system 1 to system 2 */
    double P2[3];
    icvMulMatrix_64d(   invR,
                        3,3,
                        P1,
                        1,3,
                        P2);

    /* Now we can project this point to image 2 */
    double projP[3];

    icvMulMatrix_64d(   camMatr2,
                        3,3,
                        P2,
                        1,3,
                        projP);

    point2->x = (float)(projP[0] / projP[2]);
    point2->y = (float)(projP[1] / projP[2]);

    return;
}

/*-----------------------------------------------------------------------*/
/* Compute projected infinite point for second image if first image point is known */
void icvComputeeInfiniteProject2(   CvMatr64d     rotMatr,
                                    CvMatr64d     camMatr1,
                                    CvMatr64d     camMatr2,
                                    CvPoint2D32f*  point1,
                                    CvPoint2D32f point2)
{
    double invMatr2[9];
    icvInvertMatrix_64d(camMatr2,3,invMatr2);
    double P2[3];
    double p2[3];
    p2[0] = (double)(point2.x);
    p2[1] = (double)(point2.y);
    p2[2] = 1;

    icvMulMatrix_64d(   invMatr2,
                        3,3,
                        p2,
                        1,3,
                        P2);

    /* Change system 1 to system 2 */

    double P1[3];
    icvMulMatrix_64d(   rotMatr,
                        3,3,
                        P2,
                        1,3,
                        P1);

    /* Now we can project this point to image 2 */
    double projP[3];

    icvMulMatrix_64d(   camMatr1,
                        3,3,
                        P1,
                        1,3,
                        projP);

    point1->x = (float)(projP[0] / projP[2]);
    point1->y = (float)(projP[1] / projP[2]);

    return;
}

/* Select best R and t for given cameras, points, ... */
/* For both cameras */
static int icvSelectBestRt(         int           numImages,
                                    int*          numPoints,
                                    CvPoint2D32f* imagePoints1,
                                    CvPoint2D32f* imagePoints2,
                                    CvPoint3D32f* objectPoints,

                                    CvMatr32f     cameraMatrix1,
                                    CvVect32f     distortion1,
                                    CvMatr32f     rotMatrs1,
                                    CvVect32f     transVects1,

                                    CvMatr32f     cameraMatrix2,
                                    CvVect32f     distortion2,
                                    CvMatr32f     rotMatrs2,
                                    CvVect32f     transVects2,

                                    CvMatr32f     bestRotMatr,
                                    CvVect32f     bestTransVect
                                    )
{

    /* Need to convert input data 32 -> 64 */
    CvPoint3D64d* objectPoints_64d;

    double* rotMatrs1_64d;
    double* rotMatrs2_64d;

    double* transVects1_64d;
    double* transVects2_64d;

    double cameraMatrix1_64d[9];
    double cameraMatrix2_64d[9];

    double distortion1_64d[4];
    double distortion2_64d[4];

    /* allocate memory for 64d data */
    int totalNum = 0;

    for(int i = 0; i < numImages; i++ )
    {
        totalNum += numPoints[i];
    }

    objectPoints_64d = (CvPoint3D64d*)calloc(totalNum,sizeof(CvPoint3D64d));

    rotMatrs1_64d    = (double*)calloc(numImages,sizeof(double)*9);
    rotMatrs2_64d    = (double*)calloc(numImages,sizeof(double)*9);

    transVects1_64d  = (double*)calloc(numImages,sizeof(double)*3);
    transVects2_64d  = (double*)calloc(numImages,sizeof(double)*3);

    /* Convert input data to 64d */

    icvCvt_32f_64d((float*)objectPoints, (double*)objectPoints_64d,  totalNum*3);

    icvCvt_32f_64d(rotMatrs1, rotMatrs1_64d,  numImages*9);
    icvCvt_32f_64d(rotMatrs2, rotMatrs2_64d,  numImages*9);

    icvCvt_32f_64d(transVects1, transVects1_64d,  numImages*3);
    icvCvt_32f_64d(transVects2, transVects2_64d,  numImages*3);

    /* Convert to arrays */
    icvCvt_32f_64d(cameraMatrix1, cameraMatrix1_64d, 9);
    icvCvt_32f_64d(cameraMatrix2, cameraMatrix2_64d, 9);

    icvCvt_32f_64d(distortion1, distortion1_64d, 4);
    icvCvt_32f_64d(distortion2, distortion2_64d, 4);


    /* for each R and t compute error for image pair */
    float* errors;

    errors = (float*)calloc(numImages*numImages,sizeof(float));
    if( errors == 0 )
    {
        return CV_OUTOFMEM_ERR;
    }

    int currImagePair;
    int currRt;
    for( currRt = 0; currRt < numImages; currRt++ )
    {
        int begPoint = 0;
        for(currImagePair = 0; currImagePair < numImages; currImagePair++ )
        {
            /* For current R,t R,t compute relative position of cameras */

            double convRotMatr[9];
            double convTransVect[3];

            icvCreateConvertMatrVect( rotMatrs1_64d + currRt*9,
                                      transVects1_64d + currRt*3,
                                      rotMatrs2_64d + currRt*9,
                                      transVects2_64d + currRt*3,
                                      convRotMatr,
                                      convTransVect);

            /* Project points using relative position of cameras */

            double convRotMatr2[9];
            double convTransVect2[3];

            convRotMatr2[0] = 1;
            convRotMatr2[1] = 0;
            convRotMatr2[2] = 0;

            convRotMatr2[3] = 0;
            convRotMatr2[4] = 1;
            convRotMatr2[5] = 0;

            convRotMatr2[6] = 0;
            convRotMatr2[7] = 0;
            convRotMatr2[8] = 1;

            convTransVect2[0] = 0;
            convTransVect2[1] = 0;
            convTransVect2[2] = 0;

            /* Compute error for given pair and Rt */
            /* We must project points to image and compute error */

            CvPoint2D64d* projImagePoints1;
            CvPoint2D64d* projImagePoints2;

            CvPoint3D64d* points1;
            CvPoint3D64d* points2;

            int numberPnt;
            numberPnt = numPoints[currImagePair];
            projImagePoints1 = (CvPoint2D64d*)calloc(numberPnt,sizeof(CvPoint2D64d));
            projImagePoints2 = (CvPoint2D64d*)calloc(numberPnt,sizeof(CvPoint2D64d));

            points1 = (CvPoint3D64d*)calloc(numberPnt,sizeof(CvPoint3D64d));
            points2 = (CvPoint3D64d*)calloc(numberPnt,sizeof(CvPoint3D64d));

            /* Transform object points to first camera position */
            for(int i = 0; i < numberPnt; i++ )
            {
                /* Create second camera point */
                CvPoint3D64d tmpPoint;
                tmpPoint.x = (double)(objectPoints[i].x);
                tmpPoint.y = (double)(objectPoints[i].y);
                tmpPoint.z = (double)(objectPoints[i].z);

                icvConvertPointSystem(  tmpPoint,
                                        points2+i,
                                        rotMatrs2_64d + currImagePair*9,
                                        transVects2_64d + currImagePair*3);

                /* Create first camera point using R, t */
                icvConvertPointSystem(  points2[i],
                                        points1+i,
                                        convRotMatr,
                                        convTransVect);

                CvPoint3D64d tmpPoint2 = { 0, 0, 0 };
                icvConvertPointSystem(  tmpPoint,
                                        &tmpPoint2,
                                        rotMatrs1_64d + currImagePair*9,
                                        transVects1_64d + currImagePair*3);
                /*double err;
                double dx,dy,dz;
                dx = tmpPoint2.x - points1[i].x;
                dy = tmpPoint2.y - points1[i].y;
                dz = tmpPoint2.z - points1[i].z;
                err = sqrt(dx*dx + dy*dy + dz*dz);*/
            }

#if 0
            cvProjectPointsSimple(  numPoints[currImagePair],
                                    objectPoints_64d + begPoint,
                                    rotMatrs1_64d + currRt*9,
                                    transVects1_64d + currRt*3,
                                    cameraMatrix1_64d,
                                    distortion1_64d,
                                    projImagePoints1);

            cvProjectPointsSimple(  numPoints[currImagePair],
                                    objectPoints_64d + begPoint,
                                    rotMatrs2_64d + currRt*9,
                                    transVects2_64d + currRt*3,
                                    cameraMatrix2_64d,
                                    distortion2_64d,
                                    projImagePoints2);
#endif

            /* Project with no translate and no rotation */

#if 0
            {
                double nodist[4] = {0,0,0,0};
                cvProjectPointsSimple(  numPoints[currImagePair],
                                        points1,
                                        convRotMatr2,
                                        convTransVect2,
                                        cameraMatrix1_64d,
                                        nodist,
                                        projImagePoints1);

                cvProjectPointsSimple(  numPoints[currImagePair],
                                        points2,
                                        convRotMatr2,
                                        convTransVect2,
                                        cameraMatrix2_64d,
                                        nodist,
                                        projImagePoints2);

            }
#endif

            cvProjectPointsSimple(  numPoints[currImagePair],
                                    points1,
                                    convRotMatr2,
                                    convTransVect2,
                                    cameraMatrix1_64d,
                                    distortion1_64d,
                                    projImagePoints1);

            cvProjectPointsSimple(  numPoints[currImagePair],
                                    points2,
                                    convRotMatr2,
                                    convTransVect2,
                                    cameraMatrix2_64d,
                                    distortion2_64d,
                                    projImagePoints2);

            /* points are projected. Compute error */

            int currPoint;
            double err1 = 0;
            double err2 = 0;
            double err;
            for( currPoint = 0; currPoint < numberPnt; currPoint++ )
            {
                double len1,len2;
                double dx1,dy1;
                dx1 = imagePoints1[begPoint+currPoint].x - projImagePoints1[currPoint].x;
                dy1 = imagePoints1[begPoint+currPoint].y - projImagePoints1[currPoint].y;
                len1 = sqrt(dx1*dx1 + dy1*dy1);
                err1 += len1;

                double dx2,dy2;
                dx2 = imagePoints2[begPoint+currPoint].x - projImagePoints2[currPoint].x;
                dy2 = imagePoints2[begPoint+currPoint].y - projImagePoints2[currPoint].y;
                len2 = sqrt(dx2*dx2 + dy2*dy2);
                err2 += len2;
            }

            err1 /= (float)(numberPnt);
            err2 /= (float)(numberPnt);

            err = (err1+err2) * 0.5;
            begPoint += numberPnt;

            /* Set this error to */
            errors[numImages*currImagePair+currRt] = (float)err;

            free(points1);
            free(points2);
            free(projImagePoints1);
            free(projImagePoints2);
        }
    }

    /* Just select R and t with minimal average error */

    int bestnumRt = 0;
    float minError = 0;/* Just for no warnings. Uses 'first' flag. */
    int first = 1;
    for( currRt = 0; currRt < numImages; currRt++ )
    {
        float avErr = 0;
        for(currImagePair = 0; currImagePair < numImages; currImagePair++ )
        {
            avErr += errors[numImages*currImagePair+currRt];
        }
        avErr /= (float)(numImages);

        if( first )
        {
            bestnumRt = 0;
            minError = avErr;
            first = 0;
        }
        else
        {
            if( avErr < minError )
            {
                bestnumRt = currRt;
                minError = avErr;
            }
        }

    }

    double bestRotMatr_64d[9];
    double bestTransVect_64d[3];

    icvCreateConvertMatrVect( rotMatrs1_64d + bestnumRt * 9,
                              transVects1_64d + bestnumRt * 3,
                              rotMatrs2_64d + bestnumRt * 9,
                              transVects2_64d + bestnumRt * 3,
                              bestRotMatr_64d,
                              bestTransVect_64d);

    icvCvt_64d_32f(bestRotMatr_64d,bestRotMatr,9);
    icvCvt_64d_32f(bestTransVect_64d,bestTransVect,3);


    free(errors);

    return CV_OK;
}


/* ----------------- Stereo calibration functions --------------------- */

float icvDefinePointPosition(CvPoint2D32f point1,CvPoint2D32f point2,CvPoint2D32f point)
{
    float ax = point2.x - point1.x;
    float ay = point2.y - point1.y;

    float bx = point.x - point1.x;
    float by = point.y - point1.y;

    return (ax*by - ay*bx);
}

/* Convert function for stereo warping */
int icvConvertWarpCoordinates(double coeffs[3][3],
                                CvPoint2D32f* cameraPoint,
                                CvPoint2D32f* warpPoint,
                                int direction)
{
    double x,y;
    double det;
    if( direction == CV_WARP_TO_CAMERA )
    {/* convert from camera image to warped image coordinates */
        x = warpPoint->x;
        y = warpPoint->y;

        det = (coeffs[2][0] * x + coeffs[2][1] * y + coeffs[2][2]);
        if( fabs(det) > 1e-8 )
        {
            cameraPoint->x = (float)((coeffs[0][0] * x + coeffs[0][1] * y + coeffs[0][2]) / det);
            cameraPoint->y = (float)((coeffs[1][0] * x + coeffs[1][1] * y + coeffs[1][2]) / det);
            return CV_OK;
        }
    }
    else if( direction == CV_CAMERA_TO_WARP )
    {/* convert from warped image to camera image coordinates */
        x = cameraPoint->x;
        y = cameraPoint->y;

        det = (coeffs[2][0]*x-coeffs[0][0])*(coeffs[2][1]*y-coeffs[1][1])-(coeffs[2][1]*x-coeffs[0][1])*(coeffs[2][0]*y-coeffs[1][0]);

        if( fabs(det) > 1e-8 )
        {
            warpPoint->x = (float)(((coeffs[0][2]-coeffs[2][2]*x)*(coeffs[2][1]*y-coeffs[1][1])-(coeffs[2][1]*x-coeffs[0][1])*(coeffs[1][2]-coeffs[2][2]*y))/det);
            warpPoint->y = (float)(((coeffs[2][0]*x-coeffs[0][0])*(coeffs[1][2]-coeffs[2][2]*y)-(coeffs[0][2]-coeffs[2][2]*x)*(coeffs[2][0]*y-coeffs[1][0]))/det);
            return CV_OK;
        }
    }

    return CV_BADFACTOR_ERR;
}

/* Compute stereo params using some camera params */
/* by Valery Mosyagin. int ComputeRestStereoParams(StereoParams *stereoparams) */
int icvComputeRestStereoParams(CvStereoCamera *stereoparams)
{


    icvGetQuadsTransformStruct(stereoparams);

    cvInitPerspectiveTransform( stereoparams->warpSize,
                                stereoparams->quad[0],
                                stereoparams->coeffs[0],
                                0);

    cvInitPerspectiveTransform( stereoparams->warpSize,
                                stereoparams->quad[1],
                                stereoparams->coeffs[1],
                                0);

    /* Create border for warped images */
    CvPoint2D32f corns[4];
    corns[0].x = 0;
    corns[0].y = 0;

    corns[1].x = (float)(stereoparams->camera[0]->imgSize[0]-1);
    corns[1].y = 0;

    corns[2].x = (float)(stereoparams->camera[0]->imgSize[0]-1);
    corns[2].y = (float)(stereoparams->camera[0]->imgSize[1]-1);

    corns[3].x = 0;
    corns[3].y = (float)(stereoparams->camera[0]->imgSize[1]-1);

    for(int i = 0; i < 4; i++ )
    {
        /* For first camera */
        icvConvertWarpCoordinates( stereoparams->coeffs[0],
                                corns+i,
                                stereoparams->border[0]+i,
                                CV_CAMERA_TO_WARP);

        /* For second camera */
        icvConvertWarpCoordinates( stereoparams->coeffs[1],
                                corns+i,
                                stereoparams->border[1]+i,
                                CV_CAMERA_TO_WARP);
    }

    /* Test compute  */
    {
        CvPoint2D32f warpPoints[4];
        warpPoints[0] = cvPoint2D32f(0,0);
        warpPoints[1] = cvPoint2D32f(stereoparams->warpSize.width-1,0);
        warpPoints[2] = cvPoint2D32f(stereoparams->warpSize.width-1,stereoparams->warpSize.height-1);
        warpPoints[3] = cvPoint2D32f(0,stereoparams->warpSize.height-1);

        CvPoint2D32f camPoints1[4];
        CvPoint2D32f camPoints2[4];

        for( int i = 0; i < 4; i++ )
        {
            icvConvertWarpCoordinates(stereoparams->coeffs[0],
                                camPoints1+i,
                                warpPoints+i,
                                CV_WARP_TO_CAMERA);

            icvConvertWarpCoordinates(stereoparams->coeffs[1],
                                camPoints2+i,
                                warpPoints+i,
                                CV_WARP_TO_CAMERA);
        }
    }


    /* Allocate memory for scanlines coeffs */

    stereoparams->lineCoeffs = (CvStereoLineCoeff*)calloc(stereoparams->warpSize.height,sizeof(CvStereoLineCoeff));

    /* Compute coeffs for epilines  */

    icvComputeCoeffForStereo( stereoparams);

    /* all coeffs are known */
    return CV_OK;
}

/*-------------------------------------------------------------------------------------------*/

int icvStereoCalibration( int numImages,
                            int* nums,
                            CvSize imageSize,
                            CvPoint2D32f* imagePoints1,
                            CvPoint2D32f* imagePoints2,
                            CvPoint3D32f* objectPoints,
                            CvStereoCamera* stereoparams
                           )
{
    /* Firstly we must calibrate both cameras */
    /*  Alocate memory for data */
    /* Allocate for translate vectors */
    float* transVects1;
    float* transVects2;
    float* rotMatrs1;
    float* rotMatrs2;

    transVects1 = (float*)calloc(numImages,sizeof(float)*3);
    transVects2 = (float*)calloc(numImages,sizeof(float)*3);

    rotMatrs1   = (float*)calloc(numImages,sizeof(float)*9);
    rotMatrs2   = (float*)calloc(numImages,sizeof(float)*9);

    /* Calibrate first camera */
    cvCalibrateCamera(  numImages,
                        nums,
                        imageSize,
                        imagePoints1,
                        objectPoints,
                        stereoparams->camera[0]->distortion,
                        stereoparams->camera[0]->matrix,
                        transVects1,
                        rotMatrs1,
                        1);

    /* Calibrate second camera */
    cvCalibrateCamera(  numImages,
                        nums,
                        imageSize,
                        imagePoints2,
                        objectPoints,
                        stereoparams->camera[1]->distortion,
                        stereoparams->camera[1]->matrix,
                        transVects2,
                        rotMatrs2,
                        1);

    /* Cameras are calibrated */

    stereoparams->camera[0]->imgSize[0] = (float)imageSize.width;
    stereoparams->camera[0]->imgSize[1] = (float)imageSize.height;

    stereoparams->camera[1]->imgSize[0] = (float)imageSize.width;
    stereoparams->camera[1]->imgSize[1] = (float)imageSize.height;

    icvSelectBestRt(    numImages,
                        nums,
                        imagePoints1,
                        imagePoints2,
                        objectPoints,
                        stereoparams->camera[0]->matrix,
                        stereoparams->camera[0]->distortion,
                        rotMatrs1,
                        transVects1,
                        stereoparams->camera[1]->matrix,
                        stereoparams->camera[1]->distortion,
                        rotMatrs2,
                        transVects2,
                        stereoparams->rotMatrix,
                        stereoparams->transVector
                        );

    /* Free memory */
    free(transVects1);
    free(transVects2);
    free(rotMatrs1);
    free(rotMatrs2);

    icvComputeRestStereoParams(stereoparams);

    return CV_NO_ERR;
}

#if 0
/* Find line from epipole */
static void FindLine(CvPoint2D32f epipole,CvSize imageSize,CvPoint2D32f point,CvPoint2D32f *start,CvPoint2D32f *end)
{
    CvPoint2D32f frameBeg;
    CvPoint2D32f frameEnd;
    CvPoint2D32f cross[4];
    int     haveCross[4];
    float   dist;

    haveCross[0] = 0;
    haveCross[1] = 0;
    haveCross[2] = 0;
    haveCross[3] = 0;

    frameBeg.x = 0;
    frameBeg.y = 0;
    frameEnd.x = (float)(imageSize.width);
    frameEnd.y = 0;
    haveCross[0] = icvGetCrossPieceVector(frameBeg,frameEnd,epipole,point,&cross[0]);

    frameBeg.x = (float)(imageSize.width);
    frameBeg.y = 0;
    frameEnd.x = (float)(imageSize.width);
    frameEnd.y = (float)(imageSize.height);
    haveCross[1] = icvGetCrossPieceVector(frameBeg,frameEnd,epipole,point,&cross[1]);

    frameBeg.x = (float)(imageSize.width);
    frameBeg.y = (float)(imageSize.height);
    frameEnd.x = 0;
    frameEnd.y = (float)(imageSize.height);
    haveCross[2] = icvGetCrossPieceVector(frameBeg,frameEnd,epipole,point,&cross[2]);

    frameBeg.x = 0;
    frameBeg.y = (float)(imageSize.height);
    frameEnd.x = 0;
    frameEnd.y = 0;
    haveCross[3] = icvGetCrossPieceVector(frameBeg,frameEnd,epipole,point,&cross[3]);

    int n;
    float minDist = (float)(INT_MAX);
    float maxDist = (float)(INT_MIN);

    int maxN = -1;
    int minN = -1;

    for( n = 0; n < 4; n++ )
    {
        if( haveCross[n] > 0 )
        {
            dist =  (epipole.x - cross[n].x)*(epipole.x - cross[n].x) +
                    (epipole.y - cross[n].y)*(epipole.y - cross[n].y);

            if( dist < minDist )
            {
                minDist = dist;
                minN = n;
            }

            if( dist > maxDist )
            {
                maxDist = dist;
                maxN = n;
            }
        }
    }

    if( minN >= 0 && maxN >= 0 && (minN != maxN) )
    {
        *start = cross[minN];
        *end   = cross[maxN];
    }
    else
    {
        start->x = 0;
        start->y = 0;
        end->x = 0;
        end->y = 0;
    }

    return;
}

/* Find line which cross frame by line(a,b,c) */
static void FindLineForEpiline(CvSize imageSize,float a,float b,float c,CvPoint2D32f *start,CvPoint2D32f *end)
{
    CvPoint2D32f frameBeg;
    CvPoint2D32f frameEnd;
    CvPoint2D32f cross[4];
    int     haveCross[4];
    float   dist;

    haveCross[0] = 0;
    haveCross[1] = 0;
    haveCross[2] = 0;
    haveCross[3] = 0;

    frameBeg.x = 0;
    frameBeg.y = 0;
    frameEnd.x = (float)(imageSize.width);
    frameEnd.y = 0;
    haveCross[0] = icvGetCrossLineDirect(frameBeg,frameEnd,a,b,c,&cross[0]);

    frameBeg.x = (float)(imageSize.width);
    frameBeg.y = 0;
    frameEnd.x = (float)(imageSize.width);
    frameEnd.y = (float)(imageSize.height);
    haveCross[1] = icvGetCrossLineDirect(frameBeg,frameEnd,a,b,c,&cross[1]);

    frameBeg.x = (float)(imageSize.width);
    frameBeg.y = (float)(imageSize.height);
    frameEnd.x = 0;
    frameEnd.y = (float)(imageSize.height);
    haveCross[2] = icvGetCrossLineDirect(frameBeg,frameEnd,a,b,c,&cross[2]);

    frameBeg.x = 0;
    frameBeg.y = (float)(imageSize.height);
    frameEnd.x = 0;
    frameEnd.y = 0;
    haveCross[3] = icvGetCrossLineDirect(frameBeg,frameEnd,a,b,c,&cross[3]);

    int n;
    float minDist = (float)(INT_MAX);
    float maxDist = (float)(INT_MIN);

    int maxN = -1;
    int minN = -1;

    double midPointX = imageSize.width  / 2.0;
    double midPointY = imageSize.height / 2.0;

    for( n = 0; n < 4; n++ )
    {
        if( haveCross[n] > 0 )
        {
            dist =  (float)((midPointX - cross[n].x)*(midPointX - cross[n].x) +
                            (midPointY - cross[n].y)*(midPointY - cross[n].y));

            if( dist < minDist )
            {
                minDist = dist;
                minN = n;
            }

            if( dist > maxDist )
            {
                maxDist = dist;
                maxN = n;
            }
        }
    }

    if( minN >= 0 && maxN >= 0 && (minN != maxN) )
    {
        *start = cross[minN];
        *end   = cross[maxN];
    }
    else
    {
        start->x = 0;
        start->y = 0;
        end->x = 0;
        end->y = 0;
    }

    return;

}

/* Cross lines */
static int GetCrossLines(CvPoint2D32f p1_start,CvPoint2D32f p1_end,CvPoint2D32f p2_start,CvPoint2D32f p2_end,CvPoint2D32f *cross)
{
    double ex1,ey1,ex2,ey2;
    double px1,py1,px2,py2;
    double del;
    double delA,delB,delX,delY;
    double alpha,betta;

    ex1 = p1_start.x;
    ey1 = p1_start.y;
    ex2 = p1_end.x;
    ey2 = p1_end.y;

    px1 = p2_start.x;
    py1 = p2_start.y;
    px2 = p2_end.x;
    py2 = p2_end.y;

    del = (ex1-ex2)*(py2-py1)+(ey2-ey1)*(px2-px1);
    if( del == 0)
    {
        return -1;
    }

    delA =  (px1-ex1)*(py1-py2) + (ey1-py1)*(px1-px2);
    delB =  (ex1-px1)*(ey1-ey2) + (py1-ey1)*(ex1-ex2);

    alpha =  delA / del;
    betta = -delB / del;

    if( alpha < 0 || alpha > 1.0 || betta < 0 || betta > 1.0)
    {
        return -1;
    }

    delX =  (ex1-ex2)*(py1*(px1-px2)-px1*(py1-py2))+
            (px1-px2)*(ex1*(ey1-ey2)-ey1*(ex1-ex2));

    delY =  (ey1-ey2)*(px1*(py1-py2)-py1*(px1-px2))+
            (py1-py2)*(ey1*(ex1-ex2)-ex1*(ey1-ey2));

    cross->x = (float)( delX / del);
    cross->y = (float)(-delY / del);
    return 1;
}
#endif

int icvGetCrossPieceVector(CvPoint2D32f p1_start,CvPoint2D32f p1_end,CvPoint2D32f v2_start,CvPoint2D32f v2_end,CvPoint2D32f *cross)
{
    double ex1 = p1_start.x;
    double ey1 = p1_start.y;
    double ex2 = p1_end.x;
    double ey2 = p1_end.y;

    double px1 = v2_start.x;
    double py1 = v2_start.y;
    double px2 = v2_end.x;
    double py2 = v2_end.y;

    double del = (ex1-ex2)*(py2-py1)+(ey2-ey1)*(px2-px1);
    if( del == 0)
    {
        return -1;
    }

    double delA =  (px1-ex1)*(py1-py2) + (ey1-py1)*(px1-px2);
    //double delB =  (ex1-px1)*(ey1-ey2) + (py1-ey1)*(ex1-ex2);

    double alpha =  delA / del;
    //double betta = -delB / del;

    if( alpha < 0 || alpha > 1.0 )
    {
        return -1;
    }

    double delX =  (ex1-ex2)*(py1*(px1-px2)-px1*(py1-py2))+
            (px1-px2)*(ex1*(ey1-ey2)-ey1*(ex1-ex2));

    double delY =  (ey1-ey2)*(px1*(py1-py2)-py1*(px1-px2))+
            (py1-py2)*(ey1*(ex1-ex2)-ex1*(ey1-ey2));

    cross->x = (float)( delX / del);
    cross->y = (float)(-delY / del);
    return 1;
}


int icvGetCrossLineDirect(CvPoint2D32f p1,CvPoint2D32f p2,float a,float b,float c,CvPoint2D32f* cross)
{
    double del;
    double delX,delY,delA;

    double px1,px2,py1,py2;
    double X,Y,alpha;

    px1 = p1.x;
    py1 = p1.y;

    px2 = p2.x;
    py2 = p2.y;

    del = a * (px2 - px1) + b * (py2-py1);
    if( del == 0 )
    {
        return -1;
    }

    delA = - c - a*px1 - b*py1;
    alpha = delA / del;

    if( alpha < 0 || alpha > 1.0 )
    {
        return -1;/* no cross */
    }

    delX = b * (py1*(px1-px2) - px1*(py1-py2)) + c * (px1-px2);
    delY = a * (px1*(py1-py2) - py1*(px1-px2)) + c * (py1-py2);

    X = delX / del;
    Y = delY / del;

    cross->x = (float)X;
    cross->y = (float)Y;

    return 1;
}

#if 0
static int cvComputeEpipoles( CvMatr32f camMatr1,  CvMatr32f camMatr2,
                            CvMatr32f rotMatr1,  CvMatr32f rotMatr2,
                            CvVect32f transVect1,CvVect32f transVect2,
                            CvVect32f epipole1,
                            CvVect32f epipole2)
{

    /* Copy matrix */

    CvMat ccamMatr1 = cvMat(3,3,CV_MAT32F,camMatr1);
    CvMat ccamMatr2 = cvMat(3,3,CV_MAT32F,camMatr2);
    CvMat crotMatr1 = cvMat(3,3,CV_MAT32F,rotMatr1);
    CvMat crotMatr2 = cvMat(3,3,CV_MAT32F,rotMatr2);
    CvMat ctransVect1 = cvMat(3,1,CV_MAT32F,transVect1);
    CvMat ctransVect2 = cvMat(3,1,CV_MAT32F,transVect2);
    CvMat cepipole1 = cvMat(3,1,CV_MAT32F,epipole1);
    CvMat cepipole2 = cvMat(3,1,CV_MAT32F,epipole2);


    CvMat cmatrP1   = cvMat(3,3,CV_MAT32F,0); cvmAlloc(&cmatrP1);
    CvMat cmatrP2   = cvMat(3,3,CV_MAT32F,0); cvmAlloc(&cmatrP2);
    CvMat cvectp1   = cvMat(3,1,CV_MAT32F,0); cvmAlloc(&cvectp1);
    CvMat cvectp2   = cvMat(3,1,CV_MAT32F,0); cvmAlloc(&cvectp2);
    CvMat ctmpF1    = cvMat(3,1,CV_MAT32F,0); cvmAlloc(&ctmpF1);
    CvMat ctmpM1    = cvMat(3,3,CV_MAT32F,0); cvmAlloc(&ctmpM1);
    CvMat ctmpM2    = cvMat(3,3,CV_MAT32F,0); cvmAlloc(&ctmpM2);
    CvMat cinvP1    = cvMat(3,3,CV_MAT32F,0); cvmAlloc(&cinvP1);
    CvMat cinvP2    = cvMat(3,3,CV_MAT32F,0); cvmAlloc(&cinvP2);
    CvMat ctmpMatr  = cvMat(3,3,CV_MAT32F,0); cvmAlloc(&ctmpMatr);
    CvMat ctmpVect1 = cvMat(3,1,CV_MAT32F,0); cvmAlloc(&ctmpVect1);
    CvMat ctmpVect2 = cvMat(3,1,CV_MAT32F,0); cvmAlloc(&ctmpVect2);
    CvMat cmatrF1   = cvMat(3,3,CV_MAT32F,0); cvmAlloc(&cmatrF1);
    CvMat ctmpF     = cvMat(3,3,CV_MAT32F,0); cvmAlloc(&ctmpF);
    CvMat ctmpE1    = cvMat(3,1,CV_MAT32F,0); cvmAlloc(&ctmpE1);
    CvMat ctmpE2    = cvMat(3,1,CV_MAT32F,0); cvmAlloc(&ctmpE2);

    /* Compute first */
    cvmMul( &ccamMatr1, &crotMatr1, &cmatrP1);
    cvmInvert( &cmatrP1,&cinvP1 );
    cvmMul( &ccamMatr1, &ctransVect1, &cvectp1 );

    /* Compute second */
    cvmMul( &ccamMatr2, &crotMatr2, &cmatrP2 );
    cvmInvert( &cmatrP2,&cinvP2 );
    cvmMul( &ccamMatr2, &ctransVect2, &cvectp2 );

    cvmMul( &cmatrP1, &cinvP2, &ctmpM1);
    cvmMul( &ctmpM1, &cvectp2, &ctmpVect1);
    cvmSub( &cvectp1,&ctmpVect1,&ctmpE1);

    cvmMul( &cmatrP2, &cinvP1, &ctmpM2);
    cvmMul( &ctmpM2, &cvectp1, &ctmpVect2);
    cvmSub( &cvectp2, &ctmpVect2, &ctmpE2);


    /* Need scale */

    cvmScale(&ctmpE1,&cepipole1,1.0);
    cvmScale(&ctmpE2,&cepipole2,1.0);

    cvmFree(&cmatrP1);
    cvmFree(&cmatrP1);
    cvmFree(&cvectp1);
    cvmFree(&cvectp2);
    cvmFree(&ctmpF1);
    cvmFree(&ctmpM1);
    cvmFree(&ctmpM2);
    cvmFree(&cinvP1);
    cvmFree(&cinvP2);
    cvmFree(&ctmpMatr);
    cvmFree(&ctmpVect1);
    cvmFree(&ctmpVect2);
    cvmFree(&cmatrF1);
    cvmFree(&ctmpF);
    cvmFree(&ctmpE1);
    cvmFree(&ctmpE2);

    return CV_NO_ERR;
}/* cvComputeEpipoles */
#endif

/* Compute epipoles for fundamental matrix */
int cvComputeEpipolesFromFundMatrix(CvMatr32f fundMatr,
                                         CvPoint3D32f* epipole1,
                                         CvPoint3D32f* epipole2)
{
    /* Decompose fundamental matrix using SVD ( A = U W V') */
    CvMat fundMatrC = cvMat(3,3,CV_MAT32F,fundMatr);

    CvMat* matrW = cvCreateMat(3,3,CV_MAT32F);
    CvMat* matrU = cvCreateMat(3,3,CV_MAT32F);
    CvMat* matrV = cvCreateMat(3,3,CV_MAT32F);

    /* From svd we need just last vector of U and V or last row from U' and V' */
    /* We get transposed matrixes U and V */
    cvSVD(&fundMatrC,matrW,matrU,matrV,CV_SVD_V_T|CV_SVD_U_T);

    /* Get last row from U' and compute epipole1 */
    epipole1->x = matrU->data.fl[6];
    epipole1->y = matrU->data.fl[7];
    epipole1->z = matrU->data.fl[8];

    /* Get last row from V' and compute epipole2 */
    epipole2->x = matrV->data.fl[6];
    epipole2->y = matrV->data.fl[7];
    epipole2->z = matrV->data.fl[8];

    cvReleaseMat(&matrW);
    cvReleaseMat(&matrU);
    cvReleaseMat(&matrV);
    return CV_OK;
}

int cvConvertEssential2Fundamental( CvMatr32f essMatr,
                                         CvMatr32f fundMatr,
                                         CvMatr32f cameraMatr1,
                                         CvMatr32f cameraMatr2)
{/* Fund = inv(CM1') * Ess * inv(CM2) */

    CvMat essMatrC     = cvMat(3,3,CV_MAT32F,essMatr);
    CvMat fundMatrC    = cvMat(3,3,CV_MAT32F,fundMatr);
    CvMat cameraMatr1C = cvMat(3,3,CV_MAT32F,cameraMatr1);
    CvMat cameraMatr2C = cvMat(3,3,CV_MAT32F,cameraMatr2);

    CvMat* invCM2  = cvCreateMat(3,3,CV_MAT32F);
    CvMat* tmpMatr = cvCreateMat(3,3,CV_MAT32F);
    CvMat* invCM1T = cvCreateMat(3,3,CV_MAT32F);

    cvTranspose(&cameraMatr1C,tmpMatr);
    cvInvert(tmpMatr,invCM1T);
    cvmMul(invCM1T,&essMatrC,tmpMatr);
    cvInvert(&cameraMatr2C,invCM2);
    cvmMul(tmpMatr,invCM2,&fundMatrC);

    /* Scale fundamental matrix */
    double scale;
    scale = 1.0/fundMatrC.data.fl[8];
    cvConvertScale(&fundMatrC,&fundMatrC,scale);

    cvReleaseMat(&invCM2);
    cvReleaseMat(&tmpMatr);
    cvReleaseMat(&invCM1T);

    return CV_OK;
}

/* Compute essential matrix */

int cvComputeEssentialMatrix(  CvMatr32f rotMatr,
                                    CvMatr32f transVect,
                                    CvMatr32f essMatr)
{
    float transMatr[9];

    /* Make antisymmetric matrix from transpose vector */
    transMatr[0] =   0;
    transMatr[1] = - transVect[2];
    transMatr[2] =   transVect[1];

    transMatr[3] =   transVect[2];
    transMatr[4] =   0;
    transMatr[5] = - transVect[0];

    transMatr[6] = - transVect[1];
    transMatr[7] =   transVect[0];
    transMatr[8] =   0;

    icvMulMatrix_32f(transMatr,3,3,rotMatr,3,3,essMatr);

    return CV_OK;
}
