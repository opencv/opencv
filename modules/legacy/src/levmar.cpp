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

//#define TRACKLEVMAR

typedef void (*pointer_LMJac)( const CvMat* src, CvMat* dst );
typedef void (*pointer_LMFunc)( const CvMat* src, CvMat* dst );

/* Optimization using Levenberg-Marquardt */
void cvLevenbergMarquardtOptimization(pointer_LMJac JacobianFunction,
                                    pointer_LMFunc function,
                                    /*pointer_Err error_function,*/
                                    CvMat *X0,CvMat *observRes,CvMat *resultX,
                                    int maxIter,double epsilon)
{
    /* This is not sparce method */
    /* Make optimization using  */
    /* func - function to compute */
    /* uses function to compute jacobian */

    /* Allocate memory */
    CvMat *vectX = 0;
    CvMat *vectNewX = 0;
    CvMat *resFunc = 0;
    CvMat *resNewFunc = 0;
    CvMat *error = 0;
    CvMat *errorNew = 0;
    CvMat *Jac = 0;
    CvMat *delta = 0;
    CvMat *matrJtJ = 0;
    CvMat *matrJtJN = 0;
    CvMat *matrJt = 0;
    CvMat *vectB = 0;
   
    CV_FUNCNAME( "cvLevenbegrMarquardtOptimization" );
    __BEGIN__;


    if( JacobianFunction == 0 || function == 0 || X0 == 0 || observRes == 0 || resultX == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(X0) || !CV_IS_MAT(observRes) || !CV_IS_MAT(resultX) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Some of input parameters must be a matrices" );
    }


    int numVal;
    int numFunc;
    double valError;
    double valNewError;

    numVal = X0->rows;
    numFunc = observRes->rows;

    /* test input data */
    if( X0->cols != 1 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of colomn of vector X0 must be 1" );
    }
    
    if( observRes->cols != 1 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of colomn of vector observed rusult must be 1" );
    }

    if( resultX->cols != 1 || resultX->rows != numVal )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of result vector X must be equals to X0" );
    }

    if( maxIter <= 0  )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of maximum iteration must be > 0" );
    }

    if( epsilon < 0 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Epsilon must be >= 0" );
    }

    /* copy x0 to current value of x */
    CV_CALL( vectX      = cvCreateMat(numVal, 1,      CV_64F) );
    CV_CALL( vectNewX   = cvCreateMat(numVal, 1,      CV_64F) );
    CV_CALL( resFunc    = cvCreateMat(numFunc,1,      CV_64F) );
    CV_CALL( resNewFunc = cvCreateMat(numFunc,1,      CV_64F) );
    CV_CALL( error      = cvCreateMat(numFunc,1,      CV_64F) );
    CV_CALL( errorNew   = cvCreateMat(numFunc,1,      CV_64F) );
    CV_CALL( Jac        = cvCreateMat(numFunc,numVal, CV_64F) );
    CV_CALL( delta      = cvCreateMat(numVal, 1,      CV_64F) );
    CV_CALL( matrJtJ    = cvCreateMat(numVal, numVal, CV_64F) );
    CV_CALL( matrJtJN   = cvCreateMat(numVal, numVal, CV_64F) );
    CV_CALL( matrJt     = cvCreateMat(numVal, numFunc,CV_64F) );
    CV_CALL( vectB      = cvCreateMat(numVal, 1,      CV_64F) );

    cvCopy(X0,vectX);

    /* ========== Main optimization loop ============ */
    double change;
    int currIter;
    double alpha;

    change = 1;
    currIter = 0;
    alpha = 0.001;

    do {

        /* Compute value of function */
        function(vectX,resFunc);
        /* Print result of function to file */

        /* Compute error */
        cvSub(observRes,resFunc,error);        
        
        //valError = error_function(observRes,resFunc);
        /* Need to use new version of computing error (norm) */
        valError = cvNorm(observRes,resFunc);

        /* Compute Jacobian for given point vectX */
        JacobianFunction(vectX,Jac);

        /* Define optimal delta for J'*J*delta=J'*error */
        /* compute J'J */
        cvMulTransposed(Jac,matrJtJ,1);
        
        cvCopy(matrJtJ,matrJtJN);

        /* compute J'*error */
        cvTranspose(Jac,matrJt);
        cvmMul(matrJt,error,vectB);


        /* Solve normal equation for given alpha and Jacobian */
        do
        {
            /* Increase diagonal elements by alpha */
            for( int i = 0; i < numVal; i++ )
            {
                double val;
                val = cvmGet(matrJtJ,i,i);
                cvmSet(matrJtJN,i,i,(1+alpha)*val);
            }

            /* Solve system to define delta */
            cvSolve(matrJtJN,vectB,delta,CV_SVD);

            /* We know delta and we can define new value of vector X */
            cvAdd(vectX,delta,vectNewX);

            /* Compute result of function for new vector X */
            function(vectNewX,resNewFunc);
            cvSub(observRes,resNewFunc,errorNew);

            valNewError = cvNorm(observRes,resNewFunc);

            currIter++;

            if( valNewError < valError )
            {/* accept new value */
                valError = valNewError;

                /* Compute relative change of required parameter vectorX. change = norm(curr-prev) / norm(curr) )  */
                change = cvNorm(vectX, vectNewX, CV_RELATIVE_L2);

                alpha /= 10;
                cvCopy(vectNewX,vectX);
                break;
            }
            else
            {
                alpha *= 10;
            }

        } while ( currIter < maxIter  );
        /* new value of X and alpha were accepted */

    } while ( change > epsilon && currIter < maxIter );


    /* result was computed */
    cvCopy(vectX,resultX);

    __END__;

    cvReleaseMat(&vectX);
    cvReleaseMat(&vectNewX);
    cvReleaseMat(&resFunc);
    cvReleaseMat(&resNewFunc);
    cvReleaseMat(&error);
    cvReleaseMat(&errorNew);
    cvReleaseMat(&Jac);
    cvReleaseMat(&delta);
    cvReleaseMat(&matrJtJ);
    cvReleaseMat(&matrJtJN);
    cvReleaseMat(&matrJt);
    cvReleaseMat(&vectB);

    return;
}

/*------------------------------------------------------------------------------*/
#if 0
//tests
void Jac_Func2(CvMat *vectX,CvMat *Jac)
{
    double x = cvmGet(vectX,0,0);
    double y = cvmGet(vectX,1,0);
    cvmSet(Jac,0,0,2*(x-2));
    cvmSet(Jac,0,1,2*(y+3));

    cvmSet(Jac,1,0,1);
    cvmSet(Jac,1,1,1);
    return;
}

void Res_Func2(CvMat *vectX,CvMat *res)
{
    double x = cvmGet(vectX,0,0);
    double y = cvmGet(vectX,1,0);
    cvmSet(res,0,0,(x-2)*(x-2)+(y+3)*(y+3));
    cvmSet(res,1,0,x+y);

    return;
}


double Err_Func2(CvMat *obs,CvMat *res)
{
    CvMat *tmp;
    tmp = cvCreateMat(obs->rows,1,CV_64F);
    cvSub(obs,res,tmp);

    double e;
    e = cvNorm(tmp);

    return e;
}


void TestOptimX2Y2()
{
    CvMat vectX0;
    double vectX0_dat[2];
    vectX0 = cvMat(2,1,CV_64F,vectX0_dat);
    vectX0_dat[0] = 5;
    vectX0_dat[1] = -7;

    CvMat observRes;
    double observRes_dat[2];
    observRes = cvMat(2,1,CV_64F,observRes_dat);
    observRes_dat[0] = 0;
    observRes_dat[1] = -1;
    observRes_dat[0] = 0;
    observRes_dat[1] = -1.2;

    CvMat optimX;
    double optimX_dat[2];
    optimX = cvMat(2,1,CV_64F,optimX_dat);


    LevenbegrMarquardtOptimization( Jac_Func2, Res_Func2, Err_Func2,
                                    &vectX0,&observRes,&optimX,100,0.000001);

    return;

}

#endif



