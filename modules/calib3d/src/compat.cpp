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
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

/************************************************************************************\
       Some backward compatibility stuff, to be moved to legacy or compat module
\************************************************************************************/

using cv::Ptr;

////////////////// Levenberg-Marquardt engine (the old variant) ////////////////////////

CvLevMarq::CvLevMarq()
{
    mask = prevParam = param = J = err = JtJ = JtJN = JtErr = JtJV = JtJW = Ptr<CvMat>();
    lambdaLg10 = 0; state = DONE;
    criteria = cvTermCriteria(0,0,0);
    iters = 0;
    completeSymmFlag = false;
}

CvLevMarq::CvLevMarq( int nparams, int nerrs, CvTermCriteria criteria0, bool _completeSymmFlag )
{
    mask = prevParam = param = J = err = JtJ = JtJN = JtErr = JtJV = JtJW = Ptr<CvMat>();
    init(nparams, nerrs, criteria0, _completeSymmFlag);
}

void CvLevMarq::clear()
{
    mask.release();
    prevParam.release();
    param.release();
    J.release();
    err.release();
    JtJ.release();
    JtJN.release();
    JtErr.release();
    JtJV.release();
    JtJW.release();
}

CvLevMarq::~CvLevMarq()
{
    clear();
}

void CvLevMarq::init( int nparams, int nerrs, CvTermCriteria criteria0, bool _completeSymmFlag )
{
    if( !param || param->rows != nparams || nerrs != (err ? err->rows : 0) )
        clear();
    mask = cvCreateMat( nparams, 1, CV_8U );
    cvSet(mask, cvScalarAll(1));
    prevParam = cvCreateMat( nparams, 1, CV_64F );
    param = cvCreateMat( nparams, 1, CV_64F );
    JtJ = cvCreateMat( nparams, nparams, CV_64F );
    JtJN = cvCreateMat( nparams, nparams, CV_64F );
    JtJV = cvCreateMat( nparams, nparams, CV_64F );
    JtJW = cvCreateMat( nparams, 1, CV_64F );
    JtErr = cvCreateMat( nparams, 1, CV_64F );
    if( nerrs > 0 )
    {
        J = cvCreateMat( nerrs, nparams, CV_64F );
        err = cvCreateMat( nerrs, 1, CV_64F );
    }
    prevErrNorm = DBL_MAX;
    lambdaLg10 = -3;
    criteria = criteria0;
    if( criteria.type & CV_TERMCRIT_ITER )
        criteria.max_iter = MIN(MAX(criteria.max_iter,1),1000);
    else
        criteria.max_iter = 30;
    if( criteria.type & CV_TERMCRIT_EPS )
        criteria.epsilon = MAX(criteria.epsilon, 0);
    else
        criteria.epsilon = DBL_EPSILON;
    state = STARTED;
    iters = 0;
    completeSymmFlag = _completeSymmFlag;
}

bool CvLevMarq::update( const CvMat*& _param, CvMat*& matJ, CvMat*& _err )
{
    double change;

    matJ = _err = 0;

    assert( !err.empty() );
    if( state == DONE )
    {
        _param = param;
        return false;
    }

    if( state == STARTED )
    {
        _param = param;
        cvZero( J );
        cvZero( err );
        matJ = J;
        _err = err;
        state = CALC_J;
        return true;
    }

    if( state == CALC_J )
    {
        cvMulTransposed( J, JtJ, 1 );
        cvGEMM( J, err, 1, 0, 0, JtErr, CV_GEMM_A_T );
        cvCopy( param, prevParam );
        step();
        if( iters == 0 )
            prevErrNorm = cvNorm(err, 0, CV_L2);
        _param = param;
        cvZero( err );
        _err = err;
        state = CHECK_ERR;
        return true;
    }

    assert( state == CHECK_ERR );
    errNorm = cvNorm( err, 0, CV_L2 );
    if( errNorm > prevErrNorm )
    {
        if( ++lambdaLg10 <= 16 )
        {
            step();
            _param = param;
            cvZero( err );
            _err = err;
            state = CHECK_ERR;
            return true;
        }
    }

    lambdaLg10 = MAX(lambdaLg10-1, -16);
    if( ++iters >= criteria.max_iter ||
        (change = cvNorm(param, prevParam, CV_RELATIVE_L2)) < criteria.epsilon )
    {
        _param = param;
        state = DONE;
        return true;
    }

    prevErrNorm = errNorm;
    _param = param;
    cvZero(J);
    matJ = J;
    _err = err;
    state = CALC_J;
    return true;
}


bool CvLevMarq::updateAlt( const CvMat*& _param, CvMat*& _JtJ, CvMat*& _JtErr, double*& _errNorm )
{
    double change;

    CV_Assert( err.empty() );
    if( state == DONE )
    {
        _param = param;
        return false;
    }

    if( state == STARTED )
    {
        _param = param;
        cvZero( JtJ );
        cvZero( JtErr );
        errNorm = 0;
        _JtJ = JtJ;
        _JtErr = JtErr;
        _errNorm = &errNorm;
        state = CALC_J;
        return true;
    }

    if( state == CALC_J )
    {
        cvCopy( param, prevParam );
        step();
        _param = param;
        prevErrNorm = errNorm;
        errNorm = 0;
        _errNorm = &errNorm;
        state = CHECK_ERR;
        return true;
    }

    assert( state == CHECK_ERR );
    if( errNorm > prevErrNorm )
    {
        if( ++lambdaLg10 <= 16 )
        {
            step();
            _param = param;
            errNorm = 0;
            _errNorm = &errNorm;
            state = CHECK_ERR;
            return true;
        }
    }

    lambdaLg10 = MAX(lambdaLg10-1, -16);
    if( ++iters >= criteria.max_iter ||
        (change = cvNorm(param, prevParam, CV_RELATIVE_L2)) < criteria.epsilon )
    {
        _param = param;
        state = DONE;
        return false;
    }

    prevErrNorm = errNorm;
    cvZero( JtJ );
    cvZero( JtErr );
    _param = param;
    _JtJ = JtJ;
    _JtErr = JtErr;
    state = CALC_J;
    return true;
}

void CvLevMarq::step()
{
    const double LOG10 = log(10.);
    double lambda = exp(lambdaLg10*LOG10);
    int i, j, nparams = param->rows;

    for( i = 0; i < nparams; i++ )
        if( mask->data.ptr[i] == 0 )
        {
            double *row = JtJ->data.db + i*nparams, *col = JtJ->data.db + i;
            for( j = 0; j < nparams; j++ )
                row[j] = col[j*nparams] = 0;
            JtErr->data.db[i] = 0;
        }

    if( !err )
        cvCompleteSymm( JtJ, completeSymmFlag );
#if 1
    cvCopy( JtJ, JtJN );
    for( i = 0; i < nparams; i++ )
        JtJN->data.db[(nparams+1)*i] *= 1. + lambda;
#else
    cvSetIdentity(JtJN, cvRealScalar(lambda));
    cvAdd( JtJ, JtJN, JtJN );
#endif
    cvSVD( JtJN, JtJW, 0, JtJV, CV_SVD_MODIFY_A + CV_SVD_U_T + CV_SVD_V_T );
    cvSVBkSb( JtJW, JtJV, JtJV, JtErr, param, CV_SVD_U_T + CV_SVD_V_T );
    for( i = 0; i < nparams; i++ )
        param->data.db[i] = prevParam->data.db[i] - (mask->data.ptr[i] ? param->data.db[i] : 0);
}


CV_IMPL int cvRANSACUpdateNumIters( double p, double ep, int modelPoints, int maxIters )
{
    return cv::RANSACUpdateNumIters(p, ep, modelPoints, maxIters);
}

