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

#include "cvtest.h"

int cvTsRodrigues( const CvMat* src, CvMat* dst, CvMat* jacobian )
{
    int depth;
    int i;
    float Jf[27];
    double J[27];
    CvMat _Jf, matJ = cvMat( 3, 9, CV_64F, J );

    depth = CV_MAT_DEPTH(src->type);

    if( jacobian )
    {
        assert( (jacobian->rows == 9 && jacobian->cols == 3) ||
                (jacobian->rows == 3 && jacobian->cols == 9) );
    }

    if( src->cols == 1 || src->rows == 1 )
    {
        double r[3], theta;
        CvMat _r = cvMat( src->rows, src->cols, CV_MAKETYPE(CV_64F,CV_MAT_CN(src->type)), r);

        assert( dst->rows == 3 && dst->cols == 3 );

        cvConvert( src, &_r );

        theta = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
        if( theta < DBL_EPSILON )
        {
            cvSetIdentity( dst );

            if( jacobian )
            {
                memset( J, 0, sizeof(J) );
                J[5] = J[15] = J[19] = 1;
                J[7] = J[11] = J[21] = -1;
            }
        }
        else
        {
            // omega = r/theta (~[w1, w2, w3])
            double itheta = 1./theta;
            double w1 = r[0]*itheta, w2 = r[1]*itheta, w3 = r[2]*itheta;
            double alpha = cos(theta);
            double beta = sin(theta);
            double gamma = 1 - alpha;
            double omegav[] =
            {
                0, -w3, w2,
                w3, 0, -w1,
                -w2, w1, 0
            };
            double A[] =
            {
                w1*w1, w1*w2, w1*w3,
                w2*w1, w2*w2, w2*w3,
                w3*w1, w3*w2, w3*w3
            };
            double R[9];
            CvMat _omegav = cvMat(3, 3, CV_64F, omegav);
            CvMat matA = cvMat(3, 3, CV_64F, A);
            CvMat matR = cvMat(3, 3, CV_64F, R);

            cvSetIdentity( &matR, cvRealScalar(alpha) );
            cvScaleAdd( &_omegav, cvRealScalar(beta), &matR, &matR );
            cvScaleAdd( &matA, cvRealScalar(gamma), &matR, &matR );
            cvConvert( &matR, dst );

            if( jacobian )
            {
                // m3 = [r, theta]
                double dm3din[] =
                {
                    1, 0, 0,
                    0, 1, 0,
                    0, 0, 1,
                    w1, w2, w3
                };
                // m2 = [omega, theta]
                double dm2dm3[] =
                {
                    itheta, 0, 0, -w1*itheta,
                    0, itheta, 0, -w2*itheta,
                    0, 0, itheta, -w3*itheta,
                    0, 0, 0, 1
                };
                double t0[9*4];
                double dm1dm2[21*4];
                double dRdm1[9*21];
                CvMat _dm3din = cvMat( 4, 3, CV_64FC1, dm3din );
                CvMat _dm2dm3 = cvMat( 4, 4, CV_64FC1, dm2dm3 );
                CvMat _dm1dm2 = cvMat( 21, 4, CV_64FC1, dm1dm2 );
                CvMat _dRdm1 = cvMat( 9, 21, CV_64FC1, dRdm1 );
                CvMat _dRdm1_part;
                CvMat _t0 = cvMat( 9, 4, CV_64FC1, t0 );
                CvMat _t1 = cvMat( 9, 4, CV_64FC1, dRdm1 );

                // m1 = [alpha, beta, gamma, omegav; A]
                memset( dm1dm2, 0, sizeof(dm1dm2) );
                dm1dm2[3] = -beta;
                dm1dm2[7] = alpha;
                dm1dm2[11] = beta;

                // dm1dm2(4:12,1:3) = [0 0 0 0 0 1 0 -1 0;
                //                     0 0 -1 0 0 0 1 0 0;
                //                     0 1 0 -1 0 0 0 0 0]'
                //                     -------------------
                //                     0 0 0  0 0 0 0 0 0
                dm1dm2[12 + 6] = dm1dm2[12 + 20] = dm1dm2[12 + 25] = 1;
                dm1dm2[12 + 9] = dm1dm2[12 + 14] = dm1dm2[12 + 28] = -1;

                double dm1dw[] =
                {
                    2*w1, w2, w3, w2, 0, 0, w3, 0, 0,
                    0, w1, 0, w1, 2*w2, w3, 0, w3, 0,
                    0, 0, w1, 0, 0, w2, w1, w2, 2*w3
                };

                CvMat _dm1dw = cvMat( 3, 9, CV_64FC1, dm1dw );
                CvMat _dm1dm2_part;

                cvGetSubRect( &_dm1dm2, &_dm1dm2_part, cvRect(0,12,3,9) );
                cvTranspose( &_dm1dw, &_dm1dm2_part );

                memset( dRdm1, 0, sizeof(dRdm1) );
                dRdm1[0*21] = dRdm1[4*21] = dRdm1[8*21] = 1;

                cvGetCol( &_dRdm1, &_dRdm1_part, 1 );
                cvTranspose( &_omegav, &_omegav );
                cvReshape( &_omegav, &_omegav, 1, 1 );
                cvTranspose( &_omegav, &_dRdm1_part );

                cvGetCol( &_dRdm1, &_dRdm1_part, 2 );
                cvReshape( &matA, &matA, 1, 1 );
                cvTranspose( &matA, &_dRdm1_part );

                cvGetSubRect( &_dRdm1, &_dRdm1_part, cvRect(3,0,9,9) );
                cvSetIdentity( &_dRdm1_part, cvScalarAll(beta) );

                cvGetSubRect( &_dRdm1, &_dRdm1_part, cvRect(12,0,9,9) );
                cvSetIdentity( &_dRdm1_part, cvScalarAll(gamma) );

                matJ = cvMat( 9, 3, CV_64FC1, J );

                cvMatMul( &_dRdm1, &_dm1dm2, &_t0 );
                cvMatMul( &_t0, &_dm2dm3, &_t1 );
                cvMatMul( &_t1, &_dm3din, &matJ );

                _t0 = cvMat( 3, 9, CV_64FC1, t0 );
                cvTranspose( &matJ, &_t0 );

                for( i = 0; i < 3; i++ )
                {
                    _t1 = cvMat( 3, 3, CV_64FC1, t0 + i*9 );
                    cvTranspose( &_t1, &_t1 );
                }

                cvTranspose( &_t0, &matJ );
            }
        }
    }
    else if( src->cols == 3 && src->rows == 3 )
    {
        double R[9], A[9], I[9], r[3], W[3], U[9], V[9];
        double tr, alpha, beta, theta;
        CvMat matR = cvMat( 3, 3, CV_64F, R );
        CvMat matA = cvMat( 3, 3, CV_64F, A );
        CvMat matI = cvMat( 3, 3, CV_64F, I );
        CvMat _r = cvMat( dst->rows, dst->cols, CV_MAKETYPE(CV_64F, CV_MAT_CN(dst->type)), r );
        CvMat matW = cvMat( 1, 3, CV_64F, W );
        CvMat matU = cvMat( 3, 3, CV_64F, U );
        CvMat matV = cvMat( 3, 3, CV_64F, V );

        cvConvert( src, &matR );
        cvSVD( &matR, &matW, &matU, &matV, CV_SVD_MODIFY_A + CV_SVD_U_T + CV_SVD_V_T );
        cvGEMM( &matU, &matV, 1, 0, 0, &matR, CV_GEMM_A_T );

        cvMulTransposed( &matR, &matA, 0 );
        cvSetIdentity( &matI );

        if( cvNorm( &matA, &matI, CV_C ) > 1e-3 ||
            fabs( cvDet(&matR) - 1 ) > 1e-3 )
            return 0;

        tr = (cvTrace(&matR).val[0] - 1.)*0.5;
        tr = tr > 1. ? 1. : tr < -1. ? -1. : tr;
        theta = acos(tr);
        alpha = cos(theta);
        beta = sin(theta);

        if( beta >= 1e-5 )
        {
            double dtheta_dtr = -1./sqrt(1 - tr*tr);
            double vth = 1/(2*beta);

            // om1 = [R(3,2) - R(2,3), R(1,3) - R(3,1), R(2,1) - R(1,2)]'
            double om1[] = { R[7] - R[5], R[2] - R[6], R[3] - R[1] };
            // om = om1*vth
            // r = om*theta
            double d3 = vth*theta;

            r[0] = om1[0]*d3; r[1] = om1[1]*d3; r[2] = om1[2]*d3;
            cvConvert( &_r, dst );

            if( jacobian )
            {
                // var1 = [vth;theta]
                // var = [om1;var1] = [om1;vth;theta]
                double dvth_dtheta = -vth*alpha/beta;
                double d1 = 0.5*dvth_dtheta*dtheta_dtr;
                double d2 = 0.5*dtheta_dtr;
                // dvar1/dR = dvar1/dtheta*dtheta/dR = [dvth/dtheta; 1] * dtheta/dtr * dtr/dR
                double dvardR[5*9] =
                {
                    0, 0, 0, 0, 0, 1, 0, -1, 0,
                    0, 0, -1, 0, 0, 0, 1, 0, 0,
                    0, 1, 0, -1, 0, 0, 0, 0, 0,
                    d1, 0, 0, 0, d1, 0, 0, 0, d1,
                    d2, 0, 0, 0, d2, 0, 0, 0, d2
                };
                // var2 = [om;theta]
                double dvar2dvar[] =
                {
                    vth, 0, 0, om1[0], 0,
                    0, vth, 0, om1[1], 0,
                    0, 0, vth, om1[2], 0,
                    0, 0, 0, 0, 1
                };
                double domegadvar2[] =
                {
                    theta, 0, 0, om1[0]*vth,
                    0, theta, 0, om1[1]*vth,
                    0, 0, theta, om1[2]*vth
                };

                CvMat _dvardR = cvMat( 5, 9, CV_64FC1, dvardR );
                CvMat _dvar2dvar = cvMat( 4, 5, CV_64FC1, dvar2dvar );
                CvMat _domegadvar2 = cvMat( 3, 4, CV_64FC1, domegadvar2 );
                double t0[3*5];
                CvMat _t0 = cvMat( 3, 5, CV_64FC1, t0 );

                cvMatMul( &_domegadvar2, &_dvar2dvar, &_t0 );
                cvMatMul( &_t0, &_dvardR, &matJ );
            }
        }
        else if( tr > 0 )
        {
            cvZero( dst );
            if( jacobian )
            {
                memset( J, 0, sizeof(J) );
                J[5] = J[15] = J[19] = 0.5;
                J[7] = J[11] = J[21] = -0.5;
            }
        }
        else
        {
            r[0] = theta*sqrt((R[0] + 1)*0.5);
            r[1] = theta*sqrt((R[4] + 1)*0.5)*(R[1] >= 0 ? 1 : -1);
            r[2] = theta*sqrt((R[8] + 1)*0.5)*(R[2] >= 0 ? 1 : -1);
            cvConvert( &_r, dst );

            if( jacobian )
                memset( J, 0, sizeof(J) );
        }

        if( jacobian )
        {
            for( i = 0; i < 3; i++ )
            {
                CvMat t = cvMat( 3, 3, CV_64F, J + i*9 );
                cvTranspose( &t, &t );
            }
        }
    }
    else
    {
        assert(0);
        return 0;
    }

    if( jacobian )
    {
        if( depth == CV_32F )
        {
            if( jacobian->rows == matJ.rows )
                cvConvert( &matJ, jacobian );
            else
            {
                _Jf = cvMat( matJ.rows, matJ.cols, CV_32FC1, Jf );
                cvConvert( &matJ, &_Jf );
                cvTranspose( &_Jf, jacobian );
            }
        }
        else if( jacobian->rows == matJ.rows )
            cvCopy( &matJ, jacobian );
        else
            cvTranspose( &matJ, jacobian );
    }

    return 1;
}


void
cvTsConvertHomogeneous( const CvMat* src, CvMat* dst )
{
    CvMat* src_buf = 0;
    CvMat* dst_buf = 0;
    CvMat* dst0 = dst;
    int i, count, sdims, ddims;
    int sstep1, sstep2, dstep1, dstep2;
    double *s, *d;

    if( CV_MAT_DEPTH(src->type) != CV_64F )
    {
        src_buf = cvCreateMat( src->rows, src->cols, CV_MAKETYPE(CV_64F, CV_MAT_CN(src->type)) );
        cvTsConvert( src, src_buf );
        src = src_buf;
    }

    if( CV_MAT_DEPTH(dst->type) != CV_64F )
    {
        dst_buf = cvCreateMat( dst->rows, dst->cols, CV_MAKETYPE(CV_64F, CV_MAT_CN(dst->type)) );
        dst = dst_buf;
    }

    if( src->rows > src->cols )
    {
        count = src->rows;
        sdims = CV_MAT_CN(src->type)*src->cols;
        sstep1 = src->step/sizeof(double);
        sstep2 = 1;
    }
    else
    {
        count = src->cols;
        sdims = CV_MAT_CN(src->type)*src->rows;
        if( src->rows == 1 )
        {
            sstep1 = sdims;
            sstep2 = 1;
        }
        else
        {
            sstep1 = 1;
            sstep2 = src->step/sizeof(double);
        }
    }

    if( dst->rows > dst->cols )
    {
        assert( count == dst->rows );
        ddims = CV_MAT_CN(dst->type)*dst->cols;
        dstep1 = dst->step/sizeof(double);
        dstep2 = 1;
    }
    else
    {
        assert( count == dst->cols );
        ddims = CV_MAT_CN(dst->type)*dst->rows;
        if( dst->rows == 1 )
        {
            dstep1 = ddims;
            dstep2 = 1;
        }
        else
        {
            dstep1 = 1;
            dstep2 = dst->step/sizeof(double);
        }
    }

    s = src->data.db;
    d = dst->data.db;

    if( sdims <= ddims )
    {
        int wstep = dstep2*(ddims - 1);

        for( i = 0; i < count; i++, s += sstep1, d += dstep1 )
        {
            double x = s[0];
            double y = s[sstep2];

            d[wstep] = 1;
            d[0] = x;
            d[dstep2] = y;

            if( sdims >= 3 )
            {
                d[dstep2*2] = s[sstep2*2];
                if( sdims == 4 )
                    d[dstep2*3] = s[sstep2*3];
            }
        }
    }
    else
    {
        int wstep = sstep2*(sdims - 1);

        for( i = 0; i < count; i++, s += sstep1, d += dstep1 )
        {
            double w = s[wstep];
            double x = s[0];
            double y = s[sstep2];

            w = w ? 1./w : 1;

            d[0] = x*w;
            d[dstep2] = y*w;

            if( ddims == 3 )
                d[dstep2*2] = s[sstep2*2]*w;
        }
    }

    if( dst != dst0 )
        cvTsConvert( dst, dst0 );

    cvReleaseMat( &src_buf );
    cvReleaseMat( &dst_buf );
}


void
cvTsProjectPoints( const CvMat* _3d, const CvMat* Rt, const CvMat* A,
                   CvMat* _2d, CvRNG* rng, double sigma )
{
    double p[12];
    CvMat P = cvMat( 3, 4, CV_64F, p );

    int i, count = _3d->cols;

    CvMat* temp;
    CvMat* noise = 0;

    cvMatMul( A, Rt, &P );

    if( rng )
    {
        if( sigma == 0 )
            rng = 0;
        else
        {
            noise = cvCreateMat( 1, _3d->cols, CV_64FC2 );
            cvRandArr( rng, noise, CV_RAND_NORMAL, cvScalarAll(0), cvScalarAll(sigma) );
        }
    }

    temp = cvCreateMat( 1, count, CV_64FC3 );

    for( i = 0; i < count; i++ )
    {
        const double* M = _3d->data.db + i*3;
        double* m = temp->data.db + i*3;
        double X = M[0], Y = M[1], Z = M[2];
        double u = p[0]*X + p[1]*Y + p[2]*Z + p[3];
        double v = p[4]*X + p[5]*Y + p[6]*Z + p[7];
        double s = p[8]*X + p[9]*Y + p[10]*Z + p[11];

        if( noise )
        {
            u += noise->data.db[i*2]*s;
            v += noise->data.db[i*2+1]*s;
        }

        m[0] = u;
        m[1] = v;
        m[2] = s;
    }

    cvTsConvertHomogeneous( temp, _2d );
    cvReleaseMat( &noise );
    cvReleaseMat( &temp );
}


/********************************** Rodrigues transform ********************************/

class CV_RodriguesTest : public CvArrTest
{
public:
    CV_RodriguesTest();

protected:
    int read_params( CvFileStorage* fs );
    void fill_array( int test_case_idx, int i, int j, CvMat* arr );
    int prepare_test_case( int test_case_idx );
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int );

    bool calc_jacobians;
    bool test_cpp;
};


CV_RodriguesTest::CV_RodriguesTest()
    : CvArrTest( "_3d-rodrigues", "cvRodrigues2", "" )
{
    test_array[INPUT].push(NULL);  // rotation vector
    test_array[OUTPUT].push(NULL); // rotation matrix
    test_array[OUTPUT].push(NULL); // jacobian (J)
    test_array[OUTPUT].push(NULL); // rotation vector (backward transform result)
    test_array[OUTPUT].push(NULL); // inverse transform jacobian (J1)
    test_array[OUTPUT].push(NULL); // J*J1 (or J1*J) == I(3x3)
    test_array[REF_OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);

    element_wise_relative_error = false;
    calc_jacobians = false;

    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
    default_timing_param_names = 0;
    test_cpp = false;
}


int CV_RodriguesTest::read_params( CvFileStorage* fs )
{
    int code = CvArrTest::read_params( fs );
    return code;
}


void CV_RodriguesTest::get_test_array_types_and_sizes(
    int /*test_case_idx*/, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int depth = cvTsRandInt(rng) % 2 == 0 ? CV_32F : CV_64F;
    int i, code;

    code = cvTsRandInt(rng) % 3;
    types[INPUT][0] = CV_MAKETYPE(depth, 1);

    if( code == 0 )
    {
        sizes[INPUT][0] = cvSize(1,1);
        types[INPUT][0] = CV_MAKETYPE(depth, 3);
    }
    else if( code == 1 )
        sizes[INPUT][0] = cvSize(3,1);
    else
        sizes[INPUT][0] = cvSize(1,3);

    sizes[OUTPUT][0] = cvSize(3, 3);
    types[OUTPUT][0] = CV_MAKETYPE(depth, 1);

    types[OUTPUT][1] = CV_MAKETYPE(depth, 1);

    if( cvTsRandInt(rng) % 2 )
        sizes[OUTPUT][1] = cvSize(3,9);
    else
        sizes[OUTPUT][1] = cvSize(9,3);

    types[OUTPUT][2] = types[INPUT][0];
    sizes[OUTPUT][2] = sizes[INPUT][0];

    types[OUTPUT][3] = types[OUTPUT][1];
    sizes[OUTPUT][3] = cvSize(sizes[OUTPUT][1].height, sizes[OUTPUT][1].width);

    types[OUTPUT][4] = types[OUTPUT][1];
    sizes[OUTPUT][4] = cvSize(3,3);

    calc_jacobians = cvTsRandInt(rng) % 3 != 0;
    if( !calc_jacobians )
        sizes[OUTPUT][1] = sizes[OUTPUT][3] = sizes[OUTPUT][4] = cvSize(0,0);

    for( i = 0; i < 5; i++ )
    {
        types[REF_OUTPUT][i] = types[OUTPUT][i];
        sizes[REF_OUTPUT][i] = sizes[OUTPUT][i];
    }
    test_cpp = (cvTsRandInt(rng) & 256) == 0;
}


double CV_RodriguesTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int j )
{
    return j == 4 ? 1e-2 : 1e-2;
}


void CV_RodriguesTest::fill_array( int test_case_idx, int i, int j, CvMat* arr )
{
    if( i == INPUT && j == 0 )
    {
        double r[3], theta0, theta1, f;
        CvMat _r = cvMat( arr->rows, arr->cols, CV_MAKETYPE(CV_64F,CV_MAT_CN(arr->type)), r );
        CvRNG* rng = ts->get_rng();

        r[0] = cvTsRandReal(rng)*CV_PI*2;
        r[1] = cvTsRandReal(rng)*CV_PI*2;
        r[2] = cvTsRandReal(rng)*CV_PI*2;

        theta0 = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
        theta1 = fmod(theta0, CV_PI*2);

        if( theta1 > CV_PI )
            theta1 = -(CV_PI*2 - theta1);

        f = theta1/(theta0 ? theta0 : 1);
        r[0] *= f;
        r[1] *= f;
        r[2] *= f;

        cvTsConvert( &_r, arr );
    }
    else
        CvArrTest::fill_array( test_case_idx, i, j, arr );
}


int CV_RodriguesTest::prepare_test_case( int test_case_idx )
{
    int code = CvArrTest::prepare_test_case( test_case_idx );
    return code;
}


void CV_RodriguesTest::run_func()
{
    CvMat *v2m_jac = 0, *m2v_jac = 0;
    
    if( calc_jacobians )
    {
        v2m_jac = &test_mat[OUTPUT][1];
        m2v_jac = &test_mat[OUTPUT][3];
    }

    if( !test_cpp )
    {
        cvRodrigues2( &test_mat[INPUT][0], &test_mat[OUTPUT][0], v2m_jac );
        cvRodrigues2( &test_mat[OUTPUT][0], &test_mat[OUTPUT][2], m2v_jac );
    }
    else
    {
        cv::Mat v(&test_mat[INPUT][0]), M(&test_mat[OUTPUT][0]), v2(&test_mat[OUTPUT][2]);
        cv::Mat M0 = M, v2_0 = v2;
        if( !calc_jacobians )
        {
            cv::Rodrigues(v, M);
            cv::Rodrigues(M, v2);
        }
        else
        {
            cv::Mat J1(&test_mat[OUTPUT][1]), J2(&test_mat[OUTPUT][3]);
            cv::Mat J1_0 = J1, J2_0 = J2;
            cv::Rodrigues(v, M, J1);
            cv::Rodrigues(M, v2, J2);
            if( J1.data != J1_0.data )
            {
                if( J1.size() != J1_0.size() )
                    J1 = J1.t();
                J1.convertTo(J1_0, J1_0.type());
            }
            if( J2.data != J2_0.data )
            {
                if( J2.size() != J2_0.size() )
                    J2 = J2.t();
                J2.convertTo(J2_0, J2_0.type());
            }
        }
        if( M.data != M0.data )
            M.reshape(M0.channels(), M0.rows).convertTo(M0, M0.type());
        if( v2.data != v2_0.data )
            v2.reshape(v2_0.channels(), v2_0.rows).convertTo(v2_0, v2_0.type());
    }
}


void CV_RodriguesTest::prepare_to_validation( int /*test_case_idx*/ )
{
    const CvMat* vec = &test_mat[INPUT][0];
    CvMat* m = &test_mat[REF_OUTPUT][0];
    CvMat* vec2 = &test_mat[REF_OUTPUT][2];
    CvMat* v2m_jac = 0, *m2v_jac = 0;
    double theta0, theta1;

    if( calc_jacobians )
    {
        v2m_jac = &test_mat[REF_OUTPUT][1];
        m2v_jac = &test_mat[REF_OUTPUT][3];
    }


    cvTsRodrigues( vec, m, v2m_jac );
    cvTsRodrigues( m, vec2, m2v_jac );
    cvTsCopy( vec, vec2 );

    theta0 = cvNorm( vec2, 0, CV_L2 );
    theta1 = fmod( theta0, CV_PI*2 );

    if( theta1 > CV_PI )
        theta1 = -(CV_PI*2 - theta1);
    cvScale( vec2, vec2, theta1/(theta0 ? theta0 : 1) );

    if( calc_jacobians )
    {
        //cvInvert( v2m_jac, m2v_jac, CV_SVD );
        double nrm = cvNorm(&test_mat[REF_OUTPUT][3],0,CV_C);
        if( FLT_EPSILON < nrm && nrm < 1000 )
        {
            cvTsGEMM( &test_mat[OUTPUT][1], &test_mat[OUTPUT][3],
                      1, 0, 0, &test_mat[OUTPUT][4],
                      v2m_jac->rows == 3 ? 0 : CV_GEMM_A_T + CV_GEMM_B_T );
        }
        else
        {
            cvTsSetIdentity( &test_mat[OUTPUT][4], cvScalarAll(1.) );
            cvTsCopy( &test_mat[REF_OUTPUT][2], &test_mat[OUTPUT][2] );
        }
        cvTsSetIdentity( &test_mat[REF_OUTPUT][4], cvScalarAll(1.) );
    }
}


CV_RodriguesTest rodrigues_test;


/********************************** fundamental matrix *********************************/

class CV_FundamentalMatTest : public CvArrTest
{
public:
    CV_FundamentalMatTest();

protected:
    int read_params( CvFileStorage* fs );
    void fill_array( int test_case_idx, int i, int j, CvMat* arr );
    int prepare_test_case( int test_case_idx );
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int );

    int method;
    int img_size;
    int cube_size;
    int dims;
    int f_result;
    double min_f, max_f;
    double sigma;
    bool test_cpp;
};


CV_FundamentalMatTest::CV_FundamentalMatTest()
    : CvArrTest( "_3d-fundam", "cvFindFundamentalMatrix", "" )
{
    // input arrays:
    //   0, 1 - arrays of 2d points that are passed to %func%.
    //          Can have different data type, layout, be stored in homogeneous coordinates or not.
    //   2 - array of 3d points that are projected to both view planes
    //   3 - [R|t] matrix for the second view plane (for the first one it is [I|0]
    //   4, 5 - intrinsic matrices
    test_array[INPUT].push(NULL);
    test_array[INPUT].push(NULL);
    test_array[INPUT].push(NULL);
    test_array[INPUT].push(NULL);
    test_array[INPUT].push(NULL);
    test_array[INPUT].push(NULL);
    test_array[TEMP].push(NULL);
    test_array[TEMP].push(NULL);
    test_array[OUTPUT].push(NULL);
    test_array[OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);

    element_wise_relative_error = false;

    method = 0;
    img_size = 10;
    cube_size = 10;
    min_f = 1;
    max_f = 3;
    sigma = 0;//0.1;
    f_result = 0;

    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
    default_timing_param_names = 0;
    test_cpp = false;
}


int CV_FundamentalMatTest::read_params( CvFileStorage* fs )
{
    int code = CvArrTest::read_params( fs );
    return code;
}


void CV_FundamentalMatTest::get_test_array_types_and_sizes( int /*test_case_idx*/,
                                                CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int pt_depth = cvTsRandInt(rng) % 2 == 0 ? CV_32F : CV_64F;
    double pt_count_exp = cvTsRandReal(rng)*6 + 1;
    int pt_count = cvRound(exp(pt_count_exp));

    dims = cvTsRandInt(rng) % 2 + 2;
    method = 1 << (cvTsRandInt(rng) % 4);

    if( method == CV_FM_7POINT )
        pt_count = 7;
    else
    {
        pt_count = MAX( pt_count, 8 + (method == CV_FM_8POINT) );
        if( pt_count >= 8 && cvTsRandInt(rng) % 2 )
            method |= CV_FM_8POINT;
    }

    types[INPUT][0] = CV_MAKETYPE(pt_depth, 1);

    if( cvTsRandInt(rng) % 2 )
        sizes[INPUT][0] = cvSize(pt_count, dims);
    else
    {
        sizes[INPUT][0] = cvSize(dims, pt_count);
        if( cvTsRandInt(rng) % 2 )
        {
            types[INPUT][0] = CV_MAKETYPE(pt_depth, dims);
            if( cvTsRandInt(rng) % 2 )
                sizes[INPUT][0] = cvSize(pt_count, 1);
            else
                sizes[INPUT][0] = cvSize(1, pt_count);
        }
    }

    sizes[INPUT][1] = sizes[INPUT][0];
    types[INPUT][1] = types[INPUT][0];

    sizes[INPUT][2] = cvSize(pt_count, 1 );
    types[INPUT][2] = CV_64FC3;

    sizes[INPUT][3] = cvSize(4,3);
    types[INPUT][3] = CV_64FC1;

    sizes[INPUT][4] = sizes[INPUT][5] = cvSize(3,3);
    types[INPUT][4] = types[INPUT][5] = CV_MAKETYPE(CV_64F, 1);

    sizes[TEMP][0] = cvSize(3,3);
    types[TEMP][0] = CV_64FC1;
    sizes[TEMP][1] = cvSize(pt_count,1);
    types[TEMP][1] = CV_8UC1;

    sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = cvSize(3,1);
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_64FC1;
    sizes[OUTPUT][1] = sizes[REF_OUTPUT][1] = cvSize(pt_count,1);
    types[OUTPUT][1] = types[REF_OUTPUT][1] = CV_8UC1;
    
    test_cpp = (cvTsRandInt(rng) & 256) == 0;
}


double CV_FundamentalMatTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 1e-2;
}


void CV_FundamentalMatTest::fill_array( int test_case_idx, int i, int j, CvMat* arr )
{
    double t[12];
    CvMat T;
    double* p = arr->data.db;
    CvRNG* rng = ts->get_rng();

    if( i != INPUT )
    {
        CvArrTest::fill_array( test_case_idx, i, j, arr );
        return;
    }

    switch( j )
    {
    case 0:
    case 1:
        return; // fill them later in prepare_test_case
    case 2:
        for( i = 0; i < arr->cols*3; i += 3 )
        {
            p[i] = cvTsRandReal(rng)*cube_size;
            p[i+1] = cvTsRandReal(rng)*cube_size;
            p[i+2] = cvTsRandReal(rng)*cube_size + cube_size;
        }
        break;
    case 3:
        {
        double r[3];
        CvMat rot_vec = cvMat( 3, 1, CV_64F, r );
        CvMat rot_mat = cvMat( 3, 3, CV_64F );
        r[0] = cvTsRandReal(rng)*CV_PI*2;
        r[1] = cvTsRandReal(rng)*CV_PI*2;
        r[2] = cvTsRandReal(rng)*CV_PI*2;

        cvSetData( &rot_mat, t, 4*sizeof(t[0]) );
        cvTsRodrigues( &rot_vec, &rot_mat );
        t[3] = cvTsRandReal(rng)*cube_size;
        t[7] = cvTsRandReal(rng)*cube_size;
        t[11] = cvTsRandReal(rng)*cube_size;
        T = cvMat( 3, 4, CV_64F, t );
        cvTsConvert( &T, arr );
        }
        break;
    case 4:
    case 5:
        memset( t, 0, sizeof(t) );
        t[0] = t[4] = cvTsRandReal(rng)*(max_f - min_f) + min_f;
        t[2] = (img_size*0.5 + cvTsRandReal(rng)*4. - 2.)*t[0];
        t[5] = (img_size*0.5 + cvTsRandReal(rng)*4. - 2.)*t[4];
        t[8] = 1.;
        T = cvMat( 3, 3, CV_64F, t );
        cvTsConvert( &T, arr );
        break;
    }
}


int CV_FundamentalMatTest::prepare_test_case( int test_case_idx )
{
    int code = CvArrTest::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        const CvMat* _3d = &test_mat[INPUT][2];
        CvRNG* rng = ts->get_rng();
        double Idata[] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
        CvMat I = cvMat( 3, 4, CV_64F, Idata );
        int k;

        for( k = 0; k < 2; k++ )
        {
            const CvMat* Rt = k == 0 ? &I : &test_mat[INPUT][3];
            const CvMat* A = &test_mat[INPUT][k == 0 ? 4 : 5];
            CvMat* _2d = &test_mat[INPUT][k];

            cvTsProjectPoints( _3d, Rt, A, _2d, rng, sigma );
        }
    }

    return code;
}


void CV_FundamentalMatTest::run_func()
{
    //if(!test_cpp)
    {
        f_result = cvFindFundamentalMat( &test_mat[INPUT][0], &test_mat[INPUT][1],
                    &test_mat[TEMP][0], method, MAX(sigma*3, 0.01), 0, &test_mat[TEMP][1] );
    }
    /*else
    {
        cv::findFundamentalMat(const Mat& points1, const Mat& points2,
        vector<uchar>& mask, int method=FM_RANSAC,
        double param1=3., double param2=0.99 );
        
        CV_EXPORTS Mat findFundamentalMat( const Mat& points1, const Mat& points2,
                                          int method=FM_RANSAC,
                                          double param1=3., double param2=0.99 );
    }*/

}


void CV_FundamentalMatTest::prepare_to_validation( int test_case_idx )
{
    const CvMat* Rt = &test_mat[INPUT][3];
    const CvMat* A1 = &test_mat[INPUT][4];
    const CvMat* A2 = &test_mat[INPUT][5];
    double f0[9];
    CvMat F0 = cvMat( 3, 3, CV_64FC1, f0 );

    double _invA1[9], _invA2[9], temp[9];
    CvMat invA1 = cvMat( 3, 3, CV_64F, _invA1 );
    CvMat invA2 = cvMat( 3, 3, CV_64F, _invA2 );
    CvMat R = cvMat( 3, 3, CV_64F );
    CvMat T = cvMat( 3, 3, CV_64F, temp );

    cvSetData( &R, Rt->data.db, Rt->step ); // R = Rt(:,1:3)

    // F = (A2^-T)*[t]_x*R*(A1^-1)
    cvInvert( A1, &invA1, CV_SVD );
    cvInvert( A2, &invA2, CV_SVD );

    {
    double tx = ((double*)(Rt->data.ptr))[3];
    double ty = ((double*)(Rt->data.ptr + Rt->step))[3];
    double tz = ((double*)(Rt->data.ptr + Rt->step*2))[3];

    double _t_x[] = { 0, -tz, ty, tz, 0, -tx, -ty, tx, 0 };
    CvMat t_x = cvMat( 3, 3, CV_64F, _t_x );

    cvGEMM( &invA2, &t_x, 1, 0, 0, &T, CV_GEMM_A_T );
    cvMatMul( &R, &invA1, &invA2 );
    cvMatMul( &T, &invA2, &F0 );
    cvScale( &F0, &F0, f0[8] );
    }

    double f[9];
    CvMat F = cvMat(3, 3, CV_64F, f);
    uchar* status = test_mat[TEMP][1].data.ptr;
    double err_level = get_success_error_level( test_case_idx, OUTPUT, 1 );
    uchar* mtfm1 = test_mat[REF_OUTPUT][1].data.ptr;
    uchar* mtfm2 = test_mat[OUTPUT][1].data.ptr;
    double* f_prop1 = test_mat[REF_OUTPUT][0].data.db;
    double* f_prop2 = test_mat[OUTPUT][0].data.db;

    int i, pt_count = test_mat[INPUT][2].cols;
    CvMat* p1 = cvCreateMat( 1, pt_count, CV_64FC2 );
    CvMat* p2 = cvCreateMat( 1, pt_count, CV_64FC2 );

    cvTsConvertHomogeneous( &test_mat[INPUT][0], p1 );
    cvTsConvertHomogeneous( &test_mat[INPUT][1], p2 );

    cvTsConvert( &test_mat[TEMP][0], &F );

    if( method <= CV_FM_8POINT )
        memset( status, 1, pt_count );

    for( i = 0; i < pt_count; i++ )
    {
        double x1 = p1->data.db[i*2];
        double y1 = p1->data.db[i*2+1];
        double x2 = p2->data.db[i*2];
        double y2 = p2->data.db[i*2+1];
        double n1 = 1./sqrt(x1*x1 + y1*y1 + 1);
        double n2 = 1./sqrt(x2*x2 + y2*y2 + 1);
        double t0 = fabs(f0[0]*x2*x1 + f0[1]*x2*y1 + f0[2]*x2 +
                   f0[3]*y2*x1 + f0[4]*y2*y1 + f0[5]*y2 +
                   f0[6]*x1 + f0[7]*y1 + f0[8])*n1*n2;
        double t = fabs(f[0]*x2*x1 + f[1]*x2*y1 + f[2]*x2 +
                   f[3]*y2*x1 + f[4]*y2*y1 + f[5]*y2 +
                   f[6]*x1 + f[7]*y1 + f[8])*n1*n2;
        mtfm1[i] = 1;
        mtfm2[i] = !status[i] || t0 > err_level || t < err_level;
    }

    f_prop1[0] = 1;
    f_prop1[1] = 1;
    f_prop1[2] = 0;

    f_prop2[0] = f_result != 0;
    f_prop2[1] = f[8];
    f_prop2[2] = cvDet( &F );

    cvReleaseMat( &p1 );
    cvReleaseMat( &p2 );
}


CV_FundamentalMatTest fmatrix_test;


/********************************** convert homogeneous *********************************/

class CV_ConvertHomogeneousTest : public CvArrTest
{
public:
    CV_ConvertHomogeneousTest();

protected:
    int read_params( CvFileStorage* fs );
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void fill_array( int test_case_idx, int i, int j, CvMat* arr );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int );

    int dims1, dims2;
    int pt_count;
};


CV_ConvertHomogeneousTest::CV_ConvertHomogeneousTest()
    : CvArrTest( "_3d-cvt-homogen", "cvConvertPointsHomogeniuos", "" )
{
    test_array[INPUT].push(NULL);
    test_array[OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);
    element_wise_relative_error = false;

    pt_count = dims1 = dims2 = 0;

    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
    default_timing_param_names = 0;
}


int CV_ConvertHomogeneousTest::read_params( CvFileStorage* fs )
{
    int code = CvArrTest::read_params( fs );
    return code;
}


void CV_ConvertHomogeneousTest::get_test_array_types_and_sizes( int /*test_case_idx*/,
                                                CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int pt_depth1 = cvTsRandInt(rng) % 2 == 0 ? CV_32F : CV_64F;
    int pt_depth2 = cvTsRandInt(rng) % 2 == 0 ? CV_32F : CV_64F;
    double pt_count_exp = cvTsRandReal(rng)*6 + 1;
    int t;

    pt_count = cvRound(exp(pt_count_exp));
    pt_count = MAX( pt_count, 5 );

    dims1 = 2 + (cvTsRandInt(rng) % 3);
    dims2 = 2 + (cvTsRandInt(rng) % 3);

    if( dims1 == dims2 + 2 )
        dims1--;
    else if( dims1 == dims2 - 2 )
        dims1++;

    if( cvTsRandInt(rng) % 2 )
        CV_SWAP( dims1, dims2, t );

    types[INPUT][0] = CV_MAKETYPE(pt_depth1, 1);

    if( cvTsRandInt(rng) % 2 )
        sizes[INPUT][0] = cvSize(pt_count, dims1);
    else
    {
        sizes[INPUT][0] = cvSize(dims1, pt_count);
        if( cvTsRandInt(rng) % 2 )
        {
            types[INPUT][0] = CV_MAKETYPE(pt_depth1, dims1);
            if( cvTsRandInt(rng) % 2 )
                sizes[INPUT][0] = cvSize(pt_count, 1);
            else
                sizes[INPUT][0] = cvSize(1, pt_count);
        }
    }

    types[OUTPUT][0] = CV_MAKETYPE(pt_depth2, 1);

    if( cvTsRandInt(rng) % 2 )
        sizes[OUTPUT][0] = cvSize(pt_count, dims2);
    else
    {
        sizes[OUTPUT][0] = cvSize(dims2, pt_count);
        if( cvTsRandInt(rng) % 2 )
        {
            types[OUTPUT][0] = CV_MAKETYPE(pt_depth2, dims2);
            if( cvTsRandInt(rng) % 2 )
                sizes[OUTPUT][0] = cvSize(pt_count, 1);
            else
                sizes[OUTPUT][0] = cvSize(1, pt_count);
        }
    }

    types[REF_OUTPUT][0] = types[OUTPUT][0];
    sizes[REF_OUTPUT][0] = sizes[OUTPUT][0];
}


double CV_ConvertHomogeneousTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 1e-5;
}


void CV_ConvertHomogeneousTest::fill_array( int /*test_case_idx*/, int /*i*/, int /*j*/, CvMat* arr )
{
    CvMat* temp = cvCreateMat( 1, pt_count, CV_MAKETYPE(CV_64FC1,dims1) );
    CvRNG* rng = ts->get_rng();
    CvScalar low = cvScalarAll(0), high = cvScalarAll(10);

    if( dims1 > dims2 )
        low.val[dims1-1] = 1.;

    cvRandArr( rng, temp, CV_RAND_UNI, low, high );
    cvTsConvertHomogeneous( temp, arr );
    cvReleaseMat( &temp );
}


void CV_ConvertHomogeneousTest::run_func()
{
    cvConvertPointsHomogeneous( &test_mat[INPUT][0], &test_mat[OUTPUT][0] );
}


void CV_ConvertHomogeneousTest::prepare_to_validation( int /*test_case_idx*/ )
{
    cvTsConvertHomogeneous( &test_mat[INPUT][0], &test_mat[REF_OUTPUT][0] );
}


CV_ConvertHomogeneousTest cvt_homogen_test;


/************************** compute corresponding epipolar lines ************************/

class CV_ComputeEpilinesTest : public CvArrTest
{
public:
    CV_ComputeEpilinesTest();

protected:
    int read_params( CvFileStorage* fs );
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void fill_array( int test_case_idx, int i, int j, CvMat* arr );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int );

    int which_image;
    int dims;
    int pt_count;
};


CV_ComputeEpilinesTest::CV_ComputeEpilinesTest()
    : CvArrTest( "_3d-epilines", "cvComputeCorrespondingEpilines", "" )
{
    test_array[INPUT].push(NULL);
    test_array[INPUT].push(NULL);
    test_array[OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);
    element_wise_relative_error = false;

    pt_count = dims = which_image = 0;

    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
    default_timing_param_names = 0;
}


int CV_ComputeEpilinesTest::read_params( CvFileStorage* fs )
{
    int code = CvArrTest::read_params( fs );
    return code;
}


void CV_ComputeEpilinesTest::get_test_array_types_and_sizes( int /*test_case_idx*/,
                                                CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int fm_depth = cvTsRandInt(rng) % 2 == 0 ? CV_32F : CV_64F;
    int pt_depth = cvTsRandInt(rng) % 2 == 0 ? CV_32F : CV_64F;
    int ln_depth = cvTsRandInt(rng) % 2 == 0 ? CV_32F : CV_64F;
    double pt_count_exp = cvTsRandReal(rng)*6 + 1;

    which_image = 1 + (cvTsRandInt(rng) % 2);

    pt_count = cvRound(exp(pt_count_exp));
    pt_count = MAX( pt_count, 5 );

    dims = 2 + (cvTsRandInt(rng) % 2);

    types[INPUT][0] = CV_MAKETYPE(pt_depth, 1);

    if( cvTsRandInt(rng) % 2 )
        sizes[INPUT][0] = cvSize(pt_count, dims);
    else
    {
        sizes[INPUT][0] = cvSize(dims, pt_count);
        if( cvTsRandInt(rng) % 2 )
        {
            types[INPUT][0] = CV_MAKETYPE(pt_depth, dims);
            if( cvTsRandInt(rng) % 2 )
                sizes[INPUT][0] = cvSize(pt_count, 1);
            else
                sizes[INPUT][0] = cvSize(1, pt_count);
        }
    }

    types[INPUT][1] = CV_MAKETYPE(fm_depth, 1);
    sizes[INPUT][1] = cvSize(3, 3);

    types[OUTPUT][0] = CV_MAKETYPE(ln_depth, 1);

    if( cvTsRandInt(rng) % 2 )
        sizes[OUTPUT][0] = cvSize(pt_count, 3);
    else
    {
        sizes[OUTPUT][0] = cvSize(3, pt_count);
        if( cvTsRandInt(rng) % 2 )
        {
            types[OUTPUT][0] = CV_MAKETYPE(ln_depth, 3);
            if( cvTsRandInt(rng) % 2 )
                sizes[OUTPUT][0] = cvSize(pt_count, 1);
            else
                sizes[OUTPUT][0] = cvSize(1, pt_count);
        }
    }

    types[REF_OUTPUT][0] = types[OUTPUT][0];
    sizes[REF_OUTPUT][0] = sizes[OUTPUT][0];
}


double CV_ComputeEpilinesTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 1e-5;
}


void CV_ComputeEpilinesTest::fill_array( int test_case_idx, int i, int j, CvMat* arr )
{
    CvRNG* rng = ts->get_rng();

    if( i == INPUT && j == 0 )
    {
        CvMat* temp = cvCreateMat( 1, pt_count, CV_MAKETYPE(CV_64FC1,dims) );
        cvRandArr( rng, temp, CV_RAND_UNI, cvScalar(0,0,1), cvScalarAll(10) );
        cvTsConvertHomogeneous( temp, arr );
        cvReleaseMat( &temp );
    }
    else if( i == INPUT && j == 1 )
        cvRandArr( rng, arr, CV_RAND_UNI, cvScalarAll(0), cvScalarAll(10) );
    else
        CvArrTest::fill_array( test_case_idx, i, j, arr );
}


void CV_ComputeEpilinesTest::run_func()
{
    cvComputeCorrespondEpilines( &test_mat[INPUT][0], which_image,
                                 &test_mat[INPUT][1], &test_mat[OUTPUT][0] );
}


void CV_ComputeEpilinesTest::prepare_to_validation( int /*test_case_idx*/ )
{
    CvMat* pt = cvCreateMat( 1, pt_count, CV_MAKETYPE(CV_64F, 3) );
    CvMat* lines = cvCreateMat( 1, pt_count, CV_MAKETYPE(CV_64F, 3) );
    double f[9];
    CvMat F = cvMat( 3, 3, CV_64F, f );
    int i;

    cvTsConvertHomogeneous( &test_mat[INPUT][0], pt );
    cvTsConvert( &test_mat[INPUT][1], &F );
    if( which_image == 2 )
        cvTranspose( &F, &F );

    for( i = 0; i < pt_count; i++ )
    {
        double* p = pt->data.db + i*3;
        double* l = lines->data.db + i*3;
        double t0 = f[0]*p[0] + f[1]*p[1] + f[2]*p[2];
        double t1 = f[3]*p[0] + f[4]*p[1] + f[5]*p[2];
        double t2 = f[6]*p[0] + f[7]*p[1] + f[8]*p[2];
        double d = sqrt(t0*t0 + t1*t1);
        d = d ? 1./d : 1.;
        l[0] = t0*d; l[1] = t1*d; l[2] = t2*d;
    }

    cvTsConvertHomogeneous( lines, &test_mat[REF_OUTPUT][0] );
    cvReleaseMat( &pt );
    cvReleaseMat( &lines );
}


CV_ComputeEpilinesTest epilines_test;


/* End of file. */
