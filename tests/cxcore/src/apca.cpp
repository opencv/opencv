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

#include "cxcoretest.h"

using namespace cv;

#define CHECK_C

Size sz( 200, 500);

class CV_PCATest : public CvTest
{
public:
    CV_PCATest() : CvTest( "pca", "PCA funcs" ) {}
protected:
    void run( int);
};

#if 0

void CV_PCATest::run( int )
{
    int code = CvTS::OK, err;
    int maxComponents = 1;
    Mat points( 1000, 3, CV_32FC1);

	RNG rng = *ts->get_rng(); // get ts->rng seed
	rng.fill( points, RNG::NORMAL, Scalar::all(0.0), Scalar::all(1.0) );

    float mp[] = { 3.0f, 3.0f, 3.0f }, cp[] = { 0.5f, 0.0f, 0.0f,
                                                0.0f, 1.0f, 0.0f,
                                                0.0f, 0.0f, 0.3f };
    Mat mean( 1, 3, CV_32FC1, mp ),
        cov( 3, 3, CV_32FC1, cp );
    for( int i = 0; i < points.rows; i++ )
    {
        Mat r(1, points.cols, CV_32FC1, points.ptr<float>(i));
        r =  r * cov + mean; 
    }

    PCA pca( points, Mat(), CV_PCA_DATA_AS_ROW, maxComponents );

    // check project
    Mat prjPoints = pca.project( points );
    err = 0;
    for( int i = 0; i < prjPoints.rows; i++ )
    {
        float val = prjPoints.at<float>(i,0);
        if( val > 3.0f || val < -3.0f )
            err++;
    }
	float projectErr = 0.02f;
	if( (float)err > prjPoints.rows * projectErr )
    {
        ts->printf( CvTS::LOG, "bad accuracy of project() (real = %f, permissible = %f)",
			(float)err/(float)prjPoints.rows, projectErr );
        code = CvTS::FAIL_BAD_ACCURACY;
    }

    // check backProject
    Mat points1 = pca.backProject( prjPoints );
    err = 0;
	for( int i = 0; i < points.rows; i++ ) 
	{
		if( fabs(points1.at<float>(i,0) - mean.at<float>(0,0)) > 0.15 ||
            fabs(points1.at<float>(i,1) - points.at<float>(i,1)) > 0.05 ||
            fabs(points1.at<float>(i,2) - mean.at<float>(0,2)) > 0.15 )
            err++;
	}
	float backProjectErr = 0.05f;
	if( (float)err > prjPoints.rows*backProjectErr )
    {
        ts->printf( CvTS::LOG, "bad accuracy of backProject() (real = %f, permissible = %f)",
			(float)err/(float)prjPoints.rows, backProjectErr );
        code = CvTS::FAIL_BAD_ACCURACY;
    }

	CvRNG *oldRng = ts->get_rng(); // set ts->rng seed
	*oldRng = rng.state;

    ts->set_failed_test_info( code );
}
#else
void CV_PCATest::run( int )
{
	int code = CvTS::OK;
	
	double diffPrjEps, diffBackPrjEps,
		   prjEps, backPrjEps,
		   evalEps, evecEps;
	int maxComponents = 100;
	Mat rPoints(sz, CV_32FC1), rTestPoints(sz, CV_32FC1);
	RNG rng = *ts->get_rng(); 

	rng.fill( rPoints, RNG::UNIFORM, Scalar::all(0.0), Scalar::all(1.0) );
	rng.fill( rTestPoints, RNG::UNIFORM, Scalar::all(0.0), Scalar::all(1.0) );

	PCA rPCA( rPoints, Mat(), CV_PCA_DATA_AS_ROW, maxComponents ), cPCA;

	// 1. check C++ PCA & ROW
	Mat rPrjTestPoints = rPCA.project( rTestPoints );
	Mat rBackPrjTestPoints = rPCA.backProject( rPrjTestPoints );

	Mat avg(1, sz.width, CV_32FC1 );
	reduce( rPoints, avg, 0, CV_REDUCE_AVG );
	Mat Q = rPoints - repeat( avg, rPoints.rows, 1 ), Qt = Q.t(), eval, evec;
	Q = Qt * Q;
	Q = Q /(float)rPoints.rows;

	eigen( Q, eval, evec );
	/*SVD svd(Q);
	evec = svd.vt;
	eval = svd.w;*/

	Mat subEval( maxComponents, 1, eval.type(), eval.data ),
		subEvec( maxComponents, evec.cols, evec.type(), evec.data );

#ifdef CHECK_C
	Mat prjTestPoints, backPrjTestPoints, cPoints = rPoints.t(), cTestPoints = rTestPoints.t();
	CvMat _points, _testPoints, _avg, _eval, _evec, _prjTestPoints, _backPrjTestPoints;
#endif

	// check eigen()
	double eigenEps = 1e-6;
	double err;
	for(int i = 0; i < Q.rows; i++ )
	{
		Mat v = evec.row(i).t();
		Mat Qv = Q * v;

		Mat lv = eval.at<float>(i,0) * v;
		err = norm( Qv, lv );
		if( err > eigenEps )
		{
			ts->printf( CvTS::LOG, "bad accuracy of eigen(); err = %f\n", err );
			code = CvTS::FAIL_BAD_ACCURACY;
			goto exit_func;
		}
	}
	// check pca eigenvalues
	evalEps = 1e-6, evecEps = 1;
	err = norm( rPCA.eigenvalues, subEval );
	if( err > evalEps )
	{
		ts->printf( CvTS::LOG, "pca.eigenvalues is incorrect (CV_PCA_DATA_AS_ROW); err = %f\n", err );
		code = CvTS::FAIL_BAD_ACCURACY;
		goto exit_func;
	}
	// check pca eigenvectors
	err = norm( rPCA.eigenvectors, subEvec, CV_RELATIVE_L2 );
	if( err > evecEps )
	{
		ts->printf( CvTS::LOG, "pca.eigenvectors is incorrect (CV_PCA_DATA_AS_ROW); err = %f\n", err );
		code = CvTS::FAIL_BAD_ACCURACY;
		goto exit_func;
	}
	
    prjEps = 1.265, backPrjEps = 1.265;
	for( int i = 0; i < rTestPoints.rows; i++ )
	{
		// check pca project
		Mat subEvec_t = subEvec.t();
		Mat prj = rTestPoints.row(i) - avg; prj *= subEvec_t;
		err = norm(rPrjTestPoints.row(i), prj, CV_RELATIVE_L2);
		if( err > prjEps )
		{
			ts->printf( CvTS::LOG, "bad accuracy of project() (CV_PCA_DATA_AS_ROW); err = %f\n", err );
			code = CvTS::FAIL_BAD_ACCURACY;
			goto exit_func;
		}
		// check pca backProject
		Mat backPrj = rPrjTestPoints.row(i) * subEvec + avg;
		err = norm( rBackPrjTestPoints.row(i), backPrj, CV_RELATIVE_L2 );
		if( err > backPrjEps )
		{
			ts->printf( CvTS::LOG, "bad accuracy of backProject() (CV_PCA_DATA_AS_ROW); err = %f\n", err );
			code = CvTS::FAIL_BAD_ACCURACY;
			goto exit_func;
		}
	}

	// 2. check C++ PCA & COL
	cPCA( rPoints.t(), Mat(), CV_PCA_DATA_AS_COL, maxComponents );
	diffPrjEps = 1, diffBackPrjEps = 1;
	err = norm(cPCA.project(rTestPoints.t()), rPrjTestPoints.t(), CV_RELATIVE_L2 );
	if( err > diffPrjEps )
	{
		ts->printf( CvTS::LOG, "bad accuracy of project() (CV_PCA_DATA_AS_COL); err = %f\n", err );
		code = CvTS::FAIL_BAD_ACCURACY;
		goto exit_func;
	}
	err = norm(cPCA.backProject(rPrjTestPoints.t()), rBackPrjTestPoints.t(), CV_RELATIVE_L2 );
	if( err > diffBackPrjEps )
	{
		ts->printf( CvTS::LOG, "bad accuracy of backProject() (CV_PCA_DATA_AS_COL); err = %f\n", err );
		code = CvTS::FAIL_BAD_ACCURACY;
		goto exit_func;
	}

#ifdef CHECK_C
	// 3. check C PCA & ROW
	_points = rPoints;
	_testPoints = rTestPoints;
	_avg = avg;
	_eval = eval;
	_evec = evec;
	prjTestPoints.create(rTestPoints.rows, maxComponents, rTestPoints.type() );
	backPrjTestPoints.create(rPoints.size(), rPoints.type() );
	_prjTestPoints = prjTestPoints;
	_backPrjTestPoints = backPrjTestPoints;

	cvCalcPCA( &_points, &_avg, &_eval, &_evec, CV_PCA_DATA_AS_ROW );
	cvProjectPCA( &_testPoints, &_avg, &_evec, &_prjTestPoints );
	cvBackProjectPCA( &_prjTestPoints, &_avg, &_evec, &_backPrjTestPoints );

	err = norm(prjTestPoints, rPrjTestPoints, CV_RELATIVE_L2);
	if( err > diffPrjEps )
	{
		ts->printf( CvTS::LOG, "bad accuracy of cvProjectPCA() (CV_PCA_DATA_AS_ROW); err = %f\n", err );
		code = CvTS::FAIL_BAD_ACCURACY;
		goto exit_func;
	}
	err = norm(backPrjTestPoints, rBackPrjTestPoints, CV_RELATIVE_L2);
	if( err > diffBackPrjEps )
	{
		ts->printf( CvTS::LOG, "bad accuracy of cvBackProjectPCA() (CV_PCA_DATA_AS_ROW); err = %f\n", err );
		code = CvTS::FAIL_BAD_ACCURACY;
		goto exit_func;
	}

	// 3. check C PCA & COL
	_points = cPoints;
	_testPoints = cTestPoints;
	avg = avg.t(); _avg = avg;
	eval = eval.t(); _eval = eval;
	evec = evec.t(); _evec = evec;
    prjTestPoints = prjTestPoints.t(); _prjTestPoints = prjTestPoints;
    backPrjTestPoints = backPrjTestPoints.t(); _backPrjTestPoints = backPrjTestPoints;

	cvCalcPCA( &_points, &_avg, &_eval, &_evec, CV_PCA_DATA_AS_COL );
	cvProjectPCA( &_testPoints, &_avg, &_evec, &_prjTestPoints );
	cvBackProjectPCA( &_prjTestPoints, &_avg, &_evec, &_backPrjTestPoints );

	err = norm(prjTestPoints, rPrjTestPoints.t(), CV_RELATIVE_L2 );
	if( err > diffPrjEps )
	{
		ts->printf( CvTS::LOG, "bad accuracy of cvProjectPCA() (CV_PCA_DATA_AS_COL); err = %f\n", err );
		code = CvTS::FAIL_BAD_ACCURACY;
		goto exit_func;
	}
	err = norm(backPrjTestPoints, rBackPrjTestPoints.t(), CV_RELATIVE_L2);
	if( err > diffBackPrjEps )
	{
		ts->printf( CvTS::LOG, "bad accuracy of cvBackProjectPCA() (CV_PCA_DATA_AS_COL); err = %f\n", err );
		code = CvTS::FAIL_BAD_ACCURACY;
		goto exit_func;
	}
#endif

exit_func:

	CvRNG* _rng = ts->get_rng(); 
	*_rng = rng.state;
	ts->set_failed_test_info( code );
}

#endif

//CV_PCATest pca_test;
