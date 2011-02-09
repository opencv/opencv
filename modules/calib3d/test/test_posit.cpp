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

#include "test_precomp.hpp"

using namespace cv;
using namespace std;

class CV_POSITTest : public cvtest::BaseTest
{
public:
    CV_POSITTest();
protected:
    void run(int);
};


CV_POSITTest::CV_POSITTest()
{
    test_case_count = 20;
}

void CV_POSITTest::run( int start_from )
{
    int code = cvtest::TS::OK;

    /* fixed parameters output */
    /*float rot[3][3]={  0.49010f,  0.85057f, 0.19063f,
                      -0.56948f,  0.14671f, 0.80880f,
                       0.65997f, -0.50495f, 0.55629f };

    float trans[3] = { 0.0f, 0.0f, 40.02637f };
    */

    /* Some variables */
    int i, counter;

    CvTermCriteria criteria;
    CvPoint3D32f* obj_points;
    CvPoint2D32f* img_points;
    CvPOSITObject* object;

    float angleX, angleY, angleZ;
    RNG& rng = ts->get_rng();
    int progress = 0;

    CvMat* true_rotationX = cvCreateMat( 3, 3, CV_32F );
    CvMat* true_rotationY = cvCreateMat( 3, 3, CV_32F );
    CvMat* true_rotationZ = cvCreateMat( 3, 3, CV_32F );
    CvMat* tmp_matrix = cvCreateMat( 3, 3, CV_32F );
    CvMat* true_rotation = cvCreateMat( 3, 3, CV_32F );
    CvMat* rotation = cvCreateMat( 3, 3, CV_32F );
    CvMat* translation = cvCreateMat( 3, 1, CV_32F );
    CvMat* true_translation = cvCreateMat( 3, 1, CV_32F );

    const float flFocalLength = 760.f;
    const float flEpsilon = 0.5f;

    /* Initilization */
    criteria.type = CV_TERMCRIT_EPS|CV_TERMCRIT_ITER;
    criteria.epsilon = flEpsilon;
    criteria.max_iter = 10000;

    /* Allocating source arrays; */
    obj_points = (CvPoint3D32f*)cvAlloc( 8 * sizeof(CvPoint3D32f) );
    img_points = (CvPoint2D32f*)cvAlloc( 8 * sizeof(CvPoint2D32f) );

    /* Fill points arrays with values */

    /* cube model with edge size 10 */
    obj_points[0].x = 0;  obj_points[0].y = 0;  obj_points[0].z = 0;
    obj_points[1].x = 10; obj_points[1].y = 0;  obj_points[1].z = 0;
    obj_points[2].x = 10; obj_points[2].y = 10; obj_points[2].z = 0;
    obj_points[3].x = 0;  obj_points[3].y = 10; obj_points[3].z = 0;
    obj_points[4].x = 0;  obj_points[4].y = 0;  obj_points[4].z = 10;
    obj_points[5].x = 10; obj_points[5].y = 0;  obj_points[5].z = 10;
    obj_points[6].x = 10; obj_points[6].y = 10; obj_points[6].z = 10;
    obj_points[7].x = 0;  obj_points[7].y = 10; obj_points[7].z = 10;

    /* Loop for test some random object positions */
    for( counter = start_from; counter < test_case_count; counter++ )
    {
        ts->update_context( this, counter, true );
        progress = update_progress( progress, counter, test_case_count, 0 );
        
        /* set all rotation matrix to zero */
        cvZero( true_rotationX );
        cvZero( true_rotationY );
        cvZero( true_rotationZ );
        
        /* fill random rotation matrix */
        angleX = (float)(cvtest::randReal(rng)*2*CV_PI);
        angleY = (float)(cvtest::randReal(rng)*2*CV_PI);
        angleZ = (float)(cvtest::randReal(rng)*2*CV_PI);

        true_rotationX->data.fl[0 *3+ 0] = 1;
        true_rotationX->data.fl[1 *3+ 1] = (float)cos(angleX);
        true_rotationX->data.fl[2 *3+ 2] = true_rotationX->data.fl[1 *3+ 1];
        true_rotationX->data.fl[1 *3+ 2] = -(float)sin(angleX);
        true_rotationX->data.fl[2 *3+ 1] = -true_rotationX->data.fl[1 *3+ 2];

        true_rotationY->data.fl[1 *3+ 1] = 1;
        true_rotationY->data.fl[0 *3+ 0] = (float)cos(angleY);
        true_rotationY->data.fl[2 *3+ 2] = true_rotationY->data.fl[0 *3+ 0];
        true_rotationY->data.fl[0 *3+ 2] = -(float)sin(angleY);
        true_rotationY->data.fl[2 *3+ 0] = -true_rotationY->data.fl[0 *3+ 2];

        true_rotationZ->data.fl[2 *3+ 2] = 1;
        true_rotationZ->data.fl[0 *3+ 0] = (float)cos(angleZ);
        true_rotationZ->data.fl[1 *3+ 1] = true_rotationZ->data.fl[0 *3+ 0];
        true_rotationZ->data.fl[0 *3+ 1] = -(float)sin(angleZ);
        true_rotationZ->data.fl[1 *3+ 0] = -true_rotationZ->data.fl[0 *3+ 1];

        cvMatMul( true_rotationX, true_rotationY, tmp_matrix);
        cvMatMul( tmp_matrix, true_rotationZ, true_rotation);

        /* fill translation vector */
        true_translation->data.fl[2] = (float)(cvtest::randReal(rng)*(2*flFocalLength-40) + 60);
        true_translation->data.fl[0] = (float)((cvtest::randReal(rng)*2-1)*true_translation->data.fl[2]);
        true_translation->data.fl[1] = (float)((cvtest::randReal(rng)*2-1)*true_translation->data.fl[2]);

        /* calculate perspective projection */
        for ( i = 0; i < 8; i++ )
        {
            float vec[3];
            CvMat Vec = cvMat( 3, 1, CV_32F, vec );
            CvMat Obj_point = cvMat( 3, 1, CV_32F, &obj_points[i].x );

            cvMatMul( true_rotation, &Obj_point, &Vec );

            vec[0] += true_translation->data.fl[0];
            vec[1] += true_translation->data.fl[1];
            vec[2] += true_translation->data.fl[2];

            img_points[i].x = flFocalLength * vec[0] / vec[2];
            img_points[i].y = flFocalLength * vec[1] / vec[2];
        }

        /*img_points[0].x = 0 ; img_points[0].y =   0;
        img_points[1].x = 80; img_points[1].y = -93;
        img_points[2].x = 245;img_points[2].y =  -77;
        img_points[3].x = 185;img_points[3].y =  32;
        img_points[4].x = 32; img_points[4].y = 135;
        img_points[5].x = 99; img_points[5].y = 35;
        img_points[6].x = 247; img_points[6].y = 62;
        img_points[7].x = 195; img_points[7].y = 179;
        */

        object = cvCreatePOSITObject( obj_points, 8 );
        cvPOSIT( object, img_points, flFocalLength, criteria,
                 rotation->data.fl, translation->data.fl );
        cvReleasePOSITObject( &object );

        //Mat _rotation = cvarrToMat(rotation), _true_rotation = cvarrToMat(true_rotation);
        //Mat _translation = cvarrToMat(translation), _true_translation = cvarrToMat(true_translation);
        code = cvtest::cmpEps2( ts, rotation, true_rotation, flEpsilon, false, "rotation matrix" );
        if( code < 0 )
            break;

        code = cvtest::cmpEps2( ts, translation, true_translation, flEpsilon, false, "translation vector" );
        if( code < 0 )
            break;
    }

    cvFree( &obj_points );
    cvFree( &img_points );

    cvReleaseMat( &true_rotationX );
    cvReleaseMat( &true_rotationY );
    cvReleaseMat( &true_rotationZ );
    cvReleaseMat( &tmp_matrix );
    cvReleaseMat( &true_rotation );
    cvReleaseMat( &rotation );
    cvReleaseMat( &translation );
    cvReleaseMat( &true_translation );

    if( code < 0 )
        ts->set_failed_test_info( code );
}

TEST(Calib3d_POSIT, accuracy) { CV_POSITTest test; test.safe_run(); }

/* End of file. */
