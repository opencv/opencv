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

/****************************************************************************************\

   calculate image homography 
   
\****************************************************************************************/

CV_IMPL void
cvCalcImageHomography( float* line, CvPoint3D32f* _center,
                       float* _intrinsic, float* _homography )
{
    double norm_xy, norm_xz, xy_sina, xy_cosa, xz_sina, xz_cosa, nx1, plane_dist;
    float _ry[3], _rz[3], _r_trans[9];
    CvMat rx = cvMat( 1, 3, CV_32F, line );
    CvMat ry = cvMat( 1, 3, CV_32F, _ry );
    CvMat rz = cvMat( 1, 3, CV_32F, _rz );
    CvMat r_trans = cvMat( 3, 3, CV_32F, _r_trans );
    CvMat center = cvMat( 3, 1, CV_32F, _center );

    float _sub[9];
    CvMat sub = cvMat( 3, 3, CV_32F, _sub );
    float _t_trans[3];
    CvMat t_trans = cvMat( 3, 1, CV_32F, _t_trans );

    CvMat intrinsic = cvMat( 3, 3, CV_32F, _intrinsic );
    CvMat homography = cvMat( 3, 3, CV_32F, _homography );

    if( !line || !_center || !_intrinsic || !_homography )
        CV_Error( CV_StsNullPtr, "" );

    norm_xy = cvSqrt( line[0] * line[0] + line[1] * line[1] );
    xy_cosa = line[0] / norm_xy;
    xy_sina = line[1] / norm_xy;

    norm_xz = cvSqrt( line[0] * line[0] + line[2] * line[2] );
    xz_cosa = line[0] / norm_xz;
    xz_sina = line[2] / norm_xz;

    nx1 = -xz_sina;

    _rz[0] = (float)(xy_cosa * nx1);
    _rz[1] = (float)(xy_sina * nx1);
    _rz[2] = (float)xz_cosa;
    cvScale( &rz, &rz, 1./cvNorm(&rz,0,CV_L2) );

    /*  new axe  y  */
    cvCrossProduct( &rz, &rx, &ry );
    cvScale( &ry, &ry, 1./cvNorm( &ry, 0, CV_L2 ) );

    /*  transpone rotation matrix    */
    memcpy( &_r_trans[0], line, 3*sizeof(float));
    memcpy( &_r_trans[3], _ry, 3*sizeof(float));
    memcpy( &_r_trans[6], _rz, 3*sizeof(float));

    /*  calculate center distanse from arm plane  */
    plane_dist = cvDotProduct( &center, &rz );

    /* calculate (I - r_trans)*center */
    cvSetIdentity( &sub );
    cvSub( &sub, &r_trans, &sub );
    cvMatMul( &sub, &center, &t_trans ); 

    cvMatMul( &t_trans, &rz, &sub );
    cvScaleAdd( &sub, cvRealScalar(1./plane_dist), &r_trans, &sub ); /* ? */

    cvMatMul( &intrinsic, &sub, &r_trans );
    cvInvert( &intrinsic, &sub, CV_SVD );
    cvMatMul( &r_trans, &sub, &homography );
}

/* End of file. */

