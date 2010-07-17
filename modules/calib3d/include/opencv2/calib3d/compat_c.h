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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_CALIB3D_COMPAT_C_H__
#define __OPENCV_CALIB3D_COMPAT_C_H__

#include "opencv2/imgproc/imgproc_c.h"

#ifdef __cplusplus
extern "C" {
#endif

#if !defined __cplusplus && defined _MSC_VER && _MSC_VER >= 1400
    #pragma warning(push)
    #pragma warning(disable: 4100)
#endif

/* Find fundamental matrix */
CV_INLINE  void  cvFindFundamentalMatrix( int* points1, int* points2,
                            int numpoints, int CV_UNREFERENCED(method), float* matrix )
{
    CvMat* pointsMat1;
    CvMat* pointsMat2;
    CvMat fundMatr = cvMat(3,3,CV_32F,matrix);
    int i, curr = 0;

    pointsMat1 = cvCreateMat(3,numpoints,CV_64F);
    pointsMat2 = cvCreateMat(3,numpoints,CV_64F);

    for( i = 0; i < numpoints; i++ )
    {
        cvmSet(pointsMat1,0,i,points1[curr]);//x
        cvmSet(pointsMat1,1,i,points1[curr+1]);//y
        cvmSet(pointsMat1,2,i,1.0);

        cvmSet(pointsMat2,0,i,points2[curr]);//x
        cvmSet(pointsMat2,1,i,points2[curr+1]);//y
        cvmSet(pointsMat2,2,i,1.0);
        curr += 2;
    }

    cvFindFundamentalMat(pointsMat1,pointsMat2,&fundMatr,CV_FM_RANSAC,1,0.99,0);

    cvReleaseMat(&pointsMat1);
    cvReleaseMat(&pointsMat2);
}



CV_INLINE int
cvFindChessBoardCornerGuesses( const void* arr, void* CV_UNREFERENCED(thresharr),
                               CvMemStorage * CV_UNREFERENCED(storage),
                               CvSize pattern_size, CvPoint2D32f * corners,
                               int *corner_count )
{
    return cvFindChessboardCorners( arr, pattern_size, corners,
                                    corner_count, CV_CALIB_CB_ADAPTIVE_THRESH );
}


/* Calibrates camera using multiple views of calibration pattern */
CV_INLINE void cvCalibrateCamera( int image_count, int* _point_counts,
    CvSize image_size, CvPoint2D32f* _image_points, CvPoint3D32f* _object_points,
    float* _distortion_coeffs, float* _camera_matrix, float* _translation_vectors,
    float* _rotation_matrices, int flags )
{
    int i, total = 0;
    CvMat point_counts = cvMat( image_count, 1, CV_32SC1, _point_counts );
    CvMat image_points, object_points;
    CvMat dist_coeffs = cvMat( 4, 1, CV_32FC1, _distortion_coeffs );
    CvMat camera_matrix = cvMat( 3, 3, CV_32FC1, _camera_matrix );
    CvMat rotation_matrices = cvMat( image_count, 9, CV_32FC1, _rotation_matrices );
    CvMat translation_vectors = cvMat( image_count, 3, CV_32FC1, _translation_vectors );

    for( i = 0; i < image_count; i++ )
        total += _point_counts[i];

    image_points = cvMat( total, 1, CV_32FC2, _image_points );
    object_points = cvMat( total, 1, CV_32FC3, _object_points );

    cvCalibrateCamera2( &object_points, &image_points, &point_counts, image_size,
        &camera_matrix, &dist_coeffs, &rotation_matrices, &translation_vectors,
        flags );
}


CV_INLINE void cvCalibrateCamera_64d( int image_count, int* _point_counts,
    CvSize image_size, CvPoint2D64f* _image_points, CvPoint3D64f* _object_points,
    double* _distortion_coeffs, double* _camera_matrix, double* _translation_vectors,
    double* _rotation_matrices, int flags )
{
    int i, total = 0;
    CvMat point_counts = cvMat( image_count, 1, CV_32SC1, _point_counts );
    CvMat image_points, object_points;
    CvMat dist_coeffs = cvMat( 4, 1, CV_64FC1, _distortion_coeffs );
    CvMat camera_matrix = cvMat( 3, 3, CV_64FC1, _camera_matrix );
    CvMat rotation_matrices = cvMat( image_count, 9, CV_64FC1, _rotation_matrices );
    CvMat translation_vectors = cvMat( image_count, 3, CV_64FC1, _translation_vectors );

    for( i = 0; i < image_count; i++ )
        total += _point_counts[i];

    image_points = cvMat( total, 1, CV_64FC2, _image_points );
    object_points = cvMat( total, 1, CV_64FC3, _object_points );

    cvCalibrateCamera2( &object_points, &image_points, &point_counts, image_size,
        &camera_matrix, &dist_coeffs, &rotation_matrices, &translation_vectors,
        flags );
}



/* Find 3d position of object given intrinsic camera parameters,
   3d model of the object and projection of the object into view plane */
CV_INLINE void cvFindExtrinsicCameraParams( int point_count,
    CvSize CV_UNREFERENCED(image_size), CvPoint2D32f* _image_points,
    CvPoint3D32f* _object_points, float* focal_length,
    CvPoint2D32f principal_point, float* _distortion_coeffs,
    float* _rotation_vector, float* _translation_vector )
{
    CvMat image_points = cvMat( point_count, 1, CV_32FC2, _image_points );
    CvMat object_points = cvMat( point_count, 1, CV_32FC3, _object_points );
    CvMat dist_coeffs = cvMat( 4, 1, CV_32FC1, _distortion_coeffs );
    float a[9];
    CvMat camera_matrix = cvMat( 3, 3, CV_32FC1, a );
    CvMat rotation_vector = cvMat( 1, 1, CV_32FC3, _rotation_vector );
    CvMat translation_vector = cvMat( 1, 1, CV_32FC3, _translation_vector );

    a[0] = focal_length[0]; a[4] = focal_length[1];
    a[2] = principal_point.x; a[5] = principal_point.y;
    a[1] = a[3] = a[6] = a[7] = 0.f;
    a[8] = 1.f;

    cvFindExtrinsicCameraParams2( &object_points, &image_points, &camera_matrix,
        &dist_coeffs, &rotation_vector, &translation_vector, 0 );
}


/* Variant of the previous function that takes double-precision parameters */
CV_INLINE void cvFindExtrinsicCameraParams_64d( int point_count,
    CvSize CV_UNREFERENCED(image_size), CvPoint2D64f* _image_points,
    CvPoint3D64f* _object_points, double* focal_length,
    CvPoint2D64f principal_point, double* _distortion_coeffs,
    double* _rotation_vector, double* _translation_vector )
{
    CvMat image_points = cvMat( point_count, 1, CV_64FC2, _image_points );
    CvMat object_points = cvMat( point_count, 1, CV_64FC3, _object_points );
    CvMat dist_coeffs = cvMat( 4, 1, CV_64FC1, _distortion_coeffs );
    double a[9];
    CvMat camera_matrix = cvMat( 3, 3, CV_64FC1, a );
    CvMat rotation_vector = cvMat( 1, 1, CV_64FC3, _rotation_vector );
    CvMat translation_vector = cvMat( 1, 1, CV_64FC3, _translation_vector );

    a[0] = focal_length[0]; a[4] = focal_length[1];
    a[2] = principal_point.x; a[5] = principal_point.y;
    a[1] = a[3] = a[6] = a[7] = 0.;
    a[8] = 1.;

    cvFindExtrinsicCameraParams2( &object_points, &image_points, &camera_matrix,
        &dist_coeffs, &rotation_vector, &translation_vector, 0 );
}


/* Rodrigues transform */
#define CV_RODRIGUES_M2V  0
#define CV_RODRIGUES_V2M  1

/* Converts rotation_matrix matrix to rotation_matrix vector or vice versa */
CV_INLINE void  cvRodrigues( CvMat* rotation_matrix, CvMat* rotation_vector,
                             CvMat* jacobian, int conv_type )
{
    if( conv_type == CV_RODRIGUES_V2M )
        cvRodrigues2( rotation_vector, rotation_matrix, jacobian );
    else
        cvRodrigues2( rotation_matrix, rotation_vector, jacobian );
}


/* Does reprojection of 3d object points to the view plane */
CV_INLINE void  cvProjectPoints( int point_count, CvPoint3D64f* _object_points,
    double* _rotation_vector, double*  _translation_vector,
    double* focal_length, CvPoint2D64f principal_point,
    double* _distortion, CvPoint2D64f* _image_points,
    double* _deriv_points_rotation_matrix,
    double* _deriv_points_translation_vect,
    double* _deriv_points_focal,
    double* _deriv_points_principal_point,
    double* _deriv_points_distortion_coeffs )
{
    CvMat object_points = cvMat( point_count, 1, CV_64FC3, _object_points );
    CvMat image_points = cvMat( point_count, 1, CV_64FC2, _image_points );
    CvMat rotation_vector = cvMat( 3, 1, CV_64FC1, _rotation_vector );
    CvMat translation_vector = cvMat( 3, 1, CV_64FC1, _translation_vector );
    double a[9];
    CvMat camera_matrix = cvMat( 3, 3, CV_64FC1, a );
    CvMat dist_coeffs = cvMat( 4, 1, CV_64FC1, _distortion );
    CvMat dpdr = cvMat( 2*point_count, 3, CV_64FC1, _deriv_points_rotation_matrix );
    CvMat dpdt = cvMat( 2*point_count, 3, CV_64FC1, _deriv_points_translation_vect );
    CvMat dpdf = cvMat( 2*point_count, 2, CV_64FC1, _deriv_points_focal );
    CvMat dpdc = cvMat( 2*point_count, 2, CV_64FC1, _deriv_points_principal_point );
    CvMat dpdk = cvMat( 2*point_count, 4, CV_64FC1, _deriv_points_distortion_coeffs );

    a[0] = focal_length[0]; a[4] = focal_length[1];
    a[2] = principal_point.x; a[5] = principal_point.y;
    a[1] = a[3] = a[6] = a[7] = 0.;
    a[8] = 1.;

    cvProjectPoints2( &object_points, &rotation_vector, &translation_vector,
                      &camera_matrix, &dist_coeffs, &image_points,
                      &dpdr, &dpdt, &dpdf, &dpdc, &dpdk, 0 );
}


/* Simpler version of the previous function */
CV_INLINE void  cvProjectPointsSimple( int point_count, CvPoint3D64f* _object_points,
    double* _rotation_matrix, double*  _translation_vector,
    double* _camera_matrix, double* _distortion, CvPoint2D64f* _image_points )
{
    CvMat object_points = cvMat( point_count, 1, CV_64FC3, _object_points );
    CvMat image_points = cvMat( point_count, 1, CV_64FC2, _image_points );
    CvMat rotation_matrix = cvMat( 3, 3, CV_64FC1, _rotation_matrix );
    CvMat translation_vector = cvMat( 3, 1, CV_64FC1, _translation_vector );
    CvMat camera_matrix = cvMat( 3, 3, CV_64FC1, _camera_matrix );
    CvMat dist_coeffs = cvMat( 4, 1, CV_64FC1, _distortion );

    cvProjectPoints2( &object_points, &rotation_matrix, &translation_vector,
                      &camera_matrix, &dist_coeffs, &image_points,
                      0, 0, 0, 0, 0, 0 );
}


#define cvMake2DPoints cvConvertPointsHomogeneous
#define cvMake3DPoints cvConvertPointsHomogeneous

#define cvWarpPerspectiveQMatrix cvGetPerspectiveTransform

#define cvConvertPointsHomogenious cvConvertPointsHomogeneous

#if !defined __cplusplus && defined _MSC_VER && _MSC_VER >= 1400
    #pragma warning(pop)
#endif

#ifdef __cplusplus
}
#endif

#endif
