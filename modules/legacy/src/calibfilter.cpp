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
#include <stdio.h>

#undef quad

#if _MSC_VER >= 1200
#pragma warning( disable: 4701 )
#endif

CvCalibFilter::CvCalibFilter()
{
    /* etalon data */
    etalonType = CV_CALIB_ETALON_USER;
    etalonParamCount = 0;
    etalonParams = 0;
    etalonPointCount = 0;
    etalonPoints = 0;

    /* camera data */
    cameraCount = 1;

    memset( points, 0, sizeof(points));
    memset( undistMap, 0, sizeof(undistMap));
    undistImg = 0;
    memset( latestCounts, 0, sizeof(latestCounts));
    memset( latestPoints, 0, sizeof(latestPoints));
    memset( &stereo, 0, sizeof(stereo) );
    maxPoints = 0;
    framesTotal = 15;
    framesAccepted = 0;
    isCalibrated = false;

    imgSize = cvSize(0,0);
    grayImg = 0;
    tempImg = 0;
    storage = 0;

    memset( rectMap, 0, sizeof(rectMap));
}


CvCalibFilter::~CvCalibFilter()
{
    SetCameraCount(0);
    cvFree( &etalonParams );
    cvFree( &etalonPoints );
    cvReleaseMat( &grayImg );
    cvReleaseMat( &tempImg );
    cvReleaseMat( &undistImg );
    cvReleaseMemStorage( &storage );
}


bool CvCalibFilter::SetEtalon( CvCalibEtalonType type, double* params,
                               int pointCount, CvPoint2D32f* points )
{
    int i, arrSize;

    Stop();

	if (latestPoints != NULL)
	{
		for( i = 0; i < MAX_CAMERAS; i++ )
			cvFree( latestPoints + i );
	}

    if( type == CV_CALIB_ETALON_USER || type != etalonType )
    {
		if (etalonParams != NULL)
		{
			cvFree( &etalonParams );
		}
    }

    etalonType = type;

    switch( etalonType )
    {
    case CV_CALIB_ETALON_CHESSBOARD:
        etalonParamCount = 3;
        if( !params || cvRound(params[0]) != params[0] || params[0] < 3 ||
            cvRound(params[1]) != params[1] || params[1] < 3 || params[2] <= 0 )
        {
            assert(0);
            return false;
        }

        pointCount = cvRound((params[0] - 1)*(params[1] - 1));
        break;

    case CV_CALIB_ETALON_USER:
        etalonParamCount = 0;

        if( !points || pointCount < 4 )
        {
            assert(0);
            return false;
        }
        break;

    default:
        assert(0);
        return false;
    }

    if( etalonParamCount > 0 )
    {
        arrSize = etalonParamCount * sizeof(etalonParams[0]);
        etalonParams = (double*)cvAlloc( arrSize );
    }

    arrSize = pointCount * sizeof(etalonPoints[0]);

    if( etalonPointCount != pointCount )
    {
		if (etalonPoints != NULL)
		{
			cvFree( &etalonPoints );
		}
        etalonPointCount = pointCount;
        etalonPoints = (CvPoint2D32f*)cvAlloc( arrSize );
    }

    switch( etalonType )
    {
    case CV_CALIB_ETALON_CHESSBOARD:
        {
            int etalonWidth = cvRound( params[0] ) - 1;
            int etalonHeight = cvRound( params[1] ) - 1;
            int x, y, k = 0;

            etalonParams[0] = etalonWidth;
            etalonParams[1] = etalonHeight;
            etalonParams[2] = params[2];

            for( y = 0; y < etalonHeight; y++ )
                for( x = 0; x < etalonWidth; x++ )
                {
                    etalonPoints[k++] = cvPoint2D32f( (etalonWidth - 1 - x)*params[2],
                                                      y*params[2] );
                }
        }
        break;

    case CV_CALIB_ETALON_USER:
		if (params != NULL)
		{
			memcpy( etalonParams, params, arrSize );
		}
		if (points != NULL)
		{
			memcpy( etalonPoints, points, arrSize );
		}
		break;

    default:
        assert(0);
        return false;
    }

    return true;
}


CvCalibEtalonType
CvCalibFilter::GetEtalon( int* paramCount, const double** params,
                          int* pointCount, const CvPoint2D32f** points ) const
{
    if( paramCount )
        *paramCount = etalonParamCount;

    if( params )
        *params = etalonParams;

    if( pointCount )
        *pointCount = etalonPointCount;

    if( points )
        *points = etalonPoints;

    return etalonType;
}


void CvCalibFilter::SetCameraCount( int count )
{
    Stop();
    
    if( count != cameraCount )
    {
        for( int i = 0; i < cameraCount; i++ )
        {
            cvFree( points + i );
            cvFree( latestPoints + i );
            cvReleaseMat( &undistMap[i][0] );
            cvReleaseMat( &undistMap[i][1] );
            cvReleaseMat( &rectMap[i][0] );
            cvReleaseMat( &rectMap[i][1] );
        }

        memset( latestCounts, 0, sizeof(latestPoints) );
        maxPoints = 0;
        cameraCount = count;
    }
}

   
bool CvCalibFilter::SetFrames( int frames )
{
    if( frames < 5 )
    {
        assert(0);
        return false;
    }
    
    framesTotal = frames;
    return true;
}


void CvCalibFilter::Stop( bool calibrate )
{
    int i, j;
    isCalibrated = false;

    // deallocate undistortion maps
    for( i = 0; i < cameraCount; i++ )
    {
        cvReleaseMat( &undistMap[i][0] );
        cvReleaseMat( &undistMap[i][1] );
        cvReleaseMat( &rectMap[i][0] );
        cvReleaseMat( &rectMap[i][1] );
    }

    if( calibrate && framesAccepted > 0 )
    {
        int n = framesAccepted;
        CvPoint3D32f* buffer =
            (CvPoint3D32f*)cvAlloc( n * etalonPointCount * sizeof(buffer[0]));
        CvMat mat;
        float* rotMatr = (float*)cvAlloc( n * 9 * sizeof(rotMatr[0]));
        float* transVect = (float*)cvAlloc( n * 3 * sizeof(transVect[0]));
        int* counts = (int*)cvAlloc( n * sizeof(counts[0]));

        cvInitMatHeader( &mat, 1, sizeof(CvCamera)/sizeof(float), CV_32FC1, 0 );
        memset( cameraParams, 0, cameraCount * sizeof(cameraParams[0]));

        for( i = 0; i < framesAccepted; i++ )
        {
            counts[i] = etalonPointCount;
            for( j = 0; j < etalonPointCount; j++ )
                buffer[i * etalonPointCount + j] = cvPoint3D32f( etalonPoints[j].x,
                                                                 etalonPoints[j].y, 0 );
        }

        for( i = 0; i < cameraCount; i++ )
        {
            cvCalibrateCamera( framesAccepted, counts,
                               imgSize, points[i], buffer,
                               cameraParams[i].distortion,
                               cameraParams[i].matrix,
                               transVect, rotMatr, 0 );

            cameraParams[i].imgSize[0] = (float)imgSize.width;
            cameraParams[i].imgSize[1] = (float)imgSize.height;
            
//            cameraParams[i].focalLength[0] = cameraParams[i].matrix[0];
//            cameraParams[i].focalLength[1] = cameraParams[i].matrix[4];

//            cameraParams[i].principalPoint[0] = cameraParams[i].matrix[2];
//            cameraParams[i].principalPoint[1] = cameraParams[i].matrix[5];

            memcpy( cameraParams[i].rotMatr, rotMatr, 9 * sizeof(rotMatr[0]));
            memcpy( cameraParams[i].transVect, transVect, 3 * sizeof(transVect[0]));

            mat.data.ptr = (uchar*)(cameraParams + i);
            
            /* check resultant camera parameters: if there are some INF's or NAN's,
               stop and reset results */
            if( !cvCheckArr( &mat, CV_CHECK_RANGE | CV_CHECK_QUIET, -10000, 10000 ))
                break;
        }



        isCalibrated = i == cameraCount;

        {/* calibrate stereo cameras */
            if( cameraCount == 2 )
            {
                stereo.camera[0] = &cameraParams[0];
                stereo.camera[1] = &cameraParams[1];

                icvStereoCalibration( framesAccepted, counts,
                                   imgSize,
                                   points[0],points[1],
                                   buffer,
                                   &stereo);

                for( i = 0; i < 9; i++ )
                {
                    stereo.fundMatr[i] = stereo.fundMatr[i];
                }
                
            }

        }

        cvFree( &buffer );
        cvFree( &counts );
        cvFree( &rotMatr );
        cvFree( &transVect );
    }

    framesAccepted = 0;
}


bool CvCalibFilter::FindEtalon( IplImage** imgs )
{
    return FindEtalon( (CvMat**)imgs );
}


bool CvCalibFilter::FindEtalon( CvMat** mats )
{
    bool result = true;

    if( !mats || etalonPointCount == 0 )
    {
        assert(0);
        result = false;
    }

    if( result )
    {
        int i, tempPointCount0 = etalonPointCount*2;

        for( i = 0; i < cameraCount; i++ )
        {
            if( !latestPoints[i] )
                latestPoints[i] = (CvPoint2D32f*)
                    cvAlloc( tempPointCount0*2*sizeof(latestPoints[0]));
        }

        for( i = 0; i < cameraCount; i++ )
        {
            CvSize size;
            int tempPointCount = tempPointCount0;
            bool found = false;

            if( !CV_IS_MAT(mats[i]) && !CV_IS_IMAGE(mats[i]))
            {
                assert(0);
                break;
            }

            size = cvGetSize(mats[i]);

            if( size.width != imgSize.width || size.height != imgSize.height )
            {
                imgSize = size;
            }

            if( !grayImg || grayImg->width != imgSize.width ||
                grayImg->height != imgSize.height )
            {
                cvReleaseMat( &grayImg );
                cvReleaseMat( &tempImg );
                grayImg = cvCreateMat( imgSize.height, imgSize.width, CV_8UC1 );
                tempImg = cvCreateMat( imgSize.height, imgSize.width, CV_8UC1 );
            }

            if( !storage )
                storage = cvCreateMemStorage();

            switch( etalonType )
            {
            case CV_CALIB_ETALON_CHESSBOARD:
                if( CV_MAT_CN(cvGetElemType(mats[i])) == 1 )
                    cvCopy( mats[i], grayImg );
                else
                    cvCvtColor( mats[i], grayImg, CV_BGR2GRAY );
                found = cvFindChessBoardCornerGuesses( grayImg, tempImg, storage,
                                                       cvSize( cvRound(etalonParams[0]),
                                                       cvRound(etalonParams[1])),
                                                       latestPoints[i], &tempPointCount ) != 0;
                if( found )
                    cvFindCornerSubPix( grayImg, latestPoints[i], tempPointCount,
                                        cvSize(5,5), cvSize(-1,-1),
                                        cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,10,0.1));
                break;
            default:
                assert(0);
                result = false;
                break;
            }

            latestCounts[i] = found ? tempPointCount : -tempPointCount;
            result = result && found;
        }
    }

    if( storage )
        cvClearMemStorage( storage );

    return result;
}


bool CvCalibFilter::Push( const CvPoint2D32f** pts )
{
    bool result = true;
    int i, newMaxPoints = etalonPointCount*(MAX(framesAccepted,framesTotal) + 1);

    isCalibrated = false;

    if( !pts )
    {
        for( i = 0; i < cameraCount; i++ )
            if( latestCounts[i] <= 0 )
                return false;
        pts = (const CvPoint2D32f**)latestPoints;
    }

    for( i = 0; i < cameraCount; i++ )
    {
        if( !pts[i] )
        {
            assert(0);
            break;
        }

        if( maxPoints < newMaxPoints )
        {
            CvPoint2D32f* prev = points[i];
            cvFree( points + i );
            points[i] = (CvPoint2D32f*)cvAlloc( newMaxPoints * sizeof(prev[0]));
            memcpy( points[i], prev, maxPoints * sizeof(prev[0]));
        }

        memcpy( points[i] + framesAccepted*etalonPointCount, pts[i],
                etalonPointCount*sizeof(points[0][0]));
    }

    if( maxPoints < newMaxPoints )
        maxPoints = newMaxPoints;

    result = i == cameraCount;

    if( ++framesAccepted >= framesTotal )
        Stop( true );
    return result;
}


bool CvCalibFilter::GetLatestPoints( int idx, CvPoint2D32f** pts,
                                     int* count, bool* found )
{
    int n;
    
    if( (unsigned)idx >= (unsigned)cameraCount ||
        !pts || !count || !found )
    {
        assert(0);
        return false;
    }
    
    n = latestCounts[idx];
    
    *found = n > 0;
    *count = abs(n);
    *pts = latestPoints[idx];

    return true;
}


void CvCalibFilter::DrawPoints( IplImage** dst )
{
    DrawPoints( (CvMat**)dst );
}


void CvCalibFilter::DrawPoints( CvMat** dstarr )
{
    int i, j;

    if( !dstarr )
    {
        assert(0);
        return;
    }

    if( latestCounts )
    {
        for( i = 0; i < cameraCount; i++ )
        {
            if( dstarr[i] && latestCounts[i] )
            {
                CvMat dst_stub, *dst;
                int count = 0;
                bool found = false;
                CvPoint2D32f* pts = 0;

                GetLatestPoints( i, &pts, &count, &found );

                dst = cvGetMat( dstarr[i], &dst_stub );

                static const CvScalar line_colors[] =
                {
                    {{0,0,255}},
                    {{0,128,255}},
                    {{0,200,200}},
                    {{0,255,0}},
                    {{200,200,0}},
                    {{255,0,0}},
                    {{255,0,255}}
                };

                const int colorCount = sizeof(line_colors)/sizeof(line_colors[0]);
                const int r = 4;
                CvScalar color = line_colors[0];
                CvPoint prev_pt = { 0, 0};

                for( j = 0; j < count; j++ )
                {
                    CvPoint pt;
                    pt.x = cvRound(pts[j].x);
                    pt.y = cvRound(pts[j].y);

                    if( found )
                    {
                        if( etalonType == CV_CALIB_ETALON_CHESSBOARD )
                            color = line_colors[(j/cvRound(etalonParams[0]))%colorCount];
                        else
                            color = CV_RGB(0,255,0);

                        if( j != 0 )
                            cvLine( dst, prev_pt, pt, color, 1, CV_AA );
                    }

                    cvLine( dst, cvPoint( pt.x - r, pt.y - r ),
                            cvPoint( pt.x + r, pt.y + r ), color, 1, CV_AA );

                    cvLine( dst, cvPoint( pt.x - r, pt.y + r),
                            cvPoint( pt.x + r, pt.y - r), color, 1, CV_AA );

                    cvCircle( dst, pt, r+1, color, 1, CV_AA );

                    prev_pt = pt;
                }
            }
        }
    }
}


/* Get total number of frames and already accepted pair of frames */
int CvCalibFilter::GetFrameCount( int* total ) const
{
    if( total )
        *total = framesTotal;

    return framesAccepted;
}


/* Get camera parameters for specified camera. If camera is not calibrated
   the function returns 0 */
const CvCamera* CvCalibFilter::GetCameraParams( int idx ) const
{
    if( (unsigned)idx >= (unsigned)cameraCount )
    {
        assert(0);
        return 0;
    }
    
    return isCalibrated ? cameraParams + idx : 0;
}


/* Get camera parameters for specified camera. If camera is not calibrated
   the function returns 0 */
const CvStereoCamera* CvCalibFilter::GetStereoParams() const
{
    if( !(isCalibrated && cameraCount == 2) )
    {
        assert(0);
        return 0;
    }
    
    return &stereo;
}


/* Sets camera parameters for all cameras */
bool CvCalibFilter::SetCameraParams( CvCamera* params )
{
    CvMat mat;
    int arrSize;
    
    Stop();
    
    if( !params )
    {
        assert(0);
        return false;
    }

    arrSize = cameraCount * sizeof(params[0]);

    cvInitMatHeader( &mat, 1, cameraCount * (arrSize/sizeof(float)),
                     CV_32FC1, params );
    cvCheckArr( &mat, CV_CHECK_RANGE, -10000, 10000 );

    memcpy( cameraParams, params, arrSize );
    isCalibrated = true;

    return true;
}


bool CvCalibFilter::SaveCameraParams( const char* filename )
{
    if( isCalibrated )
    {
        int i, j;
        
        FILE* f = fopen( filename, "w" );

        if( !f ) return false;

        fprintf( f, "%d\n\n", cameraCount );

        for( i = 0; i < cameraCount; i++ )
        {
            for( j = 0; j < (int)(sizeof(cameraParams[i])/sizeof(float)); j++ )
            {
                fprintf( f, "%15.10f ", ((float*)(cameraParams + i))[j] );
            }
            fprintf( f, "\n\n" );
        }

        /* Save stereo params */

        /* Save quad */
        for( i = 0; i < 2; i++ )
        {
            for( j = 0; j < 4; j++ )
            {
                fprintf(f, "%15.10f ", stereo.quad[i][j].x );
                fprintf(f, "%15.10f ", stereo.quad[i][j].y );
            }
            fprintf(f, "\n");
        }

        /* Save coeffs */
        for( i = 0; i < 2; i++ )
        {
            for( j = 0; j < 9; j++ )
            {
                fprintf(f, "%15.10lf ", stereo.coeffs[i][j/3][j%3] );
            }
            fprintf(f, "\n");
        }


        fclose(f);
        return true;
    }

    return true;
}


bool CvCalibFilter::LoadCameraParams( const char* filename )
{
    int i, j;
    int d = 0;
    FILE* f = fopen( filename, "r" );

    isCalibrated = false;

    if( !f ) return false;

    if( fscanf( f, "%d", &d ) != 1 || d <= 0 || d > 10 )
        return false;

    SetCameraCount( d );
    
    for( i = 0; i < cameraCount; i++ )
    {
        for( j = 0; j < (int)(sizeof(cameraParams[i])/sizeof(float)); j++ )
        {
            fscanf( f, "%f", &((float*)(cameraParams + i))[j] );
        }
    }


    /* Load stereo params */

    /* load quad */
    for( i = 0; i < 2; i++ )
    {
        for( j = 0; j < 4; j++ )
        {
            fscanf(f, "%f ", &(stereo.quad[i][j].x) );
            fscanf(f, "%f ", &(stereo.quad[i][j].y) );
        }
    }

    /* Load coeffs */
    for( i = 0; i < 2; i++ )
    {
        for( j = 0; j < 9; j++ )
        {
            fscanf(f, "%lf ", &(stereo.coeffs[i][j/3][j%3]) );
        }
    }
    
    
    
    
    fclose(f);

    stereo.warpSize = cvSize( cvRound(cameraParams[0].imgSize[0]), cvRound(cameraParams[0].imgSize[1]));

    isCalibrated = true;
    
    return true;
}


bool CvCalibFilter::Rectify( IplImage** srcarr, IplImage** dstarr )
{
    return Rectify( (CvMat**)srcarr, (CvMat**)dstarr );
}

bool CvCalibFilter::Rectify( CvMat** srcarr, CvMat** dstarr )
{
    int i;

    if( !srcarr || !dstarr )
    {
        assert(0);
        return false;
    }

    if( isCalibrated && cameraCount == 2 )
    {
        for( i = 0; i < cameraCount; i++ )
        {
            if( srcarr[i] && dstarr[i] )
            {
                IplImage src_stub, *src;
                IplImage dst_stub, *dst;

                src = cvGetImage( srcarr[i], &src_stub );
                dst = cvGetImage( dstarr[i], &dst_stub );

                if( src->imageData == dst->imageData )
                {
                    if( !undistImg ||
                        undistImg->width != src->width ||
                        undistImg->height != src->height ||
                        CV_MAT_CN(undistImg->type) != src->nChannels )
                    {
                        cvReleaseMat( &undistImg );
                        undistImg = cvCreateMat( src->height, src->width,
                                                 CV_8U + (src->nChannels-1)*8 );
                    }
                    cvCopy( src, undistImg );
                    src = cvGetImage( undistImg, &src_stub );
                }

                cvZero( dst );

                if( !rectMap[i][0] || rectMap[i][0]->width != src->width ||
                    rectMap[i][0]->height != src->height )
                {
                    cvReleaseMat( &rectMap[i][0] );
                    cvReleaseMat( &rectMap[i][1] );
                    rectMap[i][0] = cvCreateMat(stereo.warpSize.height,stereo.warpSize.width,CV_32FC1);
                    rectMap[i][1] = cvCreateMat(stereo.warpSize.height,stereo.warpSize.width,CV_32FC1);
                    cvComputePerspectiveMap(stereo.coeffs[i], rectMap[i][0], rectMap[i][1]);
                }
                cvRemap( src, dst, rectMap[i][0], rectMap[i][1] );
            }
        }
    }
    else
    {
        for( i = 0; i < cameraCount; i++ )
        {
            if( srcarr[i] != dstarr[i] )
                cvCopy( srcarr[i], dstarr[i] );
        }
    }

    return true;
}

bool CvCalibFilter::Undistort( IplImage** srcarr, IplImage** dstarr )
{
    return Undistort( (CvMat**)srcarr, (CvMat**)dstarr );
}


bool CvCalibFilter::Undistort( CvMat** srcarr, CvMat** dstarr )
{
    int i;

    if( !srcarr || !dstarr )
    {
        assert(0);
        return false;
    }

    if( isCalibrated )
    {
        for( i = 0; i < cameraCount; i++ )
        {
            if( srcarr[i] && dstarr[i] )
            {
                CvMat src_stub, *src;
                CvMat dst_stub, *dst;

                src = cvGetMat( srcarr[i], &src_stub );
                dst = cvGetMat( dstarr[i], &dst_stub );

                if( src->data.ptr == dst->data.ptr )
                {
                    if( !undistImg || undistImg->width != src->width ||
                        undistImg->height != src->height ||
                        CV_ARE_TYPES_EQ( undistImg, src ))
                    {
                        cvReleaseMat( &undistImg );
                        undistImg = cvCreateMat( src->height, src->width, src->type );
                    }

                    cvCopy( src, undistImg );
                    src = undistImg;
                }

            #if 1
                {
                CvMat A = cvMat( 3, 3, CV_32FC1, cameraParams[i].matrix );
                CvMat k = cvMat( 1, 4, CV_32FC1, cameraParams[i].distortion );

                if( !undistMap[i][0] || undistMap[i][0]->width != src->width ||
                     undistMap[i][0]->height != src->height )
                {
                    cvReleaseMat( &undistMap[i][0] );
                    cvReleaseMat( &undistMap[i][1] );
                    undistMap[i][0] = cvCreateMat( src->height, src->width, CV_32FC1 );
                    undistMap[i][1] = cvCreateMat( src->height, src->width, CV_32FC1 );
                    cvInitUndistortMap( &A, &k, undistMap[i][0], undistMap[i][1] );
                }

                cvRemap( src, dst, undistMap[i][0], undistMap[i][1] );
            #else
                cvUndistort2( src, dst, &A, &k );
            #endif
                }
            }
        }
    }
    else
    {
        for( i = 0; i < cameraCount; i++ )
        {
            if( srcarr[i] != dstarr[i] )
                cvCopy( srcarr[i], dstarr[i] );
        }
    }


    return true;
}

 	  	 
