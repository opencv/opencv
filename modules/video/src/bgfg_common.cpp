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

//  Function cvRefineForegroundMaskBySegm preforms FG post-processing based on segmentation
//    (all pixels of the segment will be classified as FG if majority of pixels of the region are FG).
// parameters:
//      segments - pointer to result of segmentation (for example MeanShiftSegmentation)
//      bg_model - pointer to CvBGStatModel structure
CV_IMPL void cvRefineForegroundMaskBySegm( CvSeq* segments, CvBGStatModel*  bg_model )
{
	IplImage* tmp_image = cvCreateImage(cvSize(bg_model->foreground->width,bg_model->foreground->height),
							IPL_DEPTH_8U, 1);
	for( ; segments; segments = ((CvSeq*)segments)->h_next )
	{
		CvSeq seq = *segments;
		seq.v_next = seq.h_next = NULL;
		cvZero(tmp_image);
		cvDrawContours( tmp_image, &seq, CV_RGB(0, 0, 255), CV_RGB(0, 0, 255), 10, -1);
		int num1 = cvCountNonZero(tmp_image);
        cvAnd(tmp_image, bg_model->foreground, tmp_image);
		int num2 = cvCountNonZero(tmp_image);
		if( num2 > num1*0.5 )
			cvDrawContours( bg_model->foreground, &seq, CV_RGB(0, 0, 255), CV_RGB(0, 0, 255), 10, -1);
		else
			cvDrawContours( bg_model->foreground, &seq, CV_RGB(0, 0, 0), CV_RGB(0, 0, 0), 10, -1);
	}
	cvReleaseImage(&tmp_image);
}



CV_IMPL CvSeq*
cvSegmentFGMask( CvArr* _mask, int poly1Hull0, float perimScale,
                 CvMemStorage* storage, CvPoint offset )
{
    CvMat mstub, *mask = cvGetMat( _mask, &mstub );
    CvMemStorage* tempStorage = storage ? storage : cvCreateMemStorage();
    CvSeq *contours, *c;
    int nContours = 0;
    CvContourScanner scanner;
    
    // clean up raw mask
    cvMorphologyEx( mask, mask, 0, 0, CV_MOP_OPEN, 1 );
    cvMorphologyEx( mask, mask, 0, 0, CV_MOP_CLOSE, 1 );

    // find contours around only bigger regions
    scanner = cvStartFindContours( mask, tempStorage,
        sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, offset );
    
    while( (c = cvFindNextContour( scanner )) != 0 ) 
    {
        double len = cvContourPerimeter( c );
        double q = (mask->rows + mask->cols)/perimScale; // calculate perimeter len threshold
        if( len < q ) //Get rid of blob if it's perimeter is too small
            cvSubstituteContour( scanner, 0 );
        else //Smooth it's edges if it's large enough
        {
            CvSeq* newC;
            if( poly1Hull0 ) //Polygonal approximation of the segmentation 
                newC = cvApproxPoly( c, sizeof(CvContour), tempStorage, CV_POLY_APPROX_DP, 2, 0 ); 
            else //Convex Hull of the segmentation
                newC = cvConvexHull2( c, tempStorage, CV_CLOCKWISE, 1 );
            cvSubstituteContour( scanner, newC );
            nContours++;
        }
    }
    contours = cvEndFindContours( &scanner );

    // paint the found regions back into the image
    cvZero( mask );
    for( c=contours; c != 0; c = c->h_next ) 
        cvDrawContours( mask, c, cvScalarAll(255), cvScalarAll(0), -1, CV_FILLED, 8,
            cvPoint(-offset.x,-offset.y));

    if( tempStorage != storage )
    {
        cvReleaseMemStorage( &tempStorage );
        contours = 0;
    }

    return contours;
}

/* End of file. */

