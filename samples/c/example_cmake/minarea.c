#ifdef _CH_
#pragma package <opencv>
#endif

#ifndef _EiC
#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <stdlib.h>
#endif

#define ARRAY  1

void help()
{
	printf("\nThis program demonstrates finding the minimum enclosing box or circle of a set\n"
		"of points using functions: minAreaRect() minEnclosingCircle().\n"
		"Random points are generated and then enclosed.\n"
		"Call:\n"
		"./minarea\n");
}

int main( int argc, char** argv )
{
    IplImage* img = cvCreateImage( cvSize( 500, 500 ), 8, 3 );
#if !ARRAY        
    CvMemStorage* storage = cvCreateMemStorage(0);
#endif
    help();
    cvNamedWindow( "rect & circle", 1 );
        
    for(;;)
    {
        char key;
        int i, count = rand()%100 + 1;
        CvPoint pt0, pt;
        CvBox2D box;
        CvPoint2D32f box_vtx[4];
        CvPoint2D32f center;
        CvPoint icenter;
        float radius;
#if !ARRAY            
        CvSeq* ptseq = cvCreateSeq( CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvContour),
                                     sizeof(CvPoint), storage );
        for( i = 0; i < count; i++ )
        {
            pt0.x = rand() % (img->width/2) + img->width/4;
            pt0.y = rand() % (img->height/2) + img->height/4;
            cvSeqPush( ptseq, &pt0 );
        }
#ifndef _EiC /* unfortunately, here EiC crashes */
        box = cvMinAreaRect2( ptseq, 0 );
#endif
        cvMinEnclosingCircle( ptseq, &center, &radius );
#else
        CvPoint* points = (CvPoint*)malloc( count * sizeof(points[0]));
        CvMat pointMat = cvMat( 1, count, CV_32SC2, points );

        for( i = 0; i < count; i++ )
        {
            pt0.x = rand() % (img->width/2) + img->width/4;
            pt0.y = rand() % (img->height/2) + img->height/4;
            points[i] = pt0;
        }
#ifndef _EiC
        box = cvMinAreaRect2( &pointMat, 0 );
#endif
        cvMinEnclosingCircle( &pointMat, &center, &radius );
#endif
        cvBoxPoints( box, box_vtx );
        cvZero( img );
        for( i = 0; i < count; i++ )
        {
#if !ARRAY                
            pt0 = *CV_GET_SEQ_ELEM( CvPoint, ptseq, i );
#else
            pt0 = points[i];
#endif
            cvCircle( img, pt0, 2, CV_RGB( 255, 0, 0 ), CV_FILLED, CV_AA, 0 );
        }

#ifndef _EiC
        pt0.x = cvRound(box_vtx[3].x);
        pt0.y = cvRound(box_vtx[3].y);
        for( i = 0; i < 4; i++ )
        {
            pt.x = cvRound(box_vtx[i].x);
            pt.y = cvRound(box_vtx[i].y);
            cvLine(img, pt0, pt, CV_RGB(0, 255, 0), 1, CV_AA, 0);
            pt0 = pt;
        }
#endif
        icenter.x = cvRound(center.x);
        icenter.y = cvRound(center.y);
        cvCircle( img, icenter, cvRound(radius), CV_RGB(255, 255, 0), 1, CV_AA, 0 );

        cvShowImage( "rect & circle", img );

        key = (char) cvWaitKey(0);
        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
            break;

#if !ARRAY
        cvClearMemStorage( storage );
#else
        free( points );
#endif
    }
    
    cvDestroyWindow( "rect & circle" );
    return 0;
}

#ifdef _EiC
main(1,"convexhull.c");
#endif

