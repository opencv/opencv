#include <opencv2/imgproc/imgproc_c.h>
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
void help()
{
	printf("\nThis program creates an image to demonstrate the use of the \"c\" contour\n"
			"functions: cvFindContours() and cvApproxPoly() along with the storage\n"
			"functions cvCreateMemStorage() and cvDrawContours().\n"
			"It also shows the use of a trackbar to control contour retrieval.\n"
			"\n"
            "Usage :\n"
			"./contours\n");
}

#define w 500
int levels = 3;
CvSeq* contours = 0;

void on_trackbar(int pos)
{
    IplImage* cnt_img = cvCreateImage( cvSize(w,w), 8, 3 );
    CvSeq* _contours = contours;
    int _levels = levels - 3;
    if( _levels <= 0 ) // get to the nearest face to make it look more funny
        _contours = _contours->h_next->h_next->h_next;
    cvZero( cnt_img );
    cvDrawContours( cnt_img, _contours, CV_RGB(255,0,0), CV_RGB(0,255,0), _levels, 3, CV_AA, cvPoint(0,0) );
    cvShowImage( "contours", cnt_img );
    cvReleaseImage( &cnt_img );
}

static void findCComp( IplImage* img )
{
    int x, y, cidx = 1;
    IplImage* mask = cvCreateImage( cvSize(img->width+2, img->height+2), 8, 1 );
    cvZero(mask);
    cvRectangle( mask, cvPoint(0, 0), cvPoint(mask->width-1, mask->height-1),
                 cvScalarAll(1), 1, 8, 0 );
    
    for( y = 0; y < img->height; y++ )
        for( x = 0; x < img->width; x++ )
        {
            if( CV_IMAGE_ELEM(mask, uchar, y+1, x+1) != 0 )
                continue;
            cvFloodFill(img, cvPoint(x,y), cvScalarAll(cidx),
                        cvScalarAll(0), cvScalarAll(0), 0, 4, mask);
            cidx++;
        }
}


int main( int argc, char** argv )
{
    int i, j;
    CvMemStorage* storage = cvCreateMemStorage(0);
    IplImage* img = cvCreateImage( cvSize(w,w), 8, 1 );
    IplImage* img32f = cvCreateImage( cvSize(w,w), IPL_DEPTH_32F, 1 );
    IplImage* img32s = cvCreateImage( cvSize(w,w), IPL_DEPTH_32S, 1 );
    IplImage* img3 = cvCreateImage( cvSize(w,w), 8, 3 );
    help();
    cvZero( img );

    for( i=0; i < 6; i++ )
    {
        int dx = (i%2)*250 - 30;
        int dy = (i/2)*150;
        CvScalar white = cvRealScalar(255);
        CvScalar black = cvRealScalar(0);

        if( i == 0 )
        {
            for( j = 0; j <= 10; j++ )
            {
                double angle = (j+5)*CV_PI/21;
                cvLine(img, cvPoint(cvRound(dx+100+j*10-80*cos(angle)),
                    cvRound(dy+100-90*sin(angle))),
                    cvPoint(cvRound(dx+100+j*10-30*cos(angle)),
                    cvRound(dy+100-30*sin(angle))), white, 3, 8, 0);
            }
        }

        cvEllipse( img, cvPoint(dx+150, dy+100), cvSize(100,70), 0, 0, 360, white, -1, 8, 0 );
        cvEllipse( img, cvPoint(dx+115, dy+70), cvSize(30,20), 0, 0, 360, black, -1, 8, 0 );
        cvEllipse( img, cvPoint(dx+185, dy+70), cvSize(30,20), 0, 0, 360, black, -1, 8, 0 );
        cvEllipse( img, cvPoint(dx+115, dy+70), cvSize(15,15), 0, 0, 360, white, -1, 8, 0 );
        cvEllipse( img, cvPoint(dx+185, dy+70), cvSize(15,15), 0, 0, 360, white, -1, 8, 0 );
        cvEllipse( img, cvPoint(dx+115, dy+70), cvSize(5,5), 0, 0, 360, black, -1, 8, 0 );
        cvEllipse( img, cvPoint(dx+185, dy+70), cvSize(5,5), 0, 0, 360, black, -1, 8, 0 );
        cvEllipse( img, cvPoint(dx+150, dy+100), cvSize(10,5), 0, 0, 360, black, -1, 8, 0 );
        cvEllipse( img, cvPoint(dx+150, dy+150), cvSize(40,10), 0, 0, 360, black, -1, 8, 0 );
        cvEllipse( img, cvPoint(dx+27, dy+100), cvSize(20,35), 0, 0, 360, white, -1, 8, 0 );
        cvEllipse( img, cvPoint(dx+273, dy+100), cvSize(20,35), 0, 0, 360, white, -1, 8, 0 );
    }

    cvNamedWindow( "image", 1 );
    cvShowImage( "image", img );
    cvConvert( img, img32f );
    findCComp( img32f );
    cvConvert( img32f, img32s );

    cvFindContours( img32s, storage, &contours, sizeof(CvContour),
                    CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0) );
    
    //cvFindContours( img, storage, &contours, sizeof(CvContour),
    //                CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0) );
    
    
    {
    const char* attrs[] = {"recursive", "1", 0};
    cvSave("contours.xml", contours, 0, 0, cvAttrList(attrs, 0));
    contours = (CvSeq*)cvLoad("contours.xml", storage, 0, 0);
    }

    // comment this out if you do not want approximation
    contours = cvApproxPoly( contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, 3, 1 );

    cvNamedWindow( "contours", 1 );
    cvCreateTrackbar( "levels+3", "contours", &levels, 7, on_trackbar );

    {
        CvRNG rng = cvRNG(-1);
        
        CvSeq* tcontours = contours;
        cvCvtColor( img, img3, CV_GRAY2BGR );
        while( tcontours->h_next )
            tcontours = tcontours->h_next;

        for( ; tcontours != 0; tcontours = tcontours->h_prev )
        {
            CvScalar color;
            color.val[0] = cvRandInt(&rng) % 256;
            color.val[1] = cvRandInt(&rng) % 256;
            color.val[2] = cvRandInt(&rng) % 256;
            color.val[3] = cvRandInt(&rng) % 256;
            cvDrawContours(img3, tcontours, color, color, 0, -1, 8, cvPoint(0,0));
            if( tcontours->v_next )
            {
                color.val[0] = cvRandInt(&rng) % 256;
                color.val[1] = cvRandInt(&rng) % 256;
                color.val[2] = cvRandInt(&rng) % 256;
                color.val[3] = cvRandInt(&rng) % 256;
                cvDrawContours(img3, tcontours->v_next, color, color, 1, -1, 8, cvPoint(0,0));
            }
        }
        
    }
    
    cvShowImage( "colored", img3 );
    on_trackbar(0);
    cvWaitKey(0);
    cvReleaseMemStorage( &storage );
    cvReleaseImage( &img );
    cvReleaseImage( &img32f );
    cvReleaseImage( &img32s );
    cvReleaseImage( &img3 );
    
    return 0;
}

#ifdef _EiC
main(1,"");
#endif
