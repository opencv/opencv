#ifdef _CH_
#pragma package <opencv>
#endif

#define CV_NO_BACKWARD_COMPATIBILITY

#ifndef _EiC
#include "cv.h"
#include "highgui.h"
#include <math.h>
#endif

IplImage*  image[2] = { 0, 0 }, *image0 = 0, *image1 = 0;
CvSize size;

int  w0, h0,i;
int  threshold1, threshold2;
int  l,level = 4;
int sthreshold1, sthreshold2;
int  l_comp;
int block_size = 1000;
float  parameter;
double threshold;
double rezult, min_rezult;
int filter = CV_GAUSSIAN_5x5;
CvConnectedComp *cur_comp, min_comp;
CvSeq *comp;
CvMemStorage *storage;

CvPoint pt1, pt2;

void ON_SEGMENT(int a)
{
    cvPyrSegmentation(image0, image1, storage, &comp,
                      level, threshold1+1, threshold2+1);

    /*l_comp = comp->total;

    i = 0;
    min_comp.value = cvScalarAll(0);
    while(i<l_comp)
    {
        cur_comp = (CvConnectedComp*)cvGetSeqElem ( comp, i );
        if(fabs(255- min_comp.value.val[0])>
           fabs(255- cur_comp->value.val[0]) &&
           fabs(min_comp.value.val[1])>
           fabs(cur_comp->value.val[1]) &&
           fabs(min_comp.value.val[2])>
           fabs(cur_comp->value.val[2]) )
           min_comp = *cur_comp;
        i++;
    }*/
    cvShowImage("Segmentation", image1);
}

int main( int argc, char** argv )
{
    char* filename = argc == 2 ? argv[1] : (char*)"fruits.jpg";

    if( (image[0] = cvLoadImage( filename, 1)) == 0 )
        return -1;

    cvNamedWindow("Source", 0);
    cvShowImage("Source", image[0]);

    cvNamedWindow("Segmentation", 0);

    storage = cvCreateMemStorage ( block_size );

    image[0]->width &= -(1<<level);
    image[0]->height &= -(1<<level);

    image0 = cvCloneImage( image[0] );
    image1 = cvCloneImage( image[0] );
    // segmentation of the color image
    l = 1;
    threshold1 =255;
    threshold2 =30;

    ON_SEGMENT(1);

    sthreshold1 = cvCreateTrackbar("Threshold1", "Segmentation", &threshold1, 255, ON_SEGMENT);
    sthreshold2 = cvCreateTrackbar("Threshold2", "Segmentation",  &threshold2, 255, ON_SEGMENT);

    cvShowImage("Segmentation", image1);
    cvWaitKey(0);

    cvDestroyWindow("Segmentation");
    cvDestroyWindow("Source");

    cvReleaseMemStorage(&storage );

    cvReleaseImage(&image[0]);
    cvReleaseImage(&image0);
    cvReleaseImage(&image1);

    return 0;
}

#ifdef _EiC
main(1,"pyramid_segmentation.c");
#endif
