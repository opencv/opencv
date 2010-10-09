#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include <stdio.h>

using namespace cv;

const char* model_filename = "cat.xml";
const char* image_filename = "000028.jpg";

void detect_and_draw_objects( IplImage* image, CvLatentSvmDetector* detector)
{
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* detections = 0;
    int i = 0;
	int64 start = 0, finish = 0;

	start = cvGetTickCount();
    detections = cvLatentSvmDetectObjects(image, detector, storage);
	finish = cvGetTickCount();
	printf("detection time = %.3f\n", (float)(finish - start) / (float)(cvGetTickFrequency() * 1000000.0));

    for( i = 0; i < detections->total; i++ )
    {
        CvObjectDetection detection = *(CvObjectDetection*)cvGetSeqElem( detections, i );
		CvRect bounding_box = detection.rect;
        cvRectangle( image, cvPoint(bounding_box.x, bounding_box.y),
                     cvPoint(bounding_box.x + bounding_box.width, 
							bounding_box.y + bounding_box.height),
                     CV_RGB(255,0,0), 3 );
    }
    cvReleaseMemStorage( &storage );
}


int main(int argc, char* argv[])
{
	IplImage* image = cvLoadImage(image_filename);
    CvLatentSvmDetector* detector = cvLoadLatentSvmDetector(model_filename);
    detect_and_draw_objects( image, detector );
    cvNamedWindow( "test", 0 );
    cvShowImage( "test", image );
    cvWaitKey(0);
    cvReleaseLatentSvmDetector( &detector );
    cvReleaseImage( &image );
    cvDestroyAllWindows();
    
	return 0;
}