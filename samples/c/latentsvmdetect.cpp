#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>

using namespace cv;

void help()
{
	printf( "This program demonstrated the use of the latentSVM detector.\n"
			"It reads in a trained object model and then uses that to detect the object in an image\n"
			"Call:\n"
			"./latentsvmdetect [<image_filename> <model_filename]\n"
			"  The defaults for image_filename and model_filename are cat.jpg and cat.xml respectively\n"
			"  Press any key to quit.\n");
}

const char* model_filename = "cat.xml";
const char* image_filename = "cat.jpg";

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
	help();
	if (argc > 2)
	{
		image_filename = argv[1];
		model_filename = argv[2];
	}
	IplImage* image = cvLoadImage(image_filename);
	if (!image)
	{
		printf( "Unable to load the image\n"
                "Pass it as the first parameter: latentsvmdetect <path to cat.jpg> <path to cat.xml>\n" );
		return -1;
	}
    CvLatentSvmDetector* detector = cvLoadLatentSvmDetector(model_filename);
	if (!detector)
	{
		printf( "Unable to load the model\n"
                "Pass it as the second parameter: latentsvmdetect <path to cat.jpg> <path to cat.xml>\n" );
		cvReleaseImage( &image );
		return -1;
	}
    detect_and_draw_objects( image, detector );
    cvNamedWindow( "test", 0 );
    cvShowImage( "test", image );
    cvWaitKey(0);
    cvReleaseLatentSvmDetector( &detector );
    cvReleaseImage( &image );
    cvDestroyAllWindows();
    
	return 0;
}
