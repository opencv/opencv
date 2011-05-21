#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>

#ifdef HAVE_CONFIG_H 
#include <cvconfig.h> 
#endif
#ifdef HAVE_TBB
#include "tbb/task_scheduler_init.h"
#endif

using namespace cv;

void help()
{
    printf( "This program demonstrated the use of the latentSVM detector.\n"
            "It reads in a trained object model and then uses that to detect the object in an image\n"
            "Call:\n"
            "./latentsvmdetect [--image_filename]=<image_filename, cat.jpg as default> \n"
            "       [--model_filename] = <model_filename, cat.xml as default> \n"
            "       [--threads_number] = <number of threads, -1 as default>\n"
            "  The defaults for image_filename and model_filename are cat.jpg and cat.xml respectively\n"
            "  Press any key to quit.\n");
}


void detect_and_draw_objects( IplImage* image, CvLatentSvmDetector* detector, int numThreads = -1)
{
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* detections = 0;
    int i = 0;
    int64 start = 0, finish = 0;
#ifdef HAVE_TBB
    tbb::task_scheduler_init init(tbb::task_scheduler_init::deferred);
    if (numThreads > 0)
    {
        init.initialize(numThreads);
        printf("Number of threads %i\n", numThreads);
    }
    else
    {
        printf("Number of threads is not correct for TBB version");
        return;
    }
#endif
    start = cvGetTickCount();
    detections = cvLatentSvmDetectObjects(image, detector, storage, 0.5f, numThreads);
    finish = cvGetTickCount();
    printf("detection time = %.3f\n", (float)(finish - start) / (float)(cvGetTickFrequency() * 1000000.0));

#ifdef HAVE_TBB
    init.terminate();
#endif
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

int main(int argc, const char* argv[])
{
    help();

    CommandLineParser parser(argc, argv);

    string imageFileName = parser.get<string>("image_filename", "cat.jpg");
    string modelFileName = parser.get<string>("model_filename", "cat.xml");
    int tbbNumThreads = parser.get<int>("threads_number", -1);

    IplImage* image = cvLoadImage(imageFileName.c_str());
    if (!image)
    {
        printf( "Unable to load the image\n"
                "Pass it as the first parameter: latentsvmdetect <path to cat.jpg> <path to cat.xml>\n" );
        return -1;
    }
    CvLatentSvmDetector* detector = cvLoadLatentSvmDetector(modelFileName.c_str());
    if (!detector)
    {
        printf( "Unable to load the model\n"
                "Pass it as the second parameter: latentsvmdetect <path to cat.jpg> <path to cat.xml>\n" );
        cvReleaseImage( &image );
        return -1;
    }

    detect_and_draw_objects( image, detector, tbbNumThreads );

    cvNamedWindow( "test", 0 );
    cvShowImage( "test", image );
    cvWaitKey(0);
    cvReleaseLatentSvmDetector( &detector );
    cvReleaseImage( &image );
    cvDestroyAllWindows();
    
    return 0;
}
