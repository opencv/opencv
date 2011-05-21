#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>

using namespace cv;

void help()
{
 printf("\nDo background segmentation, especially demonstrating the use of cvUpdateBGStatModel().\n"
"Learns the background at the start and then segments.\n"
"Learning is togged by the space key. Will read from file or camera\n"
"Call:\n"
"./  bgfg_segm [file name -- if no name, read from camera]\n\n");
}

//this is a sample for foreground detection functions
int main(int argc, char** argv)
{
    VideoCapture cap;
    bool update_bg_model = true;

    if( argc < 2 )
        cap.open(0);
    else
        cap.open(argv[1]);
    help();
    
    if( !cap.isOpened() )
    {
        printf("can not open camera or video file\n");
        return -1;
    }
    
    namedWindow("BG", 1);
    namedWindow("FG", 1);

    BackgroundSubtractorMOG2 bg_model;
    Mat img, fgmask;
    
    for(;;)
    {
        cap >> img;
        
        if( img.empty() )
            break;
        
        bg_model(img, fgmask, update_bg_model ? -1 : 0);
        
        imshow("image", img);
        imshow("foreground mask", fgmask);
        char k = (char)waitKey(30);
        if( k == 27 ) break;
        if( k == ' ' )
        {
            update_bg_model = !update_bg_model;
            if(update_bg_model)
            	printf("Background update is on\n");
            else
            	printf("Background update is off\n");
        }
    }

    return 0;
}
