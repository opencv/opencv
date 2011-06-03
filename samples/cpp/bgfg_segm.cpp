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
    
    namedWindow("image", CV_WINDOW_NORMAL);
    namedWindow("foreground mask", CV_WINDOW_NORMAL);
        namedWindow("foreground image", CV_WINDOW_NORMAL);
    namedWindow("mean background image", CV_WINDOW_NORMAL);

    BackgroundSubtractorMOG2 bg_model;
    Mat img, fgmask, fgimg;

    for(;;)
    {
        cap >> img;
        
        if( img.empty() )
            break;
        
        if( fgimg.empty() )
          fgimg.create(img.size(), img.type());

        //update the model
        bg_model(img, fgmask, update_bg_model ? -1 : 0);

        fgimg = Scalar::all(0);
        img.copyTo(fgimg, fgmask);

        Mat bgimg;
        bg_model.getBackgroundImage(bgimg);

        imshow("image", img);
        imshow("foreground mask", fgmask);
        imshow("foreground image", fgimg);
        if(!bgimg.empty())
          imshow("mean background image", bgimg );

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
