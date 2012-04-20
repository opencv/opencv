#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>

using namespace std;
using namespace cv;

void help()
{
 printf("\nDo background segmentation, especially demonstrating the use of cvUpdateBGStatModel().\n"
"Learns the background at the start and then segments.\n"
"Learning is togged by the space key. Will read from file or camera\n"
"Usage: \n"
"			./bgfg_segm [--camera]=<use camera, if this key is present>, [--file_name]=<path to movie file> \n\n");
}

const char* keys = 
{
	"{c |camera   |true    | use camera or not}"
	"{fn|file_name|tree.avi | movie file             }"
};

//this is a sample for foreground detection functions
int main(int argc, const char** argv)
{
	help();

	CommandLineParser parser(argc, argv, keys);
	bool useCamera = parser.get<bool>("camera");
	string file = parser.get<string>("file_name");
    VideoCapture cap;
    bool update_bg_model = true;

    if( useCamera )
        cap.open(0);
    else
		cap.open(file.c_str());
	parser.printParams();

    if( !cap.isOpened() )
    {
        printf("can not open camera or video file\n");
        return -1;
    }
    
    namedWindow("image", CV_WINDOW_NORMAL);
    namedWindow("foreground mask", CV_WINDOW_NORMAL);
    namedWindow("foreground image", CV_WINDOW_NORMAL);
    namedWindow("mean background image", CV_WINDOW_NORMAL);

    BackgroundSubtractorMOG2 bg_model;//(100, 3, 0.3, 5);
                                     
    Mat img, fgmask, fgimg;

    for(;;)
    {
        cap >> img;
        
        if( img.empty() )
            break;
        
        //cvtColor(_img, img, COLOR_BGR2GRAY);
        
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
