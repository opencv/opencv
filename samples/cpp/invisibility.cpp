#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

string keys =
    "{ help        | | Print help message. }"
    "{ input i     | 0 | Video or camera id.  }"
    "{ thr         | 20 | Threshold background. }";

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("This sample demonstrates the invisibility effect with OpenCV.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    Mat backgroundImage, currentImage;
    Mat diffImage;
    Mat foregroundMask, mask2;
    Mat result1, result2;
    Mat video, element;
   
    float threshold = parser.get<float>("thr");
    float dist;

    int camId = parser.get<int>("input");
    int key = -1;

    VideoCapture cap;
    cap.open(camId);

    while (key<0)
    {
        key = waitKey(1);
        cap >> backgroundImage;
	putText(backgroundImage, "Press any key to grab backgroud or ESC to exit.", Point(10, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
        imshow("Video", backgroundImage);
    }

    if(key ==27){
       exit(0);
    }
    else{
        cap >> backgroundImage;
        key = -1;
    }

    while (key<0)
    {
        result1 = 0;
	result2 = 0;
        key = waitKey(1);
        cap >> currentImage;
        absdiff(backgroundImage, currentImage, diffImage);
        foregroundMask = Mat::zeros(diffImage.rows, diffImage.cols, CV_8UC1);
        for(int j=0; j<diffImage.rows; ++j)
	{
            for(int i=0; i<diffImage.cols; ++i)
            {
                Vec3b pix = diffImage.at<cv::Vec3b>(j,i);
                dist = (pix[0]*pix[0] + pix[1]*pix[1] + pix[2]*pix[2]);
                dist = sqrt(dist);
                if(dist>threshold)
                {
                    foregroundMask.at<unsigned char>(j,i) = 255;
                }
            }
        }
        element = Mat::ones(5,5, CV_32F);
        morphologyEx(foregroundMask,foregroundMask,cv::MORPH_OPEN, element);
        morphologyEx(foregroundMask,foregroundMask,cv::MORPH_DILATE, element);
        bitwise_not(foregroundMask,mask2);
        bitwise_and(currentImage,currentImage,result1,mask2);
        bitwise_and(backgroundImage,backgroundImage,result2,foregroundMask);
        addWeighted(result1,1,result2,1,0,video);
        imshow("Video",video);
    }
    return 0;
}
