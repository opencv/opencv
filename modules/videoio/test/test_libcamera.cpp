#include <iostream>

#include "opencv2/videoio.hpp"
#include "opencv2/videoio/registry.hpp"
#include "opencv2/core/private.hpp"
#include "opencv2/core/utils/configuration.private.hpp"
using namespace cv;


int main()
{
    VideoCapture cap = VideoCapture(0,CAP_LIBCAMERA);
    
    if(!cap.isOpened())
        return;
 
    Mat frame;
    while(1)
    {
        cap>>frame;
        if(frame.empty())
            break;
        imshow("video", frame);
        if(waitKey(25)>0)//按下任意键退出摄像头
            break;
    }
    cap.release();
    destroyAllWindows();//关闭所有窗口
}