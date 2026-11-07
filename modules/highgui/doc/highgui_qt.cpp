#include "opencv2/highgui.hpp"

int main(int argc, char *argv[])
{
    int value = 50;
    int value2 = 0;

    namedWindow("main1",WINDOW_NORMAL);
    namedWindow("main2",WINDOW_AUTOSIZE | WINDOW_GUI_NORMAL);
    createTrackbar( "track1", "main1", &value, 255,  NULL);

    String nameb1 = "button1";
    String nameb2 = "button2";

    createButton(nameb1,callbackButton,&nameb1,QT_CHECKBOX,1);
    createButton(nameb2,callbackButton,NULL,QT_CHECKBOX,0);
    createTrackbar( "track2", NULL, &value2, 255, NULL);
    createButton("button5",callbackButton1,NULL,QT_RADIOBOX,0);
    createButton("button6",callbackButton2,NULL,QT_RADIOBOX,1);

    setMouseCallback( "main2",on_mouse,NULL );

    Mat img1 = imread("files/flower.jpg");
    VideoCapture video;
    video.open("files/hockey.avi");

    Mat img2,img3;
    while( waitKey(33) != 27 )
    {
        img1.convertTo(img2,-1,1,value);
        video >> img3;

        imshow("main1",img2);
        imshow("main2",img3);
    }

    destroyAllWindows();
    return 0;
}
