#include "opencv2/highgui/highgui.hpp"

#include <iostream>

void setExposure(int value, void *cap_ptr) {
    cv::VideoCapture& cap = *(cv::VideoCapture *)cap_ptr;

    int exposure = 1;
    switch (value) {
        case 0: exposure = 1; break;
        case 1: exposure = 2; break;
        case 2: exposure = 5; break;
        case 3: exposure = 10; break;
        case 4: exposure = 20; break;
        case 5: exposure = 39; break;
        case 6: exposure = 78; break;
        case 7: exposure = 156; break;
        case 8: exposure = 312; break;
        case 9: exposure = 625; break;
        case 10: exposure = 1250; break;
        case 11: exposure = 1250; break;
        case 12: exposure = 2500; break;
        case 13: exposure = 5000; break;
        case 14: exposure = 10000; break;
    }

    cap.set(CV_CAP_PROP_EXPOSURE, exposure);
}

void setFocus(int value, void *cap_ptr) {
    cv::VideoCapture& cap = *(cv::VideoCapture *)cap_ptr;
    cap.set(CV_CAP_PROP_FOCUS, (double)value / 100.0);
}


int main(int argc, const char *argv[]) {

    cv::VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1024);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 640);

    cap.set(CV_CAP_PROP_AUTO_FOCUS, 0);
    cap.set(CV_CAP_PROP_AUTO_EXPOSURE, 0);

    cv::namedWindow("image");

    int focus = 50;
    int temperature = 50;

    cv::createTrackbar("focus", "image", &focus, 100, setFocus, (void *)&cap);
    cv::createTrackbar("exposure", "image", &focus, 14, setExposure, (void *)&cap);

    setExposure(7, (void *)&cap);
    setFocus(focus, (void *)&cap);

    while (true) {
        cv::Mat img;
        cap >> img;

        cv::imshow("image", img);

        char key = cv::waitKey(30);
        if (key == 'q' || key == 27) break;
        if (key == 'f') cap.set(CV_CAP_PROP_AUTO_FOCUS, 1 - cap.get(CV_CAP_PROP_AUTO_FOCUS));
        if (key == 'e') cap.set(CV_CAP_PROP_AUTO_EXPOSURE, 1 - cap.get(CV_CAP_PROP_AUTO_EXPOSURE));
    }
}
