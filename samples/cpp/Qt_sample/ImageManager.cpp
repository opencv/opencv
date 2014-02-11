#include "ImageManager.h"
#include "Gui.h"
#include <sstream>
#include <iostream>
#include <opencv2/imgproc/imgproc_c.h>

#define TRHREASHOLD_BIN 57.

using namespace std;
using namespace cv;


ImageManager::ImageManager(string videoPath)
{
    this->m_thread =####new QThread();


    // detect if the string represents a number or a string (ie input id or video path)
    istringstream ss(videoPath);
    int deviceId = 0;
    if(ss >> deviceId)
    {
        cout << "ImageManager : Using video camera " << deviceId << endl;
        this->m_videoCapture = new VideoCapture(deviceId);
    }
    else
    {
        cout << "ImageManager : Using video stream " << videoPath;
        this->m_videoCapture = new VideoCapture(videoPath);
    }

    if(!this->m_videoCapture->isOpened())  // check if we succeeded
    {
        cerr << "ImageManager : cannot open video" << endl;
        throw new int (-10);
    }

    // connect the staring signal of the thread to our process slot
    QObject::connect(this->m_thread, SIGNAL(started()), this, SLOT(process()));

    // move the execution to the given thread
    this->moveToThread(this->m_thread);
}

void ImageManager::start()
{
    cout << "ImageManager : Starting thread" << endl;
    this->m_thread->start();
}

void ImageManager::stop()
{
    cout << "ImageManager : Stoping thread" << endl;
    this->m_thread->quit();
}

void ImageManager::process()
{
    for(;;)
    {
        // get a new frame
        bool isValid = this->m_videoCapture->read(this->m_currentImage); // get a new frame from camera
        // if frame is invalid (video end, camera disconnected... stop process)
        if(!isValid)
        {
            break;
        }

        // convert cv::Mat to QImage
        QImage img = Mat2QImage(this->m_currentImage);

        // send it to listeners
        emit log("input", img);

        // convert the input to grayscale
        Mat src_gray;
        cvtColor( this->m_currentImage, src_gray, CV_BGR2GRAY );
        // and send it also to listeners
        emit log("gray", Mat2QImage(src_gray));


    }

}


cv::Mat ImageManager::getCurrentImage()
{
    return this->m_currentImage;
}

QImage ImageManager::Mat2QImage(const cv::Mat img)
{
  cv::Mat _tmp;
  // convert input Mat to RGB
  switch (img.type()) {
        case CV_8UC1:
            cvtColor(img, _tmp, CV_GRAY2RGB);
            break;
        case CV_8UC3:
            cvtColor(img, _tmp, CV_BGR2RGB);
            break;
####default:
      cerr << "ImageManager : unknown format" << endl;
####  break;
        }
        assert(_tmp.isContinuous());
    // create the coresponding QImage
####QImage i(_tmp.data, _tmp.cols, _tmp.rows, _tmp.step, QImage::Format_RGB888);

    // return copy as i will be deleted add the end of the function
    return i.copy();
}
