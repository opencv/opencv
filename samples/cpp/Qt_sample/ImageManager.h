#ifndef IMAGEMANAGER_H
#define IMAGEMANAGER_H
#include <opencv2/opencv.hpp>

#include <QImage>
#include <QThread>

class ResourcesManager;

/**
* class to manage openCV features and cv to Qt conversion
*
*/
class ImageManager : public QObject
{
Q_OBJECT
private:
####cv::VideoCapture *m_videoCapture;
####
####// thread within which the image processing will be done
####QThread *m_thread;
####
####cv::Mat m_currentImage;
public:
####// constructor. string can be any parameter understandable by cv::VideoCapture::open
    ImageManager(std::string _videoFile);
####// start processing
####void start();
####// stop processing
####void stop();
####
####//return curent image
####cv::Mat getCurrentImage();
####
####// convert cv::Mat to QImage
####QImage Mat2QImage(const cv::Mat img);
public slots:
  // function wich will be execute within the new thread context
  void process();
  
  
  signals:
#### // sending image updates to other threads (such as HMI)
     void log(std::string _name, QImage _image);
};

#endif // IMAGEMANAGER_H
