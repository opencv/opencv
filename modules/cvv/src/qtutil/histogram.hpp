#ifndef CVVISUAL_HISTOGRAM_HPP
#define CVVISUAL_HISTOGRAM_HPP

#include <QWidget>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "zoomableimage.hpp"


namespace cvv
{
namespace qtutil
{

class Histogram : public QWidget
{
  Q_OBJECT
  public:
    Histogram(const cv::Mat& mat = cv::Mat{}, QWidget* parent = nullptr);

    void setMat(const cv::Mat& mat);
    std::vector<cv::Mat> calcHist(cv::Mat mat, cv::Rect rect, int bins = 256, float rangeMin = 0.0, float rangeMax = 256.0);
    cv::Mat drawHist(const std::vector<cv::Mat>& channelHists, cv::Size histSize, int lineWidth = 2, const cv::Scalar& backgroundColor = cv::Scalar(255, 255, 255));

  public slots:
    void setArea(QRectF, qreal);

  private:
    cv::Rect qrect2cvrect(const cv::Mat& mat, QRectF qrect);

    cv::Mat mat_;
    std::vector<cv::Mat> channelHists_;
    cv::Mat histMat_;
    cv::Size histSize_;
    int histLineWidth_;
    cv::Scalar histBackgroundColor_;
    ZoomableImage* zoomableImage;
};

}
}

#endif // CVVISUAL_HISTOGRAM_HPP
