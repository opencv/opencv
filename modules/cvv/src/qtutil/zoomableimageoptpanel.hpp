#ifndef CVVISUAL_ZOOMABLE_IMAGE_OPT_PANEL
#define CVVISUAL_ZOOMABLE_IMAGE_OPT_PANEL

#include <QLabel>
#include <QString>
#include <QWidget>
#include <QDoubleSpinBox>

#include "opencv2/core/core.hpp"
#include "zoomableimage.hpp"
#include "util.hpp"

namespace cvv
{
namespace qtutil
{

/*
 * @brief This Widget shows some Infos about the given cv::Mat from the given
 * ZoomabelImage
 * and has some Options for zooming
 */
class ZoomableOptPanel : public QWidget
{

	Q_OBJECT

      public:
	/**
	 * @brief the constructor
	 * @param image the ZoomableImage which will be connected
	 * @param parent the parent Widget
	 */
	ZoomableOptPanel(const ZoomableImage &zoomIm,
			 bool showHideButton=true,QWidget *parent = nullptr);

      public
slots:
	void updateMat(cv::Mat mat);
	void updateConvertStatus(const cv::Mat &,ImageConversionResult result);
	void setZoom(QRectF, qreal);

      private:
	QDoubleSpinBox *zoomSpin_;
	QLabel *labelConvert_;
	QLabel *labelDim_;
	QLabel *labelType_;
	QLabel *labelChannel_;
	QLabel *labelSize_;
	QLabel *labelDepth_;
};
}
}

#endif
