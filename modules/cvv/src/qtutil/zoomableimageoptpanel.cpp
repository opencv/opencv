#include <QVBoxLayout>
#include <QPushButton>
#include <QCheckBox>

#include "zoomableimageoptpanel.hpp"
#include "../util/util.hpp"
#include "util.hpp"

namespace cvv
{
namespace qtutil
{

ZoomableOptPanel::ZoomableOptPanel(const ZoomableImage &zoomIm, bool showHideButton, QWidget *parent)
    : QWidget{ parent }
{
	auto basicLayout = cvv::util::make_unique<QVBoxLayout>();

	basicLayout->setContentsMargins(0,0,0,0);

	auto zoomSpin = cvv::util::make_unique<QDoubleSpinBox>();
	auto labelConvert = cvv::util::make_unique<QLabel>();
	auto labelDim = cvv::util::make_unique<QLabel>();
	auto labelType = cvv::util::make_unique<QLabel>();
	auto labelChannel = cvv::util::make_unique<QLabel>();
	auto labelSize = cvv::util::make_unique<QLabel>();
	auto labelDepth = cvv::util::make_unique<QLabel>();
	auto buttonFullImage =
	    cvv::util::make_unique<QPushButton>("show full Image");

	// ConversionResult+ update mat
	connect(&zoomIm, SIGNAL(updateConversionResult(const cv::Mat &,
						       ImageConversionResult)),
		this, SLOT(updateConvertStatus(const cv::Mat &,
					       ImageConversionResult)));

	// getzoom from image
	connect(&zoomIm, SIGNAL(updateArea(QRectF, qreal)), this,
		SLOT(setZoom(QRectF, qreal)));

	// set zoom to image
	connect(zoomSpin.get(), SIGNAL(valueChanged(double)), &zoomIm,
		SLOT(setZoom(qreal)));

	// fullimage
	connect(buttonFullImage.get(), SIGNAL(clicked()), &zoomIm,
		SLOT(showFullImage()));



	zoomSpin->setMinimum(0.0);
	zoomSpin->setMaximum(2000.0);

	zoomSpin_ = zoomSpin.get();
	labelConvert_ = labelConvert.get();
	labelDim_ = labelDim.get();
	labelType_ = labelType.get();
	labelChannel_ = labelChannel.get();
	labelSize_ = labelSize.get();
	labelDepth_ = labelDepth.get();

	basicLayout->addWidget(zoomSpin.release());
	if(showHideButton)
	{
		auto checkboxShowImage= util::make_unique<QCheckBox>("Show image");
		//connect show image
		checkboxShowImage->setChecked(true);
		QObject::connect(checkboxShowImage.get(),SIGNAL(clicked(bool)),
				 &zoomIm,SLOT(setVisible(bool)));
		basicLayout->addWidget(checkboxShowImage.release());
	}
	basicLayout->addWidget(labelConvert.release());
	basicLayout->addWidget(labelSize.release());
	basicLayout->addWidget(labelDim.release());
	basicLayout->addWidget(labelType.release());
	basicLayout->addWidget(labelDepth.release());
	basicLayout->addWidget(labelChannel.release());
	basicLayout->addWidget(buttonFullImage.release());

	setLayout(basicLayout.release());

	updateMat(zoomIm.mat());
	updateConvertStatus(zoomIm.mat(),zoomIm.lastConversionResult());
}

void ZoomableOptPanel::updateConvertStatus(const cv::Mat &mat, ImageConversionResult result)
{
	labelConvert_->setText(
		QString{ "Convert Status: " }.append(conversionResultToString(result)));

	updateMat(mat);
}

void ZoomableOptPanel::updateMat(cv::Mat mat)
{
	if (mat.empty())
	{
		labelDim_->setText("empty");
		labelType_->setText("empty");
		labelChannel_->setText("empty");
		labelSize_->setText("empty");
		labelDepth_->setText("empty");
	}
	else
	{
		labelDim_->setText(QString("Dimension: %1").arg(mat.dims));
		labelChannel_->setText(
		    QString("Number of Channels: %1").arg(mat.channels()));
		labelSize_->setText(
		    QString("Size: %1/%2").arg(mat.rows).arg((mat.cols)));
		labelDepth_->setText(QString("Depth type: %1").arg(mat.depth()));

		auto type=typeToQString(mat);
		labelType_->setText(QString{"Type: "}.append(type.second));
	}
}

void ZoomableOptPanel::setZoom(QRectF, qreal zoomfac)
{
	zoomSpin_->setValue(zoomfac);
}
}
}
