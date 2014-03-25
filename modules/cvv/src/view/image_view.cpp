
#include <memory>

#include <QString>
#include <QWidget>
#include <QHBoxLayout>
#include <QGridLayout>

#include <opencv2/core/core.hpp>

#include "../qtutil/accordion.hpp"
#include "../qtutil/zoomableimageoptpanel.hpp"
#include "../qtutil/zoomableimage.hpp"
#include "../qtutil/histogram.hpp"
#include "../qtutil/histogramoptpanel.hpp"
#include "../util/util.hpp"
#include "image_view.hpp"

namespace cvv
{
namespace view
{

ImageView::ImageView(const cv::Mat &image, QWidget *parent)
	    : QWidget{ parent }, image{nullptr}
{
	auto layout = util::make_unique<QHBoxLayout>();
	auto imageLayout = util::make_unique<QGridLayout>();
	auto accor = util::make_unique<qtutil::Accordion>();

	accor->setMinimumWidth(300);
	accor->setMaximumWidth(300);

	auto zoomim = util::make_unique<qtutil::ZoomableImage>();
	accor->insert(QString("ImageInformation:"),
		std::move(util::make_unique<qtutil::ZoomableOptPanel>(*zoomim,false)));
	zoomim->setMat(image);

	auto histogram = util::make_unique<qtutil::Histogram>();
	histogram->setMat(image);
	histogram->setVisible(false);
	connect(zoomim.get(), SIGNAL(updateArea(QRectF, qreal)), histogram.get(), SLOT(setArea(QRectF, qreal)));

	accor->insert(QString("Histogram:"), std::move(util::make_unique<qtutil::HistogramOptPanel>(*histogram)));

	this->image = (*zoomim);
	imageLayout->addWidget(zoomim.release(), 0, 0);
	imageLayout->addWidget(histogram.release(), 1, 0);
	layout->addWidget(accor.release());
	layout->addLayout(imageLayout.release());

	setLayout(layout.release());
}

void ImageView::showFullImage()
{
	image->showFullImage();
}


}} //namespaces
