#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QWidget>

#include "defaultfilterview.hpp"
#include "../qtutil/accordion.hpp"
#include "../qtutil/zoomableimageoptpanel.hpp"
#include "../qtutil/zoomableimage.hpp"
#include "../qtutil/synczoomwidget.hpp"
#include "../qtutil/histogram.hpp"
#include "../qtutil/histogramoptpanel.hpp"
#include "../util/util.hpp"

namespace cvv
{
namespace view
{

DefaultFilterView::DefaultFilterView(const std::vector<cv::Mat> &images,
				     QWidget *parent)
    : FilterView{ parent }
{

	auto layout = util::make_unique<QHBoxLayout>();
	auto accor = util::make_unique<qtutil::Accordion>();
	auto imwid = util::make_unique<QWidget>();
	auto imageLayout = util::make_unique<QGridLayout>();

	accor->setMinimumWidth(250);
	accor->setMaximumWidth(250);

	std::vector<qtutil::ZoomableImage*> syncVec;
	
	size_t count = 0;
	for (auto& image : images)
	{
		auto zoomIm = util::make_unique<qtutil::ZoomableImage>();

		syncVec.push_back(zoomIm.get());

		accor->insert(
		    QString("Image Information: ") + QString::number(count),
		    std::move(
			util::make_unique<qtutil::ZoomableOptPanel>(*zoomIm)));

		zoomIm->setMat(image);

		auto histogram = util::make_unique<qtutil::Histogram>();
		histogram->setMat(image);
		histogram->setVisible(false);
		connect(zoomIm.get(), SIGNAL(updateArea(QRectF, qreal)), histogram.get(), SLOT(setArea(QRectF, qreal)));

		accor->insert(QString("Histogram: ") + QString::number(count), std::move(util::make_unique<qtutil::HistogramOptPanel>(*histogram)));

		imageLayout->addWidget(zoomIm.release(), 0, count);
    imageLayout->addWidget(histogram.release(), 1, count);

		count++;
	}

	accor->insert("Zoom synchronization",
		util::make_unique<qtutil::SyncZoomWidget>(syncVec), true, 0);

	imwid->setLayout(imageLayout.release());

	layout->addWidget(accor.release());
	layout->addWidget(imwid.release());

	setLayout(layout.release());
	//images should be seen fully at beginning
	for(auto& zoomableImage: syncVec)
	{
		zoomableImage->showFullImage();
	}
}
}
} // namespaces
