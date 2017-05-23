#include <array>
#include <vector>

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QWidget>

#include "../qtutil/accordion.hpp"
#include "../qtutil/autofilterwidget.hpp"
#include "../qtutil/zoomableimage.hpp"
#include "../qtutil/zoomableimageoptpanel.hpp"
#include "../qtutil/synczoomwidget.hpp"
#include "../util/util.hpp"
#include "singlefilterview.hpp"

namespace cvv
{
namespace view
{

SingleFilterView::SingleFilterView(const std::vector<cv::Mat> &images,
				   QWidget *parent)
    : FilterView{ parent }
{
	auto imwid = util::make_unique<QWidget>();
	auto accor = util::make_unique<qtutil::Accordion>();
	auto layout = util::make_unique<QHBoxLayout>();
	auto imageLayout = util::make_unique<QHBoxLayout>();

	accor->setMinimumWidth(280); // ggf anpassen
	accor->setMaximumWidth(280);

	auto filterSelector =
	    util::make_unique<qtutil::AutoFilterWidget<1, 1>>(this);

	qtutil::AutoFilterWidget<1, 1> *filterSel = filterSelector.get();

	accor->insert("Select a Filter", std::move(filterSelector));

	std::vector<qtutil::ZoomableImage*> syncVec;
	int count = 0;
	for (auto &image : images)
	{
		auto originalZoomIm = util::make_unique<qtutil::ZoomableImage>(image);
		accor->insert(
		    QString("Info original image ") + QString::number(count),
		    std::move(
			util::make_unique<qtutil::ZoomableOptPanel>(*originalZoomIm)));
		syncVec.push_back(originalZoomIm.get());


		auto filterZoomIm = util::make_unique<qtutil::ZoomableImage>(image.clone());

		// für das AutofilterWidget muss die output Mat gespeichert
		// werden dies übernimmt
		// das ZoomableImage
		auto filterSignals = filterSel->addEntry(
		    QString("Image: ") + QString::number(count),
		    { { util::makeRef<const cv::Mat>(/*image*/originalZoomIm->mat()) } },
		    { { util::makeRef<cv::Mat>(filterZoomIm->mat()) } });

		// connect entry=> zoomableimage
		connect(filterSignals.front().getPtr(),
			SIGNAL(signal(cv::Mat &)), filterZoomIm.get(),
			SLOT(setMatR(cv::Mat &)));

		accor->insert(
		    QString("Info filtered image ") + QString::number(count),
		    std::move(
			util::make_unique<qtutil::ZoomableOptPanel>(*filterZoomIm)));

		syncVec.push_back(filterZoomIm.get());

		auto dualImageLayout = util::make_unique<QVBoxLayout>();
		dualImageLayout->addWidget(originalZoomIm.release());
		dualImageLayout->addWidget(filterZoomIm.release());

		auto dualImageWidget = util::make_unique<QWidget>();
		dualImageWidget->setLayout(dualImageLayout.release());
		imageLayout->addWidget(dualImageWidget.release());
		count++;
	}

	imwid->setLayout(imageLayout.release());

	accor->insert("Zoom synchronization",
		std::move(util::make_unique<qtutil::SyncZoomWidget>(syncVec)), true, 1);

	layout->addWidget(accor.release());
	layout->addWidget(imwid.release());

	setLayout(layout.release());

}
}
} // namespaces
