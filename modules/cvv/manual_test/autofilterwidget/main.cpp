#include "../../src/qtutil/autofilterwidget.hpp"

#include <QWidget>
#include <QCheckBox>
#include <QLabel>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QDoubleSpinBox>
#include <QPushButton>

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

#include "../../src/qtutil/filterfunctionwidget.hpp"
#include "../../src/qtutil/filterselectorwidget.hpp"
#include "../../src/qtutil/accordion.hpp"
#include "../../src/qtutil/zoomableimage.hpp"
#include "../../src/qtutil/filter/sobelfilterwidget.hpp"
#include "../../src/util/util.hpp"
#include "../../src/impl/init.hpp"

/**
 * @brief
 * - a window will pop up
 * - it has 3 columns
 * - the first contains an accordion wit 3 elements ("button user select",
 *  "button individual filter", "afw")
 * - the 2nd column contains a black image ( 6 channels, all pixel
 * {2;1;2147483640;2147483640;2147483640;2147483640}
 * - the 3rd column contains a blue image with a black line
 * - the visible are (zoom+ area) of the left image will be applyed to the right image
 * - if "button user select" is toggled there are two checkboxes ("i1", "i2")
 * in the top of "afw"
 * - in afw the user can select filters to apply to both images (only checked ones are used)
 * - the filtered image will appear below the orginal one
 * - if the filter can not be applied to an image a red error message will appear in at the top of afw
 * - if "button individual filter" is not checked filter are applied only if there are no error messages
 * - filters are: "Sobel", "Gray filter", "Reorder channels"
 * - "Sobel" can not be applied to the left image
 * - "Sobel" represents the sobel filter
 * - "Gray filter" can apply a gray filter to both images
 * (the factors for each channel can be choosen by the user)
 * - "Reorder channels" can reorder the channels of the orginal image
 * (number and order of the output channels can be selected)
 */
int main(int argc, char *argv[])
{

	cvv::impl::initializeFilterAndViews();


	QApplication a(argc, argv);
	QWidget w{};

	cv::Mat m{ 200, 100, CV_32SC3, cv::Scalar{ 2, 1, 2147483640 } };
	cv::Mat m2{ 200, 100, CV_32FC3, cv::Scalar{ 1, 0, 0 } };

	auto matvec = cvv::qtutil::splitChannels(m);
	matvec.push_back(matvec.back());
	matvec.push_back(matvec.back());
	matvec.push_back(matvec.back());
	cv::Mat m1 = cvv::qtutil::mergeChannels(matvec);

	// something to see for sobel
	line(m2, cv::Point{ 0, 0 }, cv::Point{ 100, 100 },
	     cv::Scalar{ 0, 0, 0 }, 10, 8);

	auto i1 = cvv::util::make_unique<cvv::qtutil::ZoomableImage>(m1);
	auto i2 = cvv::util::make_unique<cvv::qtutil::ZoomableImage>(m2);
	auto o1 = cvv::util::make_unique<cvv::qtutil::ZoomableImage>();
	auto o2 = cvv::util::make_unique<cvv::qtutil::ZoomableImage>();
	// connect area on input
	QObject::connect(i1.get(), SIGNAL(updateArea(QRectF, qreal)), i2.get(),
			 SLOT(setArea(QRectF, qreal)));

	auto afw =
	    cvv::util::make_unique<cvv::qtutil::AutoFilterWidget<1, 1>>();
	// buttons
	auto busele =
	    cvv::util::make_unique<QPushButton>("user select for afw");
	auto bindiv =
	    cvv::util::make_unique<QPushButton>("individual filter for afw");
	busele->setCheckable(true);
	bindiv->setCheckable(true);
	busele->setChecked(true);
	bindiv->setChecked(true);

	QObject::connect(busele.get(), SIGNAL(clicked(bool)),
			 &(afw->slotEnableUserSelection()), SLOT(slot(bool)));
	QObject::connect(bindiv.get(), SIGNAL(clicked(bool)),
			 &(afw->slotUseFilterIndividually()), SLOT(slot(bool)));

	// add images
	auto u1 = afw->addEntry("i1", { { i1->mat() } }, { { o1->mat() } });
	auto u2 = afw->addEntry("i2", { { i2->mat() } }, { { o2->mat() } });
	// connect
	QObject::connect((u1.at(0).getPtr()), SIGNAL(signal(cv::Mat &)),
			 o1.get(), SLOT(setMatR(cv::Mat &)));
	QObject::connect((u2.at(0).getPtr()), SIGNAL(signal(cv::Mat &)),
			 o2.get(), SLOT(setMatR(cv::Mat &)));

	// build
	auto layh = cvv::util::make_unique<QHBoxLayout>();
	auto layv1 = cvv::util::make_unique<QVBoxLayout>();
	auto layv2 = cvv::util::make_unique<QVBoxLayout>();

	auto acc = cvv::util::make_unique<cvv::qtutil::Accordion>();
	acc->push_back("button user select", std::move(busele));
	acc->push_back("button individual filter", std::move(bindiv));
	acc->push_back("afw", std::move(afw));

	layv1->addWidget(i1.release());
	layv1->addWidget(o1.release());
	layv2->addWidget(i2.release());
	layv2->addWidget(o2.release());

	layh->addWidget(acc.release());
	layh->addLayout(layv1.release());
	layh->addLayout(layv2.release());

	w.setLayout(layh.release());

	w.show();
	return a.exec();
}
