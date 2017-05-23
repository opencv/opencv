#include "sobelfilterwidget.hpp"

#include "opencv2/imgproc/imgproc.hpp"

#include <QVBoxLayout>
#include <QLabel>

#include "../../util/util.hpp"
#include "../filterfunctionwidget.hpp"
#include "../filterselectorwidget.hpp"

namespace cvv
{
namespace qtutil
{

SobelFilterWidget::SobelFilterWidget(QWidget *parent)
    : FilterFunctionWidget<1, 1>{ parent }, dx_{ nullptr }, dy_{ nullptr },
      ksize_{ nullptr }, borderType_{ nullptr }, gray_{ nullptr },
      grayFilter_{ nullptr }, reorder_{ nullptr }, reorderFilter_{ nullptr }
{
	auto dx = util::make_unique<QSpinBox>();
	dx_ = *dx;
	auto dy = util::make_unique<QSpinBox>();
	dy_ = *dy;
	auto ksize = util::make_unique<QComboBox>();
	ksize_ = *ksize;
	auto borderType = util::make_unique<QComboBox>();
	borderType_ = *borderType;
	// set up elements
	dx_->setRange(0, 6);
	dy_->setRange(0, 6);
	ksize_->addItem("1");
	ksize_->addItem("3");
	ksize_->addItem("5");
	ksize_->addItem("7");
	ksize_->addItem("CV_SCHARR(-1)");
	ksize_->setCurrentIndex(1);

	borderType_->addItem("BORDER_DEFAULT");
	borderType_->addItem("BORDER_CONSTANT");
	borderType_->addItem("BORDER_REPLICATE");
	borderType_->addItem("BORDER_REFLECT");
	borderType_->addItem("BORDER_REFLECT_101");

	// connect
	QObject::connect(dx_.getPtr(), SIGNAL(valueChanged(int)),
	                 &(this->signalFilterSettingsChanged()),
	                 SIGNAL(signal()));
	QObject::connect(dy_.getPtr(), SIGNAL(valueChanged(int)),
	                 &(this->signalFilterSettingsChanged()),
	                 SIGNAL(signal()));
	QObject::connect(ksize_.getPtr(), SIGNAL(currentIndexChanged(int)),
	                 &(this->signalFilterSettingsChanged()),
	                 SIGNAL(signal()));
	QObject::connect(borderType_.getPtr(), SIGNAL(currentIndexChanged(int)),
	                 &(this->signalFilterSettingsChanged()),
	                 SIGNAL(signal()));

	// subfilter reorder
	auto reorder = util::make_unique<QCheckBox>("Reorder channels");
	reorder_ = *reorder;
	auto reorderFilter = util::make_unique<ChannelReorderFilter>();
	reorderFilter_ = *reorderFilter;
	reorder_->setChecked(false);
	reorderFilter_->setVisible(false);
	// visible
	QObject::connect(reorder_.getPtr(), SIGNAL(clicked(bool)),
	                 reorderFilter_.getPtr(), SLOT(setVisible(bool)));
	// settings
	QObject::connect(reorder_.getPtr(), SIGNAL(clicked()),
	                 &(this->signalFilterSettingsChanged()),
	                 SIGNAL(signal()));
	QObject::connect(
	    &(reorderFilter_.getPtr()->signalFilterSettingsChanged()),
	    SIGNAL(signal()), &(this->signalFilterSettingsChanged()),
	    SIGNAL(signal()));

	// subfilter gray
	auto gray = util::make_unique<QCheckBox>("Apply gray filter");
	gray_ = *gray;
	auto grayFilter = util::make_unique<GrayFilterWidget>();
	grayFilter_ = *grayFilter;
	gray_->setChecked(false);
	grayFilter_->setVisible(false);
	// visible
	QObject::connect(gray_.getPtr(), SIGNAL(clicked(bool)),
	                 grayFilter_.getPtr(), SLOT(setVisible(bool)));
	// settings
	QObject::connect(gray_.getPtr(), SIGNAL(clicked()),
	                 &(this->signalFilterSettingsChanged()),
	                 SIGNAL(signal()));
	QObject::connect(&(grayFilter_.getPtr()->signalFilterSettingsChanged()),
	                 SIGNAL(signal()),
	                 &(this->signalFilterSettingsChanged()),
	                 SIGNAL(signal()));

	// build ui
	auto lay = util::make_unique<QVBoxLayout>();
	lay->addWidget(reorder.release());
	lay->addWidget(reorderFilter.release());
	lay->addWidget(gray.release());
	lay->addWidget(grayFilter.release());
	lay->addWidget(util::make_unique<QLabel>("dx").release());
	lay->addWidget(dx.release());
	lay->addWidget(util::make_unique<QLabel>("dy").release());
	lay->addWidget(dy.release());
	lay->addWidget(util::make_unique<QLabel>("ksize").release());
	lay->addWidget(ksize.release());
	lay->addWidget(util::make_unique<QLabel>("borderType").release());
	lay->addWidget(borderType.release());
	setLayout(lay.release());
	// emit first update
	signalFilterSettingsChanged().emitSignal();
}

void SobelFilterWidget::applyFilter(InputArray in, OutputArray out) const
{
	int ksize = 3;
	switch (ksize_->currentIndex())
	{
	case 0:
		ksize = 1;
		break;
	case 1:
		ksize = 3;
		break;
	case 2:
		ksize = 5;
		break;
	case 3:
		ksize = 7;
		break;
	case 4:
		ksize = CV_SCHARR;
		break;
	}

	int borderType = cv::BORDER_DEFAULT;
	switch (borderType_->currentIndex())
	{
	case 0:
		borderType = cv::BORDER_DEFAULT;
		break;
	case 1:
		borderType = cv::BORDER_CONSTANT;
		break;
	case 2:
		borderType = cv::BORDER_REPLICATE;
		break;
	case 3:
		borderType = cv::BORDER_REFLECT;
		break;
	case 4:
		borderType = cv::BORDER_REFLECT_101;
		break;
	}

	int dx = dx_->value();
	int dy = dy_->value();
	// apply filter
	cvv::util::Reference<const cv::Mat> inar = in.at(0).get();
	cvv::util::Reference<cv::Mat> outar = out.at(0).get();
	// first reorder
	if (reorder_->isChecked())
	{
		reorderFilter_->applyFilter({ { inar } }, { { outar } });
		// out should be new input
		inar = outar.get();
	}
	// then gray
	if (gray_->isChecked())
	{
		grayFilter_->applyFilter({ { inar } }, { { outar } });
		// out should be new input
		inar = outar.get();
	}
	Sobel(inar.get(), outar.get(), -1, dx, dy, ksize, 1, 0, borderType);
}

std::pair<bool, QString> SobelFilterWidget::checkInput(InputArray in) const
{
	// check depth in CV_8U,CV_16U,CV_16S,CV_32F,CV_64F
	switch (in.at(0).get().depth())
	{
	case CV_8U:
	case CV_16U:
	case CV_16S:
	case CV_32F:
	case CV_64F:
		break;
	default:
		return { false, QString("unsupported depth: ") +
			            QString::number(in.at(0).get().depth()) };
	}

	// check subfilter
	if (gray_->isChecked())
	{
		auto resultGray = grayFilter_->checkInput(in);
		if (!resultGray.first)
		{
			return resultGray;
		}
	}
	if (reorder_->isChecked())
	{
		auto resultReorder = reorderFilter_->checkInput(in);
		if (!resultReorder.first)
		{
			return resultReorder;
		}
	}

	// check channels
	if (!(gray_->isChecked())) // gray filter => channels will be 1
	{
		if ((reorder_->isChecked()) &&
		    (reorderFilter_->outputChannels() > 4)) // no gray
		{
			return { false, "channels>4 (use gray filter or "
			                "reorder with <=4 output channels)" };
		}
		else if ((in.at(0).get().channels() > 4)) // no gray filter +
		                                          // reorder
		{
			return { false, "channels>4 (use gray filter or "
			                "reorder with <=4 output channels)" };
		}
	}

	int dx = dx_->value();
	int dy = dy_->value();
	if (dx == 0 && dy == 0)
	{
		return { false, "dx=0 and dy=0" };
	}
	// dx,dy<ksize, if sharr:  dx XOR dy
	if (dx_->value() == 0 && dy_->value() == 0)
	{
		return { false, "dx=0 and dy=0" };
	}

	int ksize = 3;
	switch (ksize_->currentIndex())
	{
	case 0:
		ksize = 1;
		break;
	case 1:
		ksize = 3;
		break;
	case 2:
		ksize = 5;
		break;
	case 3:
		ksize = 7;
		break;
	case 4:
		ksize = CV_SCHARR;
		break;
	}

	if (ksize == CV_SCHARR)
	{
		if (dx + dy != 1)
		{
			return { false, "ksize=CV_SCHARR but dx+dy != 1" };
		}
	}
	else
	{
		if ((dx >= 3 || dy >= 3) && (dx >= ksize || dy >= ksize))
		{
			return { false, "dx or dy is to big" };
		}
	}

	return { true, "" };
}
}
}
