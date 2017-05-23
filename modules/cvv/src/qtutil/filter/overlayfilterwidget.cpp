
#include <opencv2/core/core.hpp>

#include <Qt>
#include "QLabel"
#include "QSlider"
#include "QVBoxLayout"

#include "../../util/util.hpp"
#include "overlayfilterwidget.hpp"

namespace cvv
{
namespace qtutil
{

OverlayFilterWidget::OverlayFilterWidget(QWidget *parent)
    : FilterFunctionWidget<2, 1>{ parent }, opacityOfFilterImg_{ 0.5 }
{

	auto layout = util::make_unique<QVBoxLayout>();
	auto slider = util::make_unique<QSlider>(Qt::Horizontal);

	slider->setRange(0, 100);
	slider->setSliderPosition(50);
	slider->setTickPosition(QSlider::TicksAbove);
	slider->setTickInterval(10);

	connect(slider.get(), SIGNAL(valueChanged(int)), this,
	        SLOT(updateOpacity(int)));

	// Add title of slider and slider to the layout
	layout->addWidget(util::make_unique<QLabel>(
	    "Select opacity of right image").release());
	layout->addWidget(slider.release());
	setLayout(layout.release());

}

void OverlayFilterWidget::applyFilter(InputArray in, OutputArray out) const
{

	auto check = checkInput(in);
	if (!check.first)
	{
		return;
	}

	cv::addWeighted(in.at(0).get(), 1 - opacityOfFilterImg_, in.at(1).get(),
	                opacityOfFilterImg_, 0, out.at(0).get());

}

std::pair<bool, QString> OverlayFilterWidget::checkInput(InputArray in) const
{
	// check whether images have same size
	if (in.at(0).get().size() != in.at(1).get().size())
	{
		return std::make_pair(false, "Images need to have same size");
	}

	// check whether images have same number of channels
	if (in.at(0).get().channels() != in.at(1).get().channels())
	{
		return std::make_pair(
		    false, "Images need to have same number of channels");
	}


	return std::make_pair(true, "Images can be converted");
}

void OverlayFilterWidget::updateOpacity(int newOpacity)
{
	opacityOfFilterImg_ = newOpacity / 100.0;
	signalFilterSettingsChanged().emitSignal();
}
}
}
