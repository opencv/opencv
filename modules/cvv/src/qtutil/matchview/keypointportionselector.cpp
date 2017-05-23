#include <QVBoxLayout>

#include "keypointportionselector.hpp"

namespace cvv {namespace qtutil{

KeyPointPortionSelection::KeyPointPortionSelection(std::vector<cv::KeyPoint> , QWidget *parent):
	KeyPointSelection{parent}
{
	auto layout=util::make_unique<QVBoxLayout>();
	auto valueChooser=util::make_unique<KeyPointValueChooser>();
	auto selector=util::make_unique<PortionSelector>();

	selector_=selector.get();
	valueChooser_=valueChooser.get();

	connect(&(selector->signalSettingsChanged()),SIGNAL(signal()),this,SIGNAL(settingsChanged()));

	layout->addWidget(valueChooser.release());
	layout->addWidget(selector.release());

	setLayout(layout.release());
}

std::vector<cv::KeyPoint> KeyPointPortionSelection::select(const std::vector<cv::KeyPoint> &selection)
{
	return selector_->select(  selection ,
			[&](cv::KeyPoint arg1,cv::KeyPoint arg2)
			{return valueChooser_->getChoosenValue(arg1)<valueChooser_->getChoosenValue(arg2);});
}

}}
