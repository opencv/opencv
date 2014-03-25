#include <QVBoxLayout>

#include "matchportionselector.hpp"

namespace cvv {namespace qtutil{

MatchPortionSelection::MatchPortionSelection(std::vector<cv::DMatch> , QWidget *parent):
	MatchSelection{parent}
{
	auto layout=util::make_unique<QVBoxLayout>();
	auto selector=util::make_unique<PortionSelector>();

	selector_=selector.get();

	connect(&(selector->signalSettingsChanged()),SIGNAL(signal()),this,SIGNAL(settingsChanged()));

	layout->addWidget(selector.release());

	setLayout(layout.release());
}

std::vector<cv::DMatch> MatchPortionSelection::select(const std::vector<cv::DMatch> &selection)
{
	return selector_->select(  selection ,
			[&](cv::DMatch arg1,cv::DMatch arg2)
			{return arg1<arg2;});
}

}}
