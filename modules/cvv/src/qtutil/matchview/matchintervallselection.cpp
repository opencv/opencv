
#include <QVBoxLayout>

#include <algorithm>

#include "matchintervallselection.hpp"
#include "../../util/util.hpp"

namespace cvv{ namespace qtutil{

MatchIntervallSelector::MatchIntervallSelector(std::vector<cv::DMatch> matches, QWidget *parent):
	MatchSelection{parent}
{
	double min=0.0;
	double max=0.0;

	for(auto& match:matches)
	{
		min=std::min(static_cast<double>(match.distance),min);
		max=std::max(static_cast<double>(match.distance),max);
	}

	auto layout=util::make_unique<QVBoxLayout>();
	auto selector=util::make_unique<IntervallSelector>(min,max);

	selector_=selector.get();
	connect(&(selector->signalSettingsChanged()),SIGNAL(signal()),this,SIGNAL(settingsChanged()));

	layout->addWidget(selector.release());
	setLayout(layout.release());
}

std::vector<cv::DMatch> MatchIntervallSelector::select(const std::vector<cv::DMatch> &selection)
{
	return selector_->select(selection, [&](const cv::DMatch& match){return match.distance;});
}


}}
