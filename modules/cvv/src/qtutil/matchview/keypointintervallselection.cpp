
#include <QVBoxLayout>

#include <algorithm>
#include <iostream>

#include "keypointintervallselection.hpp"
#include "../../util/util.hpp"

namespace cvv{ namespace qtutil{

KeyPointIntervallSelector::KeyPointIntervallSelector(std::vector<cv::KeyPoint> keypoints, QWidget *parent):
	KeyPointSelection{parent},
	layout_{nullptr},
	selector_{nullptr},
	valueChooser_{nullptr},
	keypoints_{keypoints}
{
	auto layout=util::make_unique<QVBoxLayout>();
	auto valueChooser=util::make_unique<KeyPointValueChooser>();

	valueChooser_=valueChooser.get();

	connect(valueChooser_,SIGNAL(valueChanged()),this,SLOT(changeSelecteValue()));

	layout->setContentsMargins(0, 0, 0, 0);
	layout_=layout.get();

	layout->addWidget(valueChooser.release());
	setLayout(layout.release());

	changeSelecteValue();
}

std::vector<cv::KeyPoint> KeyPointIntervallSelector::select(const std::vector<cv::KeyPoint> &selection)
{
	return selector_->select(selection, [&](const cv::KeyPoint& key)
				{return this->valueChooser_->getChoosenValue(key);}
	);
}

void KeyPointIntervallSelector::changeSelecteValue()
{
	if(selector_){
		layout_->removeWidget(selector_);
		selector_->deleteLater();
	}
	double min=-1;
	double max=0;
	for(auto& key:keypoints_)
	{

		min=std::min(valueChooser_->getChoosenValue(key),min);
		max=std::max(valueChooser_->getChoosenValue(key),max);
	}
	auto selector=util::make_unique<IntervallSelector>(min,max);
	selector_=selector.get();
	connect(&(selector->signalSettingsChanged()),SIGNAL(signal()),this,SIGNAL(settingsChanged()));
	layout_->addWidget(selector.release());

}


}}
