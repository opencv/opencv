#include "channelreorderfilter.hpp"

#include <QLabel>

#include "../util.hpp"

namespace cvv
{
namespace qtutil
{

ChannelReorderFilter::ChannelReorderFilter(QWidget *parent)
    : FilterFunctionWidget<1, 1>{ parent }, layout_{ nullptr },
      channel_{ nullptr }, channelAssignment_{}
{
	setToolTip(
	    "nonexistant channels from source will be seen as a zero mat");
	auto lay = util::make_unique<QVBoxLayout>();
	layout_ = *lay;
	auto channel = util::make_unique<QSpinBox>();
	channel_ = *channel;

	// channelselector
	channel_->setRange(1, 10);
	QObject::connect(channel_.getPtr(), SIGNAL(valueChanged(int)), this,
	                 SLOT(setChannel(int)));

	// build ui
	layout_->addWidget(
	    util::make_unique<QLabel>("Number of channels").release());
	layout_->addWidget(channel.release());
	layout_->addWidget(util::make_unique<QLabel>(
	    "Assignment for the old channels").release());
	setLayout(lay.release());

	channel_->setValue(3);
}

std::pair<bool, QString> ChannelReorderFilter::checkInput(InputArray in) const
{
	if (in.at(0)->channels() < 1)
	{
		return { false, "<1 channel" };
	}
	return { true, "" };
}

void ChannelReorderFilter::applyFilter(InputArray in, OutputArray out) const
{
	auto chans = splitChannels(in.at(0).get());
	cv::Mat zeros = cv::Mat::zeros(chans.front().rows, chans.front().cols,
	                               chans.front().type());
	std::vector<cv::Mat> toMerge{};
	for (std::size_t i = 0; i < channelAssignment_.size(); i++)
	{
		if (static_cast<std::size_t>(
		        channelAssignment_.at(i)->value()) < chans.size())
		{
			toMerge.push_back(
			    chans.at(channelAssignment_.at(i)->value()));
		}
		else
		{
			toMerge.push_back(zeros);
		}
	}
	out.at(0).get() = mergeChannels(toMerge);
}

void ChannelReorderFilter::setChannel(std::size_t n)
{
	if (n == channelAssignment_.size())
	{
		// stop rec + update
		signalFilterSettingsChanged().emitSignal();
		return;
	}
	else if (n < channelAssignment_.size())
	{
		// remove one channel
		QSpinBox *box = channelAssignment_.back().getPtr();
		channelAssignment_.pop_back();
		layout_->removeWidget(box);
		box->setParent(nullptr);
		box->deleteLater();
	}
	else
	{
		// add one channel
		auto box = util::make_unique<QSpinBox>();
		box->setRange(0, 9);
		box->setSingleStep(1);
		box->setValue(channelAssignment_.size());
		channelAssignment_.emplace_back(*box);
		// connect
		QObject::connect(box.get(), SIGNAL(valueChanged(int)),
		                 &(this->signalFilterSettingsChanged()),
		                 SIGNAL(signal()));
		layout_->addWidget(box.release());
	}
	// rec
	setChannel(n);
}
}
}
