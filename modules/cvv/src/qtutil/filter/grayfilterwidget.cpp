#include "grayfilterwidget.hpp"

#include <QPushButton>
#include <QLabel>

#include "../filterselectorwidget.hpp"
#include "../../util/util.hpp"
#include "../util.hpp"

namespace cvv
{
namespace qtutil
{

GrayFilterWidget::GrayFilterWidget(QWidget *parent)
    : FilterFunctionWidget<1, 1>{ parent }, layout_{ nullptr },
      channel_{ nullptr }, chanValues_{}
{
	// set a tooltip
	setToolTip(
	    "nonexistant channels from source will be seen as a zero mat");
	// create the layout
	auto lay = util::make_unique<QVBoxLayout>();
	layout_ = *lay;
	// create the spinbox to select the number of channels
	auto channel = util::make_unique<QSpinBox>();
	channel_ = *channel;
	// create a button to set up the default gray filter
	auto button = util::make_unique<QPushButton>("use default rgb to gray");
	QObject::connect(button.get(), SIGNAL(clicked()), this, SLOT(setStd()));

	// set up the spinbox to select the number of channels.
	channel_->setRange(1, 10);
	// and connect it with the slot setChannel.
	QObject::connect(channel_.getPtr(), SIGNAL(valueChanged(int)), this,
			 SLOT(setChannel(int)));

	// build ui (some labels for the user are added)
	layout_->addWidget(button.release());
	layout_->addWidget(
	    util::make_unique<QLabel>("Number of channels").release());
	layout_->addWidget(channel.release());
	layout_->addWidget(
	    util::make_unique<QLabel>("Percentage for channels").release());
	setLayout(lay.release());

	// set up the default gray filter
	setStd();
}

void GrayFilterWidget::applyFilter(InputArray in, OutputArray out) const
{
	// check weather the filter can be applied
	if (!(checkInput(in).first))
	{
		return;
	}
	// the filter can be applied
	// split the cannels of the input image
	auto channels = splitChannels(in.at(0).get());
	// create a zero image
	cv::Mat tmp = cv::Mat::zeros(in.at(0).get().rows, in.at(0).get().cols,
				     in.at(0).get().depth());
	// multiply all channels with their factor and add it to tmp
	// if there are factors for more channels than the input image has, this
	// channels
	// will be ignored
	for (std::size_t i = 0;
	     ((i < channels.size()) && (i < chanValues_.size())); i++)
	{
		// multiply each channel with its factor and add the result to
		// tmp
		tmp += channels.at(i) * (chanValues_.at(i)->value());
	}
	// finally assign tmp to out
	out.at(0).get() = tmp;
}

std::pair<bool, QString> GrayFilterWidget::checkInput(InputArray) const
{
	// checks wheather the current settings are valid.
	// add up all factors
	double sum = 0;
	for (auto &elem : chanValues_)
	{
		sum += (elem->value());
	}
	// check wheather the sum is <=1
	if (sum > 1)
	{
		// the settings are invalid => return fale + a error message
		return { false, QString{ "total : " } + QString::number(sum) +
				    QString{ " > 1" } };
	}
	// the settings are valid
	return { true, "" };
}

void GrayFilterWidget::setChannel(std::size_t n)
{
	/*
	 * this function is recursive.
	 */
	if (n == chanValues_.size())
	{
		// stop recursion
		return;
	}
	else if (n < chanValues_.size())
	{
		// currently there are more channels than requested.
		// => remove one channel
		// remove a spin box from the vector
		QDoubleSpinBox *box = chanValues_.back().getPtr();
		chanValues_.pop_back();
		// remove it from the layout
		layout_->removeWidget(box);
		// reset the parent
		box->setParent(nullptr);
		// finally delete it
		box->deleteLater();
	}
	else
	{
		// currently less channel than requested
		// => add one channel
		// create a new spinbox, set its range and step size.
		auto box = util::make_unique<QDoubleSpinBox>();
		box->setRange(0, 1);
		box->setSingleStep(0.01);
		// add this box to the vector
		chanValues_.emplace_back(*box);
		// connect it to signFilterSettingsChanged_.
		QObject::connect(box.get(), SIGNAL(valueChanged(double)),
				 &(this->signalFilterSettingsChanged()),
				 SIGNAL(signal()));
		// and add it to the layout
		layout_->addWidget(box.release());
	}
	// recursion
	setChannel(n);
}

void GrayFilterWidget::setStd()
{
	// use 3 channels (b g r)
	channel_->setValue(3);
	// set factor for b
	chanValues_.at(0)->setValue(0.114);
	// set factor for g
	chanValues_.at(1)->setValue(0.587);
	// set factor for r
	chanValues_.at(2)->setValue(0.299);
}
}
}
