
#include <QVBoxLayout>

#include "matchshowsetting.hpp"

namespace cvv{ namespace qtutil {

MatchShowSetting::MatchShowSetting(std::vector<cv::DMatch>, QWidget *parent):
	MatchSettings{parent}
{
	auto layout=util::make_unique<QVBoxLayout>();
	auto button=util::make_unique<QPushButton>();

	button_=button.get();

	button_->setEnabled(true);
	button_->setCheckable(true);


	connect(button.get(),SIGNAL(clicked()),this,SLOT(updateAll()));
	connect(button.get(),SIGNAL(clicked()),this,SLOT(updateButton()));

	button->setChecked(true);
	layout->addWidget(button.release());

	setLayout(layout.release());
	updateButton();
}

void MatchShowSetting::updateButton()
{
	if(button_->isChecked()){
		button_->setText("show");
	}else{
		button_->setText("hide");
	}
}

}}
