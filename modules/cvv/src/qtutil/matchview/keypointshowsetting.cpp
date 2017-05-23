
#include <QVBoxLayout>

#include "keypointshowsetting.hpp"

namespace cvv{ namespace qtutil {

KeyPointShowSetting::KeyPointShowSetting(std::vector<cv::KeyPoint>, QWidget *parent):
	KeyPointSettings{parent}
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

void KeyPointShowSetting::updateButton()
{
	if(button_->isChecked()){
		button_->setText("show");
	}else{
		button_->setText("hide");
	}
}

}}
