#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>

#include "keypointsettingsselector.hpp"
#include "../../util/util.hpp"

namespace cvv{ namespace qtutil{

KeyPointSettingsSelector::KeyPointSettingsSelector(const std::vector<cv::KeyPoint> &univers, QWidget *parent):
	KeyPointSettings{parent},
	RegisterHelper<KeyPointSettings,std::vector<cv::KeyPoint>>{},
	univers_{univers}
{
	auto layout=util::make_unique<QVBoxLayout>();
	auto headerLayout=util::make_unique<QHBoxLayout>();
	auto closebutton=util::make_unique<QPushButton>("-");
	closebutton->setMaximumWidth(30);

	connect(closebutton.get(),SIGNAL(clicked()),this,SLOT(removeMe()));
	connect(&signalElementSelected(),SIGNAL(signal(QString)),this,SLOT(changedSetting()));

	headerLayout->addWidget(closebutton.release());
	headerLayout->addWidget(comboBox_);

	layout->setContentsMargins(0, 0, 0, 0);
	layout->addLayout(headerLayout.release());

	layout_=layout.get();
	setLayout(layout.release());
	if(this->has(this->selection())){
		changedSetting();
	}
}

void KeyPointSettingsSelector::setSettings(CVVKeyPoint &key)
{
	setting_->setSettings(key);
}

void KeyPointSettingsSelector::changedSetting()
{
	auto setting=(*this)()(univers_);
	if(setting){
		if(setting_){
			layout_->removeWidget(setting_);
			disconnect(setting_,SIGNAL(settingsChanged(KeyPointSettings&)),
				   this,SIGNAL(settingsChanged(KeyPointSettings&)));
			setting_->deleteLater();
		}
		setting_=setting.get();
		layout_->addWidget(setting.release());
		connect(setting_,SIGNAL(settingsChanged(KeyPointSettings&)),
		this,SIGNAL(settingsChanged(KeyPointSettings&)));
		setting_->updateAll();
	}
}

}}
