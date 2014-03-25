#include <QVBoxLayout>
#include <QPushButton>

#include "matchsettingsselector.hpp"
#include "../../util/util.hpp"

namespace cvv{ namespace qtutil{

MatchSettingsSelector::MatchSettingsSelector(const std::vector<cv::DMatch> &univers, QWidget *parent):
	MatchSettings{parent},
	RegisterHelper<MatchSettings,std::vector<cv::DMatch>>{},
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

void MatchSettingsSelector::setSettings(CVVMatch &match)
{
	setting_->setSettings(match);
}

void MatchSettingsSelector::changedSetting()
{
	auto setting=(*this)()(univers_);
	if(setting){
		if(setting_){
			layout_->removeWidget(setting_);
			disconnect(setting_,SIGNAL(settingsChanged(MatchSettings&)),
				   this,SIGNAL(settingsChanged(MatchSettings&)));
			setting_->deleteLater();
		}
		setting_=setting.get();
		layout_->addWidget(setting.release());
		connect(setting_,SIGNAL(settingsChanged(MatchSettings&)),
			this,SIGNAL(settingsChanged(MatchSettings&)));
		setting_->updateAll();
	}
}

}}
