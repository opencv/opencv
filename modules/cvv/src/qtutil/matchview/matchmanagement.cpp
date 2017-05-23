
#include <algorithm>

#include <QVBoxLayout>
#include <QGridLayout>
#include <QPushButton>
#include <QLabel>
#include <QFrame>

#include "matchmanagement.hpp"

namespace cvv
{
namespace qtutil
{

MatchManagement::MatchManagement(std::vector<cv::DMatch> univers,QWidget *parent) :
	MatchSettings{parent},
	univers_{univers},
	selection_{univers_}
{
	auto basicLayout=util::make_unique<QVBoxLayout>();
	auto buttonLayout=util::make_unique<QGridLayout>();
	auto settingsLayout=util::make_unique<QVBoxLayout>();
	auto selectorLayout=util::make_unique<QVBoxLayout>();

	auto buttonFrame=util::make_unique<QFrame>();
	buttonFrame->setLineWidth(1);
	buttonFrame->setFrameStyle(QFrame::Box);

	auto labelSettings=util::make_unique<QLabel>("Settings");
	auto labelSelection=util::make_unique<QLabel>("Selection");

	auto buttonAddSetting=util::make_unique<QPushButton>("Add setting");
	auto buttonAddSelection=util::make_unique<QPushButton>("Add selector");
	//auto buttonApply=util::make_unique<QPushButton>("Apply settings");
	auto showOnlySelection=util::make_unique<QCheckBox>("Show selection only");
	auto buttonApplySelection=util::make_unique<QPushButton>("Apply Selection");
	auto buttonSelectAll=util::make_unique<QPushButton>("Select all");
	auto buttonSelectNone=util::make_unique<QPushButton>("Select none");

	connect(buttonAddSetting.get(),SIGNAL(clicked()),this,SLOT(addSetting()));
	connect(buttonAddSelection.get(),SIGNAL(clicked()),this,SLOT(addSelection()));
	//connect(buttonApply.get(),SIGNAL(clicked()),this,SLOT(updateAll()));
	connect(showOnlySelection.get(),SIGNAL(clicked()),this,SLOT(updateAll()));
	connect(buttonApplySelection.get(),SIGNAL(clicked()),this,SLOT(applySelection()));
	connect(buttonSelectAll.get(),SIGNAL(clicked()),this,SLOT(selectAll()));
	connect(buttonSelectNone.get(),SIGNAL(clicked()),this,SLOT(selectNone()));

	settingsLayout_=settingsLayout.get();
	selectorLayout_=selectorLayout.get();
	showOnlySelection_=showOnlySelection.get();

	showOnlySelection->setChecked(true);

	buttonLayout->addWidget(buttonAddSetting.release(),0,0);
	buttonLayout->addWidget(buttonAddSelection.release(),0,1);
	buttonLayout->addWidget(buttonApplySelection.release(),1,0);
	//buttonLayout->addWidget(buttonApply.release(),1,1);
	buttonLayout->addWidget(showOnlySelection.release(),1,1);
	buttonLayout->addWidget(buttonSelectAll.release(),2,0);
	buttonLayout->addWidget(buttonSelectNone.release(),2,1);

	buttonFrame->setLayout(buttonLayout.release());

	basicLayout->addWidget(buttonFrame.release());
	basicLayout->addWidget(labelSettings.release());
	basicLayout->addLayout(settingsLayout.release());
	basicLayout->addWidget(labelSelection.release());
	basicLayout->addLayout(selectorLayout.release());

	basicLayout->setContentsMargins(0, 0, 0, 0);

	setLayout(basicLayout.release());

	addSelection();
	addSetting();
}

void MatchManagement::setSettings(CVVMatch &match)
{
	if(showOnlySelection_->isChecked())
	{
		if (std::find_if(selection_.begin(), selection_.end(),
				 [&](const cv::DMatch &o)
			{ return match == o; }) != selection_.end())
		{
			match.setShow(true);
		}else{
			match.setShow(false);
		}
	}/*else{
		if (std::find_if(selection_.begin(), selection_.end(),
				 [&](const cv::DMatch &o)
			{ return match == o; }) != selection_.end())
		{
			connect(this,SIGNAL(applySettingsToSelection(MatchSettings&)),
				&match,SLOT(updateSettings(MatchSettings&)));
			for(auto setting: settingsList_)
			{
				setting->setSettings(match);
			}
		}else{
			disconnect(this,SIGNAL(applySettingsToSelection(MatchSettings&)),
				&match,SLOT(updateSettings(MatchSettings&)));
			for(auto setting: settingsList_)
			{
				setting->setUnSelectedSettings(match);
			}
		}
	}*/
}

void MatchManagement::addToSelection(const cv::DMatch &match)
{
	selection_.push_back(match);
	updateAll();
}

void MatchManagement::singleSelection(const cv::DMatch &match)
{
	selection_.clear();
	selection_.push_back(match);
	updateAll();
}

void MatchManagement::setSelection(
    const std::vector<cv::DMatch> &selection)
{
	selection_.clear();
	for (auto &match : selection)
	{
		selection_.push_back(match);
	}
	updateAll();
}

void MatchManagement::addSetting()
{
	addSetting(std::move(util::make_unique<MatchSettingsSelector>(univers_)));
}


void MatchManagement::addSetting(std::unique_ptr<MatchSettingsSelector> setting)
{
	connect(setting.get(),SIGNAL(settingsChanged(MatchSettings &)),
		this,SIGNAL(settingsChanged(MatchSettings&)));

	connect(setting.get(),SIGNAL(remove(MatchSettingsSelector *)),
		this,SLOT(removeSetting(MatchSettingsSelector*)));

	settingsList_.push_back(setting.get());
	setting->setLineWidth(1);
	setting->setFrameStyle(QFrame::Box);
	settingsLayout_->addWidget(setting.release());
}

void MatchManagement::removeSetting(MatchSettingsSelector *setting)
{
	auto it = std::find(settingsList_.begin(), settingsList_.end(), setting);

	if(it == settingsList_.end())
	{
		return;
	}

	settingsList_.erase(it);
	settingsLayout_->removeWidget(setting);
	setting->deleteLater();
	updateAll();
}

void MatchManagement::addSelection()
{
	addSelection(std::move(util::make_unique<MatchSelectionSelector>(univers_)));
}

void MatchManagement::addSelection(std::unique_ptr<MatchSelectionSelector> selection)
{
	connect(selection.get(),SIGNAL(remove(MatchSelectionSelector*))
		,this,SLOT(removeSelection(MatchSelectionSelector*)));

	connect(selection.get(),SIGNAL(settingsChanged()),this,SLOT(applySelection()));
	selectorList_.push_back(selection.get());
	selection->setLineWidth(1);
	selection->setFrameStyle(QFrame::Box);
	selectorLayout_->addWidget(selection.release());
}

void MatchManagement::removeSelection(MatchSelectionSelector *selector)
{
	auto it = std::find(selectorList_.begin(), selectorList_.end(), selector);

	if(it == selectorList_.end())
	{
		return;
	}

	selectorList_.erase(it);

	selectorLayout_->removeWidget(selector);

	selector->deleteLater();
	updateAll();
}

void MatchManagement::applySelection()
{
	std::vector<cv::DMatch> currentSelection=univers_;
	for(auto& selector:selectorList_){
		currentSelection=selector->select(currentSelection);
	}
	selection_=currentSelection;
	emit updateSelection(selection_);
	updateAll();
}

}
}
