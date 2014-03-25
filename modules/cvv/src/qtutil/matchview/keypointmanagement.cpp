
#include <algorithm>

#include <QVBoxLayout>
#include <QGridLayout>
#include <QPushButton>
#include <QLabel>
#include <QFrame>

#include "keypointmanagement.hpp"

namespace cvv
{
namespace qtutil
{

KeyPointManagement::KeyPointManagement(std::vector<cv::KeyPoint> univers,QWidget *parent) :
	KeyPointSettings{parent},
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

void KeyPointManagement::setSettings(CVVKeyPoint &key)
{
	if(showOnlySelection_->isChecked())
	{
		if (std::find_if(selection_.begin(), selection_.end(),
				 [&](const cv::KeyPoint &o)
			{ return key == o; }) != selection_.end())
		{
			key.setShow(true);
		}else{
			key.setShow(false);
		}
	}/*else{
		if (std::find_if(selection_.begin(), selection_.end(),
				 [&](const cv::KeyPoint &o)
			{ return key == o; }) != selection_.end())
		{
			connect(this,SIGNAL(applySettingsToSelection(KeyPointSettings&)),
				&key,SLOT(updateSettings(KeyPointSettings&)));
			for(auto setting: settingsList_)
			{
				setting->setSettings(key);
			}
		}else{

			disconnect(this,SIGNAL(applySettingsToSelection(KeyPointSettings&)),
				&key,SLOT(updateSettings(KeyPointSettings&)));
			for(auto setting: settingsList_)
			{
				setting->setUnSelectedSettings(key);
			}
		}
	}*/

}

void KeyPointManagement::addToSelection(const cv::KeyPoint &key)
{
	selection_.push_back(key);
	emit updateSelection(selection_);
	updateAll();
}

void KeyPointManagement::singleSelection(const cv::KeyPoint &key)
{
	selection_.clear();
	selection_.push_back(key);
	emit updateSelection(selection_);
	updateAll();
}

void KeyPointManagement::setSelection(
    const std::vector<cv::KeyPoint> &selection)
{
	selection_.clear();
	for (auto &key : selection)
	{
		selection_.push_back(key);
	}
	emit updateSelection(selection_);
	updateAll();
}

void KeyPointManagement::addSetting()
{
	addSetting(std::move(util::make_unique<KeyPointSettingsSelector>(univers_)));
}


void KeyPointManagement::addSetting(std::unique_ptr<KeyPointSettingsSelector> setting)
{
	connect(setting.get(),SIGNAL(settingsChanged(KeyPointSettings &)),
		this,SIGNAL(settingsChanged(KeyPointSettings&)));

	connect(setting.get(),SIGNAL(remove(KeyPointSettingsSelector *)),
		this,SLOT(removeSetting(KeyPointSettingsSelector*)));

	settingsList_.push_back(setting.get());
	setting->setLineWidth(1);
	setting->setFrameStyle(QFrame::Box);
	settingsLayout_->addWidget(setting.release());
}

void KeyPointManagement::removeSetting(KeyPointSettingsSelector *setting)
{
	auto it = std::find(settingsList_.begin(), settingsList_.end(), setting);

	if(it == settingsList_.end())
	{
		return;
	}

	settingsList_.erase(it);
	settingsLayout_->removeWidget(setting);
	setting->deleteLater();
}

void KeyPointManagement::addSelection()
{
	addSelection(std::move(util::make_unique<KeyPointSelectionSelector>(univers_)));
}

void KeyPointManagement::addSelection(std::unique_ptr<KeyPointSelectionSelector> selection)
{
	connect(selection.get(),SIGNAL(remove(KeyPointSelectionSelector*))
		,this,SLOT(removeSelection(KeyPointSelectionSelector*)));

	connect(selection.get(),SIGNAL(settingsChanged()),this,SLOT(applySelection()));
	selectorList_.push_back(selection.get());
	selection->setLineWidth(1);
	selection->setFrameStyle(QFrame::Box);
	selectorLayout_->addWidget(selection.release());
}

void KeyPointManagement::removeSelection(KeyPointSelectionSelector *selector)
{
	auto it = std::find(selectorList_.begin(), selectorList_.end(), selector);

	if(it == selectorList_.end())
	{
		return;
	}

	selectorList_.erase(it);
	selectorLayout_->removeWidget(selector);

	selector->deleteLater();
}

void KeyPointManagement::applySelection()
{
	std::vector<cv::KeyPoint> currentSelection=univers_;
	for(auto& selector:selectorList_){
		currentSelection=selector->select(currentSelection);
	}
	selection_=currentSelection;
	emit updateSelection(selection_);
	updateAll();
}

}
}
