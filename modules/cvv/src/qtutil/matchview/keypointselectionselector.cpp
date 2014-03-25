
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>


#include "keypointselectionselector.hpp"
#include "../../util/util.hpp"

namespace cvv{ namespace qtutil{

KeyPointSelectionSelector::KeyPointSelectionSelector(const std::vector<cv::KeyPoint> &univers, QWidget *parent):
	KeyPointSelection{parent},
	RegisterHelper<KeyPointSelection,std::vector<cv::KeyPoint>>{},
	univers_{univers}
{
	auto layout=util::make_unique<QVBoxLayout>();
	auto headerLayout=util::make_unique<QHBoxLayout>();
	auto closebutton=util::make_unique<QPushButton>("-");
	closebutton->setMaximumWidth(30);

	connect(closebutton.get(),SIGNAL(clicked()),this,SLOT(removeMe()));
	connect(&signalElementSelected(),SIGNAL(signal(QString)),this,SLOT(changeSelector()));

	headerLayout->addWidget(closebutton.release());
	headerLayout->addWidget(comboBox_);

	layout->setContentsMargins(0, 0, 0, 0);
	layout->addLayout(headerLayout.release());

	layout_=layout.get();

	setLayout(layout.release());

	if(this->has(this->selection())){
		changeSelector();
	}
}

std::vector<cv::KeyPoint> cvv::qtutil::KeyPointSelectionSelector::select(const std::vector<cv::KeyPoint> &selection)
{
	return selection_->select(selection);
}

void KeyPointSelectionSelector::changeSelector()
{
	auto selection=(*this)()(univers_);

	if(selection){
		if(selection_){
			layout_->removeWidget(selection_);
			disconnect(selection_,SIGNAL(settingsChanged()),this,SIGNAL(settingsChanged()));
			selection_->deleteLater();
		}
		selection_ = selection.get();
		connect(selection.get(),SIGNAL(settingsChanged()),this,SIGNAL(settingsChanged()));
		layout_->addWidget(selection.release());
		emit settingsChanged();
	}
}

}}
