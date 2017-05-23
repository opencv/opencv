#include <QVBoxLayout>
#include <QPushButton>

#include "showinrawviewwidget.hpp"
#include "../../util/util.hpp"

namespace  cvv {namespace qtutil{

ShowInRawView::ShowInRawView(const std::vector<cv::KeyPoint> left_key,
			     const std::vector<cv::KeyPoint> right_key,
			     const std::vector<cv::DMatch> matches,
			     MatchManagement* matchmnt,
			     KeyPointManagement* keymnt,
			     QWidget *parent):
	QWidget{parent},
	matchmnt_{matchmnt},
	keymnt_{keymnt},
	rawViewWindow_{nullptr},
	left_key_{left_key},
	right_key_{right_key},
	matches_{matches}
{
	auto layout=util::make_unique<QVBoxLayout>();
	auto selectionbutton=util::make_unique<QPushButton>("show current Selection");

	connect(selectionbutton.get(),SIGNAL(clicked()),this,SLOT(getcurrentSelection()));

	layout->addWidget(selectionbutton.release());
	setLayout(layout.release());
}

ShowInRawView::~ShowInRawView()
{
	if (rawViewWindow_)
	{
		rawViewWindow_->deleteLater();
	}

}

void ShowInRawView::hideEvent(QHideEvent *)
{
	getcurrentSelection();
	rawViewWindow_->hide();
}

void ShowInRawView::getcurrentSelection()
{
	if(!rawViewWindow_){
		rawViewWindow_=new RawviewWindow{"Current Selection",left_key_,right_key_,matches_};

		connect(keymnt_,SIGNAL(updateSelection(std::vector<cv::KeyPoint>)),
			rawViewWindow_,SLOT(selectKeyPoints(std::vector<cv::KeyPoint>)));
		connect(matchmnt_,SIGNAL(updateSelection(std::vector<cv::DMatch>)),
			rawViewWindow_,SLOT(selectMatches(std::vector<cv::DMatch>)));

		connect(rawViewWindow_,SIGNAL(keyPointsSelected(std::vector<cv::KeyPoint>)),
			keymnt_,SLOT(setSelection(std::vector<cv::KeyPoint>)));
		connect(rawViewWindow_,SIGNAL(matchesSelected(std::vector<cv::DMatch>)),
			matchmnt_,SLOT(setSelection(std::vector<cv::DMatch>)));
	}
	rawViewWindow_->selectKeyPoints(keymnt_->getCurrentSelection());
	rawViewWindow_->selectMatches(matchmnt_->getCurrentSelection());
	rawViewWindow_->setVisible(true);
}


void ShowInRawView::showEvent(QShowEvent *)
{
	getcurrentSelection();
	rawViewWindow_->show();
}
}}
