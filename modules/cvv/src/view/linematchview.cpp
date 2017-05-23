
#include <algorithm>

#include <QHBoxLayout>

#include "../qtutil/accordion.hpp"
#include "../qtutil/matchview/matchscene.hpp"
#include "../qtutil/matchview/cvvkeypoint.hpp"
#include "../qtutil/matchview/cvvmatch.hpp"
#include "../qtutil/matchview/singlecolorkeypointpen.hpp"
#include "../qtutil/matchview/matchmanagement.hpp"
#include "../qtutil/matchview/matchsettingsselector.hpp"
#include "../qtutil/matchview/showinrawviewwidget.hpp"
#include "../util/util.hpp"

#include "linematchview.hpp"

namespace cvv
{
namespace view
{

LineMatchView::LineMatchView(std::vector<cv::KeyPoint> leftKeyPoints,
			     std::vector<cv::KeyPoint> rightKeyPoints,
			     std::vector<cv::DMatch> matches, cv::Mat leftIm,
			     cv::Mat rightIm, bool usetrainIdx, QWidget *parent)
    : MatchView{ parent }
{
	std::vector<cv::KeyPoint> allkeypoints;
	for(auto key:rightKeyPoints)
	{
		allkeypoints.push_back(key);
	}

	for(auto key:leftKeyPoints){
		allkeypoints.push_back(key);
	}

	auto layout = util::make_unique<QHBoxLayout>();
	auto accor = util::make_unique<qtutil::Accordion>();
	auto matchscene = util::make_unique<qtutil::MatchScene>(leftIm, rightIm);
	auto matchmnt = util::make_unique<qtutil::MatchManagement>(matches);
	auto keyPointmnt = util::make_unique<qtutil::KeyPointManagement>(allkeypoints);

	qtutil::MatchScene *matchscene_ptr = matchscene.get();
	int updateAreaDelay=std::min(std::max(matches.size(),
					      std::max(leftKeyPoints.size(),
						       rightKeyPoints.size()))/10,50lu);
	matchscene_ptr->getLeftImage().setUpdateAreaDelay(updateAreaDelay);
	matchscene_ptr->getRightImage().setUpdateAreaDelay(updateAreaDelay);

	matchManagment_ = matchmnt.get();
	keyManagment_ = keyPointmnt.get();

	connect(&matchscene_ptr->getLeftImage(),SIGNAL(updateMouseHover(QPointF,QString,bool)),
		this,SLOT(updateMousHoverOver(QPointF,QString,bool)));
	connect(&matchscene_ptr->getRightImage(),SIGNAL(updateMouseHover(QPointF,QString,bool)),
		this,SLOT(updateMousHoverOver(QPointF,QString,bool)));

	accor->setMinimumWidth(350);
	accor->setMaximumWidth(350);

	std::vector<qtutil::CVVKeyPoint *> leftKeys;
	std::vector<qtutil::CVVKeyPoint *> rightKeys;

	accor->insert("Match Settings", std::move(matchmnt));
	accor->insert("KeyPoint Settings", std::move(keyPointmnt));
	accor->insert("Show selection in rawview window",
		      std::move(util::make_unique<qtutil::ShowInRawView>(leftKeyPoints,
								 rightKeyPoints,
								 matches,
								 matchManagment_,
								 keyManagment_)));

	accor->insert("Sync Zoom ",
		      std::move(matchscene_ptr->getSyncZoomWidget()));
	accor->insert("Left Image ",
		      std::move(matchscene_ptr->getLeftMatInfoWidget()));
	accor->insert("Right Image ",
		      std::move(matchscene_ptr->getRightMatInfoWidget()));

	layout->addWidget(accor.release());
	layout->addWidget(matchscene.release());

	setLayout(layout.release());

	for (auto &keypoint : leftKeyPoints)
	{
		auto key = util::make_unique<qtutil::CVVKeyPoint>(keypoint);
		connect(keyManagment_, SIGNAL(settingsChanged(KeyPointSettings &)),
			key.get(), SLOT(updateSettings(KeyPointSettings &)));

		leftKeys.push_back(key.get());
		matchscene_ptr->addLeftKeypoint(std::move(key));
	}

	for (auto &keypoint : rightKeyPoints)
	{
		auto key = util::make_unique<qtutil::CVVKeyPoint>(keypoint);
		connect(keyManagment_, SIGNAL(settingsChanged(KeyPointSettings &)),
			key.get(), SLOT(updateSettings(KeyPointSettings &)));

		rightKeys.push_back(key.get());
		matchscene_ptr->addRightKeyPoint(std::move(key));
	}

	for (auto &match : matches)
	{
		auto cvmatch = util::make_unique<qtutil::CVVMatch>(
		    leftKeys.at(match.queryIdx),
		    rightKeys.at((usetrainIdx ? match.trainIdx : match.imgIdx)),
		    match);

		connect(matchManagment_, SIGNAL(settingsChanged(MatchSettings &)),
			cvmatch.get(), SLOT(updateSettings(MatchSettings &)));

		matchscene_ptr->addMatch(std::move(cvmatch));
	}
	matchManagment_->updateAll();
	keyManagment_->updateAll();
}
}
}
