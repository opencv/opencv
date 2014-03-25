#include "rawview.hpp"

#include <functional>

#include <QMap>
#include <QSet>
#include <QString>
#include <QVBoxLayout>
#include <QWidget>
#include <QScrollArea>

#include "../controller/view_controller.hpp"
#include "../qtutil/stfl_query_widget.hpp"
#include "../qtutil/util.hpp"

namespace cvv
{
namespace view
{

Rawview::Rawview(const std::vector<cv::KeyPoint> &keyPoints1,
                 const std::vector<cv::KeyPoint> &keyPoints2,
                 const std::vector<cv::DMatch> &matches,
				 bool usesTrainDescriptor,
                 bool showShowInViewMenu)
    : keyPoints1{keyPoints1}, keyPoints2{keyPoints2}, matches{matches},
	  showShowInViewMenu{ showShowInViewMenu },
	  usesTrainDescriptor{ usesTrainDescriptor }
{
	queryWidget = new qtutil::STFLQueryWidget();
	table = new gui::RawviewTable(this);
	QVBoxLayout *layout = new QVBoxLayout;
	layout->setContentsMargins(0, 0, 0, 0);
	layout->addWidget(queryWidget);
	layout->addWidget(table);

	setLayout(layout);
	initEngine();
	connect(queryWidget, SIGNAL(showHelp(QString)), this,
		SLOT(showHelp(QString)));
	connect(queryWidget, SIGNAL(userInputUpdate(QString)), this,
			SLOT(updateQuery(QString)));
	connect(queryWidget, SIGNAL(filterSignal(QString)), this,
			SLOT(filterQuery(QString)));
	connect(queryWidget, SIGNAL(requestSuggestions(QString)), this,
			SLOT(requestSuggestions(QString)));

	queryEngine.setElements(
				gui::createRawviewTableRows(keyPoints1, keyPoints2,
											matches, usesTrainDescriptor));
	table->updateRowGroups(queryEngine.query("#group by keypoint_type"));
}

void Rawview::initEngine()
{
	queryEngine.addStringCmdFunc("keypoint_type",
	                             [](const gui::RawviewTableRow &row)
	{
		if (row.hasSingleKeyPoint())
		{
			return "single key point";
		}
		else
		{
			return "found match";
		}
	},
	                             false);
	queryEngine.addStringCmdFunc("img_number",
	                             [](const gui::RawviewTableRow &row)
	{
		if (row.hasSingleKeyPoint())
		{
			return row.isLeftSingleKeyPoint() ? "1" : "2";
		}
		else
		{
			return "match";
		}
	},
	                             false);
	queryEngine.addFloatCmdFunc("match_distance",
	                            [](const gui::RawviewTableRow &row)
	{
		if (row.hasSingleKeyPoint())
		{
			return (float)0;
		}
		return row.matchDistance();
	});
	queryEngine.addIntegerCmdFunc("img_idx",
	                              [](const gui::RawviewTableRow &row)
	{
		if (row.hasSingleKeyPoint())
		{
			return 0;
		}
		return row.matchImgIdx();
	});
	queryEngine.addIntegerCmdFunc("query_idx",
	                              [](const gui::RawviewTableRow &row)
	{
		if (row.hasSingleKeyPoint())
		{
			return 0;
		}
		return row.matchQueryIdx();
	});
	queryEngine.addIntegerCmdFunc("train_idx",
	                              [](const gui::RawviewTableRow &row)
	{
		if (row.hasSingleKeyPoint())
		{
			return 0;
		}
		return row.matchTrainIdx();
	});
	queryEngine.addFloatCmdFunc("x_1", [](const gui::RawviewTableRow &row)
	{
		if (row.hasSingleKeyPoint())
		{
			return row.isLeftSingleKeyPoint()
			           ? row.keyPoint1XCoord()
			           : (float)0;
		}
		return row.keyPoint1XCoord();
	});
	queryEngine.addFloatCmdFunc("y_1", [](const gui::RawviewTableRow &row)
	{
		if (row.hasSingleKeyPoint())
		{
			return row.isLeftSingleKeyPoint()
			           ? row.keyPoint1YCoord()
			           : (float)0;
		}
		return row.keyPoint1YCoord();
	});
	queryEngine.addFloatCmdFunc("size_1",
	                            [](const gui::RawviewTableRow &row)
	{
		if (row.hasSingleKeyPoint())
		{
			return row.isLeftSingleKeyPoint() ? row.keyPoint1Size()
			                                  : (float)0;
		}
		return row.keyPoint1Size();
	});
	queryEngine.addFloatCmdFunc("angle_1",
	                            [](const gui::RawviewTableRow &row)
	{
		if (row.hasSingleKeyPoint())
		{
			return row.isLeftSingleKeyPoint() ? row.keyPoint1Angle()
			                                  : (float)0;
		}
		return row.keyPoint1Angle();
	});
	queryEngine.addFloatCmdFunc("response_1",
	                            [](const gui::RawviewTableRow &row)
	{
		if (row.hasSingleKeyPoint())
		{
			return row.isLeftSingleKeyPoint()
			           ? row.keyPoint1Response()
			           : (float)0;
		}
		return row.keyPoint1Response();
	});
	queryEngine.addIntegerCmdFunc("octave_1",
	                              [](const gui::RawviewTableRow &row)
	{
		if (row.hasSingleKeyPoint())
		{
			return row.isLeftSingleKeyPoint()
			           ? row.keyPoint1Octave()
			           : 0;
		}
		return row.keyPoint1Octave();
	});
	queryEngine.addIntegerCmdFunc("class_id_1",
	                              [](const gui::RawviewTableRow &row)
	{
		if (row.hasSingleKeyPoint())
		{
			return row.isLeftSingleKeyPoint()
			           ? row.keyPoint1ClassId()
			           : 0;
		}
		return row.keyPoint1ClassId();
	});
	queryEngine.addFloatCmdFunc("x_2", [](const gui::RawviewTableRow &row)
	{
		if (row.hasSingleKeyPoint())
		{
			return row.isLeftSingleKeyPoint()
			           ? (float)0
			           : row.keyPoint1XCoord();
		}
		return row.keyPoint2XCoord();
	});
	queryEngine.addFloatCmdFunc("y_2", [](const gui::RawviewTableRow &row)
	{
		if (row.hasSingleKeyPoint())
		{
			return row.isLeftSingleKeyPoint()
			           ? (float)0
			           : row.keyPoint1YCoord();
		}
		return row.keyPoint2YCoord();
	});
	queryEngine.addFloatCmdFunc("size_2",
	                            [](const gui::RawviewTableRow &row)
	{
		if (row.hasSingleKeyPoint())
		{
			return row.isLeftSingleKeyPoint() ? (float)0
			                                  : row.keyPoint1Size();
		}
		return row.keyPoint2Size();
	});
	queryEngine.addFloatCmdFunc("angle_2",
	                            [](const gui::RawviewTableRow &row)
	{
		if (row.hasSingleKeyPoint())
		{
			return row.isLeftSingleKeyPoint()
			           ? (float)0
			           : row.keyPoint1Angle();
		}
		return row.keyPoint2Angle();
	});
	queryEngine.addFloatCmdFunc("response_2",
	                            [](const gui::RawviewTableRow &row)
	{
		if (row.hasSingleKeyPoint())
		{
			return row.isLeftSingleKeyPoint()
			           ? (float)0
			           : row.keyPoint1Response();
		}
		return row.keyPoint2Response();
	});
	queryEngine.addIntegerCmdFunc("octave_2",
	                              [](const gui::RawviewTableRow &row)
	{
		if (row.hasSingleKeyPoint())
		{
			return row.isLeftSingleKeyPoint()
			           ? 0
			           : row.keyPoint1Octave();
		}
		return row.keyPoint2Octave();
	});
	queryEngine.addIntegerCmdFunc("class_id_2",
	                              [](const gui::RawviewTableRow &row)
	{
		if (row.hasSingleKeyPoint())
		{
			return row.isLeftSingleKeyPoint()
			           ? 0
			           : row.keyPoint1ClassId();
		}
		return row.keyPoint2ClassId();
	});
}

void Rawview::filterQuery(QString query)
{
	table->updateRowGroups(
	    queryEngine.query(query + " #group by keypoint_type"));
}

void Rawview::updateQuery(QString query)
{
	requestSuggestions(query);
}

void Rawview::requestSuggestions(QString query)
{
	queryWidget->showSuggestions(queryEngine.getSuggestions(query));
}

void Rawview::showHelp(QString topic)
{
	qtutil::openHelpBrowser(topic);
}

bool Rawview::doesShowShowInViewMenu()
{
	return showShowInViewMenu;
}

void Rawview::selectMatches(const std::vector<cv::KeyPoint> &keyPoints1,
                            const std::vector<cv::KeyPoint> &keyPoints2,
                            const std::vector<cv::DMatch> &matches)
{
	this->keyPoints1 = keyPoints1;
	this->keyPoints2 = keyPoints2;
	this->matches = matches;
	queryEngine.setElements(
	    gui::createRawviewTableRows(keyPoints1, keyPoints2,
									matches, usesTrainDescriptor));
	table->updateRowGroups(queryEngine.reexecuteLastQuery());
}

void Rawview::selectKeyPoints(const std::vector<cv::KeyPoint> &keyPoints)
{
	queryEngine.setElements(
	    gui::createSingleKeyPointRawviewTableRows(keyPoints));
	table->updateRowGroups(queryEngine.reexecuteLastQuery());
	matches.clear();
}

std::vector<cv::DMatch> Rawview::getMatchSelection()
{
	return table->getMatchSelection();
}

std::vector<cv::KeyPoint> Rawview::getKeyPointSelection()
{
	return table->getKeyPointSelection();
}

void Rawview::setMatchSelection(std::vector<cv::DMatch> matches)
{
	table->setMatchSelection(matches);
}

void Rawview::setKeyPointSelection(std::vector<cv::KeyPoint> keyPoints)
{
	table->setKeyPointSelection(keyPoints);
}

void Rawview::matchesKeyPointsSelected(const std::vector<cv::DMatch> &matches)
{
	std::vector<cv::KeyPoint> selectedKeyPointsVec;
	for (auto match : matches)
	{
		selectedKeyPointsVec.push_back(keyPoints1.at(match.queryIdx));
		cv::KeyPoint keyPointRight;
		if (usesTrainDescriptor)
		{
			keyPointRight = keyPoints2.at(match.trainIdx);
		}
		else
		{
			keyPointRight = keyPoints2.at(match.imgIdx);
		}
		selectedKeyPointsVec.push_back(keyPointRight);
	}
	matchesSelected(matches);
	keyPointsSelected(selectedKeyPointsVec);
}

}
}
