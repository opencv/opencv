#ifndef CVVISUAL_RAWVIEW_HPP
#define CVVISUAL_RAWVIEW_HPP

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <QWidget>
#include <QString>
#include <QSlider>
#include <QLabel>
#include <QPushButton>

#include "../stfl/stfl_engine.hpp"
#include "../impl/call.hpp"
#include "../gui/rawview_table.hpp"
#include "../gui/rawview_table_row.hpp"
#include "../util/util.hpp"
#include "match_view.hpp"

namespace cvv
{

namespace controller
{
class ViewController;
}

namespace qtutil
{
class STFLQueryWidget;
}

namespace gui
{
class RawviewTable;
class RawviewTableRow;
}

namespace view
{

/**
 * @brief A view displaying the raw data of a match call (aka the matches).
 */
class Rawview : public MatchView
{

	Q_OBJECT

public:

	/**
	 * @brief Constructor of this class.
	 *
	 * @param keypoints1 left keypoints
	 * @param keypoints2 right keypoints
	 * @param matches matches between the left and the right keypoints.
	 * @param usesTrainDescriptor Use the trainIdx property of each match to get the
	 * "right" key points?
	 * @param showShowInViewMenu Does this view show a "Show in view" item
	 * in its context menus?
	 */
	Rawview(const std::vector<cv::KeyPoint> &keyPoints1,
	const std::vector<cv::KeyPoint> &keyPoints2,
	const std::vector<cv::DMatch> &matches,
	bool usesTrainDescriptor = true,
	bool showShowInViewMenu = false);

	/**
	 * @brief Constructor.
	 * @param call from which the data for the view is taken.
	 * @param parent of this QWidget.
	 */
	Rawview(const impl::MatchCall &call, QWidget *parent = nullptr)
	    : Rawview(call.keyPoints1(), call.keyPoints2(), 
				  call.matches(), call.usesTrainDescriptor())
	{
		setParent(parent);
	}

	/**
	 * @brief Checks whether this view shows a "Show in view" item in its
	 * context menus.
	 * @return Does this view show a "Show in view" item in its context
	 * menus?
	 */
	bool doesShowShowInViewMenu();
	
	virtual std::vector<cv::DMatch> getMatchSelection();

	virtual std::vector<cv::KeyPoint> getKeyPointSelection();
	
signals:
	/**
	 * @brief Requests to update the left footer of the window that displays
	 * this view.
	 * @param newText new text of the left footer
	 */
	void updateLeftFooter(QString newText);

	/**
	 * @brief Requests to update the right footer of the window that
	 * displays this view.
	 * @param newText new text of the right footer
	 */
	void updateRightFooter(QString newText);

	/**
	 * @brief The user selected some matches.
	 * @param matches seleted matches
	 */
	void matchesSelected(const std::vector<cv::DMatch> &matches);

	/**
	 * @brief The user selected some single key points.
	 * Single key points are key points without correspoinding match.
	 * @param keyPoints seleted single key points
	 */
	void keyPointsSelected(const std::vector<cv::KeyPoint> &keyPoints);
	
public slots:

	/**
	 * @brief Show (and filter) only the given matches.
	 * @param keypoints1 (left) key points the match is refererring two
	 * @param keypoints2 (right) key points the match is refererring two
	 * @param matches matches to show
	 */
	void selectMatches(const std::vector<cv::KeyPoint> &keyPoints1,
	                   const std::vector<cv::KeyPoint> &keyPoints2,
	                   const std::vector<cv::DMatch> &matches);

	/**
	 * @brief Show (and filter) only the given single key points.
	 * @param keyPoints given key points
	 */
	void selectKeyPoints(const std::vector<cv::KeyPoint> &keyPoints);

	
	virtual void setMatchSelection(std::vector<cv::DMatch> matches);

	virtual void setKeyPointSelection(std::vector<cv::KeyPoint> keyPoints);
	
	/**
	 * @brief Issues the matchesSelected and the keyPointsSelected signal.
	 * It uses the referenced key points (via the given matches) to find the
	 * selected key points, that will be passed to the keyPointsSelected signals.
	 * @param matches the user selected matches.
	 */
	void matchesKeyPointsSelected(const std::vector<cv::DMatch> &matches);
	
private slots:

	void filterQuery(QString query);

	void updateQuery(QString query);

	void requestSuggestions(QString query);

	void showHelp(QString topic);

private:
	std::vector<cv::KeyPoint> keyPoints1;
	std::vector<cv::KeyPoint> keyPoints2;
	std::vector<cv::DMatch> matches;
	stfl::STFLEngine<gui::RawviewTableRow> queryEngine{"Rawview"};
	qtutil::STFLQueryWidget *queryWidget;
	gui::RawviewTable *table;
	bool showShowInViewMenu = false;
	bool usesTrainDescriptor = true;
	
	void initEngine();
};
}
}

#endif
