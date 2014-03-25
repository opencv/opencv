#ifndef CVVISUAL_MATCH_SELECTOR
#define CVVISUAL_MATCH_SELECTOR

#include <QFrame>

#include "opencv2/features2d/features2d.hpp"

namespace cvv{ namespace qtutil{

/**
 * @brief this class select matches from a given selection
 */
class MatchSelection:public QFrame{

	Q_OBJECT

public:
	/**
	 * @brief the constructor
	 */
	MatchSelection(QWidget * parent =nullptr):QFrame{parent}{}

	/**
	 * @brief select the matches of the given selection.
	 * @param selection a given selection
	 * @return a new selection
	 */
	virtual std::vector<cv::DMatch> select(const std::vector<cv::DMatch>& selection) = 0;

signals:
	/**
	 * @brief this signal will be emitted if settings changed.
	 */
	void settingsChanged();

};

}}
#endif
