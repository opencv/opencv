#ifndef CVVISUAL_MATCH_INTERVALL_SELECTOR
#define CVVISUAL_MATCH_INTERVALL_SELECTOR

#include "opencv2/features2d/features2d.hpp"

#include "matchselection.hpp"
#include "../intervallselector.hpp"

namespace cvv
{
namespace qtutil
{
/**
 * @brief this widget select an intervall of matches from the given selection.
 * it use IntervallSelector
 */
class MatchIntervallSelector:public MatchSelection{

	Q_OBJECT
public:
	/**
	 * @brief the constructor
	 * @param matches all matches which can be selected
	 * @param parent the parent widget
	 */
	MatchIntervallSelector(std::vector<cv::DMatch> matches,QWidget*parent=nullptr);

	/**
	 * @brief select matches from the given selecton
	 * @param selection the current selection
	 * @return the selected matches
	 */
	virtual std::vector<cv::DMatch> select(const std::vector<cv::DMatch>& selection)override;

private:
	IntervallSelector* selector_;

};

}}
#endif
