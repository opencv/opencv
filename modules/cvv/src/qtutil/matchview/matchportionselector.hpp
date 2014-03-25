#ifndef CVVISUAL_MATCH_PORTION_SELECTOR
#define CVVISUAL_MATCH_PORTION_SELECTOR

#include <vector>

#include "opencv2/features2d/features2d.hpp"

#include "matchselection.hpp"
#include "../portionselector.hpp"

namespace cvv {namespace qtutil{
/**
 * @brief this class use the PortionSelector for Matches
 */
class MatchPortionSelection:public MatchSelection{
public:
	/**
	 * @brief the constructor
	 * @param parent the parent widget
	 */
	MatchPortionSelection(std::vector<cv::DMatch>, QWidget * parent=nullptr);

	/**
	 * @brief see MatchSelection
	 */
	virtual std::vector<cv::DMatch> select(const std::vector<cv::DMatch>& selection)override;

private:
	PortionSelector* selector_;
};

}}

#endif
