#ifndef CVVISUAL_FALSE_COLOR_MATCH_PEN
#define CVVISUAL_FALSE_COLOR_MATCH_PEN

#include <vector>

#include "opencv2/features2d/features2d.hpp"

#include "matchsettings.hpp"


namespace cvv
{
namespace qtutil
{
/**
 * @brief this pen gives the falsecolor of the distance value to the match
 */
class FalseColorMatchPen : public MatchSettings
{
public:
	/**
	 * @brief the constructor
	 * @param univers all matches (for max value)
	 * @param parent the parent Widget
	 */
	FalseColorMatchPen(std::vector<cv::DMatch> univers, QWidget *parent = nullptr);

	/**
	 * @brief set the falseColor of the distance to the given match
	 */
	virtual void setSettings(CVVMatch &match) override;

private:
	double maxDistance_;
	double minDistance_=0.0;//always 0
};
}
}
#endif
