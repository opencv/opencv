#ifndef CVVISUAL_FALSE_COLOR_KEY_POINT_PEN
#define CVVISUAL_FALSE_COLOR_KEY_POINT_PEN

#include <vector>

#include "opencv2/features2d/features2d.hpp"

#include "keypointvaluechooser.hpp"
#include "keypointsettings.hpp"


namespace cvv
{
namespace qtutil
{
/**
 * @brief this pen gives the falsecolor of the distance value to the key point
 */
class FalseColorKeyPointPen : public KeyPointSettings
{

	Q_OBJECT
public:
	/**
	 * @brief the constructor
	 * @param univers all keypoints (for max value)
	 * @param parent the parent Widget
	 */
	FalseColorKeyPointPen(std::vector<cv::KeyPoint> univers, QWidget *parent = nullptr);

	/**
	 * @brief set the falseColor to the given keypoint
	 */
	virtual void setSettings(CVVKeyPoint &key) override;

private slots:

	void updateMinMax();

private:
	KeyPointValueChooser* valueChooser_;
	std::vector<cv::KeyPoint> univers_;
	double maxDistance_;
	double minDistance_=0.0;//always 0
};
}
}
#endif
