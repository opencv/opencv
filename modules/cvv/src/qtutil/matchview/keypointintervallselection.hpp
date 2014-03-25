#ifndef CVVISUAL_KEY_POINT_INTERVALL_SELECTOR
#define CVVISUAL_KEY_POINT_INTERVALL_SELECTOR

#include "opencv2/features2d/features2d.hpp"

#include "keypointselection.hpp"
#include "keypointvaluechooser.hpp"
#include "../intervallselector.hpp"

namespace cvv
{
namespace qtutil
{
/**
 * @brief this widget select an intervall of matches from the given selection.
 * it use IntervallSelector
 */
class KeyPointIntervallSelector:public KeyPointSelection{

	Q_OBJECT
public:
	/**
	 * @brief the constructor
	 * @param matches all matches which can be selected
	 * @param parent the parent widget
	 */
	KeyPointIntervallSelector(std::vector<cv::KeyPoint> key,QWidget*parent=nullptr);

	/**
	 * @brief select matches from the given selecton
	 * @param selection the current selection
	 * @return the selected matches
	 */
	virtual std::vector<cv::KeyPoint> select(const std::vector<cv::KeyPoint>& selection)override;

private slots:

	void changeSelecteValue();
private:
	QLayout* layout_;
	IntervallSelector* selector_;
	KeyPointValueChooser * valueChooser_;
	std::vector<cv::KeyPoint> keypoints_;
};

}}
#endif
