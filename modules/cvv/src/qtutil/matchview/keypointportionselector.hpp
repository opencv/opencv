#ifndef CVVISUAL_KEY_POINT_PORTION_SELECTOR
#define CVVISUAL_KEY_POINT_PORTION_SELECTOR

#include <vector>

#include "opencv2/features2d/features2d.hpp"

#include "keypointselection.hpp"
#include "keypointvaluechooser.hpp"
#include "../portionselector.hpp"

namespace cvv {namespace qtutil{

class KeyPointPortionSelection:public KeyPointSelection{
public:
	KeyPointPortionSelection(std::vector<cv::KeyPoint>, QWidget * parent=nullptr);


	virtual std::vector<cv::KeyPoint> select(const std::vector<cv::KeyPoint>& selection)override;

private:
	PortionSelector* selector_;
	KeyPointValueChooser * valueChooser_;
};

}}

#endif
