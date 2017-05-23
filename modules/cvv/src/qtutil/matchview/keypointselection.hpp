#ifndef CVVISUAL_KEY_POINT_SELECTOR
#define CVVISUAL_KEY_POINT_SELECTOR

#include <QFrame>

#include "opencv2/features2d/features2d.hpp"

namespace cvv{ namespace qtutil{

/**
 * @brief this class select keypoints from a given selection
 */
class KeyPointSelection:public QFrame{

	Q_OBJECT

public:
	/**
	 * @brief the constructor
	 */
	KeyPointSelection(QWidget * parent =nullptr):QFrame{parent}{}


	virtual std::vector<cv::KeyPoint> select(const std::vector<cv::KeyPoint>& selection) = 0;

signals:

	void settingsChanged();

};

}}
#endif
