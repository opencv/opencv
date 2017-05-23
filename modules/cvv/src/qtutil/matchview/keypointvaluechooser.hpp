#ifndef CVVISUAL_KEY_POINT_VALUE_CHOOSER
#define CVVISUAL_KEY_POINT_VALUE_CHOOSER

#include <QWidget>
#include <QComboBox>

#include "opencv2/features2d/features2d.hpp"


namespace cvv{ namespace qtutil{

/**
 * @brief this widget contains a combobox with the attributes of an keypoint as entry.
 * you cann call the method getChoosenValue which return the choosen value of the given keypoint
 */
class KeyPointValueChooser:public QWidget{

	Q_OBJECT

public:
	/**
	 * @brief the constructor
	 * @param parent the parent Widget
	 */
	KeyPointValueChooser(QWidget *parent=nullptr);

	/**
	 * @brief returns the choosen value of the given keypoint
	 * @return the choosen value of the given keypoint
	 */
	double getChoosenValue(cv::KeyPoint keypoint);

signals:

	/**
	 * @brief this signal will be emitted if the user selected an other value
	 */
	void valueChanged();

private:
	QComboBox *combBox_;

};

}}

#endif
