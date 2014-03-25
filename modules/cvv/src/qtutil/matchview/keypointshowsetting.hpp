#ifndef CVVISUAL_KEY_POINT_SHOW_SETTING
#define CVVISUAL_KEY_POINT_SHOW_SETTING

#include <vector>

#include <QPushButton>

#include "opencv2/features2d/features2d.hpp"
#include "keypointsettings.hpp"

namespace cvv{ namespace qtutil{

/**
 * @brief this class is a KeyPointSetting which hides a KeyPoint or not
 */
class KeyPointShowSetting:public KeyPointSettings{

	Q_OBJECT

public:
	/**
	 * @brief the constructor
	 * std::vector<cv::KeyPoint> this argument is for the KeyPointSettingSelector and will be ignored.
	 * @param parent
	 */
	KeyPointShowSetting(std::vector<cv::KeyPoint>,QWidget* parent=nullptr);

	/**
	 * @brief set the Settings of the given keyPoint
	 * @param key a CVVKeyPoint
	 */
	virtual void setSettings(CVVKeyPoint &key) override
		{key.setShow(button_->isChecked());}

	/*virtual void setUnSelectedSettings(CVVKeyPoint &key) override
		{key.setShow(!(button_->isChecked()));}*/
public slots:

	void updateButton();
private:
	QPushButton *button_;
};

}}

#endif
