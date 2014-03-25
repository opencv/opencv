#ifndef CVVISUAL_MATCH_SHOW_SETTING
#define CVVISUAL_MATCH_SHOW_SETTING

#include <vector>

#include <QPushButton>

#include "opencv2/features2d/features2d.hpp"
#include "matchsettings.hpp"
#include "cvvmatch.hpp"

namespace cvv{ namespace qtutil{
/**
 * @brief this class is a MatchSetting which hides a Match or not
 */
class MatchShowSetting:public MatchSettings{

	Q_OBJECT

public:
	/**
	 * @brief the constructor
	 * std::vector<cv::DMatch> this argument is for the MatchSettingSelector and will be ignored.
	 * @param parent
	 */
	MatchShowSetting(std::vector<cv::DMatch>,QWidget* parent=nullptr);

	/**
	 * @brief set the Settings of the given match
	 * @param match a cvvmatch
	 */
	virtual void setSettings(CVVMatch &match) override
		{match.setShow(button_->isChecked());}

	/*virtual void setUnSelectedSettings(CVVMatch &match) override
		{match.setShow(!(button_->isChecked()));}*/
public slots:

	void updateButton();
private:
	QPushButton *button_;
};

}}

#endif
