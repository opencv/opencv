#ifndef CVVISUAL_KEY_POINT_SETTINGS_SELECTOR
#define CVVISUAL_KEY_POINT_SETTINGS_SELECTOR

#include <vector>

#include "opencv2/features2d/features2d.hpp"

#include "keypointsettings.hpp"
#include "../registerhelper.hpp"

namespace cvv{ namespace qtutil{

/**
 * @brief this class can use diffrent KeyPointSettings
 * you can register functios which take a std::vector<cv::DMatch> as argument.
 */
class KeyPointSettingsSelector:public KeyPointSettings, public RegisterHelper<KeyPointSettings,std::vector<cv::KeyPoint>>{

	Q_OBJECT

public:
	/**
	 * @brief the constructor
	 */
	KeyPointSettingsSelector(const std::vector<cv::KeyPoint>& univers,QWidget * parent=nullptr);

	/**
	 * @brief set settings o the given keypoint
	 *
	 */
	virtual void setSettings(CVVKeyPoint &key)override;

public slots:

	/**
	 * @brief emits the remove signal this
	 */
	void removeMe()
		{emit remove(this);}

signals:
	/**
	 * @brief this signal will be emit if this selector should be removed
	 */
	void remove(KeyPointSettingsSelector*);
private slots:

	/**
	 * @brief swap the current KeyPointSetting if the user choose another.
	 */
	virtual void changedSetting();

private:
	KeyPointSettings * setting_=nullptr;
	std::vector<cv::KeyPoint> univers_;
	QLayout * layout_;

};


template <class Setting>
bool registerKeyPointSetting(const QString &name)
{
	return KeyPointSettingsSelector::registerElement(
	    name, [](std::vector<cv::KeyPoint> univers)
	{
		    return std::unique_ptr<KeyPointSettings>{ new Setting{univers}};
	});
}

}}

#endif
