#ifndef CVVISUAL_MATCH_SETTINGS_SELECTOR
#define CVVISUAL_MATCH_SETTINGS_SELECTOR

#include <vector>

#include "opencv2/features2d/features2d.hpp"

#include "matchsettings.hpp"
#include "../registerhelper.hpp"

namespace cvv{ namespace qtutil{

/**
 * @brief this class can use diffrent MatchSettings
 * you can register functios which take a std::vector<cv::DMatch> as argument.
 */
class MatchSettingsSelector:public MatchSettings, public RegisterHelper<MatchSettings,std::vector<cv::DMatch>>{

	Q_OBJECT

public:
	/**
	 * @brief the constructor
	 */
	MatchSettingsSelector(const std::vector<cv::DMatch>& univers,QWidget * parent=nullptr);

	/**
	 * @brief set settings o the given match
	 *
	 */
	virtual void setSettings(CVVMatch &match)override;

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
	void remove(MatchSettingsSelector*);
private slots:

	/**
	 * @brief swap the current MatchSetting if the user choose another.
	 */
	virtual void changedSetting();

private:
	MatchSettings * setting_=nullptr;
	std::vector<cv::DMatch> univers_;
	QLayout * layout_;

};

template <class Setting>
bool registerMatchSettings(const QString &name)
{
	return MatchSettingsSelector::registerElement(
	    name, [](std::vector<cv::DMatch> univers)
	{
		    return std::unique_ptr<MatchSettings>{ new Setting{univers}};
	});
}

}}

#endif
