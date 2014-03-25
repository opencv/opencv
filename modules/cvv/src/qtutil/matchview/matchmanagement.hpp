#ifndef CVVISUAL_MATCH_MANAGEMENT
#define CVVISUAL_MATCH_MANAGEMENT

#include <QCheckBox>

#include "opencv2/features2d/features2d.hpp"

#include "matchselectionselector.hpp"
#include "matchsettingsselector.hpp"
#include "matchsettings.hpp"
#include "cvvmatch.hpp"
#include "../../util/util.hpp"

namespace cvv
{
namespace qtutil
{

/**
 * @brief the matchmanagement class coordinates the selections and use settings for the selection.
 */
class MatchManagement : public MatchSettings
{

	Q_OBJECT

public:
	/**
	 * @brief the constructor
	 * @param univers all matches which can be selected
	 * @param parent the parent widget
	 */
	MatchManagement(std::vector<cv::DMatch> univers,QWidget *parent = nullptr);

	/**
	 * @brief set the settings if this match is selected
	 */
	virtual void setSettings(CVVMatch &match);

	/**
	 * @brief add the given MatchSettingsSelector to the list
	 */
	void addSetting(std::unique_ptr<MatchSettingsSelector>);

	/**
	 * @brief add the given MatchSelectionSelector to the list
	 */
	void addSelection(std::unique_ptr<MatchSelectionSelector>);

	/**
	 * @brief returns the current selection.
	 */
	std::vector<cv::DMatch> getCurrentSelection()
		{return selection_;}

public slots:

	//selection
	/**
	 * @brief add the given match to the current selection.
	 */
	void addToSelection(const cv::DMatch &match);

	/**
	 * @brief set the selection to the given single match
	 */
	void singleSelection(const cv::DMatch &match);

	/**
	 * @brief set the current selection to the given selection
	 */
	void setSelection(const std::vector<cv::DMatch> &selection);

	//MatchSettingSelector
	/**
	 * @brief add a new Setting
	 */
	void addSetting();

	void removeSetting(MatchSettingsSelector *setting);

	//Match Selection
	/**
	 * @brief add the given MatchSelectionSelector to the list
	 */
	void addSelection();

	/**
	 * @brief remove a given MatchSelector from the list
	 */
	void removeSelection(MatchSelectionSelector *selector);

	/**
	 * @brief select with the selections
	 */
	void applySelection();

	/**
	 * @brief set Selection to univers.
	 */
	void selectAll()
		{setSelection(univers_);}

	/**
	 * @brief set selection to an empty list.
	 */
	void selectNone()
		{setSelection(std::vector<cv::DMatch>{});}

signals:

	/**
	 * @brief this signal will be emited when the selection was changed.
	 * it can be used for syncronisation with other selector
	 */
	void updateSelection(const std::vector<cv::DMatch> &selection);

	/**
	 * @brief this singal has the same function like settingsChanged from MatchSettings,
	 * but this will be only connect to the current selection
	 */
	void applySettingsToSelection(MatchSettings&);

private:
	std::vector<cv::DMatch> univers_;
	std::vector<cv::DMatch> selection_;
	std::vector<MatchSettingsSelector*> settingsList_;
	std::vector<MatchSelectionSelector*> selectorList_;

	QLayout *settingsLayout_;
	QLayout *selectorLayout_;
	QCheckBox *showOnlySelection_;
};
}
}
#endif
