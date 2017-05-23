#ifndef CVVISUAL_KEYPOINT_MANAGEMENT
#define CVVISUAL_KEYPOINT_MANAGEMENT

#include <QCheckBox>

#include "../../util/util.hpp"

#include "opencv2/features2d/features2d.hpp"

#include "keypointselectionselector.hpp"
#include "keypointsettingsselector.hpp"
#include "keypointsettings.hpp"
#include "cvvkeypoint.hpp"

namespace cvv
{
namespace qtutil
{

/**
 * @brief the keypointmanagement class coordinates the selections and use settings for the selection.
 */
class KeyPointManagement : public KeyPointSettings
{

	Q_OBJECT

public:
	/**
	 * @brief the constructor
	 * @param univers all keypoints which can be selected
	 * @param parent the parent widget
	 */
	KeyPointManagement(std::vector<cv::KeyPoint> univers,QWidget *parent = nullptr);

	/**
	 * @brief set the settings if this KeyPoint is selected
	 */
	virtual void setSettings(CVVKeyPoint &match);

	/**
	 * @brief add the given KeyPointSettingsSelector to the list
	 */
	void addSetting(std::unique_ptr<KeyPointSettingsSelector>);

	/**
	 * @brief add the given KeyPointSelectionSelector to the list
	 */
	void addSelection(std::unique_ptr<KeyPointSelectionSelector>);

	/**
	 * @brief returns the current selection.
	 */
	std::vector<cv::KeyPoint> getCurrentSelection()
		{return selection_;}

public slots:

	//selection
	/**
	 * @brief add the given keypoint to the current selection.
	 */
	void addToSelection(const cv::KeyPoint &key);

	/**
	 * @brief set the selection to the given single match
	 */
	void singleSelection(const cv::KeyPoint &key);

	/**
	 * @brief set the current selection to the given selection
	 */
	void setSelection(const std::vector<cv::KeyPoint> &selection);

	//KeyPointSettingSelector
	/**
	 * @brief add a new Setting
	 */
	void addSetting();

	void removeSetting(KeyPointSettingsSelector *setting);

	//Match Selection
	/**
	 * @brief add a KeyPointSelectionSelector to the list
	 */
	void addSelection();

	/**
	 * @brief remove a given KeyPointSelector from the list
	 */
	void removeSelection(KeyPointSelectionSelector *selector);

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
		{setSelection(std::vector<cv::KeyPoint>{});}


signals:

	/**
	 * @brief this signal will be emited when the selection was changed.
	 * it can be used for syncronisation with other selector
	 */
	void updateSelection(const std::vector<cv::KeyPoint> &selection);

	/**
	 * @brief this singal has the same function like settingsChanged from KeyPointSettings,
	 * but this will be only connect to the current selection
	 */
	void applySettingsToSelection(KeyPointSettings&);

private:
	std::vector<cv::KeyPoint> univers_;
	std::vector<cv::KeyPoint> selection_;
	std::vector<KeyPointSettingsSelector*> settingsList_;
	std::vector<KeyPointSelectionSelector*> selectorList_;

	QLayout *settingsLayout_;
	QLayout *selectorLayout_;
	QCheckBox *showOnlySelection_;
};
}}

#endif
