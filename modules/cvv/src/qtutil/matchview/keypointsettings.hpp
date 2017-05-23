#ifndef CVVISUAL_KEYPOINT_SETTINGS
#define CVVISUAL_KEYPOINT_SETTINGS

#include <QFrame>
#include <QPen>

#include "cvvkeypoint.hpp"

namespace cvv
{
namespace qtutil
{

class CVVKeyPoint;

/**
 * @brief this abstract class returns an individual Setting for a CVVKeyPoint.
 */
class KeyPointSettings : public QFrame
{
	Q_OBJECT

      public:
	/**
	 * @brief KeyPointPen
	 * @param parent the parent Widget
	 */
	KeyPointSettings(QWidget *parent) : QFrame(parent){}

	/**
	 * @brief set individual settings for a selected cvvkeypoint
	 */
	virtual void setSettings(CVVKeyPoint &key) = 0;

	/**
	 * @brief set individual settings for a non-selected cvvkeypoint
	 */
	/*virtual void setUnSelectedSettings(CVVKeyPoint &)
		{}*/

public slots:
	/**
	 * @brief this method emits the signal settingsChanged();
	 */
	void updateAll()
		{ emit settingsChanged(*this); }

signals:
	/**
	 * @brief this signal will be emitted if the settings changed
	 * and the CVVKeyPoint must update their Settings
	 */
	void settingsChanged(KeyPointSettings &);
};
}
}
#endif
