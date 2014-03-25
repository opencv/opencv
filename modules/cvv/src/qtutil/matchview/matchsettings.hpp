#ifndef CVVISUAL_MATCH_SETTINGS
#define CVVISUAL_MATCH_SETTINGS

#include <QFrame>
#include <QPen>


namespace cvv
{
namespace qtutil
{

class CVVMatch;

/**
 * @brief this abstract class returns an individual Setting for a CVVMatch.
 */
class MatchSettings : public QFrame
{
	Q_OBJECT

      public:
	/**
	 * @brief MatchPen
	 * @param parent the parent Widget
	 */
	MatchSettings(QWidget *parent) : QFrame(parent){}

	/**
	 * @brief set individual settings for a selected cvvmatch
	 */
	virtual void setSettings(CVVMatch &match) = 0;

	/**
	 * @brief set individual settings for a non-selected cvvmatch
	 */
	/*virtual void setUnSelectedSettings(CVVMatch &)
		{}*/

public slots:
	/**
	 * @brief this method emits the signal settingsChanged();
	 */
	void updateAll()
		{emit settingsChanged(*this);}

signals:
	/**
	 * @brief this signal will be emitted if the settings changed
	 * and the CVVMatch must update their Settings
	 */
	void settingsChanged(MatchSettings &settings);
};
}
}
#endif
