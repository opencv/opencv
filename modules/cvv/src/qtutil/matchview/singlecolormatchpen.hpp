#ifndef CVVISUAL_SINGLE_COLOR_MATCHPEN
#define CVVISUAL_SINGLE_COLOR_MATCHPEN

#include <QPen>
#include <QColor>
#include <QWidget>
#include <QColorDialog>

#include "matchsettings.hpp"
#include "cvvmatch.hpp"

namespace cvv
{
namespace qtutil
{

/**
 * This MatchPen return for all CVVMatches the same Color,
 * the Color can be choosen by an QColorDialog
 */

class SingleColorMatchPen : public MatchSettings
{
	Q_OBJECT
      public:
	/**
	 * @brief the constructor
	 * @param parent the parent Widget
	 */
	SingleColorMatchPen(std::vector<cv::DMatch> ,QWidget *parent = nullptr);

	/**
	 * @brief the destructor
	 * the QColorDialog has no parent/layout it must be deleted.
	 */
	~SingleColorMatchPen()
		{colorDialog_->deleteLater();}

	/**
	 * @brief return a single Color for all CVVMatch
	 */
	virtual void setSettings(CVVMatch &match) override;

      public
slots:

	/**
	 * @brief updates the Color which will be returned in getPen(CVVMAtch&).
	 * @param color a QColor
	 */
	void updateColor(const QColor &color);

      protected
slots:
	void colorButtonClicked()
		{colorDialog_->show();}

      protected:
	QColor color_;
	QColorDialog *colorDialog_;
};
}
}
#endif
