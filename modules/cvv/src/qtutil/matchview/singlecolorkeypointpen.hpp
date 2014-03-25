#ifndef CVVISUAL_SINGLE_COLOR_KEY_PEN
#define CVVISUAL_SINGLE_COLOR_KEY_PEN

#include <QColorDialog>
#include <QPen>

#include "keypointsettings.hpp"

namespace cvv
{
namespace qtutil
{

/**
 * This KeyPointPen return for all CVVKeyPoints the same Color,
 * the Color can be choosen by an QColorDialog
 */

class SingleColorKeyPen : public KeyPointSettings
{
	Q_OBJECT
      public:
	/**
	 * @brief the consructor
	 * @param parent the parent Widget
	 */
	SingleColorKeyPen(std::vector<cv::KeyPoint> ,QWidget *parent = nullptr);

	/**
	 * @brief the destructor
	 */
	~SingleColorKeyPen()
	{ colordia_->deleteLater();}

	/**
	 * @brief this method returns the same PEn for all CVVKeyPoints
	 * @return the same Pen for all CVVKeyPoint
	 */
	virtual void setSettings(CVVKeyPoint &keypoint) override;
      public
slots:

	/**
	 * @brief this method updates the Color of the Pen which will be
	 * returned in getPen()
	 * @param color the new Color
	 */
	void updateColor(const QColor &color);

      private
slots:
	void colorButtonClicked()
		{colordia_->show();}

      private:
	QColorDialog *colordia_;
	QColor color_;
};
}
}
#endif
