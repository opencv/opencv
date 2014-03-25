#ifndef CVVISUAL_CVVMATCH
#define CVVISUAL_CVVMATCH

#include <QGraphicsObject>
#include <QPainter>
#include <QPointF>
#include <QRectF>
#include <QStyleOptionGraphicsItem>
#include <QWidget>


#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "matchsettings.hpp"
#include "cvvkeypoint.hpp"

namespace cvv
{
namespace qtutil
{

class MatchSettings;

/**
 * @brief this class represents a match which is displayed
 * a Matchscene.
 */
class CVVMatch : public QGraphicsObject,public cv::DMatch
{
	Q_OBJECT
      public:
	/**
	* @brief the constructor
	* @param left_key the left KeyPointPen
	* @param right_key the right KeyPointPen
	* @param match the match
	* @param pen a QPen
	* @param parent the parent Widget
	*/
	CVVMatch(CVVKeyPoint *left_key,
		 CVVKeyPoint *right_key,
		 const cv::DMatch &match,
		 const QPen &pen = QPen{ Qt::red },
		 QGraphicsItem *parent = nullptr);

	/**
	 * @brief returns the boundingrect of this Mathc
	 * @return the boundingrect of this Mathc
	 */
	virtual QRectF boundingRect() const;

	/**
	 * @brief the paint function
	 */
	virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *,
			   QWidget *);

	/**
	 * @brief returns the left keypoint.
	 * @return the left keypoint.
	 */
	cv::KeyPoint leftKeyPoint() const
	{
		return left_key_->keyPoint();
	}

	/**
	 * @brief returns the right keypoint.
	 * @return the right keypoint.
	 */
	cv::KeyPoint rightKeyPoint() const
	{
		return right_key_->keyPoint();
	}

	/**
	 * @brief maps the leftImagePoint to scene
	 * @return the scene point of the leftkeypoint
	 */
	QPointF leftImPointInScene() const
	{
		return left_key_->imPointInScene();
	}

	/**
	 * @brief maps the leftImagePoint to scene
	 * @return the scene point of the rightkeypoint
	 */
	QPointF rightImPointInScene() const
	{
		return right_key_->imPointInScene();
	}

	/**
	 * @brief returns the match value
	 * @return the match value
	 */
	const cv::DMatch match() const
	{
		return *this;
	}

	/**
	 * @brief returns the show value
	 * @return the show value
	 */
	bool isShown() const
	{
		return show_;
	}

	/**
	 * @brief operator ==
	 * @param o a cv::DMatch
	 * @return true if this has the same match
	 */
	bool operator==(const cv::DMatch &o);

	/**
	 * @brief get current pen
	 * @return current Pen
	 */
	QPen getPen() const
	{
		return pen_;
	}

      public
slots:

	/**
	 * @brief the match will call setSettings from settings
	 * @param settings the settings for this match
	 */
	void updateSettings(MatchSettings &settings)
	{
		settings.setSettings(*this);
	}

	/**
	 * @brief this method updates the Pen
	 * @param pen the new Pen
	 */
	void setPen(const QPen &pen);

	/**
	 * @brief if show=true the match will be visible if both keypoints are
	 * in the
	 * visibleArea of its images
	 * @param b new show value
	 */
	void setShow(const bool &b);

	/**
	 * @brief this slot will be called if the right keypoint has changed
	 * @param visible if the rightKey in the visibleArea of its image
	 */
	virtual void updateRightKey(bool visible);

	/**
	 * @brief this slot will be called if the left keypoint has changed
	 * @param visible if the leftKey in the visibleArea of its image
	 */
	virtual void updateLeftKey(bool visible);

      protected:
	CVVKeyPoint *left_key_;
	CVVKeyPoint *right_key_;
	//cv::DMatch match_;

	QPen pen_;
	bool show_;
	bool left_key_visible_;
	bool right_key_visible_;
};
}
}

#endif
