#ifndef CVVISUAL_CVV_POINT_MATCH
#define CVVISUAL_CVV_POINT_MATCH

#include <QBrush>

#include "cvvmatch.hpp"

namespace cvv
{
namespace qtutil
{

/**
 * @brief This CVVMatch will be shown as circles with a given Color and radius
 * this CVVMatches will be used in DepthView
 */
class CVVPointMatch : public CVVMatch
{
	Q_OBJECT
      public:
	/**
	* @brief the constructor
	* @param left_key the left KeyPointPen
	* @param right_key the right KeyPointPen
	* @param matchValue the match distance
	* @param isLeftKey if true the match is at Pos of the left key,
	* otherwise it is at the
	* pos of the right key
	* @param radius the radius of the MatchPoint
	* @param pen the pen
	* @param brush the brush
	* @param parent the parent Widget
	*/
	CVVPointMatch(CVVKeyPoint *left_key, CVVKeyPoint *right_key,
	              const cv::DMatch &match, bool isLeftKey = true,
	              qreal radius = 1, const QPen &pen = QPen{ Qt::red },
	              const QBrush &brush = QBrush{ Qt::red },
	              QGraphicsItem *parent = nullptr);

	/**
	 * @brief returns the boundingrect of this Mathc
	 * @return the boundingrect of this Mathc
	 */
	virtual QRectF boundingRect() const override;

	/**
	 * @brief the paint function
	 */
	virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *,
	                   QWidget *) override;

      public
slots:

	/**
	 * @brief this slot will be called if the right keypoint has changed
	 * @param visible if the rightKey in the visibleArea of its image
	 */
	virtual void updateRightKey(bool visible) override;

	/**
	 * @brief this slot will be called if the left keypoint has changed
	 * @param visible if the leftKey in the visibleArea of its image
	 */
	virtual void updateLeftKey(bool visible) override;

      protected:
	bool isLeftKey_;
	qreal radius_;
	QBrush brush_;
};
}
}
#endif
