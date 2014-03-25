#include <QBrush>

#include "cvvpointmatch.hpp"

namespace cvv
{
namespace qtutil
{

CVVPointMatch::CVVPointMatch(CVVKeyPoint *left_key, CVVKeyPoint *right_key,
                             const cv::DMatch &match, bool isLeftKey,
                             qreal radius, const QPen &pen, const QBrush &brush,
                             QGraphicsItem *parent)
    : CVVMatch{ left_key, right_key, match, pen, parent },
      isLeftKey_{ isLeftKey },
      radius_{ std::min(radius * match.distance, 10.0) }, brush_{ brush }
{
	if (isLeftKey_)
	{
		right_key_visible_ = true;
		setVisible(left_key_visible_);
	}
	else
	{
		left_key_visible_ = true;
		setVisible(right_key_visible_);
	}
}

QRectF CVVPointMatch::boundingRect() const
{
	QPointF point =
	    (isLeftKey_ ? leftImPointInScene() : rightImPointInScene());
	return QRectF{ QPointF{ point.x() - radius_, point.y() - radius_ },
		       QPointF{ point.x() + radius_, point.y() + radius_ } };
}

void CVVPointMatch::paint(QPainter *painter, const QStyleOptionGraphicsItem *,
                          QWidget *)
{
	painter->setPen(pen_);
	painter->setBrush(brush_);
	painter->drawEllipse(boundingRect());
}

void CVVPointMatch::updateRightKey(bool visible)
{
	if (!isLeftKey_)
	{
		CVVMatch::updateRightKey(visible);
	}
}

void CVVPointMatch::updateLeftKey(bool visible)
{
	if (isLeftKey_)
	{
		CVVMatch::updateLeftKey(visible);
	}
}
}
}
