
#include <algorithm>

#include "cvvmatch.hpp"

namespace cvv
{
namespace qtutil
{

CVVMatch::CVVMatch(CVVKeyPoint *left_key, CVVKeyPoint *right_key,
		   const cv::DMatch &match, const QPen &pen,
		   QGraphicsItem *parent)
    : QGraphicsObject{ parent },cv::DMatch{ match }, left_key_{ left_key }, right_key_{ right_key },
       pen_{ pen }, show_{ true },
      left_key_visible_{ left_key->imagePointisVisible() },
      right_key_visible_{ right_key_->imagePointisVisible() }
{
	//setFlag(QGraphicsItem::ItemIsSelectable);

	setVisible(show_ && left_key_visible_ && right_key_visible_);
	//setSelected(true);

	setToolTip(QString
	{ "Match distance: %1 \n queryIdx %2 \n trainIdx %3 \n imIdx %4 " }
		       .arg(distance)
		       .arg(queryIdx)
		       .arg(trainIdx)
		       .arg(imgIdx));

	connect(left_key_, SIGNAL(updatePoint(bool)), this,
		SLOT(updateLeftKey(bool)));
	connect(right_key_, SIGNAL(updatePoint(bool)), this,
		SLOT(updateRightKey(bool)));
}

QRectF CVVMatch::boundingRect() const
{
	// TODO minmax
	return QRectF{ QPointF{ std::min(leftImPointInScene().rx(),
					 rightImPointInScene().rx()),
				std::min(leftImPointInScene().ry(),
					 rightImPointInScene().ry()) },
		       QPointF{ std::max(leftImPointInScene().rx(),
					 rightImPointInScene().rx()),
				std::max(leftImPointInScene().ry(),
					 rightImPointInScene().ry()) } };
}

void CVVMatch::paint(QPainter *painter, const QStyleOptionGraphicsItem *,
		     QWidget *)
{
	painter->setPen(pen_);
	painter->drawLine(leftImPointInScene(), rightImPointInScene());
}

bool CVVMatch::operator==(const cv::DMatch &o)
{
	return o.queryIdx == queryIdx && o.trainIdx == trainIdx &&
	       o.imgIdx == imgIdx;
}

void CVVMatch::setPen(const QPen &pen)
{
	pen_ = pen;
	update();
}

void CVVMatch::setShow(const bool &b)
{
	show_ = b;
	setVisible(show_ && left_key_visible_ && right_key_visible_);
}

void CVVMatch::updateLeftKey(bool visible)
{
	left_key_visible_ = visible;
	setVisible(show_ && left_key_visible_ && right_key_visible_);
	prepareGeometryChange();
	// update();
}

void CVVMatch::updateRightKey(bool visible)
{
	right_key_visible_ = visible;
	setVisible(show_ && left_key_visible_ && right_key_visible_);
	prepareGeometryChange();
	// update();
}
}
}
