#include "cvvkeypoint.hpp"

namespace cvv
{
namespace qtutil
{

CVVKeyPoint::CVVKeyPoint(const cv::KeyPoint &key, qtutil::ZoomableImage *image,
			 QPen pen, QBrush brush, QGraphicsItem *parent)
    : QGraphicsObject{ parent }, cv::KeyPoint{ key }, image_{ image }, pen_{ pen },
      brush_{ brush }, show_{ true }
{
	//setFlag(QGraphicsItem::ItemIsSelectable, true);
	//setSelected(true);
	setToolTip(QString
	{ "KeyPoint size: %1 \n angle %2 \n response %3 " }
		       .arg(size)
		       .arg(angle)
		       .arg(response));
	if (image != nullptr)
	{
		updateImageSet(image->visibleArea(), image->zoom());
		connect(image, SIGNAL(updateArea(QRectF, qreal)), this,
			SLOT(updateImageSet(QRectF, qreal)));
	}
}

void CVVKeyPoint::paint(QPainter *painter, const QStyleOptionGraphicsItem *,
			QWidget *)
{
	painter->setPen(pen_);
	painter->setBrush(brush_);
	painter->drawEllipse(boundingRect());
}

void CVVKeyPoint::setZoomableImage(ZoomableImage *image)
{
	image_ = image;
	updateImageSet(image->visibleArea(), image->zoom());
	connect(image, SIGNAL(updateArea(QRectF, qreal)), this,
		SLOT(updateImageSet(const QRectF &, const qreal &)));
}

bool CVVKeyPoint::operator==(const cv::KeyPoint &o)
{
	return o.pt == pt && o.size == size &&
	       o.angle == angle && o.response == response &&
	       o.octave == octave && o.class_id == class_id;
}

void CVVKeyPoint::updateSettings(KeyPointSettings &settings)
{
	settings.setSettings(*this);
}



void CVVKeyPoint::setPen(const QPen &pen)
{
	pen_ = pen;
	update();
}

void CVVKeyPoint::setBrush(const QBrush &brush)
{
	brush_ = brush;
	update();
}

void CVVKeyPoint::setShow(bool b)
{
	show_ = b;
	if(image_){
		setVisible(show_&imagePointisVisible());
	}
}

QRectF CVVKeyPoint::boundingRect() const
{
	// TODO throw image==nullptr
	return QRectF{
		QPointF{ imPointInScene().x() - 3, imPointInScene().y() - 3 },
		QPointF{ imPointInScene().x() + 3, imPointInScene().y() + 3 }
	};
}

void CVVKeyPoint::updateImageSet(const QRectF &, const qreal &zoom)
{
	imagePointInScene_=image_->mapImagePointToParent(
				QPointF{ pt.x, pt.y });

	bool isInVisibleArea=imagePointisVisible();
	setVisible(show_ && isInVisibleArea);
	emit updatePoint(isInVisibleArea);
	zoom_ = zoom;
	prepareGeometryChange();
	// update();
}
}
}
