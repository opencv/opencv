#ifndef CVVISUAL_CVVKEYPOINT
#define CVVISUAL_CVVKEYPOINT

#include <QGraphicsObject>
#include <QPainter>
#include <QPointF>
#include <QRectF>
#include <QStyleOptionGraphicsItem>
#include <QWidget>
#include <QGraphicsScene>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "keypointsettings.hpp"
#include "../zoomableimage.hpp"

namespace cvv
{
namespace qtutil
{

class KeyPointSettings;

/**
 * @brief this class represents a Keypoint which is displayed
 *  a Matchscene.
 **/
class CVVKeyPoint : public QGraphicsObject,public cv::KeyPoint
{
	Q_OBJECT
      public:
	/**
	 * @brief the construor
	 * @param key the keypoint with the image point
	 * @param image the zoomable image
	 */
	CVVKeyPoint(const cv::KeyPoint &key,
		    qtutil::ZoomableImage *image = nullptr,
		    QPen pen = QPen{ Qt::red },
		    QBrush brush = QBrush{ Qt::red },
		    QGraphicsItem *parent = nullptr);

	/**
	 * @brief this method maps the imagepoint to the scene
	 * @return maps the imagepoint to the scene
	 */
	QPointF imPointInScene() const
		{return imagePointInScene_;}

	/**
	 * @brief boundingRect
	 * @return the boundingRect
	 */
	QRectF boundingRect() const;

	/**
	 * @brief returns the keypoint
	 * @return the keypoint
	 */
	cv::KeyPoint keyPoint() const
		{return *this;}

	/**
	 * @brief the paint function.
	 */
	void paint(QPainter *painter, const QStyleOptionGraphicsItem *,
		   QWidget *);

	/**
	 * @brief returns true if this keypoint is in the visble area of its
	 * image
	 * @return true if this keypoint is in the visble area of its image
	 */
	bool imagePointisVisible()
		{return image_->visibleArea().contains(pt.x, pt.y);	}

	/**
	 * @brief if show is true this keypoint will be visible if it is the
	 * visibleArea
	 * @return the show Value
	 */
	bool isShown() const
		{return show_;}

	bool operator==(const cv::KeyPoint &o);

	QPen getPen() const
		{return pen_;}

	QBrush getBrush() const
		{return brush_;	}

signals:
	/**
	 * @brief this signal will be emited when the imagepoint in the scene
	 * has changed
	 * @param visible it is true if this keypoint is in the visibleArea
	 */
	void updatePoint(bool visible);

      public
slots:
	/**
	 * @brief updates the settings of this KeyPoint
	 * @param settings the object which has new settings for this keypoint
	 */
	void updateSettings(KeyPointSettings &settings);

	void setPen(const QPen &pen);

	/**
	 * @brief updates the brush of this KeyPoint
	 * @param brush a new brush
	 */
	void setBrush(const QBrush &brush);

	/**
	 * @brief if show is true this keypoint will be visible if it is the
	 * visibleArea
	 * @param b the new show Value
	 */
	void setShow(bool b);

	/**
	 * @brief updates the coordinates and visibleState of this KeyPoint
	 * @param visibleArea the visibleArea of the ZoomableImage
	 * @param zoom the zoomfactor
	 */
	void updateImageSet(const QRectF &, const qreal &zoom);

	/**
	 * @brief this method sets and connects this keypoint which the given
	 * ZoomableImage.
	 * the ZoomableImage should be in a QGraphicScene and should have same
	 * parent
	 * @param image the image
	 */
	void setZoomableImage(ZoomableImage *image);

      private:
	qtutil::ZoomableImage *image_=nullptr;

	QPen pen_;
	QBrush brush_;
	qreal zoom_;
	bool show_;

	QPointF imagePointInScene_;
};
}
}
#endif
