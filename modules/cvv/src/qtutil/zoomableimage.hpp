#ifndef CVVISUAL_ZOOMABLEIMAGE_HPP
#define CVVISUAL_ZOOMABLEIMAGE_HPP

#include <QWidget>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsTextItem>
#include <QRectF>
#include <QGraphicsPixmapItem>
#include <QWheelEvent>
#include <QApplication>
#include <QTimer>
#include <QScrollBar>

#include "opencv2/core/core.hpp"

#include "util.hpp"
#include "../util/util.hpp"
#include "../util/observer_ptr.hpp"

namespace cvv
{
namespace qtutil
{
namespace structures
{
/**
 * @brief A graphics view with overwritten event handlers.
 */
class ZoomableImageGraphicsView : public QGraphicsView
{
      public:
	/**
	 * @brief Constructor
	 */
	ZoomableImageGraphicsView() : QGraphicsView{}
	{
	}

      protected:
	/**
	 * @brief Ignores the wheel event if ctrl is pressed.
	 * @param event The event.
	 */
	virtual void wheelEvent(QWheelEvent *event) override;

	/**
	 * @brief Ignores the mouse move event.
	 * @param event The event.
	 */
	virtual void mouseMoveEvent(QMouseEvent * event) override
	{
		event->ignore();
	}
};
}

/**
 * @brief The ZoomableImage class shows an image and adds zoom functionality.
 */
class ZoomableImage : public QWidget
{
	Q_OBJECT
      public:
	/**
	 * @brief Constructor
	 * @param mat The image to display
	 * @param parent The parent widget.
	 */
	ZoomableImage(const cv::Mat &mat = cv::Mat{},
		      QWidget *parent = nullptr);

	/**
	 * @brief Destructor
	 */
	~ZoomableImage()
	{
		//disconnect everything from custom signals
		QObject::disconnect(this, SIGNAL(updateArea(QRectF,qreal)), 0, 0);
		QObject::disconnect(this, SIGNAL(updateConversionResult(cv::Mat,ImageConversionResult)), 0, 0);
		QObject::disconnect(this, SIGNAL(updateMouseHover(QPointF,QString,bool)), 0, 0);

		//disconnect all slots
		QObject::disconnect((view_->horizontalScrollBar()),
				 &QScrollBar::valueChanged, this,
				 &ZoomableImage::viewScrolled);
		QObject::disconnect((view_->verticalScrollBar()),
				 &QScrollBar::valueChanged, this,
				 &ZoomableImage::viewScrolled);
		QObject::disconnect(&updateAreaTimer_,SIGNAL(timeout()),this,SLOT(emitUpdateArea()));


		//stop timer from being activated
		updateAreaTimer_.stop();
		updateAreaQueued_=true;
	}

	/**
	 * @brief Returns the current image.
	 * @return The current image.
	 */
	const cv::Mat &mat() const
	{
		return mat_;
	}

	/**
	 * @brief Returns the current image.
	 * @return The current image.
	 */
	cv::Mat &mat()
	{
		return mat_;
	}

	/**
	 * @brief Returns the visible area of the image.
	 * @return The visible area of the image.
	 */
	QRectF visibleArea() const;

	/**
	 * @brief Returns the zoom factor.
	 * @return The zoom factor.
	 */
	qreal zoom() const
	{
		return zoom_;
	}

	QPointF mapImagePointToParent(QPointF) const;

	/**
	 * @brief Returns the threshold that determines wheather the pixelvalues
	 * are shown.
	 * @return
	 */
	qreal threshold() const
	{
		return threshold_;
	}

	/**
	 * @brief The overridden resize event (from QWidget).
	 */
	virtual void resizeEvent(QResizeEvent *) override
	{
		queueUpdateArea();
	}

	/**
	 * @brief Returns the width of the image.
	 * @return The width of the image.
	 */
	int imageWidth() const
	{
		return mat_.cols;
	}

	/**
	 * @brief Returns the height of the image.
	 * @return The height of the image.
	 */
	int imageHeight() const
	{
		return mat_.rows;
	}

	/**
	 * @brief Returns weather pixel values are shown depending on threshold.
	 * @return Weather pixel values are shown depending on threshold.
	 */
	bool autoShowValues() const
	{
		return autoShowValues_;
	}

	/**
	 * @brief Returns the current scroll factor for CTRL+scroll.
	 * @return The current scroll factor for CTRL+scroll.
	 */
	qreal scrollFactorCTRL() const
	{
		return scrollFactorCTRL_;
	}

	/**
	 * @brief Returns the current scroll factor for CTRL+shift+scroll.
	 * @return The current scroll factor for CTRL+shift+scroll.
	 */
	qreal scrollFactorCTRLShift() const
	{
		return scrollFactorCTRLShift_;
	}

	/**
	 * @brief Returns the image managed by this item.
	 * @return The image managed by this item.
	 */
	QPixmap fullImage() const
	{
		return pixmap_->pixmap();
	}

	/**
	 * @brief Returns the currently visible area as an pixmap.
	 * @return The currently visible area as an pixmap.
	 */
	QPixmap visibleImage() const
	{
		return QPixmap::grabWidget(view_->viewport());
	}

	/**
	 * @brief Returns the last image conversion result.
	 * @return The last image conversion result.
	 */
	ImageConversionResult lastConversionResult() const
	{
		return lastConversionResult_;
	}

signals:
	/**
	 * @brief Emmited whenever the image is updated. It passes the
	 * conversion result
	 * and the image.
	 */
	void updateConversionResult(const cv::Mat &,
				    ImageConversionResult) const;

	/**
	 *@brief Emitted whenever the visible area changes. Passes the visible
	 *area and zoom factor.
	 */
	void updateArea(QRectF, qreal) const;

	/**
	 * @brief Updates information at mouse position (position, channel values of the pixel, whether the position is in the image)
	 */
	void updateMouseHover(QPointF,QString,bool);

      public
slots:
	/**
	 * @brief Sets the curent visible area (the center) and zoom factor.
	 * @param rect The area.
	 * @param zoom The zoom.
	 */
	void setArea(QRectF rect, qreal zoom);

	/**
	 * @brief Updates the image to display.
	 * @param mat The new image to display.
	 */
	void setMatR(cv::Mat &mat)
	{
		setMat(mat);
	}

	/**
	 * @brief Updates the image to display.
	 * @param mat The new image to display.
	 */

	void setMat(cv::Mat mat);

	/**
	 * @brief Updates the zoom factor.
	 * @param factor The zoom factor.
	 */
	void setZoom(qreal factor = 1);

	/**
	 * @brief Sets weather pixel values are shown depending on threshold.
	 * @param enable If true pixel values are shown depending on threshold.
	 */
	void setAutoShowValues(bool enable = true)
	{
		autoShowValues_ = enable;
	}

	/**
	 * @brief Sets the threshold that determines wheather the pixelvalues
	 * are shown.
	 * @param threshold The threshold.
	 */
	void setThreshold(qreal threshold = 60)
	{
		threshold_ = threshold;
	}

	/**
	 * @brief Resizes the image so it is fully visible.
	 */
	void showFullImage();

	/**
	 * @brief Returns the current CTRL+scroll zoom factor.
	 * @return The current scroll factor.
	 */
	void setCTRLZoomFactor(qreal factor = 1.025)
	{
		scrollFactorCTRL_ = factor;
	}

	/**
	 * @brief Returns the current CTRL+ shift+scroll zoom factor.
	 * @return The current scroll factor.
	 */
	void setCTRLShiftZoomFactor(qreal factor = 1.01)
	{
		scrollFactorCTRLShift_ = factor;
	}

	/**
	 * @brief The overridden wheel event (from QWidget).
	 * @param event The event.
	 */
	virtual void wheelEvent(QWheelEvent *event) override;

	/**
	 * @brief Sets the new delay between two updateArea signals.
	 * (if 0 the signal will be emitted as soon as all the events in the window system's event queue have been processed)
	 * @param i The new delay. (ignored if <0)
	 */
	void setUpdateAreaDelay(int i)
	{
		if(i>=0)
		{
			updateAreaDelay_=i;
		}
	}

protected:
	/**
	 * @brief The mouse move event will output informations regarding the pixelvalue and the position in the image with updateMouseHover().
	 * @param event The event
	 */
	virtual void mouseMoveEvent(QMouseEvent * event) override;

      private
slots:
	/**
	 * @brief Called when the graphic view is scrolled.
	 */
	void viewScrolled()
	{
		queueUpdateArea();
	}

	/**
	 * @brief Draws the pixel value for all visible pixels.
	 */
	void drawValues();

	/**
	 * @brief On right click a menu to save the current visible image or the full image will appear.
	 * @param pos The position of the right click.
	 */
	void rightClick(const QPoint &pos);

	/**
	 * @brief This function will emit updateArea() and set updateAreaQueued_ = false.
	 * It will be called by updateAreaTimer_.
	 */
	void emitUpdateArea();

	/**
	 * @brief This function will start the timer updateAreaTimer_ with the delay updateAreaDelay_ and set updateAreaQueued_ = true.
	 * If updateAreaQueued_ was already set this function will do nothing.
	 */
	void queueUpdateArea();

      private:
	/**
	 * @brief The image to display.
	 */
	cv::Mat mat_;

	/**
	 * @brief The pixmap containing the converted image.
	 */
	util::ObserverPtr<QGraphicsPixmapItem> pixmap_;
	/**
	 * @brief The graphics view showing the scene.
	 */
	util::ObserverPtr<structures::ZoomableImageGraphicsView> view_;
	/**
	 * @brief The scene containing the pixmap.
	 */
	util::ObserverPtr<QGraphicsScene> scene_;
	/**
	 * @brief The current zoom factor.
	 */
	qreal zoom_;

	/**
	 * @brief The current threshold.
	 */
	qreal threshold_;
	/**
	 * @brief Weather pixel values are shown depending on threshold.
	 */
	bool autoShowValues_;
	/**
	 * @brief The values on the graphics scene.
	 */
	std::vector<QGraphicsTextItem *> values_;
	/**
	 * @brief The factor multiplied to the number of scrolled pixels.
	 */
	qreal scrollFactorCTRL_;

	/**
	 * @brief The factor multiplied to the number of scrolled pixels.
	 */
	qreal scrollFactorCTRLShift_;

	/**
	 * @brief Will call emitUpdateArea() on timeout.
	 */
	QTimer updateAreaTimer_;

	/**
	 * @brief Whether updateAreaTimer_ is running.
	 */
	bool updateAreaQueued_;

	/**
	 * @brief The delay for updateAreaTimer_.
	 */
	int updateAreaDelay_;

	/**
	 * @brief The last image conversion result.
	 */
	ImageConversionResult lastConversionResult_;
};
}
}
#endif
