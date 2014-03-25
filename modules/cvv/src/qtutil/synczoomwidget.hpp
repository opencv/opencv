#ifndef CVVISUAL_SYNC_ZOOM_WIDGET
#define CVVISUAL_SYNC_ZOOM_WIDGET

#include <QWidget>
#include <QButtonGroup>

#include "zoomableimage.hpp"

namespace cvv
{
namespace qtutil
{

/**
 * @brief this class can set a Master for a given list of ZoomabelImmages.
 * when the master is zoomed all other images in the list
 * will show the same area this the same zoomlevel
 */
class SyncZoomWidget : public QWidget
{

	Q_OBJECT

      public:
	/**
	 * @brief the constructor
	 * @param images a list of zoomabel images
	 * @param parent the parent Widget
	 */
	SyncZoomWidget(std::vector<ZoomableImage *> images,
		       QWidget *parent = nullptr);

	/**
	 *@brief the destructor
	 */
	~SyncZoomWidget()
	{
		buttonGroup_->deleteLater();
	}
      public
slots:
	/**
	 * @brief this method set the master of the master id=images.size none master will be set
	 * @param id the id of the master image
	 */
	void selectMaster(int id);
signals:

	/**
	 * @brief a signal for the zoom syncronisation
	 */
	void updateArea(QRectF, qreal) const;

      private:
	void disconnectMaster();

	std::vector<ZoomableImage *> images_;
	size_t currentIdx_;
	QButtonGroup *buttonGroup_;
};
}
}

#endif
