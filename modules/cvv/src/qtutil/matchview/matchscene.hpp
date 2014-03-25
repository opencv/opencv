#ifndef CVVISUAL_MATCH_SCENE
#define CVVISUAL_MATCH_SCENE

#include <vector>

#include <QWidget>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsProxyWidget>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "cvvmatch.hpp"
#include "cvvkeypoint.hpp"
#include "zoomableproxyobject.hpp"
#include "../synczoomwidget.hpp"
#include "../zoomableimage.hpp"
#include "../zoomableimageoptpanel.hpp"
#include "../../util/util.hpp"

namespace cvv
{
namespace qtutil
{

namespace structures
{
/**
 * @brief class for internal use in MatchScene
 */
class MatchSceneGraphicsView : public QGraphicsView
{
	Q_OBJECT
      public:
	MatchSceneGraphicsView(QGraphicsScene *scene, QWidget *parent = nullptr)
	    : QGraphicsView{ scene, parent }
	{
	}

      protected:
	virtual void resizeEvent(QResizeEvent *event) override
	{
		QGraphicsView::resizeEvent(event);
		emit signalResized();
	}

	virtual void contextMenuEvent(QContextMenuEvent *event)
	{
		emit signalContextMenu(event->globalPos());
		//event->ignore();
	}
signals:
	void signalResized();
	void signalContextMenu(const QPoint& );
};
}

/**
 * @brief this scene shows two (zoomable)images with keypoints and matches.
 */
class MatchScene : public QGraphicsView
{
	Q_OBJECT
      public:
	/**
	 * @brief the constructor
	 * @param imageLeft the left image
	 * @param imageRight the right iamge
	 * @param parent the parent Widget
	 */
	MatchScene(const cv::Mat& imageLeft,const cv::Mat& imageRight,
		   QWidget *parent = nullptr);

	/**
	 * @brief returns the lett image.
	 */
	ZoomableImage& getLeftImage()
		{return *leftImage_;}

	/**
	 * @brief returns the right image.
	 */
	ZoomableImage& getRightImage()
		{return *rightImage_;}

	/**
	 * @brief returns a ZoomableOptPanel of the left Image
	 * @return a ZoomableOptPanel of the left Image
	 */
	std::unique_ptr<ZoomableOptPanel> getLeftMatInfoWidget()
		{return util::make_unique<ZoomableOptPanel>(*leftImage_,false);}

	/**
	 * @brief returns a ZoomableOptPanel of the right Image
	 * @return a ZoomableOptPanel of the right Image
	 */
	std::unique_ptr<ZoomableOptPanel> getRightMatInfoWidget()
		{return util::make_unique<ZoomableOptPanel>(*rightImage_,false);}


	/**
	 * @brief get SyncZoomWidget
	 * @return SyncZoomWidget for the images
	 */
	std::unique_ptr<SyncZoomWidget> getSyncZoomWidget();

	/**
	 * @brief add keypoint to the left image.
	 */
	void addLeftKeypoint(std::unique_ptr<CVVKeyPoint>);

	/**
	 * @brief add keypoint to the right image.
	 */
	void addRightKeyPoint(std::unique_ptr<CVVKeyPoint>);

	/**
	 * @brief add Match.
	 */
	void addMatch(std::unique_ptr<CVVMatch>);

public slots:

	/**
	 * @brief select all visible Items.
	 */
	void selectAllVisible();

      private
slots:
	/**
	 * @brief an slot which will be calles when the MatchSceneGraphicsView was resized.
	 */
	void viewReized();

	/**
	 * @brief On right click a menu to save the current visible image or the zoomable images.
	 * @param pos The position of the right click.
	 */
	void rightClick(const QPoint &pos);

      private:
	structures::MatchSceneGraphicsView *graphicView_;
	QGraphicsScene *graphicScene_;

	qtutil::ZoomableImage *leftImage_;
	qtutil::ZoomableImage *rightImage_;

	structures::ZoomableProxyObject *leftImWidget_;
	structures::ZoomableProxyObject *rightImWidget_;
};
}
}
#endif
