
#include <QHBoxLayout>
#include <QPoint>
#include <QScrollBar>
#include <QPainterPath>
#include <QTransform>
#include <QFileDialog>
#include <QMenu>
#include <QAction>

#include "matchscene.hpp"

namespace cvv
{
namespace qtutil
{

MatchScene::MatchScene(const cv::Mat &imageLeft, const cv::Mat &imageRight, QWidget *parent)
    : QGraphicsView{ parent }
{
	auto basicLayout = util::make_unique<QHBoxLayout>();

	auto graphicScene = util::make_unique<QGraphicsScene>(this);
	graphicScene_ = graphicScene.get();
	auto graphicView =
	    util::make_unique<structures::MatchSceneGraphicsView>(
		graphicScene.release());
	graphicView_ = graphicView.get();

	auto leftImage = util::make_unique<ZoomableImage>(imageLeft);
	auto rightImage = util::make_unique<ZoomableImage>(imageRight);
	leftImage_ = leftImage.get();
	rightImage_ = rightImage.get();

	auto leftImWidget = util::make_unique<structures::ZoomableProxyObject>(leftImage.release());
	auto rightImWidget = util::make_unique<structures::ZoomableProxyObject>( rightImage.release() );

	leftImWidget_=leftImWidget.get();
	rightImWidget_=rightImWidget.get();

	graphicScene_->addItem(leftImWidget.release());
	graphicScene_->addItem(rightImWidget.release());

	leftImWidget_->setFlag(QGraphicsItem::ItemIsFocusable);
	rightImWidget_->setFlag(QGraphicsItem::ItemIsFocusable);

	basicLayout->setContentsMargins(0, 0, 0, 0);
	basicLayout->addWidget(graphicView.release());
	setLayout(basicLayout.release());

	connect(graphicView_, SIGNAL(signalResized()), this,
		SLOT(viewReized()));
	connect(graphicView_, SIGNAL(signalContextMenu(QPoint)), this,
		SLOT(rightClick(QPoint)));

	// rightklick
	setContextMenuPolicy(Qt::CustomContextMenu);
	QObject::connect(this,
			 SIGNAL(customContextMenuRequested(const QPoint &)),
			 this, SLOT(rightClick(QPoint)));
}

std::unique_ptr<SyncZoomWidget> MatchScene::getSyncZoomWidget()
{
	std::vector<ZoomableImage *> images;
	images.push_back(leftImage_);
	images.push_back(rightImage_);
	return util::make_unique<SyncZoomWidget>(images);
}

void MatchScene::addLeftKeypoint(std::unique_ptr<CVVKeyPoint> keypoint)
{
	keypoint->setZoomableImage(leftImage_);
	graphicScene_->addItem(keypoint.release());
}

void MatchScene::addRightKeyPoint(std::unique_ptr<CVVKeyPoint> keypoint)
{
	keypoint->setZoomableImage(rightImage_);
	graphicScene_->addItem(keypoint.release());
}

void MatchScene::addMatch(std::unique_ptr<CVVMatch> cvmatch)
{
	graphicScene_->addItem(cvmatch.release());
}

void MatchScene::selectAllVisible()
{
	QPainterPath selectionPath{};
	selectionPath.addRect(graphicScene_->itemsBoundingRect());
	graphicScene_->setSelectionArea(selectionPath, QTransform{});
}

void MatchScene::viewReized()
{
	int width = graphicView_->viewport()->width();
	int heigth = graphicView_->viewport()->height();

	// left
	leftImWidget_->setPos(0, 0);
	leftImWidget_->setMinimumSize((width / 2), heigth);
	leftImWidget_->setMaximumSize(width / 2, heigth);

	// right
	rightImWidget_->setPos(width / 2, 0);
	rightImWidget_->setMinimumSize(width / 2, heigth);
	rightImWidget_->setMaximumSize(width / 2, heigth);
	rightImWidget_->update();
	leftImWidget_->update();
	graphicView_->setSceneRect(0, 0, width, heigth);
}

void MatchScene::rightClick(const QPoint &pos)
{
	QPoint p = pos;
	QMenu menu;

	menu.addAction("Save visible image");
	menu.addAction("Save left image (orginal)");
	menu.addAction("Save left image (visible)");
	menu.addAction("Save right image (orginal)");
	menu.addAction("Save right image (visible)");

	QAction *item = menu.exec(p);
	if (item)
	{
		QString fileName = QFileDialog::getSaveFileName(
		    this, tr("Save File"), ".",
		    tr("BMP (*.bmp);;GIF (*.gif);;JPG (*.jpg);;PNG (*.png);;"
		       "PBM (*.pbm);;PGM (*.pgm);;PPM (*.ppm);;XBM (*.xbm);;"
		       "XPM (*.xpm)"));
		if (fileName == "")
		{
			return;
		}
		QPixmap pmap;

		QString str=item->text();

		if(str.contains("left"))
		{
			if(str.contains("orginal")){
				pmap = leftImage_->fullImage();
			}else{
				pmap = leftImage_->visibleImage();
			}
		}else if(str.contains("right"))
		{
			if(str.contains("orginal")){
				pmap = rightImage_->fullImage();
			}else{
				pmap = rightImage_->visibleImage();
			}
		}else{
			pmap = QPixmap::grabWidget(graphicView_->viewport());
		}
		pmap.save(fileName, 0, 100);
	}
}
}
}
