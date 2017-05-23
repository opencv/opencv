#ifndef CVVISUAL_ZOOMALBE_PROXY_OBJECT
#define CVVISUAL_ZOOMALBE_PROXY_OBJECT

#include <QGraphicsProxyWidget>
#include <QGraphicsSceneContextMenuEvent>

#include "../zoomableimage.hpp"

namespace cvv
{
namespace qtutil
{
namespace structures
{
/**
 * @brief spezific class for MatchScene
 */
class ZoomableProxyObject : public QGraphicsProxyWidget
{
      public:
	ZoomableProxyObject(ZoomableImage *zoom);

	~ZoomableProxyObject()
	{
	}

      protected:
	virtual void contextMenuEvent(QGraphicsSceneContextMenuEvent *event)
	{
		event->ignore();
	}

	virtual void wheelEvent(QGraphicsSceneWheelEvent *event);

      private:
	ZoomableImage *image_;
};

}
}
}
#endif
