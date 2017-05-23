#ifndef CVVISUAL_IMAGE_VIEW_HPP
#define CVVISUAL_IMAGE_VIEW_HPP

#include <QString>
#include <QWidget>

#include <opencv2/core/core.hpp>

#include "../qtutil/zoomableimage.hpp"
#include "../util/observer_ptr.hpp"

namespace cvv
{
namespace view
{

/**
 * @brief Shows one image.
 */
class ImageView : public QWidget
{
	Q_OBJECT

signals:

	/**
	 * @brief update left Footer.
	 * Signal to update the left side of the footer with newText.
	 * @param newText to update the footer with.
	 */
	void updateLeftFooter(const QString &newText);

	/**
	 * @brief update right Footer.
	 * Signal to update the right side of the footer with newText.
	 * @param newText to update the footer with.
	 */
	void updateRightFoooter(const QString &newText);

      public:
	/**
	 * @brief Constructor.
	 * @param image to show.
	 * @param parent of this QWidget.
	 **/
	ImageView(const cv::Mat &image, QWidget *parent = nullptr);
	
	/**
	 * @brief Shows the full image.
	 */
	void showFullImage();
	
      private:
	util::ObserverPtr<qtutil::ZoomableImage> image;
};
}
} // namespaces

#endif

