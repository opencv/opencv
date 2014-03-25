#ifndef CVVISUAL_OVERLAY_FILTER_WIDGET_HPP
#define CVVISUAL_OVERLAY_FILTER_WIDGET_HPP

#include <unordered_map>

#include "../../util/observer_ptr.hpp"
#include "../filterselectorwidget.hpp"

namespace cvv
{
namespace qtutil
{

/**
* @brief Class providing functionality to compute an overlay image of two
*	input matrices.
*/
class OverlayFilterWidget : public FilterFunctionWidget<2, 1>
{
	Q_OBJECT
      public:
	/**
	 * @brief The input type.
	 */
	using InputArray = FilterFunctionWidget<2, 1>::InputArray;
	// std::array<util::Reference<const cv::Mat>,2>

	/**
	 * @brief The output type.
	 */
	using OutputArray = FilterFunctionWidget<2, 1>::OutputArray;
	// std::array<util::Reference<cv::Mat>,1>

	/**
	* @brief Constructs OverlayFilterWidget with default opacity 0,5.
	* @param parent The parent of the widget
	*/
	OverlayFilterWidget(QWidget *parent = nullptr);

	/**
	* The opacity of the second image while overlaying is indicated by
	* opacityOfFilterImg_.
	* @brief Overlays the original images
	* @param in Array of input matrices
	* @param out Array of output matrices
	*/
	void applyFilter(InputArray in, OutputArray out) const;

	/**
	* Checks whether the matrices have the same size and same number of
	* channels.
	* @brief Checks whether matrices in "in" can be processed by Overlayfilter
	* @param in Array of input matrices
	*/
	std::pair<bool, QString> checkInput(InputArray in) const;

      private:
	double opacityOfFilterImg_;
	//< Opacity of the second input image when ovelaying

      private
slots:
	/**
	* @brief Sets opacityOfFilterImg_ and emits signFilterSettingsChanged.
	* @param op New opacity
	*/
	void updateOpacity(int op);
};
}
}

#endif
