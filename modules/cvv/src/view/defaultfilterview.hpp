#ifndef CVVISUAL_DEFAULT_FILTER_VIEW
#define CVVISUAL_DEFAULT_FILTER_VIEW

#include <QWidget>

#include "opencv2/core/core.hpp"

#include "filter_view.hpp"

#include "../impl/filter_call.hpp"

namespace cvv
{
namespace view
{

/**
 * @brief This Filterview shows only the given images and has no other options.
*/
class DefaultFilterView : public cvv::view::FilterView
{
	Q_OBJECT
      public:
	/**
	 * @brief Standard constructor for FilterView
	 * @param images A List of images
	 * @param parent The parent of this QWidget
	 */
	DefaultFilterView(const std::vector<cv::Mat> &images,
			  QWidget *parent = nullptr);

	/**
	 * @brief Constructor using a filter call to get its data from.
	 * @param call to get the data from.
	 * @param parent of this QWidget.
	 */
	DefaultFilterView(const cvv::impl::FilterCall &call,
			  QWidget *parent = nullptr)
	    : DefaultFilterView{ { call.original(), call.result() }, parent }
	{
	}

	~DefaultFilterView()
	{
	}
};
}
} // namespaces
#endif
