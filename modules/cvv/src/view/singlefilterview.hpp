#ifndef CVVISUAL_SINGLE_FILTER_VIEW
#define CVVISUAL_SINGLE_FILTER_VIEW

#include <QWidget>

#include "opencv2/core/core.hpp"

#include "filter_view.hpp"
#include "../impl/filter_call.hpp"

namespace cvv
{
namespace view
{

/**
 * @brief This Filterview applies the same filter for all given images.
*/
class SingleFilterView : public cvv::view::FilterView
{
	Q_OBJECT
      public:
	/**
	 * @brief Cnstructor
	 * @param images a vector of images which will be shown
	 * @param parent the parent Widget
	 */
	SingleFilterView(const std::vector<cv::Mat> &images,
			 QWidget *parent = nullptr);

	/**
	 * @brief Constructor using a filter call to get its data from.
	 * @param call Call to get the data from.
	 * @param parent Parent of this QWidget.
	 */
	SingleFilterView(const cvv::impl::FilterCall &call,
			 QWidget *parent = nullptr)
	    : SingleFilterView{ { call.original(), call.result() }, parent }
	{
	}
};
}
} // namespaces
#endif
