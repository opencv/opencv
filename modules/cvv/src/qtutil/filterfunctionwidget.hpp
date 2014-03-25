#ifndef CVVISUAL_FILTERFUNCTIONWIDGET_HPP
#define CVVISUAL_FILTERFUNCTIONWIDGET_HPP
// STD
#include <array>
#include <type_traits>

// QT
#include <QWidget>
#include <QString>

// OCV
#include "opencv2/core/core.hpp"

// cvv
#include "signalslot.hpp"
#include "../util/util.hpp"

namespace cvv
{
namespace qtutil
{

/**
 * @brief The input type for FilterFunctionWidgets.
 */
template <std::size_t In>
using CvvInputArray = std::array<util::Reference<const cv::Mat>, In>;

/**
 * @brief The output type for FilterFunctionWidgets.
 */
template <std::size_t Out>
using CvvOutputArray = std::array<util::Reference<cv::Mat>, Out>;

/**
 * @brief The type for the input of the filter.
 *
 * Inherit from it if you want to provide an image filter.
 * Use the widget to let the user choose parameters.
 * Emit stateChanged when user input leads to different parameters.
 *
 * @tparam In The number of input images.
 * @tparam Out The number of output images.
 */
template <std::size_t In, std::size_t Out>
class FilterFunctionWidget : public QWidget
{
	static_assert(Out > 0, "Out should be >0.");

      public:
	/**
	 * @brief The input type.
	 */
	using InputArray = CvvInputArray<In>;

	/**
	 * @brief The output type.
	 */
	using OutputArray = CvvOutputArray<Out>;

	/**
	 * @brief Constructor
	 * @param parent Parent widget.
	 */
	FilterFunctionWidget(QWidget *parent = nullptr)
	    : QWidget{ parent }, signFilterSettingsChanged_{}
	{
	}

	/**
	 * @brief Applys the filter to in and saves the result in out.
	 * @param in The input images.
	 * @param out The output images.
	 */
	virtual void applyFilter(InputArray in, OutputArray out) const = 0;

	/**
	 * @brief Checks whether input can be progressed by the applyFilter
	 *function.
	 * @param in The input images.
	 * @return bool = true: the filter can be executed.
	 *		bool = false: the filter cant be executed (e.g. images
	 *have wrong depth)
	 *		QString = message for the user (e.g. why the filter can't
	 *be progressed.)
	 */
	virtual std::pair<bool, QString> checkInput(InputArray in) const = 0;

	const Signal &signalFilterSettingsChanged() const
	{
		return signFilterSettingsChanged_;
	}

      private:
	/**
	 * @brief Signal to emit when user input leads to different parameters.
	 */
	const Signal signFilterSettingsChanged_;
};
}
} // end namespaces qtutil, cvv
#endif // CVVISUAL_FILTERFUNCTIONWIDGET_HPP
