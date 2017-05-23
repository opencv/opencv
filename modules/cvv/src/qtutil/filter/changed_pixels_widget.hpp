#ifndef CVVISUAL_CHANGED_PIXELS_WIDGET_HPP
#define CVVISUAL_CHANGED_PIXELS_WIDGET_HPP

#include "../filterfunctionwidget.hpp"

namespace cvv
{
namespace qtutil
{

/**
 * @brief A Comparator that will create a Mat that highlights exactly the changed
 *		pixels (black) and leaves unchanged pixels white.
 */
class ChangedPixelsWidget : public FilterFunctionWidget<2, 1>
{
	Q_OBJECT
public:

	/**
	 * @brief Constructor
	 */
	ChangedPixelsWidget(QWidget* parent = nullptr);

	/**
	 * @brief Applys the filter to in and saves the result in out.
	 * @param in The input images.
	 * @param out The output images.
	 */
	void applyFilter(InputArray in, OutputArray out) const override;

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
	std::pair<bool, QString> checkInput(InputArray in) const override;

};

}
}


#endif
