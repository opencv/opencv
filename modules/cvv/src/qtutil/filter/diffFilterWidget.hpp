
#ifndef CVVISUAL_DIFF_FILTER_WIDGET_HPP
#define CVVISUAL_DIFF_FILTER_WIDGET_HPP

#include <unordered_map>

#include "../filterselectorwidget.hpp"

namespace cvv
{
namespace qtutil
{

/**
* @brief Enum of the possible types of difference filters.
*/
enum class DiffFilterType
{
	HUE = 0,
	SATURATION = 1,
	VALUE = 2,
	LUMINANCE = VALUE,
	GRAYSCALE = 3
};

/**
* @brief Class providing functionality to compute a difference image of two
*	input matrices.
*/
class DiffFilterFunction : public FilterFunctionWidget<2, 1>
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
	* @brief Constructs DiffFilterFunction with default filter grayscale.
	* @param parent The parent of the widget
	*/
	DiffFilterFunction(QWidget *parent = nullptr);

	/**
	* @brief Applys difference filter specified by filterType_.
	* @param in Array of input matrices
	* @param out Array of output matrices
	*/
	void applyFilter(InputArray in, OutputArray out) const;

	/**
	* @brief Checks whether matrices in 'in' can be processed by this
	* DiffFilter
	*/
	std::pair<bool, QString> checkInput(InputArray in) const;

      private:
	DiffFilterType filterType_;
	//< Type of difference filter that is to be applied

	std::unordered_map<std::string, std::function<void(void)>> filterMap_;
	//< Map of all available filters with their names

	/**
	* @brief Extracts the names of all available filters from filterMap_.
	*/
	QStringList extractStringListfromMap() const;

      private
slots:
	/**
	* @brief Sets filterType_ and emits signFilterSettingsChanged.
	* @param type The name of the new DiffFilterType.
	*/
	void updateFilterType(const QString &type);
};
}
}

#endif
