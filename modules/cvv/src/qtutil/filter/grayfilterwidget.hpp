#ifndef CVVISUAL_GRAYFILTERWIDGET_HPP
#define CVVISUAL_GRAYFILTERWIDGET_HPP

#include <vector>

#include <QVBoxLayout>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QWidget>
#include <QObject>
#include <QString>

#include "opencv2/core/core.hpp"

#include "../filterfunctionwidget.hpp"
#include "../../util/observer_ptr.hpp"

namespace cvv
{
namespace qtutil
{

/**
 * @brief Represents a gray filter.
 *
 * The user can select the factors used for every channel.
 */
class GrayFilterWidget : public FilterFunctionWidget<1, 1>
{
	Q_OBJECT
      public:
	/**
	 * @brief The input type.
	 */
	using InputArray = FilterFunctionWidget<1, 1>::InputArray;

	/**
	 * @brief The output type.
	 */
	using OutputArray = FilterFunctionWidget<1, 1>::OutputArray;

	/**
	 * @brief Constructor
	 */
	GrayFilterWidget(QWidget *parent = nullptr);

	/**
	 * @brief Applys the filter to in and saves the result in out.
	 * @param in The input images.
	 * @param out The output images.
	 */
	virtual void applyFilter(InputArray in, OutputArray out) const override;

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
	virtual std::pair<bool, QString> checkInput(InputArray) const override;

      private
slots:
	/**
	 * @brief Sets the number of channels.
	 * @param n The number of channels.
	 */
	void setChannel(int n)
	{
		setChannel(static_cast<std::size_t>(n));
	}

	/**
	 * @brief Sets the number of channels.
	 * @param n The number of channels.
	 */
	void setChannel(std::size_t n);

	/**
	 * @brief Sets the standard gray filter. (0.299*R + 0.587*G + 0.114*B)
	 */
	void setStd();

      private:
	/**
	 * @brief The layout.
	 */
	util::ObserverPtr<QVBoxLayout> layout_;
	/**
	 * @brief The spinbox to select the number of channels.
	 */
	util::ObserverPtr<QSpinBox> channel_;
	/**
	 * @brief Spin boxes for the factor for each channel.
	 */
	std::vector<util::ObserverPtr<QDoubleSpinBox>> chanValues_;
};
}
}

#endif // CVVISUAL_GRAYFILTERWIDGET_HPP
