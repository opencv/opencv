#ifndef CVVISUAL_INTERVALLSELECTOR_HPP
#define CVVISUAL_INTERVALLSELECTOR_HPP

#include <vector>
#include <algorithm>

#include <QWidget>
#include <QDoubleSpinBox>
#include <QRadioButton>
#include <QCheckBox>
#include <QLabel>
#include <QVBoxLayout>
#include <QButtonGroup>

#include "../util/util.hpp"
#include "../util/observer_ptr.hpp"
#include "signalslot.hpp"

namespace cvv
{
namespace qtutil
{

/**
 * @brief Selects elements from a double range. The elements are converted to a
 * double using the
 * given DoubleExtractor functor.
 */
class IntervallSelector : public QWidget
{
      public:
	/**
	 * @brief Constructor
	 * @param min Minimal value
	 * @param max Maximal value
	 * @param parent Parent widget
	 */
	IntervallSelector(double min, double max, QWidget *parent = nullptr)
	    : QWidget{ parent }, sigSettingsChanged_{}, min_{ nullptr },
	      max_{ nullptr }, complement_{ nullptr }
	{
		auto minb = util::make_unique<QDoubleSpinBox>();
		min_ = *minb;

		auto maxb = util::make_unique<QDoubleSpinBox>();
		max_ = *maxb;
		auto complement =
		    util::make_unique<QCheckBox>("Select the complement");
		complement_ = *complement;

		// set ranges
		minb->setRange(min, max);
		maxb->setValue(min);
		maxb->setRange(min, max);
		maxb->setValue(max);
		// connect
		QObject::connect(min_.getPtr(), SIGNAL(valueChanged(double)),
				 &sigSettingsChanged_, SIGNAL(signal()));
		QObject::connect(max_.getPtr(), SIGNAL(valueChanged(double)),
				 &sigSettingsChanged_, SIGNAL(signal()));
		QObject::connect(complement_.getPtr(), SIGNAL(clicked()),
				 &sigSettingsChanged_, SIGNAL(signal()));
		// build ui
		auto lay = util::make_unique<QVBoxLayout>();
		lay->setContentsMargins(0, 0, 0, 0);
		lay->addWidget(
		util::make_unique<QLabel>(QString{"From lower bound ("}+
					  QString::number(min)+QString{"):"}).release());
		lay->addWidget(minb.release());
		lay->addWidget(
		    util::make_unique<QLabel>(QString{"To upper bound ("}+
					      QString::number(max)+QString{"):"}).release());
		lay->addWidget(maxb.release());
		lay->addWidget(complement.release());
		setLayout(lay.release());
	}

	/**
	 * @brief Returns elements from the selected range.
	 * @param selection The selection.
	 * @param extr Extractor functor (has to be double extr(Type))
	 * @return the selected values
	 */
	template <class Type, class DoubleExtractor>
	std::vector<Type> select(std::vector<Type> selection,
				 DoubleExtractor extr) const
	{
		std::vector<Type> result;

		bool complement = complement_->isChecked();

		std::copy_if(selection.begin(), selection.end(),
			     std::back_insert_iterator<std::vector<Type>>(
				 result),
			     [=](Type t1)
		{
			return complement !=
			       // check weather the element is in the interval
			       (((min_->value()) <= extr(t1)) &&
				(extr(t1) <= max_->value()));
		});
		return result;
	}

	/**
	 * @brief Returns the signal emitted when settings are changed.
	 * @return The signal emitted when settings are changed.
	 */
	const Signal &signalSettingsChanged() const
	{
		return sigSettingsChanged_;
	}

      private:
	/**
	 * @brief Emitted when settings are changed.
	 */
	const Signal sigSettingsChanged_;
	/**
	 * @brief Spinbox to select the lower bound
	 */
	util::ObserverPtr<QDoubleSpinBox> min_;
	/**
	 * @brief Spinbox to select the upper bound
	 */
	util::ObserverPtr<QDoubleSpinBox> max_;
	/**
	 * @brief Weather the complement should be selected
	 */
	util::ObserverPtr<QCheckBox> complement_;
};
}
}
#endif // CVVISUAL_INTERVALLSELECTOR_HPP
