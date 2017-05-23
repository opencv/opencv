#ifndef CVVISUAL_PORTIONSELECTOR_HPP
#define CVVISUAL_PORTIONSELECTOR_HPP

#include <vector>
#include <algorithm>
#include <limits>

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
 * @brief Provides a function to select a portion of a set.
 *
 * The highest | lowest n ^ n% of a given selection can be selected.
 * Optionally the complement can be used.
 */
class PortionSelector : public QWidget
{
      public:
	/**
	 * @brief Constructor
	 * @param parent Parent widget
	 */
	PortionSelector(QWidget *parent = nullptr)
	    : QWidget{ parent }, sigSettingsChanged_{}, highest_{ nullptr },
	      lowest_{ nullptr }, complement_{ nullptr },
	      percentVal_{ nullptr }, numberVal_{ nullptr },
	      percent_{ nullptr }, number_{ nullptr },
	      bgroupNumberPerc_{ util::make_unique<QButtonGroup>() }
	{
		// create elements
		auto highest = util::make_unique<QCheckBox>("highest");
		auto lowest = util::make_unique<QCheckBox>("lowest");
		auto complement =
		    util::make_unique<QCheckBox>("select the complement");

		auto percentVal = util::make_unique<QDoubleSpinBox>();
		auto numberVal = util::make_unique<QSpinBox>();
		auto percent = util::make_unique<QRadioButton>("(100*n)%");
		auto number = util::make_unique<QRadioButton>("n");

		highest_ = *highest;
		lowest_ = *lowest;
		complement_ = *complement;
		percentVal_ = *percentVal;
		numberVal_ = *numberVal;
		percent_ = *percent;
		number_ = *number;

		// button group
		bgroupNumberPerc_->addButton(number.get());
		bgroupNumberPerc_->addButton(percent.get());
		bgroupNumberPerc_->setExclusive(true);

		// connect subwidgets
		QObject::connect(number_.getPtr(), SIGNAL(toggled(bool)),
				 numberVal_.getPtr(), SLOT(setVisible(bool)));
		QObject::connect(percent_.getPtr(), SIGNAL(toggled(bool)),
				 percentVal_.getPtr(), SLOT(setVisible(bool)));

		// settings
		percentVal->setRange(0, 1);
		percentVal->setSingleStep(0.01);
		numberVal->setRange(0, std::numeric_limits<int>::max());

		number_->setChecked(true);
		percentVal_->setVisible(false);
		numberVal_->setVisible(true);

		// connect state changed
		QObject::connect(highest.get(), SIGNAL(clicked()),
				 &sigSettingsChanged_, SIGNAL(signal()));
		QObject::connect(lowest.get(), SIGNAL(clicked()),
				 &sigSettingsChanged_, SIGNAL(signal()));
		QObject::connect(bgroupNumberPerc_.get(),
				 SIGNAL(buttonClicked(int)),
				 &sigSettingsChanged_, SIGNAL(signal()));
		QObject::connect(numberVal.get(), SIGNAL(valueChanged(int)),
				 &sigSettingsChanged_, SIGNAL(signal()));
		QObject::connect(percentVal.get(), SIGNAL(valueChanged(double)),
				 &sigSettingsChanged_, SIGNAL(signal()));
		QObject::connect(complement.get(), SIGNAL(clicked()),
				 &sigSettingsChanged_, SIGNAL(signal()));

		// build ui
		auto lay = util::make_unique<QVBoxLayout>();
		lay->setContentsMargins(0, 0, 0, 0);
		lay->addWidget(
		    util::make_unique<QLabel>("select the").release());
		lay->addWidget(highest.release());
		lay->addWidget(lowest.release());
		lay->addWidget(number.release());
		lay->addWidget(percent.release());
		lay->addWidget(util::make_unique<QLabel>("with n =").release());
		lay->addWidget(numberVal.release());
		lay->addWidget(percentVal.release());
		lay->addWidget(complement.release());
		setLayout(lay.release());
	}

	/**
	 * @brief Returns elements from the selected range.
	 * @param selection The selection.
	 * @param comp Comparison function object used for sorting
	 * (bool cmp(const Type &a, const Type &b))
	 * @return the selected values
	 */
	template <class Type, class Compare>
	std::vector<Type> select(std::vector<Type> selection,
				 Compare comp) const
	{
		// number
		int n = numberVal_->value();
		if (percent_->isChecked())
		{
			n = (percentVal_->value()) * selection.size();
		}

		std::sort(selection.begin(),selection.end(),comp);

		// lowest selected value
		std::size_t lower = (lowest_->isChecked()) ? n : 0;
		// highest selected value
		std::size_t upper =
		    selection.size() - ((highest_->isChecked()) ? n : 0);

		// element
		bool complement = complement_->isChecked();

		// filter
		std::vector<Type> result;
		for (std::size_t i = 0; i < selection.size(); i++)
		{
			if (complement != ((i < lower) || (i >= upper)))
			{
				// copy value
				result.push_back(selection.at(i));
			}
		}
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
	 * @brief Whether the highest elements should be selected
	 */
	util::ObserverPtr<QCheckBox> highest_;
	/**
	 * @brief Whether the lowest should be selected
	 */
	util::ObserverPtr<QCheckBox> lowest_;
	/**
	 * @brief Whether the complement should be selected
	 */
	util::ObserverPtr<QCheckBox> complement_;
	/**
	 * @brief The percent value
	 */
	util::ObserverPtr<QDoubleSpinBox> percentVal_;
	/**
	 * @brief The number of elements
	 */
	util::ObserverPtr<QSpinBox> numberVal_;
	/**
	 * @brief Whether the percent value should be used
	 */
	util::ObserverPtr<QRadioButton> percent_;
	/**
	 * @brief Whether the number value should be used
	 */
	util::ObserverPtr<QRadioButton> number_;
	/**
	 * @brief Button group for number/percentage
	 */
	std::unique_ptr<QButtonGroup> bgroupNumberPerc_;
};
}
} // namespace cvv, namespace qtutil

#endif // CVVISUAL_PORTIONSELECTOR_HPP
