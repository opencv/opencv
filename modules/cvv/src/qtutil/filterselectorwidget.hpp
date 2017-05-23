#ifndef CVVISUAL_FILTERSELECTORWIDGET_HPP
#define CVVISUAL_FILTERSELECTORWIDGET_HPP
// STD
#include <stdexcept>
#include <array>
#include <type_traits>

// QT
#include <QWidget>
#include <QString>
#include <QComboBox>
#include <QPushButton>

// OCV
#include "opencv2/core/core.hpp"

// CVV
#include "signalslot.hpp"
#include "registerhelper.hpp"
#include "filterfunctionwidget.hpp"
#include "../util/util.hpp"
#include "../util/observer_ptr.hpp"

namespace cvv
{
namespace qtutil
{

// forward
template <std::size_t In, std::size_t Out, class Filter>
bool registerFilter(const QString &name);

/**
 * @brief The FilterSelectorWidget class
 */
template <std::size_t In, std::size_t Out>
class FilterSelectorWidget
    : public RegisterHelper<FilterFunctionWidget<In, Out>, QWidget *>,
      public FilterFunctionWidget<In, Out>
{
	static_assert(Out > 0, "Out must not be 0!");

      public:
	/**
	 * @brief The input type.
	 */
	using InputArray = typename FilterFunctionWidget<In, Out>::InputArray;

	/**
	 * @brief The output type.
	 */
	using OutputArray = typename FilterFunctionWidget<In, Out>::OutputArray;

	/**
	 * @brief Constuctor
	 * @param parent The parent widget.
	 */
	FilterSelectorWidget(QWidget *parent = nullptr)
	    : RegisterHelper<FilterFunctionWidget<In, Out>, QWidget *>{},
	      FilterFunctionWidget<In, Out>{ parent }, layout_{ nullptr },
	      currentFilter_{ nullptr }, slotFilterSelected_{ [this]()
	{
		this->updatedSelectedFilter();
	} }
	{
		auto lay = util::make_unique<QVBoxLayout>();
		layout_ = *lay;
		this->layout_->setAlignment(Qt::AlignTop);
		this->layout_->setSpacing(0);
		this->layout_->addWidget((this->comboBox_));
		// connect elem selected with update for it
		QObject::connect(&(this->signalElementSelected()),
		                 SIGNAL(signal(QString)),
		                 &(this->slotFilterSelected_), SLOT(slot()));
		this->setLayout(lay.release());
		// update for initial selection (if it is valid)
		if (this->has(this->selection()))
		{
			updatedSelectedFilter();
		}
		// add an apply button
		auto button = util::make_unique<QPushButton>("apply");
		// connect it
		QObject::connect(button.get(), SIGNAL(clicked()),
		                 &(this->signalFilterSettingsChanged()),
		                 SIGNAL(signal()));
		this->layout_->addWidget(button.release());
	}

	~FilterSelectorWidget()
	{
	}

	/**
	 * @brief Applies the selected filter.
	 * @param in Input images.
	 * @param out Output images.
	 * @throw std::invalid_argument checkInput(in).first==false
	 */
	virtual void applyFilter(InputArray in, OutputArray out) const override
	{
		auto check = checkInput(in);
		if (!check.first)
		{
			throw std::invalid_argument{
				check.second.toStdString()
			};
		}
		return currentFilter_->applyFilter(in, out);
	}

	/**
	 * @brief Checks whether input can be progressed by the selected filter.
	 * @param in The input images.
	 * @return bool = true: the filter can be executed.
	 *		bool = false: the filter cant be executed (e.g. images
	 *have wrong depth)
	 *		QString = message for the user (e.g. why the filter can't
	 *be progressed.)
	 */
	virtual std::pair<bool, QString> checkInput(InputArray in) const
	    override
	{
		if (!currentFilter_)
		{
			return { false, "No entry selected." };
		}
		return currentFilter_->checkInput(in);
	}

	/**
	 * @brief Registers a FilterFunctionWidget with a given name.
	 * @param name The name.
	 * @return true: If the function was registered. false: If the name was
	 * taken
	 * (the function was not registered!)
	 */
	template <class Filter> static bool registerFilter(const QString &name)
	{
		return qtutil::registerFilter<In, Out, Filter>(name);
	}

      protected:
	/**
	 * @brief The layout
	 */
	util::ObserverPtr<QVBoxLayout> layout_;

      private:
	/**
	 * @brief Performs the update after a selection occurred.
	 */
	void updatedSelectedFilter()
	{
		if ((this->currentFilter_))
		{
			layout_->removeWidget((this->currentFilter_.getPtr()));
			// disconnect
			QObject::disconnect(
			    &(this->currentFilter_
			          ->signalFilterSettingsChanged()),
			    0, &(this->signalFilterSettingsChanged()), 0);
			currentFilter_->deleteLater();
		}
		auto filt = (*this)()(nullptr);
		this->currentFilter_ = *filt;
		this->layout_->insertWidget(2, filt.release());
		// pass signal
		QObject::connect(
		    &(this->currentFilter_->signalFilterSettingsChanged()),
		    SIGNAL(signal()), &(this->signalFilterSettingsChanged()),
		    SIGNAL(signal()));
		// settings changed
		this->signalFilterSettingsChanged().emitSignal();
	}

	/**
	 * @brief the current filter
	 */
	util::ObserverPtr<FilterFunctionWidget<In, Out>> currentFilter_;
	/**
	 * @brief Slot called when user changes selection
	 */
	const Slot slotFilterSelected_;
}; // FilterSelectorWidget

/**
 * @brief Registers a FilterFunctionWidget with a given name.
 * @param name The name.
 * @return true: If the function was registered. false: If the name was taken
 * (the function was not registered!)
 */
template <std::size_t In, std::size_t Out, class Filter>
bool registerFilter(const QString &name)
{
	return FilterSelectorWidget<In, Out>::registerElement(
	    name, [](QWidget *parent)
	{
		    return std::unique_ptr<FilterFunctionWidget<In, Out>>{
			    new Filter{ parent }
		    };
	    });
}
}
} // end namespaces qtutil, cvv
#endif // CVVISUAL_FILTERSELECTORWIDGET_HPP
