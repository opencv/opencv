#ifndef CVVISUAL_AUTOFILTERWIDGET_HPP
#define CVVISUAL_AUTOFILTERWIDGET_HPP

#include <array>
#include <vector>
#include <chrono>

#include "opencv2/core/core.hpp"

#include <QWidget>
#include <QCheckBox>
#include <QVBoxLayout>
#include <QLabel>
#include <QString>

#include "filterselectorwidget.hpp"
#include "signalslot.hpp"
#include "../util/util.hpp"
#include "../util/observer_ptr.hpp"
#include "signalslot.hpp"

namespace cvv
{
namespace qtutil
{

template <std::size_t In, std::size_t Out> class AutoFilterWidget;

/**
 * @brief Contains internal structures or classes.
 *
 * Stores the image input/output, the name and update signals for all output
 *images.
 * Also provides the label to pass messages from the filter and provides a check
 *box to select
 * the input to be filtered (can be deactivated).
 */
namespace structures
{

/**
 * @brief Represents an entry of an autofilterwidget.
 */
template <std::size_t In, std::size_t Out>
class AutoFilterWidgetEntry : public QWidget
{
      public:
	/**
	 * The input type for a filter.
	 */
	using InputArray = typename AutoFilterWidget<In, Out>::InputArray;
	/**
	 * The type of an output parameter of a filter.
	 */
	using OutputArray = typename AutoFilterWidget<In, Out>::OutputArray;

	/**
	 * @brief Constructor
	 * @param name The name shown to the user.
	 * @param in Image input
	 * @param out Image output
	 * @param parent Parent widget
	 */
	AutoFilterWidgetEntry(const QString &name, InputArray in,
	                      OutputArray out, QWidget *parent = nullptr)
	    : QWidget{ parent }, name_{ name }, checkBox_{ nullptr },
	      message_{ nullptr }, in_(in), out_(out), signals_()
	{
		auto box = util::make_unique<QCheckBox>(name);
		checkBox_ = *box;

		auto msg = util::make_unique<QLabel>();
		message_ = *msg;

		auto lay = util::make_unique<QVBoxLayout>();
		lay->setAlignment(Qt::AlignTop);
		lay->setSpacing(0);
		lay->setContentsMargins(0, 0, 0, 0);
		lay->addWidget(box.release());
		lay->addWidget(msg.release());
		message_->setVisible(false);
		setLayout(lay.release());
		enableUserSelection(true);
	}

	/**
	 * @brief Destructor
	 */
	~AutoFilterWidgetEntry()
	{
	}

	/**
	 * @brief Checks wheather the check box is checked.
	 */
	operator bool() const
	{
		return checkBox_->isChecked();
	}

	/**
	 * @brief Returns the image input.
	 * @return The image input.
	 */
	InputArray input() const
	{
		return in_;
	}

	/**
	 * @brief Returns the image output.
	 * @return The image output.
	 */
	OutputArray output()
	{
		return out_;
	}

	/**
	 * @brief Returns references to the update signals.
	 * @return References to the update signals.
	 */
	std::vector<util::Reference<const SignalMatRef>> signalsRef() const
	{
		std::vector<util::Reference<const SignalMatRef>> result{};
		for (auto &elem : signals_)
		{
			result.emplace_back(elem);
		}
		return result;
	}

	/**
	 * @brief Emits all update signals.
	*/
	void emitAll() const
	{
		for (std::size_t i = 0; i < Out; i++)
		{
			signals_.at(i).emitSignal(out_.at(i).get());
		}
	}

	/**
	 * @brief Sets the message to display.
	 * @param msg The message to display (if msg == "" no message will be
	 * shown.
	 */
	void setMessage(const QString &msg = "")
	{
		if (msg == "")
		{
			message_->setVisible(false);
			return;
		}
		message_->setVisible(true);
		message_->setText(QString("<font color='red'>") + name_ +
		                  QString(": ") + msg + QString("</font>"));
	}

	/**
	 * @brief Enables/disables the checkbox.
	 * @param enabled If true the box will be enabled.
	 * If false the box will be disabled and checked.
	 */
	void enableUserSelection(bool enabled = true)
	{
		if (!enabled)
		{
			checkBox_->setChecked(true);
		}
		checkBox_->setVisible(enabled);
	}

	/**
	 * @brief The display name.
	 */
	QString name_;
	/**
	 * @brief The check box.
	 */
	util::ObserverPtr<QCheckBox> checkBox_;
	/**
	 * @brief The label to display messages.
	 */
	util::ObserverPtr<QLabel> message_;
	/**
	 * @brief Image input.
	 */
	InputArray in_;
	/**
	 * @brief Image output.
	 */
	OutputArray out_;
	/**
	 * @brief The update signals for the output.
	 */
	std::array<const SignalMatRef, Out> signals_;
};

} // structures

/**
 * @brief The AutoFilterWidget class automatically applies the selected filter
 * to all added entries.
 */
template <std::size_t In, std::size_t Out>
class AutoFilterWidget : public FilterSelectorWidget<In, Out>
{
      public:
	/**
	 * The input type for a filter.
	 */
	using InputArray = typename FilterSelectorWidget<In, Out>::InputArray;
	/**
	 * The type of an output parameter of a filter.
	 */
	using OutputArray = typename FilterSelectorWidget<In, Out>::OutputArray;

	/**
	 * @brief Constructor.
	 * @param parent The parent widget.
	 */
	AutoFilterWidget(QWidget *parent = nullptr)
	    : FilterSelectorWidget<In, Out>{ parent },
	      slotEnableUserSelection_{ [this](bool b)
	{
		this->enableUserSelection(b);
	} },
	      slotUseFilterIndividually_{ [this](bool b)
	{
		this->useFilterIndividually(b);
	} },
	      entryLayout_{ nullptr }, applyFilterIndividually_{ false },
	      entries_{}, earliestActivationTime_{}, slotApplyFilter_{ [this]()
	{
		this->autoApplyFilter();
	} },
	      userSelection_{ true }
	{
		// add sublayout
		auto lay = util::make_unique<QVBoxLayout>();
		entryLayout_ = *lay;
		lay->setContentsMargins(0, 0, 0, 0);
		this->layout_->insertLayout(0, lay.release());
		// connect auto filter slot
		QObject::connect(&(this->signalFilterSettingsChanged()),
		                 SIGNAL(signal()), &(this->slotApplyFilter_),
		                 SLOT(slot()));
	}

	/**
	 * @brief Adds an entry.
	 * @param name The name of the enty.
	 * @param in The image input.
	 * @param out The image output.
	 * @return The update signals for all output images.
	 */
	std::vector<util::Reference<const SignalMatRef>>
	addEntry(const QString &name, InputArray in, OutputArray out)
	{
		auto elem = util::make_unique<
		    structures::AutoFilterWidgetEntry<In, Out>>(name, in, out);
		auto result = elem->signalsRef();
		elem->enableUserSelection(userSelection_);
		// store element
		entries_.emplace_back(*elem);
		// add it to the widget
		entryLayout_->addWidget(elem.release());
		return result;
	}

	/**
	 * @brief Removes all entries.
	 */
	void removeAll()
	{
		structures::AutoFilterWidgetEntry<In, Out> *elemToDelete;
		for (auto &elem : entries_)
		{
			elemToDelete = elem.getPtr();
			// remove from layout
			entryLayout_->removeWidget(elemToDelete);
			// delete the element
			elemToDelete->deleteLater();
		}
		entries_.clear();
	}

	/**
	 * @brief Enabels / disables the user to select entries to filter per
	 * combo boxes.
	 * @param enabled If true it will be enabled.
	 */
	void enableUserSelection(bool enabled = true)
	{
		userSelection_ = enabled;
		for (auto &elem : entries_)
		{
			elem.get().enableUserSelection(userSelection_);
		}
	}

	/**
	 * @brief Sets whether the filter will be applied to entries it can be
	 * applied to
	 * even when one other entry cant apply the filter.
	 * @param individually If true each entry that can apply the filter does
	 * so.
	 */
	void useFilterIndividually(bool individually = true)
	{
		applyFilterIndividually_ = individually;
	}

	/**
	 * @brief Returns a slot object that calls enableUserSelection.
	 * @return A slot object that calls enableUserSelection.
	 */
	const SlotBool &slotEnableUserSelection() const
	{
		return slotEnableUserSelection_;
	}

	/**
	 * @brief Returns a slot object that calls seFilterIndividually.
	 * @return A slot object that calls seFilterIndividually.
	 */
	const SlotBool &slotUseFilterIndividually() const
	{
		return slotUseFilterIndividually_;
	}

      private:
	/**
	* @brief calls enableUserSelection
	*/
	const SlotBool slotEnableUserSelection_;
	/**
	 * @brief calls seFilterIndividually.
	 */
	const SlotBool slotUseFilterIndividually_;
	/**
	 * @brief Applies the filter when some settings where changed.
	 */
	void autoApplyFilter()
	{
		auto start = std::chrono::high_resolution_clock::now();
		// activate again?
		if (start < earliestActivationTime_)
		{
			return;
		}
		// apply filter
		if (!applyFilterIndividually_)
		{
			// only apply all filters at once
			// check wheather all filters can be applied
			std::size_t failed = 0;
			for (auto &elem : entries_)
			{
				// activated?
				if (elem.get())
				{
					auto check = this->checkInput(
					    elem.get().input());

					if (!check.first)
					{
						// elem cant apply filter
						failed++;
						elem.get().setMessage(
						    check.second);
					}
					else
					{
						// elem can apply filter. delete
						// message
						elem.get().setMessage("");
					}
				}
				else
				{
					// delete message
					elem.get().setMessage("");
				}
			}
			if (failed)
			{
				// one filter failed
				return;
			}
			// all can apply filter
			// apply filters
			for (auto &elem : entries_)
			{
				// activated?
				if (elem.get())
				{
					this->applyFilter(elem.get().input(),
					                  elem.get().output());
					elem.get().emitAll();
				};
			}
		}
		else
		{ // applyFilterIndividually_==true
			// filters can be applied individually
			for (auto &elem : entries_)
			{
				// activated?
				if (elem.get())
				{
					auto check = this->checkInput(
					    elem.get().input());
					if (!check.first)
					{
						// set message
						elem.get().setMessage(
						    check.second);
					}
					else
					{
						// apply filter+set message
						elem.get().setMessage("");
						this->applyFilter(
						    elem.get().input(),
						    elem.get().output());
						elem.get().emitAll();
					}
				}
				else
				{
					// delete message
					elem.get().setMessage("");
				}
			}
		}
		// update activation time
		earliestActivationTime_ =
		    std::chrono::high_resolution_clock::now() +
		    (std::chrono::high_resolution_clock::now() -
		     start); // duration
	}

	/**
	 * @brief The layout containing the entries.
	 */
	util::ObserverPtr<QVBoxLayout> entryLayout_;
	/**
	 * @brief Each entry that can apply the filter does so.
	 */
	bool applyFilterIndividually_;
	/**
	 * @brief The entries.
	 */
	std::vector<util::Reference<structures::AutoFilterWidgetEntry<In, Out>>>
	entries_;
	/**
	 * @brief Time for the earliest next activation for the filter.
	 */
	std::chrono::time_point<std::chrono::high_resolution_clock>
	earliestActivationTime_;
	/**
	 * @brief Slot called when filter settings change.
	 */
	Slot slotApplyFilter_;
	/**
	 * @brief Whether user selection is enabled
	 */
	bool userSelection_;
};
}
}
#endif // CVVISUAL_AUTOFILTERWIDGET_HPP
