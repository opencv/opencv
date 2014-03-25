#ifndef CVVISUAL_REGISTERHELPER_HPP
#define CVVISUAL_REGISTERHELPER_HPP
// std
#include <map>
#include <vector>
#include <stdexcept>
#include <memory>
#include <functional>
// QT
#include <QWidget>
#include <QString>
#include <QComboBox>
#include <QVBoxLayout>
// cvv
#include "signalslot.hpp"

namespace cvv
{
namespace qtutil
{
/**
 * @brief The RegisterHelper class can be inherited to gain a mechanism to
 *register fabric functions
 * for QWidgets.
 *
 * The registered functions are shared between all instances of a class.
 * A QComboBox is provided for user selection.
 * The content of the QComboBox is updated whenever a function is registered.
 *
 * Inheriting classes have to delete the member comboBox_ on destruction!
 * (e.g. by putting it into a layout)
 */
template <class Value, class... Args> class RegisterHelper
{
      public:
	/**
	 * @brief Constructor
	 */
	RegisterHelper()
	    : comboBox_{ new QComboBox{} }, signElementSelected_{},
	      slotElementRegistered_{ [&](const QString &name)
	{
		comboBox_->addItem(name);
	} }
	{
		// elem registered
		QObject::connect(&signalElementRegistered(),
		                 &SignalQString::signal,
		                 &slotElementRegistered_, &SlotQString::slot);
		// connect
		QObject::connect(comboBox_, &QComboBox::currentTextChanged,
		                 &signalElementSelected(),
		                 &SignalQString::signal);
		// add current list of elements
		for (auto &elem : registeredElementsMap())
		{
			comboBox_->addItem(elem.first);
		}

	}

	~RegisterHelper()
	{
	}

	/**
	 * @brief Returns the current selection from the QComboBox
	 * @return The current selection from the QComboBox
	 */
	QString selection() const
	{
		return comboBox_->currentText();
	}

	/**
	 * @brief Checks whether a function was registered with the name.
	 * @param name The name to look up
	 * @return true if there is a function. false otherwise
	 */
	static bool has(const QString &name)
	{
		return registeredElementsMap().find(name) !=
		       registeredElementsMap().end();
	}

	/**
	 * @brief Returns the names of all registered functions.
	 * @return The names of all registered functions.
	 */
	static std::vector<QString> registeredElements()
	{
		std::vector<QString> result{};
		for (auto &elem : registeredElementsMap())
		{
			result.push_back(elem.first);
		};
		return result;
	}

	/**
	 * @brief Registers a function.
	 * @param name The name.
	 * @param fabric The fabric function.
	 * @return true if the function was registered. false if the name was
	 * taken
	 * (the function was not registered!)
	 */
	static bool registerElement(
	    const QString &name,
	    const std::function<std::unique_ptr<Value>(Args...)> &fabric)
	{
		if (has(name))
		{
			return false;
		};

		registeredElementsMap().emplace(name, fabric);

		signalElementRegistered().emitSignal(name);

		return true;
	}

	/**
	 * @brief Selects an function according to name.
	 * @param name The name of the function to select.
	 * @return true if the function was selected. false if no function has
	 * name.
	 */
	bool select(const QString &name)
	{
		if (!has(name))
		{
			return false;
		}
		comboBox_->setCurrentText(name);
		return true;
	}

	/**
	 * @brief Returns the function according to the current selection of the
	 * QComboBox.
	 * @throw std::out_of_range If there is no such function.
	 * @return The function according to the current selection of the
	 * QComboBox.
	 */
	std::function<std::unique_ptr<Value>(Args...)> operator()()
	{
		return (*this)(selection());
	}

	/**
	 * @brief Returns the function according to name.
	 * @param The name of a registered function.
	 * @throw std::out_of_range If there is no such function.
	 * @return The function according to name.
	 */
	std::function<std::unique_ptr<Value>(Args...)>
	operator()(const QString &name)
	{
		return registeredElementsMap().at(name);
	}

	/**
	 * @brief Returns a signal emitted whenever a function is registered.
	 * @return A signal emitted whenever a function is registered.
	 */
	static const SignalQString &signalElementRegistered()
	{
		static const SignalQString signElementRegistered_{};
		return signElementRegistered_;
	}

	/**
	 * @brief Returns the signal emitted whenever a new element in the
	 * combobox is selected.
	 * (passes the selected string)
	 * @return The signal emitted whenever a new element in the combobox is
	 * selected.
	 * (passes the selected string)
	 */
	const SignalQString &signalElementSelected() const
	{
		return signElementSelected_;
	}

      protected:
	/**
	 * @brief QComboBox containing all names of registered functions
	 */
	QComboBox *comboBox_;

      private:
	/**
	 * @brief Signal emitted whenever a new element in the combobox is
	 * selected.
	 * (passes the selected string)
	 */
	const SignalQString signElementSelected_;

	/**
	 * @brief Slot called whenever a function is registered
	 */
	const SlotQString slotElementRegistered_;

	/**
	 * @brief Returns the map of registered functions and their names.
	 * @return The map of registered functions and their names.
	 */
	static std::map<QString,
	                std::function<std::unique_ptr<Value>(Args...)>> &
	registeredElementsMap()
	{
		static std::map<QString,
		                std::function<std::unique_ptr<Value>(Args...)>>
		map{};
		return map;
	}
};
}
} // end namespaces qtutil, cvv

#endif // CVVISUAL_REGISTERHELPER_HPP
