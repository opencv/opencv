#ifndef CVVISUAL_ACCORDION_HPP
#define CVVISUAL_ACCORDION_HPP
// STD
#include <memory>
#include <stdexcept>
#include <map>
#include <limits>
// QT
#include <QWidget>
#include <QString>
#include <QVBoxLayout>
// CVV
#include "collapsable.hpp"
#include "../util/util.hpp"
#include "../util/observer_ptr.hpp"

namespace cvv
{
namespace qtutil
{
/**
 * @brief The Accordion class.
 *
 * Contains multiple widgets and their title. These get stored in collapsables.
 * The collapsables are stored in a collumn.
 */
class Accordion : public QWidget
{
	Q_OBJECT
      public:
	/**
	 * @brief The handle type to access elements
	 */
	using Handle = QWidget *;

	/**
	 * @brief Constructs an empty accordion.
	 * @param parent The parent widget
	 */
	explicit Accordion(QWidget *parent = nullptr);

	~Accordion()
	{
	}

	/**
	 * @brief Returns the element corrsponding to handle
	 * @throw std::out_of_range If there is no element corresponding to
	 * handle
	 * @return The element corrsponding to handle
	 */
	Collapsable &element(Handle handle)
	{
		return *elements_.at(handle);
	}

	const Collapsable &element(Handle handle) const
	{
		return *elements_.at(handle);
	}

	/**
	 * @brief Sets the title above the element.
	 * @param handle The element
	 * @param title The new title.
	 * @throw std::out_of_range If there is no element corresponding to
	 * handle
	 */
	void setTitle(Handle handle, const QString &title)
	{
		element(handle).setTitle(title);
	}

	/**
	 * @brief Returns the current title above the element.
	 * @param handle The element
	 * @throw std::out_of_range If there is no element corresponding to
	 * handle
	 * @return The current title above the element.
	 */
	QString title(Handle handle) const
	{
		return element(handle).title();
	}

	/**
	 * @brief Collapses an element
	 * @param handle The element to collapse
	 * @param b
	 * @parblock
	 * 		true: collapses the widget
	 * 		false: expands the widget
	 * @endparblock
	 * @throw std::out_of_range If there is no element corresponding to
	 * handle
	 */
	void collapse(Handle handle, bool b = true)
	{
		element(handle).collapse(b);
	}

	/**
	 * @brief Expands an element
	 * @param handle Element to expand
	 * @param b
	 * @parblock
	 * 		true: expands the widget
	 * 		false: collapses the widget
	 * @endparblock
	 * @throw std::out_of_range If there is no element corresponding to
	 * handle
	 */
	void expand(Handle handle, bool b = true)
	{
		collapse(handle, !b);
	}

	/**
	 * @brief Collapses all elements
	 * @param b
	 * @parblock
	 * 		true: collapses all elements
	 * 		false: expands all elements
	 * @endparblock
	 */
	void collapseAll(bool b = true);

	/**
	 * @brief Expands all elements
	 * @param b
	 * @parblock
	 * 		true: expands all elements
	 * 		false: collapses all elements
	 * @endparblock
	 */
	void expandAll(bool b = true)
	{
		collapseAll(!b);
	}

	/**
	 * @brief Makes the element invisible
	 * @param handle The element
	 * @param b
	 * @parblock
	 * 		true: makes the element invisible
	 * 		false: makes the element visible
	 * @endparblock
	 * @throw std::out_of_range If there is no element corresponding to
	 * handle
	 */
	void hide(Handle handle, bool b = true)
	{
		element(handle).setVisible(!b);
	}

	/**
	 * @brief Makes the element visible
	 * @param handle The element
	 * @param b
	 * @parblock
	 * 		true: makes the element visible
	 * 		false: makes the element invisible
	 * @endparblock
	 * @throw std::out_of_range If there is no element corresponding to
	 * handle
	 */
	void show(Handle handle, bool b = true)
	{
		hide(handle, !b);
	}

	/**
	 * @brief Sets all elements' visibility to !b
	 * @param b
	 * @parblock
	 * 		true: makes all elements invisible
	 * 		false: makes all elements visible
	 * @endparblock
	 */
	void hideAll(bool b = true);

	/**
	 * @brief Sets all elements' visibility to b
	 * @param b
	 * @parblock
	 * 		true: makes all elements visible
	 * 		false: makes all elements invisible
	 * @endparblock
	 */
	void showAll(bool b = true)
	{
		hideAll(!b);
	}

	/**
	 * @brief Inserts a widget at the given position
	 * @param title The title to display
	 * @param widget The widget to display
	 * @param isCollapsed Whether the widget is collapsed after creation
	 * @param position The position. If it is greater than the number of
	 *elements the widget
	 *	will be added to the end
	 * @return The handle to access the element
	 */
	Handle insert(const QString &title, std::unique_ptr<QWidget> widget,
	              bool isCollapsed = true,
	              std::size_t position =
	                  std::numeric_limits<std::size_t>::max());

	/**
	 * @brief Adds a widget to the end of the Accordion
	 * @param title The title to display
	 * @param widget The widget to display
	 * @param isCollapsed Whether the widget is collapsed after creation
	 * @return The handle to access the element
	 */
	Handle push_back(const QString &title, std::unique_ptr<QWidget> widget,
	                 bool isCollapsed = true)
	{
		return insert(title, std::move(widget), isCollapsed);
	}

	/**
	 * @brief Adds a widget to the front of the Accordion
	 * @param title The title to display
	 * @param widget The widget to display
	 * @param isCollapsed Whether the widget is collapsed after creation
	 * @return The handle to access the element
	 */
	Handle push_front(const QString &title, std::unique_ptr<QWidget> widget,
	                  bool isCollapsed = true)
	{
		return insert(title, std::move(widget), isCollapsed, 0);
	}

	/**
	 * @brief Removes the element and deletes it immediately.
	 * @param handle Handle of the element
	 * @param del
	 * @throw std::out_of_range If there is no element corresponding to
	 * handle
	 */
	void remove(Handle handle);

	/**
	 * @brief Removes all elements and deletes them immediately.
	 * @param del
	 */
	void clear();

	/**
	 * @brief Removes an element and returns its title and Collapsable.
	 * (ownership remains)
	 * @param handle Handle of the element
	 * @throw std::out_of_range If there is no element corresponding to
	 * handle
	 * @return Title and reference
	 */
	std::pair<QString, Collapsable *> pop(Handle handle);

	/**
	 * @brief Removes all elements from the Accordion and returns their
	 *titles
	 *	and Collapsables  (ownership remains)
	 * @return A vector containing all titles and references
	 */
	std::vector<std::pair<QString, Collapsable *>> popAll();

	/**
	 * @brief Returns the number of elements
	 * @return The number of elements
	 */
	std::size_t size() const
	{
		return elements_.size();
	}

      private:
	/**
	 * @brief Storage for all elements
	 */
	std::map<Handle, Collapsable *> elements_;

	/**
	 * @brief Layout for all elements
	 */
	util::ObserverPtr<QVBoxLayout> layout_;
}; // Accordion
}
} // end namespaces qtutil, cvv
#endif // CVVISUAL_ACCORDION_HPP
