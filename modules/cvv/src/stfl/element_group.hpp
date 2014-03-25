#ifndef CVVISUAL_ELEMENTGROUP_HPP
#define CVVISUAL_ELEMENTGROUP_HPP

#include <functional>
#include <QList>
#include <QStringList>
#include <QString>
#include <stdint.h>
#include <stdexcept>


namespace cvv
{
namespace stfl
{

/**
 * @brief A group of elements with a title.
 */
template <class Element> class ElementGroup
{
public:
	/**
	 * @brief Contructs an empty ElementGroup.
	 */
	ElementGroup()
	{
	}

	/**
	 * @brief Constructs a new ElementGroup
	 * @param _titles title of this group, consisting of several sub titles
	 * @param _elements elements of this group
	 */
	ElementGroup(QStringList _titles, QList<Element> &_elements)
	    : titles{ _titles }, elements{ _elements }
	{
	}
	/**
	 * @brief Checks whether or not this group contains the element.
	 * @param element element to be checked
	 * @return does this group contain the given element?
	 */
	bool contains(const Element &element)
	{
		return this->elements.contains(element);
	}

	/**
	 * @brief Return the inherited elements.
	 * @return the interited elements
	 */
	QList<Element> getElements()
	{
		return this->elements;
	}

	/**
	 * @brief Returns the number of elements in this group.
	 * @return number of elements in this group
	 */
	size_t size() const
	{
		return this->elements.size();
	}

	/**
	 * @brief Returns the title (consisting of sub titles).
	 * @return the group title
	 */
	QStringList getTitles() const
	{
		return this->titles;
	}

	/**
	 * @brief Checks whether this an the given element group have the same
	 * titles.
	 * @param other given other element group
	 * @return Does this element group and the given have the same titles.
	 */
	bool hasSameTitles(ElementGroup<Element> &other)
	{
		for (auto title : other.getTitles())
		{
			if (!titles.contains(title))
			{
				return false;
			}
		}
		return other.getTitles().size() == titles.size();
	}

	/**
	 * @brief Checks whether this an the given element have the same list of
	 * elements.
	 * @param other given other element group
	 * @param elementCompFunc element comparison function that gets two
	 * elements passed
	 * and returns true if both can be considered equal
	 * @return Does this element group and the given have the same list of
	 * elements.
	 */
	bool
	hasSameElementList(ElementGroup<Element> &other,
	                   std::function<bool(const Element &, const Element &)>
	                       elementCompFunc)
	{
		if (other.getElements().size() != elements.size())
		{
			return false;
		}
		for (int i = 0; i < elements.size(); i++)
		{
			if (!elementCompFunc(elements[i], other.get(i)))
			{
				return false;
			}
		}
		return true;
	}

	/**
	 * @brief Get the element at the given index (in this group).
	 *
	 * @param index given index
	 * @return element at the given index
	 * @throws std::invalid_argument if no such element exists
	 */
	Element get(size_t index)
	{
		if (index >= size())
		{
			throw std::invalid_argument{
				"there is no call with this id"
			};
		}
		return this->elements[index];
	}

	/**
	 * @brief Remove the element at the given index.
	 * @param index given element index
	 */
	void removeElement(size_t index)
	{
		if (index < static_cast<size_t>(elements.size()))
		{
			elements.removeAt(index);
		}
	}

	/**
	 * @brief Sets the inhereted elements.
	 * @param newElements new elements of this group
	 */
	void setElements(QList<Element> newElements)
	{
		elements = newElements;
	}

private:
	QStringList titles;
	QList<Element> elements;
};
}
}
#endif
