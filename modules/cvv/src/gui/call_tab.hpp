#ifndef CVVISUAL_CALL_TAB_HPP
#define CVVISUAL_CALL_TAB_HPP

#include <QString>
#include <QWidget>

#include "../util/util.hpp"

namespace cvv
{
namespace gui
{

/**
 * @brief Super class of the inner part of a tab or window.
 * A call tab.
 * The inner part of a tab or a window.
 * Super class for actual call tabs containing views.
 */
class CallTab : public QWidget
{
	Q_OBJECT
      public:
	/**
	 * @brief Returns the name of this tab.
	 * @return current name
	 */
	const QString getName() const
	{
		return name;
	}

	/**
	 * @brief Sets the name of this tab.
	 * @param name new name
	 */
	void setName(const QString &newName)
	{
		name = newName;
	}

	/**
	 * @brief Returns the of this CallTab.
	 * @return the ID of the CallTab
	 * (ID is equal to the ID of the associated call in derived classes)
	 */
	virtual size_t getId() const
	{
		return 0;
	}

      private:
	QString name;
};
}
} // namespaces

#endif
