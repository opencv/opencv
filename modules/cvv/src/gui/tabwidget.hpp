#ifndef CVVISUAL_TABWIDGET
#define CVVISUAL_TABWIDGET

#include <QTabWidget>
#include <QTabBar>


namespace cvv
{
namespace gui
{

/**
 * @brief A simple to QTabWidget Subclass, enabling the access to protected
 * members.
 */
class TabWidget : public QTabWidget
{

      public:
	/**
	 * @brief Constructor of this class.
	 */
	TabWidget(QWidget *parent) : QTabWidget(parent)
	{
	};

	/**
	 * @brief Returns the shown tab bar.
	 * This method helps to access the member tabBar which has by default
	 * only a protected setter.
	 * @return shown tab bar.
	 */
	QTabBar *getTabBar() const
	{
		return tabBar();
	}
};
}
}

#endif
