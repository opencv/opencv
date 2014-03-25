#include <sstream>

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QApplication>

#include "acctester.hpp"

/**
 * @brief
 * - a window with serval buttons ("pfront", "pback", "insert elem at pos 5",
 * "remove last element inserted at 5", "clear", "hideAll", "hide5", "showAll",
 * "show5", "collapseAll", "collapse5", "expandAll", "expand5") and an empty area
 * - "pfront" will insert a collapsable with text and title 0 at the beginning
 * - "pback" will insert a collapsable with text and title end at the end
 * - "clear" will delete all elements
 * - "hideAll" will hide all collapsables
 * - "showAll" will show all collapsables
 * - "collapseAll" will collapse all collapsables
 * - "expandAll" will expand all collapsables
 * - "insert elem at pos 5" will insert a collapsable (elem5) with text and
 * title 5 at position 5
 * - "remove last element inserted at 5" will remove the last elem5
 * - "hide5" will hide the last elem5
 * - "show5" will show the last elem5
 * - "collapse5" will collapse the last elem5
 * - "expand5" will expand the last elem5
 * - if there is no elem5 a window with following text will pop up:
	no last element inserted at 5. (maybe already deleted)
 * - if "insert elem at pos 5" is used 2 times and "remove last element inserted at 5"
 *  is calles there is NO elem5 (the second is deleted, the first one is not set as elem5)
 *
 * - beginning /end / pos 5 refer to the empty area at the bottom
 * - all buttons behave as described
 * - the area displays scrollbars if needed
 */
int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	Acctester w{};

	w.show();
	return a.exec();
}
