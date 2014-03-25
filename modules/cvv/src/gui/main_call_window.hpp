#ifndef CVVISUAL_MAINCALLWINDOW_HPP
#define CVVISUAL_MAINCALLWINDOW_HPP

#include <memory>

#include <QCloseEvent>

#include "call_window.hpp"
#include "overview_panel.hpp"
#include "../controller/view_controller.hpp"
#include "../util/util.hpp"

namespace cvv
{

namespace controller
{
class ViewController;
}

namespace gui
{

class OverviewPanel;

/**
 * @brief A call window also inheriting the overview panel.
 */
class MainCallWindow : public CallWindow
{

	Q_OBJECT

      public:
	/**
	 * @brief Constructs a new main call window.
	 * @param controller view controller inheriting this main window
	 * @param id id of this main window
	 * @param ovPanel inherited overview panel
	 */
	MainCallWindow(util::Reference<controller::ViewController> controller,
	               size_t id, OverviewPanel *ovPanel);

	~MainCallWindow()
	{
	}

	/**
	 * @brief Show the overview tab.
	 */
	void showOverviewTab();

	/**
	 * @brief Hides the close window.
	 */
	void hideCloseWindow();

      protected:
	void closeEvent(QCloseEvent *event);

      private:
	OverviewPanel *ovPanel;
};
}
}
#endif
