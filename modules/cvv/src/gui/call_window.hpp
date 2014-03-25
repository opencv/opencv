#ifndef CVVISUAL_CALLWINDOW_HPP
#define CVVISUAL_CALLWINDOW_HPP

#include <vector>
#include <map>

#include <QTabWidget>
#include <QMainWindow>
#include <QString>
#include <vector>
#include <QLabel>
#include <QKeyEvent>
#include <QPoint>
#include <QCloseEvent>
#include <QPushButton>

#include "call_tab.hpp"
#include "../controller/view_controller.hpp"
#include "../util/util.hpp"
#include "tabwidget.hpp"

namespace cvv
{

namespace controller
{
class ViewController;
}

namespace gui
{

/**
 * @brief Window inheriting some call tabs with in a tab widget.
 */
class CallWindow : public QMainWindow
{

	Q_OBJECT

public:
	/**
	 * @brief Contructs a new call window.
	 * @param controller view controller that this window belongs to
	 * @param id id of the window
	 */
	CallWindow(util::Reference<controller::ViewController> controller,
	           size_t id);

	/**
	 * @brief Shows an "Exit program" button.
	 */
	void showExitProgramButton();

	/**
	 * @brief Add a new tab to the inherited tab widget.
	 * @param tab new tab
	 */
	void addTab(CallTab *tab);

	/**
	 * @brief Get the id of this window.
	 * @return id of this window.
	 */
	size_t getId();

	/**
	 * @brief Remove the given tab from this window.
	 * @param given tab to remove
	 */
	void removeTab(CallTab *tab);

	/**
	 * @brief Remove the given tab from this window.
	 * @param id of the given tab
	 */
	void removeTab(size_t tabId);

	/**
	 * @brief Show the given tab.
	 * @param given tab
	 */
	void showTab(CallTab *tab);

	/**
	 * @brief Show the given tab.
	 * @param id of the given tab
	 */
	void showTab(size_t tabId);
	
	/**
	 * @brief Examines whether or not the given is inherited in this window.
	 * @param id of the given tab
	 */
	bool hasTab(size_t tabId);

	/**
	 * @brief Returns the number of tabs shown in this window.
	 * @return number of tabs
	 */
	size_t tabCount();

	/**
	 * @brief Returns the ids of the available call tabs.
	 * @return available call tabs' ids
	 */
	std::vector<size_t> getCallTabIds();

public slots:
	
	/**
	 * @brief Update the left footer with the given text.
	 * @param newText given text
	 */
	void updateLeftFooter(QString newText);

	/**
	 * @brief Update the right footer with the given text.
	 * @param newText given text
	 */
	void updateRightFooter(QString newText);

private slots:
	void contextMenuRequested(const QPoint &location);

	void contextMenuAction(QAction *action);

	void tabCloseRequested(int index);

	void step();

	void fastForward();

	void closeApp();

protected:
	size_t id;
	util::Reference<controller::ViewController> controller;
	TabWidget *tabWidget;
	QMainWindow *window;
	QPushButton *closeButton;
	QPushButton *stepButton;
	QPushButton *fastForwardButton;
	std::map<size_t, CallTab *> tabMap;
	QLabel *leftFooter;
	QLabel *rightFooter;
	int currentContextMenuTabId = -1;
	int tabOffset = 0;

	void initMenu();

	void initTabs();

	void initFooter();

	void closeEvent(QCloseEvent *event);

	size_t getCallTabIdByTabIndex(int index);

	bool hasTabAtIndex(int index);
};
}
}

#endif
