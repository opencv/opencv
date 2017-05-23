#ifndef CVVISUAL_VIEWCONTROLLER_HPP
#define CVVISUAL_VIEWCONTROLLER_HPP

#include <vector>
#include <algorithm>
#include <iostream>
#include <map>
#include <memory>

#include <functional>
#include <utility>
#include <QString>

#include "../util/util.hpp"
#include "../impl/call.hpp"
#include "../gui/call_window.hpp"
#include "../gui/call_tab.hpp"


namespace cvv
{

namespace gui
{
class CallTab;
class CallWindow;
class MainCallWindow;
class OverviewPanel;
}

namespace controller
{

/**
 * @brief Modes that this cvv application can be running in.
 */
enum class Mode
{
	/**
         * @brief The normal mode.
         */
	NORMAL = 0,
	/**
         * @brief The cvv UI is hidden.
         */
	HIDE = 1,
	/**
         * @brief The cvv UI stops only at the final call
         * The final call is the call which is called after `cvv::finalShow()`)
         */
	FAST_FORWARD = 2
};

class ViewController;

/**
 * @brief Typedef for a function that creates a CallTab from a impl::Call.
 */
using TabFactory =
    std::function<std::unique_ptr<gui::CallTab>(util::Reference<impl::Call>)>;

/**
 * @brief Controlls the windows, call tabs and the event fetch loop.
 * Its the layer between the low level model (aka DataController) an the high
 * level GUI (aka CallTab, OverviewPanel, ...).
 */
class ViewController
{
      public:
	/**
	 * @brief The default contructor for this class.
	 */
	ViewController();

	/**
	 * @brief Clean up.
	 */
	~ViewController();

	/**
	 * @brief Adds the new call tab type.
	 * @param typeName name of the new type
	 * @param constr function constructing an instance of this  call tab
	 * type
	 * @return an instance of the new call tab type
	 */
	static void addCallType(const QString typeName, TabFactory constr);

	/**
	 * @brief Adds a new call and shows it in the overview table.
	 * @param data new call (data)
	 */
	void addCall(util::Reference<impl::Call> data);

	/**
	 * @brief Execute the Qt event loop.
	 */
	void exec();

	/**
	 * @brief Get the call with the given id.
	 * @param id given id
	 * @return call with the given id
	 */
	impl::Call &getCall(size_t id);

	/**
	 * @brief Get the current setting [key] in the given scope.
	 * Please use `setDefaultSetting` to set a default value that's other
	 * than
	 * an empty QString.
	 * @param scope given scope (e.g. 'Overview')
	 * @param key settings key (e.g. 'autoOpenTabs')
	 * @return settings string
	 */
	QString getSetting(const QString &scope, const QString &key);

	/**
	 * @brief Get the inherited call windows with tabs.
	 * @return the inherited CallWindows
	 */
	std::vector<util::Reference<gui::CallWindow>> getTabWindows();

	/**
	 * @brief Get the inherited main window.
	 * @return the inherited main window
	 */
	util::Reference<gui::MainCallWindow> getMainWindow();

	/**
	 * @brief Move the call tab with the given id to a new window.
	 * @param tabId given call tab id
	 */
	void moveCallTabToNewWindow(size_t tabId);

	/**
	 * @brief Move the given call tab to the given window.
	 * @param tabId id of the given call tab
	 * @param windowId id of the given window (0 is the main window)
	 */
	void moveCallTabToWindow(size_t tabId, size_t windowId);

	/**
	 * @brief Removes the call tab with the given id.
	 * @param tabId given id
	 * @param deleteCall if deleteCall and deleteIt are true, it also
	 * deletes the proper Call
	 */
	void removeCallTab(size_t tabId, bool deleteIt = true,
	                   bool deleteCall = false, bool updateUI = true);

	/**
	 * @brief Opens the users default browser with the topic help page.
	 * Current URL: cvv.mostlynerdless.de/help.php?topic=[topic]
	 *
	 * Topics can be added via appending the doc/topics.yml file.
	 *
	 * @param topic help topic
	 */
	void openHelpBrowser(const QString &topic);

	/**
	 * @brief Resume the execution of the calling program.
	 */
	void resumeProgramExecution();

	/**
	 * @brief Set the default setting for a given stettings key and scope.
	 * It doesn't override existing settings.
	 * @param scope given settings scope
	 * @param key given settings key
	 * @param value default value of the setting
	 */
	void setDefaultSetting(const QString &scope, const QString &key,
	                       const QString &value);

	/**
	 * @brief Set the setting for a given stettings key and scope.
	 * @param scope given settings scope
	 * @param key given settings key
	 * @param value new value of the setting
	 */
	void setSetting(const QString &scope, const QString &key,
	                const QString &value);

	/**
	 * @brief Show the given call tab and bring it's window to the front.
	 * @note It's not guaranteed that it really brings the tabs' window to the front.
	 * @param tabId id of the given call tab
	 */
	void showCallTab(size_t tabId);

	/**
	 * @brief Shows the tab and opens it if neccessary.
	 * @param tabId id of the tab
	 */
	void showAndOpenCallTab(size_t tabId);
	
	/**
	 * @brief Opens the tab it if neccessary.
	 * @param tabId id of the tab
	 */
	void openCallTab(size_t tabId);

	/**
	 * @brief Show the overview tab (and table) and bring it's window to the
	 * front.
	 * @note The latter is not guaranteed.
	 */
	void showOverview();

	/**
	 * @brief Get the window in which the given tab lays currently.
	 * @param tabId id of the given call tab
	 * @return current window
	 */
	gui::CallWindow *getCurrentWindowOfTab(size_t tabId);

	/**
	 * @brief Returns the call tab with the given id and constructs it if
	 * doesn't exit.
	 * @param tabId given id
	 * @return call tab with given id
	 */
	gui::CallTab *getCallTab(size_t tabId);

	/**
	 * @brief Remove the window from the internal data structures.
	 * @param windowId id of the window
	 * @note Only call this method if you now the implacations of deleting
	 * the window.
	 */
	void removeWindowFromMaps(size_t windowId);

	/**
	 * @brief Shows an "Exit program" button on each window.
	 */
	void showExitProgramButton();

	/**
	 * @brief Removes the empty windows.
	 * @note It's safer to call the removeEmptyWindowsWithDelay method
	 * instead.
	 */
	void removeEmptyWindows();

	/**
	 * @brief Removes the empty windows with a small delay.
	 */
	void removeEmptyWindowsWithDelay();

	/**
	 * @brief Checks whether or not is useful to call the
	 * removeEmptyWindows() method.
	 * @return Is is useful to call the removeEmptyWindows() method?
	 * @note Please don't call this method outside a periodcally called
	 * method.
	 */
	bool shouldRunRemoveEmptyWindows();

	/**
	 * @brief Set the mode that this application is running in.
	 * @param newMode mode to be set
	 */
	void setMode(Mode newMode);

	/**
	 * @brief Returns the mode this program is running in.
	 * @return the current mode, NROMAL, HIDE or FAST_FORWARD
	 */
	Mode getMode();

	/**
	 * @brief Checks whether or not the `cvv::finalCall()` method has been
	 * called?
	 * @return Has the `cvv::finalCall()` method been called?
	 */
	bool hasFinalCall();

      private:
	static std::map<QString, TabFactory> callTabType;

	std::map<size_t, std::unique_ptr<gui::CallWindow>> windowMap{};
	gui::MainCallWindow *mainWindow;

	std::map<size_t, std::unique_ptr<gui::CallTab>> callTabMap{};
	gui::OverviewPanel *ovPanel;
	bool doesShowExitProgramButton = false;
	/**
	 * @brief Counter == 0 <=> you should run `removeEmptyWindows()`.
	 */
	bool shouldRunRemoveEmptyWindows_ = true;

	Mode mode = Mode::NORMAL;

	bool ownsQApplication = false;

	size_t max_window_id = 0;

	bool hasCall(size_t id);

	void updateMode();

	void hideAll();
};
}
}

#endif
