#ifndef CVVISUAL_FILTER_CALL_TAB_HPP
#define CVVISUAL_FILTER_CALL_TAB_HPP

#include <QString>
#include <QWidget>

#include "multiview_call_tab.hpp"
#include "../view/filter_view.hpp"
#include "../impl/filter_call.hpp"

namespace cvv
{
namespace gui
{

/** Filter Call Tab.
 * @brief Inner part of a tab, contains a FilterView.
 * The inner part of a tab or window
 * containing a FilterView.
 * Allows to switch views and to access the help.
 */
class FilterCallTab
    : public MultiViewCallTab<cvv::view::FilterView, cvv::impl::FilterCall>
{
	Q_OBJECT

      public:
	/**
	 * @brief Short constructor named after the Call, using the requested View
	 * from the Call or, if no or invalid request, default view.
	 * Initializes the FilterCallTab with the requested or default view and names it
	 * after the associated FilterCall.
	 * @param filterCall - the FilterCall containing the information to be
	 * visualized.
	 */
	FilterCallTab(const cvv::impl::FilterCall &filterCall)
	    : FilterCallTab{
		      filterCall, filterCall.requestedView()
	      }
	{
	}

	/**
	 * @brief Constructor with possibility to select view.
	 * Note that the default view is still created first.
	 * @param call - the MatchCall containing the information to be
	 * visualized.
	 * @param filterViewId - ID of the View to be set up. If a view of this name does
	 * not exist, the default view will be used.
	 */
	FilterCallTab(const cvv::impl::FilterCall &filterCall, const QString& filterViewId)
	    : MultiViewCallTab<cvv::view::FilterView, cvv::impl::FilterCall>{
		      filterCall, filterViewId, QString{ "default_filter_view" }, QString{ "DefaultFilterView" }
	      }
	{
	}

	~FilterCallTab()
	{
	}

	/**
	 * @brief Register the template class to the map of FilterViews.
	 * View needs to offer a constructor of the form View(const
	 * cvv::impl::FilterCall&, QWidget*).
	 * @param name to register the class under.
	 * @tparam View - Class to register.
	 * @return true when the view was registered and false when the name was
	 * already taken.
	 */
	template <class View>
	static bool registerFilterView(const QString &name)
	{
		return registerView<View>(name);
	}
};
}
} // namespaces

#endif
