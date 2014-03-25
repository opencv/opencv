#ifndef CVVISUAL_EXTENSION_API_HPP
#define CVVISUAL_EXTENSION_API_HPP

#include <opencv2/core/core.hpp>

#include <QString>
#include <QWidget>

#include "../impl/call.hpp"
#include "../controller/view_controller.hpp"
#include "../view/filter_view.hpp"
#include "../gui/match_call_tab.hpp"
#include "../gui/filter_call_tab.hpp"
#include "../qtutil/filterselectorwidget.hpp"

namespace cvv
{
namespace extend
{

/**
 * @brief Introduces a new filter-view.
 * @param name of the new FilterView.
 * @tparam FView A FilterView. Needs to have a constructor of the form
 * FView(const cvv::impl::FilterCall&, QWidget*).
 */
template <class FView> void addFilterView(const QString name)
{
	cvv::gui::FilterCallTab::registerFilterView<FView>(name);
}

/**
 * @brief Introduces a new match-view.
 * @param name of the new MatchView.
 * @tparam MView A MatchView. Needs to have a constructor of the form
 * MView(const cvv::impl::MatchCall&, QWidget*).
 */
template <class MView> void addMatchView(const QString name)
{
	cvv::gui::MatchCallTab::registerMatchView<MView>(name);
}

using TabFactory = controller::TabFactory;
/**
 * @brief Introduces a new call-type.
 * @param factory A function that recieves a reference to a call and should
 * return the appropriate
 * window.
 */
void addCallType(const QString name, TabFactory factory);

template <std::size_t In, std::size_t Out, class Filter>
/**
 * @brief Introduces a new filter for the filter-selector-widget.
 */
bool registerFilter(const QString &name)
{
	return cvv::qtutil::registerFilter<In, Out, Filter>(name);
}
}
} // namespaces cvv::extend

#endif
