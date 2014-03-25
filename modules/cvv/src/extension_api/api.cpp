#include "api.hpp"

#include "../gui/filter_call_tab.hpp"
#include "../gui/match_call_tab.hpp"

namespace cvv
{
namespace extend
{

void addCallType(const QString name, TabFactory factory)
{
	controller::ViewController::addCallType(name, factory);
}
}
} // namespaces cvv::extend
