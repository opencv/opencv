#include "opencv2/cvv/final_show.hpp"

#include "data_controller.hpp"

namespace cvv
{
namespace impl
{

void finalShow()
{
	auto &controller = impl::dataController();
	if (controller.numCalls() != 0)
	{
		controller.lastCall();
	}
	impl::deleteDataController();
}
}
} // namespaces cvv::impl
