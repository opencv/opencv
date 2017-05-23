#include "filter_call.hpp"

#include "data_controller.hpp"

#include "../util/util.hpp"

namespace cvv
{
namespace impl
{

FilterCall::FilterCall(cv::InputArray in, cv::InputArray out,
                       impl::CallMetaData data, QString type,
                       QString description, QString requestedView)
    : Call{ data,                   std::move(type),
	    std::move(description), std::move(requestedView) },
      input_{ in.getMat().clone() }, output_{ out.getMat().clone() }
{
}

const cv::Mat &FilterCall::matrixAt(size_t index) const
{
	switch (index)
	{
	case 0:
		return original();
	case 1:
		return result();
	default:
		throw std::out_of_range{ "" };
	}
}

void debugFilterCall(cv::InputArray original, cv::InputArray result,
                     const CallMetaData &data, const char *description,
                     const char *view, const char *filter)
{
	dataController().addCall(util::make_unique<FilterCall>(
	    original, result, data, filter,
	    description ? QString::fromLocal8Bit(description)
	                : QString{ "<no description>" },
	    view ? QString::fromLocal8Bit(view) : QString{}));
}
}
} // namespaces cvv::impl
