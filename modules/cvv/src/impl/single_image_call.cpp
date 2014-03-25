#include "single_image_call.hpp"

#include <QString>

#include "data_controller.hpp"

#include "../util/util.hpp"

namespace cvv
{
namespace impl
{

SingleImageCall::SingleImageCall(cv::InputArray img, impl::CallMetaData data,
                                 QString type, QString description,
                                 QString requestedView)
    : Call{ data,                   std::move(type),
	    std::move(description), std::move(requestedView) },
      img{ img.getMat().clone() }
{
}

const cv::Mat &SingleImageCall::matrixAt(size_t index) const
{
	if (index)
	{
		throw std::out_of_range{ "" };
	}
	return img;
}

void debugSingleImageCall(cv::InputArray img, const CallMetaData &data,
                          const char *description, const char *view,
                          const char *filter)
{
	dataController().addCall(util::make_unique<SingleImageCall>(
	    img, data, filter, description ? QString::fromLocal8Bit(description)
	                                   : QString{ "<no description>" },
	    view ? QString::fromLocal8Bit(view) : QString{}));
}
}
} // namespaces cvv::impl
