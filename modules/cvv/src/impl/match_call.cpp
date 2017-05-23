#include "match_call.hpp"

#include <stdexcept>

#include <QString>

#include "data_controller.hpp"

#include "../util/util.hpp"

namespace cvv
{
namespace impl
{

MatchCall::MatchCall(cv::InputArray img1, std::vector<cv::KeyPoint> keypoints1,
                     cv::InputArray img2, std::vector<cv::KeyPoint> keypoints2,
                     std::vector<cv::DMatch> matches, impl::CallMetaData data,
                     QString type, QString description, QString requestedView,
                     bool useTrainDescriptor)
    : Call{ data,                   std::move(type),
	    std::move(description), std::move(requestedView) },
      img1_{ img1.getMat().clone() }, keypoints1_{ std::move(keypoints1) },
      img2_{ img2.getMat().clone() }, keypoints2_{ std::move(keypoints2) },
      matches_{ std::move(matches) }, usesTrainDescriptor_{ useTrainDescriptor }
{
}

const cv::Mat &MatchCall::matrixAt(size_t index) const
{
	switch (index)
	{
	case 0:
		return img1();
	case 1:
		return img2();
	default:
		throw std::out_of_range{ "" };
	}
}

void debugMatchCall(cv::InputArray img1, std::vector<cv::KeyPoint> keypoints1,
                    cv::InputArray img2, std::vector<cv::KeyPoint> keypoints2,
                    std::vector<cv::DMatch> matches, const CallMetaData &data,
                    const char *description, const char *view,
                    bool useTrainDescriptor)
{
	dataController().addCall(util::make_unique<MatchCall>(
	    img1, std::move(keypoints1), img2, std::move(keypoints2),
	    std::move(matches), data, "match",
	    description ? QString::fromLocal8Bit(description)
	                : QString{ "<no description>" },
	    view ? QString::fromLocal8Bit(view) : QString{},
	    useTrainDescriptor));
}
}
} // namespaces cvv::impl
