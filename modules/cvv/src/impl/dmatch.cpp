#include "opencv2/cvv/dmatch.hpp"

#include "opencv2/cvv/call_meta_data.hpp"
#include "match_call.hpp"

namespace cvv
{
namespace impl
{

void debugDMatch(cv::InputArray img1, std::vector<cv::KeyPoint> keypoints1,
                 cv::InputArray img2, std::vector<cv::KeyPoint> keypoints2,
                 std::vector<cv::DMatch> matches, const CallMetaData &data,
                 const char *description, const char *view,
                 bool useTrainDescriptor)
{
	debugMatchCall(img1, std::move(keypoints1), img2, std::move(keypoints2),
	               std::move(matches), data, description, view,
	               useTrainDescriptor);
}
}
} // namespaces
