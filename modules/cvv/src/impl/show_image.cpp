#include "opencv2/cvv/show_image.hpp"

#include "opencv2/cvv/call_meta_data.hpp"
#include "single_image_call.hpp"

namespace cvv
{
namespace impl
{

void showImage(cv::InputArray img, const CallMetaData &data,
               const char *description, const char *view)
{
	debugSingleImageCall(img, data, description, view, "singleImage");
}
}
} // namespaces
