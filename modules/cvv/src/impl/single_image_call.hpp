#ifndef CVVISUAL_SINGLE_IMAGE_CALL_HPP
#define CVVISUAL_SINGLE_IMAGE_CALL_HPP

#include "call.hpp"

#include <QString>

#include "opencv2/core/core.hpp"

namespace cvv
{
namespace impl
{

/**
 * All data of a filter-call: Location, original image and result.
 */
class SingleImageCall : public Call
{
      public:
	/**
	 * @brief Constructs a SingleImageCall.
	 */
	SingleImageCall(cv::InputArray img, impl::CallMetaData data,
	                QString type, QString description,
	                QString requestedView);

	size_t matrixCount() const override
	{
		return 1;
	}
	const cv::Mat &matrixAt(size_t index) const override;

	/**
	 * @returns the original image
	 */
	const cv::Mat &mat() const
	{
		return img;
	}

      private:
	cv::Mat img;
};

/**
 * Constructs a SingleImageCall and adds it to the global data-controller.
 */
void debugSingleImageCall(cv::InputArray img, const CallMetaData &data,
                          const char *description, const char *view,
                          const char *filter);
}
} // namespaces

#endif
