#ifndef CVVISUAL_FILTER_CALL_HPP
#define CVVISUAL_FILTER_CALL_HPP

#include <QString>

#include "call.hpp"

#include "opencv2/core/core.hpp"

namespace cvv
{
namespace impl
{

/**
 * All data of a filter-call: Location, original image and result.
 */
class FilterCall : public Call
{
      public:
	/**
	 * @brief Constructs a FilterCall.
	 */
	FilterCall(cv::InputArray in, cv::InputArray out,
	           impl::CallMetaData data, QString type, QString description,
	           QString requestedView);

	size_t matrixCount() const override
	{
		return 2;
	}
	const cv::Mat &matrixAt(size_t index) const override;

	/**
	 * @returns the original image
	 */
	const cv::Mat &original() const
	{
		return input_;
	}
	/**
	 * @returns the filtered image
	 */
	const cv::Mat &result() const
	{
		return output_;
	}

      private:
	// TODO: in case we REALLY want to support several input-images: make
	// this a std::vector
	// TODO: those are typedefs for references, make it clean:
	cv::Mat input_;
	cv::Mat output_;
};

/**
 * Constructs a FilterCall and adds it to the global data-controller.
 */
void debugFilterCall(cv::InputArray original, cv::InputArray result,
                     const CallMetaData &data, const char *description,
                     const char *view, const char *filter);
}
} // namespaces

#endif
