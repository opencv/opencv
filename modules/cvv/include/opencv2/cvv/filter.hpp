#ifndef CVVISUAL_DEBUG_FILTER_HPP
#define CVVISUAL_DEBUG_FILTER_HPP

#include <string>

#include "opencv2/core/core.hpp"

#include "call_meta_data.hpp"
#include "debug_mode.hpp"

namespace cvv
{

namespace impl
{
// implementation outside API
void debugFilter(cv::InputArray original, cv::InputArray result,
                 const CallMetaData &data, const char *description,
                 const char *view);
} // namespace impl

#ifdef CVVISUAL_DEBUGMODE
static inline void
debugFilter(cv::InputArray original, cv::InputArray result,
            impl::CallMetaData metaData = impl::CallMetaData(),
            const char *description = nullptr, const char *view = nullptr)
{
	if (debugMode())
	{
		impl::debugFilter(original, result, metaData, description,
		                  view);
	}
}
static inline void debugFilter(cv::InputArray original, cv::InputArray result,
                               impl::CallMetaData metaData,
                               const ::std::string &description,
                               const ::std::string &view = "")
{
	if (debugMode())
	{
		impl::debugFilter(original, result, metaData,
		                  description.c_str(), view.c_str());
	}
}
#else
/**
 * @brief Use the debug-framework to compare two images (from which the second
 * is intended to be the result of
 * a filter applied to the first).
 */
static inline void debugFilter(cv::InputArray, cv::InputArray,
                               impl::CallMetaData = impl::CallMetaData(),
                               const char * = nullptr, const char * = nullptr)
{
}

/**
 * Dito.
 */
static inline void debugFilter(cv::InputArray, cv::InputArray,
                               impl::CallMetaData, const ::std::string &,
                               const ::std::string &)
{
}
#endif

} // namespace cvv

#endif
