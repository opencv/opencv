#ifndef CVVISUAL_DEBUG_SHOW_IMAGE_HPP
#define CVVISUAL_DEBUG_SHOW_IMAGE_HPP

#include <string>

#include "opencv2/core/core.hpp"

#include "call_meta_data.hpp"
#include "debug_mode.hpp"

namespace cvv
{

namespace impl
{
// implementation outside API
void showImage(cv::InputArray img, const CallMetaData &data,
               const char *description, const char *view);
} // namespace impl

#ifdef CVVISUAL_DEBUGMODE
static inline void showImage(cv::InputArray img,
                             impl::CallMetaData metaData = impl::CallMetaData(),
                             const char *description = nullptr,
                             const char *view = nullptr)
{
	if (debugMode())
	{
		impl::showImage(img, metaData, description, view);
	}
}
static inline void showImage(cv::InputArray img, impl::CallMetaData metaData,
                             const ::std::string &description,
                             const ::std::string &view = "")
{
	if (debugMode())
	{
		impl::showImage(img, metaData, description.c_str(),
		                view.c_str());
	}
}
#else
/**
 * Use the debug-framework to show a single image.
 */
static inline void showImage(cv::InputArray,
                             impl::CallMetaData = impl::CallMetaData(),
                             const char * = nullptr, const char * = nullptr)
{
}
/**
 * Dito.
 */
static inline void showImage(cv::InputArray, impl::CallMetaData,
                             const ::std::string &, const ::std::string &)
{
}
#endif

} // namespace cvv

#endif
