// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"

#include <vector>

#include <opencv2/core/utils/logger.defines.hpp>
#undef CV_LOG_STRIP_LEVEL
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE + 1
#include "opencv2/core/utils/logger.hpp"
#include "opencv2/core/utils/filesystem.hpp"
#include "opencv2/core/utils/filesystem.private.hpp"

namespace cv { namespace samples {

static cv::Ptr< std::vector<cv::String> > g_data_search_path;
static cv::Ptr< std::vector<cv::String> > g_data_search_subdir;

static std::vector<cv::String>& _getDataSearchPath()
{
    if (g_data_search_path.empty())
        g_data_search_path.reset(new std::vector<cv::String>());
    return *(g_data_search_path.get());
}

static std::vector<cv::String>& _getDataSearchSubDirectory()
{
    if (g_data_search_subdir.empty())
    {
        g_data_search_subdir.reset(new std::vector<cv::String>());
        g_data_search_subdir->push_back("samples/data");
        g_data_search_subdir->push_back("data");
        g_data_search_subdir->push_back("");
    }
    return *(g_data_search_subdir.get());
}


CV_EXPORTS void addSamplesDataSearchPath(const cv::String& path)
{
    if (utils::fs::isDirectory(path))
        _getDataSearchPath().push_back(path);
}
CV_EXPORTS void addSamplesDataSearchSubDirectory(const cv::String& subdir)
{
    _getDataSearchSubDirectory().push_back(subdir);
}

cv::String findFile(const cv::String& relative_path, bool required, bool silentMode)
{
#if OPENCV_HAVE_FILESYSTEM_SUPPORT
    CV_LOG_DEBUG(NULL, cv::format("cv::samples::findFile('%s', %s)", relative_path.c_str(), required ? "true" : "false"));
    cv::String result = cv::utils::findDataFile(relative_path,
                                                "OPENCV_SAMPLES_DATA_PATH",
                                                &_getDataSearchPath(),
                                                &_getDataSearchSubDirectory());
    if (result != relative_path && !silentMode)
    {
        CV_LOG_WARNING(NULL, "cv::samples::findFile('" << relative_path << "') => '" << result << "'");
    }
    if (result.empty() && required)
        CV_Error(cv::Error::StsError, cv::format("OpenCV samples: Can't find required data file: %s", relative_path.c_str()));
    return result;
#else
    CV_UNUSED(relative_path);
    CV_UNUSED(required);
    CV_UNUSED(silentMode);
    CV_Error(Error::StsNotImplemented, "File system support is disabled in this OpenCV build!");
#endif
}


}} // namespace
