// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_UTILS_FILESYSTEM_HPP
#define OPENCV_UTILS_FILESYSTEM_HPP

namespace cv { namespace utils { namespace fs {


CV_EXPORTS bool exists(const cv::String& path);
CV_EXPORTS bool isDirectory(const cv::String& path);


CV_EXPORTS cv::String getcwd();


CV_EXPORTS bool createDirectory(const cv::String& path);
CV_EXPORTS bool createDirectories(const cv::String& path);

#ifdef __OPENCV_BUILD
// TODO
//CV_EXPORTS cv::String getTempDirectory();

/**
 * @brief Returns directory to store OpenCV cache files
 * Create sub-directory in common OpenCV cache directory if it doesn't exist.
 * @param sub_directory_name name of sub-directory. NULL or "" value asks to return root cache directory.
 * @param configuration_name optional name of configuration parameter name which overrides default behavior.
 * @return Path to cache directory. Returns empty string if cache directories support is not available. Returns "disabled" if cache disabled by user.
 */
CV_EXPORTS cv::String getCacheDirectory(const char* sub_directory_name, const char* configuration_name = NULL);

#endif

}}} // namespace

#endif // OPENCV_UTILS_FILESYSTEM_HPP
