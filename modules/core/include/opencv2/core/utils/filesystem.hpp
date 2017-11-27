// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_UTILS_FILESYSTEM_HPP
#define OPENCV_UTILS_FILESYSTEM_HPP

namespace cv { namespace utils { namespace fs {


CV_EXPORTS bool exists(const cv::String& path);
CV_EXPORTS bool isDirectory(const cv::String& path);

CV_EXPORTS void remove_all(const cv::String& path);


CV_EXPORTS cv::String getcwd();

/** Join path components */
CV_EXPORTS cv::String join(const cv::String& base, const cv::String& path);

/**
 * Generate a list of all files that match the globbing pattern.
 *
 * Result entries are prefixed by base directory path.
 *
 * @param directory base directory
 * @param pattern filter pattern (based on '*'/'?' symbols). Use empty string to disable filtering and return all results
 * @param[out] result result of globing.
 * @param recursive scan nested directories too
 * @param includeDirectories include directories into results list
 */
CV_EXPORTS void glob(const cv::String& directory, const cv::String& pattern,
        CV_OUT std::vector<cv::String>& result,
        bool recursive = false, bool includeDirectories = false);

/**
 * Generate a list of all files that match the globbing pattern.
 *
 * @param directory base directory
 * @param pattern filter pattern (based on '*'/'?' symbols). Use empty string to disable filtering and return all results
 * @param[out] result globbing result with relative paths from base directory
 * @param recursive scan nested directories too
 * @param includeDirectories include directories into results list
 */
CV_EXPORTS void glob_relative(const cv::String& directory, const cv::String& pattern,
        CV_OUT std::vector<cv::String>& result,
        bool recursive = false, bool includeDirectories = false);


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
