#ifndef CVVISUAL_QTUTIL_HPP
#define CVVISUAL_QTUTIL_HPP

#include <limits>
#include <vector>
#include <stdexcept>

#include <QImage>
#include <QPixmap>
#include <QSet>

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"


namespace cvv
{
namespace qtutil
{

/**
 * @brief Represents the staus of an image conversion.
 */
enum class ImageConversionResult
{
	SUCCESS,
	MAT_EMPTY,
	MAT_NOT_2D,
	FLOAT_OUT_OF_0_TO_1,
	NUMBER_OF_CHANNELS_NOT_SUPPORTED,
	MAT_INVALID_SIZE,
	MAT_UNSUPPORTED_DEPTH
};

/**
 * @brief Converts a cv::Mat to a QImage.
 * @param mat The mat to convert.
 * @param skipFloatRangeTest If true Mats with floating types will be checked
 *  wheather all values are within [0,1].
 * @param threads Number of threads to use (0 will use 1 thread).
 * @return The status of the conversion and the converted mat.
 */
std::pair<ImageConversionResult, QImage>
convertMatToQImage(const cv::Mat &mat, bool skipFloatRangeTest = true,
		   unsigned int threads =
		       std::numeric_limits<unsigned int>::max());

/**
 * @brief Converts a cv::Mat to a QPixmap.
 * @param mat The mat to convert.
 * @param skipFloatRangeTest If true Mats with floating types will be checked
 *  wheather all values are within [0,1].
 * @param threads Number of threads to use (0 will use 1 thread).
 * @return The status of the conversion and the converted mat.
 */
std::pair<ImageConversionResult, QPixmap>
convertMatToQPixmap(const cv::Mat &mat, bool skipFloatRangeTest = true,
		    unsigned int threads =
			std::numeric_limits<unsigned int>::max());

/**
 * @brief Creates a QSet<QString> with the given string as an inherited value.
 */
QSet<QString> createStringSet(QString string);

/**
 * @brief Returns a string containing the type of the mat.
 * @param mat The mat.
 * @return A string containing the type of the mat.
 * (first = flase when the depth is unknown)
 */
std::pair<bool, QString> typeToQString(const cv::Mat &mat);

/**
 * @brief Returns a string descripton to a image conversion result.
 * @param result The image conversion result.
 * @return The descripton.
 */
QString conversionResultToString(const ImageConversionResult &result);

/**
 * @brief Splits a mat in multiple one channel mats.
 * @param mat The mat.
 * @return The splitted mats.
 */
std::vector<cv::Mat> splitChannels(const cv::Mat &mat);

/**
 * @brief Merges multiple one channel mats into one.
 * @param mats The mats to merge.
 * @return Merged mat
 * @throw std::invalid_argument If the images have different depths. Or one mat
 * has more than 1
 * channel.
 */
cv::Mat mergeChannels(std::vector<cv::Mat> mats);

/**
 * @brief Opens the users default browser with the topic help page.
 * Current URL: cvv.mostlynerdless.de/help.php?topic=[topic]
 *
 * Topics can be added via appending the doc/topics.yml file.
 *
 * @param topic help topic
 */
void openHelpBrowser(const QString &topic);

/**
 * @brief Set the default setting for a given stettings key and scope.
 * It doesn't override existing settings.
 * @param scope given settings scope
 * @param key given settings key
 * @param value default value of the setting
 */
void setDefaultSetting(const QString &scope, const QString &key,
		       const QString &value);

/**
 * @brief Set the setting for a given stettings key and scope.
 * @param scope given settings scope
 * @param key given settings key
 * @param value new value of the setting
 */
void setSetting(const QString &scope, const QString &key, const QString &value);

/**
 * @brief Get the current setting [key] in the given scope.
 * Please use `setDefaultSetting` to set a default value that's other than
 * an empty QString.
 * @param scope given scope (e.g. 'Overview')
 * @param key settings key (e.g. 'autoOpenTabs')
 * @return settings string
 */
QString getSetting(const QString &scope, const QString &key);
}
}

#endif
