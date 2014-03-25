#ifndef CVVISUAL_TYPES_HPP
#define CVVISUAL_TYPES_HPP

#include "opencv2/core/core.hpp"


namespace cvv
{
namespace qtutil
{

namespace structures
{
/**
 * Helper struct to convert an opencv depth type (an int) back to a type.
 */
template <int depth> struct DepthTypeConverter
{
	static_assert(!(depth == CV_8U || depth == CV_8S || depth == CV_16U ||
			depth == CV_16S || depth == CV_32S || depth == CV_32F ||
			depth == CV_64F),
		      "Conversion of unknown type");
	// using type;
};
/**
 * Helper struct to convert an opencv depth type (an int) back to a type.
 */
template <> struct DepthTypeConverter<CV_8U>
{
	using type = uint8_t;
};
/**
 * Helper struct to convert an opencv depth type (an int) back to a type.
 */
template <> struct DepthTypeConverter<CV_8S>
{
	using type = int8_t;
};
/**
 * Helper struct to convert an opencv depth type (an int) back to a type.
 */
template <> struct DepthTypeConverter<CV_16U>
{
	using type = uint16_t;
};
/**
 * Helper struct to convert an opencv depth type (an int) back to a type.
 */
template <> struct DepthTypeConverter<CV_16S>
{
	using type = int16_t;
};
/**
 * Helper struct to convert an opencv depth type (an int) back to a type.
 */
template <> struct DepthTypeConverter<CV_32S>
{
	using type = int32_t;
};
/**
 * Helper struct to convert an opencv depth type (an int) back to a type.
 */
template <> struct DepthTypeConverter<CV_32F>
{
	using type = float;
};
/**
 * Helper struct to convert an opencv depth type (an int) back to a type.
 */
template <> struct DepthTypeConverter<CV_64F>
{
	using type = double;
};

/**
 * Helper struct to convert an opencv depth type and a number of channels (1 to
 * 4) into
 * a c++ type.
 */
template <class depthtype, int channels> struct PixelTypeConverter
{
	static_assert(channels >= 1 && channels <= 10,
		      "Illegal number of channels");
	using type = cv::Vec<depthtype, channels>;
};
}

/**
 * Converts an opencv depth type (an int) back to a type.
 */
template <int depth>
using DepthType = typename structures::DepthTypeConverter<depth>::type;

/**
 * Converts an opencv depth type and a number of channels (1 to 4) into a c++
 * type.
 */
template <int depth, int channels>
using PixelType =
    typename structures::PixelTypeConverter<DepthType<depth>, channels>::type;

// convert a depth value to uchar

/**
 * @brief Converts the value to an uchar.
 * @param value The value to convert.
 * @return The converted value.
 */
template <int depth> uchar convertTo8U(const DepthType<depth>) // = delete;
{
	// specializing deleted function-templates is an error in clang, so
	// catch it here:
	static_assert(
	    depth != depth, // we are not allowed to write false directly
	    "The general version of convertTo8U must never be called");
	// avoid warnings:
	return {};
}

/**
 * @brief Converts the value to an uchar.
 * @param value The value to convert.
 * @return The converted value.
 */
template <> inline uchar convertTo8U<CV_8U>(const DepthType<CV_8U> value)
{
	return value;
}

/**
 * @brief Converts the value to an uchar.
 * @param value The value to convert.
 * @return The converted value.
 */
template <> inline uchar convertTo8U<CV_16U>(const DepthType<CV_16U> value)
{
	return cv::saturate_cast<DepthType<CV_8U>>(value / 256);
}

/**
 * @brief Converts the value to an uchar.
 * @param value The value to convert.
 * @return The converted value.
 */
template <> inline uchar convertTo8U<CV_16S>(const DepthType<CV_16S> value)
{
	return convertTo8U<CV_8U>((value / 256) + 128);
}

/**
 * @brief Converts the value to an uchar.
 * @param value The value to convert.
 * @return The converted value.
 */
template <> inline uchar convertTo8U<CV_8S>(const DepthType<CV_8S> value)
{
	return convertTo8U<CV_16S>(cv::saturate_cast<DepthType<CV_16S>>(value) *
				   256);
}

/**
 * @brief Converts the value to an uchar.
 * @param value The value to convert.
 * @return The converted value.
 */
template <> inline uchar convertTo8U<CV_32S>(const DepthType<CV_32S> value)
{
	return convertTo8U<CV_16S>(
	    cv::saturate_cast<DepthType<CV_16S>>(value / (256 * 256)));
}

/**
 * @brief Converts the value to an uchar.
 * @param value The value to convert.
 * @return The converted value.
 */
template <> inline uchar convertTo8U<CV_32F>(const DepthType<CV_32F> value)
{
	return cv::saturate_cast<DepthType<CV_8U>>(value * 256.0);
}

/**
 * @brief Converts the value to an uchar.
 * @param value The value to convert.
 * @return The converted value.
 */
template <> inline uchar convertTo8U<CV_64F>(const DepthType<CV_64F> value)
{
	return cv::saturate_cast<DepthType<CV_8U>>(value * 256.0);
}
}
} // namespaces

#endif // TYPES_HPP
