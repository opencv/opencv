#ifndef CVVISUAL_COLOR_UTIL
#define CVVISUAL_COLOR_UTIL

#include <vector>

#include <QColor>

#include "opencv2/core/core.hpp"

namespace cvv
{
namespace qtutil
{
/**
 * @brief Returns the false color. (BGR)
 * @param value The double to convert.
 * @return The false color.
 */
cv::Vec<uint8_t,3> inline falseColor(double d)
{
	static const std::vector<cv::Vec<uint8_t,3>> points{
		{176,0,13}, // 0.0
		{163,0,31}, // 0.1
		{131,0,75}, // 0.2
		{86,0,137}, // 0.3
		{36,0,205}, // 0.4
		{0,0,255}, // 0.5
		{0,49,255}, // 0.6
		{0,119,255}, // 0.7
		{0,183,255}, // 0.8
		{0,229,255}, // 0.9
		{0,251,255} // 1.0
	};

	if(d<0)
	{
		return points[0];
	}
	if(d>1)
	{
		return points[10];
	}

	int low = std::floor(d/0.1);
	int high = std::ceil(d/0.1);

	//interp. factor
	double factorHi = d/0.1 - low;
	double factorLo = 1 - factorHi;


	return factorLo*(points[low]) + factorHi*(points[high]);
		/*{cv::saturate_cast<uint8_t>( (factorLo*(points[low][0]) + factorHi*(points[high][0]))),
		cv::saturate_cast<uint8_t>( (factorLo*(points[low][1]) + factorHi*(points[high][1]))),
		cv::saturate_cast<uint8_t>( (factorLo*(points[low][2]) + factorHi*(points[high][2])))};*/
}

QColor inline getFalseColor(double value, double max, double min)
{
	cv::Vec<uint8_t,3> color;
	if(value<=min)
	{
		color=falseColor(0);
	} else if(value>=max)
	{
		color=falseColor(1);
	}else if(max<=min)
	{
		color=falseColor(0);
	}else {
		double val01 = (value-min) / (max - min);
		color=falseColor(val01);
	}

	return QColor{color[2],color[1],color[0]};
}

}
}
#endif
