/*
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#ifndef AVUTIL_PARSEUTILS_H
#define AVUTIL_PARSEUTILS_H

#include <time.h>

#include "rational.h"

/**
 * @file
 * misc parsing utilities
 */

/**
 * Parse str and put in width_ptr and height_ptr the detected values.
 *
 * @param[in,out] width_ptr pointer to the variable which will contain the detected
 * width value
 * @param[in,out] height_ptr pointer to the variable which will contain the detected
 * height value
 * @param[in] str the string to parse: it has to be a string in the format
 * width x height or a valid video size abbreviation.
 * @return >= 0 on success, a negative error code otherwise
 */
int av_parse_video_size(int *width_ptr, int *height_ptr, const char *str);

/**
 * Parse str and store the detected values in *rate.
 *
 * @param[in,out] rate pointer to the AVRational which will contain the detected
 * frame rate
 * @param[in] str the string to parse: it has to be a string in the format
 * rate_num / rate_den, a float number or a valid video rate abbreviation
 * @return >= 0 on success, a negative error code otherwise
 */
int av_parse_video_rate(AVRational *rate, const char *str);

/**
 * Put the RGBA values that correspond to color_string in rgba_color.
 *
 * @param color_string a string specifying a color. It can be the name of
 * a color (case insensitive match) or a [0x|#]RRGGBB[AA] sequence,
 * possibly followed by "@" and a string representing the alpha
 * component.
 * The alpha component may be a string composed by "0x" followed by an
 * hexadecimal number or a decimal number between 0.0 and 1.0, which
 * represents the opacity value (0x00/0.0 means completely transparent,
 * 0xff/1.0 completely opaque).
 * If the alpha component is not specified then 0xff is assumed.
 * The string "random" will result in a random color.
 * @param slen length of the initial part of color_string containing the
 * color. It can be set to -1 if color_string is a null terminated string
 * containing nothing else than the color.
 * @return >= 0 in case of success, a negative value in case of
 * failure (for example if color_string cannot be parsed).
 */
int av_parse_color(uint8_t *rgba_color, const char *color_string, int slen,
                   void *log_ctx);

/**
 * Parse timestr and return in *time a corresponding number of
 * microseconds.
 *
 * @param timeval puts here the number of microseconds corresponding
 * to the string in timestr. If the string represents a duration, it
 * is the number of microseconds contained in the time interval.  If
 * the string is a date, is the number of microseconds since 1st of
 * January, 1970 up to the time of the parsed date.  If timestr cannot
 * be successfully parsed, set *time to INT64_MIN.

 * @param timestr a string representing a date or a duration.
 * - If a date the syntax is:
 * @code
 * [{YYYY-MM-DD|YYYYMMDD}[T|t| ]]{{HH[:MM[:SS[.m...]]]}|{HH[MM[SS[.m...]]]}}[Z]
 * now
 * @endcode
 * If the value is "now" it takes the current time.
 * Time is local time unless Z is appended, in which case it is
 * interpreted as UTC.
 * If the year-month-day part is not specified it takes the current
 * year-month-day.
 * - If a duration the syntax is:
 * @code
 * [-]HH[:MM[:SS[.m...]]]
 * [-]S+[.m...]
 * @endcode
 * @param duration flag which tells how to interpret timestr, if not
 * zero timestr is interpreted as a duration, otherwise as a date
 * @return 0 in case of success, a negative value corresponding to an
 * AVERROR code otherwise
 */
int av_parse_time(int64_t *timeval, const char *timestr, int duration);

/**
 * Attempt to find a specific tag in a URL.
 *
 * syntax: '?tag1=val1&tag2=val2...'. Little URL decoding is done.
 * Return 1 if found.
 */
int av_find_info_tag(char *arg, int arg_size, const char *tag1, const char *info);

/**
 * Convert the decomposed UTC time in tm to a time_t value.
 */
time_t av_timegm(struct tm *tm);

#endif /* AVUTIL_PARSEUTILS_H */
