/* quirc -- QR-code recognition library
 * Copyright (C) 2010-2012 Daniel Beer <dlbeer@gmail.com>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef QUIRC_H_
#define QUIRC_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct quirc;

/* Obtain the library version string. */
const char *quirc_version(void);

/* Construct a new QR-code recognizer. This function will return NULL
 * if sufficient memory could not be allocated.
 */
struct quirc *quirc_new(void);

/* Destroy a QR-code recognizer. */
void quirc_destroy(struct quirc *q);

/* Resize the QR-code recognizer. The size of an image must be
 * specified before codes can be analyzed.
 *
 * This function returns 0 on success, or -1 if sufficient memory could
 * not be allocated.
 */
int quirc_resize(struct quirc *q, int w, int h);

/* These functions are used to process images for QR-code recognition.
 * quirc_begin() must first be called to obtain access to a buffer into
 * which the input image should be placed. Optionally, the current
 * width and height may be returned.
 *
 * After filling the buffer, quirc_end() should be called to process
 * the image for QR-code recognition. The locations and content of each
 * code may be obtained using accessor functions described below.
 */
uint8_t *quirc_begin(struct quirc *q, int *w, int *h);
void quirc_end(struct quirc *q);

/* This structure describes a location in the input image buffer. */
struct quirc_point {
	int	x;
	int	y;
};

/* This enum describes the various decoder errors which may occur. */
typedef enum {
	QUIRC_SUCCESS = 0,
	QUIRC_ERROR_INVALID_GRID_SIZE,
	QUIRC_ERROR_INVALID_VERSION,
	QUIRC_ERROR_FORMAT_ECC,
	QUIRC_ERROR_DATA_ECC,
	QUIRC_ERROR_UNKNOWN_DATA_TYPE,
	QUIRC_ERROR_DATA_OVERFLOW,
	QUIRC_ERROR_DATA_UNDERFLOW
} quirc_decode_error_t;

/* Return a string error message for an error code. */
const char *quirc_strerror(quirc_decode_error_t err);

/* Limits on the maximum size of QR-codes and their content. */
#define QUIRC_MAX_BITMAP	3917
#define QUIRC_MAX_PAYLOAD	8896

/* QR-code ECC types. */
#define QUIRC_ECC_LEVEL_M     0
#define QUIRC_ECC_LEVEL_L     1
#define QUIRC_ECC_LEVEL_H     2
#define QUIRC_ECC_LEVEL_Q     3

/* QR-code data types. */
#define QUIRC_DATA_TYPE_NUMERIC       1
#define QUIRC_DATA_TYPE_ALPHA         2
#define QUIRC_DATA_TYPE_BYTE          4
#define QUIRC_DATA_TYPE_KANJI         8

/* Common character encodings */
#define QUIRC_ECI_ISO_8859_1		1
#define QUIRC_ECI_IBM437		2
#define QUIRC_ECI_ISO_8859_2		4
#define QUIRC_ECI_ISO_8859_3		5
#define QUIRC_ECI_ISO_8859_4		6
#define QUIRC_ECI_ISO_8859_5		7
#define QUIRC_ECI_ISO_8859_6		8
#define QUIRC_ECI_ISO_8859_7		9
#define QUIRC_ECI_ISO_8859_8		10
#define QUIRC_ECI_ISO_8859_9		11
#define QUIRC_ECI_WINDOWS_874		13
#define QUIRC_ECI_ISO_8859_13		15
#define QUIRC_ECI_ISO_8859_15		17
#define QUIRC_ECI_SHIFT_JIS		20
#define QUIRC_ECI_UTF_8			26

/* This structure is used to return information about detected QR codes
 * in the input image.
 */
struct quirc_code {
	/* The four corners of the QR-code, from top left, clockwise */
	struct quirc_point	corners[4];

	/* The number of cells across in the QR-code. The cell bitmap
	 * is a bitmask giving the actual values of cells. If the cell
	 * at (x, y) is black, then the following bit is set:
	 *
	 *     cell_bitmap[i >> 3] & (1 << (i & 7))
	 *
	 * where i = (y * size) + x.
	 */
	int			size;
	uint8_t			cell_bitmap[QUIRC_MAX_BITMAP];
};

/* This structure holds the decoded QR-code data */
struct quirc_data {
	/* Various parameters of the QR-code. These can mostly be
	 * ignored if you only care about the data.
	 */
	int			version;
	int			ecc_level;
	int			mask;

	/* This field is the highest-valued data type found in the QR
	 * code.
	 */
	int			data_type;

	/* Data payload. For the Kanji datatype, payload is encoded as
	 * Shift-JIS. For all other datatypes, payload is ASCII text.
	 */
	uint8_t			payload[QUIRC_MAX_PAYLOAD];
	int			payload_len;

	/* ECI assignment number */
	uint32_t		eci;
};

/* Return the number of QR-codes identified in the last processed
 * image.
 */
int quirc_count(const struct quirc *q);

/* Extract the QR-code specified by the given index. */
void quirc_extract(const struct quirc *q, int index,
		   struct quirc_code *code);

/* Decode a QR-code, returning the payload data. */
quirc_decode_error_t quirc_decode(const struct quirc_code *code,
				  struct quirc_data *data);
/* flip the QR code horizontaly (mirror flip) */
void quirc_flip(struct quirc_code *code);

#ifdef __cplusplus
}
#endif

#endif
