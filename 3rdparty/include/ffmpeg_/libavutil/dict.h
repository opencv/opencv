/*
 *
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

/**
 * @file
 * Public dictionary API.
 */

#ifndef AVUTIL_DICT_H
#define AVUTIL_DICT_H

#define AV_DICT_MATCH_CASE      1
#define AV_DICT_IGNORE_SUFFIX   2
#define AV_DICT_DONT_STRDUP_KEY 4
#define AV_DICT_DONT_STRDUP_VAL 8
#define AV_DICT_DONT_OVERWRITE 16   ///< Don't overwrite existing entries.
#define AV_DICT_APPEND         32   /**< If the entry already exists, append to it.  Note that no
                                      delimiter is added, the strings are simply concatenated. */

typedef struct {
    char *key;
    char *value;
} AVDictionaryEntry;

typedef struct AVDictionary AVDictionary;

/**
 * Get a dictionary entry with matching key.
 *
 * @param prev Set to the previous matching element to find the next.
 *             If set to NULL the first matching element is returned.
 * @param flags Allows case as well as suffix-insensitive comparisons.
 * @return Found entry or NULL, changing key or value leads to undefined behavior.
 */
AVDictionaryEntry *
av_dict_get(AVDictionary *m, const char *key, const AVDictionaryEntry *prev, int flags);

/**
 * Set the given entry in *pm, overwriting an existing entry.
 *
 * @param pm pointer to a pointer to a dictionary struct. If *pm is NULL
 * a dictionary struct is allocated and put in *pm.
 * @param key entry key to add to *pm (will be av_strduped depending on flags)
 * @param value entry value to add to *pm (will be av_strduped depending on flags).
 *        Passing a NULL value will cause an existing tag to be deleted.
 * @return >= 0 on success otherwise an error code <0
 */
int av_dict_set(AVDictionary **pm, const char *key, const char *value, int flags);

/**
 * Copy entries from one AVDictionary struct into another.
 * @param dst pointer to a pointer to a AVDictionary struct. If *dst is NULL,
 *            this function will allocate a struct for you and put it in *dst
 * @param src pointer to source AVDictionary struct
 * @param flags flags to use when setting entries in *dst
 * @note metadata is read using the AV_DICT_IGNORE_SUFFIX flag
 */
void av_dict_copy(AVDictionary **dst, AVDictionary *src, int flags);

/**
 * Free all the memory allocated for an AVDictionary struct.
 */
void av_dict_free(AVDictionary **m);

#endif // AVUTIL_DICT_H
