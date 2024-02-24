/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_PRIVATE_UTIL_H
#define OPENEXR_PRIVATE_UTIL_H

#include <stdint.h>

static inline int
compute_sampled_lines (int height, int y_sampling, int start_y)
{
    int nlines;

    if (y_sampling <= 1) return height;

    if (height == 1)
        nlines = (start_y % y_sampling) == 0 ? 1 : 0;
    else
    {
        int start, end;

        /* computed the number of times y % ysampling == 0, by
         * computing interval based on first and last time that occurs
         * on the given range
         */
        start = start_y % y_sampling;
        if (start != 0)
            start = start_y + (y_sampling - start);
        else
            start = start_y;
        end = start_y + height - 1;
        end -= (end % y_sampling);

        if (start > end)
            nlines = 0;
        else
            nlines = (end - start) / y_sampling + 1;
    }

    return nlines;
}

#endif /* OPENEXR_PRIVATE_UTIL_H */
