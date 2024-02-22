/*
 * Copyright (c) 1991-1997 Sam Leffler
 * Copyright (c) 1991-1997 Silicon Graphics, Inc.
 *
 * Permission to use, copy, modify, distribute, and sell this software and
 * its documentation for any purpose is hereby granted without fee, provided
 * that (i) the above copyright notices and this permission notice appear in
 * all copies of the software and related documentation, and (ii) the names of
 * Sam Leffler and Silicon Graphics may not be used in any advertising or
 * publicity relating to the software without the specific, prior written
 * permission of Sam Leffler and Silicon Graphics.
 *
 * THE SOFTWARE IS PROVIDED "AS-IS" AND WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS, IMPLIED OR OTHERWISE, INCLUDING WITHOUT LIMITATION, ANY
 * WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 *
 * IN NO EVENT SHALL SAM LEFFLER OR SILICON GRAPHICS BE LIABLE FOR
 * ANY SPECIAL, INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY KIND,
 * OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER OR NOT ADVISED OF THE POSSIBILITY OF DAMAGE, AND ON ANY THEORY OF
 * LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THIS SOFTWARE.
 */

#include <stdio.h>
#include <string.h>

/*
 * Hack program to construct tables used to find
 * runs of zeros and ones in Group 3 Fax encoding.
 */

dumparray(name, runs) char *name;
unsigned char runs[256];
{
    int i;
    char *sep;
    printf("static unsigned char %s[256] = {\n", name);
    sep = "    ";
    for (i = 0; i < 256; i++)
    {
        printf("%s%d", sep, runs[i]);
        if (((i + 1) % 16) == 0)
        {
            printf(",	/* 0x%02x - 0x%02x */\n", i - 15, i);
            sep = "    ";
        }
        else
            sep = ", ";
    }
    printf("\n};\n");
}

main()
{
    unsigned char runs[2][256];

    memset(runs[0], 0, 256 * sizeof(char));
    memset(runs[1], 0, 256 * sizeof(char));
    {
        register int run, runlen, i;
        runlen = 1;
        for (run = 0x80; run != 0xff; run = (run >> 1) | 0x80)
        {
            for (i = run - 1; i >= 0; i--)
            {
                runs[1][run | i] = runlen;
                runs[0][(~(run | i)) & 0xff] = runlen;
            }
            runlen++;
        }
        runs[1][0xff] = runs[0][0] = 8;
    }
    dumparray("bruns", runs[0]);
    dumparray("wruns", runs[1]);
}
