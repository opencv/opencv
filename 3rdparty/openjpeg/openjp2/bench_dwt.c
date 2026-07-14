/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
 * Copyright (c) 2017, IntoPix SA <contact@intopix.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "opj_includes.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/times.h>
#endif /* _WIN32 */

OPJ_INT32 getValue(OPJ_UINT32 i)
{
    return ((OPJ_INT32)i % 511) - 256;
}

void init_tilec(opj_tcd_tilecomp_t * l_tilec,
                OPJ_INT32 x0,
                OPJ_INT32 y0,
                OPJ_INT32 x1,
                OPJ_INT32 y1,
                OPJ_UINT32 numresolutions,
                OPJ_BOOL irreversible)
{
    opj_tcd_resolution_t* l_res;
    OPJ_UINT32 resno, l_level_no;
    size_t i, nValues;

    memset(l_tilec, 0, sizeof(*l_tilec));
    l_tilec->x0 = x0;
    l_tilec->y0 = y0;
    l_tilec->x1 = x1;
    l_tilec->y1 = y1;
    nValues = (size_t)(l_tilec->x1 - l_tilec->x0) *
              (size_t)(l_tilec->y1 - l_tilec->y0);
    l_tilec->data = (OPJ_INT32*) opj_malloc(sizeof(OPJ_INT32) * nValues);
    assert(l_tilec->data != NULL);
    for (i = 0; i < nValues; i++) {
        OPJ_INT32 val = getValue((OPJ_UINT32)i);
        if (irreversible) {
            OPJ_FLOAT32 fVal = (OPJ_FLOAT32)val;
            memcpy(&l_tilec->data[i], &fVal, sizeof(OPJ_FLOAT32));
        } else {
            l_tilec->data[i] = val;
        }
    }
    l_tilec->numresolutions = numresolutions;
    l_tilec->minimum_num_resolutions = numresolutions;
    l_tilec->resolutions = (opj_tcd_resolution_t*) opj_calloc(
                               l_tilec->numresolutions,
                               sizeof(opj_tcd_resolution_t));

    l_level_no = l_tilec->numresolutions;
    l_res = l_tilec->resolutions;

    /* Adapted from opj_tcd_init_tile() */
    for (resno = 0; resno < l_tilec->numresolutions; ++resno) {

        --l_level_no;

        /* border for each resolution level (global) */
        l_res->x0 = opj_int_ceildivpow2(l_tilec->x0, (OPJ_INT32)l_level_no);
        l_res->y0 = opj_int_ceildivpow2(l_tilec->y0, (OPJ_INT32)l_level_no);
        l_res->x1 = opj_int_ceildivpow2(l_tilec->x1, (OPJ_INT32)l_level_no);
        l_res->y1 = opj_int_ceildivpow2(l_tilec->y1, (OPJ_INT32)l_level_no);

        ++l_res;
    }
}

void free_tilec(opj_tcd_tilecomp_t * l_tilec)
{
    opj_free(l_tilec->data);
    opj_free(l_tilec->resolutions);
}

void usage(void)
{
    printf(
        "bench_dwt [-decode|encode] [-I] [-size value] [-check] [-display]\n");
    printf(
        "          [-num_resolutions val] [-offset x y] [-num_threads val]\n");
    exit(1);
}


OPJ_FLOAT64 opj_clock(void)
{
#ifdef _WIN32
    /* _WIN32: use QueryPerformance (very accurate) */
    LARGE_INTEGER freq, t ;
    /* freq is the clock speed of the CPU */
    QueryPerformanceFrequency(&freq) ;
    /* cout << "freq = " << ((double) freq.QuadPart) << endl; */
    /* t is the high resolution performance counter (see MSDN) */
    QueryPerformanceCounter(& t) ;
    return freq.QuadPart ? (t.QuadPart / (OPJ_FLOAT64) freq.QuadPart) : 0 ;
#else
    /* Unix or Linux: use resource usage */
    struct rusage t;
    OPJ_FLOAT64 procTime;
    /* (1) Get the rusage data structure at this moment (man getrusage) */
    getrusage(0, &t);
    /* (2) What is the elapsed time ? - CPU time = User time + System time */
    /* (2a) Get the seconds */
    procTime = (OPJ_FLOAT64)(t.ru_utime.tv_sec + t.ru_stime.tv_sec);
    /* (2b) More precisely! Get the microseconds part ! */
    return (procTime + (OPJ_FLOAT64)(t.ru_utime.tv_usec + t.ru_stime.tv_usec) *
            1e-6) ;
#endif
}

static OPJ_FLOAT64 opj_wallclock(void)
{
#ifdef _WIN32
    return opj_clock();
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (OPJ_FLOAT64)tv.tv_sec + 1e-6 * (OPJ_FLOAT64)tv.tv_usec;
#endif
}

int main(int argc, char** argv)
{
    int num_threads = 0;
    opj_tcd_t tcd;
    opj_tcd_image_t tcd_image;
    opj_tcd_tile_t tcd_tile;
    opj_tcd_tilecomp_t tilec;
    opj_image_t image;
    opj_image_comp_t image_comp;
    opj_thread_pool_t* tp;
    OPJ_INT32 i, j, k;
    OPJ_BOOL display = OPJ_FALSE;
    OPJ_BOOL check = OPJ_FALSE;
    OPJ_INT32 size = 16384 - 1;
    OPJ_FLOAT64 start, stop;
    OPJ_FLOAT64 start_wc, stop_wc;
    OPJ_UINT32 offset_x = ((OPJ_UINT32)size + 1) / 2 - 1;
    OPJ_UINT32 offset_y = ((OPJ_UINT32)size + 1) / 2 - 1;
    OPJ_UINT32 num_resolutions = 6;
    OPJ_BOOL bench_decode = OPJ_TRUE;
    OPJ_BOOL irreversible = OPJ_FALSE;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-encode") == 0) {
            bench_decode = OPJ_FALSE;
        } else if (strcmp(argv[i], "-decode") == 0) {
            bench_decode = OPJ_TRUE;
        } else if (strcmp(argv[i], "-display") == 0) {
            display = OPJ_TRUE;
        } else if (strcmp(argv[i], "-check") == 0) {
            check = OPJ_TRUE;
        } else if (strcmp(argv[i], "-I") == 0) {
            irreversible = OPJ_TRUE;
        } else if (strcmp(argv[i], "-size") == 0 && i + 1 < argc) {
            size = atoi(argv[i + 1]);
            i ++;
        } else if (strcmp(argv[i], "-num_threads") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[i + 1]);
            i ++;
        } else if (strcmp(argv[i], "-num_resolutions") == 0 && i + 1 < argc) {
            num_resolutions = (OPJ_UINT32)atoi(argv[i + 1]);
            if (num_resolutions == 0 || num_resolutions > 32) {
                fprintf(stderr,
                        "Invalid value for num_resolutions. Should be >= 1 and <= 32\n");
                exit(1);
            }
            i ++;
        } else if (strcmp(argv[i], "-offset") == 0 && i + 2 < argc) {
            offset_x = (OPJ_UINT32)atoi(argv[i + 1]);
            offset_y = (OPJ_UINT32)atoi(argv[i + 2]);
            i += 2;
        } else {
            usage();
        }
    }

    if (irreversible && check) {
        /* Due to irreversible inverse DWT not being symmetric of forward */
        /* See BUG_WEIRD_TWO_INVK in dwt.c */
        printf("-I and -check aren't compatible\n");
        exit(1);
    }

    tp = opj_thread_pool_create(num_threads);

    init_tilec(&tilec, (OPJ_INT32)offset_x, (OPJ_INT32)offset_y,
               (OPJ_INT32)offset_x + size, (OPJ_INT32)offset_y + size,
               num_resolutions, irreversible);

    if (display) {
        printf("Before\n");
        k = 0;
        for (j = 0; j < tilec.y1 - tilec.y0; j++) {
            for (i = 0; i < tilec.x1 - tilec.x0; i++) {
                if (irreversible) {
                    printf("%f ", ((OPJ_FLOAT32*)tilec.data)[k]);
                } else {
                    printf("%d ", tilec.data[k]);
                }
                k ++;
            }
            printf("\n");
        }
    }

    memset(&tcd, 0, sizeof(tcd));
    tcd.thread_pool = tp;
    tcd.whole_tile_decoding = OPJ_TRUE;
    tcd.win_x0 = (OPJ_UINT32)tilec.x0;
    tcd.win_y0 = (OPJ_UINT32)tilec.y0;
    tcd.win_x1 = (OPJ_UINT32)tilec.x1;
    tcd.win_y1 = (OPJ_UINT32)tilec.y1;
    tcd.tcd_image = &tcd_image;
    memset(&tcd_image, 0, sizeof(tcd_image));
    tcd_image.tiles = &tcd_tile;
    memset(&tcd_tile, 0, sizeof(tcd_tile));
    tcd_tile.x0 = tilec.x0;
    tcd_tile.y0 = tilec.y0;
    tcd_tile.x1 = tilec.x1;
    tcd_tile.y1 = tilec.y1;
    tcd_tile.numcomps = 1;
    tcd_tile.comps = &tilec;
    tcd.image = &image;
    memset(&image, 0, sizeof(image));
    image.numcomps = 1;
    image.comps = &image_comp;
    memset(&image_comp, 0, sizeof(image_comp));
    image_comp.dx = 1;
    image_comp.dy = 1;

    start = opj_clock();
    start_wc = opj_wallclock();
    if (bench_decode) {
        if (irreversible)  {
            opj_dwt_decode_real(&tcd, &tilec, tilec.numresolutions);
        } else {
            opj_dwt_decode(&tcd, &tilec, tilec.numresolutions);
        }
    } else {
        if (irreversible)  {
            opj_dwt_encode_real(&tcd, &tilec);
        } else {
            opj_dwt_encode(&tcd, &tilec);
        }
    }
    stop = opj_clock();
    stop_wc = opj_wallclock();
    printf("time for %s: total = %.03f s, wallclock = %.03f s\n",
           bench_decode ? "dwt_decode" : "dwt_encode",
           stop - start,
           stop_wc - start_wc);

    if (display) {
        if (bench_decode) {
            printf("After IDWT\n");
        } else {
            printf("After FDWT\n");
        }
        k = 0;
        for (j = 0; j < tilec.y1 - tilec.y0; j++) {
            for (i = 0; i < tilec.x1 - tilec.x0; i++) {
                if (irreversible) {
                    printf("%f ", ((OPJ_FLOAT32*)tilec.data)[k]);
                } else {
                    printf("%d ", tilec.data[k]);
                }
                k ++;
            }
            printf("\n");
        }
    }

    if ((display || check) && !irreversible) {

        if (bench_decode) {
            opj_dwt_encode(&tcd, &tilec);
        } else {
            opj_dwt_decode(&tcd, &tilec, tilec.numresolutions);
        }


        if (display && !irreversible) {
            if (bench_decode) {
                printf("After FDWT\n");
            } else {
                printf("After IDWT\n");
            }
            k = 0;
            for (j = 0; j < tilec.y1 - tilec.y0; j++) {
                for (i = 0; i < tilec.x1 - tilec.x0; i++) {
                    if (irreversible) {
                        printf("%f ", ((OPJ_FLOAT32*)tilec.data)[k]);
                    } else {
                        printf("%d ", tilec.data[k]);
                    }
                    k ++;
                }
                printf("\n");
            }
        }

    }

    if (check) {

        size_t idx;
        size_t nValues = (size_t)(tilec.x1 - tilec.x0) *
                         (size_t)(tilec.y1 - tilec.y0);
        for (idx = 0; idx < nValues; idx++) {
            if (tilec.data[idx] != getValue((OPJ_UINT32)idx)) {
                printf("Difference found at idx = %u\n", (OPJ_UINT32)idx);
                exit(1);
            }
        }
    }

    free_tilec(&tilec);

    opj_thread_pool_destroy(tp);
    return 0;
}
