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

#undef NDEBUG

#include "opj_includes.h"

int main()
{
    OPJ_UINT32 i, j, w, h;
    OPJ_INT32 buffer[ 99 * 101 ];
    OPJ_BOOL ret;
    opj_sparse_array_int32_t* sa;

    sa = opj_sparse_array_int32_create(0, 1, 1, 1);
    assert(sa == NULL);
    opj_sparse_array_int32_free(sa);

    sa = opj_sparse_array_int32_create(1, 0, 1, 1);
    assert(sa == NULL);

    sa = opj_sparse_array_int32_create(1, 1, 0, 1);
    assert(sa == NULL);

    sa = opj_sparse_array_int32_create(1, 1, 1, 0);
    assert(sa == NULL);

    sa = opj_sparse_array_int32_create(99, 101, ~0U, ~0U);
    assert(sa == NULL);

    sa = opj_sparse_array_int32_create(99, 101, 15, 17);
    opj_sparse_array_int32_free(sa);

    sa = opj_sparse_array_int32_create(99, 101, 15, 17);
    ret = opj_sparse_array_int32_read(sa, 0, 0, 0, 1, buffer, 1, 1, OPJ_FALSE);
    assert(!ret);
    ret = opj_sparse_array_int32_read(sa, 0, 0, 1, 0, buffer, 1, 1, OPJ_FALSE);
    assert(!ret);
    ret = opj_sparse_array_int32_read(sa, 0, 0, 100, 1, buffer, 1, 1, OPJ_FALSE);
    assert(!ret);
    ret = opj_sparse_array_int32_read(sa, 0, 0, 1, 102, buffer, 1, 1, OPJ_FALSE);
    assert(!ret);
    ret = opj_sparse_array_int32_read(sa, 1, 0, 0, 1, buffer, 1, 1, OPJ_FALSE);
    assert(!ret);
    ret = opj_sparse_array_int32_read(sa, 0, 1, 1, 0, buffer, 1, 1, OPJ_FALSE);
    assert(!ret);
    ret = opj_sparse_array_int32_read(sa, 99, 101, 99, 101, buffer, 1, 1,
                                      OPJ_FALSE);
    assert(!ret);

    buffer[0] = 1;
    ret = opj_sparse_array_int32_read(sa, 0, 0, 1, 1, buffer, 1, 1, OPJ_FALSE);
    assert(ret);
    assert(buffer[0] == 0);

    memset(buffer, 0xFF, sizeof(buffer));
    ret = opj_sparse_array_int32_read(sa, 0, 0, 99, 101, buffer, 1, 99, OPJ_FALSE);
    assert(ret);
    for (i = 0; i < 99 * 101; i++) {
        assert(buffer[i] == 0);
    }

    buffer[0] = 1;
    ret = opj_sparse_array_int32_write(sa, 4, 5, 4 + 1, 5 + 1, buffer, 1, 1,
                                       OPJ_FALSE);
    assert(ret);

    buffer[0] = 2;
    ret = opj_sparse_array_int32_write(sa, 4, 5, 4 + 1, 5 + 1, buffer, 1, 1,
                                       OPJ_FALSE);
    assert(ret);

    buffer[0] = 0;
    buffer[1] = 0xFF;
    ret = opj_sparse_array_int32_read(sa, 4, 5, 4 + 1, 5 + 1, buffer, 1, 1,
                                      OPJ_FALSE);
    assert(ret);
    assert(buffer[0] == 2);
    assert(buffer[1] == 0xFF);

    buffer[0] = 0xFF;
    buffer[1] = 0xFF;
    buffer[2] = 0xFF;
    ret = opj_sparse_array_int32_read(sa, 4, 5, 4 + 1, 5 + 2, buffer, 0, 1,
                                      OPJ_FALSE);
    assert(ret);
    assert(buffer[0] == 2);
    assert(buffer[1] == 0);
    assert(buffer[2] == 0xFF);

    buffer[0] = 3;
    ret = opj_sparse_array_int32_write(sa, 4, 5, 4 + 1, 5 + 1, buffer, 0, 1,
                                       OPJ_FALSE);
    assert(ret);

    buffer[0] = 0;
    buffer[1] = 0xFF;
    ret = opj_sparse_array_int32_read(sa, 4, 5, 4 + 1, 5 + 1, buffer, 1, 1,
                                      OPJ_FALSE);
    assert(ret);
    assert(buffer[0] == 3);
    assert(buffer[1] == 0xFF);

    w = 15 + 1;
    h = 17 + 1;
    memset(buffer, 0xFF, sizeof(buffer));
    ret = opj_sparse_array_int32_read(sa, 2, 1, 2 + w, 1 + h, buffer, 1, w,
                                      OPJ_FALSE);
    assert(ret);
    for (j = 0; j < h; j++) {
        for (i = 0; i < w; i++) {
            if (i == 4 - 2 && j == 5 - 1) {
                assert(buffer[ j * w + i ] == 3);
            } else {
                assert(buffer[ j * w + i ] == 0);
            }
        }
    }

    opj_sparse_array_int32_free(sa);


    sa = opj_sparse_array_int32_create(99, 101, 15, 17);
    memset(buffer, 0xFF, sizeof(buffer));
    ret = opj_sparse_array_int32_read(sa, 0, 0, 2, 1, buffer, 2, 4, OPJ_FALSE);
    assert(ret);
    assert(buffer[0] == 0);
    assert(buffer[1] == -1);
    assert(buffer[2] == 0);

    buffer[0] = 1;
    buffer[2] = 3;
    ret = opj_sparse_array_int32_write(sa, 0, 0, 2, 1, buffer, 2, 4, OPJ_FALSE);
    assert(ret);

    memset(buffer, 0xFF, sizeof(buffer));
    ret = opj_sparse_array_int32_read(sa, 0, 0, 2, 1, buffer, 2, 4, OPJ_FALSE);
    assert(ret);
    assert(buffer[0] == 1);
    assert(buffer[1] == -1);
    assert(buffer[2] == 3);

    opj_sparse_array_int32_free(sa);

    return 0;
}
