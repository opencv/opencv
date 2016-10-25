/* This is AGAST and OAST, an optimal and accelerated corner detector
              based on the accelerated segment tests
   Below is the original copyright and the references */

/*
Copyright (C) 2010  Elmar Mair
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

    *Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

    *Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

    *Neither the name of the University of Cambridge nor the names of
     its contributors may be used to endorse or promote products derived
     from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
The references are:
 * Adaptive and Generic Corner Detection Based on the Accelerated Segment Test,
   Elmar Mair and Gregory D. Hager and Darius Burschka
   and Michael Suppa and Gerhard Hirzinger ECCV 2010
   URL: http://www6.in.tum.de/Main/ResearchAgast
*/

#include "agast_score.hpp"

#ifdef _MSC_VER
#pragma warning( disable : 4127 )
#endif

namespace cv
{

void makeAgastOffsets(int pixel[16], int rowStride, int type)
{
    static const int offsets16[][2] =
    {
        {-3,  0}, {-3, -1}, {-2, -2}, {-1, -3}, {0, -3}, { 1, -3}, { 2, -2}, { 3, -1},
        { 3,  0}, { 3,  1}, { 2,  2}, { 1,  3}, {0,  3}, {-1,  3}, {-2,  2}, {-3,  1}
    };

    static const int offsets12d[][2] =
    {
        {-3,  0}, {-2, -1}, {-1, -2}, {0, -3}, { 1, -2}, { 2, -1},
        { 3,  0}, { 2,  1}, { 1,  2}, {0,  3}, {-1,  2}, {-2,  1}
    };

    static const int offsets12s[][2] =
    {
        {-2,  0}, {-2, -1}, {-1, -2}, {0, -2}, { 1, -2}, { 2, -1},
        { 2,  0}, { 2,  1}, { 1,  2}, {0,  2}, {-1,  2}, {-2,  1}
    };

    static const int offsets8[][2] =
    {
        {-1,  0}, {-1, -1}, {0, -1}, { 1, -1},
        { 1,  0}, { 1,  1}, {0,  1}, {-1,  1}
    };

    const int (*offsets)[2] = type == AgastFeatureDetector::OAST_9_16 ? offsets16 :
                              type == AgastFeatureDetector::AGAST_7_12d ? offsets12d :
                              type == AgastFeatureDetector::AGAST_7_12s ? offsets12s :
                              type == AgastFeatureDetector::AGAST_5_8 ? offsets8  : 0;

    CV_Assert(pixel && offsets);

    int k = 0;
    for( ; k < 16; k++ )
        pixel[k] = offsets[k][0] + offsets[k][1] * rowStride;
}

#if (defined __i386__ || defined(_M_IX86) || defined __x86_64__ || defined(_M_X64))
// 16 pixel mask
template<>
int agast_cornerScore<AgastFeatureDetector::OAST_9_16>(const uchar* ptr, const int pixel[], int threshold)
{
    int bmin = threshold;
    int bmax = 255;
    int b_test = (bmax + bmin) / 2;

    short offset0 = (short) pixel[0];
    short offset1 = (short) pixel[1];
    short offset2 = (short) pixel[2];
    short offset3 = (short) pixel[3];
    short offset4 = (short) pixel[4];
    short offset5 = (short) pixel[5];
    short offset6 = (short) pixel[6];
    short offset7 = (short) pixel[7];
    short offset8 = (short) pixel[8];
    short offset9 = (short) pixel[9];
    short offset10 = (short) pixel[10];
    short offset11 = (short) pixel[11];
    short offset12 = (short) pixel[12];
    short offset13 = (short) pixel[13];
    short offset14 = (short) pixel[14];
    short offset15 = (short) pixel[15];

    while(true)
    {
        const int cb = *ptr + b_test;
        const int c_b = *ptr - b_test;
        if(ptr[offset0] > cb)
          if(ptr[offset2] > cb)
            if(ptr[offset4] > cb)
              if(ptr[offset5] > cb)
                if(ptr[offset7] > cb)
                  if(ptr[offset3] > cb)
                    if(ptr[offset1] > cb)
                      if(ptr[offset6] > cb)
                        if(ptr[offset8] > cb)
                          goto is_a_corner;
                        else
                          if(ptr[offset15] > cb)
                            goto is_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset13] > cb)
                          if(ptr[offset14] > cb)
                            if(ptr[offset15] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset8] > cb)
                        if(ptr[offset9] > cb)
                          if(ptr[offset10] > cb)
                            if(ptr[offset6] > cb)
                              goto is_a_corner;
                            else
                              if(ptr[offset11] > cb)
                                if(ptr[offset12] > cb)
                                  if(ptr[offset13] > cb)
                                    if(ptr[offset14] > cb)
                                      if(ptr[offset15] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset10] > cb)
                      if(ptr[offset11] > cb)
                        if(ptr[offset12] > cb)
                          if(ptr[offset8] > cb)
                            if(ptr[offset9] > cb)
                              if(ptr[offset6] > cb)
                                goto is_a_corner;
                              else
                                if(ptr[offset13] > cb)
                                  if(ptr[offset14] > cb)
                                    if(ptr[offset15] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              if(ptr[offset1] > cb)
                                if(ptr[offset13] > cb)
                                  if(ptr[offset14] > cb)
                                    if(ptr[offset15] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset1] > cb)
                              if(ptr[offset13] > cb)
                                if(ptr[offset14] > cb)
                                  if(ptr[offset15] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else if(ptr[offset7] < c_b)
                  if(ptr[offset14] > cb)
                    if(ptr[offset15] > cb)
                      if(ptr[offset1] > cb)
                        if(ptr[offset3] > cb)
                          if(ptr[offset6] > cb)
                            goto is_a_corner;
                          else
                            if(ptr[offset13] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset10] > cb)
                            if(ptr[offset11] > cb)
                              if(ptr[offset12] > cb)
                                if(ptr[offset13] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset8] > cb)
                          if(ptr[offset9] > cb)
                            if(ptr[offset10] > cb)
                              if(ptr[offset11] > cb)
                                if(ptr[offset12] > cb)
                                  if(ptr[offset13] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else if(ptr[offset14] < c_b)
                    if(ptr[offset8] < c_b)
                      if(ptr[offset9] < c_b)
                        if(ptr[offset10] < c_b)
                          if(ptr[offset11] < c_b)
                            if(ptr[offset12] < c_b)
                              if(ptr[offset13] < c_b)
                                if(ptr[offset6] < c_b)
                                  goto is_a_corner;
                                else
                                  if(ptr[offset15] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  if(ptr[offset14] > cb)
                    if(ptr[offset15] > cb)
                      if(ptr[offset1] > cb)
                        if(ptr[offset3] > cb)
                          if(ptr[offset6] > cb)
                            goto is_a_corner;
                          else
                            if(ptr[offset13] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset10] > cb)
                            if(ptr[offset11] > cb)
                              if(ptr[offset12] > cb)
                                if(ptr[offset13] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset8] > cb)
                          if(ptr[offset9] > cb)
                            if(ptr[offset10] > cb)
                              if(ptr[offset11] > cb)
                                if(ptr[offset12] > cb)
                                  if(ptr[offset13] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
              else if(ptr[offset5] < c_b)
                if(ptr[offset12] > cb)
                  if(ptr[offset13] > cb)
                    if(ptr[offset14] > cb)
                      if(ptr[offset15] > cb)
                        if(ptr[offset1] > cb)
                          if(ptr[offset3] > cb)
                            goto is_a_corner;
                          else
                            if(ptr[offset10] > cb)
                              if(ptr[offset11] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset8] > cb)
                            if(ptr[offset9] > cb)
                              if(ptr[offset10] > cb)
                                if(ptr[offset11] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset6] > cb)
                          if(ptr[offset7] > cb)
                            if(ptr[offset8] > cb)
                              if(ptr[offset9] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else if(ptr[offset12] < c_b)
                  if(ptr[offset7] < c_b)
                    if(ptr[offset8] < c_b)
                      if(ptr[offset9] < c_b)
                        if(ptr[offset10] < c_b)
                          if(ptr[offset11] < c_b)
                            if(ptr[offset13] < c_b)
                              if(ptr[offset6] < c_b)
                                goto is_a_corner;
                              else
                                if(ptr[offset14] < c_b)
                                  if(ptr[offset15] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                if(ptr[offset12] > cb)
                  if(ptr[offset13] > cb)
                    if(ptr[offset14] > cb)
                      if(ptr[offset15] > cb)
                        if(ptr[offset1] > cb)
                          if(ptr[offset3] > cb)
                            goto is_a_corner;
                          else
                            if(ptr[offset10] > cb)
                              if(ptr[offset11] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset8] > cb)
                            if(ptr[offset9] > cb)
                              if(ptr[offset10] > cb)
                                if(ptr[offset11] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset6] > cb)
                          if(ptr[offset7] > cb)
                            if(ptr[offset8] > cb)
                              if(ptr[offset9] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else if(ptr[offset12] < c_b)
                  if(ptr[offset7] < c_b)
                    if(ptr[offset8] < c_b)
                      if(ptr[offset9] < c_b)
                        if(ptr[offset10] < c_b)
                          if(ptr[offset11] < c_b)
                            if(ptr[offset13] < c_b)
                              if(ptr[offset14] < c_b)
                                if(ptr[offset6] < c_b)
                                  goto is_a_corner;
                                else
                                  if(ptr[offset15] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
            else if(ptr[offset4] < c_b)
              if(ptr[offset11] > cb)
                if(ptr[offset12] > cb)
                  if(ptr[offset13] > cb)
                    if(ptr[offset10] > cb)
                      if(ptr[offset14] > cb)
                        if(ptr[offset15] > cb)
                          if(ptr[offset1] > cb)
                            goto is_a_corner;
                          else
                            if(ptr[offset8] > cb)
                              if(ptr[offset9] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset6] > cb)
                            if(ptr[offset7] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset9] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset5] > cb)
                          if(ptr[offset6] > cb)
                            if(ptr[offset7] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset9] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset1] > cb)
                        if(ptr[offset3] > cb)
                          if(ptr[offset14] > cb)
                            if(ptr[offset15] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else if(ptr[offset11] < c_b)
                if(ptr[offset7] < c_b)
                  if(ptr[offset8] < c_b)
                    if(ptr[offset9] < c_b)
                      if(ptr[offset10] < c_b)
                        if(ptr[offset6] < c_b)
                          if(ptr[offset5] < c_b)
                            if(ptr[offset3] < c_b)
                              goto is_a_corner;
                            else
                              if(ptr[offset12] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset12] < c_b)
                              if(ptr[offset13] < c_b)
                                if(ptr[offset14] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset12] < c_b)
                            if(ptr[offset13] < c_b)
                              if(ptr[offset14] < c_b)
                                if(ptr[offset15] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                goto is_not_a_corner;
            else
              if(ptr[offset11] > cb)
                if(ptr[offset12] > cb)
                  if(ptr[offset13] > cb)
                    if(ptr[offset10] > cb)
                      if(ptr[offset14] > cb)
                        if(ptr[offset15] > cb)
                          if(ptr[offset1] > cb)
                            goto is_a_corner;
                          else
                            if(ptr[offset8] > cb)
                              if(ptr[offset9] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset6] > cb)
                            if(ptr[offset7] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset9] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset5] > cb)
                          if(ptr[offset6] > cb)
                            if(ptr[offset7] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset9] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset1] > cb)
                        if(ptr[offset3] > cb)
                          if(ptr[offset14] > cb)
                            if(ptr[offset15] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else if(ptr[offset11] < c_b)
                if(ptr[offset7] < c_b)
                  if(ptr[offset8] < c_b)
                    if(ptr[offset9] < c_b)
                      if(ptr[offset10] < c_b)
                        if(ptr[offset12] < c_b)
                          if(ptr[offset13] < c_b)
                            if(ptr[offset6] < c_b)
                              if(ptr[offset5] < c_b)
                                goto is_a_corner;
                              else
                                if(ptr[offset14] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              if(ptr[offset14] < c_b)
                                if(ptr[offset15] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                goto is_not_a_corner;
          else if(ptr[offset2] < c_b)
            if(ptr[offset9] > cb)
              if(ptr[offset10] > cb)
                if(ptr[offset11] > cb)
                  if(ptr[offset8] > cb)
                    if(ptr[offset12] > cb)
                      if(ptr[offset13] > cb)
                        if(ptr[offset14] > cb)
                          if(ptr[offset15] > cb)
                            goto is_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset7] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset5] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset7] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset4] > cb)
                          if(ptr[offset5] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset7] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset3] > cb)
                        if(ptr[offset4] > cb)
                          if(ptr[offset5] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset7] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset1] > cb)
                      if(ptr[offset12] > cb)
                        if(ptr[offset13] > cb)
                          if(ptr[offset14] > cb)
                            if(ptr[offset15] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                goto is_not_a_corner;
            else if(ptr[offset9] < c_b)
              if(ptr[offset7] < c_b)
                if(ptr[offset8] < c_b)
                  if(ptr[offset6] < c_b)
                    if(ptr[offset5] < c_b)
                      if(ptr[offset4] < c_b)
                        if(ptr[offset3] < c_b)
                          if(ptr[offset1] < c_b)
                            goto is_a_corner;
                          else
                            if(ptr[offset10] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset10] < c_b)
                            if(ptr[offset11] < c_b)
                              if(ptr[offset12] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset10] < c_b)
                          if(ptr[offset11] < c_b)
                            if(ptr[offset12] < c_b)
                              if(ptr[offset13] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset10] < c_b)
                        if(ptr[offset11] < c_b)
                          if(ptr[offset12] < c_b)
                            if(ptr[offset13] < c_b)
                              if(ptr[offset14] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset10] < c_b)
                      if(ptr[offset11] < c_b)
                        if(ptr[offset12] < c_b)
                          if(ptr[offset13] < c_b)
                            if(ptr[offset14] < c_b)
                              if(ptr[offset15] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                goto is_not_a_corner;
            else
              goto is_not_a_corner;
          else
            if(ptr[offset9] > cb)
              if(ptr[offset10] > cb)
                if(ptr[offset11] > cb)
                  if(ptr[offset8] > cb)
                    if(ptr[offset12] > cb)
                      if(ptr[offset13] > cb)
                        if(ptr[offset14] > cb)
                          if(ptr[offset15] > cb)
                            goto is_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset7] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset5] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset7] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset4] > cb)
                          if(ptr[offset5] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset7] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset3] > cb)
                        if(ptr[offset4] > cb)
                          if(ptr[offset5] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset7] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset1] > cb)
                      if(ptr[offset12] > cb)
                        if(ptr[offset13] > cb)
                          if(ptr[offset14] > cb)
                            if(ptr[offset15] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                goto is_not_a_corner;
            else if(ptr[offset9] < c_b)
              if(ptr[offset7] < c_b)
                if(ptr[offset8] < c_b)
                  if(ptr[offset10] < c_b)
                    if(ptr[offset11] < c_b)
                      if(ptr[offset6] < c_b)
                        if(ptr[offset5] < c_b)
                          if(ptr[offset4] < c_b)
                            if(ptr[offset3] < c_b)
                              goto is_a_corner;
                            else
                              if(ptr[offset12] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset12] < c_b)
                              if(ptr[offset13] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset12] < c_b)
                            if(ptr[offset13] < c_b)
                              if(ptr[offset14] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset12] < c_b)
                          if(ptr[offset13] < c_b)
                            if(ptr[offset14] < c_b)
                              if(ptr[offset15] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                goto is_not_a_corner;
            else
              goto is_not_a_corner;
        else if(ptr[offset0] < c_b)
          if(ptr[offset2] > cb)
            if(ptr[offset9] > cb)
              if(ptr[offset7] > cb)
                if(ptr[offset8] > cb)
                  if(ptr[offset6] > cb)
                    if(ptr[offset5] > cb)
                      if(ptr[offset4] > cb)
                        if(ptr[offset3] > cb)
                          if(ptr[offset1] > cb)
                            goto is_a_corner;
                          else
                            if(ptr[offset10] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset10] > cb)
                            if(ptr[offset11] > cb)
                              if(ptr[offset12] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset10] > cb)
                          if(ptr[offset11] > cb)
                            if(ptr[offset12] > cb)
                              if(ptr[offset13] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset10] > cb)
                        if(ptr[offset11] > cb)
                          if(ptr[offset12] > cb)
                            if(ptr[offset13] > cb)
                              if(ptr[offset14] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset10] > cb)
                      if(ptr[offset11] > cb)
                        if(ptr[offset12] > cb)
                          if(ptr[offset13] > cb)
                            if(ptr[offset14] > cb)
                              if(ptr[offset15] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                goto is_not_a_corner;
            else if(ptr[offset9] < c_b)
              if(ptr[offset10] < c_b)
                if(ptr[offset11] < c_b)
                  if(ptr[offset8] < c_b)
                    if(ptr[offset12] < c_b)
                      if(ptr[offset13] < c_b)
                        if(ptr[offset14] < c_b)
                          if(ptr[offset15] < c_b)
                            goto is_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset7] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset5] < c_b)
                            if(ptr[offset6] < c_b)
                              if(ptr[offset7] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset4] < c_b)
                          if(ptr[offset5] < c_b)
                            if(ptr[offset6] < c_b)
                              if(ptr[offset7] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset3] < c_b)
                        if(ptr[offset4] < c_b)
                          if(ptr[offset5] < c_b)
                            if(ptr[offset6] < c_b)
                              if(ptr[offset7] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset1] < c_b)
                      if(ptr[offset12] < c_b)
                        if(ptr[offset13] < c_b)
                          if(ptr[offset14] < c_b)
                            if(ptr[offset15] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                goto is_not_a_corner;
            else
              goto is_not_a_corner;
          else if(ptr[offset2] < c_b)
            if(ptr[offset4] > cb)
              if(ptr[offset11] > cb)
                if(ptr[offset7] > cb)
                  if(ptr[offset8] > cb)
                    if(ptr[offset9] > cb)
                      if(ptr[offset10] > cb)
                        if(ptr[offset6] > cb)
                          if(ptr[offset5] > cb)
                            if(ptr[offset3] > cb)
                              goto is_a_corner;
                            else
                              if(ptr[offset12] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset12] > cb)
                              if(ptr[offset13] > cb)
                                if(ptr[offset14] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset12] > cb)
                            if(ptr[offset13] > cb)
                              if(ptr[offset14] > cb)
                                if(ptr[offset15] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else if(ptr[offset11] < c_b)
                if(ptr[offset12] < c_b)
                  if(ptr[offset13] < c_b)
                    if(ptr[offset10] < c_b)
                      if(ptr[offset14] < c_b)
                        if(ptr[offset15] < c_b)
                          if(ptr[offset1] < c_b)
                            goto is_a_corner;
                          else
                            if(ptr[offset8] < c_b)
                              if(ptr[offset9] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            if(ptr[offset7] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset9] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset5] < c_b)
                          if(ptr[offset6] < c_b)
                            if(ptr[offset7] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset9] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset1] < c_b)
                        if(ptr[offset3] < c_b)
                          if(ptr[offset14] < c_b)
                            if(ptr[offset15] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                goto is_not_a_corner;
            else if(ptr[offset4] < c_b)
              if(ptr[offset5] > cb)
                if(ptr[offset12] > cb)
                  if(ptr[offset7] > cb)
                    if(ptr[offset8] > cb)
                      if(ptr[offset9] > cb)
                        if(ptr[offset10] > cb)
                          if(ptr[offset11] > cb)
                            if(ptr[offset13] > cb)
                              if(ptr[offset6] > cb)
                                goto is_a_corner;
                              else
                                if(ptr[offset14] > cb)
                                  if(ptr[offset15] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else if(ptr[offset12] < c_b)
                  if(ptr[offset13] < c_b)
                    if(ptr[offset14] < c_b)
                      if(ptr[offset15] < c_b)
                        if(ptr[offset1] < c_b)
                          if(ptr[offset3] < c_b)
                            goto is_a_corner;
                          else
                            if(ptr[offset10] < c_b)
                              if(ptr[offset11] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset8] < c_b)
                            if(ptr[offset9] < c_b)
                              if(ptr[offset10] < c_b)
                                if(ptr[offset11] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset6] < c_b)
                          if(ptr[offset7] < c_b)
                            if(ptr[offset8] < c_b)
                              if(ptr[offset9] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else if(ptr[offset5] < c_b)
                if(ptr[offset7] > cb)
                  if(ptr[offset14] > cb)
                    if(ptr[offset8] > cb)
                      if(ptr[offset9] > cb)
                        if(ptr[offset10] > cb)
                          if(ptr[offset11] > cb)
                            if(ptr[offset12] > cb)
                              if(ptr[offset13] > cb)
                                if(ptr[offset6] > cb)
                                  goto is_a_corner;
                                else
                                  if(ptr[offset15] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else if(ptr[offset14] < c_b)
                    if(ptr[offset15] < c_b)
                      if(ptr[offset1] < c_b)
                        if(ptr[offset3] < c_b)
                          if(ptr[offset6] < c_b)
                            goto is_a_corner;
                          else
                            if(ptr[offset13] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset10] < c_b)
                            if(ptr[offset11] < c_b)
                              if(ptr[offset12] < c_b)
                                if(ptr[offset13] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset8] < c_b)
                          if(ptr[offset9] < c_b)
                            if(ptr[offset10] < c_b)
                              if(ptr[offset11] < c_b)
                                if(ptr[offset12] < c_b)
                                  if(ptr[offset13] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else if(ptr[offset7] < c_b)
                  if(ptr[offset3] < c_b)
                    if(ptr[offset1] < c_b)
                      if(ptr[offset6] < c_b)
                        if(ptr[offset8] < c_b)
                          goto is_a_corner;
                        else
                          if(ptr[offset15] < c_b)
                            goto is_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset13] < c_b)
                          if(ptr[offset14] < c_b)
                            if(ptr[offset15] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset8] < c_b)
                        if(ptr[offset9] < c_b)
                          if(ptr[offset10] < c_b)
                            if(ptr[offset6] < c_b)
                              goto is_a_corner;
                            else
                              if(ptr[offset11] < c_b)
                                if(ptr[offset12] < c_b)
                                  if(ptr[offset13] < c_b)
                                    if(ptr[offset14] < c_b)
                                      if(ptr[offset15] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset10] < c_b)
                      if(ptr[offset11] < c_b)
                        if(ptr[offset12] < c_b)
                          if(ptr[offset8] < c_b)
                            if(ptr[offset9] < c_b)
                              if(ptr[offset6] < c_b)
                                goto is_a_corner;
                              else
                                if(ptr[offset13] < c_b)
                                  if(ptr[offset14] < c_b)
                                    if(ptr[offset15] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              if(ptr[offset1] < c_b)
                                if(ptr[offset13] < c_b)
                                  if(ptr[offset14] < c_b)
                                    if(ptr[offset15] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset1] < c_b)
                              if(ptr[offset13] < c_b)
                                if(ptr[offset14] < c_b)
                                  if(ptr[offset15] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  if(ptr[offset14] < c_b)
                    if(ptr[offset15] < c_b)
                      if(ptr[offset1] < c_b)
                        if(ptr[offset3] < c_b)
                          if(ptr[offset6] < c_b)
                            goto is_a_corner;
                          else
                            if(ptr[offset13] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset10] < c_b)
                            if(ptr[offset11] < c_b)
                              if(ptr[offset12] < c_b)
                                if(ptr[offset13] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset8] < c_b)
                          if(ptr[offset9] < c_b)
                            if(ptr[offset10] < c_b)
                              if(ptr[offset11] < c_b)
                                if(ptr[offset12] < c_b)
                                  if(ptr[offset13] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
              else
                if(ptr[offset12] > cb)
                  if(ptr[offset7] > cb)
                    if(ptr[offset8] > cb)
                      if(ptr[offset9] > cb)
                        if(ptr[offset10] > cb)
                          if(ptr[offset11] > cb)
                            if(ptr[offset13] > cb)
                              if(ptr[offset14] > cb)
                                if(ptr[offset6] > cb)
                                  goto is_a_corner;
                                else
                                  if(ptr[offset15] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else if(ptr[offset12] < c_b)
                  if(ptr[offset13] < c_b)
                    if(ptr[offset14] < c_b)
                      if(ptr[offset15] < c_b)
                        if(ptr[offset1] < c_b)
                          if(ptr[offset3] < c_b)
                            goto is_a_corner;
                          else
                            if(ptr[offset10] < c_b)
                              if(ptr[offset11] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset8] < c_b)
                            if(ptr[offset9] < c_b)
                              if(ptr[offset10] < c_b)
                                if(ptr[offset11] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset6] < c_b)
                          if(ptr[offset7] < c_b)
                            if(ptr[offset8] < c_b)
                              if(ptr[offset9] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
            else
              if(ptr[offset11] > cb)
                if(ptr[offset7] > cb)
                  if(ptr[offset8] > cb)
                    if(ptr[offset9] > cb)
                      if(ptr[offset10] > cb)
                        if(ptr[offset12] > cb)
                          if(ptr[offset13] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset5] > cb)
                                goto is_a_corner;
                              else
                                if(ptr[offset14] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              if(ptr[offset14] > cb)
                                if(ptr[offset15] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else if(ptr[offset11] < c_b)
                if(ptr[offset12] < c_b)
                  if(ptr[offset13] < c_b)
                    if(ptr[offset10] < c_b)
                      if(ptr[offset14] < c_b)
                        if(ptr[offset15] < c_b)
                          if(ptr[offset1] < c_b)
                            goto is_a_corner;
                          else
                            if(ptr[offset8] < c_b)
                              if(ptr[offset9] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            if(ptr[offset7] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset9] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset5] < c_b)
                          if(ptr[offset6] < c_b)
                            if(ptr[offset7] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset9] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset1] < c_b)
                        if(ptr[offset3] < c_b)
                          if(ptr[offset14] < c_b)
                            if(ptr[offset15] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                goto is_not_a_corner;
          else
            if(ptr[offset9] > cb)
              if(ptr[offset7] > cb)
                if(ptr[offset8] > cb)
                  if(ptr[offset10] > cb)
                    if(ptr[offset11] > cb)
                      if(ptr[offset6] > cb)
                        if(ptr[offset5] > cb)
                          if(ptr[offset4] > cb)
                            if(ptr[offset3] > cb)
                              goto is_a_corner;
                            else
                              if(ptr[offset12] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset12] > cb)
                              if(ptr[offset13] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset12] > cb)
                            if(ptr[offset13] > cb)
                              if(ptr[offset14] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset12] > cb)
                          if(ptr[offset13] > cb)
                            if(ptr[offset14] > cb)
                              if(ptr[offset15] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                goto is_not_a_corner;
            else if(ptr[offset9] < c_b)
              if(ptr[offset10] < c_b)
                if(ptr[offset11] < c_b)
                  if(ptr[offset8] < c_b)
                    if(ptr[offset12] < c_b)
                      if(ptr[offset13] < c_b)
                        if(ptr[offset14] < c_b)
                          if(ptr[offset15] < c_b)
                            goto is_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset7] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset5] < c_b)
                            if(ptr[offset6] < c_b)
                              if(ptr[offset7] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset4] < c_b)
                          if(ptr[offset5] < c_b)
                            if(ptr[offset6] < c_b)
                              if(ptr[offset7] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset3] < c_b)
                        if(ptr[offset4] < c_b)
                          if(ptr[offset5] < c_b)
                            if(ptr[offset6] < c_b)
                              if(ptr[offset7] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset1] < c_b)
                      if(ptr[offset12] < c_b)
                        if(ptr[offset13] < c_b)
                          if(ptr[offset14] < c_b)
                            if(ptr[offset15] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                goto is_not_a_corner;
            else
              goto is_not_a_corner;
        else
          if(ptr[offset7] > cb)
            if(ptr[offset8] > cb)
              if(ptr[offset9] > cb)
                if(ptr[offset6] > cb)
                  if(ptr[offset5] > cb)
                    if(ptr[offset4] > cb)
                      if(ptr[offset3] > cb)
                        if(ptr[offset2] > cb)
                          if(ptr[offset1] > cb)
                            goto is_a_corner;
                          else
                            if(ptr[offset10] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset10] > cb)
                            if(ptr[offset11] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset10] > cb)
                          if(ptr[offset11] > cb)
                            if(ptr[offset12] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset10] > cb)
                        if(ptr[offset11] > cb)
                          if(ptr[offset12] > cb)
                            if(ptr[offset13] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset10] > cb)
                      if(ptr[offset11] > cb)
                        if(ptr[offset12] > cb)
                          if(ptr[offset13] > cb)
                            if(ptr[offset14] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  if(ptr[offset10] > cb)
                    if(ptr[offset11] > cb)
                      if(ptr[offset12] > cb)
                        if(ptr[offset13] > cb)
                          if(ptr[offset14] > cb)
                            if(ptr[offset15] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
              else
                goto is_not_a_corner;
            else
              goto is_not_a_corner;
          else if(ptr[offset7] < c_b)
            if(ptr[offset8] < c_b)
              if(ptr[offset9] < c_b)
                if(ptr[offset6] < c_b)
                  if(ptr[offset5] < c_b)
                    if(ptr[offset4] < c_b)
                      if(ptr[offset3] < c_b)
                        if(ptr[offset2] < c_b)
                          if(ptr[offset1] < c_b)
                            goto is_a_corner;
                          else
                            if(ptr[offset10] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset10] < c_b)
                            if(ptr[offset11] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset10] < c_b)
                          if(ptr[offset11] < c_b)
                            if(ptr[offset12] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset10] < c_b)
                        if(ptr[offset11] < c_b)
                          if(ptr[offset12] < c_b)
                            if(ptr[offset13] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset10] < c_b)
                      if(ptr[offset11] < c_b)
                        if(ptr[offset12] < c_b)
                          if(ptr[offset13] < c_b)
                            if(ptr[offset14] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  if(ptr[offset10] < c_b)
                    if(ptr[offset11] < c_b)
                      if(ptr[offset12] < c_b)
                        if(ptr[offset13] < c_b)
                          if(ptr[offset14] < c_b)
                            if(ptr[offset15] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
              else
                goto is_not_a_corner;
            else
              goto is_not_a_corner;
          else
            goto is_not_a_corner;

        is_a_corner:
            bmin = b_test;
            goto end;

        is_not_a_corner:
            bmax = b_test;
            goto end;

        end:

        if(bmin == bmax - 1 || bmin == bmax)
            return bmin;
        b_test = (bmin + bmax) / 2;
    }
}

// 12 pixel mask in diamond format
template<>
int agast_cornerScore<AgastFeatureDetector::AGAST_7_12d>(const uchar* ptr, const int pixel[], int threshold)
{
    int bmin = threshold;
    int bmax = 255;
    int b_test = (bmax + bmin)/2;

    short offset0 = (short) pixel[0];
    short offset1 = (short) pixel[1];
    short offset2 = (short) pixel[2];
    short offset3 = (short) pixel[3];
    short offset4 = (short) pixel[4];
    short offset5 = (short) pixel[5];
    short offset6 = (short) pixel[6];
    short offset7 = (short) pixel[7];
    short offset8 = (short) pixel[8];
    short offset9 = (short) pixel[9];
    short offset10 = (short) pixel[10];
    short offset11 = (short) pixel[11];

    while(true)
    {
        const int cb = *ptr + b_test;
        const int c_b = *ptr - b_test;
        if(ptr[offset0] > cb)
          if(ptr[offset5] > cb)
            if(ptr[offset2] > cb)
              if(ptr[offset9] > cb)
                if(ptr[offset1] > cb)
                  if(ptr[offset6] > cb)
                    if(ptr[offset3] > cb)
                      if(ptr[offset4] > cb)
                        goto is_a_corner;
                      else
                        if(ptr[offset10] > cb)
                          if(ptr[offset11] > cb)
                            goto is_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset8] > cb)
                        if(ptr[offset10] > cb)
                          if(ptr[offset11] > cb)
                            goto is_a_corner;
                          else
                            if(ptr[offset4] > cb)
                              if(ptr[offset7] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset11] > cb)
                      if(ptr[offset3] > cb)
                        if(ptr[offset4] > cb)
                          goto is_a_corner;
                        else
                          if(ptr[offset10] > cb)
                            goto is_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset8] > cb)
                          if(ptr[offset10] > cb)
                            goto is_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  if(ptr[offset6] > cb)
                    if(ptr[offset7] > cb)
                      if(ptr[offset8] > cb)
                        if(ptr[offset4] > cb)
                          if(ptr[offset3] > cb)
                            goto is_a_corner;
                          else
                            if(ptr[offset10] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset10] > cb)
                            if(ptr[offset11] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
              else
                if(ptr[offset3] > cb)
                  if(ptr[offset4] > cb)
                    if(ptr[offset1] > cb)
                      if(ptr[offset6] > cb)
                        goto is_a_corner;
                      else
                        if(ptr[offset11] > cb)
                          goto is_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset6] > cb)
                        if(ptr[offset7] > cb)
                          if(ptr[offset8] > cb)
                            goto is_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
            else
              if(ptr[offset9] > cb)
                if(ptr[offset7] > cb)
                  if(ptr[offset8] > cb)
                    if(ptr[offset1] > cb)
                      if(ptr[offset10] > cb)
                        if(ptr[offset11] > cb)
                          goto is_a_corner;
                        else
                          if(ptr[offset6] > cb)
                            if(ptr[offset4] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset6] > cb)
                          if(ptr[offset3] > cb)
                            if(ptr[offset4] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset6] > cb)
                        if(ptr[offset4] > cb)
                          if(ptr[offset3] > cb)
                            goto is_a_corner;
                          else
                            if(ptr[offset10] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset10] > cb)
                            if(ptr[offset11] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                goto is_not_a_corner;
          else
            if(ptr[offset5] < c_b)
              if(ptr[offset9] > cb)
                if(ptr[offset3] < c_b)
                  if(ptr[offset4] < c_b)
                    if(ptr[offset11] > cb)
                      if(ptr[offset1] > cb)
                        if(ptr[offset8] > cb)
                          if(ptr[offset10] > cb)
                            if(ptr[offset2] > cb)
                              goto is_a_corner;
                            else
                              if(ptr[offset7] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            if(ptr[offset2] < c_b)
                              if(ptr[offset7] < c_b)
                                if(ptr[offset8] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset6] > cb)
                          if(ptr[offset7] > cb)
                            if(ptr[offset8] > cb)
                              if(ptr[offset10] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            if(ptr[offset2] < c_b)
                              if(ptr[offset7] < c_b)
                                if(ptr[offset1] < c_b)
                                  goto is_a_corner;
                                else
                                  if(ptr[offset8] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                    else
                      if(ptr[offset2] < c_b)
                        if(ptr[offset7] < c_b)
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset8] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset11] > cb)
                      if(ptr[offset8] > cb)
                        if(ptr[offset10] > cb)
                          if(ptr[offset1] > cb)
                            if(ptr[offset2] > cb)
                              goto is_a_corner;
                            else
                              if(ptr[offset7] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset7] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  if(ptr[offset11] > cb)
                    if(ptr[offset10] > cb)
                      if(ptr[offset3] > cb)
                        if(ptr[offset1] > cb)
                          if(ptr[offset2] > cb)
                            goto is_a_corner;
                          else
                            if(ptr[offset7] > cb)
                              if(ptr[offset8] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset6] > cb)
                            if(ptr[offset7] > cb)
                              if(ptr[offset8] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset8] > cb)
                          if(ptr[offset1] > cb)
                            if(ptr[offset2] > cb)
                              goto is_a_corner;
                            else
                              if(ptr[offset7] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset7] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
              else
                if(ptr[offset9] < c_b)
                  if(ptr[offset2] > cb)
                    if(ptr[offset1] > cb)
                      if(ptr[offset4] > cb)
                        if(ptr[offset10] > cb)
                          if(ptr[offset3] > cb)
                            if(ptr[offset11] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            if(ptr[offset7] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset11] < c_b)
                                  if(ptr[offset10] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset6] < c_b)
                          if(ptr[offset7] < c_b)
                            if(ptr[offset8] < c_b)
                              if(ptr[offset10] < c_b)
                                if(ptr[offset4] < c_b)
                                  goto is_a_corner;
                                else
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset6] < c_b)
                        if(ptr[offset7] < c_b)
                          if(ptr[offset8] < c_b)
                            if(ptr[offset4] < c_b)
                              if(ptr[offset3] < c_b)
                                goto is_a_corner;
                              else
                                if(ptr[offset10] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              if(ptr[offset10] < c_b)
                                if(ptr[offset11] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset6] < c_b)
                      if(ptr[offset7] < c_b)
                        if(ptr[offset8] < c_b)
                          if(ptr[offset4] < c_b)
                            if(ptr[offset3] < c_b)
                              goto is_a_corner;
                            else
                              if(ptr[offset10] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset10] < c_b)
                              if(ptr[offset11] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset2] < c_b)
                            if(ptr[offset1] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  if(ptr[offset2] > cb)
                    if(ptr[offset1] > cb)
                      if(ptr[offset3] > cb)
                        if(ptr[offset4] > cb)
                          if(ptr[offset10] > cb)
                            if(ptr[offset11] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    if(ptr[offset2] < c_b)
                      if(ptr[offset3] < c_b)
                        if(ptr[offset4] < c_b)
                          if(ptr[offset7] < c_b)
                            if(ptr[offset1] < c_b)
                              if(ptr[offset6] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
            else
              if(ptr[offset2] > cb)
                if(ptr[offset10] > cb)
                  if(ptr[offset11] > cb)
                    if(ptr[offset9] > cb)
                      if(ptr[offset1] > cb)
                        if(ptr[offset3] > cb)
                          goto is_a_corner;
                        else
                          if(ptr[offset8] > cb)
                            goto is_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset6] > cb)
                          if(ptr[offset7] > cb)
                            if(ptr[offset8] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset1] > cb)
                        if(ptr[offset3] > cb)
                          if(ptr[offset4] > cb)
                            goto is_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                if(ptr[offset9] > cb)
                  if(ptr[offset7] > cb)
                    if(ptr[offset8] > cb)
                      if(ptr[offset10] > cb)
                        if(ptr[offset11] > cb)
                          if(ptr[offset1] > cb)
                            goto is_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
        else if(ptr[offset0] < c_b)
          if(ptr[offset2] > cb)
            if(ptr[offset5] > cb)
              if(ptr[offset7] > cb)
                if(ptr[offset6] > cb)
                  if(ptr[offset4] > cb)
                    if(ptr[offset3] > cb)
                      if(ptr[offset1] > cb)
                        goto is_a_corner;
                      else
                        if(ptr[offset8] > cb)
                          goto is_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset9] > cb)
                        if(ptr[offset8] > cb)
                          if(ptr[offset10] > cb)
                            goto is_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset9] > cb)
                      if(ptr[offset8] > cb)
                        if(ptr[offset10] > cb)
                          if(ptr[offset11] > cb)
                            goto is_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                if(ptr[offset9] < c_b)
                  if(ptr[offset8] < c_b)
                    if(ptr[offset10] < c_b)
                      if(ptr[offset11] < c_b)
                        if(ptr[offset7] < c_b)
                          if(ptr[offset1] < c_b)
                            goto is_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
            else
              if(ptr[offset9] < c_b)
                if(ptr[offset7] < c_b)
                  if(ptr[offset8] < c_b)
                    if(ptr[offset5] < c_b)
                      if(ptr[offset1] < c_b)
                        if(ptr[offset10] < c_b)
                          if(ptr[offset11] < c_b)
                            goto is_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset4] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            if(ptr[offset3] < c_b)
                              if(ptr[offset4] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset6] < c_b)
                          if(ptr[offset4] < c_b)
                            if(ptr[offset3] < c_b)
                              goto is_a_corner;
                            else
                              if(ptr[offset10] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset10] < c_b)
                              if(ptr[offset11] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset10] < c_b)
                        if(ptr[offset11] < c_b)
                          if(ptr[offset1] < c_b)
                            goto is_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                goto is_not_a_corner;
          else
            if(ptr[offset2] < c_b)
              if(ptr[offset9] > cb)
                if(ptr[offset5] > cb)
                  if(ptr[offset1] < c_b)
                    if(ptr[offset4] < c_b)
                      if(ptr[offset10] < c_b)
                        if(ptr[offset3] < c_b)
                          if(ptr[offset11] < c_b)
                            goto is_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        if(ptr[offset6] > cb)
                          if(ptr[offset7] > cb)
                            if(ptr[offset8] > cb)
                              if(ptr[offset11] > cb)
                                if(ptr[offset10] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset6] > cb)
                        if(ptr[offset7] > cb)
                          if(ptr[offset8] > cb)
                            if(ptr[offset10] > cb)
                              if(ptr[offset4] > cb)
                                goto is_a_corner;
                              else
                                if(ptr[offset11] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset6] > cb)
                      if(ptr[offset7] > cb)
                        if(ptr[offset8] > cb)
                          if(ptr[offset4] > cb)
                            if(ptr[offset3] > cb)
                              goto is_a_corner;
                            else
                              if(ptr[offset10] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset10] > cb)
                              if(ptr[offset11] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  if(ptr[offset3] < c_b)
                    if(ptr[offset4] < c_b)
                      if(ptr[offset5] < c_b)
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] < c_b)
                            goto is_a_corner;
                          else
                            if(ptr[offset11] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            if(ptr[offset7] < c_b)
                              if(ptr[offset8] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset1] < c_b)
                          if(ptr[offset10] < c_b)
                            if(ptr[offset11] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
              else
                if(ptr[offset9] < c_b)
                  if(ptr[offset5] < c_b)
                    if(ptr[offset1] < c_b)
                      if(ptr[offset6] < c_b)
                        if(ptr[offset3] < c_b)
                          if(ptr[offset4] < c_b)
                            goto is_a_corner;
                          else
                            if(ptr[offset10] < c_b)
                              if(ptr[offset11] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset8] < c_b)
                            if(ptr[offset10] < c_b)
                              if(ptr[offset11] < c_b)
                                goto is_a_corner;
                              else
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset7] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset11] < c_b)
                          if(ptr[offset3] < c_b)
                            if(ptr[offset4] < c_b)
                              goto is_a_corner;
                            else
                              if(ptr[offset10] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset8] < c_b)
                              if(ptr[offset10] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset6] < c_b)
                        if(ptr[offset7] < c_b)
                          if(ptr[offset8] < c_b)
                            if(ptr[offset4] < c_b)
                              if(ptr[offset3] < c_b)
                                goto is_a_corner;
                              else
                                if(ptr[offset10] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              if(ptr[offset10] < c_b)
                                if(ptr[offset11] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset10] < c_b)
                      if(ptr[offset11] < c_b)
                        if(ptr[offset1] < c_b)
                          if(ptr[offset3] < c_b)
                            goto is_a_corner;
                          else
                            if(ptr[offset8] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            if(ptr[offset7] < c_b)
                              if(ptr[offset8] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  if(ptr[offset3] < c_b)
                    if(ptr[offset4] < c_b)
                      if(ptr[offset5] < c_b)
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] < c_b)
                            goto is_a_corner;
                          else
                            if(ptr[offset11] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            if(ptr[offset7] < c_b)
                              if(ptr[offset8] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset1] < c_b)
                          if(ptr[offset10] < c_b)
                            if(ptr[offset11] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
            else
              if(ptr[offset9] < c_b)
                if(ptr[offset7] < c_b)
                  if(ptr[offset8] < c_b)
                    if(ptr[offset5] < c_b)
                      if(ptr[offset1] < c_b)
                        if(ptr[offset10] < c_b)
                          if(ptr[offset11] < c_b)
                            goto is_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset4] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            if(ptr[offset3] < c_b)
                              if(ptr[offset4] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset6] < c_b)
                          if(ptr[offset4] < c_b)
                            if(ptr[offset3] < c_b)
                              goto is_a_corner;
                            else
                              if(ptr[offset10] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset10] < c_b)
                              if(ptr[offset11] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset10] < c_b)
                        if(ptr[offset11] < c_b)
                          if(ptr[offset1] < c_b)
                            goto is_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                if(ptr[offset5] > cb)
                  if(ptr[offset9] > cb)
                    if(ptr[offset6] > cb)
                      if(ptr[offset7] > cb)
                        if(ptr[offset8] > cb)
                          if(ptr[offset4] > cb)
                            if(ptr[offset3] > cb)
                              goto is_a_corner;
                            else
                              if(ptr[offset10] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset10] > cb)
                              if(ptr[offset11] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
        else
          if(ptr[offset5] > cb)
            if(ptr[offset9] > cb)
              if(ptr[offset6] > cb)
                if(ptr[offset7] > cb)
                  if(ptr[offset4] > cb)
                    if(ptr[offset3] > cb)
                      if(ptr[offset8] > cb)
                        goto is_a_corner;
                      else
                        if(ptr[offset1] > cb)
                          if(ptr[offset2] > cb)
                            goto is_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset8] > cb)
                        if(ptr[offset10] > cb)
                          goto is_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset11] > cb)
                      if(ptr[offset8] > cb)
                        if(ptr[offset10] > cb)
                          goto is_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                goto is_not_a_corner;
            else
              if(ptr[offset2] > cb)
                if(ptr[offset3] > cb)
                  if(ptr[offset4] > cb)
                    if(ptr[offset7] > cb)
                      if(ptr[offset1] > cb)
                        if(ptr[offset6] > cb)
                          goto is_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        if(ptr[offset6] > cb)
                          if(ptr[offset8] > cb)
                            goto is_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                goto is_not_a_corner;
          else
            if(ptr[offset5] < c_b)
              if(ptr[offset9] < c_b)
                if(ptr[offset6] < c_b)
                  if(ptr[offset7] < c_b)
                    if(ptr[offset4] < c_b)
                      if(ptr[offset3] < c_b)
                        if(ptr[offset8] < c_b)
                          goto is_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset2] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset8] < c_b)
                          if(ptr[offset10] < c_b)
                            goto is_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset11] < c_b)
                        if(ptr[offset8] < c_b)
                          if(ptr[offset10] < c_b)
                            goto is_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                if(ptr[offset2] < c_b)
                  if(ptr[offset3] < c_b)
                    if(ptr[offset4] < c_b)
                      if(ptr[offset7] < c_b)
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] < c_b)
                            goto is_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            if(ptr[offset8] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
            else
              goto is_not_a_corner;

        is_a_corner:
            bmin = b_test;
            goto end;

        is_not_a_corner:
            bmax = b_test;
            goto end;

        end:

        if(bmin == bmax - 1 || bmin == bmax)
            return bmin;
        b_test = (bmin + bmax) / 2;
    }
}

//12 pixel mask in square format
template<>
int agast_cornerScore<AgastFeatureDetector::AGAST_7_12s>(const uchar* ptr, const int pixel[], int threshold)
{
    int bmin = threshold;
    int bmax = 255;
    int b_test = (bmax + bmin)/2;

    short offset0 = (short) pixel[0];
    short offset1 = (short) pixel[1];
    short offset2 = (short) pixel[2];
    short offset3 = (short) pixel[3];
    short offset4 = (short) pixel[4];
    short offset5 = (short) pixel[5];
    short offset6 = (short) pixel[6];
    short offset7 = (short) pixel[7];
    short offset8 = (short) pixel[8];
    short offset9 = (short) pixel[9];
    short offset10 = (short) pixel[10];
    short offset11 = (short) pixel[11];

    while(true)
    {
        const int cb = *ptr + b_test;
        const int c_b = *ptr - b_test;
        if(ptr[offset0] > cb)
          if(ptr[offset5] > cb)
            if(ptr[offset2] < c_b)
              if(ptr[offset7] > cb)
                if(ptr[offset9] < c_b)
                  goto is_not_a_corner;
                else
                  if(ptr[offset9] > cb)
                    if(ptr[offset1] < c_b)
                      if(ptr[offset6] < c_b)
                        goto is_not_a_corner;
                      else
                        if(ptr[offset6] > cb)
                          if(ptr[offset8] > cb)
                            if(ptr[offset4] > cb)
                              if(ptr[offset3] > cb)
                                goto is_a_corner;
                              else
                                if(ptr[offset10] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              if(ptr[offset10] > cb)
                                if(ptr[offset11] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset1] > cb)
                        if(ptr[offset6] < c_b)
                          if(ptr[offset8] > cb)
                            if(ptr[offset10] > cb)
                              if(ptr[offset11] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          if(ptr[offset6] > cb)
                            if(ptr[offset8] > cb)
                              if(ptr[offset4] > cb)
                                if(ptr[offset3] > cb)
                                  goto is_a_corner;
                                else
                                  if(ptr[offset10] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset8] > cb)
                              if(ptr[offset10] > cb)
                                if(ptr[offset11] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                      else
                        if(ptr[offset6] < c_b)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset6] > cb)
                            if(ptr[offset8] > cb)
                              if(ptr[offset4] > cb)
                                if(ptr[offset3] > cb)
                                  goto is_a_corner;
                                else
                                  if(ptr[offset10] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
              else
                goto is_not_a_corner;
            else
              if(ptr[offset2] > cb)
                if(ptr[offset7] < c_b)
                  if(ptr[offset9] < c_b)
                    if(ptr[offset1] < c_b)
                      goto is_not_a_corner;
                    else
                      if(ptr[offset1] > cb)
                        if(ptr[offset6] > cb)
                          if(ptr[offset3] > cb)
                            if(ptr[offset4] > cb)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            if(ptr[offset3] > cb)
                              if(ptr[offset4] > cb)
                                if(ptr[offset11] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset3] > cb)
                              if(ptr[offset4] > cb)
                                if(ptr[offset11] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset9] > cb)
                      if(ptr[offset1] < c_b)
                        goto is_not_a_corner;
                      else
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] < c_b)
                            if(ptr[offset11] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  goto is_a_corner;
                                else
                                  if(ptr[offset10] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  goto is_a_corner;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              if(ptr[offset11] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset10] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset1] < c_b)
                        goto is_not_a_corner;
                      else
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] > cb)
                            if(ptr[offset3] > cb)
                              if(ptr[offset4] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                else
                  if(ptr[offset9] < c_b)
                    if(ptr[offset7] > cb)
                      if(ptr[offset1] < c_b)
                        if(ptr[offset6] < c_b)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset6] > cb)
                            if(ptr[offset3] > cb)
                              if(ptr[offset4] > cb)
                                if(ptr[offset8] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] > cb)
                            if(ptr[offset3] > cb)
                              if(ptr[offset4] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset8] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                    else
                      if(ptr[offset1] < c_b)
                        goto is_not_a_corner;
                      else
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] > cb)
                            if(ptr[offset3] > cb)
                              if(ptr[offset4] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                  else
                    if(ptr[offset7] > cb)
                      if(ptr[offset9] > cb)
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset3] > cb)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] < c_b)
                              if(ptr[offset11] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset10] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset4] > cb)
                                        goto is_a_corner;
                                      else
                                        if(ptr[offset11] > cb)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                if(ptr[offset11] > cb)
                                  if(ptr[offset3] > cb)
                                    if(ptr[offset4] > cb)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset10] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset8] > cb)
                                      if(ptr[offset10] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset3] > cb)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset10] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                      else
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset8] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset8] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                    else
                      if(ptr[offset9] > cb)
                        if(ptr[offset1] < c_b)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] < c_b)
                              if(ptr[offset11] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset10] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                if(ptr[offset11] > cb)
                                  if(ptr[offset3] > cb)
                                    if(ptr[offset4] > cb)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset10] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset8] > cb)
                                      if(ptr[offset10] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset1] < c_b)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
              else
                if(ptr[offset7] > cb)
                  if(ptr[offset9] < c_b)
                    goto is_not_a_corner;
                  else
                    if(ptr[offset9] > cb)
                      if(ptr[offset1] < c_b)
                        if(ptr[offset6] < c_b)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset6] > cb)
                            if(ptr[offset8] > cb)
                              if(ptr[offset4] > cb)
                                if(ptr[offset3] > cb)
                                  goto is_a_corner;
                                else
                                  if(ptr[offset10] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] < c_b)
                            if(ptr[offset8] > cb)
                              if(ptr[offset10] > cb)
                                if(ptr[offset11] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset3] > cb)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset3] > cb)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  goto is_not_a_corner;
          else
            if(ptr[offset5] < c_b)
              if(ptr[offset9] < c_b)
                if(ptr[offset7] > cb)
                  if(ptr[offset2] < c_b)
                    goto is_not_a_corner;
                  else
                    if(ptr[offset2] > cb)
                      if(ptr[offset1] < c_b)
                        goto is_not_a_corner;
                      else
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] > cb)
                            if(ptr[offset3] > cb)
                              if(ptr[offset4] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  if(ptr[offset7] < c_b)
                    if(ptr[offset2] < c_b)
                      if(ptr[offset1] > cb)
                        if(ptr[offset6] > cb)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            if(ptr[offset8] < c_b)
                              if(ptr[offset4] < c_b)
                                if(ptr[offset3] < c_b)
                                  goto is_a_corner;
                                else
                                  if(ptr[offset10] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset4] < c_b)
                                if(ptr[offset3] < c_b)
                                  goto is_a_corner;
                                else
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset10] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset3] < c_b)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                    else
                      if(ptr[offset2] > cb)
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset3] < c_b)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset3] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    if(ptr[offset8] < c_b)
                                      if(ptr[offset11] < c_b)
                                        if(ptr[offset10] < c_b)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset4] < c_b)
                                        goto is_a_corner;
                                      else
                                        if(ptr[offset11] < c_b)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                    else
                                      if(ptr[offset3] < c_b)
                                        if(ptr[offset4] < c_b)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                      else
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset3] < c_b)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] > cb)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                  else
                    if(ptr[offset2] < c_b)
                      goto is_not_a_corner;
                    else
                      if(ptr[offset2] > cb)
                        if(ptr[offset1] < c_b)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
              else
                if(ptr[offset9] > cb)
                  if(ptr[offset7] < c_b)
                    if(ptr[offset2] > cb)
                      if(ptr[offset1] < c_b)
                        goto is_not_a_corner;
                      else
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] > cb)
                            if(ptr[offset10] > cb)
                              if(ptr[offset11] > cb)
                                if(ptr[offset3] > cb)
                                  goto is_a_corner;
                                else
                                  if(ptr[offset8] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset10] > cb)
                                if(ptr[offset11] > cb)
                                  if(ptr[offset3] > cb)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset8] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset10] > cb)
                                if(ptr[offset11] > cb)
                                  if(ptr[offset3] > cb)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset8] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset2] < c_b)
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] > cb)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset8] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset8] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset7] > cb)
                      if(ptr[offset2] < c_b)
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                      else
                        if(ptr[offset2] > cb)
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] < c_b)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset1] > cb)
                              if(ptr[offset6] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    if(ptr[offset3] > cb)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset8] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      if(ptr[offset3] > cb)
                                        goto is_a_corner;
                                      else
                                        if(ptr[offset8] > cb)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      if(ptr[offset3] > cb)
                                        goto is_a_corner;
                                      else
                                        if(ptr[offset8] > cb)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                goto is_not_a_corner;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] < c_b)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset1] > cb)
                              if(ptr[offset6] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                goto is_not_a_corner;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                    else
                      if(ptr[offset2] < c_b)
                        goto is_not_a_corner;
                      else
                        if(ptr[offset2] > cb)
                          if(ptr[offset1] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset1] > cb)
                              if(ptr[offset6] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    if(ptr[offset3] > cb)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset8] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      if(ptr[offset3] > cb)
                                        goto is_a_corner;
                                      else
                                        if(ptr[offset8] > cb)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      if(ptr[offset3] > cb)
                                        goto is_a_corner;
                                      else
                                        if(ptr[offset8] > cb)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                else
                  if(ptr[offset2] < c_b)
                    if(ptr[offset7] > cb)
                      goto is_not_a_corner;
                    else
                      if(ptr[offset7] < c_b)
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] > cb)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset8] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset8] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset2] > cb)
                      if(ptr[offset7] > cb)
                        if(ptr[offset1] < c_b)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset7] < c_b)
                          if(ptr[offset1] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset1] > cb)
                              if(ptr[offset6] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset3] > cb)
                                    if(ptr[offset4] > cb)
                                      if(ptr[offset10] > cb)
                                        if(ptr[offset11] > cb)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  if(ptr[offset3] > cb)
                                    if(ptr[offset4] > cb)
                                      if(ptr[offset10] > cb)
                                        if(ptr[offset11] > cb)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset1] > cb)
                              if(ptr[offset6] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset3] > cb)
                                    if(ptr[offset4] > cb)
                                      if(ptr[offset10] > cb)
                                        if(ptr[offset11] > cb)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  if(ptr[offset3] > cb)
                                    if(ptr[offset4] > cb)
                                      if(ptr[offset10] > cb)
                                        if(ptr[offset11] > cb)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
            else
              if(ptr[offset2] < c_b)
                if(ptr[offset7] > cb)
                  if(ptr[offset9] < c_b)
                    goto is_not_a_corner;
                  else
                    if(ptr[offset9] > cb)
                      if(ptr[offset1] < c_b)
                        if(ptr[offset6] < c_b)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset6] > cb)
                            if(ptr[offset8] > cb)
                              if(ptr[offset10] > cb)
                                if(ptr[offset11] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] > cb)
                            if(ptr[offset8] > cb)
                              if(ptr[offset10] > cb)
                                if(ptr[offset11] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                if(ptr[offset2] > cb)
                  if(ptr[offset7] < c_b)
                    if(ptr[offset9] < c_b)
                      if(ptr[offset1] < c_b)
                        goto is_not_a_corner;
                      else
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] > cb)
                            if(ptr[offset3] > cb)
                              if(ptr[offset4] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset9] > cb)
                        if(ptr[offset1] < c_b)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset10] > cb)
                                if(ptr[offset11] > cb)
                                  if(ptr[offset3] > cb)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset8] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    if(ptr[offset3] > cb)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset8] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    if(ptr[offset3] > cb)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset8] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset1] < c_b)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                  else
                    if(ptr[offset9] < c_b)
                      if(ptr[offset7] > cb)
                        if(ptr[offset1] < c_b)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset1] < c_b)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                    else
                      if(ptr[offset7] > cb)
                        if(ptr[offset9] > cb)
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] < c_b)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset1] > cb)
                              if(ptr[offset6] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    if(ptr[offset3] > cb)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset8] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      if(ptr[offset3] > cb)
                                        goto is_a_corner;
                                      else
                                        if(ptr[offset8] > cb)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      if(ptr[offset3] > cb)
                                        goto is_a_corner;
                                      else
                                        if(ptr[offset8] > cb)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                goto is_not_a_corner;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset1] > cb)
                              if(ptr[offset6] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset3] > cb)
                                    if(ptr[offset4] > cb)
                                      if(ptr[offset10] > cb)
                                        if(ptr[offset11] > cb)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  if(ptr[offset3] > cb)
                                    if(ptr[offset4] > cb)
                                      if(ptr[offset10] > cb)
                                        if(ptr[offset11] > cb)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                      else
                        if(ptr[offset9] > cb)
                          if(ptr[offset1] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset1] > cb)
                              if(ptr[offset6] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    if(ptr[offset3] > cb)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset8] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      if(ptr[offset3] > cb)
                                        goto is_a_corner;
                                      else
                                        if(ptr[offset8] > cb)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      if(ptr[offset3] > cb)
                                        goto is_a_corner;
                                      else
                                        if(ptr[offset8] > cb)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset1] > cb)
                              if(ptr[offset6] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset3] > cb)
                                    if(ptr[offset4] > cb)
                                      if(ptr[offset10] > cb)
                                        if(ptr[offset11] > cb)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  if(ptr[offset3] > cb)
                                    if(ptr[offset4] > cb)
                                      if(ptr[offset10] > cb)
                                        if(ptr[offset11] > cb)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                else
                  if(ptr[offset7] > cb)
                    if(ptr[offset9] < c_b)
                      goto is_not_a_corner;
                    else
                      if(ptr[offset9] > cb)
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
        else if(ptr[offset0] < c_b)
          if(ptr[offset5] < c_b)
            if(ptr[offset9] > cb)
              if(ptr[offset2] > cb)
                goto is_not_a_corner;
              else
                if(ptr[offset2] < c_b)
                  if(ptr[offset7] > cb)
                    if(ptr[offset1] > cb)
                      goto is_not_a_corner;
                    else
                      if(ptr[offset1] < c_b)
                        if(ptr[offset6] < c_b)
                          if(ptr[offset3] < c_b)
                            if(ptr[offset4] < c_b)
                              goto is_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                        else
                          if(ptr[offset6] > cb)
                            if(ptr[offset3] < c_b)
                              if(ptr[offset4] < c_b)
                                if(ptr[offset11] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset3] < c_b)
                              if(ptr[offset4] < c_b)
                                if(ptr[offset11] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset7] < c_b)
                      if(ptr[offset1] > cb)
                        if(ptr[offset6] > cb)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            if(ptr[offset3] < c_b)
                              if(ptr[offset4] < c_b)
                                if(ptr[offset8] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] < c_b)
                            if(ptr[offset3] < c_b)
                              if(ptr[offset4] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                        else
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset8] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                    else
                      if(ptr[offset1] > cb)
                        goto is_not_a_corner;
                      else
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] < c_b)
                            if(ptr[offset3] < c_b)
                              if(ptr[offset4] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                else
                  goto is_not_a_corner;
            else
              if(ptr[offset9] < c_b)
                if(ptr[offset7] > cb)
                  if(ptr[offset2] > cb)
                    goto is_not_a_corner;
                  else
                    if(ptr[offset2] < c_b)
                      if(ptr[offset1] > cb)
                        goto is_not_a_corner;
                      else
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] > cb)
                            if(ptr[offset11] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  goto is_a_corner;
                                else
                                  if(ptr[offset10] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  goto is_a_corner;
                                else
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              if(ptr[offset11] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset10] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  if(ptr[offset7] < c_b)
                    if(ptr[offset2] > cb)
                      if(ptr[offset1] > cb)
                        if(ptr[offset6] > cb)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            if(ptr[offset8] < c_b)
                              if(ptr[offset4] < c_b)
                                if(ptr[offset3] < c_b)
                                  goto is_a_corner;
                                else
                                  if(ptr[offset10] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] > cb)
                            if(ptr[offset8] < c_b)
                              if(ptr[offset10] < c_b)
                                if(ptr[offset11] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset3] < c_b)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset8] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                        else
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset3] < c_b)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                    else
                      if(ptr[offset2] < c_b)
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset3] < c_b)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] > cb)
                              if(ptr[offset11] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset10] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset4] < c_b)
                                        goto is_a_corner;
                                      else
                                        if(ptr[offset11] < c_b)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                if(ptr[offset11] < c_b)
                                  if(ptr[offset3] < c_b)
                                    if(ptr[offset4] < c_b)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset8] < c_b)
                                      if(ptr[offset10] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                      else
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset3] < c_b)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] > cb)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                  else
                    if(ptr[offset2] > cb)
                      goto is_not_a_corner;
                    else
                      if(ptr[offset2] < c_b)
                        if(ptr[offset1] > cb)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] > cb)
                              if(ptr[offset11] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset10] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                if(ptr[offset11] < c_b)
                                  if(ptr[offset3] < c_b)
                                    if(ptr[offset4] < c_b)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset8] < c_b)
                                      if(ptr[offset10] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
              else
                if(ptr[offset2] > cb)
                  goto is_not_a_corner;
                else
                  if(ptr[offset2] < c_b)
                    if(ptr[offset7] > cb)
                      if(ptr[offset1] > cb)
                        goto is_not_a_corner;
                      else
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] < c_b)
                            if(ptr[offset3] < c_b)
                              if(ptr[offset4] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset7] < c_b)
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset8] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset8] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                      else
                        if(ptr[offset1] > cb)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
          else
            if(ptr[offset5] > cb)
              if(ptr[offset2] > cb)
                if(ptr[offset7] < c_b)
                  if(ptr[offset9] > cb)
                    goto is_not_a_corner;
                  else
                    if(ptr[offset9] < c_b)
                      if(ptr[offset1] > cb)
                        if(ptr[offset6] > cb)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            if(ptr[offset8] < c_b)
                              if(ptr[offset10] < c_b)
                                if(ptr[offset11] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] > cb)
                            if(ptr[offset8] < c_b)
                              if(ptr[offset10] < c_b)
                                if(ptr[offset11] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset8] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                        else
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  if(ptr[offset7] > cb)
                    if(ptr[offset9] < c_b)
                      if(ptr[offset1] > cb)
                        if(ptr[offset6] < c_b)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset6] > cb)
                            if(ptr[offset3] > cb)
                              if(ptr[offset4] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset8] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset8] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                    else
                      if(ptr[offset9] > cb)
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset3] > cb)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] < c_b)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset3] > cb)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset8] > cb)
                                      if(ptr[offset10] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset3] > cb)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset10] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                      else
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] < c_b)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset8] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset8] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
              else
                if(ptr[offset2] < c_b)
                  if(ptr[offset7] < c_b)
                    if(ptr[offset9] > cb)
                      if(ptr[offset1] > cb)
                        goto is_not_a_corner;
                      else
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] > cb)
                            if(ptr[offset3] < c_b)
                              if(ptr[offset4] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset9] < c_b)
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] > cb)
                              if(ptr[offset10] < c_b)
                                if(ptr[offset11] < c_b)
                                  if(ptr[offset3] < c_b)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset8] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset8] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset8] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                      else
                        if(ptr[offset1] > cb)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                  else
                    if(ptr[offset7] > cb)
                      if(ptr[offset9] < c_b)
                        if(ptr[offset1] > cb)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] > cb)
                              if(ptr[offset10] < c_b)
                                if(ptr[offset11] < c_b)
                                  if(ptr[offset3] < c_b)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset8] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset8] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset8] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset9] > cb)
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] < c_b)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset3] > cb)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset10] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset1] < c_b)
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset8] > cb)
                                        if(ptr[offset11] > cb)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      if(ptr[offset3] < c_b)
                                        if(ptr[offset11] < c_b)
                                          if(ptr[offset10] < c_b)
                                            goto is_a_corner;
                                          else
                                            goto is_not_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset8] > cb)
                                      if(ptr[offset10] > cb)
                                        if(ptr[offset4] > cb)
                                          goto is_a_corner;
                                        else
                                          if(ptr[offset11] > cb)
                                            goto is_a_corner;
                                          else
                                            goto is_not_a_corner;
                                      else
                                        if(ptr[offset3] > cb)
                                          if(ptr[offset4] > cb)
                                            goto is_a_corner;
                                          else
                                            goto is_not_a_corner;
                                        else
                                          goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset3] < c_b)
                                    if(ptr[offset4] < c_b)
                                      if(ptr[offset10] < c_b)
                                        if(ptr[offset11] < c_b)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                goto is_not_a_corner;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset4] > cb)
                                      if(ptr[offset3] > cb)
                                        goto is_a_corner;
                                      else
                                        if(ptr[offset10] > cb)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                    else
                                      if(ptr[offset10] > cb)
                                        if(ptr[offset11] > cb)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset1] < c_b)
                              if(ptr[offset6] > cb)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset3] < c_b)
                                    if(ptr[offset4] < c_b)
                                      if(ptr[offset10] < c_b)
                                        if(ptr[offset11] < c_b)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  if(ptr[offset3] < c_b)
                                    if(ptr[offset4] < c_b)
                                      if(ptr[offset10] < c_b)
                                        if(ptr[offset11] < c_b)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                    else
                      if(ptr[offset9] > cb)
                        if(ptr[offset1] > cb)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset9] < c_b)
                          if(ptr[offset1] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset1] < c_b)
                              if(ptr[offset6] > cb)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset8] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      if(ptr[offset3] < c_b)
                                        goto is_a_corner;
                                      else
                                        if(ptr[offset8] < c_b)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      if(ptr[offset3] < c_b)
                                        goto is_a_corner;
                                      else
                                        if(ptr[offset8] < c_b)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset1] < c_b)
                              if(ptr[offset6] > cb)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset3] < c_b)
                                    if(ptr[offset4] < c_b)
                                      if(ptr[offset10] < c_b)
                                        if(ptr[offset11] < c_b)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  if(ptr[offset3] < c_b)
                                    if(ptr[offset4] < c_b)
                                      if(ptr[offset10] < c_b)
                                        if(ptr[offset11] < c_b)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                else
                  if(ptr[offset7] > cb)
                    if(ptr[offset9] < c_b)
                      goto is_not_a_corner;
                    else
                      if(ptr[offset9] > cb)
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset3] > cb)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] < c_b)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset3] > cb)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset10] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset3] > cb)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset10] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset9] < c_b)
                      if(ptr[offset7] < c_b)
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] > cb)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
            else
              if(ptr[offset2] > cb)
                if(ptr[offset7] < c_b)
                  if(ptr[offset9] > cb)
                    goto is_not_a_corner;
                  else
                    if(ptr[offset9] < c_b)
                      if(ptr[offset1] > cb)
                        if(ptr[offset6] > cb)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            if(ptr[offset8] < c_b)
                              if(ptr[offset10] < c_b)
                                if(ptr[offset11] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] > cb)
                            if(ptr[offset8] < c_b)
                              if(ptr[offset10] < c_b)
                                if(ptr[offset11] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset8] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                        else
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                if(ptr[offset2] < c_b)
                  if(ptr[offset7] > cb)
                    if(ptr[offset9] > cb)
                      if(ptr[offset1] > cb)
                        goto is_not_a_corner;
                      else
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] > cb)
                            if(ptr[offset3] < c_b)
                              if(ptr[offset4] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset9] < c_b)
                        if(ptr[offset1] > cb)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] > cb)
                              if(ptr[offset10] < c_b)
                                if(ptr[offset11] < c_b)
                                  if(ptr[offset3] < c_b)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset8] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset8] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset8] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset1] > cb)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                  else
                    if(ptr[offset9] > cb)
                      if(ptr[offset7] < c_b)
                        if(ptr[offset1] > cb)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset1] > cb)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                    else
                      if(ptr[offset7] < c_b)
                        if(ptr[offset9] < c_b)
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] > cb)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset1] < c_b)
                              if(ptr[offset6] > cb)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset8] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      if(ptr[offset3] < c_b)
                                        goto is_a_corner;
                                      else
                                        if(ptr[offset8] < c_b)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      if(ptr[offset3] < c_b)
                                        goto is_a_corner;
                                      else
                                        if(ptr[offset8] < c_b)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                goto is_not_a_corner;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset1] < c_b)
                              if(ptr[offset6] > cb)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset3] < c_b)
                                    if(ptr[offset4] < c_b)
                                      if(ptr[offset10] < c_b)
                                        if(ptr[offset11] < c_b)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  if(ptr[offset3] < c_b)
                                    if(ptr[offset4] < c_b)
                                      if(ptr[offset10] < c_b)
                                        if(ptr[offset11] < c_b)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                      else
                        if(ptr[offset9] < c_b)
                          if(ptr[offset1] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset1] < c_b)
                              if(ptr[offset6] > cb)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset8] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      if(ptr[offset3] < c_b)
                                        goto is_a_corner;
                                      else
                                        if(ptr[offset8] < c_b)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      if(ptr[offset3] < c_b)
                                        goto is_a_corner;
                                      else
                                        if(ptr[offset8] < c_b)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset1] < c_b)
                              if(ptr[offset6] > cb)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset3] < c_b)
                                    if(ptr[offset4] < c_b)
                                      if(ptr[offset10] < c_b)
                                        if(ptr[offset11] < c_b)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  if(ptr[offset3] < c_b)
                                    if(ptr[offset4] < c_b)
                                      if(ptr[offset10] < c_b)
                                        if(ptr[offset11] < c_b)
                                          goto is_a_corner;
                                        else
                                          goto is_not_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                else
                  if(ptr[offset7] < c_b)
                    if(ptr[offset9] > cb)
                      goto is_not_a_corner;
                    else
                      if(ptr[offset9] < c_b)
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] > cb)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
        else
          if(ptr[offset5] < c_b)
            if(ptr[offset7] > cb)
              goto is_not_a_corner;
            else
              if(ptr[offset7] < c_b)
                if(ptr[offset2] > cb)
                  if(ptr[offset9] > cb)
                    goto is_not_a_corner;
                  else
                    if(ptr[offset9] < c_b)
                      if(ptr[offset1] > cb)
                        if(ptr[offset6] > cb)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            if(ptr[offset8] < c_b)
                              if(ptr[offset4] < c_b)
                                if(ptr[offset3] < c_b)
                                  goto is_a_corner;
                                else
                                  if(ptr[offset10] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset3] < c_b)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset3] < c_b)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  if(ptr[offset2] < c_b)
                    if(ptr[offset9] > cb)
                      if(ptr[offset1] < c_b)
                        if(ptr[offset6] > cb)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            if(ptr[offset3] < c_b)
                              if(ptr[offset4] < c_b)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset8] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset8] < c_b)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                    else
                      if(ptr[offset9] < c_b)
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset3] < c_b)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] > cb)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset3] < c_b)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset8] < c_b)
                                      if(ptr[offset10] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                      else
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] > cb)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset8] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset8] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                  else
                    if(ptr[offset9] > cb)
                      goto is_not_a_corner;
                    else
                      if(ptr[offset9] < c_b)
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] > cb)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset3] < c_b)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] > cb)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
              else
                goto is_not_a_corner;
          else
            if(ptr[offset5] > cb)
              if(ptr[offset7] > cb)
                if(ptr[offset2] < c_b)
                  if(ptr[offset9] < c_b)
                    goto is_not_a_corner;
                  else
                    if(ptr[offset9] > cb)
                      if(ptr[offset1] > cb)
                        if(ptr[offset6] < c_b)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset6] > cb)
                            if(ptr[offset8] > cb)
                              if(ptr[offset4] > cb)
                                if(ptr[offset3] > cb)
                                  goto is_a_corner;
                                else
                                  if(ptr[offset10] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset3] > cb)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset3] > cb)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  if(ptr[offset2] > cb)
                    if(ptr[offset9] < c_b)
                      if(ptr[offset1] > cb)
                        if(ptr[offset6] < c_b)
                          goto is_not_a_corner;
                        else
                          if(ptr[offset6] > cb)
                            if(ptr[offset3] > cb)
                              if(ptr[offset4] > cb)
                                goto is_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                          else
                            goto is_not_a_corner;
                      else
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset8] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset6] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset8] > cb)
                                    goto is_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                    else
                      if(ptr[offset9] > cb)
                        if(ptr[offset1] < c_b)
                          if(ptr[offset6] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset3] > cb)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] < c_b)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset3] > cb)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset8] > cb)
                                      if(ptr[offset10] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset3] > cb)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset10] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                      else
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  goto is_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] < c_b)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset8] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset8] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                  else
                    if(ptr[offset9] < c_b)
                      goto is_not_a_corner;
                    else
                      if(ptr[offset9] > cb)
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] < c_b)
                            goto is_not_a_corner;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset3] > cb)
                                    goto is_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto is_a_corner;
                                    else
                                      goto is_not_a_corner;
                                  else
                                    goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                            else
                              goto is_not_a_corner;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] < c_b)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset3] > cb)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset10] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                          else
                            if(ptr[offset6] < c_b)
                              goto is_not_a_corner;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset3] > cb)
                                      goto is_a_corner;
                                    else
                                      if(ptr[offset10] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                  else
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto is_a_corner;
                                      else
                                        goto is_not_a_corner;
                                    else
                                      goto is_not_a_corner;
                                else
                                  goto is_not_a_corner;
                              else
                                goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
              else
                goto is_not_a_corner;
            else
              goto is_not_a_corner;

        is_a_corner:
            bmin = b_test;
            goto end;

        is_not_a_corner:
            bmax = b_test;
            goto end;

        end:

        if(bmin == bmax - 1 || bmin == bmax)
            return bmin;
        b_test = (bmin + bmax) / 2;
    }
}

// 8 pixel mask
template<>
int agast_cornerScore<AgastFeatureDetector::AGAST_5_8>(const uchar* ptr, const int pixel[], int threshold)
{
    int bmin = threshold;
    int bmax = 255;
    int b_test = (bmax + bmin)/2;

    short offset0 = (short) pixel[0];
    short offset1 = (short) pixel[1];
    short offset2 = (short) pixel[2];
    short offset3 = (short) pixel[3];
    short offset4 = (short) pixel[4];
    short offset5 = (short) pixel[5];
    short offset6 = (short) pixel[6];
    short offset7 = (short) pixel[7];

    while(true)
    {
        const int cb = *ptr + b_test;
        const int c_b = *ptr - b_test;
        if(ptr[offset0] > cb)
          if(ptr[offset2] > cb)
            if(ptr[offset3] > cb)
              if(ptr[offset5] > cb)
                if(ptr[offset1] > cb)
                  if(ptr[offset4] > cb)
                    goto is_a_corner;
                  else
                    if(ptr[offset7] > cb)
                      goto is_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  if(ptr[offset4] > cb)
                    if(ptr[offset6] > cb)
                      goto is_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
              else
                if(ptr[offset1] > cb)
                  if(ptr[offset4] > cb)
                    goto is_a_corner;
                  else
                    if(ptr[offset7] > cb)
                      goto is_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  goto is_not_a_corner;
            else
              if(ptr[offset7] > cb)
                if(ptr[offset6] > cb)
                  if(ptr[offset5] > cb)
                    if(ptr[offset1] > cb)
                      goto is_a_corner;
                    else
                      if(ptr[offset4] > cb)
                        goto is_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset1] > cb)
                      goto is_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                if(ptr[offset5] < c_b)
                  if(ptr[offset3] < c_b)
                    if(ptr[offset7] < c_b)
                      if(ptr[offset4] < c_b)
                        if(ptr[offset6] < c_b)
                          goto is_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
          else
            if(ptr[offset5] > cb)
              if(ptr[offset7] > cb)
                if(ptr[offset6] > cb)
                  if(ptr[offset1] > cb)
                    goto is_a_corner;
                  else
                    if(ptr[offset4] > cb)
                      goto is_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                goto is_not_a_corner;
            else
              if(ptr[offset5] < c_b)
                if(ptr[offset3] < c_b)
                  if(ptr[offset2] < c_b)
                    if(ptr[offset1] < c_b)
                      if(ptr[offset4] < c_b)
                        goto is_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      if(ptr[offset4] < c_b)
                        if(ptr[offset6] < c_b)
                          goto is_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset7] < c_b)
                      if(ptr[offset4] < c_b)
                        if(ptr[offset6] < c_b)
                          goto is_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                goto is_not_a_corner;
        else if(ptr[offset0] < c_b)
          if(ptr[offset2] < c_b)
            if(ptr[offset7] > cb)
              if(ptr[offset3] < c_b)
                if(ptr[offset5] < c_b)
                  if(ptr[offset1] < c_b)
                    if(ptr[offset4] < c_b)
                      goto is_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    if(ptr[offset4] < c_b)
                      if(ptr[offset6] < c_b)
                        goto is_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  if(ptr[offset1] < c_b)
                    if(ptr[offset4] < c_b)
                      goto is_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
              else
                if(ptr[offset5] > cb)
                  if(ptr[offset3] > cb)
                    if(ptr[offset4] > cb)
                      if(ptr[offset6] > cb)
                        goto is_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
            else
              if(ptr[offset7] < c_b)
                if(ptr[offset3] < c_b)
                  if(ptr[offset5] < c_b)
                    if(ptr[offset1] < c_b)
                      goto is_a_corner;
                    else
                      if(ptr[offset4] < c_b)
                        if(ptr[offset6] < c_b)
                          goto is_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset1] < c_b)
                      goto is_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  if(ptr[offset6] < c_b)
                    if(ptr[offset5] < c_b)
                      if(ptr[offset1] < c_b)
                        goto is_a_corner;
                      else
                        if(ptr[offset4] < c_b)
                          goto is_a_corner;
                        else
                          goto is_not_a_corner;
                    else
                      if(ptr[offset1] < c_b)
                        goto is_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
              else
                if(ptr[offset3] < c_b)
                  if(ptr[offset5] < c_b)
                    if(ptr[offset1] < c_b)
                      if(ptr[offset4] < c_b)
                        goto is_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      if(ptr[offset4] < c_b)
                        if(ptr[offset6] < c_b)
                          goto is_a_corner;
                        else
                          goto is_not_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    if(ptr[offset1] < c_b)
                      if(ptr[offset4] < c_b)
                        goto is_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  goto is_not_a_corner;
          else
            if(ptr[offset5] > cb)
              if(ptr[offset3] > cb)
                if(ptr[offset2] > cb)
                  if(ptr[offset1] > cb)
                    if(ptr[offset4] > cb)
                      goto is_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    if(ptr[offset4] > cb)
                      if(ptr[offset6] > cb)
                        goto is_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  if(ptr[offset7] > cb)
                    if(ptr[offset4] > cb)
                      if(ptr[offset6] > cb)
                        goto is_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
              else
                goto is_not_a_corner;
            else
              if(ptr[offset5] < c_b)
                if(ptr[offset7] < c_b)
                  if(ptr[offset6] < c_b)
                    if(ptr[offset1] < c_b)
                      goto is_a_corner;
                    else
                      if(ptr[offset4] < c_b)
                        goto is_a_corner;
                      else
                        goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
              else
                goto is_not_a_corner;
        else
          if(ptr[offset3] > cb)
            if(ptr[offset5] > cb)
              if(ptr[offset2] > cb)
                if(ptr[offset1] > cb)
                  if(ptr[offset4] > cb)
                    goto is_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  if(ptr[offset4] > cb)
                    if(ptr[offset6] > cb)
                      goto is_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
              else
                if(ptr[offset7] > cb)
                  if(ptr[offset4] > cb)
                    if(ptr[offset6] > cb)
                      goto is_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
                else
                  goto is_not_a_corner;
            else
              goto is_not_a_corner;
          else
            if(ptr[offset3] < c_b)
              if(ptr[offset5] < c_b)
                if(ptr[offset2] < c_b)
                  if(ptr[offset1] < c_b)
                    if(ptr[offset4] < c_b)
                      goto is_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    if(ptr[offset4] < c_b)
                      if(ptr[offset6] < c_b)
                        goto is_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                else
                  if(ptr[offset7] < c_b)
                    if(ptr[offset4] < c_b)
                      if(ptr[offset6] < c_b)
                        goto is_a_corner;
                      else
                        goto is_not_a_corner;
                    else
                      goto is_not_a_corner;
                  else
                    goto is_not_a_corner;
              else
                goto is_not_a_corner;
            else
              goto is_not_a_corner;

        is_a_corner:
            bmin=b_test;
            goto end;

        is_not_a_corner:
            bmax=b_test;
            goto end;

        end:

        if(bmin == bmax - 1 || bmin == bmax)
            return bmin;
        b_test = (bmin + bmax) / 2;
    }
}
#else // !(defined __i386__ || defined(_M_IX86) || defined __x86_64__ || defined(_M_X64))


int agast_tree_search(const uint32_t table_struct32[], int pixel_[], const unsigned char* const ptr, int threshold)
{
    const int cb = *ptr + threshold;
    const int c_b = *ptr - threshold;
    int index;
    int offset;
    int cmpresult;
    index = 0;
    while ((table_struct32[index]>>16)!=0)
    {
        offset=(int) pixel_[table_struct32[index]>>28];
        if ((table_struct32[index]&(1<<12))!=0)
            cmpresult=(ptr[offset] < c_b);
        else
            cmpresult=(ptr[offset] > cb);
        if (cmpresult)
            index =(table_struct32[index]>>16)&0xfff;
        else
            index =table_struct32[index]&0xfff;
    }
    return (int)(table_struct32[index]&0xff);
}

// universal pixel mask
int AGAST_ALL_SCORE(const uchar* ptr, const int pixel[], int threshold, int agasttype)
{
    int bmin = threshold;
    int bmax = 255;
    int b_test = (bmax + bmin)/2;
    uint32_t *table_struct;

    int result;
    static const uint32_t table_5_8_corner_struct[] =
    {       0x00010026,0x20020017,0x3003000c,0x50040009,0x10050007,0x406d0006,0x706d006c,0x4008006c,
        0x606d006c,0x100a006c,0x406d000b,0x706d006c,0x700d0012,0x600e006c,0x500f0011,0x106d0010,
        0x406d006c,0x106d006c,0x5013106c,0x3014106c,0x7015106c,0x4016106c,0x606d106c,0x5018001c,
        0x7019006c,0x601a006c,0x106d001b,0x406d006c,0x501d106c,0x301e106c,0x201f1023,0x10201021,
        0x406d106c,0x4022106c,0x606d106c,0x7024106c,0x4025106c,0x606d106c,0x00271058,0x20281049,
        0x70290035,0x302a1031,0x502b102f,0x102c102d,0x406d106c,0x402e106c,0x606d106c,0x1030106c,
        0x406d106c,0x5032006c,0x3033006c,0x4034006c,0x606d006c,0x70361041,0x3037103c,0x5038103b,
        0x106d1039,0x403a106c,0x606d106c,0x106d106c,0x603d106c,0x503e1040,0x106d103f,0x406d106c,
        0x106d106c,0x3042106c,0x50431047,0x10441045,0x406d106c,0x4046106c,0x606d106c,0x1048106c,
        0x406d106c,0x504a0053,0x304b006c,0x204c0050,0x104d004e,0x406d006c,0x404f006c,0x606d006c,
        0x7051006c,0x4052006c,0x606d006c,0x5054106c,0x7055106c,0x6056106c,0x106d1057,0x406d106c,
        0x30590062,0x505a006c,0x205b005f,0x105c005d,0x406d006c,0x405e006c,0x606d006c,0x7060006c,
        0x4061006c,0x606d006c,0x3063106c,0x5064106c,0x20651069,0x10661067,0x406d106c,0x4068106c,
        0x606d106c,0x706a106c,0x406b106c,0x606d106c,0x000000fe,0x000000ff};

    static const uint32_t table_7_12d_corner_struct[] =
    {       0x000100b5,0x50020036,0x20030025,0x9004001d,0x10050015,0x6006000f,0x3007000a,0x41870008,
        0xa0090186,0xb1870186,0x800b0186,0xa00c0186,0xb187000d,0x400e0186,0x71870186,0xb0100186,
        0x30110013,0x41870012,0xa1870186,0x80140186,0xa1870186,0x60160186,0x70170186,0x80180186,
        0x4019001b,0x3187001a,0xa1870186,0xa01c0186,0xb1870186,0x301e0186,0x401f0186,0x10200022,
        0x61870021,0xb1870186,0x60230186,0x70240186,0x81870186,0x90260186,0x70270186,0x80280186,
        0x10290030,0xa02a002d,0xb187002b,0x602c0186,0x41870186,0x602e0186,0x302f0186,0x41870186,
        0x60310186,0x40320034,0x31870033,0xa1870186,0xa0350186,0xb1870186,0x503710a1,0x9038006b,
        0x3039105b,0x403a1053,0xb03b004d,0x103c0044,0x803d0040,0xa03e0186,0x2187003f,0x71870186,
        0x60411186,0x20421186,0x70431186,0x81871186,0x60450048,0x70460186,0x80470186,0xa1870186,
        0x60491186,0x204a1186,0x704b1186,0x1187104c,0x81871186,0x204e1186,0x704f1186,0x10501051,
        0x61871186,0x60521186,0x81871186,0xb0540186,0x80550186,0xa0560186,0x10570059,0x21870058,
        0x71870186,0x605a0186,0x71870186,0xb05c0186,0xa05d0186,0x305e0065,0x105f0062,0x21870060,
        0x70610186,0x81870186,0x60630186,0x70640186,0x81870186,0x80660186,0x10670069,0x21870068,
        0x71870186,0x606a0186,0x71870186,0x906c1093,0x206d0087,0x106e007f,0x406f0077,0xa0700072,
        0x30710186,0xb1870186,0x60731186,0x70741186,0x80751186,0xb0761186,0xa1871186,0x60781186,
        0x70791186,0x807a1186,0xa07b107d,0x4187107c,0xb1871186,0x307e1186,0x41871186,0x60801186,
        0x70811186,0x80821186,0x40831085,0x31871084,0xa1871186,0xa0861186,0xb1871186,0x60881186,
        0x70891186,0x808a108f,0x408b108d,0x3187108c,0xa1871186,0xa08e1186,0xb1871186,0x20901186,
        0x10911186,0x30921186,0x41871186,0x20940099,0x10950186,0x30960186,0x40970186,0xa0980186,
        0xb1870186,0x209a1186,0x309b1186,0x409c1186,0x709d1186,0x109e109f,0x61871186,0x60a01186,
        0x81871186,0x20a200ae,0xa0a30186,0xb0a40186,0x90a500ab,0x10a600a8,0x318700a7,0x81870186,
        0x60a90186,0x70aa0186,0x81870186,0x10ac0186,0x30ad0186,0x41870186,0x90af0186,0x70b00186,
        0x80b10186,0xa0b20186,0xb0b30186,0x118700b4,0x61870186,0x00b6115a,0x20b700e2,0x50b800cc,
        0x70b900c5,0x60ba0186,0x40bb00c1,0x30bc00be,0x118700bd,0x81870186,0x90bf0186,0x80c00186,
        0xa1870186,0x90c20186,0x80c30186,0xa0c40186,0xb1870186,0x90c61186,0x80c71186,0xa0c81186,
        0xb0c91186,0x70ca1186,0x118710cb,0x61871186,0x90cd1186,0x70ce1186,0x80cf1186,0x50d010de,
        0x10d110d8,0xa0d210d5,0xb18710d3,0x60d41186,0x41871186,0x60d61186,0x30d71186,0x41871186,
        0x60d91186,0x40da10dc,0x318710db,0xa1871186,0xa0dd1186,0xb1871186,0xa0df1186,0xb0e01186,
        0x118710e1,0x61871186,0x20e3113a,0x90e4010b,0x50e500ff,0x10e610f7,0x40e710ef,0xa0e810ea,
        0x30e91186,0xb1871186,0x60eb0186,0x70ec0186,0x80ed0186,0xb0ee0186,0xa1870186,0x60f00186,
        0x70f10186,0x80f20186,0xa0f300f5,0x418700f4,0xb1870186,0x30f60186,0x41870186,0x60f80186,
        0x70f90186,0x80fa0186,0x40fb00fd,0x318700fc,0xa1870186,0xa0fe0186,0xb1870186,0x31001186,
        0x41011186,0x51021108,0x11031105,0x61871104,0xb1871186,0x61061186,0x71071186,0x81871186,
        0x11091186,0xa10a1186,0xb1871186,0x910c112e,0x510d1126,0x110e111e,0x610f1118,0x31101113,
        0x41871111,0xa1121186,0xb1871186,0x81141186,0xa1151186,0xb1871116,0x41171186,0x71871186,
        0xb1191186,0x311a111c,0x4187111b,0xa1871186,0x811d1186,0xa1871186,0x611f1186,0x71201186,
        0x81211186,0x41221124,0x31871123,0xa1871186,0xa1251186,0xb1871186,0xa1271186,0xb1281186,
        0x1129112b,0x3187112a,0x81871186,0x612c1186,0x712d1186,0x81871186,0x312f1186,0x41301186,
        0x51311137,0x11321134,0x61871133,0xb1871186,0x61351186,0x71361186,0x81871186,0x11381186,
        0xa1391186,0xb1871186,0x913b1150,0x713c1186,0x813d1186,0x513e114c,0x113f1146,0xa1401143,
        0xb1871141,0x61421186,0x41871186,0x61441186,0x31451186,0x41871186,0x61471186,0x4148114a,
        0x31871149,0xa1871186,0xa14b1186,0xb1871186,0xa14d1186,0xb14e1186,0x1187114f,0x61871186,
        0x51510186,0x91520186,0x61530186,0x71540186,0x81550186,0x41560158,0x31870157,0xa1870186,
        0xa1590186,0xb1870186,0x515b0170,0x915c0168,0x615d0186,0x715e0186,0x415f0165,0x31600163,
        0x81870161,0x11620186,0x21870186,0x81640186,0xa1870186,0xb1660186,0x81670186,0xa1870186,
        0x21690186,0x316a0186,0x416b0186,0x716c0186,0x116d016e,0x61870186,0x616f0186,0x81870186,
        0x51711186,0x9172117e,0x61731186,0x71741186,0x4175117b,0x31761179,0x81871177,0x11781186,
        0x21871186,0x817a1186,0xa1871186,0xb17c1186,0x817d1186,0xa1871186,0x217f1186,0x31801186,
        0x41811186,0x71821186,0x11831184,0x61871186,0x61851186,0x81871186,0x000000fe,0x000000ff};

    static const uint32_t table_7_12s_corner_struct[] =
    {       0x0001032b,0x50020104,0x20031026,0x70040748,0x97481005,0x90060748,0x1007100f,0x67481008,
        0x60090748,0x800a0748,0x400b000d,0x3749000c,0xa7490748,0xa00e0748,0xb7490748,0x1010001e,
        0x60111014,0x80120748,0xa0130748,0xb7490748,0x6015001b,0x80160748,0x40170019,0x37490018,
        0xa7490748,0xa01a0748,0xb7490748,0x801c0748,0xa01d0748,0xb7490748,0x6748101f,0x60200748,
        0x80210748,0x40220024,0x37490023,0xa7490748,0xa0250748,0xb7490748,0x202700e1,0x70281059,
        0x90291035,0x1748102a,0x102b0748,0x602c002e,0x302d0748,0x47490748,0x602f1032,0x30300748,
        0x40310748,0xb7490748,0x30330748,0x40340748,0xb7490748,0x9036004d,0x17481037,0x10380748,
        0x6039103f,0xb03a0748,0x303b003d,0x4749003c,0xa7490748,0x803e0748,0xa7490748,0x60400047,
        0x30410044,0x47490042,0xa0430748,0xb7490748,0x80450748,0xa0460748,0xb7490748,0xb0480748,
        0x3049004b,0x4749004a,0xa7490748,0x804c0748,0xa7490748,0x1748104e,0x104f0748,0x60500052,
        0x30510748,0x47490748,0x60531056,0x30540748,0x40550748,0xb7490748,0x30570748,0x40580748,
        0xb7490748,0x905a107d,0x705b0071,0x105c1061,0x6748105d,0x605e0748,0x305f0748,0x40600748,
        0x87490748,0x1062006c,0x60630065,0x30640748,0x47490748,0x60661069,0x30670748,0x40680748,
        0xb7490748,0x306a0748,0x406b0748,0xb7490748,0x6748106d,0x606e0748,0x306f0748,0x40700748,
        0x87490748,0x17481072,0x10730748,0x60740076,0x30750748,0x47490748,0x6077107a,0x30780748,
        0x40790748,0xb7490748,0x307b0748,0x407c0748,0xb7490748,0x707e00bd,0x907f00a7,0x10801088,
        0x67481081,0x60820748,0x80830748,0x40840086,0x37490085,0xa7490748,0xa0870748,0xb7490748,
        0x1089009f,0x608a1090,0xb08b0748,0x308c008e,0x4749008d,0xa7490748,0x808f0748,0xa7490748,
        0x60910099,0x30920095,0x47490093,0xa0940748,0xb7490748,0x80960748,0xa0970748,0x47490098,
        0xb7490748,0xb09a0748,0x309b009d,0x4749009c,0xa7490748,0x809e0748,0xa7490748,0x674810a0,
        0x60a10748,0x80a20748,0x40a300a5,0x374900a4,0xa7490748,0xa0a60748,0xb7490748,0x10a810ad,
        0x674810a9,0x60aa0748,0x30ab0748,0x40ac0748,0x87490748,0x10ae00b8,0x60af00b1,0x30b00748,
        0x47490748,0x60b210b5,0x30b30748,0x40b40748,0xb7490748,0x30b60748,0x40b70748,0xb7490748,
        0x674810b9,0x60ba0748,0x30bb0748,0x40bc0748,0x87490748,0x90be00d5,0x174810bf,0x10c00748,
        0x60c110c7,0xb0c20748,0x30c300c5,0x474900c4,0xa7490748,0x80c60748,0xa7490748,0x60c800cf,
        0x30c900cc,0x474900ca,0xa0cb0748,0xb7490748,0x80cd0748,0xa0ce0748,0xb7490748,0xb0d00748,
        0x30d100d3,0x474900d2,0xa7490748,0x80d40748,0xa7490748,0x174810d6,0x10d70748,0x60d800da,
        0x30d90748,0x47490748,0x60db10de,0x30dc0748,0x40dd0748,0xb7490748,0x30df0748,0x40e00748,
        0xb7490748,0x70e20748,0x974810e3,0x90e40748,0x10e510ed,0x674810e6,0x60e70748,0x80e80748,
        0x40e900eb,0x374900ea,0xa7490748,0xa0ec0748,0xb7490748,0x10ee00fc,0x60ef10f2,0x80f00748,
        0xa0f10748,0xb7490748,0x60f300f9,0x80f40748,0x40f500f7,0x374900f6,0xa7490748,0xa0f80748,
        0xb7490748,0x80fa0748,0xa0fb0748,0xb7490748,0x674810fd,0x60fe0748,0x80ff0748,0x41000102,
        0x37490101,0xa7490748,0xa1030748,0xb7490748,0x51051253,0x9106118c,0x71070119,0x27481108,
        0x21090748,0x1748110a,0x110b0748,0x610c0110,0x310d0748,0x410e0748,0xa10f0748,0xb7490748,
        0x61111115,0x31120748,0x41130748,0xa1140748,0xb7490748,0x31160748,0x41170748,0xa1180748,
        0xb7490748,0x711a117a,0x211b1136,0x111c0124,0x6748011d,0x611e1748,0x811f1748,0x41201122,
        0x37491121,0xa7491748,0xa1231748,0xb7491748,0x1125112e,0x67480126,0x61271748,0x4128112b,
        0x37491129,0x812a1748,0xa7491748,0x812c1748,0xa12d1748,0xb7491748,0x6748012f,0x61301748,
        0x81311748,0x41321134,0x37491133,0xa7491748,0xa1351748,0xb7491748,0x21370160,0x11381140,
        0x67480139,0x613a1748,0x813b1748,0x413c113e,0x3749113d,0xa7491748,0xa13f1748,0xb7491748,
        0x11410158,0x61420146,0x31430748,0x41440748,0xa1450748,0xb7490748,0x61471154,0x4148014e,
        0xa149014b,0x314a0748,0xb7490748,0x814c1748,0xb14d1748,0xa7491748,0x814f1748,0xa1501152,
        0x47491151,0xb7491748,0x31531748,0x47491748,0x31550748,0x41560748,0xa1570748,0xb7490748,
        0x67480159,0x615a1748,0x815b1748,0x415c115e,0x3749115d,0xa7491748,0xa15f1748,0xb7491748,
        0x11610169,0x67480162,0x61631748,0x81641748,0x41651167,0x37491166,0xa7491748,0xa1681748,
        0xb7491748,0x116a1172,0x6748016b,0x616c1748,0x816d1748,0x416e1170,0x3749116f,0xa7491748,
        0xa1711748,0xb7491748,0x67480173,0x61741748,0x81751748,0x41761178,0x37491177,0xa7491748,
        0xa1791748,0xb7491748,0x2748117b,0x217c0748,0x1748117d,0x117e0748,0x617f0183,0x31800748,
        0x41810748,0xa1820748,0xb7490748,0x61841188,0x31850748,0x41860748,0xa1870748,0xb7490748,
        0x31890748,0x418a0748,0xa18b0748,0xb7490748,0x918d020d,0x718e11b0,0x218f019f,0x17481190,
        0x11910748,0x61920196,0xa1930748,0xb1940748,0x37490195,0x87490748,0x6197119b,0xa1980748,
        0xb1990748,0x3749019a,0x87490748,0xa19c0748,0xb19d0748,0x3749019e,0x87490748,0x21a01748,
        0x11a111a5,0x674801a2,0x61a31748,0x31a41748,0x47491748,0x11a601ab,0x674801a7,0x61a81748,
        0x31a91748,0x41aa1748,0x87491748,0x674801ac,0x61ad1748,0x31ae1748,0x41af1748,0x87491748,
        0x71b101fb,0x21b211c9,0x11b311b8,0x674811b4,0x61b50748,0x81b60748,0xa1b70748,0xb7490748,
        0x11b901c4,0x61ba01bd,0x81bb0748,0xa1bc0748,0xb7490748,0x61be11c1,0x81bf0748,0xa1c00748,
        0xb7490748,0x81c20748,0xa1c30748,0xb7490748,0x674811c5,0x61c60748,0x81c70748,0xa1c80748,
        0xb7490748,0x21ca01e4,0x11cb11d0,0x674811cc,0x61cd0748,0x81ce0748,0xa1cf0748,0xb7490748,
        0x11d101df,0x61d201d6,0xa1d30748,0xb1d40748,0x374901d5,0x87490748,0x61d711db,0xa1d80748,
        0xb1d90748,0x374901da,0x87490748,0xa1dc0748,0xb1dd0748,0x374901de,0x87490748,0x674811e0,
        0x61e10748,0x81e20748,0xa1e30748,0xb7490748,0x11e511ea,0x674811e6,0x61e70748,0x81e80748,
        0xa1e90748,0xb7490748,0x11eb01f6,0x61ec01ef,0x81ed0748,0xa1ee0748,0xb7490748,0x61f011f3,
        0x81f10748,0xa1f20748,0xb7490748,0x81f40748,0xa1f50748,0xb7490748,0x674811f7,0x61f80748,
        0x81f90748,0xa1fa0748,0xb7490748,0x274811fc,0x21fd0748,0x174811fe,0x11ff0748,0x62000204,
        0xa2010748,0xb2020748,0x37490203,0x87490748,0x62051209,0xa2060748,0xb2070748,0x37490208,
        0x87490748,0xa20a0748,0xb20b0748,0x3749020c,0x87490748,0x220e1220,0x7748020f,0x72101748,
        0x12111215,0x67480212,0x62131748,0x32141748,0x47491748,0x1216021b,0x67480217,0x62181748,
        0x32191748,0x421a1748,0x87491748,0x6748021c,0x621d1748,0x321e1748,0x421f1748,0x87491748,
        0x22210748,0x72220232,0x17481223,0x12240748,0x62250229,0x32260748,0x42270748,0xa2280748,
        0xb7490748,0x622a122e,0x322b0748,0x422c0748,0xa22d0748,0xb7490748,0x322f0748,0x42300748,
        0xa2310748,0xb7490748,0x72331243,0x17481234,0x12350748,0x6236023a,0x32370748,0x42380748,
        0xa2390748,0xb7490748,0x623b123f,0x323c0748,0x423d0748,0xa23e0748,0xb7490748,0x32400748,
        0x42410748,0xa2420748,0xb7490748,0x17481244,0x12450748,0x6246024a,0x32470748,0x42480748,
        0xa2490748,0xb7490748,0x624b124f,0x324c0748,0x424d0748,0xa24e0748,0xb7490748,0x32500748,
        0x42510748,0xa2520748,0xb7490748,0x2254126e,0x72550748,0x97481256,0x92570748,0x1258125d,
        0x67481259,0x625a0748,0x825b0748,0xa25c0748,0xb7490748,0x125e0269,0x625f0262,0x82600748,
        0xa2610748,0xb7490748,0x62631266,0x82640748,0xa2650748,0xb7490748,0x82670748,0xa2680748,
        0xb7490748,0x6748126a,0x626b0748,0x826c0748,0xa26d0748,0xb7490748,0x226f0311,0x727012a2,
        0x92711281,0x17481272,0x12730748,0x62740278,0x32750748,0x42760748,0xa2770748,0xb7490748,
        0x6279127d,0x327a0748,0x427b0748,0xa27c0748,0xb7490748,0x327e0748,0x427f0748,0xa2800748,
        0xb7490748,0x92820292,0x17481283,0x12840748,0x62850289,0xa2860748,0xb2870748,0x37490288,
        0x87490748,0x628a128e,0xa28b0748,0xb28c0748,0x3749028d,0x87490748,0xa28f0748,0xb2900748,
        0x37490291,0x87490748,0x17481293,0x12940748,0x62950299,0x32960748,0x42970748,0xa2980748,
        0xb7490748,0x629a129e,0x329b0748,0x429c0748,0xa29d0748,0xb7490748,0x329f0748,0x42a00748,
        0xa2a10748,0xb7490748,0x92a312c4,0x72a402b4,0x174812a5,0x12a60748,0x62a702ab,0x32a80748,
        0x42a90748,0xa2aa0748,0xb7490748,0x62ac12b0,0x32ad0748,0x42ae0748,0xa2af0748,0xb7490748,
        0x32b10748,0x42b20748,0xa2b30748,0xb7490748,0x174812b5,0x12b60748,0x62b702bb,0x32b80748,
        0x42b90748,0xa2ba0748,0xb7490748,0x62bc12c0,0x32bd0748,0x42be0748,0xa2bf0748,0xb7490748,
        0x32c10748,0x42c20748,0xa2c30748,0xb7490748,0x72c502f0,0x92c602e0,0x12c712cc,0x674812c8,
        0x62c90748,0x82ca0748,0xa2cb0748,0xb7490748,0x12cd02db,0x62ce02d2,0xa2cf0748,0xb2d00748,
        0x374902d1,0x87490748,0x62d312d7,0xa2d40748,0xb2d50748,0x374902d6,0x87490748,0xa2d80748,
        0xb2d90748,0x374902da,0x87490748,0x674812dc,0x62dd0748,0x82de0748,0xa2df0748,0xb7490748,
        0x174812e1,0x12e20748,0x62e302e7,0x32e40748,0x42e50748,0xa2e60748,0xb7490748,0x62e812ec,
        0x32e90748,0x42ea0748,0xa2eb0748,0xb7490748,0x32ed0748,0x42ee0748,0xa2ef0748,0xb7490748,
        0x92f10301,0x174812f2,0x12f30748,0x62f402f8,0xa2f50748,0xb2f60748,0x374902f7,0x87490748,
        0x62f912fd,0xa2fa0748,0xb2fb0748,0x374902fc,0x87490748,0xa2fe0748,0xb2ff0748,0x37490300,
        0x87490748,0x17481302,0x13030748,0x63040308,0x33050748,0x43060748,0xa3070748,0xb7490748,
        0x6309130d,0x330a0748,0x430b0748,0xa30c0748,0xb7490748,0x330e0748,0x430f0748,0xa3100748,
        0xb7490748,0x73120748,0x97481313,0x93140748,0x1315131a,0x67481316,0x63170748,0x83180748,
        0xa3190748,0xb7490748,0x131b0326,0x631c031f,0x831d0748,0xa31e0748,0xb7490748,0x63201323,
        0x83210748,0xa3220748,0xb7490748,0x83240748,0xa3250748,0xb7490748,0x67481327,0x63280748,
        0x83290748,0xa32a0748,0xb7490748,0x032c1655,0x532d1431,0x932e0360,0x2748032f,0x23301748,
        0x7331033d,0x17480332,0x13331748,0x63341336,0x33351748,0x47491748,0x6337033a,0x33381748,
        0x43391748,0xb7491748,0x333b1748,0x433c1748,0xb7491748,0x733e1354,0x133f0344,0x67480340,
        0x63411748,0x33421748,0x43431748,0x87491748,0x1345134f,0x63461348,0x33471748,0x47491748,
        0x6349034c,0x334a1748,0x434b1748,0xb7491748,0x334d1748,0x434e1748,0xb7491748,0x67480350,
        0x63511748,0x33521748,0x43531748,0x87491748,0x17480355,0x13561748,0x63571359,0x33581748,
        0x47491748,0x635a035d,0x335b1748,0x435c1748,0xb7491748,0x335e1748,0x435f1748,0xb7491748,
        0x936113ff,0x7362037b,0x27480363,0x23641748,0x17480365,0x13661748,0x6367036d,0xb3681748,
        0x3369136b,0x4749136a,0xa7491748,0x836c1748,0xa7491748,0x636e1375,0x336f1372,0x47491370,
        0xa3711748,0xb7491748,0x83731748,0xa3741748,0xb7491748,0xb3761748,0x33771379,0x47491378,
        0xa7491748,0x837a1748,0xa7491748,0x737c13e6,0x237d039d,0x137e0386,0x6748037f,0x63801748,
        0x83811748,0x43821384,0x37491383,0xa7491748,0xa3851748,0xb7491748,0x13871395,0x6388038b,
        0x83891748,0xa38a1748,0xb7491748,0x638c1392,0x838d1748,0x438e1390,0x3749138f,0xa7491748,
        0xa3911748,0xb7491748,0x83931748,0xa3941748,0xb7491748,0x67480396,0x63971748,0x83981748,
        0x4399139b,0x3749139a,0xa7491748,0xa39c1748,0xb7491748,0x239e13c6,0x139f03a7,0x674803a0,
        0x63a11748,0x83a21748,0x43a313a5,0x374913a4,0xa7491748,0xa3a61748,0xb7491748,0x13a813be,
        0x63a903af,0xb3aa1748,0x33ab13ad,0x474913ac,0xa7491748,0x83ae1748,0xa7491748,0x63b013b8,
        0x33b113b4,0x474913b2,0xa3b31748,0xb7491748,0x83b51748,0xa3b61748,0x474913b7,0xb7491748,
        0xb3b91748,0x33ba13bc,0x474913bb,0xa7491748,0x83bd1748,0xa7491748,0x674803bf,0x63c01748,
        0x83c11748,0x43c213c4,0x374913c3,0xa7491748,0xa3c51748,0xb7491748,0x13c703cf,0x674803c8,
        0x63c91748,0x83ca1748,0x43cb13cd,0x374913cc,0xa7491748,0xa3ce1748,0xb7491748,0x13d013de,
        0x63d103d4,0x83d21748,0xa3d31748,0xb7491748,0x63d513db,0x83d61748,0x43d713d9,0x374913d8,
        0xa7491748,0xa3da1748,0xb7491748,0x83dc1748,0xa3dd1748,0xb7491748,0x674803df,0x63e01748,
        0x83e11748,0x43e213e4,0x374913e3,0xa7491748,0xa3e51748,0xb7491748,0x274803e7,0x23e81748,
        0x174803e9,0x13ea1748,0x63eb03f1,0xb3ec1748,0x33ed13ef,0x474913ee,0xa7491748,0x83f01748,
        0xa7491748,0x63f213f9,0x33f313f6,0x474913f4,0xa3f51748,0xb7491748,0x83f71748,0xa3f81748,
        0xb7491748,0xb3fa1748,0x33fb13fd,0x474913fc,0xa7491748,0x83fe1748,0xa7491748,0x27480400,
        0x24011748,0x7402040e,0x17480403,0x14041748,0x64051407,0x34061748,0x47491748,0x6408040b,
        0x34091748,0x440a1748,0xb7491748,0x340c1748,0x440d1748,0xb7491748,0x740f1425,0x14100415,
        0x67480411,0x64121748,0x34131748,0x44141748,0x87491748,0x14161420,0x64171419,0x34181748,
        0x47491748,0x641a041d,0x341b1748,0x441c1748,0xb7491748,0x341e1748,0x441f1748,0xb7491748,
        0x67480421,0x64221748,0x34231748,0x44241748,0x87491748,0x17480426,0x14271748,0x6428142a,
        0x34291748,0x47491748,0x642b042e,0x342c1748,0x442d1748,0xb7491748,0x342f1748,0x44301748,
        0xb7491748,0x5432057d,0x2433048b,0x7434144d,0x97480435,0x94361748,0x1437043c,0x67480438,
        0x64391748,0x843a1748,0xa43b1748,0xb7491748,0x143d1448,0x643e0441,0x843f1748,0xa4401748,
        0xb7491748,0x64421445,0x84431748,0xa4441748,0xb7491748,0x84461748,0xa4471748,0xb7491748,
        0x67480449,0x644a1748,0x844b1748,0xa44c1748,0xb7491748,0x744e0748,0x944f145f,0x14500454,
        0x67481451,0x64520748,0x34530748,0x47490748,0x1455145a,0x67481456,0x64570748,0x34580748,
        0x44590748,0x87490748,0x6748145b,0x645c0748,0x345d0748,0x445e0748,0x87490748,0x9460047b,
        0x14611469,0x67481462,0x64630748,0x84640748,0x44650467,0x37490466,0xa7490748,0xa4680748,
        0xb7490748,0x146a0473,0x6748146b,0x646c0748,0x446d0470,0x3749046e,0x846f0748,0xa7490748,
        0x84710748,0xa4720748,0xb7490748,0x67481474,0x64750748,0x84760748,0x44770479,0x37490478,
        0xa7490748,0xa47a0748,0xb7490748,0x147c0480,0x6748147d,0x647e0748,0x347f0748,0x47490748,
        0x14811486,0x67481482,0x64830748,0x34840748,0x44850748,0x87490748,0x67481487,0x64880748,
        0x34890748,0x448a0748,0x87490748,0x248c1547,0x748d14c9,0x948e049e,0x1748048f,0x14901748,
        0x64910495,0x34921748,0x44931748,0xa4941748,0xb7491748,0x6496149a,0x34971748,0x44981748,
        0xa4991748,0xb7491748,0x349b1748,0x449c1748,0xa49d1748,0xb7491748,0x949f14b9,0x14a004a5,
        0x674804a1,0x64a21748,0x84a31748,0xa4a41748,0xb7491748,0x14a614b4,0x64a704ab,0xa4a81748,
        0xb4a91748,0x374914aa,0x87491748,0x64ac14b0,0xa4ad1748,0xb4ae1748,0x374914af,0x87491748,
        0xa4b11748,0xb4b21748,0x374914b3,0x87491748,0x674804b5,0x64b61748,0x84b71748,0xa4b81748,
        0xb7491748,0x174804ba,0x14bb1748,0x64bc04c0,0x34bd1748,0x44be1748,0xa4bf1748,0xb7491748,
        0x64c114c5,0x34c21748,0x44c31748,0xa4c41748,0xb7491748,0x34c61748,0x44c71748,0xa4c81748,
        0xb7491748,0x74ca0515,0x94cb14db,0x174804cc,0x14cd1748,0x64ce04d2,0xa4cf1748,0xb4d01748,
        0x374914d1,0x87491748,0x64d314d7,0xa4d41748,0xb4d51748,0x374914d6,0x87491748,0xa4d81748,
        0xb4d91748,0x374914da,0x87491748,0x94dc0505,0x14dd04e5,0x674814de,0x64df0748,0x84e00748,
        0x44e104e3,0x374904e2,0xa7490748,0xa4e40748,0xb7490748,0x14e614fd,0x64e714eb,0x34e81748,
        0x44e91748,0xa4ea1748,0xb7491748,0x64ec04f9,0x44ed14f3,0xa4ee04f0,0x84ef0748,0xb7490748,
        0x34f11748,0xb4f21748,0xa7491748,0x84f40748,0xa4f504f7,0x474904f6,0xb7490748,0x34f80748,
        0x47490748,0x34fa1748,0x44fb1748,0xa4fc1748,0xb7491748,0x674814fe,0x64ff0748,0x85000748,
        0x45010503,0x37490502,0xa7490748,0xa5040748,0xb7490748,0x17480506,0x15071748,0x6508050c,
        0x35091748,0x450a1748,0xa50b1748,0xb7491748,0x650d1511,0x350e1748,0x450f1748,0xa5101748,
        0xb7491748,0x35121748,0x45131748,0xa5141748,0xb7491748,0x95160526,0x17480517,0x15181748,
        0x6519051d,0x351a1748,0x451b1748,0xa51c1748,0xb7491748,0x651e1522,0x351f1748,0x45201748,
        0xa5211748,0xb7491748,0x35231748,0x45241748,0xa5251748,0xb7491748,0x95271537,0x17480528,
        0x15291748,0x652a052e,0xa52b1748,0xb52c1748,0x3749152d,0x87491748,0x652f1533,0xa5301748,
        0xb5311748,0x37491532,0x87491748,0xa5341748,0xb5351748,0x37491536,0x87491748,0x17480538,
        0x15391748,0x653a053e,0x353b1748,0x453c1748,0xa53d1748,0xb7491748,0x653f1543,0x35401748,
        0x45411748,0xa5421748,0xb7491748,0x35441748,0x45451748,0xa5461748,0xb7491748,0x75480564,
        0x97481549,0x954a0748,0x154b0553,0x6748154c,0x654d0748,0x854e0748,0x454f0551,0x37490550,
        0xa7490748,0xa5520748,0xb7490748,0x1554155c,0x67481555,0x65560748,0x85570748,0x4558055a,
        0x37490559,0xa7490748,0xa55b0748,0xb7490748,0x6748155d,0x655e0748,0x855f0748,0x45600562,
        0x37490561,0xa7490748,0xa5630748,0xb7490748,0x95651748,0x75661748,0x1567056c,0x67480568,
        0x65691748,0x856a1748,0xa56b1748,0xb7491748,0x156d1578,0x656e0571,0x856f1748,0xa5701748,
        0xb7491748,0x65721575,0x85731748,0xa5741748,0xb7491748,0x85761748,0xa5771748,0xb7491748,
        0x67480579,0x657a1748,0x857b1748,0xa57c1748,0xb7491748,0x257e0598,0x757f1748,0x97480580,
        0x95811748,0x15820587,0x67480583,0x65841748,0x85851748,0xa5861748,0xb7491748,0x15881593,
        0x6589058c,0x858a1748,0xa58b1748,0xb7491748,0x658d1590,0x858e1748,0xa58f1748,0xb7491748,
        0x85911748,0xa5921748,0xb7491748,0x67480594,0x65951748,0x85961748,0xa5971748,0xb7491748,
        0x2599163b,0x759a05cc,0x959b05ab,0x1748059c,0x159d1748,0x659e05a2,0x359f1748,0x45a01748,
        0xa5a11748,0xb7491748,0x65a315a7,0x35a41748,0x45a51748,0xa5a61748,0xb7491748,0x35a81748,
        0x45a91748,0xa5aa1748,0xb7491748,0x95ac15bc,0x174805ad,0x15ae1748,0x65af05b3,0xa5b01748,
        0xb5b11748,0x374915b2,0x87491748,0x65b415b8,0xa5b51748,0xb5b61748,0x374915b7,0x87491748,
        0xa5b91748,0xb5ba1748,0x374915bb,0x87491748,0x174805bd,0x15be1748,0x65bf05c3,0x35c01748,
        0x45c11748,0xa5c21748,0xb7491748,0x65c415c8,0x35c51748,0x45c61748,0xa5c71748,0xb7491748,
        0x35c91748,0x45ca1748,0xa5cb1748,0xb7491748,0x95cd05ee,0x75ce15de,0x174805cf,0x15d01748,
        0x65d105d5,0x35d21748,0x45d31748,0xa5d41748,0xb7491748,0x65d615da,0x35d71748,0x45d81748,
        0xa5d91748,0xb7491748,0x35db1748,0x45dc1748,0xa5dd1748,0xb7491748,0x174805df,0x15e01748,
        0x65e105e5,0x35e21748,0x45e31748,0xa5e41748,0xb7491748,0x65e615ea,0x35e71748,0x45e81748,
        0xa5e91748,0xb7491748,0x35eb1748,0x45ec1748,0xa5ed1748,0xb7491748,0x75ef161a,0x95f0160a,
        0x15f105f6,0x674805f2,0x65f31748,0x85f41748,0xa5f51748,0xb7491748,0x15f71605,0x65f805fc,
        0xa5f91748,0xb5fa1748,0x374915fb,0x87491748,0x65fd1601,0xa5fe1748,0xb5ff1748,0x37491600,
        0x87491748,0xa6021748,0xb6031748,0x37491604,0x87491748,0x67480606,0x66071748,0x86081748,
        0xa6091748,0xb7491748,0x1748060b,0x160c1748,0x660d0611,0x360e1748,0x460f1748,0xa6101748,
        0xb7491748,0x66121616,0x36131748,0x46141748,0xa6151748,0xb7491748,0x36171748,0x46181748,
        0xa6191748,0xb7491748,0x961b162b,0x1748061c,0x161d1748,0x661e0622,0xa61f1748,0xb6201748,
        0x37491621,0x87491748,0x66231627,0xa6241748,0xb6251748,0x37491626,0x87491748,0xa6281748,
        0xb6291748,0x3749162a,0x87491748,0x1748062c,0x162d1748,0x662e0632,0x362f1748,0x46301748,
        0xa6311748,0xb7491748,0x66331637,0x36341748,0x46351748,0xa6361748,0xb7491748,0x36381748,
        0x46391748,0xa63a1748,0xb7491748,0x763c1748,0x9748063d,0x963e1748,0x163f0644,0x67480640,
        0x66411748,0x86421748,0xa6431748,0xb7491748,0x16451650,0x66460649,0x86471748,0xa6481748,
        0xb7491748,0x664a164d,0x864b1748,0xa64c1748,0xb7491748,0x864e1748,0xa64f1748,0xb7491748,
        0x67480651,0x66521748,0x86531748,0xa6541748,0xb7491748,0x565616cf,0x77480657,0x76581748,
        0x26590675,0x9748065a,0x965b1748,0x165c0664,0x6748065d,0x665e1748,0x865f1748,0x46601662,
        0x37491661,0xa7491748,0xa6631748,0xb7491748,0x1665166d,0x67480666,0x66671748,0x86681748,
        0x4669166b,0x3749166a,0xa7491748,0xa66c1748,0xb7491748,0x6748066e,0x666f1748,0x86701748,
        0x46711673,0x37491672,0xa7491748,0xa6741748,0xb7491748,0x267616b3,0x96770687,0x1678167c,
        0x67480679,0x667a1748,0x367b1748,0x47491748,0x167d0682,0x6748067e,0x667f1748,0x36801748,
        0x46811748,0x87491748,0x67480683,0x66841748,0x36851748,0x46861748,0x87491748,0x968816a3,
        0x16890691,0x6748068a,0x668b1748,0x868c1748,0x468d168f,0x3749168e,0xa7491748,0xa6901748,
        0xb7491748,0x1692169b,0x67480693,0x66941748,0x46951698,0x37491696,0x86971748,0xa7491748,
        0x86991748,0xa69a1748,0xb7491748,0x6748069c,0x669d1748,0x869e1748,0x469f16a1,0x374916a0,
        0xa7491748,0xa6a21748,0xb7491748,0x16a416a8,0x674806a5,0x66a61748,0x36a71748,0x47491748,
        0x16a906ae,0x674806aa,0x66ab1748,0x36ac1748,0x46ad1748,0x87491748,0x674806af,0x66b01748,
        0x36b11748,0x46b21748,0x87491748,0x974806b4,0x96b51748,0x16b606be,0x674806b7,0x66b81748,
        0x86b91748,0x46ba16bc,0x374916bb,0xa7491748,0xa6bd1748,0xb7491748,0x16bf16c7,0x674806c0,
        0x66c11748,0x86c21748,0x46c316c5,0x374916c4,0xa7491748,0xa6c61748,0xb7491748,0x674806c8,
        0x66c91748,0x86ca1748,0x46cb16cd,0x374916cc,0xa7491748,0xa6ce1748,0xb7491748,0x56d00748,
        0x76d10748,0x26d216ee,0x974816d3,0x96d40748,0x16d506dd,0x674816d6,0x66d70748,0x86d80748,
        0x46d906db,0x374906da,0xa7490748,0xa6dc0748,0xb7490748,0x16de16e6,0x674816df,0x66e00748,
        0x86e10748,0x46e206e4,0x374906e3,0xa7490748,0xa6e50748,0xb7490748,0x674816e7,0x66e80748,
        0x86e90748,0x46ea06ec,0x374906eb,0xa7490748,0xa6ed0748,0xb7490748,0x26ef072c,0x96f01700,
        0x16f106f5,0x674816f2,0x66f30748,0x36f40748,0x47490748,0x16f616fb,0x674816f7,0x66f80748,
        0x36f90748,0x46fa0748,0x87490748,0x674816fc,0x66fd0748,0x36fe0748,0x46ff0748,0x87490748,
        0x9701071c,0x1702170a,0x67481703,0x67040748,0x87050748,0x47060708,0x37490707,0xa7490748,
        0xa7090748,0xb7490748,0x170b0714,0x6748170c,0x670d0748,0x470e0711,0x3749070f,0x87100748,
        0xa7490748,0x87120748,0xa7130748,0xb7490748,0x67481715,0x67160748,0x87170748,0x4718071a,
        0x37490719,0xa7490748,0xa71b0748,0xb7490748,0x171d0721,0x6748171e,0x671f0748,0x37200748,
        0x47490748,0x17221727,0x67481723,0x67240748,0x37250748,0x47260748,0x87490748,0x67481728,
        0x67290748,0x372a0748,0x472b0748,0x87490748,0x9748172d,0x972e0748,0x172f0737,0x67481730,
        0x67310748,0x87320748,0x47330735,0x37490734,0xa7490748,0xa7360748,0xb7490748,0x17381740,
        0x67481739,0x673a0748,0x873b0748,0x473c073e,0x3749073d,0xa7490748,0xa73f0748,0xb7490748,
        0x67481741,0x67420748,0x87430748,0x47440746,0x37490745,0xa7490748,0xa7470748,0xb7490748,
        0x000000fe,0x000000ff};

    static const uint32_t table_9_16_corner_struct[] =
    {       0x00010138,0x200200d3,0x4003008a,0x50040051,0x70050027,0x30060016,0x1007000d,0x6008000a,
        0x82ad0009,0xf2ad02ac,0xd00b02ac,0xe00c02ac,0xf2ad02ac,0x800e02ac,0x900f02ac,0xa01002ac,
        0x62ad0011,0xb01202ac,0xc01302ac,0xd01402ac,0xe01502ac,0xf2ad02ac,0xa01702ac,0xb01802ac,
        0xc01902ac,0x801a0023,0x901b001f,0x62ad001c,0xd01d02ac,0xe01e02ac,0xf2ad02ac,0x102002ac,
        0xd02102ac,0xe02202ac,0xf2ad02ac,0x102402ac,0xd02502ac,0xe02602ac,0xf2ad02ac,0x70281041,
        0xe0290038,0xf02a02ac,0x102b0032,0x302c002e,0x62ad002d,0xd2ad02ac,0xa02f02ac,0xb03002ac,
        0xc03102ac,0xd2ad02ac,0x803302ac,0x903402ac,0xa03502ac,0xb03602ac,0xc03702ac,0xd2ad02ac,
        0xe03912ac,0x803a12ac,0x903b12ac,0xa03c12ac,0xb03d12ac,0xc03e12ac,0xd03f12ac,0x62ad1040,
        0xf2ad12ac,0xe04202ac,0xf04302ac,0x1044004b,0x30450047,0x62ad0046,0xd2ad02ac,0xa04802ac,
        0xb04902ac,0xc04a02ac,0xd2ad02ac,0x804c02ac,0x904d02ac,0xa04e02ac,0xb04f02ac,0xc05002ac,
        0xd2ad02ac,0x5052106e,0xc0530064,0xd05402ac,0xe05502ac,0xf056005e,0x1057005a,0x32ad0058,
        0xa05902ac,0xb2ad02ac,0x805b02ac,0x905c02ac,0xa05d02ac,0xb2ad02ac,0x605f02ac,0x706002ac,
        0x806102ac,0x906202ac,0xa06302ac,0xb2ad02ac,0xc06512ac,0x706612ac,0x806712ac,0x906812ac,
        0xa06912ac,0xb06a12ac,0xd06b12ac,0x62ad106c,0xe06d12ac,0xf2ad12ac,0xc06f0080,0xd07002ac,
        0xe07102ac,0xf072007a,0x10730076,0x32ad0074,0xa07502ac,0xb2ad02ac,0x807702ac,0x907802ac,
        0xa07902ac,0xb2ad02ac,0x607b02ac,0x707c02ac,0x807d02ac,0x907e02ac,0xa07f02ac,0xb2ad02ac,
        0xc08112ac,0x708212ac,0x808312ac,0x908412ac,0xa08512ac,0xb08612ac,0xd08712ac,0xe08812ac,
        0x62ad1089,0xf2ad12ac,0x408b10b1,0xb08c00a1,0xc08d02ac,0xd08e02ac,0xa08f009d,0xe0900098,
        0xf0910094,0x12ad0092,0x809302ac,0x92ad02ac,0x609502ac,0x709602ac,0x809702ac,0x92ad02ac,
        0x509902ac,0x609a02ac,0x709b02ac,0x809c02ac,0x92ad02ac,0x109e02ac,0x309f02ac,0xe0a002ac,
        0xf2ad02ac,0xb0a212ac,0x70a312ac,0x80a412ac,0x90a512ac,0xa0a612ac,0x60a710ad,0x50a810aa,
        0x32ad10a9,0xc2ad12ac,0xc0ab12ac,0xd0ac12ac,0xe2ad12ac,0xc0ae12ac,0xd0af12ac,0xe0b012ac,
        0xf2ad12ac,0xb0b200c7,0xc0b302ac,0xd0b402ac,0xa0b500c3,0xe0b600be,0xf0b700ba,0x12ad00b8,
        0x80b902ac,0x92ad02ac,0x60bb02ac,0x70bc02ac,0x80bd02ac,0x92ad02ac,0x50bf02ac,0x60c002ac,
        0x70c102ac,0x80c202ac,0x92ad02ac,0x10c402ac,0x30c502ac,0xe0c602ac,0xf2ad02ac,0xb0c812ac,
        0x70c912ac,0x80ca12ac,0x90cb12ac,0xa0cc12ac,0xc0cd12ac,0xd0ce12ac,0x60cf10d1,0x52ad10d0,
        0xe2ad12ac,0xe0d212ac,0xf2ad12ac,0x20d4110a,0x90d500ef,0xa0d602ac,0xb0d702ac,0x80d800ea,
        0xc0d900e5,0xd0da00e1,0xe0db00de,0xf2ad00dc,0x60dd02ac,0x72ad02ac,0x50df02ac,0x60e002ac,
        0x72ad02ac,0x40e202ac,0x50e302ac,0x60e402ac,0x72ad02ac,0x30e602ac,0x40e702ac,0x50e802ac,
        0x60e902ac,0x72ad02ac,0x10eb02ac,0xc0ec02ac,0xd0ed02ac,0xe0ee02ac,0xf2ad02ac,0x90f012ac,
        0x70f112ac,0x80f212ac,0x60f31104,0x50f410ff,0x40f510fb,0x30f610f8,0x12ad10f7,0xa2ad12ac,
        0xa0f912ac,0xb0fa12ac,0xc2ad12ac,0xa0fc12ac,0xb0fd12ac,0xc0fe12ac,0xd2ad12ac,0xa10012ac,
        0xb10112ac,0xc10212ac,0xd10312ac,0xe2ad12ac,0xa10512ac,0xb10612ac,0xc10712ac,0xd10812ac,
        0xe10912ac,0xf2ad12ac,0x910b0125,0xa10c02ac,0xb10d02ac,0x810e0120,0xc10f011b,0xd1100117,
        0xe1110114,0xf2ad0112,0x611302ac,0x72ad02ac,0x511502ac,0x611602ac,0x72ad02ac,0x411802ac,
        0x511902ac,0x611a02ac,0x72ad02ac,0x311c02ac,0x411d02ac,0x511e02ac,0x611f02ac,0x72ad02ac,
        0x112102ac,0xc12202ac,0xd12302ac,0xe12402ac,0xf2ad02ac,0x912612ac,0x712712ac,0x812812ac,
        0xa12912ac,0xb12a12ac,0x612b1134,0x512c1131,0x412d112f,0x32ad112e,0xc2ad12ac,0xc13012ac,
        0xd2ad12ac,0xc13212ac,0xd13312ac,0xe2ad12ac,0xc13512ac,0xd13612ac,0xe13712ac,0xf2ad12ac,
        0x01391270,0x213a0170,0x913b0155,0x713c02ac,0x813d02ac,0x613e014f,0x513f014a,0x41400146,
        0x31410143,0x12ad0142,0xa2ad02ac,0xa14402ac,0xb14502ac,0xc2ad02ac,0xa14702ac,0xb14802ac,
        0xc14902ac,0xd2ad02ac,0xa14b02ac,0xb14c02ac,0xc14d02ac,0xd14e02ac,0xe2ad02ac,0xa15002ac,
        0xb15102ac,0xc15202ac,0xd15302ac,0xe15402ac,0xf2ad02ac,0x915612ac,0xa15712ac,0xb15812ac,
        0x8159116b,0xc15a1166,0xd15b1162,0xe15c115f,0xf2ad115d,0x615e12ac,0x72ad12ac,0x516012ac,
        0x616112ac,0x72ad12ac,0x416312ac,0x516412ac,0x616512ac,0x72ad12ac,0x316712ac,0x416812ac,
        0x516912ac,0x616a12ac,0x72ad12ac,0x116c12ac,0xc16d12ac,0xd16e12ac,0xe16f12ac,0xf2ad12ac,
        0x21711242,0x41720198,0xb1730182,0x717402ac,0x817502ac,0x917602ac,0xa17702ac,0x6178017e,
        0x5179017b,0x32ad017a,0xc2ad02ac,0xc17c02ac,0xd17d02ac,0xe2ad02ac,0xc17f02ac,0xd18002ac,
        0xe18102ac,0xf2ad02ac,0xb18312ac,0xc18412ac,0xd18512ac,0xa1861194,0xe187118f,0xf188118b,
        0x12ad1189,0x818a12ac,0x92ad12ac,0x618c12ac,0x718d12ac,0x818e12ac,0x92ad12ac,0x519012ac,
        0x619112ac,0x719212ac,0x819312ac,0x92ad12ac,0x119512ac,0x319612ac,0xe19712ac,0xf2ad12ac,
        0x41991220,0x519a01b6,0xc19b01a4,0x719c02ac,0x819d02ac,0x919e02ac,0xa19f02ac,0xb1a002ac,
        0xd1a102ac,0x62ad01a2,0xe1a302ac,0xf2ad02ac,0xc1a512ac,0xd1a612ac,0xe1a712ac,0xf1a811b0,
        0x11a911ac,0x32ad11aa,0xa1ab12ac,0xb2ad12ac,0x81ad12ac,0x91ae12ac,0xa1af12ac,0xb2ad12ac,
        0x61b112ac,0x71b212ac,0x81b312ac,0x91b412ac,0xa1b512ac,0xb2ad12ac,0x51b71204,0x71b801d1,
        0xe1b901c1,0x81ba02ac,0x91bb02ac,0xa1bc02ac,0xb1bd02ac,0xc1be02ac,0xd1bf02ac,0x62ad01c0,
        0xf2ad02ac,0xe1c212ac,0xf1c312ac,0x11c411cb,0x31c511c7,0x62ad11c6,0xd2ad12ac,0xa1c812ac,
        0xb1c912ac,0xc1ca12ac,0xd2ad12ac,0x81cc12ac,0x91cd12ac,0xa1ce12ac,0xb1cf12ac,0xc1d012ac,
        0xd2ad12ac,0x71d211f4,0x31d311e3,0x11d411da,0x61d511d7,0x82ad11d6,0xf2ad12ac,0xd1d812ac,
        0xe1d912ac,0xf2ad12ac,0x81db12ac,0x91dc12ac,0xa1dd12ac,0x62ad11de,0xb1df12ac,0xc1e012ac,
        0xd1e112ac,0xe1e212ac,0xf2ad12ac,0xa1e412ac,0xb1e512ac,0xc1e612ac,0x81e711f0,0x91e811ec,
        0x62ad11e9,0xd1ea12ac,0xe1eb12ac,0xf2ad12ac,0x11ed12ac,0xd1ee12ac,0xe1ef12ac,0xf2ad12ac,
        0x11f112ac,0xd1f212ac,0xe1f312ac,0xf2ad12ac,0xe1f512ac,0xf1f612ac,0x11f711fe,0x31f811fa,
        0x62ad11f9,0xd2ad12ac,0xa1fb12ac,0xb1fc12ac,0xc1fd12ac,0xd2ad12ac,0x81ff12ac,0x920012ac,
        0xa20112ac,0xb20212ac,0xc20312ac,0xd2ad12ac,0xc205020e,0x720602ac,0x820702ac,0x920802ac,
        0xa20902ac,0xb20a02ac,0xd20b02ac,0xe20c02ac,0x62ad020d,0xf2ad02ac,0xc20f12ac,0xd21012ac,
        0xe21112ac,0xf212121a,0x12131216,0x32ad1214,0xa21512ac,0xb2ad12ac,0x821712ac,0x921812ac,
        0xa21912ac,0xb2ad12ac,0x621b12ac,0x721c12ac,0x821d12ac,0x921e12ac,0xa21f12ac,0xb2ad12ac,
        0xb221022c,0x722202ac,0x822302ac,0x922402ac,0xa22502ac,0xc22602ac,0xd22702ac,0x6228022a,
        0x52ad0229,0xe2ad02ac,0xe22b02ac,0xf2ad02ac,0xb22d12ac,0xc22e12ac,0xd22f12ac,0xa230123e,
        0xe2311239,0xf2321235,0x12ad1233,0x823412ac,0x92ad12ac,0x623612ac,0x723712ac,0x823812ac,
        0x92ad12ac,0x523a12ac,0x623b12ac,0x723c12ac,0x823d12ac,0x92ad12ac,0x123f12ac,0x324012ac,
        0xe24112ac,0xf2ad12ac,0x92430255,0x724402ac,0x824502ac,0xa24602ac,0xb24702ac,0x62480251,
        0x5249024e,0x424a024c,0x32ad024b,0xc2ad02ac,0xc24d02ac,0xd2ad02ac,0xc24f02ac,0xd25002ac,
        0xe2ad02ac,0xc25202ac,0xd25302ac,0xe25402ac,0xf2ad02ac,0x925612ac,0xa25712ac,0xb25812ac,
        0x8259126b,0xc25a1266,0xd25b1262,0xe25c125f,0xf2ad125d,0x625e12ac,0x72ad12ac,0x526012ac,
        0x626112ac,0x72ad12ac,0x426312ac,0x526412ac,0x626512ac,0x72ad12ac,0x326712ac,0x426812ac,
        0x526912ac,0x626a12ac,0x72ad12ac,0x126c12ac,0xc26d12ac,0xd26e12ac,0xe26f12ac,0xf2ad12ac,
        0x7271028e,0x827202ac,0x927302ac,0x62740288,0x52750283,0x4276027f,0x3277027c,0x2278027a,
        0x12ad0279,0xa2ad02ac,0xa27b02ac,0xb2ad02ac,0xa27d02ac,0xb27e02ac,0xc2ad02ac,0xa28002ac,
        0xb28102ac,0xc28202ac,0xd2ad02ac,0xa28402ac,0xb28502ac,0xc28602ac,0xd28702ac,0xe2ad02ac,
        0xa28902ac,0xb28a02ac,0xc28b02ac,0xd28c02ac,0xe28d02ac,0xf2ad02ac,0x728f12ac,0x829012ac,
        0x929112ac,0x629212a6,0x529312a1,0x4294129d,0x3295129a,0x22961298,0x12ad1297,0xa2ad12ac,
        0xa29912ac,0xb2ad12ac,0xa29b12ac,0xb29c12ac,0xc2ad12ac,0xa29e12ac,0xb29f12ac,0xc2a012ac,
        0xd2ad12ac,0xa2a212ac,0xb2a312ac,0xc2a412ac,0xd2a512ac,0xe2ad12ac,0xa2a712ac,0xb2a812ac,
        0xc2a912ac,0xd2aa12ac,0xe2ab12ac,0xf2ad12ac,0x000000fe,0x000000ff};


    switch(agasttype) {
      case AgastFeatureDetector::AGAST_5_8:
        table_struct=(uint32_t *)(table_5_8_corner_struct);
        break;
      case AgastFeatureDetector::AGAST_7_12d:
        table_struct=(uint32_t *)(table_7_12d_corner_struct);
        break;
      case AgastFeatureDetector::AGAST_7_12s:
        table_struct=(uint32_t *)(table_7_12s_corner_struct);
        break;
      case AgastFeatureDetector::OAST_9_16:
      default:
        table_struct=(uint32_t *)(table_9_16_corner_struct);
        break;
    }

    while(true)
    {
        result = agast_tree_search(table_struct, (int *)pixel, ptr, b_test);
        if (result == 254)
            bmax = b_test;
        else
            bmin = b_test;

        if(bmin == bmax - 1 || bmin == bmax)
            return bmin;
        b_test = (bmin + bmax) / 2;
    }
}

// 8 pixel mask
template<>
int agast_cornerScore<AgastFeatureDetector::AGAST_5_8>(const uchar* ptr, const int pixel[], int threshold)
{
    return AGAST_ALL_SCORE(ptr, pixel, threshold, AgastFeatureDetector::AGAST_5_8);
}

// 12 pixel mask in square format
template<>
int agast_cornerScore<AgastFeatureDetector::AGAST_7_12d>(const uchar* ptr, const int pixel[], int threshold)
{
    return AGAST_ALL_SCORE(ptr, pixel, threshold, AgastFeatureDetector::AGAST_7_12d);
}

// 12 pixel mask in diamond format
template<>
int agast_cornerScore<AgastFeatureDetector::AGAST_7_12s>(const uchar* ptr, const int pixel[], int threshold)
{
    return AGAST_ALL_SCORE(ptr, pixel, threshold, AgastFeatureDetector::AGAST_7_12s);
}

// 16 pixel mask
template<>
int agast_cornerScore<AgastFeatureDetector::OAST_9_16>(const uchar* ptr, const int pixel[], int threshold)
{
    return AGAST_ALL_SCORE(ptr, pixel, threshold, AgastFeatureDetector::OAST_9_16);
}

#endif // !(defined __i386__ || defined(_M_IX86) || defined __x86_64__ || defined(_M_X64))

} // namespace cv
