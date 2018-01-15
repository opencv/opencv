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

#include "precomp.hpp"
#include "agast_score.hpp"

namespace cv
{

#if (defined __i386__ || defined(_M_IX86) || defined __x86_64__ || defined(_M_X64))

static void AGAST_5_8(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold)
{

    cv::Mat img;
    if(!_img.getMat().isContinuous())
      img = _img.getMat().clone();
    else
      img = _img.getMat();

    size_t total = 0;
    int xsize = img.cols;
    int ysize = img.rows;
    size_t nExpectedCorners = keypoints.capacity();
    int x, y;
    int xsizeB = xsize - 2;
    int ysizeB = ysize - 1;
    int width;

    keypoints.resize(0);

    int pixel_5_8_[16];
    makeAgastOffsets(pixel_5_8_, (int)img.step, AgastFeatureDetector::AGAST_5_8);

    short offset0 = (short) pixel_5_8_[0];
    short offset1 = (short) pixel_5_8_[1];
    short offset2 = (short) pixel_5_8_[2];
    short offset3 = (short) pixel_5_8_[3];
    short offset4 = (short) pixel_5_8_[4];
    short offset5 = (short) pixel_5_8_[5];
    short offset6 = (short) pixel_5_8_[6];
    short offset7 = (short) pixel_5_8_[7];

    width = xsize;

    for(y = 1; y < ysizeB; y++)
    {
        x = 0;
        while(true)
        {
          homogeneous:
          {
            x++;
            if(x > xsizeB)
                break;
            else
            {
                const unsigned char* const ptr = img.ptr() + y*width + x;
                const int cb = *ptr + threshold;
                const int c_b = *ptr - threshold;
                if(ptr[offset0] > cb)
                  if(ptr[offset2] > cb)
                    if(ptr[offset3] > cb)
                      if(ptr[offset5] > cb)
                        if(ptr[offset1] > cb)
                          if(ptr[offset4] > cb)
                            goto success_structured;
                          else
                            if(ptr[offset7] > cb)
                              goto success_structured;
                            else
                              goto homogeneous;
                        else
                          if(ptr[offset4] > cb)
                            if(ptr[offset6] > cb)
                              goto success_structured;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                      else
                        if(ptr[offset1] > cb)
                          if(ptr[offset4] > cb)
                            goto success_homogeneous;
                          else
                            if(ptr[offset7] > cb)
                              goto success_homogeneous;
                            else
                              goto homogeneous;
                        else
                          goto homogeneous;
                    else
                      if(ptr[offset7] > cb)
                        if(ptr[offset6] > cb)
                          if(ptr[offset5] > cb)
                            if(ptr[offset1] > cb)
                              goto success_structured;
                            else
                              if(ptr[offset4] > cb)
                                goto success_structured;
                              else
                                goto homogeneous;
                          else
                            if(ptr[offset1] > cb)
                              goto success_homogeneous;
                            else
                              goto homogeneous;
                        else
                          goto homogeneous;
                      else
                        if(ptr[offset5] < c_b)
                          if(ptr[offset3] < c_b)
                            if(ptr[offset7] < c_b)
                              if(ptr[offset4] < c_b)
                                if(ptr[offset6] < c_b)
                                  goto success_structured;
                                else
                                  goto structured;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                  else
                    if(ptr[offset5] > cb)
                      if(ptr[offset7] > cb)
                        if(ptr[offset6] > cb)
                          if(ptr[offset1] > cb)
                            goto success_homogeneous;
                          else
                            if(ptr[offset4] > cb)
                              goto success_homogeneous;
                            else
                              goto homogeneous;
                        else
                          goto homogeneous;
                      else
                        goto homogeneous;
                    else
                      if(ptr[offset5] < c_b)
                        if(ptr[offset3] < c_b)
                          if(ptr[offset2] < c_b)
                            if(ptr[offset1] < c_b)
                              if(ptr[offset4] < c_b)
                                goto success_structured;
                              else
                                goto homogeneous;
                            else
                              if(ptr[offset4] < c_b)
                                if(ptr[offset6] < c_b)
                                  goto success_structured;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            if(ptr[offset7] < c_b)
                              if(ptr[offset4] < c_b)
                                if(ptr[offset6] < c_b)
                                  goto success_structured;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                        else
                          goto homogeneous;
                      else
                        goto homogeneous;
                else
                if(ptr[offset0] < c_b)
                  if(ptr[offset2] < c_b)
                    if(ptr[offset7] > cb)
                      if(ptr[offset3] < c_b)
                        if(ptr[offset5] < c_b)
                          if(ptr[offset1] < c_b)
                            if(ptr[offset4] < c_b)
                              goto success_structured;
                            else
                              goto structured;
                          else
                            if(ptr[offset4] < c_b)
                              if(ptr[offset6] < c_b)
                                goto success_structured;
                              else
                                goto structured;
                            else
                              goto homogeneous;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset4] < c_b)
                              goto success_structured;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                      else
                        if(ptr[offset5] > cb)
                          if(ptr[offset3] > cb)
                            if(ptr[offset4] > cb)
                              if(ptr[offset6] > cb)
                                goto success_structured;
                              else
                                goto structured;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                    else
                      if(ptr[offset7] < c_b)
                        if(ptr[offset3] < c_b)
                          if(ptr[offset5] < c_b)
                            if(ptr[offset1] < c_b)
                              goto success_structured;
                            else
                              if(ptr[offset4] < c_b)
                                if(ptr[offset6] < c_b)
                                  goto success_structured;
                                else
                                  goto structured;
                              else
                                goto homogeneous;
                          else
                            if(ptr[offset1] < c_b)
                              goto success_homogeneous;
                            else
                              goto homogeneous;
                        else
                          if(ptr[offset6] < c_b)
                            if(ptr[offset5] < c_b)
                              if(ptr[offset1] < c_b)
                                goto success_structured;
                              else
                                if(ptr[offset4] < c_b)
                                  goto success_structured;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset1] < c_b)
                                goto success_homogeneous;
                              else
                                goto homogeneous;
                          else
                            goto homogeneous;
                      else
                        if(ptr[offset3] < c_b)
                          if(ptr[offset5] < c_b)
                            if(ptr[offset1] < c_b)
                              if(ptr[offset4] < c_b)
                                goto success_structured;
                              else
                                goto homogeneous;
                            else
                              if(ptr[offset4] < c_b)
                                if(ptr[offset6] < c_b)
                                  goto success_structured;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            if(ptr[offset1] < c_b)
                              if(ptr[offset4] < c_b)
                                goto success_homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                        else
                          goto homogeneous;
                  else
                    if(ptr[offset5] > cb)
                      if(ptr[offset3] > cb)
                        if(ptr[offset2] > cb)
                          if(ptr[offset1] > cb)
                            if(ptr[offset4] > cb)
                              goto success_structured;
                            else
                              goto homogeneous;
                          else
                            if(ptr[offset4] > cb)
                              if(ptr[offset6] > cb)
                                goto success_structured;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                        else
                          if(ptr[offset7] > cb)
                            if(ptr[offset4] > cb)
                              if(ptr[offset6] > cb)
                                goto success_structured;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                      else
                        goto homogeneous;
                    else
                      if(ptr[offset5] < c_b)
                        if(ptr[offset7] < c_b)
                          if(ptr[offset6] < c_b)
                            if(ptr[offset1] < c_b)
                              goto success_homogeneous;
                            else
                              if(ptr[offset4] < c_b)
                                goto success_homogeneous;
                              else
                                goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                      else
                        goto homogeneous;
                else
                  if(ptr[offset3] > cb)
                    if(ptr[offset5] > cb)
                      if(ptr[offset2] > cb)
                        if(ptr[offset1] > cb)
                          if(ptr[offset4] > cb)
                            goto success_homogeneous;
                          else
                            goto homogeneous;
                        else
                          if(ptr[offset4] > cb)
                            if(ptr[offset6] > cb)
                              goto success_homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                      else
                        if(ptr[offset7] > cb)
                          if(ptr[offset4] > cb)
                            if(ptr[offset6] > cb)
                              goto success_homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                    else
                      goto homogeneous;
                  else
                    if(ptr[offset3] < c_b)
                      if(ptr[offset5] < c_b)
                        if(ptr[offset2] < c_b)
                          if(ptr[offset1] < c_b)
                            if(ptr[offset4] < c_b)
                              goto success_homogeneous;
                            else
                              goto homogeneous;
                          else
                            if(ptr[offset4] < c_b)
                              if(ptr[offset6] < c_b)
                                goto success_homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                        else
                          if(ptr[offset7] < c_b)
                            if(ptr[offset4] < c_b)
                              if(ptr[offset6] < c_b)
                                goto success_homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                      else
                        goto homogeneous;
                    else
                      goto homogeneous;
            }
          }
          structured:
          {
            x++;
            if(x > xsizeB)
                break;
            else
            {
                const unsigned char* const ptr = img.ptr() + y*width + x;
                const int cb = *ptr + threshold;
                const int c_b = *ptr - threshold;
                if(ptr[offset0] > cb)
                  if(ptr[offset2] > cb)
                    if(ptr[offset3] > cb)
                      if(ptr[offset5] > cb)
                        if(ptr[offset7] > cb)
                          if(ptr[offset1] > cb)
                            goto success_structured;
                          else
                            if(ptr[offset4] > cb)
                              if(ptr[offset6] > cb)
                                goto success_structured;
                              else
                                goto structured;
                            else
                              goto structured;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset4] > cb)
                              goto success_structured;
                            else
                              goto structured;
                          else
                            if(ptr[offset4] > cb)
                              if(ptr[offset6] > cb)
                                goto success_structured;
                              else
                                goto structured;
                            else
                              goto structured;
                      else
                        if(ptr[offset7] > cb)
                          if(ptr[offset1] > cb)
                            goto success_structured;
                          else
                            goto structured;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset4] > cb)
                              goto success_structured;
                            else
                              goto structured;
                          else
                            goto structured;
                    else
                      if(ptr[offset7] > cb)
                        if(ptr[offset6] > cb)
                          if(ptr[offset5] > cb)
                            if(ptr[offset1] > cb)
                              goto success_structured;
                            else
                              if(ptr[offset4] > cb)
                                goto success_structured;
                              else
                                goto structured;
                          else
                            if(ptr[offset1] > cb)
                              goto success_structured;
                            else
                              goto structured;
                        else
                          goto structured;
                      else
                        if(ptr[offset5] < c_b)
                          if(ptr[offset3] < c_b)
                            if(ptr[offset7] < c_b)
                              if(ptr[offset4] < c_b)
                                if(ptr[offset6] < c_b)
                                  goto success_structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto structured;
                  else
                    if(ptr[offset5] > cb)
                      if(ptr[offset7] > cb)
                        if(ptr[offset6] > cb)
                          if(ptr[offset1] > cb)
                            goto success_structured;
                          else
                            if(ptr[offset4] > cb)
                              goto success_structured;
                            else
                              goto structured;
                        else
                          goto structured;
                      else
                        goto structured;
                    else
                      if(ptr[offset5] < c_b)
                        if(ptr[offset3] < c_b)
                          if(ptr[offset2] < c_b)
                            if(ptr[offset1] < c_b)
                              if(ptr[offset4] < c_b)
                                goto success_structured;
                              else
                                goto structured;
                            else
                              if(ptr[offset4] < c_b)
                                if(ptr[offset6] < c_b)
                                  goto success_structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            if(ptr[offset7] < c_b)
                              if(ptr[offset4] < c_b)
                                if(ptr[offset6] < c_b)
                                  goto success_homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                        else
                          goto structured;
                      else
                        goto homogeneous;
                else
                if(ptr[offset0] < c_b)
                  if(ptr[offset2] < c_b)
                    if(ptr[offset7] > cb)
                      if(ptr[offset3] < c_b)
                        if(ptr[offset5] < c_b)
                          if(ptr[offset1] < c_b)
                            if(ptr[offset4] < c_b)
                              goto success_structured;
                            else
                              goto structured;
                          else
                            if(ptr[offset4] < c_b)
                              if(ptr[offset6] < c_b)
                                goto success_structured;
                              else
                                goto structured;
                            else
                              goto structured;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset4] < c_b)
                              goto success_structured;
                            else
                              goto structured;
                          else
                            goto structured;
                      else
                        if(ptr[offset5] > cb)
                          if(ptr[offset3] > cb)
                            if(ptr[offset4] > cb)
                              if(ptr[offset6] > cb)
                                goto success_structured;
                              else
                                goto structured;
                            else
                              goto structured;
                          else
                            goto homogeneous;
                        else
                          goto structured;
                    else
                      if(ptr[offset7] < c_b)
                        if(ptr[offset3] < c_b)
                          if(ptr[offset5] < c_b)
                            if(ptr[offset1] < c_b)
                              goto success_structured;
                            else
                              if(ptr[offset4] < c_b)
                                if(ptr[offset6] < c_b)
                                  goto success_structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            if(ptr[offset1] < c_b)
                              goto success_structured;
                            else
                              goto structured;
                        else
                          if(ptr[offset6] < c_b)
                            if(ptr[offset5] < c_b)
                              if(ptr[offset1] < c_b)
                                goto success_structured;
                              else
                                if(ptr[offset4] < c_b)
                                  goto success_structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset1] < c_b)
                                goto success_structured;
                              else
                                goto structured;
                          else
                            goto structured;
                      else
                        if(ptr[offset3] < c_b)
                          if(ptr[offset5] < c_b)
                            if(ptr[offset1] < c_b)
                              if(ptr[offset4] < c_b)
                                goto success_homogeneous;
                              else
                                goto homogeneous;
                            else
                              if(ptr[offset4] < c_b)
                                if(ptr[offset6] < c_b)
                                  goto success_homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            if(ptr[offset1] < c_b)
                              if(ptr[offset4] < c_b)
                                goto success_homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                        else
                          goto homogeneous;
                  else
                    if(ptr[offset5] > cb)
                      if(ptr[offset3] > cb)
                        if(ptr[offset2] > cb)
                          if(ptr[offset1] > cb)
                            if(ptr[offset4] > cb)
                              goto success_structured;
                            else
                              goto structured;
                          else
                            if(ptr[offset4] > cb)
                              if(ptr[offset6] > cb)
                                goto success_structured;
                              else
                                goto structured;
                            else
                              goto structured;
                        else
                          if(ptr[offset7] > cb)
                            if(ptr[offset4] > cb)
                              if(ptr[offset6] > cb)
                                goto success_homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                      else
                        goto structured;
                    else
                      if(ptr[offset5] < c_b)
                        if(ptr[offset7] < c_b)
                          if(ptr[offset6] < c_b)
                            if(ptr[offset1] < c_b)
                              goto success_structured;
                            else
                              if(ptr[offset4] < c_b)
                                goto success_structured;
                              else
                                goto structured;
                          else
                            goto structured;
                        else
                          goto structured;
                      else
                        goto homogeneous;
                else
                  if(ptr[offset3] > cb)
                    if(ptr[offset5] > cb)
                      if(ptr[offset2] > cb)
                        if(ptr[offset1] > cb)
                          if(ptr[offset4] > cb)
                            goto success_homogeneous;
                          else
                            goto homogeneous;
                        else
                          if(ptr[offset4] > cb)
                            if(ptr[offset6] > cb)
                              goto success_homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                      else
                        if(ptr[offset7] > cb)
                          if(ptr[offset4] > cb)
                            if(ptr[offset6] > cb)
                              goto success_homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                    else
                      goto homogeneous;
                  else
                    if(ptr[offset3] < c_b)
                      if(ptr[offset5] < c_b)
                        if(ptr[offset2] < c_b)
                          if(ptr[offset1] < c_b)
                            if(ptr[offset4] < c_b)
                              goto success_homogeneous;
                            else
                              goto homogeneous;
                          else
                            if(ptr[offset4] < c_b)
                              if(ptr[offset6] < c_b)
                                goto success_homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                        else
                          if(ptr[offset7] < c_b)
                            if(ptr[offset4] < c_b)
                              if(ptr[offset6] < c_b)
                                goto success_homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                      else
                        goto homogeneous;
                    else
                      goto homogeneous;
            }
          }
            success_homogeneous:
            if(total == nExpectedCorners)
            {
                if(nExpectedCorners == 0)
                {
                    nExpectedCorners = 512;
                    keypoints.reserve(nExpectedCorners);
                }
                else
                {
                    nExpectedCorners *= 2;
                    keypoints.reserve(nExpectedCorners);
                }
            }
            keypoints.push_back(KeyPoint(Point2f((float)x, (float)y), 7.0f));
            total++;
            goto homogeneous;
            success_structured:
            if(total == nExpectedCorners)
            {
                if(nExpectedCorners == 0)
                {
                    nExpectedCorners = 512;
                    keypoints.reserve(nExpectedCorners);
                }
                else
                {
                    nExpectedCorners *= 2;
                    keypoints.reserve(nExpectedCorners);
                }
            }
            keypoints.push_back(KeyPoint(Point2f((float)x, (float)y), 7.0f));
            total++;
            goto structured;
        }
    }
}

static void AGAST_7_12d(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold)
{
    cv::Mat img;
    if(!_img.getMat().isContinuous())
      img = _img.getMat().clone();
    else
      img = _img.getMat();

    size_t total = 0;
    int xsize = img.cols;
    int ysize = img.rows;
    size_t nExpectedCorners = keypoints.capacity();
    int x, y;
    int xsizeB = xsize - 4;
    int ysizeB = ysize - 3;
    int width;

    keypoints.resize(0);

    int pixel_7_12d_[16];
    makeAgastOffsets(pixel_7_12d_, (int)img.step, AgastFeatureDetector::AGAST_7_12d);

    short offset0 = (short) pixel_7_12d_[0];
    short offset1 = (short) pixel_7_12d_[1];
    short offset2 = (short) pixel_7_12d_[2];
    short offset3 = (short) pixel_7_12d_[3];
    short offset4 = (short) pixel_7_12d_[4];
    short offset5 = (short) pixel_7_12d_[5];
    short offset6 = (short) pixel_7_12d_[6];
    short offset7 = (short) pixel_7_12d_[7];
    short offset8 = (short) pixel_7_12d_[8];
    short offset9 = (short) pixel_7_12d_[9];
    short offset10 = (short) pixel_7_12d_[10];
    short offset11 = (short) pixel_7_12d_[11];

    width = xsize;

    for(y = 3; y < ysizeB; y++)
    {
        x = 2;
        while(true)
        {
          homogeneous:
          {
            x++;
            if(x > xsizeB)
                break;
            else
            {
                const unsigned char* const ptr = img.ptr() + y*width + x;
                const int cb = *ptr + threshold;
                const int c_b = *ptr - threshold;
                if(ptr[offset0] > cb)
                  if(ptr[offset5] > cb)
                    if(ptr[offset2] > cb)
                      if(ptr[offset9] > cb)
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] > cb)
                            if(ptr[offset3] > cb)
                              if(ptr[offset4] > cb)
                                goto success_homogeneous;
                              else
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto success_structured;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto success_structured;
                                  else
                                    if(ptr[offset4] > cb)
                                      if(ptr[offset7] > cb)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            if(ptr[offset11] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  goto success_homogeneous;
                                else
                                  if(ptr[offset10] > cb)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              goto homogeneous;
                        else
                          if(ptr[offset6] > cb)
                            if(ptr[offset7] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset3] > cb)
                                    goto success_structured;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto success_structured;
                                    else
                                      goto homogeneous;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_structured;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                      else
                        if(ptr[offset3] > cb)
                          if(ptr[offset4] > cb)
                            if(ptr[offset1] > cb)
                              if(ptr[offset6] > cb)
                                goto success_homogeneous;
                              else
                                if(ptr[offset11] > cb)
                                  goto success_homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset7] > cb)
                                  if(ptr[offset8] > cb)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                    else
                      if(ptr[offset9] > cb)
                        if(ptr[offset7] > cb)
                          if(ptr[offset8] > cb)
                            if(ptr[offset1] > cb)
                              if(ptr[offset10] > cb)
                                if(ptr[offset11] > cb)
                                  goto success_homogeneous;
                                else
                                  if(ptr[offset6] > cb)
                                    if(ptr[offset4] > cb)
                                      goto success_structured;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset3] > cb)
                                    if(ptr[offset4] > cb)
                                      goto success_structured;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset3] > cb)
                                    goto success_homogeneous;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                      else
                        goto homogeneous;
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
                                      goto success_structured;
                                    else
                                      if(ptr[offset7] > cb)
                                        goto success_structured;
                                      else
                                        goto structured;
                                  else
                                    goto homogeneous;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset2] < c_b)
                                      if(ptr[offset7] < c_b)
                                        if(ptr[offset8] < c_b)
                                          goto success_structured;
                                        else
                                          goto structured;
                                      else
                                        goto structured;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset7] > cb)
                                    if(ptr[offset8] > cb)
                                      if(ptr[offset10] > cb)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset2] < c_b)
                                      if(ptr[offset7] < c_b)
                                        if(ptr[offset1] < c_b)
                                          goto success_structured;
                                        else
                                          if(ptr[offset8] < c_b)
                                            goto success_structured;
                                          else
                                            goto structured;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                            else
                              if(ptr[offset2] < c_b)
                                if(ptr[offset7] < c_b)
                                  if(ptr[offset1] < c_b)
                                    if(ptr[offset6] < c_b)
                                      goto success_structured;
                                    else
                                      goto homogeneous;
                                  else
                                    if(ptr[offset6] < c_b)
                                      if(ptr[offset8] < c_b)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            if(ptr[offset11] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset1] > cb)
                                    if(ptr[offset2] > cb)
                                      goto success_structured;
                                    else
                                      if(ptr[offset7] > cb)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                  else
                                    if(ptr[offset6] > cb)
                                      if(ptr[offset7] > cb)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                        else
                          if(ptr[offset11] > cb)
                            if(ptr[offset10] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset1] > cb)
                                  if(ptr[offset2] > cb)
                                    goto success_homogeneous;
                                  else
                                    if(ptr[offset7] > cb)
                                      if(ptr[offset8] > cb)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  if(ptr[offset6] > cb)
                                    if(ptr[offset7] > cb)
                                      if(ptr[offset8] > cb)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset8] > cb)
                                  if(ptr[offset1] > cb)
                                    if(ptr[offset2] > cb)
                                      goto success_homogeneous;
                                    else
                                      if(ptr[offset7] > cb)
                                        goto success_homogeneous;
                                      else
                                        goto homogeneous;
                                  else
                                    if(ptr[offset6] > cb)
                                      if(ptr[offset7] > cb)
                                        goto success_homogeneous;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                      else
                        if(ptr[offset9] < c_b)
                          if(ptr[offset2] > cb)
                            if(ptr[offset1] > cb)
                              if(ptr[offset4] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset3] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_structured;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset7] < c_b)
                                      if(ptr[offset8] < c_b)
                                        if(ptr[offset11] < c_b)
                                          if(ptr[offset10] < c_b)
                                            goto success_structured;
                                          else
                                            goto structured;
                                        else
                                          goto structured;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset7] < c_b)
                                    if(ptr[offset8] < c_b)
                                      if(ptr[offset10] < c_b)
                                        if(ptr[offset4] < c_b)
                                          goto success_structured;
                                        else
                                          if(ptr[offset11] < c_b)
                                            goto success_structured;
                                          else
                                            goto structured;
                                      else
                                        if(ptr[offset3] < c_b)
                                          if(ptr[offset4] < c_b)
                                            goto success_structured;
                                          else
                                            goto structured;
                                        else
                                          goto homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset7] < c_b)
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset4] < c_b)
                                      if(ptr[offset3] < c_b)
                                        goto success_structured;
                                      else
                                        if(ptr[offset10] < c_b)
                                          goto success_structured;
                                        else
                                          goto homogeneous;
                                    else
                                      if(ptr[offset10] < c_b)
                                        if(ptr[offset11] < c_b)
                                          goto success_structured;
                                        else
                                          goto homogeneous;
                                      else
                                        goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset7] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto success_homogeneous;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto success_homogeneous;
                                      else
                                        goto homogeneous;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto success_homogeneous;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  if(ptr[offset2] < c_b)
                                    if(ptr[offset1] < c_b)
                                      if(ptr[offset3] < c_b)
                                        if(ptr[offset4] < c_b)
                                          goto success_structured;
                                        else
                                          goto homogeneous;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                        else
                          if(ptr[offset2] > cb)
                            if(ptr[offset1] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                          else
                            if(ptr[offset2] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset7] < c_b)
                                    if(ptr[offset1] < c_b)
                                      if(ptr[offset6] < c_b)
                                        goto success_homogeneous;
                                      else
                                        goto homogeneous;
                                    else
                                      if(ptr[offset6] < c_b)
                                        if(ptr[offset8] < c_b)
                                          goto success_homogeneous;
                                        else
                                          goto homogeneous;
                                      else
                                        goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                    else
                      if(ptr[offset2] > cb)
                        if(ptr[offset10] > cb)
                          if(ptr[offset11] > cb)
                            if(ptr[offset9] > cb)
                              if(ptr[offset1] > cb)
                                if(ptr[offset3] > cb)
                                  goto success_homogeneous;
                                else
                                  if(ptr[offset8] > cb)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset7] > cb)
                                    if(ptr[offset8] > cb)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset1] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                      else
                        if(ptr[offset9] > cb)
                          if(ptr[offset7] > cb)
                            if(ptr[offset8] > cb)
                              if(ptr[offset10] > cb)
                                if(ptr[offset11] > cb)
                                  if(ptr[offset1] > cb)
                                    goto success_homogeneous;
                                  else
                                    if(ptr[offset6] > cb)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                else if(ptr[offset0] < c_b)
                  if(ptr[offset2] > cb)
                    if(ptr[offset5] > cb)
                      if(ptr[offset7] > cb)
                        if(ptr[offset6] > cb)
                          if(ptr[offset4] > cb)
                            if(ptr[offset3] > cb)
                              if(ptr[offset1] > cb)
                                goto success_homogeneous;
                              else
                                if(ptr[offset8] > cb)
                                  goto success_homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset9] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    goto success_structured;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            if(ptr[offset9] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto success_structured;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                        else
                          goto homogeneous;
                      else
                        if(ptr[offset9] < c_b)
                          if(ptr[offset8] < c_b)
                            if(ptr[offset10] < c_b)
                              if(ptr[offset11] < c_b)
                                if(ptr[offset7] < c_b)
                                  if(ptr[offset1] < c_b)
                                    goto success_structured;
                                  else
                                    if(ptr[offset6] < c_b)
                                      goto success_structured;
                                    else
                                      goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                    else
                      if(ptr[offset9] < c_b)
                        if(ptr[offset7] < c_b)
                          if(ptr[offset8] < c_b)
                            if(ptr[offset5] < c_b)
                              if(ptr[offset1] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto success_structured;
                                  else
                                    if(ptr[offset6] < c_b)
                                      if(ptr[offset4] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto homogeneous;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset3] < c_b)
                                      if(ptr[offset4] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto success_structured;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset10] < c_b)
                                if(ptr[offset11] < c_b)
                                  if(ptr[offset1] < c_b)
                                    goto success_homogeneous;
                                  else
                                    if(ptr[offset6] < c_b)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                      else
                        goto homogeneous;
                  else
                    if(ptr[offset2] < c_b)
                      if(ptr[offset9] > cb)
                        if(ptr[offset5] > cb)
                          if(ptr[offset1] < c_b)
                            if(ptr[offset4] < c_b)
                              if(ptr[offset10] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto success_structured;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset7] > cb)
                                    if(ptr[offset8] > cb)
                                      if(ptr[offset11] > cb)
                                        if(ptr[offset10] > cb)
                                          goto success_structured;
                                        else
                                          goto structured;
                                      else
                                        goto structured;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset7] > cb)
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset4] > cb)
                                        goto success_structured;
                                      else
                                        if(ptr[offset11] > cb)
                                          goto success_structured;
                                        else
                                          goto structured;
                                    else
                                      if(ptr[offset3] > cb)
                                        if(ptr[offset4] > cb)
                                          goto success_structured;
                                        else
                                          goto structured;
                                      else
                                        goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset7] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset3] > cb)
                                      goto success_structured;
                                    else
                                      if(ptr[offset10] > cb)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                  else
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                        else
                          if(ptr[offset3] < c_b)
                            if(ptr[offset4] < c_b)
                              if(ptr[offset5] < c_b)
                                if(ptr[offset1] < c_b)
                                  if(ptr[offset6] < c_b)
                                    goto success_homogeneous;
                                  else
                                    if(ptr[offset11] < c_b)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset7] < c_b)
                                      if(ptr[offset8] < c_b)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset1] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                      else
                        if(ptr[offset9] < c_b)
                          if(ptr[offset5] < c_b)
                            if(ptr[offset1] < c_b)
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    goto success_homogeneous;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto success_structured;
                                      else
                                        if(ptr[offset4] < c_b)
                                          if(ptr[offset7] < c_b)
                                            goto success_structured;
                                          else
                                            goto structured;
                                        else
                                          goto homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset11] < c_b)
                                  if(ptr[offset3] < c_b)
                                    if(ptr[offset4] < c_b)
                                      goto success_homogeneous;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto success_homogeneous;
                                      else
                                        goto homogeneous;
                                  else
                                    if(ptr[offset8] < c_b)
                                      if(ptr[offset10] < c_b)
                                        goto success_homogeneous;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset7] < c_b)
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset4] < c_b)
                                      if(ptr[offset3] < c_b)
                                        goto success_structured;
                                      else
                                        if(ptr[offset10] < c_b)
                                          goto success_structured;
                                        else
                                          goto homogeneous;
                                    else
                                      if(ptr[offset10] < c_b)
                                        if(ptr[offset11] < c_b)
                                          goto success_structured;
                                        else
                                          goto homogeneous;
                                      else
                                        goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            if(ptr[offset10] < c_b)
                              if(ptr[offset11] < c_b)
                                if(ptr[offset1] < c_b)
                                  if(ptr[offset3] < c_b)
                                    goto success_homogeneous;
                                  else
                                    if(ptr[offset8] < c_b)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset7] < c_b)
                                      if(ptr[offset8] < c_b)
                                        goto success_homogeneous;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                        else
                          if(ptr[offset3] < c_b)
                            if(ptr[offset4] < c_b)
                              if(ptr[offset5] < c_b)
                                if(ptr[offset1] < c_b)
                                  if(ptr[offset6] < c_b)
                                    goto success_homogeneous;
                                  else
                                    if(ptr[offset11] < c_b)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset7] < c_b)
                                      if(ptr[offset8] < c_b)
                                        goto success_homogeneous;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset1] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                    else
                      if(ptr[offset9] < c_b)
                        if(ptr[offset7] < c_b)
                          if(ptr[offset8] < c_b)
                            if(ptr[offset5] < c_b)
                              if(ptr[offset1] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto success_homogeneous;
                                  else
                                    if(ptr[offset6] < c_b)
                                      if(ptr[offset4] < c_b)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset3] < c_b)
                                      if(ptr[offset4] < c_b)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto success_homogeneous;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto success_homogeneous;
                                      else
                                        goto homogeneous;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto success_homogeneous;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset10] < c_b)
                                if(ptr[offset11] < c_b)
                                  if(ptr[offset1] < c_b)
                                    goto success_homogeneous;
                                  else
                                    if(ptr[offset6] < c_b)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                      else
                        if(ptr[offset5] > cb)
                          if(ptr[offset9] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset7] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset3] > cb)
                                      goto success_homogeneous;
                                    else
                                      if(ptr[offset10] > cb)
                                        goto success_homogeneous;
                                      else
                                        goto homogeneous;
                                  else
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto success_homogeneous;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                else
                  if(ptr[offset5] > cb)
                    if(ptr[offset9] > cb)
                      if(ptr[offset6] > cb)
                        if(ptr[offset7] > cb)
                          if(ptr[offset4] > cb)
                            if(ptr[offset3] > cb)
                              if(ptr[offset8] > cb)
                                goto success_homogeneous;
                              else
                                if(ptr[offset1] > cb)
                                  if(ptr[offset2] > cb)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  goto success_homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            if(ptr[offset11] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  goto success_homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                        else
                          goto homogeneous;
                      else
                        goto homogeneous;
                    else
                      if(ptr[offset2] > cb)
                        if(ptr[offset3] > cb)
                          if(ptr[offset4] > cb)
                            if(ptr[offset7] > cb)
                              if(ptr[offset1] > cb)
                                if(ptr[offset6] > cb)
                                  goto success_homogeneous;
                                else
                                  goto homogeneous;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset8] > cb)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                      else
                        goto homogeneous;
                  else
                    if(ptr[offset5] < c_b)
                      if(ptr[offset9] < c_b)
                        if(ptr[offset6] < c_b)
                          if(ptr[offset7] < c_b)
                            if(ptr[offset4] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset8] < c_b)
                                  goto success_homogeneous;
                                else
                                  if(ptr[offset1] < c_b)
                                    if(ptr[offset2] < c_b)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset11] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                      else
                        if(ptr[offset2] < c_b)
                          if(ptr[offset3] < c_b)
                            if(ptr[offset4] < c_b)
                              if(ptr[offset7] < c_b)
                                if(ptr[offset1] < c_b)
                                  if(ptr[offset6] < c_b)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset8] < c_b)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                    else
                      goto homogeneous;
            }
          }
          structured:
          {
            x++;
            if(x > xsizeB)
                break;
            else
            {
                const unsigned char* const ptr = img.ptr() + y*width + x;
                const int cb = *ptr + threshold;
                const int c_b = *ptr - threshold;
                if(ptr[offset0] > cb)
                  if(ptr[offset5] > cb)
                    if(ptr[offset2] > cb)
                      if(ptr[offset9] > cb)
                        if(ptr[offset1] > cb)
                          if(ptr[offset6] > cb)
                            if(ptr[offset3] > cb)
                              if(ptr[offset4] > cb)
                                goto success_structured;
                              else
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto success_structured;
                                  else
                                    if(ptr[offset4] > cb)
                                      if(ptr[offset7] > cb)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            if(ptr[offset11] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  goto success_structured;
                                else
                                  if(ptr[offset10] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              goto structured;
                        else
                          if(ptr[offset6] > cb)
                            if(ptr[offset7] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset3] > cb)
                                    goto success_structured;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                          else
                            goto structured;
                      else
                        if(ptr[offset3] > cb)
                          if(ptr[offset4] > cb)
                            if(ptr[offset1] > cb)
                              if(ptr[offset6] > cb)
                                goto success_structured;
                              else
                                if(ptr[offset11] > cb)
                                  goto success_structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset7] > cb)
                                  if(ptr[offset8] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            goto structured;
                        else
                          goto structured;
                    else
                      if(ptr[offset9] > cb)
                        if(ptr[offset7] > cb)
                          if(ptr[offset8] > cb)
                            if(ptr[offset1] > cb)
                              if(ptr[offset10] > cb)
                                if(ptr[offset11] > cb)
                                  goto success_structured;
                                else
                                  if(ptr[offset6] > cb)
                                    if(ptr[offset4] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset3] > cb)
                                    if(ptr[offset4] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset3] > cb)
                                    goto success_structured;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                goto structured;
                          else
                            goto structured;
                        else
                          goto structured;
                      else
                        goto structured;
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
                                      goto success_structured;
                                    else
                                      if(ptr[offset7] > cb)
                                        goto success_structured;
                                      else
                                        goto structured;
                                  else
                                    goto structured;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset2] < c_b)
                                      if(ptr[offset7] < c_b)
                                        if(ptr[offset8] < c_b)
                                          goto success_structured;
                                        else
                                          goto structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset7] > cb)
                                    if(ptr[offset8] > cb)
                                      if(ptr[offset10] > cb)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset2] < c_b)
                                      if(ptr[offset7] < c_b)
                                        if(ptr[offset1] < c_b)
                                          goto success_structured;
                                        else
                                          if(ptr[offset8] < c_b)
                                            goto success_structured;
                                          else
                                            goto structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                            else
                              if(ptr[offset2] < c_b)
                                if(ptr[offset7] < c_b)
                                  if(ptr[offset1] < c_b)
                                    if(ptr[offset6] < c_b)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    if(ptr[offset6] < c_b)
                                      if(ptr[offset8] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            if(ptr[offset11] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset1] > cb)
                                    if(ptr[offset2] > cb)
                                      goto success_structured;
                                    else
                                      if(ptr[offset7] > cb)
                                        goto success_structured;
                                      else
                                        goto structured;
                                  else
                                    if(ptr[offset6] > cb)
                                      if(ptr[offset7] > cb)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                        else
                          if(ptr[offset11] > cb)
                            if(ptr[offset10] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset1] > cb)
                                  if(ptr[offset2] > cb)
                                    goto success_structured;
                                  else
                                    if(ptr[offset7] > cb)
                                      if(ptr[offset8] > cb)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                else
                                  if(ptr[offset6] > cb)
                                    if(ptr[offset7] > cb)
                                      if(ptr[offset8] > cb)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset8] > cb)
                                  if(ptr[offset1] > cb)
                                    if(ptr[offset2] > cb)
                                      goto success_structured;
                                    else
                                      if(ptr[offset7] > cb)
                                        goto success_structured;
                                      else
                                        goto structured;
                                  else
                                    if(ptr[offset6] > cb)
                                      if(ptr[offset7] > cb)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                else
                                  goto structured;
                            else
                              goto structured;
                          else
                            goto structured;
                      else
                        if(ptr[offset9] < c_b)
                          if(ptr[offset2] > cb)
                            if(ptr[offset1] > cb)
                              if(ptr[offset4] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset3] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset7] < c_b)
                                      if(ptr[offset8] < c_b)
                                        if(ptr[offset11] < c_b)
                                          if(ptr[offset10] < c_b)
                                            goto success_structured;
                                          else
                                            goto structured;
                                        else
                                          goto structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset7] < c_b)
                                    if(ptr[offset8] < c_b)
                                      if(ptr[offset10] < c_b)
                                        if(ptr[offset4] < c_b)
                                          goto success_structured;
                                        else
                                          if(ptr[offset11] < c_b)
                                            goto success_structured;
                                          else
                                            goto structured;
                                      else
                                        if(ptr[offset3] < c_b)
                                          if(ptr[offset4] < c_b)
                                            goto success_structured;
                                          else
                                            goto structured;
                                        else
                                          goto structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset7] < c_b)
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset4] < c_b)
                                      if(ptr[offset3] < c_b)
                                        goto success_structured;
                                      else
                                        if(ptr[offset10] < c_b)
                                          goto success_structured;
                                        else
                                          goto structured;
                                    else
                                      if(ptr[offset10] < c_b)
                                        if(ptr[offset11] < c_b)
                                          goto success_structured;
                                        else
                                          goto structured;
                                      else
                                        goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset7] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto success_structured;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                else
                                  if(ptr[offset2] < c_b)
                                    if(ptr[offset1] < c_b)
                                      if(ptr[offset3] < c_b)
                                        if(ptr[offset4] < c_b)
                                          goto success_structured;
                                        else
                                          goto structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                        else
                          if(ptr[offset2] > cb)
                            if(ptr[offset1] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                          else
                            if(ptr[offset2] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset7] < c_b)
                                    if(ptr[offset1] < c_b)
                                      if(ptr[offset6] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      if(ptr[offset6] < c_b)
                                        if(ptr[offset8] < c_b)
                                          goto success_structured;
                                        else
                                          goto structured;
                                      else
                                        goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                            else
                              goto homogeneous;
                    else
                      if(ptr[offset2] > cb)
                        if(ptr[offset10] > cb)
                          if(ptr[offset11] > cb)
                            if(ptr[offset9] > cb)
                              if(ptr[offset1] > cb)
                                if(ptr[offset3] > cb)
                                  goto success_structured;
                                else
                                  if(ptr[offset8] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset7] > cb)
                                    if(ptr[offset8] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset1] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            goto structured;
                        else
                          goto structured;
                      else
                        if(ptr[offset9] > cb)
                          if(ptr[offset7] > cb)
                            if(ptr[offset8] > cb)
                              if(ptr[offset10] > cb)
                                if(ptr[offset11] > cb)
                                  if(ptr[offset1] > cb)
                                    goto success_structured;
                                  else
                                    if(ptr[offset6] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                          else
                            goto structured;
                        else
                          goto structured;
                else if(ptr[offset0] < c_b)
                  if(ptr[offset2] > cb)
                    if(ptr[offset5] > cb)
                      if(ptr[offset7] > cb)
                        if(ptr[offset6] > cb)
                          if(ptr[offset4] > cb)
                            if(ptr[offset3] > cb)
                              if(ptr[offset1] > cb)
                                goto success_structured;
                              else
                                if(ptr[offset8] > cb)
                                  goto success_structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset9] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            if(ptr[offset9] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                        else
                          goto structured;
                      else
                        if(ptr[offset9] < c_b)
                          if(ptr[offset8] < c_b)
                            if(ptr[offset10] < c_b)
                              if(ptr[offset11] < c_b)
                                if(ptr[offset7] < c_b)
                                  if(ptr[offset1] < c_b)
                                    goto success_structured;
                                  else
                                    if(ptr[offset6] < c_b)
                                      goto success_structured;
                                    else
                                      goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                          else
                            goto structured;
                        else
                          goto structured;
                    else
                      if(ptr[offset9] < c_b)
                        if(ptr[offset7] < c_b)
                          if(ptr[offset8] < c_b)
                            if(ptr[offset5] < c_b)
                              if(ptr[offset1] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto success_structured;
                                  else
                                    if(ptr[offset6] < c_b)
                                      if(ptr[offset4] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset3] < c_b)
                                      if(ptr[offset4] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto success_structured;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset10] < c_b)
                                if(ptr[offset11] < c_b)
                                  if(ptr[offset1] < c_b)
                                    goto success_structured;
                                  else
                                    if(ptr[offset6] < c_b)
                                      goto success_structured;
                                    else
                                      goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            goto structured;
                        else
                          goto structured;
                      else
                        goto structured;
                  else
                    if(ptr[offset2] < c_b)
                      if(ptr[offset9] > cb)
                        if(ptr[offset5] > cb)
                          if(ptr[offset1] < c_b)
                            if(ptr[offset4] < c_b)
                              if(ptr[offset10] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto success_structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset7] > cb)
                                    if(ptr[offset8] > cb)
                                      if(ptr[offset11] > cb)
                                        if(ptr[offset10] > cb)
                                          goto success_structured;
                                        else
                                          goto structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset7] > cb)
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset4] > cb)
                                        goto success_structured;
                                      else
                                        if(ptr[offset11] > cb)
                                          goto success_structured;
                                        else
                                          goto structured;
                                    else
                                      if(ptr[offset3] > cb)
                                        if(ptr[offset4] > cb)
                                          goto success_structured;
                                        else
                                          goto structured;
                                      else
                                        goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset7] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset3] > cb)
                                      goto success_structured;
                                    else
                                      if(ptr[offset10] > cb)
                                        goto success_structured;
                                      else
                                        goto structured;
                                  else
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                        else
                          if(ptr[offset3] < c_b)
                            if(ptr[offset4] < c_b)
                              if(ptr[offset5] < c_b)
                                if(ptr[offset1] < c_b)
                                  if(ptr[offset6] < c_b)
                                    goto success_structured;
                                  else
                                    if(ptr[offset11] < c_b)
                                      goto success_structured;
                                    else
                                      goto structured;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset7] < c_b)
                                      if(ptr[offset8] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset1] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              goto structured;
                          else
                            goto structured;
                      else
                        if(ptr[offset9] < c_b)
                          if(ptr[offset5] < c_b)
                            if(ptr[offset1] < c_b)
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    goto success_structured;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                else
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto success_structured;
                                      else
                                        if(ptr[offset4] < c_b)
                                          if(ptr[offset7] < c_b)
                                            goto success_structured;
                                          else
                                            goto structured;
                                        else
                                          goto structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset11] < c_b)
                                  if(ptr[offset3] < c_b)
                                    if(ptr[offset4] < c_b)
                                      goto success_structured;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                  else
                                    if(ptr[offset8] < c_b)
                                      if(ptr[offset10] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset7] < c_b)
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset4] < c_b)
                                      if(ptr[offset3] < c_b)
                                        goto success_structured;
                                      else
                                        if(ptr[offset10] < c_b)
                                          goto success_structured;
                                        else
                                          goto structured;
                                    else
                                      if(ptr[offset10] < c_b)
                                        if(ptr[offset11] < c_b)
                                          goto success_structured;
                                        else
                                          goto structured;
                                      else
                                        goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            if(ptr[offset10] < c_b)
                              if(ptr[offset11] < c_b)
                                if(ptr[offset1] < c_b)
                                  if(ptr[offset3] < c_b)
                                    goto success_structured;
                                  else
                                    if(ptr[offset8] < c_b)
                                      goto success_structured;
                                    else
                                      goto structured;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset7] < c_b)
                                      if(ptr[offset8] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                        else
                          if(ptr[offset3] < c_b)
                            if(ptr[offset4] < c_b)
                              if(ptr[offset5] < c_b)
                                if(ptr[offset1] < c_b)
                                  if(ptr[offset6] < c_b)
                                    goto success_structured;
                                  else
                                    if(ptr[offset11] < c_b)
                                      goto success_structured;
                                    else
                                      goto structured;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset7] < c_b)
                                      if(ptr[offset8] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset1] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              goto structured;
                          else
                            goto structured;
                    else
                      if(ptr[offset9] < c_b)
                        if(ptr[offset7] < c_b)
                          if(ptr[offset8] < c_b)
                            if(ptr[offset5] < c_b)
                              if(ptr[offset1] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto success_structured;
                                  else
                                    if(ptr[offset6] < c_b)
                                      if(ptr[offset4] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset3] < c_b)
                                      if(ptr[offset4] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto success_structured;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset10] < c_b)
                                if(ptr[offset11] < c_b)
                                  if(ptr[offset1] < c_b)
                                    goto success_structured;
                                  else
                                    if(ptr[offset6] < c_b)
                                      goto success_structured;
                                    else
                                      goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            goto structured;
                        else
                          goto structured;
                      else
                        if(ptr[offset5] > cb)
                          if(ptr[offset9] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset7] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset3] > cb)
                                      goto success_structured;
                                    else
                                      if(ptr[offset10] > cb)
                                        goto success_structured;
                                      else
                                        goto structured;
                                  else
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                          else
                            goto homogeneous;
                        else
                          goto structured;
                else
                  if(ptr[offset5] > cb)
                    if(ptr[offset9] > cb)
                      if(ptr[offset6] > cb)
                        if(ptr[offset7] > cb)
                          if(ptr[offset4] > cb)
                            if(ptr[offset3] > cb)
                              if(ptr[offset8] > cb)
                                goto success_structured;
                              else
                                if(ptr[offset1] > cb)
                                  if(ptr[offset2] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  goto success_structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            if(ptr[offset11] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  goto success_structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                        else
                          goto structured;
                      else
                        goto structured;
                    else
                      if(ptr[offset2] > cb)
                        if(ptr[offset3] > cb)
                          if(ptr[offset4] > cb)
                            if(ptr[offset7] > cb)
                              if(ptr[offset1] > cb)
                                if(ptr[offset6] > cb)
                                  goto success_structured;
                                else
                                  goto structured;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset8] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              goto structured;
                          else
                            goto structured;
                        else
                          goto structured;
                      else
                        goto structured;
                  else
                    if(ptr[offset5] < c_b)
                      if(ptr[offset9] < c_b)
                        if(ptr[offset6] < c_b)
                          if(ptr[offset7] < c_b)
                            if(ptr[offset4] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset8] < c_b)
                                  goto success_structured;
                                else
                                  if(ptr[offset1] < c_b)
                                    if(ptr[offset2] < c_b)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    goto success_structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset11] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    goto success_structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            goto structured;
                        else
                          goto structured;
                      else
                        if(ptr[offset2] < c_b)
                          if(ptr[offset3] < c_b)
                            if(ptr[offset4] < c_b)
                              if(ptr[offset7] < c_b)
                                if(ptr[offset1] < c_b)
                                  if(ptr[offset6] < c_b)
                                    goto success_structured;
                                  else
                                    goto structured;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset8] < c_b)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                          else
                            goto structured;
                        else
                          goto structured;
                    else
                      goto homogeneous;
            }
          }
          success_homogeneous:
            if(total == nExpectedCorners)
            {
                if(nExpectedCorners == 0)
                {
                    nExpectedCorners = 512;
                    keypoints.reserve(nExpectedCorners);
                }
                else
                {
                    nExpectedCorners *= 2;
                    keypoints.reserve(nExpectedCorners);
                }
            }
            keypoints.push_back(KeyPoint(Point2f((float)x, (float)y), 7.0f));
            total++;
            goto homogeneous;
          success_structured:
            if(total == nExpectedCorners)
            {
                if(nExpectedCorners == 0)
                {
                    nExpectedCorners = 512;
                    keypoints.reserve(nExpectedCorners);
                }
                else
                {
                    nExpectedCorners *= 2;
                    keypoints.reserve(nExpectedCorners);
                }
            }
            keypoints.push_back(KeyPoint(Point2f((float)x, (float)y), 7.0f));
            total++;
            goto structured;
        }
    }
}


static void AGAST_7_12s(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold)
{
    cv::Mat img;
    if(!_img.getMat().isContinuous())
      img = _img.getMat().clone();
    else
      img = _img.getMat();

    size_t total = 0;
    int xsize = img.cols;
    int ysize = img.rows;
    size_t nExpectedCorners = keypoints.capacity();
    int x, y;
    int xsizeB=xsize - 3; //2, +1 due to faster test x>xsizeB
    int ysizeB=ysize - 2;
    int width;

    keypoints.resize(0);

    int pixel_7_12s_[16];
    makeAgastOffsets(pixel_7_12s_, (int)img.step, AgastFeatureDetector::AGAST_7_12s);

    short offset0 = (short) pixel_7_12s_[0];
    short offset1 = (short) pixel_7_12s_[1];
    short offset2 = (short) pixel_7_12s_[2];
    short offset3 = (short) pixel_7_12s_[3];
    short offset4 = (short) pixel_7_12s_[4];
    short offset5 = (short) pixel_7_12s_[5];
    short offset6 = (short) pixel_7_12s_[6];
    short offset7 = (short) pixel_7_12s_[7];
    short offset8 = (short) pixel_7_12s_[8];
    short offset9 = (short) pixel_7_12s_[9];
    short offset10 = (short) pixel_7_12s_[10];
    short offset11 = (short) pixel_7_12s_[11];

    width = xsize;

    for(y = 2; y < ysizeB; y++)
    {
        x = 1;
        while(true)
        {
          homogeneous:
          {
            x++;
            if(x > xsizeB)
                break;
            else
            {
                const unsigned char* const ptr = img.ptr() + y*width + x;
                const int cb = *ptr + threshold;
                const int c_b = *ptr - threshold;
                if(ptr[offset0] > cb)
                  if(ptr[offset2] > cb)
                    if(ptr[offset5] > cb)
                      if(ptr[offset9] > cb)
                        if(ptr[offset7] > cb)
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  goto success_structured;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset4] > cb)
                                      goto success_structured;
                                    else
                                      if(ptr[offset11] > cb)
                                        goto success_structured;
                                      else
                                        goto structured;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset11] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    goto success_structured;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto success_structured;
                                    else
                                      goto homogeneous;
                                else
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset10] > cb)
                                      goto success_structured;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset3] > cb)
                                    goto success_structured;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto success_structured;
                                    else
                                      goto homogeneous;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_structured;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset11] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  goto success_homogeneous;
                                else
                                  if(ptr[offset10] > cb)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            goto homogeneous;
                      else
                        if(ptr[offset3] > cb)
                          if(ptr[offset4] > cb)
                            if(ptr[offset7] > cb)
                              if(ptr[offset1] > cb)
                                if(ptr[offset6] > cb)
                                  goto success_homogeneous;
                                else
                                  if(ptr[offset11] > cb)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset8] > cb)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset1] > cb)
                                if(ptr[offset6] > cb)
                                  goto success_homogeneous;
                                else
                                  if(ptr[offset11] > cb)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                    else
                      if(ptr[offset9] < c_b)
                        if(ptr[offset7] < c_b)
                          if(ptr[offset5] < c_b)
                            if(ptr[offset1] > cb)
                              if(ptr[offset4] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset3] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto homogeneous;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset8] < c_b)
                                      if(ptr[offset11] < c_b)
                                        if(ptr[offset10] < c_b)
                                          goto success_structured;
                                        else
                                          goto structured;
                                      else
                                        goto structured;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset4] < c_b)
                                        goto success_structured;
                                      else
                                        if(ptr[offset11] < c_b)
                                          goto success_structured;
                                        else
                                          goto structured;
                                    else
                                      if(ptr[offset3] < c_b)
                                        if(ptr[offset4] < c_b)
                                          goto success_structured;
                                        else
                                          goto structured;
                                      else
                                        goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto success_structured;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            if(ptr[offset1] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_structured;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset3] > cb)
                              if(ptr[offset4] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                      else
                        if(ptr[offset10] > cb)
                          if(ptr[offset11] > cb)
                            if(ptr[offset9] > cb)
                              if(ptr[offset7] > cb)
                                if(ptr[offset1] > cb)
                                  if(ptr[offset3] > cb)
                                    goto success_homogeneous;
                                  else
                                    if(ptr[offset8] > cb)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  if(ptr[offset6] > cb)
                                    if(ptr[offset8] > cb)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset1] > cb)
                                  if(ptr[offset3] > cb)
                                    goto success_homogeneous;
                                  else
                                    if(ptr[offset8] > cb)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset1] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                  else
                    if(ptr[offset7] > cb)
                      if(ptr[offset9] > cb)
                        if(ptr[offset8] > cb)
                          if(ptr[offset5] > cb)
                            if(ptr[offset1] > cb)
                              if(ptr[offset10] > cb)
                                if(ptr[offset11] > cb)
                                  goto success_homogeneous;
                                else
                                  if(ptr[offset6] > cb)
                                    if(ptr[offset4] > cb)
                                      goto success_structured;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset3] > cb)
                                    if(ptr[offset4] > cb)
                                      goto success_structured;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset3] > cb)
                                    goto success_homogeneous;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            if(ptr[offset10] > cb)
                              if(ptr[offset11] > cb)
                                if(ptr[offset1] > cb)
                                  goto success_homogeneous;
                                else
                                  if(ptr[offset6] > cb)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                        else
                          goto homogeneous;
                      else
                        goto homogeneous;
                    else
                      if(ptr[offset7] < c_b)
                        if(ptr[offset5] < c_b)
                          if(ptr[offset2] < c_b)
                            if(ptr[offset6] < c_b)
                              if(ptr[offset4] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset1] < c_b)
                                    goto success_homogeneous;
                                  else
                                    if(ptr[offset8] < c_b)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  if(ptr[offset9] < c_b)
                                    if(ptr[offset8] < c_b)
                                      if(ptr[offset10] < c_b)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset9] < c_b)
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              goto homogeneous;
                          else
                            if(ptr[offset9] < c_b)
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto success_homogeneous;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto success_homogeneous;
                                      else
                                        goto homogeneous;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto success_homogeneous;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                        else
                          goto homogeneous;
                      else
                        goto homogeneous;
                else if(ptr[offset0] < c_b)
                  if(ptr[offset2] < c_b)
                    if(ptr[offset9] < c_b)
                      if(ptr[offset5] < c_b)
                        if(ptr[offset7] < c_b)
                          if(ptr[offset1] < c_b)
                            if(ptr[offset6] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  goto success_structured;
                                else
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset4] < c_b)
                                      goto success_structured;
                                    else
                                      if(ptr[offset11] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset11] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    goto success_structured;
                                  else
                                    if(ptr[offset10] < c_b)
                                      goto success_structured;
                                    else
                                      goto homogeneous;
                                else
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset10] < c_b)
                                      goto success_structured;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            if(ptr[offset6] < c_b)
                              if(ptr[offset8] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset3] < c_b)
                                    goto success_structured;
                                  else
                                    if(ptr[offset10] < c_b)
                                      goto success_structured;
                                    else
                                      goto homogeneous;
                                else
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto success_structured;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                        else
                          if(ptr[offset1] < c_b)
                            if(ptr[offset11] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  goto success_homogeneous;
                                else
                                  if(ptr[offset10] < c_b)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset4] < c_b)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            goto homogeneous;
                      else
                        if(ptr[offset10] < c_b)
                          if(ptr[offset11] < c_b)
                            if(ptr[offset7] < c_b)
                              if(ptr[offset1] < c_b)
                                if(ptr[offset3] < c_b)
                                  goto success_homogeneous;
                                else
                                  if(ptr[offset8] < c_b)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset8] < c_b)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset1] < c_b)
                                if(ptr[offset3] < c_b)
                                  goto success_homogeneous;
                                else
                                  if(ptr[offset8] < c_b)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                    else
                      if(ptr[offset9] > cb)
                        if(ptr[offset5] > cb)
                          if(ptr[offset7] > cb)
                            if(ptr[offset1] < c_b)
                              if(ptr[offset4] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset3] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto homogeneous;
                                else
                                  if(ptr[offset6] > cb)
                                    if(ptr[offset8] > cb)
                                      if(ptr[offset11] > cb)
                                        if(ptr[offset10] > cb)
                                          goto success_structured;
                                        else
                                          goto structured;
                                      else
                                        goto structured;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset4] > cb)
                                        goto success_structured;
                                      else
                                        if(ptr[offset11] > cb)
                                          goto success_structured;
                                        else
                                          goto structured;
                                    else
                                      if(ptr[offset3] > cb)
                                        if(ptr[offset4] > cb)
                                          goto success_structured;
                                        else
                                          goto structured;
                                      else
                                        goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset3] > cb)
                                      goto success_structured;
                                    else
                                      if(ptr[offset10] > cb)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                  else
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            if(ptr[offset1] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto success_structured;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                        else
                          if(ptr[offset3] < c_b)
                            if(ptr[offset4] < c_b)
                              if(ptr[offset5] < c_b)
                                if(ptr[offset7] < c_b)
                                  if(ptr[offset1] < c_b)
                                    if(ptr[offset6] < c_b)
                                      goto success_structured;
                                    else
                                      if(ptr[offset11] < c_b)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                  else
                                    if(ptr[offset6] < c_b)
                                      if(ptr[offset8] < c_b)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  if(ptr[offset1] < c_b)
                                    if(ptr[offset6] < c_b)
                                      goto success_homogeneous;
                                    else
                                      if(ptr[offset11] < c_b)
                                        goto success_homogeneous;
                                      else
                                        goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset1] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                      else
                        if(ptr[offset3] < c_b)
                          if(ptr[offset4] < c_b)
                            if(ptr[offset5] < c_b)
                              if(ptr[offset7] < c_b)
                                if(ptr[offset1] < c_b)
                                  if(ptr[offset6] < c_b)
                                    goto success_homogeneous;
                                  else
                                    if(ptr[offset11] < c_b)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset8] < c_b)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset1] < c_b)
                                  if(ptr[offset6] < c_b)
                                    goto success_homogeneous;
                                  else
                                    if(ptr[offset11] < c_b)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset1] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                  else
                    if(ptr[offset7] > cb)
                      if(ptr[offset5] > cb)
                        if(ptr[offset2] > cb)
                          if(ptr[offset6] > cb)
                            if(ptr[offset4] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset1] > cb)
                                  goto success_homogeneous;
                                else
                                  if(ptr[offset8] > cb)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset9] > cb)
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset10] > cb)
                                      goto success_structured;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset9] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_structured;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          if(ptr[offset9] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset3] > cb)
                                    goto success_homogeneous;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                      else
                        goto homogeneous;
                    else
                      if(ptr[offset7] < c_b)
                        if(ptr[offset9] < c_b)
                          if(ptr[offset8] < c_b)
                            if(ptr[offset5] < c_b)
                              if(ptr[offset1] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto success_homogeneous;
                                  else
                                    if(ptr[offset6] < c_b)
                                      if(ptr[offset4] < c_b)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset3] < c_b)
                                      if(ptr[offset4] < c_b)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto success_homogeneous;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto success_homogeneous;
                                      else
                                        goto homogeneous;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto success_homogeneous;
                                      else
                                        goto homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset10] < c_b)
                                if(ptr[offset11] < c_b)
                                  if(ptr[offset1] < c_b)
                                    goto success_homogeneous;
                                  else
                                    if(ptr[offset6] < c_b)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                      else
                        goto homogeneous;
                else
                  if(ptr[offset5] > cb)
                    if(ptr[offset7] > cb)
                      if(ptr[offset9] > cb)
                        if(ptr[offset6] > cb)
                          if(ptr[offset4] > cb)
                            if(ptr[offset3] > cb)
                              if(ptr[offset8] > cb)
                                goto success_homogeneous;
                              else
                                if(ptr[offset1] > cb)
                                  if(ptr[offset2] > cb)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  goto success_homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            if(ptr[offset11] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  goto success_homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                        else
                          goto homogeneous;
                      else
                        if(ptr[offset2] > cb)
                          if(ptr[offset3] > cb)
                            if(ptr[offset4] > cb)
                              if(ptr[offset1] > cb)
                                if(ptr[offset6] > cb)
                                  goto success_homogeneous;
                                else
                                  goto homogeneous;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset8] > cb)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          goto homogeneous;
                    else
                      goto homogeneous;
                  else
                    if(ptr[offset5] < c_b)
                      if(ptr[offset7] < c_b)
                        if(ptr[offset9] < c_b)
                          if(ptr[offset6] < c_b)
                            if(ptr[offset4] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset8] < c_b)
                                  goto success_homogeneous;
                                else
                                  if(ptr[offset1] < c_b)
                                    if(ptr[offset2] < c_b)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                            else
                              if(ptr[offset11] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  goto homogeneous;
                              else
                                goto homogeneous;
                          else
                            goto homogeneous;
                        else
                          if(ptr[offset2] < c_b)
                            if(ptr[offset3] < c_b)
                              if(ptr[offset4] < c_b)
                                if(ptr[offset1] < c_b)
                                  if(ptr[offset6] < c_b)
                                    goto success_homogeneous;
                                  else
                                    goto homogeneous;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset8] < c_b)
                                      goto success_homogeneous;
                                    else
                                      goto homogeneous;
                                  else
                                    goto homogeneous;
                              else
                                goto homogeneous;
                            else
                              goto homogeneous;
                          else
                            goto homogeneous;
                      else
                        goto homogeneous;
                    else
                      goto homogeneous;
            }
          }
          structured:
          {
            x++;
            if(x > xsizeB)
                break;
            else
            {
                const unsigned char* const ptr = img.ptr() + y*width + x;
                const int cb = *ptr + threshold;
                const int c_b = *ptr - threshold;
                if(ptr[offset0] > cb)
                  if(ptr[offset2] > cb)
                    if(ptr[offset5] > cb)
                      if(ptr[offset9] > cb)
                        if(ptr[offset7] > cb)
                          if(ptr[offset1] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  goto success_structured;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset4] > cb)
                                      goto success_structured;
                                    else
                                      if(ptr[offset11] > cb)
                                        goto success_structured;
                                      else
                                        goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset11] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    goto success_structured;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                else
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset10] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                goto structured;
                          else
                            if(ptr[offset6] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset3] > cb)
                                    goto success_structured;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                        else
                          if(ptr[offset1] > cb)
                            if(ptr[offset11] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  goto success_structured;
                                else
                                  if(ptr[offset10] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            goto structured;
                      else
                        if(ptr[offset3] > cb)
                          if(ptr[offset4] > cb)
                            if(ptr[offset7] > cb)
                              if(ptr[offset1] > cb)
                                if(ptr[offset6] > cb)
                                  goto success_structured;
                                else
                                  if(ptr[offset11] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset8] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset1] > cb)
                                if(ptr[offset6] > cb)
                                  goto success_structured;
                                else
                                  if(ptr[offset11] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                              else
                                goto structured;
                          else
                            goto structured;
                        else
                          goto structured;
                    else
                      if(ptr[offset7] < c_b)
                        if(ptr[offset9] < c_b)
                          if(ptr[offset5] < c_b)
                            if(ptr[offset1] > cb)
                              if(ptr[offset4] > cb)
                                if(ptr[offset10] > cb)
                                  if(ptr[offset3] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset8] < c_b)
                                      if(ptr[offset11] < c_b)
                                        if(ptr[offset10] < c_b)
                                          goto success_structured;
                                        else
                                          goto structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset4] < c_b)
                                        goto success_structured;
                                      else
                                        if(ptr[offset11] < c_b)
                                          goto success_structured;
                                        else
                                          goto structured;
                                    else
                                      if(ptr[offset3] < c_b)
                                        if(ptr[offset4] < c_b)
                                          goto success_structured;
                                        else
                                          goto structured;
                                      else
                                        goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto success_structured;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            if(ptr[offset1] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                        else
                          if(ptr[offset10] > cb)
                            if(ptr[offset11] > cb)
                              if(ptr[offset9] > cb)
                                if(ptr[offset1] > cb)
                                  if(ptr[offset3] > cb)
                                    goto success_structured;
                                  else
                                    if(ptr[offset8] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                else
                                  goto structured;
                              else
                                if(ptr[offset1] > cb)
                                  if(ptr[offset3] > cb)
                                    if(ptr[offset4] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              goto structured;
                          else
                            goto structured;
                      else
                        if(ptr[offset10] > cb)
                          if(ptr[offset11] > cb)
                            if(ptr[offset9] > cb)
                              if(ptr[offset1] > cb)
                                if(ptr[offset3] > cb)
                                  goto success_structured;
                                else
                                  if(ptr[offset8] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset7] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset1] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset4] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            goto structured;
                        else
                          goto structured;
                  else
                    if(ptr[offset7] > cb)
                      if(ptr[offset9] > cb)
                        if(ptr[offset8] > cb)
                          if(ptr[offset5] > cb)
                            if(ptr[offset1] > cb)
                              if(ptr[offset10] > cb)
                                if(ptr[offset11] > cb)
                                  goto success_structured;
                                else
                                  if(ptr[offset6] > cb)
                                    if(ptr[offset4] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset3] > cb)
                                    if(ptr[offset4] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset6] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset3] > cb)
                                    goto success_structured;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                goto structured;
                          else
                            if(ptr[offset10] > cb)
                              if(ptr[offset11] > cb)
                                if(ptr[offset1] > cb)
                                  goto success_structured;
                                else
                                  if(ptr[offset6] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                        else
                          goto structured;
                      else
                        goto structured;
                    else
                      if(ptr[offset7] < c_b)
                        if(ptr[offset5] < c_b)
                          if(ptr[offset2] < c_b)
                            if(ptr[offset6] < c_b)
                              if(ptr[offset4] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset1] < c_b)
                                    goto success_structured;
                                  else
                                    if(ptr[offset8] < c_b)
                                      goto success_structured;
                                    else
                                      goto structured;
                                else
                                  if(ptr[offset9] < c_b)
                                    if(ptr[offset8] < c_b)
                                      if(ptr[offset10] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset9] < c_b)
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              goto structured;
                          else
                            if(ptr[offset9] < c_b)
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto success_structured;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                        else
                          goto structured;
                      else
                        goto structured;
                else if(ptr[offset0] < c_b)
                  if(ptr[offset2] < c_b)
                    if(ptr[offset11] < c_b)
                      if(ptr[offset3] < c_b)
                        if(ptr[offset5] < c_b)
                          if(ptr[offset9] < c_b)
                            if(ptr[offset7] < c_b)
                              if(ptr[offset1] < c_b)
                                if(ptr[offset4] < c_b)
                                  goto success_structured;
                                else
                                  if(ptr[offset10] < c_b)
                                    goto success_structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset4] < c_b)
                                      goto success_structured;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset1] < c_b)
                                if(ptr[offset4] < c_b)
                                  goto success_structured;
                                else
                                  if(ptr[offset10] < c_b)
                                    goto success_structured;
                                  else
                                    goto structured;
                              else
                                goto structured;
                          else
                            if(ptr[offset4] < c_b)
                              if(ptr[offset7] < c_b)
                                if(ptr[offset1] < c_b)
                                  goto success_structured;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset8] < c_b)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset1] < c_b)
                                  goto success_structured;
                                else
                                  goto structured;
                            else
                              goto structured;
                        else
                          if(ptr[offset10] < c_b)
                            if(ptr[offset9] < c_b)
                              if(ptr[offset7] < c_b)
                                if(ptr[offset1] < c_b)
                                  goto success_structured;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset8] < c_b)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset1] < c_b)
                                  goto success_structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset1] < c_b)
                                if(ptr[offset4] < c_b)
                                  goto success_structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            if(ptr[offset7] > cb)
                              if(ptr[offset9] > cb)
                                if(ptr[offset5] > cb)
                                  if(ptr[offset4] > cb)
                                    if(ptr[offset6] > cb)
                                      if(ptr[offset8] > cb)
                                        if(ptr[offset10] > cb)
                                          goto success_structured;
                                        else
                                          goto structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                      else
                        if(ptr[offset9] < c_b)
                          if(ptr[offset8] < c_b)
                            if(ptr[offset10] < c_b)
                              if(ptr[offset7] < c_b)
                                if(ptr[offset1] < c_b)
                                  goto success_structured;
                                else
                                  if(ptr[offset6] < c_b)
                                    goto success_structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset1] < c_b)
                                  goto success_structured;
                                else
                                  goto structured;
                            else
                              goto structured;
                          else
                            goto structured;
                        else
                          if(ptr[offset5] > cb)
                            if(ptr[offset7] > cb)
                              if(ptr[offset9] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset6] > cb)
                                    if(ptr[offset8] > cb)
                                      if(ptr[offset3] > cb)
                                        goto success_structured;
                                      else
                                        if(ptr[offset10] > cb)
                                          goto success_structured;
                                        else
                                          goto structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                          else
                            goto structured;
                    else
                      if(ptr[offset4] < c_b)
                        if(ptr[offset5] < c_b)
                          if(ptr[offset7] < c_b)
                            if(ptr[offset6] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset1] < c_b)
                                  goto success_structured;
                                else
                                  if(ptr[offset8] < c_b)
                                    goto success_structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset9] < c_b)
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset10] < c_b)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              goto structured;
                          else
                            if(ptr[offset1] < c_b)
                              if(ptr[offset6] < c_b)
                                if(ptr[offset3] < c_b)
                                  goto success_structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                        else
                          if(ptr[offset7] > cb)
                            if(ptr[offset9] > cb)
                              if(ptr[offset5] > cb)
                                if(ptr[offset6] > cb)
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                          else
                            goto structured;
                      else
                        if(ptr[offset5] > cb)
                          if(ptr[offset7] > cb)
                            if(ptr[offset9] > cb)
                              if(ptr[offset6] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset4] > cb)
                                      goto success_structured;
                                    else
                                      if(ptr[offset11] > cb)
                                        goto success_structured;
                                      else
                                        goto homogeneous;
                                  else
                                    if(ptr[offset3] > cb)
                                      if(ptr[offset4] > cb)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                          else
                            goto structured;
                        else
                          goto structured;
                  else
                    if(ptr[offset7] > cb)
                      if(ptr[offset5] > cb)
                        if(ptr[offset2] > cb)
                          if(ptr[offset6] > cb)
                            if(ptr[offset4] > cb)
                              if(ptr[offset3] > cb)
                                if(ptr[offset1] > cb)
                                  goto success_structured;
                                else
                                  if(ptr[offset8] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset9] > cb)
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset10] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset9] > cb)
                                if(ptr[offset8] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            goto structured;
                        else
                          if(ptr[offset9] > cb)
                            if(ptr[offset6] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset3] > cb)
                                    goto success_structured;
                                  else
                                    if(ptr[offset10] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                          else
                            goto structured;
                      else
                        goto structured;
                    else
                      if(ptr[offset7] < c_b)
                        if(ptr[offset9] < c_b)
                          if(ptr[offset8] < c_b)
                            if(ptr[offset5] < c_b)
                              if(ptr[offset1] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    goto success_structured;
                                  else
                                    if(ptr[offset6] < c_b)
                                      if(ptr[offset4] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset3] < c_b)
                                      if(ptr[offset4] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      goto success_structured;
                                    else
                                      if(ptr[offset10] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        goto success_structured;
                                      else
                                        goto structured;
                                    else
                                      goto structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset10] < c_b)
                                if(ptr[offset11] < c_b)
                                  if(ptr[offset1] < c_b)
                                    goto success_structured;
                                  else
                                    if(ptr[offset6] < c_b)
                                      goto success_structured;
                                    else
                                      goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            goto structured;
                        else
                          goto structured;
                      else
                        goto structured;
                else
                  if(ptr[offset5] > cb)
                    if(ptr[offset7] > cb)
                      if(ptr[offset9] > cb)
                        if(ptr[offset6] > cb)
                          if(ptr[offset4] > cb)
                            if(ptr[offset3] > cb)
                              if(ptr[offset8] > cb)
                                goto success_structured;
                              else
                                if(ptr[offset1] > cb)
                                  if(ptr[offset2] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  goto success_structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            if(ptr[offset11] > cb)
                              if(ptr[offset8] > cb)
                                if(ptr[offset10] > cb)
                                  goto success_structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                        else
                          goto structured;
                      else
                        if(ptr[offset2] > cb)
                          if(ptr[offset3] > cb)
                            if(ptr[offset4] > cb)
                              if(ptr[offset1] > cb)
                                if(ptr[offset6] > cb)
                                  goto success_structured;
                                else
                                  goto structured;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset8] > cb)
                                    goto success_structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              goto structured;
                          else
                            goto structured;
                        else
                          goto structured;
                    else
                      goto structured;
                  else
                    if(ptr[offset5] < c_b)
                      if(ptr[offset7] < c_b)
                        if(ptr[offset9] < c_b)
                          if(ptr[offset6] < c_b)
                            if(ptr[offset4] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset8] < c_b)
                                  goto success_structured;
                                else
                                  if(ptr[offset1] < c_b)
                                    if(ptr[offset2] < c_b)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    goto success_structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                            else
                              if(ptr[offset11] < c_b)
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset10] < c_b)
                                    goto success_structured;
                                  else
                                    goto structured;
                                else
                                  goto structured;
                              else
                                goto structured;
                          else
                            goto structured;
                        else
                          if(ptr[offset2] < c_b)
                            if(ptr[offset3] < c_b)
                              if(ptr[offset4] < c_b)
                                if(ptr[offset1] < c_b)
                                  if(ptr[offset6] < c_b)
                                    goto success_structured;
                                  else
                                    goto structured;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset8] < c_b)
                                      goto success_structured;
                                    else
                                      goto structured;
                                  else
                                    goto structured;
                              else
                                goto structured;
                            else
                              goto structured;
                          else
                            goto structured;
                      else
                        goto structured;
                    else
                      goto homogeneous;
            }
          }
          success_homogeneous:
            if(total == nExpectedCorners)
            {
                if(nExpectedCorners == 0)
                {
                    nExpectedCorners = 512;
                    keypoints.reserve(nExpectedCorners);
                }
                else
                {
                    nExpectedCorners *= 2;
                    keypoints.reserve(nExpectedCorners);
                }
            }
            keypoints.push_back(KeyPoint(Point2f((float)x, (float)y), 7.0f));
            total++;
            goto homogeneous;
          success_structured:
            if(total == nExpectedCorners)
            {
                if(nExpectedCorners == 0)
                {
                    nExpectedCorners = 512;
                    keypoints.reserve(nExpectedCorners);
                }
                else
                {
                    nExpectedCorners *= 2;
                    keypoints.reserve(nExpectedCorners);
                }
            }
            keypoints.push_back(KeyPoint(Point2f((float)x, (float)y), 7.0f));
            total++;
            goto structured;
        }
    }
}

static void OAST_9_16(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold)
{
    cv::Mat img;
    if(!_img.getMat().isContinuous())
      img = _img.getMat().clone();
    else
      img = _img.getMat();

    size_t total = 0;
    int xsize = img.cols;
    int ysize = img.rows;
    size_t nExpectedCorners = keypoints.capacity();
    int x, y;
    int xsizeB=xsize - 4;
    int ysizeB=ysize - 3;
    int width;

    keypoints.resize(0);

    int pixel_9_16_[16];
    makeAgastOffsets(pixel_9_16_, (int)img.step, AgastFeatureDetector::OAST_9_16);

    short offset0 = (short) pixel_9_16_[0];
    short offset1 = (short) pixel_9_16_[1];
    short offset2 = (short) pixel_9_16_[2];
    short offset3 = (short) pixel_9_16_[3];
    short offset4 = (short) pixel_9_16_[4];
    short offset5 = (short) pixel_9_16_[5];
    short offset6 = (short) pixel_9_16_[6];
    short offset7 = (short) pixel_9_16_[7];
    short offset8 = (short) pixel_9_16_[8];
    short offset9 = (short) pixel_9_16_[9];
    short offset10 = (short) pixel_9_16_[10];
    short offset11 = (short) pixel_9_16_[11];
    short offset12 = (short) pixel_9_16_[12];
    short offset13 = (short) pixel_9_16_[13];
    short offset14 = (short) pixel_9_16_[14];
    short offset15 = (short) pixel_9_16_[15];

    width = xsize;

    for(y = 3; y < ysizeB; y++)
    {
        x = 2;
        while(true)
        {
            x++;
            if(x > xsizeB)
                break;
            else
            {
                const unsigned char* const ptr = img.ptr() + y*width + x;
                const int cb = *ptr + threshold;
                const int c_b = *ptr - threshold;
                if(ptr[offset0] > cb)
                  if(ptr[offset2] > cb)
                    if(ptr[offset4] > cb)
                      if(ptr[offset5] > cb)
                        if(ptr[offset7] > cb)
                          if(ptr[offset3] > cb)
                            if(ptr[offset1] > cb)
                              if(ptr[offset6] > cb)
                                if(ptr[offset8] > cb)
                                  {}
                                else
                                  if(ptr[offset15] > cb)
                                    {}
                                  else
                                    continue;
                              else
                                if(ptr[offset13] > cb)
                                  if(ptr[offset14] > cb)
                                    if(ptr[offset15] > cb)
                                      {}
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              if(ptr[offset8] > cb)
                                if(ptr[offset9] > cb)
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset6] > cb)
                                      {}
                                    else
                                      if(ptr[offset11] > cb)
                                        if(ptr[offset12] > cb)
                                          if(ptr[offset13] > cb)
                                            if(ptr[offset14] > cb)
                                              if(ptr[offset15] > cb)
                                                {}
                                              else
                                                continue;
                                            else
                                              continue;
                                          else
                                            continue;
                                        else
                                          continue;
                                      else
                                        continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                          else
                            if(ptr[offset10] > cb)
                              if(ptr[offset11] > cb)
                                if(ptr[offset12] > cb)
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset9] > cb)
                                      if(ptr[offset6] > cb)
                                        {}
                                      else
                                        if(ptr[offset13] > cb)
                                          if(ptr[offset14] > cb)
                                            if(ptr[offset15] > cb)
                                              {}
                                            else
                                              continue;
                                          else
                                            continue;
                                        else
                                          continue;
                                    else
                                      if(ptr[offset1] > cb)
                                        if(ptr[offset13] > cb)
                                          if(ptr[offset14] > cb)
                                            if(ptr[offset15] > cb)
                                              {}
                                            else
                                              continue;
                                          else
                                            continue;
                                        else
                                          continue;
                                      else
                                        continue;
                                  else
                                    if(ptr[offset1] > cb)
                                      if(ptr[offset13] > cb)
                                        if(ptr[offset14] > cb)
                                          if(ptr[offset15] > cb)
                                            {}
                                          else
                                            continue;
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                else
                                  continue;
                              else
                                continue;
                            else
                              continue;
                        else if(ptr[offset7] < c_b)
                          if(ptr[offset14] > cb)
                            if(ptr[offset15] > cb)
                              if(ptr[offset1] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset6] > cb)
                                    {}
                                  else
                                    if(ptr[offset13] > cb)
                                      {}
                                    else
                                      continue;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      if(ptr[offset12] > cb)
                                        if(ptr[offset13] > cb)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset8] > cb)
                                  if(ptr[offset9] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        if(ptr[offset12] > cb)
                                          if(ptr[offset13] > cb)
                                            {}
                                          else
                                            continue;
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              continue;
                          else if(ptr[offset14] < c_b)
                            if(ptr[offset8] < c_b)
                              if(ptr[offset9] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    if(ptr[offset12] < c_b)
                                      if(ptr[offset13] < c_b)
                                        if(ptr[offset6] < c_b)
                                          {}
                                        else
                                          if(ptr[offset15] < c_b)
                                            {}
                                          else
                                            continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                            else
                              continue;
                          else
                            continue;
                        else
                          if(ptr[offset14] > cb)
                            if(ptr[offset15] > cb)
                              if(ptr[offset1] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset6] > cb)
                                    {}
                                  else
                                    if(ptr[offset13] > cb)
                                      {}
                                    else
                                      continue;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      if(ptr[offset12] > cb)
                                        if(ptr[offset13] > cb)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset8] > cb)
                                  if(ptr[offset9] > cb)
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        if(ptr[offset12] > cb)
                                          if(ptr[offset13] > cb)
                                            {}
                                          else
                                            continue;
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              continue;
                          else
                            continue;
                      else if(ptr[offset5] < c_b)
                        if(ptr[offset12] > cb)
                          if(ptr[offset13] > cb)
                            if(ptr[offset14] > cb)
                              if(ptr[offset15] > cb)
                                if(ptr[offset1] > cb)
                                  if(ptr[offset3] > cb)
                                    {}
                                  else
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                else
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset9] > cb)
                                      if(ptr[offset10] > cb)
                                        if(ptr[offset11] > cb)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset7] > cb)
                                    if(ptr[offset8] > cb)
                                      if(ptr[offset9] > cb)
                                        if(ptr[offset10] > cb)
                                          if(ptr[offset11] > cb)
                                            {}
                                          else
                                            continue;
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              continue;
                          else
                            continue;
                        else if(ptr[offset12] < c_b)
                          if(ptr[offset7] < c_b)
                            if(ptr[offset8] < c_b)
                              if(ptr[offset9] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    if(ptr[offset13] < c_b)
                                      if(ptr[offset6] < c_b)
                                        {}
                                      else
                                        if(ptr[offset14] < c_b)
                                          if(ptr[offset15] < c_b)
                                            {}
                                          else
                                            continue;
                                        else
                                          continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                            else
                              continue;
                          else
                            continue;
                        else
                          continue;
                      else
                        if(ptr[offset12] > cb)
                          if(ptr[offset13] > cb)
                            if(ptr[offset14] > cb)
                              if(ptr[offset15] > cb)
                                if(ptr[offset1] > cb)
                                  if(ptr[offset3] > cb)
                                    {}
                                  else
                                    if(ptr[offset10] > cb)
                                      if(ptr[offset11] > cb)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                else
                                  if(ptr[offset8] > cb)
                                    if(ptr[offset9] > cb)
                                      if(ptr[offset10] > cb)
                                        if(ptr[offset11] > cb)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset6] > cb)
                                  if(ptr[offset7] > cb)
                                    if(ptr[offset8] > cb)
                                      if(ptr[offset9] > cb)
                                        if(ptr[offset10] > cb)
                                          if(ptr[offset11] > cb)
                                            {}
                                          else
                                            continue;
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              continue;
                          else
                            continue;
                        else if(ptr[offset12] < c_b)
                          if(ptr[offset7] < c_b)
                            if(ptr[offset8] < c_b)
                              if(ptr[offset9] < c_b)
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    if(ptr[offset13] < c_b)
                                      if(ptr[offset14] < c_b)
                                        if(ptr[offset6] < c_b)
                                          {}
                                        else
                                          if(ptr[offset15] < c_b)
                                            {}
                                          else
                                            continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                            else
                              continue;
                          else
                            continue;
                        else
                          continue;
                    else if(ptr[offset4] < c_b)
                      if(ptr[offset11] > cb)
                        if(ptr[offset12] > cb)
                          if(ptr[offset13] > cb)
                            if(ptr[offset10] > cb)
                              if(ptr[offset14] > cb)
                                if(ptr[offset15] > cb)
                                  if(ptr[offset1] > cb)
                                    {}
                                  else
                                    if(ptr[offset8] > cb)
                                      if(ptr[offset9] > cb)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                else
                                  if(ptr[offset6] > cb)
                                    if(ptr[offset7] > cb)
                                      if(ptr[offset8] > cb)
                                        if(ptr[offset9] > cb)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset5] > cb)
                                  if(ptr[offset6] > cb)
                                    if(ptr[offset7] > cb)
                                      if(ptr[offset8] > cb)
                                        if(ptr[offset9] > cb)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              if(ptr[offset1] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset14] > cb)
                                    if(ptr[offset15] > cb)
                                      {}
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                          else
                            continue;
                        else
                          continue;
                      else if(ptr[offset11] < c_b)
                        if(ptr[offset7] < c_b)
                          if(ptr[offset8] < c_b)
                            if(ptr[offset9] < c_b)
                              if(ptr[offset10] < c_b)
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset5] < c_b)
                                    if(ptr[offset3] < c_b)
                                      {}
                                    else
                                      if(ptr[offset12] < c_b)
                                        {}
                                      else
                                        continue;
                                  else
                                    if(ptr[offset12] < c_b)
                                      if(ptr[offset13] < c_b)
                                        if(ptr[offset14] < c_b)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                else
                                  if(ptr[offset12] < c_b)
                                    if(ptr[offset13] < c_b)
                                      if(ptr[offset14] < c_b)
                                        if(ptr[offset15] < c_b)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                continue;
                            else
                              continue;
                          else
                            continue;
                        else
                          continue;
                      else
                        continue;
                    else
                      if(ptr[offset11] > cb)
                        if(ptr[offset12] > cb)
                          if(ptr[offset13] > cb)
                            if(ptr[offset10] > cb)
                              if(ptr[offset14] > cb)
                                if(ptr[offset15] > cb)
                                  if(ptr[offset1] > cb)
                                    {}
                                  else
                                    if(ptr[offset8] > cb)
                                      if(ptr[offset9] > cb)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                else
                                  if(ptr[offset6] > cb)
                                    if(ptr[offset7] > cb)
                                      if(ptr[offset8] > cb)
                                        if(ptr[offset9] > cb)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset5] > cb)
                                  if(ptr[offset6] > cb)
                                    if(ptr[offset7] > cb)
                                      if(ptr[offset8] > cb)
                                        if(ptr[offset9] > cb)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              if(ptr[offset1] > cb)
                                if(ptr[offset3] > cb)
                                  if(ptr[offset14] > cb)
                                    if(ptr[offset15] > cb)
                                      {}
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                          else
                            continue;
                        else
                          continue;
                      else if(ptr[offset11] < c_b)
                        if(ptr[offset7] < c_b)
                          if(ptr[offset8] < c_b)
                            if(ptr[offset9] < c_b)
                              if(ptr[offset10] < c_b)
                                if(ptr[offset12] < c_b)
                                  if(ptr[offset13] < c_b)
                                    if(ptr[offset6] < c_b)
                                      if(ptr[offset5] < c_b)
                                        {}
                                      else
                                        if(ptr[offset14] < c_b)
                                          {}
                                        else
                                          continue;
                                    else
                                      if(ptr[offset14] < c_b)
                                        if(ptr[offset15] < c_b)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                            else
                              continue;
                          else
                            continue;
                        else
                          continue;
                      else
                        continue;
                  else if(ptr[offset2] < c_b)
                    if(ptr[offset9] > cb)
                      if(ptr[offset10] > cb)
                        if(ptr[offset11] > cb)
                          if(ptr[offset8] > cb)
                            if(ptr[offset12] > cb)
                              if(ptr[offset13] > cb)
                                if(ptr[offset14] > cb)
                                  if(ptr[offset15] > cb)
                                    {}
                                  else
                                    if(ptr[offset6] > cb)
                                      if(ptr[offset7] > cb)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                else
                                  if(ptr[offset5] > cb)
                                    if(ptr[offset6] > cb)
                                      if(ptr[offset7] > cb)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset4] > cb)
                                  if(ptr[offset5] > cb)
                                    if(ptr[offset6] > cb)
                                      if(ptr[offset7] > cb)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset5] > cb)
                                    if(ptr[offset6] > cb)
                                      if(ptr[offset7] > cb)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                          else
                            if(ptr[offset1] > cb)
                              if(ptr[offset12] > cb)
                                if(ptr[offset13] > cb)
                                  if(ptr[offset14] > cb)
                                    if(ptr[offset15] > cb)
                                      {}
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                            else
                              continue;
                        else
                          continue;
                      else
                        continue;
                    else if(ptr[offset9] < c_b)
                      if(ptr[offset7] < c_b)
                        if(ptr[offset8] < c_b)
                          if(ptr[offset6] < c_b)
                            if(ptr[offset5] < c_b)
                              if(ptr[offset4] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset1] < c_b)
                                    {}
                                  else
                                    if(ptr[offset10] < c_b)
                                      {}
                                    else
                                      continue;
                                else
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      if(ptr[offset12] < c_b)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    if(ptr[offset12] < c_b)
                                      if(ptr[offset13] < c_b)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              if(ptr[offset10] < c_b)
                                if(ptr[offset11] < c_b)
                                  if(ptr[offset12] < c_b)
                                    if(ptr[offset13] < c_b)
                                      if(ptr[offset14] < c_b)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                          else
                            if(ptr[offset10] < c_b)
                              if(ptr[offset11] < c_b)
                                if(ptr[offset12] < c_b)
                                  if(ptr[offset13] < c_b)
                                    if(ptr[offset14] < c_b)
                                      if(ptr[offset15] < c_b)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                            else
                              continue;
                        else
                          continue;
                      else
                        continue;
                    else
                      continue;
                  else
                    if(ptr[offset9] > cb)
                      if(ptr[offset10] > cb)
                        if(ptr[offset11] > cb)
                          if(ptr[offset8] > cb)
                            if(ptr[offset12] > cb)
                              if(ptr[offset13] > cb)
                                if(ptr[offset14] > cb)
                                  if(ptr[offset15] > cb)
                                    {}
                                  else
                                    if(ptr[offset6] > cb)
                                      if(ptr[offset7] > cb)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                else
                                  if(ptr[offset5] > cb)
                                    if(ptr[offset6] > cb)
                                      if(ptr[offset7] > cb)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset4] > cb)
                                  if(ptr[offset5] > cb)
                                    if(ptr[offset6] > cb)
                                      if(ptr[offset7] > cb)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              if(ptr[offset3] > cb)
                                if(ptr[offset4] > cb)
                                  if(ptr[offset5] > cb)
                                    if(ptr[offset6] > cb)
                                      if(ptr[offset7] > cb)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                          else
                            if(ptr[offset1] > cb)
                              if(ptr[offset12] > cb)
                                if(ptr[offset13] > cb)
                                  if(ptr[offset14] > cb)
                                    if(ptr[offset15] > cb)
                                      {}
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                            else
                              continue;
                        else
                          continue;
                      else
                        continue;
                    else if(ptr[offset9] < c_b)
                      if(ptr[offset7] < c_b)
                        if(ptr[offset8] < c_b)
                          if(ptr[offset10] < c_b)
                            if(ptr[offset11] < c_b)
                              if(ptr[offset6] < c_b)
                                if(ptr[offset5] < c_b)
                                  if(ptr[offset4] < c_b)
                                    if(ptr[offset3] < c_b)
                                      {}
                                    else
                                      if(ptr[offset12] < c_b)
                                        {}
                                      else
                                        continue;
                                  else
                                    if(ptr[offset12] < c_b)
                                      if(ptr[offset13] < c_b)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                else
                                  if(ptr[offset12] < c_b)
                                    if(ptr[offset13] < c_b)
                                      if(ptr[offset14] < c_b)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset12] < c_b)
                                  if(ptr[offset13] < c_b)
                                    if(ptr[offset14] < c_b)
                                      if(ptr[offset15] < c_b)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              continue;
                          else
                            continue;
                        else
                          continue;
                      else
                        continue;
                    else
                      continue;
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
                                    {}
                                  else
                                    if(ptr[offset10] > cb)
                                      {}
                                    else
                                      continue;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      if(ptr[offset12] > cb)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    if(ptr[offset12] > cb)
                                      if(ptr[offset13] > cb)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              if(ptr[offset10] > cb)
                                if(ptr[offset11] > cb)
                                  if(ptr[offset12] > cb)
                                    if(ptr[offset13] > cb)
                                      if(ptr[offset14] > cb)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                          else
                            if(ptr[offset10] > cb)
                              if(ptr[offset11] > cb)
                                if(ptr[offset12] > cb)
                                  if(ptr[offset13] > cb)
                                    if(ptr[offset14] > cb)
                                      if(ptr[offset15] > cb)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                            else
                              continue;
                        else
                          continue;
                      else
                        continue;
                    else if(ptr[offset9] < c_b)
                      if(ptr[offset10] < c_b)
                        if(ptr[offset11] < c_b)
                          if(ptr[offset8] < c_b)
                            if(ptr[offset12] < c_b)
                              if(ptr[offset13] < c_b)
                                if(ptr[offset14] < c_b)
                                  if(ptr[offset15] < c_b)
                                    {}
                                  else
                                    if(ptr[offset6] < c_b)
                                      if(ptr[offset7] < c_b)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                else
                                  if(ptr[offset5] < c_b)
                                    if(ptr[offset6] < c_b)
                                      if(ptr[offset7] < c_b)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset5] < c_b)
                                    if(ptr[offset6] < c_b)
                                      if(ptr[offset7] < c_b)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset5] < c_b)
                                    if(ptr[offset6] < c_b)
                                      if(ptr[offset7] < c_b)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                          else
                            if(ptr[offset1] < c_b)
                              if(ptr[offset12] < c_b)
                                if(ptr[offset13] < c_b)
                                  if(ptr[offset14] < c_b)
                                    if(ptr[offset15] < c_b)
                                      {}
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                            else
                              continue;
                        else
                          continue;
                      else
                        continue;
                    else
                      continue;
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
                                      {}
                                    else
                                      if(ptr[offset12] > cb)
                                        {}
                                      else
                                        continue;
                                  else
                                    if(ptr[offset12] > cb)
                                      if(ptr[offset13] > cb)
                                        if(ptr[offset14] > cb)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                else
                                  if(ptr[offset12] > cb)
                                    if(ptr[offset13] > cb)
                                      if(ptr[offset14] > cb)
                                        if(ptr[offset15] > cb)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                continue;
                            else
                              continue;
                          else
                            continue;
                        else
                          continue;
                      else if(ptr[offset11] < c_b)
                        if(ptr[offset12] < c_b)
                          if(ptr[offset13] < c_b)
                            if(ptr[offset10] < c_b)
                              if(ptr[offset14] < c_b)
                                if(ptr[offset15] < c_b)
                                  if(ptr[offset1] < c_b)
                                    {}
                                  else
                                    if(ptr[offset8] < c_b)
                                      if(ptr[offset9] < c_b)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset7] < c_b)
                                      if(ptr[offset8] < c_b)
                                        if(ptr[offset9] < c_b)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset5] < c_b)
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset7] < c_b)
                                      if(ptr[offset8] < c_b)
                                        if(ptr[offset9] < c_b)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              if(ptr[offset1] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset14] < c_b)
                                    if(ptr[offset15] < c_b)
                                      {}
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                          else
                            continue;
                        else
                          continue;
                      else
                        continue;
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
                                        {}
                                      else
                                        if(ptr[offset14] > cb)
                                          if(ptr[offset15] > cb)
                                            {}
                                          else
                                            continue;
                                        else
                                          continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                            else
                              continue;
                          else
                            continue;
                        else if(ptr[offset12] < c_b)
                          if(ptr[offset13] < c_b)
                            if(ptr[offset14] < c_b)
                              if(ptr[offset15] < c_b)
                                if(ptr[offset1] < c_b)
                                  if(ptr[offset3] < c_b)
                                    {}
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                else
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset9] < c_b)
                                      if(ptr[offset10] < c_b)
                                        if(ptr[offset11] < c_b)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset7] < c_b)
                                    if(ptr[offset8] < c_b)
                                      if(ptr[offset9] < c_b)
                                        if(ptr[offset10] < c_b)
                                          if(ptr[offset11] < c_b)
                                            {}
                                          else
                                            continue;
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              continue;
                          else
                            continue;
                        else
                          continue;
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
                                          {}
                                        else
                                          if(ptr[offset15] > cb)
                                            {}
                                          else
                                            continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                            else
                              continue;
                          else if(ptr[offset14] < c_b)
                            if(ptr[offset15] < c_b)
                              if(ptr[offset1] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset6] < c_b)
                                    {}
                                  else
                                    if(ptr[offset13] < c_b)
                                      {}
                                    else
                                      continue;
                                else
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      if(ptr[offset12] < c_b)
                                        if(ptr[offset13] < c_b)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset9] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        if(ptr[offset12] < c_b)
                                          if(ptr[offset13] < c_b)
                                            {}
                                          else
                                            continue;
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              continue;
                          else
                            continue;
                        else if(ptr[offset7] < c_b)
                          if(ptr[offset3] < c_b)
                            if(ptr[offset1] < c_b)
                              if(ptr[offset6] < c_b)
                                if(ptr[offset8] < c_b)
                                  {}
                                else
                                  if(ptr[offset15] < c_b)
                                    {}
                                  else
                                    continue;
                              else
                                if(ptr[offset13] < c_b)
                                  if(ptr[offset14] < c_b)
                                    if(ptr[offset15] < c_b)
                                      {}
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              if(ptr[offset8] < c_b)
                                if(ptr[offset9] < c_b)
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset6] < c_b)
                                      {}
                                    else
                                      if(ptr[offset11] < c_b)
                                        if(ptr[offset12] < c_b)
                                          if(ptr[offset13] < c_b)
                                            if(ptr[offset14] < c_b)
                                              if(ptr[offset15] < c_b)
                                                {}
                                              else
                                                continue;
                                            else
                                              continue;
                                          else
                                            continue;
                                        else
                                          continue;
                                      else
                                        continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                          else
                            if(ptr[offset10] < c_b)
                              if(ptr[offset11] < c_b)
                                if(ptr[offset12] < c_b)
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset9] < c_b)
                                      if(ptr[offset6] < c_b)
                                        {}
                                      else
                                        if(ptr[offset13] < c_b)
                                          if(ptr[offset14] < c_b)
                                            if(ptr[offset15] < c_b)
                                              {}
                                            else
                                              continue;
                                          else
                                            continue;
                                        else
                                          continue;
                                    else
                                      if(ptr[offset1] < c_b)
                                        if(ptr[offset13] < c_b)
                                          if(ptr[offset14] < c_b)
                                            if(ptr[offset15] < c_b)
                                              {}
                                            else
                                              continue;
                                          else
                                            continue;
                                        else
                                          continue;
                                      else
                                        continue;
                                  else
                                    if(ptr[offset1] < c_b)
                                      if(ptr[offset13] < c_b)
                                        if(ptr[offset14] < c_b)
                                          if(ptr[offset15] < c_b)
                                            {}
                                          else
                                            continue;
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                else
                                  continue;
                              else
                                continue;
                            else
                              continue;
                        else
                          if(ptr[offset14] < c_b)
                            if(ptr[offset15] < c_b)
                              if(ptr[offset1] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset6] < c_b)
                                    {}
                                  else
                                    if(ptr[offset13] < c_b)
                                      {}
                                    else
                                      continue;
                                else
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      if(ptr[offset12] < c_b)
                                        if(ptr[offset13] < c_b)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset8] < c_b)
                                  if(ptr[offset9] < c_b)
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        if(ptr[offset12] < c_b)
                                          if(ptr[offset13] < c_b)
                                            {}
                                          else
                                            continue;
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              continue;
                          else
                            continue;
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
                                          {}
                                        else
                                          if(ptr[offset15] > cb)
                                            {}
                                          else
                                            continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                            else
                              continue;
                          else
                            continue;
                        else if(ptr[offset12] < c_b)
                          if(ptr[offset13] < c_b)
                            if(ptr[offset14] < c_b)
                              if(ptr[offset15] < c_b)
                                if(ptr[offset1] < c_b)
                                  if(ptr[offset3] < c_b)
                                    {}
                                  else
                                    if(ptr[offset10] < c_b)
                                      if(ptr[offset11] < c_b)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                else
                                  if(ptr[offset8] < c_b)
                                    if(ptr[offset9] < c_b)
                                      if(ptr[offset10] < c_b)
                                        if(ptr[offset11] < c_b)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset6] < c_b)
                                  if(ptr[offset7] < c_b)
                                    if(ptr[offset8] < c_b)
                                      if(ptr[offset9] < c_b)
                                        if(ptr[offset10] < c_b)
                                          if(ptr[offset11] < c_b)
                                            {}
                                          else
                                            continue;
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              continue;
                          else
                            continue;
                        else
                          continue;
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
                                        {}
                                      else
                                        if(ptr[offset14] > cb)
                                          {}
                                        else
                                          continue;
                                    else
                                      if(ptr[offset14] > cb)
                                        if(ptr[offset15] > cb)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                            else
                              continue;
                          else
                            continue;
                        else
                          continue;
                      else if(ptr[offset11] < c_b)
                        if(ptr[offset12] < c_b)
                          if(ptr[offset13] < c_b)
                            if(ptr[offset10] < c_b)
                              if(ptr[offset14] < c_b)
                                if(ptr[offset15] < c_b)
                                  if(ptr[offset1] < c_b)
                                    {}
                                  else
                                    if(ptr[offset8] < c_b)
                                      if(ptr[offset9] < c_b)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                else
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset7] < c_b)
                                      if(ptr[offset8] < c_b)
                                        if(ptr[offset9] < c_b)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset5] < c_b)
                                  if(ptr[offset6] < c_b)
                                    if(ptr[offset7] < c_b)
                                      if(ptr[offset8] < c_b)
                                        if(ptr[offset9] < c_b)
                                          {}
                                        else
                                          continue;
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              if(ptr[offset1] < c_b)
                                if(ptr[offset3] < c_b)
                                  if(ptr[offset14] < c_b)
                                    if(ptr[offset15] < c_b)
                                      {}
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                          else
                            continue;
                        else
                          continue;
                      else
                        continue;
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
                                      {}
                                    else
                                      if(ptr[offset12] > cb)
                                        {}
                                      else
                                        continue;
                                  else
                                    if(ptr[offset12] > cb)
                                      if(ptr[offset13] > cb)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                else
                                  if(ptr[offset12] > cb)
                                    if(ptr[offset13] > cb)
                                      if(ptr[offset14] > cb)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset12] > cb)
                                  if(ptr[offset13] > cb)
                                    if(ptr[offset14] > cb)
                                      if(ptr[offset15] > cb)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              continue;
                          else
                            continue;
                        else
                          continue;
                      else
                        continue;
                    else if(ptr[offset9] < c_b)
                      if(ptr[offset10] < c_b)
                        if(ptr[offset11] < c_b)
                          if(ptr[offset8] < c_b)
                            if(ptr[offset12] < c_b)
                              if(ptr[offset13] < c_b)
                                if(ptr[offset14] < c_b)
                                  if(ptr[offset15] < c_b)
                                    {}
                                  else
                                    if(ptr[offset6] < c_b)
                                      if(ptr[offset7] < c_b)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                else
                                  if(ptr[offset5] < c_b)
                                    if(ptr[offset6] < c_b)
                                      if(ptr[offset7] < c_b)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset5] < c_b)
                                    if(ptr[offset6] < c_b)
                                      if(ptr[offset7] < c_b)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              if(ptr[offset3] < c_b)
                                if(ptr[offset4] < c_b)
                                  if(ptr[offset5] < c_b)
                                    if(ptr[offset6] < c_b)
                                      if(ptr[offset7] < c_b)
                                        {}
                                      else
                                        continue;
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                          else
                            if(ptr[offset1] < c_b)
                              if(ptr[offset12] < c_b)
                                if(ptr[offset13] < c_b)
                                  if(ptr[offset14] < c_b)
                                    if(ptr[offset15] < c_b)
                                      {}
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                            else
                              continue;
                        else
                          continue;
                      else
                        continue;
                    else
                      continue;
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
                                    {}
                                  else
                                    if(ptr[offset10] > cb)
                                      {}
                                    else
                                      continue;
                                else
                                  if(ptr[offset10] > cb)
                                    if(ptr[offset11] > cb)
                                      {}
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset10] > cb)
                                  if(ptr[offset11] > cb)
                                    if(ptr[offset12] > cb)
                                      {}
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              if(ptr[offset10] > cb)
                                if(ptr[offset11] > cb)
                                  if(ptr[offset12] > cb)
                                    if(ptr[offset13] > cb)
                                      {}
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                          else
                            if(ptr[offset10] > cb)
                              if(ptr[offset11] > cb)
                                if(ptr[offset12] > cb)
                                  if(ptr[offset13] > cb)
                                    if(ptr[offset14] > cb)
                                      {}
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                            else
                              continue;
                        else
                          if(ptr[offset10] > cb)
                            if(ptr[offset11] > cb)
                              if(ptr[offset12] > cb)
                                if(ptr[offset13] > cb)
                                  if(ptr[offset14] > cb)
                                    if(ptr[offset15] > cb)
                                      {}
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                            else
                              continue;
                          else
                            continue;
                      else
                        continue;
                    else
                      continue;
                  else if(ptr[offset7] < c_b)
                    if(ptr[offset8] < c_b)
                      if(ptr[offset9] < c_b)
                        if(ptr[offset6] < c_b)
                          if(ptr[offset5] < c_b)
                            if(ptr[offset4] < c_b)
                              if(ptr[offset3] < c_b)
                                if(ptr[offset2] < c_b)
                                  if(ptr[offset1] < c_b)
                                    {}
                                  else
                                    if(ptr[offset10] < c_b)
                                      {}
                                    else
                                      continue;
                                else
                                  if(ptr[offset10] < c_b)
                                    if(ptr[offset11] < c_b)
                                      {}
                                    else
                                      continue;
                                  else
                                    continue;
                              else
                                if(ptr[offset10] < c_b)
                                  if(ptr[offset11] < c_b)
                                    if(ptr[offset12] < c_b)
                                      {}
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                            else
                              if(ptr[offset10] < c_b)
                                if(ptr[offset11] < c_b)
                                  if(ptr[offset12] < c_b)
                                    if(ptr[offset13] < c_b)
                                      {}
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                          else
                            if(ptr[offset10] < c_b)
                              if(ptr[offset11] < c_b)
                                if(ptr[offset12] < c_b)
                                  if(ptr[offset13] < c_b)
                                    if(ptr[offset14] < c_b)
                                      {}
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                            else
                              continue;
                        else
                          if(ptr[offset10] < c_b)
                            if(ptr[offset11] < c_b)
                              if(ptr[offset12] < c_b)
                                if(ptr[offset13] < c_b)
                                  if(ptr[offset14] < c_b)
                                    if(ptr[offset15] < c_b)
                                      {}
                                    else
                                      continue;
                                  else
                                    continue;
                                else
                                  continue;
                              else
                                continue;
                            else
                              continue;
                          else
                            continue;
                      else
                        continue;
                    else
                      continue;
                  else
                    continue;
            }
            if(total == nExpectedCorners)
            {
                if(nExpectedCorners == 0)
                {
                    nExpectedCorners = 512;
                    keypoints.reserve(nExpectedCorners);
                }
                else
                {
                    nExpectedCorners *= 2;
                    keypoints.reserve(nExpectedCorners);
                }
            }
            keypoints.push_back(KeyPoint(Point2f((float)x, (float)y), 7.0f));
            total++;
        }
    }
}



#else // !(defined __i386__ || defined(_M_IX86) || defined __x86_64__ || defined(_M_X64))

static void AGAST_ALL(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold, int agasttype)
{
    cv::Mat img;
    if(!_img.getMat().isContinuous())
      img = _img.getMat().clone();
    else
      img = _img.getMat();

    int agastbase;
    int result;
    uint32_t *table_struct1;
    uint32_t *table_struct2;
    static const uint32_t table_5_8_struct1[] =
    {
        0x00010026,0x20020017,0x3003000c,0x50040009,0x10050007,0x406f0006,0x706f006c,0x4008006c,
        0x606f006c,0x100a006c,0x406d000b,0x706d006c,0x700d0012,0x600e006c,0x500f0011,0x106f0010,
        0x406f006c,0x106d006c,0x5013106c,0x3014106c,0x7015106c,0x4016106c,0x606f106e,0x5018001c,
        0x7019006c,0x601a006c,0x106d001b,0x406d006c,0x501d106c,0x301e106c,0x201f1023,0x10201021,
        0x406f106c,0x4022106c,0x606f106c,0x7024106c,0x4025106c,0x606f106c,0x00271058,0x20281049,
        0x70290035,0x302a1031,0x502b102f,0x102c102d,0x406f106e,0x402e106c,0x606f106e,0x1030106c,
        0x406f106c,0x5032006c,0x3033006c,0x4034006c,0x606f006e,0x70361041,0x3037103c,0x5038103b,
        0x106f1039,0x403a106c,0x606f106e,0x106d106c,0x603d106c,0x503e1040,0x106f103f,0x406f106c,
        0x106d106c,0x3042106c,0x50431047,0x10441045,0x406f106c,0x4046106c,0x606f106c,0x1048106c,
        0x406d106c,0x504a0053,0x304b006c,0x204c0050,0x104d004e,0x406f006c,0x404f006c,0x606f006c,
        0x7051006c,0x4052006c,0x606f006c,0x5054106c,0x7055106c,0x6056106c,0x106d1057,0x406d106c,
        0x30590062,0x505a006c,0x205b005f,0x105c005d,0x406d006c,0x405e006c,0x606d006c,0x7060006c,
        0x4061006c,0x606d006c,0x3063106c,0x5064106c,0x20651069,0x10661067,0x406d106c,0x4068106c,
        0x606d106c,0x706a106c,0x406b106c,0x606d106c,0x000000fc,0x000000fd,0x000000fe,0x000000ff
    };

    static const uint32_t table_5_8_struct2[] =
    {
        0x0001002a,0x2002001b,0x30030010,0x5004000c,0x70050008,0x10730006,0x40070072,0x60730072,
        0x1009000a,0x40730072,0x400b0072,0x60730072,0x700d000e,0x10730072,0x100f0072,0x40730072,
        0x70110016,0x60120072,0x50130015,0x10730014,0x40730072,0x10730072,0x50171072,0x30181070,
        0x70191070,0x401a1072,0x60731072,0x501c0020,0x701d0072,0x601e0072,0x1073001f,0x40730072,
        0x50211070,0x30221072,0x20231027,0x10241025,0x40731072,0x40261072,0x60731072,0x70281070,
        0x40291070,0x60711070,0x002b105c,0x202c104d,0x702d0039,0x302e1035,0x502f1033,0x10301031,
        0x40731072,0x40321072,0x60731072,0x10341072,0x40731072,0x50360072,0x30370070,0x40380072,
        0x60730072,0x703a1045,0x303b1040,0x503c103f,0x1073103d,0x403e1072,0x60731072,0x10731072,
        0x60411072,0x50421044,0x10731043,0x40731072,0x10731072,0x30461070,0x5047104b,0x10481049,
        0x40711070,0x404a1070,0x60711070,0x104c1070,0x40711070,0x504e0057,0x304f0072,0x20500054,
        0x10510052,0x40730072,0x40530072,0x60730072,0x70550070,0x40560070,0x60710070,0x50581070,
        0x70591072,0x605a1072,0x1073105b,0x40731072,0x305d0066,0x505e0070,0x205f0063,0x10600061,
        0x40710070,0x40620070,0x60710070,0x70640070,0x40650070,0x60710070,0x30671070,0x50681070,
        0x2069106d,0x106a106b,0x40711070,0x406c1070,0x60711070,0x706e1070,0x406f1070,0x60711070,
        0x000000fc,0x000000fd,0x000000fe,0x000000ff
    };

    static const uint32_t table_7_12d_struct1[] =
    {
        0x000100b5,0x50020036,0x20030025,0x9004001d,0x10050015,0x6006000f,0x3007000a,0x41870008,
        0xa0090186,0xb1890186,0x800b0186,0xa00c0186,0xb189000d,0x400e0186,0x71890188,0xb0100186,
        0x30110013,0x41870012,0xa1870186,0x80140186,0xa1870186,0x60160186,0x70170186,0x80180186,
        0x4019001b,0x3189001a,0xa1890186,0xa01c0186,0xb1890186,0x301e0186,0x401f0186,0x10200022,
        0x61870021,0xb1870186,0x60230186,0x70240186,0x81870186,0x90260186,0x70270186,0x80280186,
        0x10290030,0xa02a002d,0xb187002b,0x602c0186,0x41890186,0x602e0186,0x302f0186,0x41890186,
        0x60310186,0x40320034,0x31870033,0xa1870186,0xa0350186,0xb1870186,0x503710a1,0x9038006b,
        0x3039105b,0x403a1053,0xb03b004d,0x103c0044,0x803d0040,0xa03e0186,0x2189003f,0x71890188,
        0x60411186,0x20421186,0x70431188,0x81891188,0x60450048,0x70460186,0x80470186,0xa1890188,
        0x60491186,0x204a1186,0x704b1186,0x1189104c,0x81891188,0x204e1186,0x704f1186,0x10501051,
        0x61891186,0x60521186,0x81891186,0xb0540186,0x80550186,0xa0560186,0x10570059,0x21890058,
        0x71890186,0x605a0186,0x71890186,0xb05c0186,0xa05d0186,0x305e0065,0x105f0062,0x21870060,
        0x70610186,0x81890186,0x60630186,0x70640186,0x81890186,0x80660186,0x10670069,0x21870068,
        0x71870186,0x606a0186,0x71870186,0x906c1093,0x206d0087,0x106e007f,0x406f0077,0xa0700072,
        0x30710186,0xb1890186,0x60731186,0x70741186,0x80751186,0xb0761188,0xa1891188,0x60781186,
        0x70791186,0x807a1186,0xa07b107d,0x4189107c,0xb1891188,0x307e1186,0x41891188,0x60801186,
        0x70811186,0x80821186,0x40831085,0x31891084,0xa1891186,0xa0861186,0xb1891186,0x60881186,
        0x70891186,0x808a108f,0x408b108d,0x3187108c,0xa1871186,0xa08e1186,0xb1871186,0x20901186,
        0x10911186,0x30921186,0x41891186,0x20940099,0x10950186,0x30960186,0x40970186,0xa0980186,
        0xb1870186,0x209a1186,0x309b1186,0x409c1186,0x709d1186,0x109e109f,0x61871186,0x60a01186,
        0x81871186,0x20a200ae,0xa0a30186,0xb0a40186,0x90a500ab,0x10a600a8,0x318700a7,0x81870186,
        0x60a90186,0x70aa0186,0x81870186,0x10ac0186,0x30ad0186,0x41870186,0x90af0186,0x70b00186,
        0x80b10186,0xa0b20186,0xb0b30186,0x118700b4,0x61870186,0x00b6115a,0x20b700e2,0x50b800cc,
        0x70b900c5,0x60ba0186,0x40bb00c1,0x30bc00be,0x118700bd,0x81870186,0x90bf0186,0x80c00186,
        0xa1890186,0x90c20186,0x80c30186,0xa0c40186,0xb1890186,0x90c61186,0x80c71186,0xa0c81186,
        0xb0c91186,0x70ca1186,0x118910cb,0x61891186,0x90cd1186,0x70ce1186,0x80cf1186,0x50d010de,
        0x10d110d8,0xa0d210d5,0xb18910d3,0x60d41186,0x41891188,0x60d61186,0x30d71186,0x41891188,
        0x60d91186,0x40da10dc,0x318910db,0xa1891186,0xa0dd1186,0xb1891186,0xa0df1186,0xb0e01186,
        0x118710e1,0x61871186,0x20e3113a,0x90e4010b,0x50e500ff,0x10e610f7,0x40e710ef,0xa0e810ea,
        0x30e91186,0xb1891186,0x60eb0186,0x70ec0186,0x80ed0186,0xb0ee0188,0xa1890188,0x60f00186,
        0x70f10186,0x80f20186,0xa0f300f5,0x418900f4,0xb1890188,0x30f60186,0x41890188,0x60f80186,
        0x70f90186,0x80fa0186,0x40fb00fd,0x318900fc,0xa1890186,0xa0fe0186,0xb1890186,0x31001186,
        0x41011186,0x51021108,0x11031105,0x61871104,0xb1871186,0x61061186,0x71071186,0x81891186,
        0x11091186,0xa10a1186,0xb1871186,0x910c112e,0x510d1126,0x110e111e,0x610f1118,0x31101113,
        0x41871111,0xa1121186,0xb1891186,0x81141186,0xa1151186,0xb1891116,0x41171186,0x71891188,
        0xb1191186,0x311a111c,0x4187111b,0xa1871186,0x811d1186,0xa1871186,0x611f1186,0x71201186,
        0x81211186,0x41221124,0x31891123,0xa1891186,0xa1251186,0xb1891186,0xa1271186,0xb1281186,
        0x1129112b,0x3187112a,0x81871186,0x612c1186,0x712d1186,0x81871186,0x312f1186,0x41301186,
        0x51311137,0x11321134,0x61871133,0xb1871186,0x61351186,0x71361186,0x81871186,0x11381186,
        0xa1391186,0xb1871186,0x913b1150,0x713c1186,0x813d1186,0x513e114c,0x113f1146,0xa1401143,
        0xb1871141,0x61421186,0x41891186,0x61441186,0x31451186,0x41891186,0x61471186,0x4148114a,
        0x31871149,0xa1871186,0xa14b1186,0xb1871186,0xa14d1186,0xb14e1186,0x1187114f,0x61871186,
        0x51510186,0x91520186,0x61530186,0x71540186,0x81550186,0x41560158,0x31870157,0xa1870186,
        0xa1590186,0xb1870186,0x515b0170,0x915c0168,0x615d0186,0x715e0186,0x415f0165,0x31600163,
        0x81870161,0x11620186,0x21870186,0x81640186,0xa1870186,0xb1660186,0x81670186,0xa1870186,
        0x21690186,0x316a0186,0x416b0186,0x716c0186,0x116d016e,0x61870186,0x616f0186,0x81870186,
        0x51711186,0x9172117e,0x61731186,0x71741186,0x4175117b,0x31761179,0x81871177,0x11781186,
        0x21871186,0x817a1186,0xa1871186,0xb17c1186,0x817d1186,0xa1871186,0x217f1186,0x31801186,
        0x41811186,0x71821186,0x11831184,0x61871186,0x61851186,0x81871186,0x000000fc,0x000000fd,
        0x000000fe,0x000000ff
    };

    static const uint32_t table_7_12d_struct2[] =
    {
        0x000100b5,0x50020036,0x20030025,0x9004001d,0x10050015,0x6006000f,0x3007000a,0x41890008,
        0xa0090188,0xb1890188,0x800b0188,0xa00c0188,0xb189000d,0x400e0188,0x71890188,0xb0100188,
        0x30110013,0x41890012,0xa1890188,0x80140188,0xa1890188,0x60160188,0x70170188,0x80180188,
        0x4019001b,0x3189001a,0xa1890188,0xa01c0188,0xb1890188,0x301e0188,0x401f0188,0x10200022,
        0x61890021,0xb1890188,0x60230188,0x70240188,0x81890188,0x90260188,0x70270188,0x80280188,
        0x10290030,0xa02a002d,0xb189002b,0x602c0188,0x41890188,0x602e0188,0x302f0188,0x41890188,
        0x60310188,0x40320034,0x31890033,0xa1890188,0xa0350188,0xb1890188,0x503710a1,0x9038006b,
        0x3039105b,0x403a1053,0xb03b004d,0x103c0044,0x803d0040,0xa03e0188,0x2189003f,0x71890188,
        0x60411188,0x20421188,0x70431188,0x81891188,0x60450048,0x70460188,0x80470188,0xa1890188,
        0x60491188,0x204a1188,0x704b1188,0x1189104c,0x81891188,0x204e1188,0x704f1188,0x10501051,
        0x61891188,0x60521188,0x81891188,0xb0540188,0x80550188,0xa0560188,0x10570059,0x21890058,
        0x71890188,0x605a0188,0x71890188,0xb05c0188,0xa05d0188,0x305e0065,0x105f0062,0x21890060,
        0x70610188,0x81890188,0x60630188,0x70640188,0x81890188,0x80660188,0x10670069,0x21890068,
        0x71890188,0x606a0188,0x71890188,0x906c1093,0x206d0087,0x106e007f,0x406f0077,0xa0700072,
        0x30710188,0xb1890188,0x60731188,0x70741188,0x80751188,0xb0761188,0xa1891188,0x60781188,
        0x70791188,0x807a1188,0xa07b107d,0x4189107c,0xb1891188,0x307e1188,0x41891188,0x60801188,
        0x70811188,0x80821188,0x40831085,0x31891084,0xa1891188,0xa0861188,0xb1891188,0x60881188,
        0x70891188,0x808a108f,0x408b108d,0x3189108c,0xa1891188,0xa08e1188,0xb1891188,0x20901188,
        0x10911188,0x30921188,0x41891188,0x20940099,0x10950188,0x30960188,0x40970188,0xa0980188,
        0xb1890188,0x209a1186,0x309b1188,0x409c1188,0x709d1188,0x109e109f,0x61891188,0x60a01188,
        0x81891188,0x20a200ae,0xa0a30188,0xb0a40188,0x90a500ab,0x10a600a8,0x318900a7,0x81890188,
        0x60a90188,0x70aa0188,0x81890188,0x10ac0188,0x30ad0188,0x41890188,0x90af0188,0x70b00188,
        0x80b10188,0xa0b20188,0xb0b30188,0x118900b4,0x61890188,0x00b6115a,0x20b700e2,0x50b800cc,
        0x70b900c5,0x60ba0188,0x40bb00c1,0x30bc00be,0x118900bd,0x81890188,0x90bf0188,0x80c00188,
        0xa1890188,0x90c20188,0x80c30188,0xa0c40188,0xb1890188,0x90c61188,0x80c71188,0xa0c81188,
        0xb0c91188,0x70ca1188,0x118910cb,0x61891188,0x90cd1188,0x70ce1188,0x80cf1188,0x50d010de,
        0x10d110d8,0xa0d210d5,0xb18910d3,0x60d41188,0x41891188,0x60d61188,0x30d71188,0x41891188,
        0x60d91188,0x40da10dc,0x318910db,0xa1891188,0xa0dd1188,0xb1891188,0xa0df1188,0xb0e01188,
        0x118910e1,0x61891188,0x20e3113a,0x90e4010b,0x50e500ff,0x10e610f7,0x40e710ef,0xa0e810ea,
        0x30e91188,0xb1891188,0x60eb0188,0x70ec0188,0x80ed0188,0xb0ee0188,0xa1890188,0x60f00188,
        0x70f10188,0x80f20188,0xa0f300f5,0x418900f4,0xb1890188,0x30f60188,0x41890188,0x60f80188,
        0x70f90188,0x80fa0188,0x40fb00fd,0x318900fc,0xa1890188,0xa0fe0188,0xb1890188,0x31001188,
        0x41011188,0x51021108,0x11031105,0x61891104,0xb1891188,0x61061188,0x71071188,0x81891188,
        0x11091188,0xa10a1188,0xb1891188,0x910c112e,0x510d1126,0x110e111e,0x610f1118,0x31101113,
        0x41891111,0xa1121188,0xb1891188,0x81141188,0xa1151188,0xb1891116,0x41171188,0x71891188,
        0xb1191188,0x311a111c,0x4189111b,0xa1891188,0x811d1188,0xa1891188,0x611f1188,0x71201188,
        0x81211188,0x41221124,0x31891123,0xa1891188,0xa1251188,0xb1891188,0xa1271188,0xb1281188,
        0x1129112b,0x3189112a,0x81891188,0x612c1188,0x712d1188,0x81891188,0x312f1188,0x41301188,
        0x51311137,0x11321134,0x61891133,0xb1891188,0x61351188,0x71361188,0x81891188,0x11381188,
        0xa1391188,0xb1891188,0x913b1150,0x713c1188,0x813d1188,0x513e114c,0x113f1146,0xa1401143,
        0xb1891141,0x61421188,0x41891188,0x61441188,0x31451188,0x41891188,0x61471188,0x4148114a,
        0x31891149,0xa1891188,0xa14b1188,0xb1891188,0xa14d1188,0xb14e1188,0x1189114f,0x61891188,
        0x51510188,0x91520186,0x61530188,0x71540188,0x81550188,0x41560158,0x31890157,0xa1890188,
        0xa1590188,0xb1890188,0x515b0170,0x915c0168,0x615d0188,0x715e0188,0x415f0165,0x31600163,
        0x81890161,0x11620188,0x21890188,0x81640188,0xa1890188,0xb1660188,0x81670188,0xa1890188,
        0x21690188,0x316a0188,0x416b0188,0x716c0188,0x116d016e,0x61890188,0x616f0188,0x81890188,
        0x51711186,0x9172117e,0x61731188,0x71741188,0x4175117b,0x31761179,0x81891177,0x11781188,
        0x21891188,0x817a1188,0xa1891188,0xb17c1188,0x817d1188,0xa1891188,0x217f1188,0x31801188,
        0x41811188,0x71821188,0x11831184,0x61891188,0x61851188,0x81891188,0x000000fc,0x000000fd,
        0x000000fe,0x000000ff
    };

    static const uint32_t table_7_12s_struct1[] =
    {
        0x00010091,0x20020064,0x50030031,0x90040026,0x7005001c,0x10060015,0x6007000f,0x3008000b,
        0x41590009,0xa00a0156,0xb1590158,0x800c0156,0xa00d0156,0x4159000e,0xb1590158,0xb0100156,
        0x30110013,0x41590012,0xa1590156,0x80140156,0xa1590156,0x60160156,0x80170156,0x4018001a,
        0x31590019,0xa1590156,0xa01b0156,0xb1590156,0x101d0156,0xb01e0023,0x301f0021,0x41570020,
        0xa1570156,0x80220156,0xa1570156,0x60240156,0x30250156,0x41570156,0x30270156,0x40280156,
        0x7029002e,0x102a002c,0x6157002b,0xb1570156,0x602d0156,0x81570156,0x102f0156,0x61570030,
        0xb1570156,0x90321055,0x70331050,0x5034104b,0x10350044,0x4036003d,0xa0370039,0x30380156,
        0xb1590158,0x603a1156,0x803b1156,0xb03c1158,0xa1591158,0x603e1156,0x803f1156,0xa0401042,
        0x41591041,0xb1591158,0x30431156,0x41591158,0x60451156,0x80461156,0x40471049,0x31591048,
        0xa1591156,0xa04a1156,0xb1591156,0x104c0156,0x304d0156,0x404e0156,0xa04f0156,0xb1590156,
        0x10510156,0x30520156,0x40530156,0xa0540156,0xb1570156,0xa0560156,0xb0570156,0x90580061,
        0x7059005e,0x105a005c,0x3157005b,0x81570156,0x605d0156,0x81570156,0x105f0156,0x31570060,
        0x81570156,0x10620156,0x30630156,0x41570156,0x7065007a,0x90660156,0x80670156,0x50680076,
        0x10690070,0xa06a006d,0xb157006b,0x606c0156,0x41590156,0x606e0156,0x306f0156,0x41590156,
        0x60710156,0x40720074,0x31570073,0xa1570156,0xa0750156,0xb1570156,0xa0770156,0xb0780156,
        0x11570079,0x61570156,0x707b1156,0x507c1156,0x207d1089,0x607e1156,0x407f1085,0x30801082,
        0x11571081,0x81571156,0x90831156,0x80841156,0xa1591156,0x90861156,0x80871156,0xa0881156,
        0xb1591156,0x908a1156,0x608b1156,0x808c1156,0x408d108f,0x3157108e,0xa1571156,0xa0901156,
        0xb1571156,0x0092112c,0x209310ff,0x909410c2,0x509510b7,0x709610ad,0x109710a6,0x609810a0,
        0x3099109c,0x4159109a,0xa09b1156,0xb1591158,0x809d1156,0xa09e1156,0x4159109f,0xb1591158,
        0xb0a11156,0x30a210a4,0x415910a3,0xa1591156,0x80a51156,0xa1591156,0x60a71156,0x80a81156,
        0x40a910ab,0x315910aa,0xa1591156,0xa0ac1156,0xb1591156,0x10ae1156,0xb0af10b4,0x30b010b2,
        0x415710b1,0xa1571156,0x80b31156,0xa1571156,0x60b51156,0x30b61156,0x41571156,0xa0b81156,
        0xb0b91156,0x70ba10bf,0x10bb10bd,0x315710bc,0x81571156,0x60be1156,0x81571156,0x10c01156,
        0x315710c1,0x81571156,0x90c300f0,0x50c400e1,0x70c500dc,0x10c610d5,0x40c710ce,0xa0c810ca,
        0x30c91156,0xb1591158,0x60cb0156,0x80cc0156,0xb0cd0158,0xa1590158,0x60cf0156,0x80d00156,
        0xa0d100d3,0x415900d2,0xb1590158,0x30d40156,0x41590158,0x60d60156,0x80d70156,0x40d800da,
        0x315900d9,0xa1590156,0xa0db0156,0xb1590156,0x10dd1156,0x30de1156,0x40df1156,0xa0e01156,
        0xb1591156,0x30e21156,0x40e31156,0x50e410ed,0x70e510ea,0x10e610e8,0x615910e7,0xb1591156,
        0x60e91156,0x81591156,0x10eb1156,0x615710ec,0xb1571156,0x10ee1156,0xa0ef1156,0xb1571156,
        0x30f11156,0x40f21156,0x50f310fc,0x70f410f9,0x10f510f7,0x615710f6,0xb1571156,0x60f81156,
        0x81571156,0x10fa1156,0x615710fb,0xb1571156,0x10fd1156,0xa0fe1156,0xb1571156,0x71000116,
        0x51010156,0x2102010e,0x61030156,0x4104010a,0x31050107,0x11570106,0x81570156,0x91080156,
        0x81090156,0xa1590156,0x910b0156,0x810c0156,0xa10d0156,0xb1590156,0x910f0156,0x61100156,
        0x81110156,0x41120114,0x31570113,0xa1570156,0xa1150156,0xb1570156,0x71171156,0x91181156,
        0x81191156,0x511a1128,0x111b1122,0xa11c111f,0xb157111d,0x611e1156,0x41591156,0x61201156,
        0x31211156,0x41591156,0x61231156,0x41241126,0x31571125,0xa1571156,0xa1271156,0xb1571156,
        0xa1291156,0xb12a1156,0x1157112b,0x61571156,0x512d0141,0x712e0156,0x912f013a,0x61300156,
        0x41310137,0x31320135,0x81570133,0x11340156,0x21570156,0x81360156,0xa1570156,0xb1380156,
        0x81390156,0xa1570156,0x213b0156,0x313c0156,0x413d0156,0x113e013f,0x61570156,0x61400156,
        0x81570156,0x51421156,0x71431156,0x9144114f,0x61451156,0x4146114c,0x3147114a,0x81571148,
        0x11491156,0x21571156,0x814b1156,0xa1571156,0xb14d1156,0x814e1156,0xa1571156,0x21501156,
        0x31511156,0x41521156,0x11531154,0x61571156,0x61551156,0x81571156,0x000000fc,0x000000fd,
        0x000000fe,0x000000ff
    };

    static const uint32_t table_7_12s_struct2[] =
    {
        0x00010092,0x20020065,0x50030031,0x90040026,0x7005001c,0x10060015,0x6007000f,0x3008000b,
        0x41400009,0xa00a013f,0xb140013f,0x800c013f,0xa00d013f,0x4140000e,0xb140013f,0xb010013f,
        0x30110013,0x41400012,0xa140013f,0x8014013f,0xa140013f,0x6016013f,0x8017013f,0x4018001a,
        0x31400019,0xa140013f,0xa01b013f,0xb140013f,0x101d013f,0xb01e0023,0x301f0021,0x41400020,
        0xa140013f,0x8022013f,0xa140013f,0x6024013f,0x3025013f,0x4140013f,0x3027013f,0x4028013f,
        0x7029002e,0x102a002c,0x6140002b,0xb140013f,0x602d013f,0x8140013f,0x102f013f,0x61400030,
        0xb140013f,0x70321059,0x90331050,0x5034104b,0x10350044,0x4036003d,0xa0370039,0x3038013f,
        0xb140013f,0x603a113f,0x803b113f,0xb03c113f,0xa140113f,0x603e113f,0x803f113f,0xa0401042,
        0x41401041,0xb140113f,0x3043113f,0x4140113f,0x6045113f,0x8046113f,0x40471049,0x31401048,
        0xa140113f,0xa04a113f,0xb140113f,0x104c013f,0x304d013f,0x404e013f,0xa04f013f,0xb140013f,
        0xa051013f,0xb052013f,0x90530056,0x1054013f,0x31400055,0x8140013f,0x1057013f,0x3058013f,
        0x4140013f,0xa05a013f,0xb05b013f,0x905c0062,0x105d005f,0x3140005e,0x8140013f,0x6060013f,
        0x8061013f,0x7140013f,0x1063013f,0x3064013f,0x4140013f,0x7066007b,0x9067013f,0x8068013f,
        0x50690077,0x106a0071,0xa06b006e,0xb140006c,0x606d013f,0x4140013f,0x606f013f,0x3070013f,
        0x4140013f,0x6072013f,0x40730075,0x31400074,0xa140013f,0xa076013f,0xb140013f,0xa078013f,
        0xb079013f,0x1140007a,0x6140013f,0x707c113f,0x507d113f,0x207e108a,0x607f113f,0x40801086,
        0x30811083,0x11401082,0x8140113f,0x9084113f,0x8085113f,0xa140113f,0x9087113f,0x8088113f,
        0xa089113f,0xb140113f,0x908b113f,0x608c113f,0x808d113f,0x408e1090,0x3140108f,0xa140113f,
        0xa091113f,0xb140113f,0x00931113,0x209410e6,0xb09510c8,0x309610b9,0x509710a9,0x909810a3,
        0x709910a0,0x109a109c,0x4140109b,0xa140113f,0x609d113f,0x809e113f,0x4140109f,0xa140113f,
        0x10a1113f,0x414010a2,0xa140113f,0x40a4113f,0x70a510a8,0x114010a6,0x60a7113f,0x8140113f,
        0x1140113f,0xa0aa10b2,0x90ab10b0,0x70ac10af,0x114010ad,0x60ae113f,0x8140113f,0x1140113f,
        0x10b1113f,0x4140113f,0x70b3013f,0x90b4013f,0x50b5013f,0x40b6013f,0x60b7013f,0x80b8013f,
        0xa140013f,0x90ba10c0,0x80bb113f,0xa0bc113f,0x70bd10bf,0x114010be,0x6140113f,0x1140113f,
        0x50c1013f,0x70c2013f,0x90c3013f,0x40c4013f,0x60c5013f,0x80c6013f,0x314000c7,0xa140013f,
        0x40c910dc,0x50ca10d5,0x70cb10d2,0x60cc113f,0x30cd10cf,0x114010ce,0x8140113f,0x90d0113f,
        0x80d1113f,0xa140113f,0x10d3113f,0x60d4113f,0x3140113f,0x70d6013f,0x90d7013f,0x50d8013f,
        0x60d9013f,0x80da013f,0xa0db013f,0xb140013f,0x50dd013f,0x70de013f,0x90df013f,0x60e0013f,
        0x80e1013f,0xa0e200e4,0x414000e3,0xb140013d,0x30e5013f,0x4140013f,0x70e700fd,0x50e8013f,
        0x20e900f5,0x60ea013f,0x40eb00f1,0x30ec00ee,0x114000ed,0x8140013f,0x90ef013f,0x80f0013f,
        0xa140013f,0x90f2013f,0x80f3013f,0xa0f4013f,0xb140013f,0x90f6013f,0x60f7013f,0x80f8013f,
        0x40f900fb,0x314000fa,0xa140013f,0xa0fc013f,0xb140013f,0x70fe113f,0x90ff113f,0x8100113f,
        0x5101110f,0x11021109,0xa1031106,0xb1401104,0x6105113f,0x4140113f,0x6107113f,0x3108113f,
        0x4140113f,0x610a113f,0x410b110d,0x3140110c,0xa140113f,0xa10e113f,0xb140113f,0xa110113f,
        0xb111113f,0x11401112,0x6140113f,0x51140128,0x7115013f,0x91160121,0x6117013f,0x4118011e,
        0x3119011c,0x8140011a,0x111b013f,0x2140013f,0x811d013f,0xa140013f,0xb11f013f,0x8120013f,
        0xa140013f,0x2122013f,0x3123013f,0x4124013f,0x11250126,0x6140013f,0x6127013f,0x8140013f,
        0x5129113d,0x712a113f,0x912b1136,0x612c113f,0x412d1133,0x312e1131,0x8140112f,0x1130113f,
        0x2140113f,0x8132113f,0xa140113f,0xb134113f,0x8135113f,0xa140113f,0x2137113f,0x3138113f,
        0x4139113f,0x113a113b,0x6140113f,0x613c113f,0x8140113f,0x000000fc,0x000000fd,0x000000fe,
        0x000000ff
    };

    static const uint32_t table_9_16_struct[] =
    {
        0x00010138,0x200200d3,0x4003008a,0x50040051,0x70050027,0x30060016,0x1007000d,0x6008000a,
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
        0xc2a912ac,0xd2aa12ac,0xe2ab12ac,0xf2ad12ac,0x000000fc,0x000000fd,0x000000fe,0x000000ff
    };
    switch(agasttype) {
      case AgastFeatureDetector::AGAST_5_8:
        agastbase=0;
        table_struct1=(uint32_t *)(table_5_8_struct1);
        table_struct2=(uint32_t *)(table_5_8_struct2);
        break;
      case AgastFeatureDetector::AGAST_7_12d:
        agastbase=2;
        table_struct1=(uint32_t *)(table_7_12d_struct1);
        table_struct2=(uint32_t *)(table_7_12d_struct2);
        break;
      case AgastFeatureDetector::AGAST_7_12s:
        agastbase=1;
        table_struct1=(uint32_t *)(table_7_12s_struct1);
        table_struct2=(uint32_t *)(table_7_12s_struct2);
        break;
      case AgastFeatureDetector::OAST_9_16:
      default:
        agastbase=2;
        table_struct1=(uint32_t *)(table_9_16_struct);
        table_struct2=(uint32_t *)(table_9_16_struct);
        break;
    }

    size_t total = 0;
    int xsize = img.cols;
    int ysize = img.rows;
    size_t nExpectedCorners = keypoints.capacity();
    int x, y;
    int xsizeB = xsize - (agastbase + 2);
    int ysizeB = ysize - (agastbase + 1);
    int width;

    keypoints.resize(0);

    int pixel[16];
    makeAgastOffsets(pixel, (int)img.step, agasttype);

    width = xsize;

    for(y = agastbase+1; y < ysizeB; y++)
    {
        x = agastbase;
        while(true)
        {
          homogeneous:
          {
            x++;
            if(x > xsizeB)
                break;
            else
            {
                const unsigned char* const ptr = img.ptr() + y*width + x;
                result = agast_tree_search(table_struct1, pixel, ptr, threshold);
                switch (result)
                {
                case 252:
                    goto homogeneous;
                case 253:
                    goto success_homogeneous;
                case 254:
                    goto structured;
                case 255:
                    goto success_structured;
                }
            }
          }
          structured:
          {
            x++;
            if(x > xsizeB)
                break;
            else
            {
                const unsigned char* const ptr = img.ptr() + y*width + x;
                result = agast_tree_search(table_struct2, pixel, ptr, threshold);
                switch (result)
                {
                case 252:
                    goto homogeneous;
                case 253:
                    goto success_homogeneous;
                case 254:
                    goto structured;
                case 255:
                    goto success_structured;
                }
            }
          }
            success_homogeneous:
            if(total == nExpectedCorners)
            {
                if(nExpectedCorners == 0)
                {
                    nExpectedCorners = 512;
                    keypoints.reserve(nExpectedCorners);
                }
                else
                {
                    nExpectedCorners *= 2;
                    keypoints.reserve(nExpectedCorners);
                }
            }
            keypoints.push_back(KeyPoint(Point2f((float)x, (float)y), 7.0f));
            total++;
            goto homogeneous;
            success_structured:
            if(total == nExpectedCorners)
            {
                if(nExpectedCorners == 0)
                {
                    nExpectedCorners = 512;
                    keypoints.reserve(nExpectedCorners);
                }
                else
                {
                    nExpectedCorners *= 2;
                    keypoints.reserve(nExpectedCorners);
                }
            }
            keypoints.push_back(KeyPoint(Point2f((float)x, (float)y), 7.0f));
            total++;
            goto structured;
        }
    }
}

static void AGAST_5_8(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold)
{
    AGAST_ALL(_img, keypoints, threshold, AgastFeatureDetector::AGAST_5_8);
}

static void AGAST_7_12d(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold)
{
    AGAST_ALL(_img, keypoints, threshold, AgastFeatureDetector::AGAST_7_12d);
}

static void AGAST_7_12s(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold)
{
    AGAST_ALL(_img, keypoints, threshold, AgastFeatureDetector::AGAST_7_12s);
}

static void OAST_9_16(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold)
{
    AGAST_ALL(_img, keypoints, threshold, AgastFeatureDetector::OAST_9_16);
}

#endif // !(defined __i386__ || defined(_M_IX86) || defined __x86_64__ || defined(_M_X64))

void AGAST(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression)
{
    CV_INSTRUMENT_REGION()

    AGAST(_img, keypoints, threshold, nonmax_suppression, AgastFeatureDetector::OAST_9_16);
}

class AgastFeatureDetector_Impl : public AgastFeatureDetector
{
public:
    AgastFeatureDetector_Impl( int _threshold, bool _nonmaxSuppression, int _type )
    : threshold(_threshold), nonmaxSuppression(_nonmaxSuppression), type((short)_type)
    {}

    void detect( InputArray _image, std::vector<KeyPoint>& keypoints, InputArray _mask )
    {
        CV_INSTRUMENT_REGION()

        Mat mask = _mask.getMat(), grayImage;
        UMat ugrayImage;
        _InputArray gray = _image;
        if( _image.type() != CV_8U )
        {
            _OutputArray ogray = _image.isUMat() ? _OutputArray(ugrayImage) : _OutputArray(grayImage);
            cvtColor( _image, ogray, COLOR_BGR2GRAY );
            gray = ogray;
        }
        keypoints.clear();
        AGAST( gray, keypoints, threshold, nonmaxSuppression, type );
        KeyPointsFilter::runByPixelsMask( keypoints, mask );
    }

    void set(int prop, double value)
    {
        if(prop == THRESHOLD)
            threshold = cvRound(value);
        else if(prop == NONMAX_SUPPRESSION)
            nonmaxSuppression = value != 0;
        else
            CV_Error(Error::StsBadArg, "");
    }

    double get(int prop) const
    {
        if(prop == THRESHOLD)
            return threshold;
        if(prop == NONMAX_SUPPRESSION)
            return nonmaxSuppression;
        CV_Error(Error::StsBadArg, "");
        return 0;
    }

    void setThreshold(int threshold_) { threshold = threshold_; }
    int getThreshold() const { return threshold; }

    void setNonmaxSuppression(bool f) { nonmaxSuppression = f; }
    bool getNonmaxSuppression() const { return nonmaxSuppression; }

    void setType(int type_) { type = type_; }
    int getType() const { return type; }

    int threshold;
    bool nonmaxSuppression;
    int type;
};

Ptr<AgastFeatureDetector> AgastFeatureDetector::create( int threshold, bool nonmaxSuppression, int type )
{
    return makePtr<AgastFeatureDetector_Impl>(threshold, nonmaxSuppression, type);
}

void AGAST(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression, int type)
{
    CV_INSTRUMENT_REGION()

    std::vector<KeyPoint> kpts;

    // detect
    switch(type) {
      case AgastFeatureDetector::AGAST_5_8:
        AGAST_5_8(_img, kpts, threshold);
        break;
      case AgastFeatureDetector::AGAST_7_12d:
        AGAST_7_12d(_img, kpts, threshold);
        break;
      case AgastFeatureDetector::AGAST_7_12s:
        AGAST_7_12s(_img, kpts, threshold);
        break;
      case AgastFeatureDetector::OAST_9_16:
        OAST_9_16(_img, kpts, threshold);
        break;
    }

    cv::Mat img = _img.getMat();

    // score
    int pixel_[16];
    makeAgastOffsets(pixel_, (int)img.step, type);

    std::vector<KeyPoint>::iterator kpt;
    for(kpt = kpts.begin(); kpt != kpts.end(); ++kpt)
    {
        switch(type) {
          case AgastFeatureDetector::AGAST_5_8:
            kpt->response = (float)agast_cornerScore<AgastFeatureDetector::AGAST_5_8>
                (&img.at<uchar>((int)kpt->pt.y, (int)kpt->pt.x), pixel_, threshold);
            break;
          case AgastFeatureDetector::AGAST_7_12d:
            kpt->response = (float)agast_cornerScore<AgastFeatureDetector::AGAST_7_12d>
                (&img.at<uchar>((int)kpt->pt.y, (int)kpt->pt.x), pixel_, threshold);
            break;
          case AgastFeatureDetector::AGAST_7_12s:
            kpt->response = (float)agast_cornerScore<AgastFeatureDetector::AGAST_7_12s>
                (&img.at<uchar>((int)kpt->pt.y, (int)kpt->pt.x), pixel_, threshold);
            break;
          case AgastFeatureDetector::OAST_9_16:
            kpt->response = (float)agast_cornerScore<AgastFeatureDetector::OAST_9_16>
                (&img.at<uchar>((int)kpt->pt.y, (int)kpt->pt.x), pixel_, threshold);
            break;
        }
    }

    // suppression
    if(nonmax_suppression)
    {
        size_t j;
        size_t curr_idx;
        size_t lastRow = 0, next_lastRow = 0;
        size_t num_Corners = kpts.size();
        size_t lastRowCorner_ind = 0, next_lastRowCorner_ind = 0;

        std::vector<int> nmsFlags;
        std::vector<KeyPoint>::iterator currCorner_nms;
        std::vector<KeyPoint>::const_iterator currCorner;

        currCorner = kpts.begin();

        nmsFlags.resize((int)num_Corners);

        // set all flags to MAXIMUM
        for(j = 0; j < num_Corners; j++)
            nmsFlags[j] = -1;

        for(curr_idx = 0; curr_idx < num_Corners; curr_idx++)
        {
            int t;
            // check above
            if(lastRow + 1 < currCorner->pt.y)
            {
                lastRow = next_lastRow;
                lastRowCorner_ind = next_lastRowCorner_ind;
            }
            if(next_lastRow != currCorner->pt.y)
            {
                next_lastRow = (size_t) currCorner->pt.y;
                next_lastRowCorner_ind = curr_idx;
            }
            if(lastRow + 1 == currCorner->pt.y)
            {
                // find the corner above the current one
                while( (kpts[lastRowCorner_ind].pt.x < currCorner->pt.x)
                    && (kpts[lastRowCorner_ind].pt.y == lastRow) )
                    lastRowCorner_ind++;

                if( (kpts[lastRowCorner_ind].pt.x == currCorner->pt.x)
                 && (lastRowCorner_ind != curr_idx) )
                {
                    size_t w = lastRowCorner_ind;
                    // find the maximum in this block
                    while(nmsFlags[w] != -1)
                        w = nmsFlags[w];

                    if(kpts[curr_idx].response < kpts[w].response)
                        nmsFlags[curr_idx] = (int)w;
                    else
                        nmsFlags[w] = (int)curr_idx;
                }
            }

            // check left
            t = (int)curr_idx - 1;
            if( (curr_idx != 0) && (kpts[t].pt.y == currCorner->pt.y)
             && (kpts[t].pt.x + 1 == currCorner->pt.x) )
            {
                int currCornerMaxAbove_ind = nmsFlags[curr_idx];
                // find the maximum in that area
                while(nmsFlags[t] != -1)
                    t = nmsFlags[t];
                // no maximum above
                if(currCornerMaxAbove_ind == -1)
                {
                    if((size_t)t != curr_idx)
                    {
                        if ( kpts[curr_idx].response < kpts[t].response )
                            nmsFlags[curr_idx] = t;
                        else
                            nmsFlags[t] = (int)curr_idx;
                    }
                }
                else // maximum above
                {
                    if(t != currCornerMaxAbove_ind)
                    {
                        if(kpts[currCornerMaxAbove_ind].response < kpts[t].response)
                        {
                            nmsFlags[currCornerMaxAbove_ind] = t;
                            nmsFlags[curr_idx] = t;
                        }
                        else
                        {
                            nmsFlags[t] = currCornerMaxAbove_ind;
                            nmsFlags[curr_idx] = currCornerMaxAbove_ind;
                        }
                    }
                }
            }
            ++currCorner;
        }

        // collecting maximum corners
        for(curr_idx = 0; curr_idx < num_Corners; curr_idx++)
        {
            if (nmsFlags[curr_idx] == -1)
                keypoints.push_back(kpts[curr_idx]);
        }
    } else
    {
      keypoints = kpts;
    }
}

String AgastFeatureDetector::getDefaultName() const
{
    return(Feature2D::getDefaultName() + ".AgastFeatureDetector");
}

} // END NAMESPACE CV
