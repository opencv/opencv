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


#ifndef __OPENCV_FEATURES_2D_AGAST_HPP__
#define __OPENCV_FEATURES_2D_AGAST_HPP__

#ifdef __cplusplus

#include "precomp.hpp"
namespace cv
{

void makeAgastOffsets(int pixel[16], int row_stride, int type);

template<int type>
int agast_cornerScore(const uchar* ptr, const int pixel[], int threshold);

}
#endif
#endif
