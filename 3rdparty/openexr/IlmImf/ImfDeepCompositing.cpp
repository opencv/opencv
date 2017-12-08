///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012, Weta Digital Ltd
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Weta Digital nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission. 
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

#include "ImfDeepCompositing.h"

#include "ImfNamespace.h"
#include <algorithm>
#include <vector>

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using std::sort;
using std::vector;
DeepCompositing::DeepCompositing()
{
}

DeepCompositing::~DeepCompositing()
{
}

void 
DeepCompositing::composite_pixel (float outputs[],
                                  const float* inputs[],
                                  const char*channel_names[],
                                  int num_channels,
                                  int num_samples,
                                  int sources)
{
    for(int i=0;i<num_channels;i++) outputs[i]=0.0;
    // no samples? do nothing
   if(num_samples==0)
   {
       return;
   }
   
   vector<int> sort_order;
   if(sources>1)
   {
       sort_order.resize(num_samples);
       for(int i=0;i<num_samples;i++) sort_order[i]=i;
       sort(&sort_order[0],inputs,channel_names,num_channels,num_samples,sources);
   }
   
   
   for(int i=0;i<num_samples;i++)
   {
       int s=(sources>1) ? sort_order[i] : i;
       float alpha=outputs[2]; 
       if(alpha>=1.0) return;
       
       for(int c=0;c<num_channels;c++)
       {
           outputs[c]+=(1.0-alpha)*inputs[c][s];
       }
   }   
}

struct sort_helper
{
    const float ** inputs;
    bool operator() (int a,int b) 
    {
        if(inputs[0][a] < inputs[0][b]) return true;
        if(inputs[0][a] > inputs[0][b]) return false;
        if(inputs[1][a] < inputs[1][b]) return true;
        if(inputs[1][a] > inputs[1][b]) return false;
        return a<b;
    }
    sort_helper(const float ** i) : inputs(i) {}
};

void
DeepCompositing::sort(int order[], const float* inputs[], const char* channel_names[], int num_channels, int num_samples, int sources)
{
  std::sort(order+0,order+num_samples,sort_helper(inputs));
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
